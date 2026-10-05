#!/usr/bin/env python3
"""Generate a candidate ledger: call the model, record what it said, bank it.

Generation only. Nothing here compiles, simulates, proves or scores anything --
that is the whole point. The output is `candidates.jsonl`, one record per model
call, which any scorer (Tier S differential simulation, Tier F formal
equivalence) can consume later without re-running the model.

    export RTLREPAIR_LLM_URL=https://<modal-app>.modal.run/v1
    export RTLREPAIR_LLM_API_KEY=<the served endpoint's key>
    python generate_ledger.py --model OriGen_Fix --bank

There are two case sets, with different prompt grammars and different oracles.
Which one you get is selected by `--bench`:

*Default -- the 85-case spec x locate grid* (configs/vericodegen/main85_inputs.jsonl).
One run is banked per arm, keyed `(model_slot, version_id, arm)` -- the arm name
(`spec0_loc0` ... `spec1_loc1`) is used verbatim as the signal component, matching
the identifiers already in generated/second_model/verdicts.jsonl. The `diag*`
signal grammar does not apply: these cases lint and compile clean, so there is no
tool diagnostic to give.

All 85 cases are generated, not the 68 formally-tractable subset. Banking the
full set makes narrowing it a rescore rather than a regeneration.

*`--bench` -- the 459-case held-out set* (data/repairbench_heldout.jsonl, 374 lint
+ 85 semantic). This is the arm matrix repairbench_eval.py drives, under the
`diag*` grammar, and banking it as a ledger is what makes `base_diag` /
`base_diag_spec` rescorable -- the eight pre-split arms could only be regenerated.

    python generate_ledger.py --model base --bench --arms diag,diag_spec --bank

The prompt is built by `repair_agent._build_prompt` and the completion parsed by
`repair_agent._split_explanation_and_code` + `bench_common.extract_rtl`, both
imported rather than reimplemented so the ledger cannot drift from the driver.
The tool diagnostic is a *stored* string on the benchmark item, so generation
needs no verilator and no iverilog -- only scoring does.

Held-out records carry `bucket` and `mutation`, because the two buckets are
scored by different oracles (lint: `verilator --lint-only`; semantic: golden
differential simulation) and a scorer must dispatch without re-reading the
benchmark file.

The credential is read from the environment by backend.llm_client and is never
written to the ledger, to run.json, or to any log.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(HERE.parents[3])
sys.path.insert(0, str(HERE))

import prompts  # noqa: E402
# repairbench_eval puts ../ and ../datagen on sys.path at import, which is what
# makes repair_agent and bench_common importable here. Import it first.
import repairbench_eval as rbe  # noqa: E402
import repair_agent  # noqa: E402
from bench_common import extract_rtl  # noqa: E402
from backend import llm_client  # noqa: E402
from storage import RunKey, default_version_id, put_run, write_index  # noqa: E402
from storage.layout import slot_for  # noqa: E402

INPUTS = os.path.join(ROOT, "configs", "vericodegen", "main85_inputs.jsonl")
EXTRACTOR = "prompts.extract_module"

HELDOUT = os.path.join(ROOT, "data", "repairbench_heldout.jsonl")
# The `diag*` grammar: every arm carries the tool diagnostic, `spec` adds intent
# and `locate` adds the oracle line. Names match storage.layout.signal_type and
# the arm names in configs/experiments.json (base_diag, base_diag_spec, ...).
DIAG_ARMS = [("diag", False, False), ("diag_spec", True, False),
             ("diag_locate", False, True), ("diag_spec_locate", True, True)]
DIAG_EXTRACTOR = "repair_agent._split_explanation_and_code+bench_common.extract_rtl"

# Per prompt style: how the turn is built, how the completion is parsed, and the
# completion budget. All three are ported from repairbench_eval.py's dispatch --
# a ledger built with the wrong style would look entirely valid and be silently
# incomparable to the banked arm it claims to reproduce.
#
# VeriReason gets 2048 because reasoning precedes the code; at 1024 longer items
# truncate before the module and the arm measures the budget, not the model.
# (VERIREASON_DUMP is not honoured here -- the ledger *is* the dump, and unlike
# that append-mode file every record carries a case_id and an arm.)
PROMPT_STYLES = {
    "default":    {"max_tokens": 1024, "extractor": DIAG_EXTRACTOR},
    "origen":     {"max_tokens": 1024,
                   "extractor": "prompts.extract_module|bench_common.extract_rtl"},
    "verireason": {"max_tokens": 2048,
                   "extractor": "prompts.extract_verireason|bench_common.extract_rtl"},
}


def build_turn(style: str, broken: str, err: str, spec, loc):
    """(system, user) for one prompt style -- the same call the driver makes."""
    if style == "origen":
        return prompts.origen_prompt(broken, err, spec=spec, locate=loc)
    if style == "verireason":
        return prompts.verireason_prompt(broken, err, spec=spec, locate=loc)
    return (repair_agent._REPAIR_SYSTEM,
            repair_agent._build_prompt(broken, err, spec=spec, locate=loc))


def extract_for(style: str, raw: str):
    """Parse a completion the way the driver parses it for this style."""
    if style == "origen":
        return prompts.extract_module(raw) or extract_rtl(raw)
    if style == "verireason":
        return prompts.extract_verireason(raw) or extract_rtl(raw)
    _, fixed = repair_agent._split_explanation_and_code(raw)
    return extract_rtl(fixed) if fixed else None

_print_lock = threading.Lock()


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode()).hexdigest()


def load_cases(path: str, case_set: str | None = None) -> list[dict]:
    cases = [json.loads(line) for line in open(path) if line.strip()]
    if case_set:
        keep = set(json.load(open(case_set)))
        cases = [c for c in cases
                 if c["internal_metadata"]["case_id"] in keep
                 or c["internal_metadata"]["anonymous_id"] in keep]
        if not cases:
            raise SystemExit(f"--case-set {case_set} matched no case in {path}")
    return cases


def load_heldout(path: str, bucket: str | None = None) -> list[dict]:
    """The 459-case held-out set. `bucket` narrows to 'lint' or 'semantic'."""
    items = [json.loads(line) for line in open(path) if line.strip()]
    if bucket:
        items = [i for i in items if i.get("bucket") == bucket]
        if not items:
            raise SystemExit(f"--bucket {bucket} matched no item in {path}")
    return items


_ORACLE_LINES: dict[str, str] | None = None


def _oracle_line_fallback(case_id: str) -> str | None:
    """The 2x2 grid's oracle buggy line for `case_id`, or None.

    Read lazily and cached: the held-out semantic cases and the main85 grid are
    the same 85 cases under two prompt grammars, and share case_ids, so the
    grid's `oracle_location` is authoritative wherever the diff heuristic in
    `repairbench_eval.buggy_line` comes up empty. Only the line text is used --
    the grid also carries a line number, which the diag grammar's locate block
    has no slot for, so the two arms disclose the same fact either way.
    """
    global _ORACLE_LINES
    if _ORACLE_LINES is None:
        _ORACLE_LINES = {}
        try:
            with open(INPUTS) as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    j = json.loads(line)
                    ol = (j.get("model_input") or {}).get("oracle_location") or {}
                    text = (ol.get("broken_line") or "").strip()
                    if text:
                        _ORACLE_LINES[j["internal_metadata"]["case_id"]] = text
        except OSError:
            pass  # grid inputs absent: fall back to the empty-block behaviour
    return _ORACLE_LINES.get(case_id)


def generate_one_diag(item: dict, arm: str, use_spec: bool, use_loc: bool,
                      specs: dict, args) -> dict:
    """One held-out model call -> one ledger record. Never raises.

    Mirrors repairbench_eval.eval_one's generation half exactly: same prompt
    builder, same system text, same max_tokens, same extraction chain. The
    scoring half is deliberately absent -- that is the point of the ledger.
    """
    broken, err = rbe.parse_item(item)
    seed_id = item["seed_id"]
    # buggy_line reads the golden off disk, so --locate is a generation-time
    # dependency on scoring-side data. It returns None when the seed has no
    # golden, which silently empties the block -- recorded as had_locate.
    #
    # It also returns None when the mutated line happens to appear verbatim
    # somewhere else in the golden: the heuristic is "present in broken, absent
    # from golden", and a repeated line defeats it (2 of the 85 semantic cases
    # -- an off_by_one_literal and an op_swap in a counter). Those cases would
    # silently ship an empty locate block, making the arm a duplicate of its
    # no-locate twin and biasing the measured effect toward null. The 2x2 grid's
    # inputs carry the authoritative oracle for exactly these cases, keyed by
    # the same case_id, so fall back to it rather than degrade the arm. Where
    # both are available they agree on all 83.
    loc = rbe.buggy_line(broken, seed_id) if use_loc else None
    if use_loc and loc is None:
        loc = _oracle_line_fallback(item["id"])
    if args.anon:
        broken, err = rbe.anonymize(broken, err, seed_id)
    spec = rbe.spec_for(seed_id, specs) if use_spec else None
    style = args.prompt_style
    system, user = build_turn(style, broken, err, spec, loc)
    extractor = PROMPT_STYLES[style]["extractor"]

    rec = {
        "case_id": item["id"],       # held-out items key on `id`, not `case_id`
        "seed_id": seed_id,
        "bucket": item.get("bucket"),
        "mutation": item.get("mutation"),
        "arm": arm,
        "anon": bool(args.anon),
        "prompt_style": style,
        "had_spec": spec is not None,
        "had_locate": loc is not None,
        "prompt_sha256": _sha(user),
        "system_sha256": _sha(system),
    }
    try:
        texts = llm_client.generate_strict(
            user, model=args.model, n=1, temp=args.temp,
            max_tokens=args.max_tokens, system=system)
        raw = texts[0] if texts else ""
    except Exception as exc:  # noqa: BLE001 - a failed call is data, not a crash
        status = llm_client.last_call_status()
        rec.update({"raw": None, "extracted_rtl": None, "parsed_ok": False,
                    "extractor": extractor, "finish_reason": None,
                    "usage": None, "error_type": type(exc).__name__,
                    "error": str(exc)[:300],
                    "served_model": status.get("served_model"),
                    "elapsed_ms": status.get("elapsed_ms")})
        return rec

    status = llm_client.last_call_status()
    finishes = status.get("finish_reasons") or []
    fixed = extract_for(style, raw)
    if fixed:
        fixed = fixed.rstrip() + "\n"  # avoid cosmetic EOFNEWLINE lint failure
    rec.update({
        "raw": raw,
        "raw_sha256": _sha(raw),
        "extracted_rtl": fixed or None,
        "extractor": extractor,
        "parsed_ok": bool(fixed),
        "finish_reason": finishes[0] if finishes else None,
        "usage": status.get("usage"),
        "error_type": None,
        "error": None,
        "served_model": status.get("served_model"),
        "elapsed_ms": status.get("elapsed_ms"),
    })
    return rec


def generate_one(case: dict, arm: str, use_spec: bool, use_loc: bool,
                 model: str, temp: float, max_tokens: int) -> dict:
    """One model call -> one ledger record. Never raises: a failure is a record.

    `last_call_status()` is read here, inside the worker, because it is
    thread-local by design -- reading it after the pool joins would attribute
    another worker's finish_reason and token counts to this case.
    """
    meta, mi = case["internal_metadata"], case["model_input"]
    user = prompts.spec_locate_prompt(mi["broken_rtl"], mi.get("full_spec"),
                                      mi.get("oracle_location"), use_spec, use_loc)
    rec = {
        "case_id": meta["case_id"],
        "anon_id": meta["anonymous_id"],
        "seed_id": meta["source_seed_id"],
        "cluster_id": meta["cluster_id"],
        "mutation_family": meta["mutation_family"],
        "arm": arm,
        "prompt_sha256": _sha(user),
        "system_sha256": _sha(prompts.SPEC_LOCATE_SYSTEM),
    }
    try:
        # generate_strict, never generate: the non-strict path silently returns
        # canned RTL from serve_stub on any failure, which would land in the
        # ledger indistinguishable from a real completion.
        texts = llm_client.generate_strict(
            user, model=model, n=1, temp=temp, max_tokens=max_tokens,
            system=prompts.SPEC_LOCATE_SYSTEM)
        raw = texts[0] if texts else ""
    except Exception as exc:  # noqa: BLE001 - a failed call is data, not a crash
        status = llm_client.last_call_status()
        rec.update({"raw": None, "extracted_rtl": None, "parsed_ok": False,
                    "extractor": EXTRACTOR, "finish_reason": None, "usage": None,
                    "error_type": type(exc).__name__, "error": str(exc)[:300],
                    "served_model": status.get("served_model"),
                    "elapsed_ms": status.get("elapsed_ms")})
        return rec

    status = llm_client.last_call_status()
    finishes = status.get("finish_reasons") or []
    mod = prompts.extract_module(raw)
    rec.update({
        "raw": raw,
        "raw_sha256": _sha(raw),
        "extracted_rtl": mod,
        "extractor": EXTRACTOR,
        "parsed_ok": bool(mod),
        "finish_reason": finishes[0] if finishes else None,
        "usage": status.get("usage"),
        "error_type": None,
        "error": None,
        "served_model": status.get("served_model"),
        "elapsed_ms": status.get("elapsed_ms"),
    })
    return rec


def run_arm(cases: list[dict], arm: str, use_spec: bool, use_loc: bool,
            args, gen=None) -> list[dict]:
    """`gen(case, arm, use_spec, use_loc) -> record`; defaults to the main85 path."""
    if gen is None:
        def gen(case, arm, use_spec, use_loc):
            return generate_one(case, arm, use_spec, use_loc,
                                args.model, args.temp, args.max_tokens)

    records: list[dict] = []
    done = 0

    def work(case):
        nonlocal done
        rec = gen(case, arm, use_spec, use_loc)
        with _print_lock:
            done += 1
            if done % 10 == 0 or done == len(cases):
                print(f"  {arm}: {done}/{len(cases)}", flush=True)
        return rec

    if args.workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            records = list(ex.map(work, cases))
    else:
        records = [work(c) for c in cases]
    # ex.map yields in input order, so the ledger is deterministic in case order.
    return records


def bank(records: list[dict], arm: str, args) -> str:
    key = RunKey(model=args.slot or slot_for(args.model),
                 version=args.version or default_version_id(),
                 signal=arm)
    meta = {
        "arm": arm,
        "generation": {
            # Endpoint identity, never the credential.
            "endpoint": os.environ.get("RTLREPAIR_LLM_URL") or llm_client.DEFAULT_LLM_URL,
            "requested_model": args.model,
            "temp": args.temp,
            "max_tokens": args.max_tokens,
            "n": 1,
            "workers": args.workers,
            "prompt_grammar": (f"build_turn({args.prompt_style})" if args.bench
                               else "prompts.spec_locate_prompt"),
            "extractor": (PROMPT_STYLES[args.prompt_style]["extractor"]
                          if args.bench else EXTRACTOR),
            "cases": os.path.relpath(args.cases, ROOT),
            "n_cases": len(records),
            # `anon` and `prompt_style` only exist in the held-out grammar; the
            # main85 prompts carry no VerilogEval identity to strip and have one
            # fixed grammar. system_sha256 is per-record there (styles differ).
            **({"anon": bool(args.anon), "prompt_style": args.prompt_style}
               if args.bench else
               {"system_sha256": _sha(prompts.SPEC_LOCATE_SYSTEM)}),
        },
        "scored": False,
    }
    if args.bench:
        buckets: dict[str, int] = {}
        for r in records:
            buckets[r.get("bucket") or "?"] = buckets.get(r.get("bucket") or "?", 0) + 1
        meta["generation"]["by_bucket"] = buckets
    dest = put_run(key, ledger=records, meta=meta, root=args.root,
                   overwrite=args.overwrite)
    write_index(args.root)
    return dest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True,
                    help="model id as the endpoint serves it (e.g. OriGen_Fix)")
    ap.add_argument("--slot", help="override the banked model slot; by default the "
                    "slot is derived from --model via storage.layout.MODEL_ALIASES")
    ap.add_argument("--arms", help="comma-separated subset of "
                    + ",".join(a for a, _, _ in prompts.SPEC_LOCATE_ARMS)
                    + " (or of " + ",".join(a for a, _, _ in DIAG_ARMS)
                    + " under --bench)")
    ap.add_argument("--bench", nargs="?", const=HELDOUT, default=None,
                    metavar="JSONL",
                    help="generate the held-out set under the diag* grammar "
                         f"instead of the main85 grid (default {os.path.relpath(HELDOUT, ROOT)})")
    ap.add_argument("--bucket", choices=("lint", "semantic"),
                    help="with --bench, restrict to one bucket (default: both)")
    ap.add_argument("--prompt-style", choices=tuple(PROMPT_STYLES), default="default",
                    help="with --bench, the prompt grammar. Must match the arm's "
                         "prompt_style in configs/experiments.json -- a mismatch "
                         "produces a valid-looking ledger that is silently "
                         "incomparable to the arm it claims to reproduce")
    ap.add_argument("--no-anon", dest="anon", action="store_false",
                    help="with --bench, keep the VerilogEval identity. The banked "
                         "arms were all --anon (configs/experiments.json defaults)")
    ap.set_defaults(anon=True)
    ap.add_argument("--cases", default=None, help="cases JSONL (overrides the default "
                    "for whichever grammar is selected)")
    ap.add_argument("--case-set", help="JSON list of case_ids to restrict to")
    ap.add_argument("--limit", type=int, default=0, help="first N cases (smoke test)")
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--max-tokens", type=int, default=0,
                    help="default 4096 for the main85 grid, 1024 under --bench "
                         "(the value repair_agent.repair hardcodes)")
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent calls; keep arms you compare on the same value")
    ap.add_argument("--out", help="write candidates.jsonl here instead of banking")
    ap.add_argument("--bank", action="store_true", help="bank into the results tree")
    ap.add_argument("--version", help="version_id (default YYYYMMDD-<git sha>)")
    ap.add_argument("--root", help="tree root (default $RTLREPAIR_STORE)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)

    if not (args.bank or args.out):
        raise SystemExit("pass --bank (into the tree) or --out <path>")
    if not args.bench and (args.bucket or not args.anon
                           or args.prompt_style != "default"):
        raise SystemExit("--bucket, --no-anon and --prompt-style apply only to --bench")
    if not args.max_tokens:
        args.max_tokens = (PROMPT_STYLES[args.prompt_style]["max_tokens"]
                           if args.bench else 4096)

    all_arms = DIAG_ARMS if args.bench else prompts.SPEC_LOCATE_ARMS
    arms = all_arms
    if args.arms:
        want = {a.strip() for a in args.arms.split(",")}
        arms = [a for a in all_arms if a[0] in want]
        unknown = want - {a[0] for a in all_arms}
        if unknown:
            raise SystemExit(f"unknown arm(s): {sorted(unknown)}")

    if args.bench:
        args.cases = args.cases or args.bench
        cases = load_heldout(args.cases, args.bucket)
        specs = rbe.load_specs()
        if not specs:
            raise SystemExit(
                f"no specs loaded from {rbe.EVAL_TASKS}; a --spec arm would be "
                "silently identical to its no-spec twin. Aborting.")

        def gen(case, arm, use_spec, use_loc):
            return generate_one_diag(case, arm, use_spec, use_loc, specs, args)
    else:
        args.cases = args.cases or INPUTS
        cases = load_cases(args.cases, args.case_set)
        gen = None
    if args.limit:
        cases = cases[:args.limit]

    health = llm_client.health_check()
    if not health.get("ok"):
        raise SystemExit(
            f"endpoint not reachable: {health.get('error')}\n"
            f"  RTLREPAIR_LLM_URL={os.environ.get('RTLREPAIR_LLM_URL', '(default localhost)')}\n"
            "Generation is aborted rather than run against a stub.")
    print(f"{len(cases)} cases x {len(arms)} arm(s) = {len(cases) * len(arms)} calls "
          f"on {args.model} @ {os.environ.get('RTLREPAIR_LLM_URL') or llm_client.DEFAULT_LLM_URL}")

    for arm, use_spec, use_loc in arms:
        records = run_arm(cases, arm, use_spec, use_loc, args, gen)
        ok = sum(1 for r in records if r["parsed_ok"])
        err = sum(1 for r in records if r["error"])
        trunc = sum(1 for r in records if r["finish_reason"] == "length")
        print(f"{arm}: {len(records)} calls, {ok} parsed, {err} errored, {trunc} truncated")
        if args.bench and use_loc:
            # buggy_line needs the golden, and its diff heuristic also misses a
            # mutated line that recurs verbatim in the golden; the 2x2 grid's
            # oracle covers the latter. Anything still empty here makes the arm
            # a duplicate of its twin on that case, so say so loudly.
            noloc = sum(1 for r in records if not r["had_locate"])
            if noloc:
                print(f"  WARNING: {noloc}/{len(records)} had no oracle line "
                      "(no golden on disk, and no grid oracle to fall back to) "
                      "-- locate block was empty, so those cases duplicate the "
                      "no-locate arm")
        if args.out:
            # A directory always gets one file per arm. Keying this off the arm
            # *count* meant a single-arm run wrote straight to args.out -- so
            # pointing it at a directory raised IsADirectoryError only after
            # every call had been paid for and the records were unrecoverable.
            path = (os.path.join(args.out, f"candidates_{arm}.jsonl")
                    if len(arms) > 1 or os.path.isdir(args.out) else args.out)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as fh:
                for r in records:
                    fh.write(json.dumps(r) + "\n")
            print(f"  wrote {path}")
        if args.bank:
            print(f"  banked -> {bank(records, arm, args)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
