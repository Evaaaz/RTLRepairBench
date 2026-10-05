"""Evaluate a model's tool-in-the-loop repair on the held-out categorized RepairBench.

For each (broken RTL, tool error), ask the model (via repair_agent.repair)
for a fix, then RE-VALIDATE with the same external tool that defined the bug:
  - lint-class:     the fixed RTL must pass `verilator --lint-only` (was failing).
  - semantic-class: the fixed RTL must compile AND match the golden under
                    differential iverilog simulation (was diverging).

Ablation flags:
  --anon   Strip the VerilogEval problem identity (rename the module to TopModule
           and scrub the Prob-id/filename out of the tool error) so the model
           cannot recall the golden from a leaked name. Establishes the true
           leak-free repair floor.
  --spec   Inject the design's natural-language specification (the VerilogEval
           `instruction`, joined by seed_id) into the repair prompt -- the
           spec-conditioned arm that tests whether intent closes the gap.
  --temp   Sampling temperature for the repair model (0.0 = greedy, for
           deterministic paired comparisons).

Generation goes to the served model at RTLREPAIR_LLM_URL (pass env inline);
validation is local.

Usage:
  RTLREPAIR_LLM_URL=... RTLREPAIR_LLM_API_KEY=... \
    .venv/bin/python benchmarks/repairbench_eval.py --model base --anon --temp 0.0
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import threading
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))   # .../code/benchmarks
CODE = os.path.dirname(HERE)                          # .../code  (holds the FIXED gates)
if CODE not in sys.path:
    sys.path.insert(0, CODE)
if HERE not in sys.path:
    sys.path.append(HERE)

from benchmarks import prompts  # noqa: E402
from benchmarks.bench_common import extract_rtl  # noqa: E402
from benchmarks import repair_agent  # noqa: E402  (moved out of backend/ upstream)
from datagen.diff_testbench import parse_module, run_diff_test  # noqa: E402
from datagen.run_verilator_for_repairs import lint_source  # noqa: E402
from project_paths import data_root, output_root  # noqa: E402

# --- location-independent data resolution --------------------------------------
# All data inputs come from $RTLREPAIR_DATA. The shared resolver supports both
# this tree's historical nested checkout and a standalone extraction.
DATA = str(data_root())
OUT_DIR = os.environ.get("RTLREPAIR_OUT") or str(output_root() / "repairbench")
SEEDS = os.environ.get(
    "RTLREPAIR_SEEDS", str(output_root() / "repairbench" / "seeds")
)
EVAL_TASKS = os.environ.get("RTLREPAIR_EVAL_TASKS", os.path.join(DATA, "eval_tasks.jsonl"))
ANON_NAME = "TopModule"

_RTL = re.compile(r"```systemverilog\n(.*?)```", re.S)
_ERR = re.compile(r"reported:\s*```\n?(.*?)```", re.S)


def parse_item(item: dict) -> tuple[str, str]:
    """Pull the broken RTL and the tool error out of the benchmark prompt."""
    user = item["messages"][1]["content"]
    rtl = _RTL.search(user)
    err = _ERR.search(user)
    return (rtl.group(1).strip() if rtl else ""), (err.group(1).strip() if err else "")


def anonymize(broken: str, err: str, seed_id: str) -> tuple[str, str]:
    """Remove the leaked VerilogEval identity: module name + filename in the error."""
    pat = re.compile(r"\b" + re.escape(seed_id) + r"\b")
    return pat.sub(ANON_NAME, broken), pat.sub(ANON_NAME, err)


def load_specs() -> dict:
    """Map seed_id -> natural-language spec (VerilogEval `instruction`).

    Goldens are named `<Prob>_..._ref`; eval task ids drop the `_ref`. We index
    by both the raw id and a few normalized variants so the join is robust.
    """
    specs: dict[str, str] = {}
    if not os.path.exists(EVAL_TASKS):
        return specs
    for line in open(EVAL_TASKS):
        d = json.loads(line)
        instr = (d.get("instruction") or "").strip()
        if not instr:
            continue
        rawid = str(d.get("id", ""))
        key = rawid.split(":")[-1]  # strip any `verilogeval:` prefix
        specs[key] = instr
        specs[key + "_ref"] = instr
    return specs


def spec_for(seed_id: str, specs: dict):
    return (
        specs.get(seed_id)
        or specs.get(seed_id.replace("_ref", ""))
        or specs.get(seed_id.split(":")[-1])
    )


def golden_src(seed_id: str, anon: bool) -> str | None:
    path = os.path.join(SEEDS, seed_id + ".sv")
    if not os.path.exists(path):
        return None
    src = open(path).read()
    if anon:
        src = re.sub(r"\b" + re.escape(seed_id) + r"\b", ANON_NAME, src)
    return src


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def semantic_golden_errors(
    items: list[dict],
    *,
    seeds_dir: str | Path = SEEDS,
    require_manifest: bool = True,
) -> list[str]:
    """Return coverage/hash errors for the semantic golden reference set."""

    required = sorted(
        {str(item["seed_id"]) for item in items if item.get("bucket") == "semantic"}
    )
    if not required:
        return []
    root = Path(seeds_dir)
    missing = [seed_id for seed_id in required if not (root / f"{seed_id}.sv").is_file()]
    errors: list[str] = []
    if missing:
        errors.append(
            f"missing {len(missing)}/{len(required)} semantic goldens: "
            + ", ".join(missing[:8])
        )

    manifest_path = root / "_manifest.sha256.json"
    if not manifest_path.is_file():
        if require_manifest:
            errors.append(f"golden SHA-256 manifest is missing: {manifest_path}")
        return errors
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        errors.append(f"golden manifest is unreadable: {type(exc).__name__}: {exc}")
        return errors
    if not isinstance(manifest, dict):
        errors.append("golden manifest must be a JSON object")
        return errors
    for seed_id in required:
        path = root / f"{seed_id}.sv"
        if not path.is_file():
            continue
        expected = manifest.get(seed_id)
        if not isinstance(expected, str):
            errors.append(f"golden manifest has no hash for {seed_id}")
        elif _sha256_file(path) != expected:
            errors.append(f"golden SHA-256 mismatch for {seed_id}")
    return errors


def benchmark_structure_errors(items: list[dict]) -> list[str]:
    """Return schema errors that must be fixed before any model calls are made."""

    if not items:
        return ["benchmark contains zero cases"]
    errors: list[str] = []
    ids = [item.get("id") for item in items]
    if any(not isinstance(case_id, str) or not case_id.strip() for case_id in ids):
        errors.append("every benchmark case must have a non-empty string id")
    if len(set(ids)) != len(ids):
        errors.append("benchmark case ids are not unique")
    for index, item in enumerate(items):
        label = item.get("id") or f"row {index}"
        if item.get("bucket") not in {"lint", "semantic"}:
            errors.append(f"{label}: bucket must be lint or semantic")
        if not isinstance(item.get("seed_id"), str) or not item["seed_id"].strip():
            errors.append(f"{label}: seed_id is missing")
        if not isinstance(item.get("mutation"), str) or not item["mutation"].strip():
            errors.append(f"{label}: mutation is missing")
        try:
            broken, tool_error = parse_item(item)
        except (IndexError, KeyError, TypeError):
            broken, tool_error = "", ""
        if not broken:
            errors.append(f"{label}: broken RTL is not parseable")
        if not tool_error:
            errors.append(f"{label}: verifier diagnostic is not parseable")
    return errors


def run_completeness_errors(
    items: list[dict], records: list[dict], *, require_specs: bool
) -> list[str]:
    """Return fail-closed arm-level completeness errors."""

    errors: list[str] = []
    if not items:
        errors.append("benchmark contains zero cases")
    item_ids = [item.get("id") for item in items]
    record_ids = [record.get("id") for record in records]
    if any(item_id is None for item_id in item_ids):
        errors.append("benchmark contains a case with no id")
    if len(set(item_ids)) != len(item_ids):
        errors.append("benchmark case ids are not unique")
    if len(records) != len(items):
        errors.append(f"record count {len(records)} does not match case count {len(items)}")
    if collections.Counter(record_ids) != collections.Counter(item_ids):
        errors.append("result ids do not exactly match benchmark ids")
    no_llm = sum(not bool(record.get("used_llm")) for record in records)
    if no_llm:
        errors.append(f"{no_llm}/{len(records)} cases have no successful model response")
    if require_specs:
        no_spec = sum(not bool(record.get("had_spec")) for record in records)
        if no_spec:
            errors.append(f"{no_spec}/{len(records)} spec-arm cases have no joined specification")
    return errors


def _write_result(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    with temporary.open("w") as handle:
        json.dump(payload, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def buggy_line(broken: str, seed_id: str) -> str | None:
    """Oracle localization: the single line present in the broken RTL but not the golden.

    For single-edit mutations this is the mutated line. Leak-safe (the line content
    carries no Prob-id), so it is used verbatim even under --anon.
    """
    golden = golden_src(seed_id, anon=False)
    if golden is None:
        return None
    g = {l.strip() for l in golden.splitlines() if l.strip()}
    for l in broken.splitlines():
        s = l.strip()
        if s and s not in g:
            return s
    return None


def recovers_lint(seed_id: str, fixed: str, anon: bool) -> bool:
    name = (ANON_NAME if anon else seed_id) + ".sv"
    return not lint_source(name, fixed).failed


def recovers_semantic(seed_id: str, fixed: str, anon: bool) -> bool:
    golden = golden_src(seed_id, anon)
    if golden is None:
        return False
    info = parse_module(golden)
    if info is None:
        return False
    res = run_diff_test(golden, fixed, info)
    return bool(res.compiled and not res.diverged)


_AC = None


def claude_repair(broken: str, err: str, spec, model: str, locate=None):
    """Repair via the Anthropic API, reusing the exact repair prompt + system."""
    global _AC
    import anthropic
    if _AC is None:
        _AC = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    prompt = repair_agent._build_prompt(broken, err, spec=spec, locate=locate)
    kwargs = dict(
        model=model, max_tokens=1024,
        system=repair_agent._REPAIR_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    if not (("opus-4" in model) or ("sonnet-4" in model)):
        kwargs["temperature"] = 0.0  # newer Claude models deprecate temperature
    resp = _AC.messages.create(**kwargs)
    text = resp.content[0].text if resp.content else ""
    _, fixed = repair_agent._split_explanation_and_code(text)
    return fixed


def origen_repair(broken: str, err: str, spec, model: str, temp: float, locate=None):
    """Repair via a served model (vLLM/OpenAI-compatible) using OriGen_Fix's NATIVE
    prompt template. OriGen's format requires a task description, so this is run in
    the diagnostic+spec condition (see README). Served as chat, not raw completion —
    the header-priming of the original template is dropped; we ask for the full module.
    """
    from backend import llm_client
    system, user = prompts.origen_prompt(broken, err, spec=spec, locate=locate)
    outs = llm_client.generate_strict(user, model=model, n=1, temp=temp,
                                      max_tokens=1024, system=system)
    return outs[0] if outs else ""


# Serializes VERIREASON_DUMP appends. A raw completion can exceed the 8 KB
# stdio buffer, so one fh.write() may become several write() syscalls -- under
# --workers > 1 that lets two records interleave into a corrupt JSONL line.
_DUMP_LOCK = threading.Lock()


def verireason_repair(broken: str, err: str, spec, model: str, temp: float, locate=None):
    """Repair via a served model using VeriReason's NATIVE <think>/<answer> format.

    Completion budget defaults to 2048 (env VERIREASON_MAX_TOKENS): reasoning
    precedes the code, so the 1024 used elsewhere truncates before the module
    on longer items — that would measure the budget, not the model. Set
    VERIREASON_DUMP=<path.jsonl> to append every raw completion for a
    truncation audit (records with no closing </answer> are flagged).

    Give each arm its **own** dump path. The file is opened in append mode and
    carries no arm field, so pointing two arms at one path fuses them
    irreversibly — that is how the banked vr_raw_vrqwen.jsonl ended up holding
    both vrqwen arms (918 = 2x459) with no way to split it.
    """
    from backend import llm_client
    system, user = prompts.verireason_prompt(broken, err, spec=spec, locate=locate)
    max_toks = int(os.environ.get("VERIREASON_MAX_TOKENS", "2048"))
    outs = llm_client.generate_strict(user, model=model, n=1, temp=temp,
                                      max_tokens=max_toks, system=system)
    raw = outs[0] if outs else ""
    dump = os.environ.get("VERIREASON_DUMP")
    if dump and raw:
        # finish_reason is the only honest truncation signal: the model routinely
        # emits EOS without closing </answer>, so truncated_guess fires on ~99% of
        # completions while real max_tokens truncation is under 1%. Kept only for
        # continuity with dumps collected before finish_reason was recorded.
        fr = llm_client.last_call_status().get("finish_reasons") or []
        line = json.dumps({"truncated_guess": "</answer>" not in raw,
                           "finish_reason": fr[0] if fr else None,
                           "truncated": fr[:1] == ["length"],
                           "raw": raw}) + "\n"
        with _DUMP_LOCK, open(dump, "a") as fh:
            fh.write(line)
    return raw


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base")
    ap.add_argument("--backend", default="rtlrepair", choices=["rtlrepair", "claude"])
    ap.add_argument("--claude-model", default="claude-opus-4-8")
    ap.add_argument("--bench", default=os.path.join(DATA, "repairbench_heldout.jsonl"))
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--anon", action="store_true", help="strip leaked Prob-id / module name")
    ap.add_argument("--spec", action="store_true", help="inject the VerilogEval spec")
    ap.add_argument("--locate", action="store_true", help="inject the oracle buggy line")
    ap.add_argument("--prompt-style", default="default", choices=sorted(prompts.PROMPTERS),
                    help="prompt template; 'origen' uses OriGen_Fix's native format, "
                         "'verireason' uses VeriReason's think/answer format")
    ap.add_argument("--allow-empty-seeds", action="store_true",
                    help="permit semantic scoring with no goldens (returns 0 recovery) — off by default")
    ap.add_argument(
        "--allow-unverified-seeds",
        action="store_true",
        help="allow semantic goldens without the SHA-256 manifest; diagnostic use only",
    )
    ap.add_argument(
        "--allow-incomplete-run",
        action="store_true",
        help="write an arm with missing model responses or specs; diagnostic use only",
    )
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--workers", type=int, default=1,
                    help="concurrent in-flight items (1 = sequential, the default). "
                         "vLLM kernels are not batch-invariant, so greedy output can "
                         "differ between concurrency levels: keep every arm you intend "
                         "to compare on the SAME --workers value.")
    args = ap.parse_args()

    benchmark_items = [json.loads(line) for line in open(args.bench) if line.strip()]
    structure_errors = benchmark_structure_errors(benchmark_items)
    if structure_errors:
        sys.exit(
            "ERROR: benchmark preflight failed:\n  - "
            + "\n  - ".join(structure_errors)
        )
    items = benchmark_items
    if args.limit:
        items = items[: args.limit]
    limited_run = len(items) != len(benchmark_items)

    # Loud-fail rather than silently reporting false negatives when any semantic
    # golden is missing or has drifted from its validated manifest.
    has_semantic = any(it.get("bucket") == "semantic" for it in items)
    golden_errors = semantic_golden_errors(
        items,
        require_manifest=not args.allow_unverified_seeds,
    )
    if has_semantic and golden_errors and not args.allow_empty_seeds:
        sys.exit(
            "ERROR: semantic golden preflight failed:\n  - "
            + "\n  - ".join(golden_errors)
            + "\nRegenerate the validated goldens with make_seeds.py, or use an explicit "
            "diagnostic-only override."
        )
    specs = load_specs() if args.spec else {}
    if args.spec:
        matched = sum(1 for it in items if spec_for(it["seed_id"], specs))
        print(f"spec join: matched {matched}/{len(items)} items", flush=True)
        if matched != len(items) and not args.allow_incomplete_run:
            sys.exit(
                f"ERROR: spec arm joined only {matched}/{len(items)} specifications; "
                "refusing to spend model calls on an incomplete arm"
            )

    by = collections.defaultdict(lambda: {"n": 0, "recovered": 0, "used_llm": 0, "spec": 0})
    _progress = {"done": 0}
    _progress_lock = threading.Lock()

    def eval_one(idx_item):
        i, item = idx_item
        broken, err = parse_item(item)
        bucket, seed_id = item["bucket"], item["seed_id"]
        locate = buggy_line(broken, seed_id) if args.locate else None
        if args.anon:
            broken, err = anonymize(broken, err, seed_id)
        spec = spec_for(seed_id, specs) if args.spec else None
        fixed, used_llm = "", False
        try:
            if args.backend == "claude":
                raw_fixed = claude_repair(broken, err, spec, args.claude_model, locate=locate)
                used_llm = bool(raw_fixed)
                if raw_fixed:
                    fixed = extract_rtl(raw_fixed)
            elif args.prompt_style == "origen":
                raw_fixed = origen_repair(broken, err, spec, args.model, args.temp, locate=locate)
                used_llm = bool(raw_fixed)
                if raw_fixed:
                    fixed = prompts.extract_module(raw_fixed) or extract_rtl(raw_fixed)
            elif args.prompt_style == "verireason":
                raw_fixed = verireason_repair(broken, err, spec, args.model, args.temp, locate=locate)
                used_llm = bool(raw_fixed)
                if raw_fixed:
                    fixed = prompts.extract_verireason(raw_fixed) or extract_rtl(raw_fixed)
            else:
                rr = repair_agent.repair(broken, err, model=args.model, temperature=args.temp, spec=spec, locate=locate)
                used_llm = bool(getattr(rr, "used_llm", False))
                if rr.fixed_rtl:
                    fixed = extract_rtl(rr.fixed_rtl)
            if fixed:
                fixed = fixed.rstrip() + "\n"  # avoid cosmetic EOFNEWLINE lint failure
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] repair error: {exc}", flush=True)
        ok = False
        if fixed:
            ok = recovers_lint(seed_id, fixed, args.anon) if bucket == "lint" \
                else recovers_semantic(seed_id, fixed, args.anon)
        # A Tier-S verdict is an estimate, so any arm we might want to re-score at the
        # stronger tier has to keep its candidate. Off by default: writing RTL next to a
        # scored arm changes what the release payload contains.
        _keep = os.environ.get("RTLREPAIR_KEEP_CANDIDATES", "").strip()
        if _keep and fixed:
            _d = Path(_keep)
            _d.mkdir(parents=True, exist_ok=True)
            (_d / f"{item.get('id')}.sv").write_text(fixed, encoding="utf-8")
        rec = {
            "id": item.get("id"), "bucket": bucket, "mutation": item["mutation"],
            "seed_id": seed_id, "recovered": ok, "used_llm": used_llm,
            "had_spec": spec is not None,
        }
        with _progress_lock:
            _progress["done"] += 1
            done = _progress["done"]
        if done % 25 == 0:
            print(f"  {done}/{len(items)} done", flush=True)
        return rec

    if args.workers > 1:
        # ThreadPoolExecutor.map yields results in INPUT order, so `records`
        # stays item-ordered no matter what order the workers finish in. The
        # scoring gates each mkdtemp their own workdir, so they do not collide.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
            records = list(ex.map(eval_one, enumerate(items)))
    else:
        records = [eval_one(x) for x in enumerate(items)]

    # Aggregate after the fact so counters never depend on completion order.
    for rec in records:
        b = by[rec["bucket"]]
        b["n"] += 1
        b["recovered"] += int(rec["recovered"])
        b["used_llm"] += int(rec["used_llm"])
        b["spec"] += int(rec["had_spec"])

    model_label = args.claude_model if args.backend == "claude" else args.model
    summary = {
        "model": model_label, "backend": args.backend, "n": len(items),
        "anon": args.anon, "spec": args.spec, "locate": args.locate, "temp": args.temp,
        "by_bucket": {},
    }
    for b, d in by.items():
        summary["by_bucket"][b] = {
            "n": d["n"], "recovered": d["recovered"],
            "recovery_rate": round(d["recovered"] / d["n"], 4) if d["n"] else 0.0,
            "used_llm_rate": round(d["used_llm"] / d["n"], 4) if d["n"] else 0.0,
            "spec_rate": round(d["spec"] / d["n"], 4) if d["n"] else 0.0,
        }
    completeness_errors = run_completeness_errors(
        items, records, require_specs=args.spec
    )
    diagnostic_overrides = []
    if has_semantic and args.allow_empty_seeds:
        diagnostic_overrides.append("allow_empty_seeds")
    if has_semantic and args.allow_unverified_seeds:
        diagnostic_overrides.append("allow_unverified_seeds")
    if args.allow_incomplete_run:
        diagnostic_overrides.append("allow_incomplete_run")
    if limited_run:
        diagnostic_overrides.append(
            f"limit:{len(items)}/{len(benchmark_items)}"
        )
    summary["completeness"] = {
        "complete": not completeness_errors,
        "errors": completeness_errors,
        "diagnostic_overrides": diagnostic_overrides,
        "paper_valid": not completeness_errors and not diagnostic_overrides,
    }
    print(json.dumps(summary, indent=2))
    safe = re.sub(r"[^A-Za-z0-9]+", "-", model_label)
    tag = safe + ("_anon" if args.anon else "") + ("_spec" if args.spec else "") + ("_loc" if args.locate else "")
    if args.prompt_style != "default":
        tag += "_" + args.prompt_style
    if limited_run:
        tag += f"_limit{len(items)}"
    out = Path(args.out or os.path.join(OUT_DIR, f"rb_{tag}.json"))
    payload = {"summary": summary, "records": records}
    if completeness_errors and not args.allow_incomplete_run:
        incomplete = out.with_name(f"{out.stem}.incomplete{out.suffix}")
        _write_result(incomplete, payload)
        sys.exit(
            "ERROR: arm completeness audit failed:\n  - "
            + "\n  - ".join(completeness_errors)
            + f"\nIncomplete diagnostic output was preserved at {incomplete}; "
            "the canonical output was not overwritten."
        )
    _write_result(out, payload)
    print("wrote", out)


if __name__ == "__main__":
    main()
