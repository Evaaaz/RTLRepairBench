#!/usr/bin/env python3
"""X3 -- the counterexample contrast on the Modal-served open-weight models.

Same experiment as ``x1_counterexample_power.py``, run on the four open-weight
models whose Counterexample cell in the decomposition table is blank. Kept as a
separate script rather than another env-var branch in x1 for two reasons:

* x1 writes to ``generated/counterexample/<tag>/``, which is **git-tracked** and
  holds the published llama / gpt-4.1 / gpt-4o-mini cells. A forgotten tag there
  silently replaces a number that is in the paper. This script writes to its own
  tree and refuses to overwrite an existing result.
* The open-weight arms were not generated in x1's prompt dialect. VeriReason and
  VR-Qwen are served raw-completion with a think/answer contract; x1's single
  hardcoded system text and its bare ``(module.*?endmodule)`` regex would run
  them off-distribution *and* mis-extract, grabbing the first module out of the
  reasoning block. Each model here keeps the system text and extractor its own
  banked arms used.

Design, paired and otherwise identical to the published contrast: take every
banked ``diag`` baseline the formal oracle refuted, replay the proof's own
witness, and ask for a revision twice --

  concrete : original module + previous candidate + the witness trace
  generic  : original module + previous candidate + "it still did not pass"

so the contrast isolates the executable trace from mere re-exposure. Scored by
reset-aware differential simulation against the golden; reported as exact
McNemar on the discordant pairs. Exploratory: nothing here enters the
confirmatory family.

Inputs are all local and already banked -- no formal toolchain is needed, because
the witnesses were retained when the Tier F verdicts were produced:

  generated/tierf_openweight/results.jsonl                        which baselines were refuted
  generated/tierf_openweight/jobs.jsonl                           the failing candidate + golden
  generated/tierf_openweight/artifacts/<call>/formal/replay/witness.json   the trace

Usage:

    python x3_counterexample_openweight.py --model base \\
        --served-model base --endpoint https://<app>.modal.run/v1
"""
from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import sys
import threading
from collections import Counter
from math import comb
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from project_paths import output_root, project_root  # noqa: E402

SOURCE_ROOT = str(project_root())
ROOT = str(output_root().parent)
for _p in (str(_CODE_DIR), str(_CODE_DIR / "benchmarks"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import prompts  # noqa: E402
import repairbench_eval as rbe  # noqa: E402
import reset_aware_sim  # noqa: E402
from backend import llm_client  # noqa: E402

# The generic repair system text, byte-identical to the one x1 uses for the
# published rows, so base/v5 stay comparable to them.
GENERIC_SYSTEM = (
    "You are an expert RTL debugging engineer. Repair the supplied compile-clean "
    "SystemVerilog module. Return the complete replacement module, from `module` through "
    "`endmodule`; never return a patch, explanation, testbench, or placeholder. Preserve "
    "the given module interface and emit synthesizable RTL.")

# VeriReason's own contract. Appending its think/answer instruction is what keeps
# the revision on-distribution and makes extract_verireason applicable; without
# it the model still reasons, and the reasoning block gets extracted as the fix.
VERIREASON_SYSTEM = "Please act as a professional verilog designer."
VERIREASON_TAIL = (
    "\n\nFirst reason about the bug inside <think> </think> tags, then output the "
    "complete corrected module inside <answer> </answer> tags in a "
    "```verilog code fence.")

ORIGEN_SYSTEM = "You are a professional Verilog designer."

# Per model slot: the served-arm conventions its banked baseline arm used.
STYLES = {
    "base":       {"system": GENERIC_SYSTEM,     "style": "default",    "max_tokens": 1024},
    "v5":         {"system": GENERIC_SYSTEM,     "style": "default",    "max_tokens": 1024},
    "verireason": {"system": VERIREASON_SYSTEM,  "style": "verireason", "max_tokens": 2048},
    "vrqwen":     {"system": VERIREASON_SYSTEM,  "style": "verireason", "max_tokens": 2048},
    # OriGen emits raw Verilog after "### Response:" with no fence, which
    # extract_module already handles; its system text is minimal because the
    # native instruction is self-contained.
    "origen":     {"system": ORIGEN_SYSTEM,      "style": "origen",     "max_tokens": 1024},
}

_print_lock = threading.Lock()


def extract_for(style: str, raw: str):
    if style == "verireason":
        return prompts.extract_verireason(raw)
    return prompts.extract_module(raw)


def witness_text(call_id: str, artifacts: Path) -> str | None:
    """Render the retained proof witness as the concrete counterexample.

    The open-weight verdicts were produced by the yosys `sat` / smtbmc path,
    which writes `formal/replay/witness.json` -- an Icarus-replayed trace -- not
    the sby `trace_aiw.yw` x1 reads. Same evidence, different file: the driven
    inputs and the first divergence, which is what the concrete arm carries.
    """
    path = artifacts / call_id / "formal" / "replay" / "witness.json"
    if not path.is_file():
        return None
    try:
        w = json.loads(path.read_text())
    except Exception:
        return None
    trace, div = w.get("input_trace") or [], w.get("first_divergence") or {}
    if not trace or not div:
        return None
    lines = []
    for i, step in enumerate(trace):
        cells = "  ".join(f"{k}={v}" for k, v in step.items() if k != "clk")
        lines.append(f"  cycle {i}: {cells}")
    lines.append(
        f"  cycle {div.get('cycle')}: output {div.get('output')} "
        f"expected {div.get('expected')}, got {div.get('got')}")
    return ("The verifier produced this counterexample trace, from initialisation through "
            "the first divergence. Values are the driven inputs, the reference output and "
            "the candidate's output:\n" + "\n".join(lines))


def build_prompt(broken: str, prev: str, trace: str | None, style: str) -> str:
    parts = ["Revise a previous repair attempt for this original problem.",
             "\nOriginal buggy module:\n```systemverilog\n" + broken.strip() + "\n```",
             "\nPrevious candidate:\n```systemverilog\n" + prev.strip() + "\n```"]
    parts.append("\n" + trace if trace else
                 "\nThe previous candidate still did not pass verification.")
    parts.append("\nReturn only the complete corrected module.")
    user = "\n".join(parts)
    return user + VERIREASON_TAIL if style == "verireason" else user


def mcnemar(b: int, c: int) -> float:
    """Exact two-sided McNemar on the discordant counts."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / (2 ** n))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=tuple(STYLES),
                    help="banked model slot; selects prompt style and extractor")
    ap.add_argument("--served-model", required=True,
                    help="model id as the endpoint serves it (base, tuned, verireason, vrqwen)")
    ap.add_argument("--endpoint", required=True,
                    help="OpenAI-compatible /v1 root. Required rather than defaulted: the "
                         "shared default points at a different provider entirely")
    ap.add_argument("--tag", help="output directory name (default: --model)")
    ap.add_argument("--run", default="generated/tierf_openweight",
                    help="banked Tier F run tree, relative to the output root")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="first N refuted cases (smoke test)")
    ap.add_argument("--out", help="override the output directory")
    ap.add_argument("--overwrite", action="store_true",
                    help="permit replacing an existing result.json")
    ap.add_argument("--publish", action="store_true",
                    help="also merge this result into the public contrast artifact "
                         "artifacts/public/capability/counterexample_contrast.json, keyed by "
                         "--tag. Other models' entries are left untouched; the table reads "
                         "that file, so this is what puts the cell in the paper.")
    args = ap.parse_args(argv)

    tag = args.tag or args.model
    out = Path(args.out or f"{ROOT}/generated/counterexample_openweight/{tag}")
    if (out / "result.json").exists() and not args.overwrite:
        raise SystemExit(f"{out}/result.json exists; pass --overwrite to replace it")

    run = Path(f"{ROOT}/{args.run}")
    cfg = STYLES[args.model]

    # Endpoint identity is checked before any paid call: a wrong --served-model
    # against a live endpoint would otherwise 404 per-case and read as failure.
    os.environ["RTLREPAIR_LLM_URL"] = args.endpoint
    health = llm_client.health_check()
    if not health.get("ok"):
        raise SystemExit(f"endpoint not reachable: {health.get('error')}")
    served = list(health.get("models") or [])
    if args.served_model not in served:
        raise SystemExit(f"{args.served_model!r} not served at {args.endpoint}; got {served}")

    # Refuted baselines: select on the formal status, exactly as the published
    # rows do. Selecting on witness-file presence instead would pull in cases
    # whose verdict was UNSUPPORTED, which is a different population.
    refuted = []
    for line in open(run / "results.jsonl"):
        r = json.loads(line)
        model, arm, case = r["call_id"].split("__", 2)
        if model == args.model and arm == "diag" and \
                (r.get("formal_verdict") or {}).get("status") == "COUNTEREXAMPLE":
            refuted.append((r["call_id"], case, r["seed_id"]))

    cand_rtl = {}
    for line in open(run / "jobs.jsonl"):
        j = json.loads(line)
        cand_rtl[j["call_id"]] = (j.get("candidate_rtl"), j.get("golden_rtl"))

    heldout = {}
    for line in open(f"{SOURCE_ROOT}/data/repairbench_heldout.jsonl"):
        it = json.loads(line)
        heldout[it["id"]] = it

    jobs, skipped, kinds = [], [], {}
    for call_id, case, seed_id in refuted:
        prev, golden = cand_rtl.get(call_id, (None, None))
        item = heldout.get(case)
        if not prev or not golden or item is None:
            skipped.append((case, "no banked candidate/golden/item")); continue
        trace = witness_text(call_id, run / "artifacts")
        if trace is None:
            skipped.append((case, "no retained witness")); continue
        broken, err = rbe.parse_item(item)
        broken, err = rbe.anonymize(broken, err, seed_id)   # the arms were --anon
        kinds[case] = "formal"
        for a in ("concrete", "generic"):
            jobs.append((case, golden, broken, prev, a, trace))
    if args.limit:
        keep = {j[0] for j in jobs[:args.limit * 2]}
        jobs = [j for j in jobs if j[0] in keep]

    n_cases = len(jobs) // 2
    print(f"X3 [{args.model} @ {args.served_model}]  {len(refuted)} refuted baselines "
          f"-> {n_cases} paired cases, {len(jobs)} calls  ({len(skipped)} skipped)", flush=True)

    res, errors, done = {}, [], 0

    def one(job):
        nonlocal done
        case, golden, broken, prev, arm, trace = job
        user = build_prompt(broken, prev, trace if arm == "concrete" else None, cfg["style"])
        try:
            texts = llm_client.generate_strict(
                user, model=args.served_model, n=1, temp=0.0,
                max_tokens=cfg["max_tokens"], system=cfg["system"])
            cand = extract_for(cfg["style"], texts[0] if texts else "")
        except Exception as exc:  # noqa: BLE001 - a failed call is data
            return case, arm, None, str(exc)[:120]
        if cand:  # retain candidates so a rescore costs no new calls
            d = out / arm
            d.mkdir(parents=True, exist_ok=True)
            (d / f"{case}.sv").write_text(cand)
        ok = bool(cand) and reset_aware_sim.equivalent(golden, cand)
        with _print_lock:
            done += 1
            if done % 20 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
        return case, arm, ok, None

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        for case, arm, ok, err in ex.map(one, jobs):
            res.setdefault(case, {})[arm] = ok
            if err:
                errors.append({"case": case, "arm": arm, "error": err})

    paired = {k: a for k, a in res.items()
              if a.get("concrete") is not None and a.get("generic") is not None}
    conc = sum(1 for a in paired.values() if a["concrete"])
    gen = sum(1 for a in paired.values() if a["generic"])
    b = sum(1 for a in paired.values() if a["concrete"] and not a["generic"])
    c = sum(1 for a in paired.values() if a["generic"] and not a["concrete"])
    n = len(paired)

    result = {
        "model": args.model, "served_model": args.served_model, "endpoint": args.endpoint,
        "prompt_style": cfg["style"], "max_tokens": cfg["max_tokens"],
        "n_paired": n, "concrete_recovered": conc, "generic_recovered": gen,
        "concrete_pct": round(100 * conc / n, 1) if n else None,
        "generic_pct": round(100 * gen / n, 1) if n else None,
        "risk_diff_pp": round(100 * (conc - gen) / n, 1) if n else None,
        "concrete_only": b, "generic_only": c, "mcnemar_p": mcnemar(b, c),
        "n_refuted_baselines": len(refuted), "n_pairs_attempted": len(jobs) // 2,
        "skipped": skipped, "errors": errors, "n_errors": len(errors),
        "witness_kind": dict(Counter(kinds.values())),
        "scored_by": "reset_aware_differential_simulation", "exploratory": True,
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")

    if args.publish:
        pub = Path(SOURCE_ROOT) / "artifacts" / "public" / "capability" / \
            "counterexample_contrast.json"
        existing = json.loads(pub.read_text()) if pub.exists() else {}
        existing[tag] = result
        pub.write_text(json.dumps(existing, indent=2, sort_keys=True) + "\n")
        print(f"  published -> {pub.relative_to(Path(SOURCE_ROOT))} [{tag}]")
    print(f"\nCOUNTEREXAMPLE CONTRAST ({args.model}, n={n} of {len(jobs)//2} attempted, "
          f"{len(errors)} call errors)")
    print(f"  concrete witness : {conc}/{n} = {result['concrete_pct']}%")
    print(f"  generic notice   : {gen}/{n} = {result['generic_pct']}%")
    print(f"  risk difference  : {result['risk_diff_pp']:+}pp   discordant {b}/{c}, "
          f"exact McNemar p={result['mcnemar_p']:.3f}")
    print(f"  wrote {out}/result.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
