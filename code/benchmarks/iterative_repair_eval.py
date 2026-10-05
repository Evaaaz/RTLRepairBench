"""Experiment 2: iterative testbench-feedback repair.

Every other RTLRepair repair number is single-shot. Here the model proposes a fix,
we RUN it under differential simulation, and if it still diverges we feed back the
concrete failure (which output diverges, at which cycle) and let it try again, up to
K rounds. This tests whether the survivors are fixable with more compute-per-bug (a
search gap) or are genuinely under-determined (an information gap no iteration closes).

Recovery is reported cumulatively per round, by mutation class. Backend claude (the
served base endpoint is the intended base-first run when credits allow).

Usage:
  ANTHROPIC_API_KEY=... .venv/bin/python benchmarks/iterative_repair_eval.py \
    --backend claude --rounds 3 --bench data/repairbench_sem85.jsonl
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
for _p in (HERE, CODE, os.path.join(CODE, "datagen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pathlib as _pl  # noqa: E402
ROOT = str(_pl.Path(os.environ.get("RTLREPAIR_ROOT", "")) if os.environ.get("RTLREPAIR_ROOT") else _pl.Path(__file__).resolve().parents[2])

import repair_agent  # noqa: E402
from bench_common import extract_rtl  # noqa: E402
from repairbench_eval import (  # noqa: E402
    parse_item, anonymize, golden_src, claude_repair, ANON_NAME,
)
from diff_testbench import parse_module, run_diff_test  # noqa: E402


def divergence_feedback(golden: str, fixed: str) -> tuple[bool, str]:
    """Return (passed, feedback). passed=True iff fixed matches golden under diff-sim."""
    info = parse_module(golden)
    if info is None:
        return False, "The module could not be parsed."
    res = run_diff_test(golden, fixed, info)
    if not res.compiled:
        return False, "Your revised module does not compile/simulate. Fix it and try again."
    if not res.diverged:
        return True, ""
    outs = ", ".join(f"`{o}`" for o in (res.out_diffs or {}).keys()) or "an output"
    cyc = getattr(res, "first_cycle", None)
    where = f", first at cycle {cyc}" if cyc is not None else ""
    return False, (f"Your revised module still diverges from the reference: output(s) {outs} "
                   f"are still wrong{where}. This is a behavioral bug. Reconsider the intended "
                   f"logic and return a corrected module.")


def one_fix(cur_rtl: str, err: str, args) -> str:
    if args.backend == "claude":
        raw = claude_repair(cur_rtl, err, None, args.claude_model)
    else:
        rr = repair_agent.repair(cur_rtl, err, model=args.model, temperature=args.temp)
        raw = rr.fixed_rtl or ""
    fixed = extract_rtl(raw) if raw else ""
    return (fixed.rstrip() + "\n") if fixed else ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="claude", choices=["rtlrepair", "claude"])
    ap.add_argument("--model", default="base")
    ap.add_argument("--claude-model", default="claude-opus-4-8")
    ap.add_argument("--bench", default=os.path.join(ROOT, "data", "repairbench_sem85.jsonl"))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--temp", type=float, default=0.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    items = [json.loads(l) for l in open(args.bench) if json.loads(l).get("bucket") == "semantic"]
    if args.limit:
        items = items[: args.limit]

    # cumulative recovery at each round (0..rounds); by mutation
    cum = [0] * (args.rounds + 1)
    by_mut = collections.defaultdict(lambda: [0] * (args.rounds + 1))
    tot_mut = collections.Counter()
    records = []
    for idx, item in enumerate(items):
        broken, err = parse_item(item)
        seed_id, mut = item["seed_id"], item["mutation"]
        broken, err = anonymize(broken, err, seed_id)
        golden = golden_src(seed_id, anon=True)
        tot_mut[mut] += 1
        solved_round = None
        cur, cur_err = broken, err
        for r in range(args.rounds + 1):
            fixed = one_fix(cur, cur_err, args)
            if not fixed:
                break
            passed, fb = divergence_feedback(golden, fixed) if golden else (False, "")
            if passed:
                solved_round = r
                break
            cur, cur_err = fixed, fb  # iterate: repair the previous attempt given its divergence
        # tally: a case solved at round r counts as solved for all rounds >= r
        for r in range(args.rounds + 1):
            if solved_round is not None and solved_round <= r:
                cum[r] += 1
                by_mut[mut][r] += 1
        records.append({"id": item.get("id"), "mutation": mut, "solved_round": solved_round})
        if (idx + 1) % 20 == 0:
            print(f"  {idx+1}/{len(items)}  (solved so far @final: {cum[-1]})", flush=True)

    n = len(items)
    model_label = args.claude_model if args.backend == "claude" else args.model
    print(f"\nITERATIVE REPAIR ({model_label}, {n} semantic cases, {args.rounds} feedback rounds)")
    print("cumulative recovery by round:")
    for r in range(args.rounds + 1):
        print(f"  round {r}: {cum[r]}/{n} = {cum[r]/n*100:.1f}%")
    print("\nby mutation (round 0 -> final):")
    for m in sorted(tot_mut):
        print(f"  {m:<22} {by_mut[m][0]}/{tot_mut[m]} -> {by_mut[m][-1]}/{tot_mut[m]}")

    out = args.out or os.path.join(ROOT, "generated", f"iterative_{model_label.replace('/','-')}.json")
    json.dump({"model": model_label, "n": n, "rounds": args.rounds,
               "cumulative": cum, "by_mutation": {m: by_mut[m] for m in tot_mut},
               "tot_mut": dict(tot_mut), "records": records}, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
