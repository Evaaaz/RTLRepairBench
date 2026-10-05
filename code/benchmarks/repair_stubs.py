"""repair_stubs -- re-run only the stub-corrupted tasks of a completed run.

When the model endpoint blips mid-benchmark, those tasks fall back to the
deterministic serve_stub (all n samples identical, opening with ``// STUB``) and
score 0, understating the run. This re-runs ONLY those tasks against the (now
healthy) endpoint and patches ``<model>/result.json`` in place, leaving the
already-good tasks untouched.

Usage (endpoint env must be set, like run_all):
    python benchmarks/repair_stubs.py --model tunedv6
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

_THIS = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_THIS)
for _p in (_THIS, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pathlib as _pl  # noqa: E402
_ROOT = str(_pl.Path(os.environ.get("RTLREPAIR_ROOT", "")) if os.environ.get("RTLREPAIR_ROOT") else _pl.Path(__file__).resolve().parents[2])

import bench_common  # noqa: E402
import make_benchmark_table as mbt  # noqa: E402


def _is_stub(run_name: str, tid: str) -> bool:
    raws = glob.glob(os.path.join(bench_common.RESULTS_DIR, run_name, tid, "sample_*", "raw.txt"))
    texts = []
    for r in raws:
        try:
            texts.append(open(r, encoding="utf-8").read())
        except OSError:
            pass
    if not texts:
        return False
    if any("// STUB" in t for t in texts):
        return True
    return len(texts) > 1 and len(set(texts)) == 1


def main():
    ap = argparse.ArgumentParser(description="Re-run + patch stub-corrupted benchmark tasks.")
    ap.add_argument("--model", required=True, help="served model id whose result.json to repair")
    ap.add_argument("--tasks", default=None, help="tasks jsonl (default data/eval_tasks.jsonl)")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--temp", type=float, default=0.8)
    args = ap.parse_args()

    rj = os.path.join(bench_common.RESULTS_DIR, args.model, "result.json")
    res = json.load(open(rj))
    probs = res["problems"]

    stub_ids = [p["id"] for p in probs if _is_stub(args.model, p["id"])]
    print("[{0}] {1} stub tasks to repair: {2}".format(args.model, len(stub_ids), stub_ids))
    if not stub_ids:
        print("nothing to repair.")
        return

    all_tasks = bench_common.load_tasks(args.tasks)
    keep = set(stub_ids)
    subset = [t for t in all_tasks if str(t.get("id")) in keep]
    subpath = os.path.join(bench_common.RESULTS_DIR, "_repair_{0}.jsonl".format(args.model))
    with open(subpath, "w", encoding="utf-8") as fh:
        for t in subset:
            fh.write(json.dumps(t) + "\n")

    # Re-run ONLY the stub tasks to a side run dir.
    row = bench_common.run_pass_at_k(
        model=args.model, n=args.n, temp=args.temp,
        tasks_path=subpath, run_name="{0}_repair".format(args.model),
    )
    fresh = {p["id"]: p for p in row["problems"]}

    fixed = 0
    for i, p in enumerate(probs):
        if p["id"] in fresh:
            probs[i] = fresh[p["id"]]
            fixed += 1

    still = [tid for tid in stub_ids if _is_stub("{0}_repair".format(args.model), tid)]
    res["problems"] = probs
    res["compile_pass"] = bench_common._avg([p["compile_pass"] for p in probs])
    res["testbench_pass"] = bench_common._avg([p["test_pass"] for p in probs])
    res["is_placeholder"] = bool(still)
    json.dump(res, open(rj, "w"), indent=2)

    p1 = mbt.mean_pass_at_k(probs, 1)
    p3 = mbt.mean_pass_at_k(probs, 3)
    print("[{0}] patched {1} tasks; still_stub={2}".format(args.model, fixed, len(still)))
    print("  NEW compile={0:.3f} test={1:.3f} pass@1={2:.3f} pass@3={3:.3f} placeholder={4}".format(
        res["compile_pass"], res["testbench_pass"], p1, p3, res["is_placeholder"]))


if __name__ == "__main__":
    main()
