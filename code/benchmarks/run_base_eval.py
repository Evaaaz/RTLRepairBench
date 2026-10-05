"""run_base_eval -- evaluate the BASE (untuned) model on the benchmark suite.

Thin driver: loads ``benchmarks/benchmark_tasks.jsonl`` (or a fixture / built-in
task), samples n=20 @ temp=0.8 per task for pass@k, plus a greedy temp=0 pass for
a clean compile-pass row, scores with evaluate_outputs, and writes raw outputs +
``result.json`` under ``generated/benchmark_results/base/``.

Falls back to ``backend.serve_stub`` when no model server is reachable; in that
case the result is flagged ``is_placeholder=True``.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pathlib as _pl  # noqa: E402
_REPO_ROOT = str(_pl.Path(os.environ.get("RTLREPAIR_ROOT", "")) if os.environ.get("RTLREPAIR_ROOT") else _pl.Path(__file__).resolve().parents[2])

import bench_common  # noqa: E402

MODEL_DEFAULT = "base"
RUN_NAME = "base"


def main():
    ap = argparse.ArgumentParser(description="Run base-model RTL benchmark.")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--tasks", default=None, help="benchmark_tasks.jsonl path")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--temp", type=float, default=0.8)
    args = ap.parse_args()

    row = bench_common.run_pass_at_k(
        model=args.model,
        n=args.n,
        temp=args.temp,
        tasks_path=args.tasks,
        run_name=RUN_NAME,
    )
    out = os.path.join(bench_common.RESULTS_DIR, RUN_NAME, "result.json")
    print(json.dumps({
        "model": row["model"],
        "compile_pass": row["compile_pass"],
        "testbench_pass": row["testbench_pass"],
        "is_placeholder": row["is_placeholder"],
        "result_json": out,
    }, indent=2))


if __name__ == "__main__":
    main()
