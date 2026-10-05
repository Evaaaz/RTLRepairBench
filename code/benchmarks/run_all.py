"""run_all -- the headline benchmark in one command, fully logged to W&B.

Runs the three Layer-1 evals over the held-out suite and assembles the headline
table (the README section-7 result):

    base   pass@k   (benchmarks.bench_common.run_pass_at_k, model="base")
    tuned  pass@k   (model="tuned")
    repair fail->pass rate  (run_repair_eval.run_repair, model="tuned"  => "tuned+repair")

then writes ``generated/benchmark_results/table.{md,csv}`` and logs the whole
sweep to Weights & Biases: per-model scalars, the table as a wandb.Table, and the
result files as an artifact. W&B logging is a no-op when wandb is unavailable or
disabled (see ``benchmarks/wandb_logging.py``), so this command runs anywhere.

The repair row is merged onto the ``tuned`` row's ``repair_fail_to_pass`` column,
matching the existing table layout.

Usage:
  # Real run (needs a served base+tuned vLLM endpoint -- see README_TRAIN s5-6):
  RTLREPAIR_LLM_URL=http://localhost:8000/v1 RTLREPAIR_LLM_API_KEY=EMPTY \\
      python benchmarks/run_all.py --tasks data/eval_tasks.jsonl --n 20 --temp 0.8

  # Offline smoke (serve_stub fallback -> rows flagged PLACEHOLDER), W&B offline:
  WANDB_MODE=offline python benchmarks/run_all.py --n 3
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
import make_benchmark_table as mbt  # noqa: E402
import run_repair_eval  # noqa: E402
import wandb_logging  # noqa: E402


def _git_commit() -> str:
    head = os.path.join(_REPO_ROOT, ".git", "HEAD")
    try:
        with open(head, "r", encoding="utf-8") as fh:
            ref = fh.read().strip()
        if ref.startswith("ref:"):
            path = os.path.join(_REPO_ROOT, ".git", ref.split(" ", 1)[1])
            with open(path, "r", encoding="utf-8") as fh:
                return fh.read().strip()[:12]
        return ref[:12]
    except OSError:
        return "unknown"


def _make_subset(tasks_path, limit):
    """Write the first ``limit`` tasks to a temp JSONL and return its path."""
    tasks = bench_common.load_tasks(tasks_path)[: max(1, limit)]
    os.makedirs(bench_common.RESULTS_DIR, exist_ok=True)
    out = os.path.join(bench_common.RESULTS_DIR, "_subset_{0}.jsonl".format(limit))
    with open(out, "w", encoding="utf-8") as fh:
        for t in tasks:
            fh.write(json.dumps(t) + "\n")
    print("[subset] using first {0} tasks -> {1}".format(len(tasks), out))
    return out


def main():
    ap = argparse.ArgumentParser(description="Run the full base/tuned/repair benchmark + log to W&B.")
    ap.add_argument("--tasks", default=None, help="benchmark tasks JSONL (default: data/eval_tasks.jsonl)")
    ap.add_argument("--n", type=int, default=20, help="samples per task for pass@k")
    ap.add_argument("--temp", type=float, default=0.8, help="sampling temperature for pass@k")
    ap.add_argument("--no-repair", action="store_true", help="skip the repair fail->pass eval")
    ap.add_argument("--limit", type=int, default=None,
                    help="cap to the first N tasks (fast smoke/demo run)")
    ap.add_argument("--models", default="base,tuned",
                    help="comma-separated served model ids for pass@k rows (e.g. base,tuned,tunedv6)")
    ap.add_argument("--repair-model", default="tuned",
                    help="served model id used for the repair fail->pass eval")
    ap.add_argument("--run-name", default=None, help="W&B run name (default: bench-<commit>)")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-entity", default=None)
    args = ap.parse_args()

    tasks_path = args.tasks
    if args.limit:
        tasks_path = _make_subset(tasks_path, args.limit)
    commit = _git_commit()
    endpoint = os.environ.get("RTLREPAIR_LLM_URL", "(unset -> serve_stub)")

    run = wandb_logging.init_run(
        name=args.run_name or "bench-{0}".format(commit),
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "n": args.n,
            "temp": args.temp,
            "tasks": tasks_path or bench_common.DEFAULT_TASKS,
            "endpoint": endpoint,
            "git_commit": commit,
            "served_base": "Qwen/Qwen2.5-Coder-7B-Instruct",
            "served_tuned": "rtlrepair/qwen2.5-coder-7b-tuned",
        },
    )

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    rows = {}

    # pass@k for each requested model id (e.g. base, tuned, tunedv6)
    for model in models:
        print("== pass@k: {0} (n={1}, temp={2}) ==".format(model, args.n, args.temp))
        row = bench_common.run_pass_at_k(
            model=model, n=args.n, temp=args.temp, tasks_path=tasks_path, run_name=model,
            wandb_run=run,
        )
        wandb_logging.log_passk_row(run, row)
        rows[model] = row

    # repair fail->pass loop -> merged onto the repair-model's row
    if not args.no_repair and args.repair_model in rows:
        rm = args.repair_model
        print("== repair fail->pass: {0}+repair ==".format(rm))
        rep = run_repair_eval.run_repair(rm, tasks_path=tasks_path)
        wandb_logging.log_repair_row(run, rep, model="{0}+repair".format(rm))
        rows[rm]["repair_fail_to_pass"] = rep.get("repair_fail_to_pass")
        rows[rm]["is_placeholder"] = bool(
            rows[rm].get("is_placeholder") or rep.get("is_placeholder")
        )

    # headline table (one row per model, in requested order)
    table_rows = [rows[m] for m in models]
    paths = mbt.make_table(table_rows)
    wandb_logging.log_table(
        run,
        table_rows,
        artifact_files=[paths["md"], paths["csv"]]
        + [os.path.join(bench_common.RESULTS_DIR, m, "result.json") for m in models]
        + [os.path.join(bench_common.RESULTS_DIR, "repair", "result.json")],
    )
    wandb_logging.finish(run)

    placeholder = any(r.get("is_placeholder") for r in table_rows)
    print(json.dumps({
        "table_md": paths["md"],
        "table_csv": paths["csv"],
        "is_placeholder": placeholder,
        "wandb": bool(run),
    }, indent=2))
    if placeholder:
        print("\n[!] Results are PLACEHOLDER (no live model server). Set RTLREPAIR_LLM_URL "
              "to a served base+tuned vLLM endpoint for the real headline numbers.")


if __name__ == "__main__":
    main()
