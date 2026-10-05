"""run_repair_eval -- measure the repair loop's fail-to-pass rate.

For each benchmark task we first generate a candidate (greedy). If it already
passes, it is not a repair opportunity. For the failing candidates we run the
repair path -- ``repair_agent.explain_error`` for an explanation plus a
model regeneration conditioned on the tool error -- then re-score.

``repair_fail_to_pass`` = (# initially-failing tasks that pass after repair) /
(# initially-failing tasks). Raw outputs + logs land under
generated/benchmark_results/repair/.

Falls back to backend.serve_stub offline (result flagged is_placeholder=True);
note the deterministic stub will regenerate identical RTL, so offline repair
rate is honestly ~0 and flagged PLACEHOLDER.
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

MODEL_DEFAULT = "tuned"
RUN_NAME = "repair"


def _build_repair_prompt(task, failing_rtl, error_text, explanation):
    instr = task.get("instruction") or "Fix the SystemVerilog module."
    return (
        "{0}\n\n"
        "The following SystemVerilog failed verification.\n\n"
        "```systemverilog\n{1}\n```\n\n"
        "Tool error:\n{2}\n\n"
        "Guidance:\n{3}\n\n"
        "Return the corrected synthesizable module only, in a "
        "```systemverilog code block."
    ).format(instr.strip(), failing_rtl.strip(), error_text.strip(),
             (explanation or "").strip())


def run_repair(model, tasks_path=None, run_name=RUN_NAME):
    from backend import llm_client
    import repair_agent

    tasks = bench_common.load_tasks(tasks_path)
    run_dir = os.path.join(bench_common.RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    initially_failing = 0
    repaired = 0
    is_placeholder = False
    records = []

    for task in tasks:
        task_id = str(task.get("id", "task"))
        # Greedy first attempt.
        try:
            first = llm_client.generate(
                bench_common.build_prompt(task), model=model, n=1, temp=0.0
            )
        except Exception:
            first = [""]
        comp = first[0] if first else ""
        pre_dir = os.path.join(run_dir, task_id, "pre")
        pre = bench_common.evaluate_completion(task, comp, pre_dir, "cand")
        pre_pass = pre.get("test_pass")
        pre_pass = pre_pass if pre_pass is not None else pre.get("compile_pass")

        rec = {"id": task_id, "pre_pass": bool(pre_pass)}
        if pre_pass:
            records.append(rec)
            continue

        initially_failing += 1
        failing_rtl = bench_common.extract_rtl(comp) or comp
        error_text = pre.get("log", "") or "verification failed"
        explanation = repair_agent.explain_error(failing_rtl, error_text, model=model)

        # Regenerate conditioned on the error.
        try:
            second = llm_client.generate(
                _build_repair_prompt(task, failing_rtl, error_text, explanation),
                model=model, n=1, temp=0.2,
            )
        except Exception:
            second = [""]
        repaired_comp = second[0] if second else ""

        # Offline stub returns identical canned RTL -> repair can't change a fail.
        if repaired_comp.strip() == comp.strip():
            is_placeholder = True

        post_dir = os.path.join(run_dir, task_id, "post")
        post = bench_common.evaluate_completion(task, repaired_comp, post_dir, "repaired")
        post_pass = post.get("test_pass")
        post_pass = post_pass if post_pass is not None else post.get("compile_pass")
        if post_pass:
            repaired += 1
        rec.update({
            "post_pass": bool(post_pass),
            "explanation": explanation,
        })
        with open(os.path.join(post_dir, "explanation.md"), "w", encoding="utf-8") as fh:
            fh.write(explanation)
        records.append(rec)

    rate = (repaired / initially_failing) if initially_failing else None
    row = {
        "model": model,
        "repair_fail_to_pass": rate,
        "initially_failing": initially_failing,
        "repaired": repaired,
        "is_placeholder": is_placeholder,
        "records": records,
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(row, fh, indent=2)
    return row


def main():
    ap = argparse.ArgumentParser(description="Run repair fail-to-pass benchmark.")
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--tasks", default=None)
    args = ap.parse_args()
    row = run_repair(args.model, tasks_path=args.tasks)
    out = os.path.join(bench_common.RESULTS_DIR, RUN_NAME, "result.json")
    print(json.dumps({
        "model": row["model"],
        "repair_fail_to_pass": row["repair_fail_to_pass"],
        "initially_failing": row["initially_failing"],
        "repaired": row["repaired"],
        "is_placeholder": row["is_placeholder"],
        "result_json": out,
    }, indent=2))


if __name__ == "__main__":
    main()
