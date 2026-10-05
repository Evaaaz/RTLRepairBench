"""run_ablation_tooluse -- ablation: performance WITH vs WITHOUT tool use.

For each held-out task and each model we measure functional pass rate two ways:

* TOOL-USE OFF = a single greedy (temp=0) generation, scored once. No feedback.
* TOOL-USE ON  = the SAME greedy round-0 generation, then up to ``max_rounds``
  repair rounds, each conditioned on the real verilator/iverilog error log;
  re-scored each round. Pass = passes within ``max_rounds`` rounds.

Because tool-OFF is literally round 0 of the tool-ON run, the only difference
between the two conditions is the repair loop (no temperature confound).

This is a separate greedy functional-pass metric from the temp-sampled pass@k
table; do not compare the two directly.

    RTLREPAIR_LLM_URL=... RTLREPAIR_LLM_API_KEY=... \\
      python benchmarks/run_ablation_tooluse.py --models base,tuned --max-rounds 3
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Optional

_THIS = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_THIS)
for _p in (_THIS, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pathlib as _pl  # noqa: E402
_ROOT = str(_pl.Path(os.environ.get("RTLREPAIR_ROOT", "")) if os.environ.get("RTLREPAIR_ROOT") else _pl.Path(__file__).resolve().parents[2])

import bench_common  # noqa: E402
import wandb_logging  # noqa: E402
from run_repair_eval import _build_repair_prompt  # noqa: E402

RUN_NAME = "ablation"
MAX_ROUNDS_DEFAULT = 3
TEMP_REPAIR_DEFAULT = 0.2


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))


def _display(model: str) -> str:
    return "v5" if model == "tuned" else model


def _pass_of(res: Dict) -> bool:
    """A pass = test_pass when measured, else fall back to compile_pass."""
    tp = res.get("test_pass")
    return bool(tp if tp is not None else res.get("compile_pass"))


def _is_stub(text: str) -> bool:
    return "STUB" in (text or "")[:80]


def _run_one_task(task, model, run_dir, max_rounds, temp_repair):
    """Greedy round-0 (OFF), then up to max_rounds repair rounds (ON delta)."""
    from backend import llm_client
    import repair_agent

    task_id = str(task.get("id", "task"))
    prompt = bench_common.build_prompt(task)

    # ---- round 0: greedy, the tool-OFF measurement ----
    try:
        first = llm_client.generate(prompt, model=model, n=1, temp=0.0)
    except Exception:
        first = [""]
    round0 = first[0] if first else ""
    r0_dir = os.path.join(run_dir, task_id, "round_0")
    res = bench_common.evaluate_completion(task, round0, r0_dir, "cand")
    passed = _pass_of(res)

    placeholder = _is_stub(round0)
    round_results = [{"round": 0, "pass": passed, "tool": res.get("tool")}]
    pass_round = 0 if passed else None
    cur_comp, cur_res = round0, res

    # ---- rounds 1..max_rounds: repair conditioned on the real tool error ----
    r = 1
    while not passed and r <= max_rounds:
        failing_rtl = bench_common.extract_rtl(cur_comp) or cur_comp
        error_text = cur_res.get("log", "") or "verification failed"
        explanation = repair_agent.explain_error(failing_rtl, error_text, model=model)
        repair_prompt = _build_repair_prompt(task, failing_rtl, error_text, explanation)
        try:
            regen = llm_client.generate(repair_prompt, model=model, n=1, temp=temp_repair)
        except Exception:
            regen = [""]
        cur_comp = regen[0] if regen else ""
        # Offline stub returns identical / canned RTL -> repair can't change a fail.
        if _is_stub(cur_comp) or cur_comp.strip() == round0.strip():
            placeholder = True
        rd_dir = os.path.join(run_dir, task_id, "round_{0}".format(r))
        cur_res = bench_common.evaluate_completion(task, cur_comp, rd_dir, "repaired")
        passed = _pass_of(cur_res)
        round_results.append({"round": r, "pass": passed, "tool": cur_res.get("tool")})
        if passed:
            pass_round = r
            break
        r += 1

    return {
        "id": task_id,
        "pass_off": bool(round_results[0]["pass"]),
        "pass_on": pass_round is not None,
        "pass_round": pass_round,
        "round_results": round_results,
        "is_placeholder": placeholder,
    }


def run_tooluse_ablation(model, tasks_path=None, max_rounds=MAX_ROUNDS_DEFAULT,
                         temp_repair=TEMP_REPAIR_DEFAULT, run_name=RUN_NAME,
                         wandb_run=None, limit=None):
    tasks = bench_common.load_tasks(tasks_path)
    if limit:
        tasks = tasks[: max(1, limit)]
    run_dir = os.path.join(bench_common.RESULTS_DIR, run_name, _safe(model))
    os.makedirs(run_dir, exist_ok=True)

    records: List[Dict] = []
    n_done = off_hits = on_hits = 0
    for task in tasks:
        rec = _run_one_task(task, model, run_dir, max_rounds, temp_repair)
        records.append(rec)
        n_done += 1
        off_hits += int(rec["pass_off"])
        on_hits += int(rec["pass_on"])
        if wandb_run is not None:
            try:
                wandb_run.log({
                    "live/{0}/tasks_done".format(model): n_done,
                    "live/{0}/running_pass_off".format(model): off_hits / n_done,
                    "live/{0}/running_pass_on".format(model): on_hits / n_done,
                })
            except Exception:
                pass

    n = len(records) or 1
    pass_off = off_hits / n
    pass_on = on_hits / n
    # cumulative pass rate by round (r0 == pass_off, r_max == pass_on; monotonic)
    per_round = []
    for k in range(max_rounds + 1):
        hits = sum(1 for r in records if r["pass_round"] is not None and r["pass_round"] <= k)
        per_round.append(hits / n)
    repaired_by_round = {
        rd: sum(1 for r in records if r["pass_round"] == rd) for rd in range(1, max_rounds + 1)
    }
    row = {
        "model": model,
        "n_tasks": len(records),
        "pass_off": pass_off,
        "pass_on": pass_on,
        "lift": pass_on - pass_off,
        "initially_failing": sum(1 for r in records if not r["pass_off"]),
        "repaired_by_round": repaired_by_round,
        "per_round_cumulative": per_round,
        "max_rounds": max_rounds,
        "temp_repair": temp_repair,
        "is_placeholder": any(r["is_placeholder"] for r in records),
        "records": records,
        "run_dir": run_dir,
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(row, fh, indent=2)
    return row


# --------------------------------------------------------------------------- #
# Output table + charts
# --------------------------------------------------------------------------- #
def _fmt(v):
    return "{0:.2f}".format(v) if isinstance(v, (int, float)) else "-"


def _write_table(rows, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    cols = ["model", "tool_off", "tool_on", "lift"]
    lines = ["# Tool-use ablation (greedy round-0 vs +verify/repair)", "",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]
    import csv
    md = os.path.join(out_dir, "ablation_table.md")
    cv = os.path.join(out_dir, "ablation_table.csv")
    with open(cv, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in rows:
            disp = _display(r["model"])
            lines.append("| {0} | {1} | {2} | {3} |".format(
                disp, _fmt(r["pass_off"]), _fmt(r["pass_on"]), _fmt(r["lift"])))
            w.writerow([disp, "{0:.4f}".format(r["pass_off"]),
                        "{0:.4f}".format(r["pass_on"]), "{0:.4f}".format(r["lift"])])
    lines += ["", "_tool_off = single greedy generation; tool_on = + up to {0} verify/repair "
              "rounds on the real verilator/iverilog error. {1} held-out tasks._".format(
                  rows[0]["max_rounds"], rows[0]["n_tasks"]), ""]
    with open(md, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    return {"md": md, "csv": cv}


def _charts(rows, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except Exception as exc:
        print("[charts skipped] {0}".format(exc))
        return
    labels = [_display(r["model"]) for r in rows]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    # (1) grouped bar: off vs on per model
    x = np.arange(len(rows)); w = 0.35
    b1 = ax1.bar(x - w / 2, [r["pass_off"] for r in rows], w, label="tool OFF (1-shot)", color="#64748B")
    b2 = ax1.bar(x + w / 2, [r["pass_on"] for r in rows], w, label="tool ON (verify+repair)", color="#2ACFBE")
    ax1.bar_label(b1, fmt="%.2f", fontsize=9); ax1.bar_label(b2, fmt="%.2f", fontsize=9)
    ax1.set_xticks(x); ax1.set_xticklabels(labels); ax1.set_ylim(0, 1.02)
    ax1.set_ylabel("functional pass rate"); ax1.set_title("Tool use OFF vs ON"); ax1.legend()
    # (2) cumulative pass vs repair round, base vs v5 as 2 lines
    colors = {"base": "#1f77b4", "v5": "#d62728"}
    for r in rows:
        disp = _display(r["model"])
        rounds = list(range(len(r["per_round_cumulative"])))
        ax2.plot(rounds, r["per_round_cumulative"], marker="o",
                 label=disp, color=colors.get(disp, None), linewidth=2)
    ax2.set_xticks(list(range(rows[0]["max_rounds"] + 1)))
    ax2.set_xlabel("repair round (0 = no tool use)"); ax2.set_ylim(0, 1.02)
    ax2.set_title("Cumulative pass vs repair round"); ax2.grid(alpha=0.3); ax2.legend()
    png = os.path.join(out_dir, "ablation.png")
    plt.tight_layout(); plt.savefig(png, dpi=130)
    print("chart -> {0}".format(png))


def _wandb_two_line(rows):
    """Two runs sharing metric 'pass_at_round' -> 2 colored lines on one panel."""
    try:
        import wandb
    except Exception as exc:
        print("[wandb two-line skipped] {0}".format(exc))
        return
    for r in rows:
        name = "abl-{0}".format(_display(r["model"]))
        run = wandb.init(project=os.environ.get("WANDB_PROJECT", "rtlrepair-bench"),
                         entity=os.environ.get("WANDB_ENTITY"),
                         name=name, job_type="ablation", reinit=True)
        for k, cum in enumerate(r["per_round_cumulative"]):
            run.log({"pass_at_round": cum, "round": k})
        print("{0} -> {1}".format(name, getattr(run, "url", "(offline)")))
        run.finish()


def _git_commit():
    head = os.path.join(_ROOT, ".git", "HEAD")
    try:
        with open(head, "r", encoding="utf-8") as fh:
            ref = fh.read().strip()
        if ref.startswith("ref:"):
            with open(os.path.join(_ROOT, ".git", ref.split(" ", 1)[1]), "r", encoding="utf-8") as fh:
                return fh.read().strip()[:12]
        return ref[:12]
    except OSError:
        return "unknown"


def main():
    ap = argparse.ArgumentParser(description="Tool-use ablation: verify+repair ON vs OFF.")
    ap.add_argument("--models", default="base,tuned", help="comma-separated served model ids")
    ap.add_argument("--tasks", default=None, help="tasks JSONL (default: data/eval_tasks.jsonl)")
    ap.add_argument("--max-rounds", type=int, default=MAX_ROUNDS_DEFAULT)
    ap.add_argument("--temp-repair", type=float, default=TEMP_REPAIR_DEFAULT)
    ap.add_argument("--limit", type=int, default=None, help="cap to first N tasks (smoke)")
    ap.add_argument("--wandb-project", default=None)
    ap.add_argument("--wandb-entity", default=None)
    args = ap.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    endpoint = os.environ.get("RTLREPAIR_LLM_URL", "(unset -> serve_stub)")
    run = wandb_logging.init_run(
        name="ablation-tooluse",
        project=args.wandb_project,
        entity=args.wandb_entity,
        config={
            "max_rounds": args.max_rounds,
            "temp_repair": args.temp_repair,
            "round0_temp": 0.0,
            "tasks": args.tasks or bench_common.DEFAULT_TASKS,
            "endpoint": endpoint,
            "git_commit": _git_commit(),
        },
    )

    rows = []
    for model in models:
        print("== tool-use ablation: {0} (max_rounds={1}) ==".format(model, args.max_rounds))
        row = run_tooluse_ablation(
            model, tasks_path=args.tasks, max_rounds=args.max_rounds,
            temp_repair=args.temp_repair, wandb_run=run, limit=args.limit,
        )
        wandb_logging.log_ablation_row(run, row, _display(model))
        print("   {0}: tool_off={1:.3f}  tool_on={2:.3f}  lift=+{3:.3f}  placeholder={4}".format(
            _display(model), row["pass_off"], row["pass_on"], row["lift"], row["is_placeholder"]))
        rows.append(row)
    wandb_logging.finish(run)

    # two-line cumulative-pass overlay (separate runs sharing a metric name)
    _wandb_two_line(rows)

    out_dir = os.path.join(bench_common.RESULTS_DIR, "ablation")
    paths = _write_table(rows, out_dir)
    _charts(rows, out_dir)

    placeholder = any(r["is_placeholder"] for r in rows)
    print(json.dumps({
        "table_md": paths["md"],
        "chart": os.path.join(out_dir, "ablation.png"),
        "summary": {_display(r["model"]): {"off": round(r["pass_off"], 4),
                                            "on": round(r["pass_on"], 4),
                                            "lift": round(r["lift"], 4)} for r in rows},
        "is_placeholder": placeholder,
    }, indent=2))
    if placeholder:
        print("\n[!] PLACEHOLDER: the endpoint returned stub/identical RTL for some tasks "
              "(flaky or offline). Numbers are INVALID; re-run against a healthy endpoint.")


if __name__ == "__main__":
    main()
