"""wandb_logging -- optional Weights & Biases logging for the benchmark harness.

The harness must run with or without W&B: on a laptop with no login, in CI, or on
a pod that is fully wired to wandb. So every function here degrades to a no-op when
wandb is unavailable or disabled, and never raises into the eval path.

Disable explicitly with ``RTLREPAIR_NO_WANDB=1`` (or the standard ``WANDB_DISABLED=true``).
Force offline (no network/login, writes a local run dir) with ``WANDB_MODE=offline``.

Logged for a full ``run_all`` sweep:
  * config: n, temp, tasks file, served-model ids, endpoint, git commit;
  * per-model summary scalars (compile_pass / testbench_pass / pass@1 / pass@3 /
    repair_fail_to_pass / is_placeholder), namespaced ``bench/<model>/<metric>``;
  * the aggregated results table as a ``wandb.Table``;
  * ``table.md`` / ``table.csv`` / per-model ``result.json`` as a run Artifact.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence

import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import make_benchmark_table as mbt  # noqa: E402

_DEFAULT_PROJECT = os.environ.get("WANDB_PROJECT", "rtlrepair-bench")
_DEFAULT_ENTITY = os.environ.get("WANDB_ENTITY")  # None -> wandb default/login


def _truthy(val: Optional[str]) -> bool:
    return str(val or "").strip().lower() in {"1", "true", "yes", "on"}


def _disabled() -> bool:
    return _truthy(os.environ.get("RTLREPAIR_NO_WANDB")) or _truthy(
        os.environ.get("WANDB_DISABLED")
    )


def init_run(
    name: str,
    config: Optional[Dict] = None,
    project: Optional[str] = None,
    entity: Optional[str] = None,
):
    """Start a wandb run, or return None if wandb is unavailable/disabled.

    Never raises -- a logging failure must not take down a benchmark run.
    """
    if _disabled():
        print("[wandb] disabled via env -- skipping logging.")
        return None
    try:
        import wandb  # noqa: F401
    except Exception:
        print("[wandb] not installed -- skipping logging (pip install wandb).")
        return None
    try:
        run = wandb.init(
            project=project or _DEFAULT_PROJECT,
            entity=entity or _DEFAULT_ENTITY,
            name=name,
            job_type="benchmark",
            config=config or {},
            mode=os.environ.get("WANDB_MODE"),  # None -> online; "offline" for local-only
        )
        print("[wandb] logging to {0}".format(getattr(run, "url", "(offline)")))
        return run
    except Exception as exc:  # pragma: no cover - network/login issues
        print("[wandb] init failed ({0}) -- skipping logging.".format(exc))
        return None


def _pass_at_k(row: Dict, k: int) -> Optional[float]:
    problems = row.get("problems")
    if problems:
        return mbt.mean_pass_at_k(problems, k)
    return None


def log_passk_row(run, row: Dict) -> Dict:
    """Log a pass@k eval row (from bench_common.run_pass_at_k). Returns its metrics."""
    model = str(row.get("model", "model"))
    metrics = {
        "compile_pass": row.get("compile_pass"),
        "testbench_pass": row.get("testbench_pass"),
        "pass_at_1": _pass_at_k(row, 1),
        "pass_at_3": _pass_at_k(row, 3),
        "is_placeholder": 1.0 if row.get("is_placeholder") else 0.0,
    }
    _log_namespaced(run, model, metrics)
    return metrics


def log_repair_row(run, row: Dict, model: str = "tuned+repair") -> Dict:
    """Log a repair eval row (from run_repair_eval.run_repair). Returns its metrics."""
    metrics = {
        "repair_fail_to_pass": row.get("repair_fail_to_pass"),
        "initially_failing": row.get("initially_failing"),
        "repaired": row.get("repaired"),
        "is_placeholder": 1.0 if row.get("is_placeholder") else 0.0,
    }
    _log_namespaced(run, model, metrics)
    return metrics


def log_ablation_row(run, row: Dict, model: str) -> Dict:
    """Log a tool-use ablation row (from run_ablation_tooluse). Returns its metrics."""
    rb = row.get("repaired_by_round") or {}
    metrics = {
        "pass_tool_off": row.get("pass_off"),
        "pass_tool_on": row.get("pass_on"),
        "lift": row.get("lift"),
        "initially_failing": row.get("initially_failing"),
        "repaired_round_1": rb.get(1, rb.get("1")),
        "repaired_round_2": rb.get(2, rb.get("2")),
        "repaired_round_3": rb.get(3, rb.get("3")),
        "is_placeholder": 1.0 if row.get("is_placeholder") else 0.0,
    }
    _log_namespaced(run, model, metrics)
    return metrics


def _log_namespaced(run, model: str, metrics: Dict) -> None:
    if run is None:
        return
    payload = {
        "bench/{0}/{1}".format(model, k): v
        for k, v in metrics.items()
        if v is not None
    }
    if payload:
        try:
            run.log(payload)
            run.summary.update(payload)
        except Exception:
            pass


def log_table(run, rows: Sequence[Dict], artifact_files: Optional[List[str]] = None) -> None:
    """Log the aggregated benchmark table as a wandb.Table + upload result files."""
    if run is None:
        return
    try:
        import wandb

        derived = [mbt._derive_row(r) for r in rows]
        cols = list(mbt.COLUMNS) + ["note"]
        table = wandb.Table(columns=cols)
        for d in derived:
            note = "PLACEHOLDER" if d.get("is_placeholder") else "measured"
            # Pass raw values: numeric columns stay None-or-Number (wandb infers a
            # nullable numeric type); rendering None as "-" would force String and
            # collide with a later numeric row.
            table.add_data(*[d.get(c) for c in mbt.COLUMNS], note)
        run.log({"bench/table": table})

        files = [f for f in (artifact_files or []) if f and os.path.exists(f)]
        if files:
            art = wandb.Artifact("benchmark-results", type="benchmark")
            for f in files:
                # Namespace by parent dir so the three per-model result.json files
                # don't collide on basename inside the artifact.
                name = os.path.join(os.path.basename(os.path.dirname(f)), os.path.basename(f))
                art.add_file(f, name=name)
            run.log_artifact(art)
    except Exception as exc:  # pragma: no cover
        print("[wandb] table/artifact logging failed ({0}).".format(exc))


def finish(run) -> None:
    if run is None:
        return
    try:
        run.finish()
    except Exception:
        pass
