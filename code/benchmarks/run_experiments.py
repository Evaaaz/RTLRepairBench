#!/usr/bin/env python3
"""Config-driven RepairBench runner.

Reads an experiment matrix (configs/experiments.json), turns each experiment into
a `repairbench_eval.py` invocation, runs them, and writes one result JSON per
experiment into $RTLREPAIR_OUT (default generated/repairbench). Then point analyze_repairbench.py
at that dir for the per-bucket table + between-arm McNemar contrasts.

Usage:
  python run_experiments.py --dry-run            # print the plan, run nothing
  python run_experiments.py                      # run every experiment
  python run_experiments.py --only base_diag base_diag_spec
  RTLREPAIR_LLM_URL=... ANTHROPIC_API_KEY=... ORIGEN_LLM_URL=... python run_experiments.py

Nothing runs under --dry-run, so it is safe to inspect the exact commands first.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from project_paths import output_root  # noqa: E402

DEFAULT_CONFIG = os.path.join(HERE, "configs", "experiments.json")
OUT_DIR = os.environ.get("RTLREPAIR_OUT") or str(output_root() / "repairbench")


def build_command(exp: dict, defaults: dict, bench: str | None,
                  workers: int | None = None) -> tuple[list[str], dict]:
    """Return (argv, env_overrides) for one experiment."""
    cfg = {**defaults, **{k: v for k, v in exp.items() if not k.startswith("_")}}
    signal = cfg.get("signal", [])
    argv = [sys.executable, os.path.join(HERE, "repairbench_eval.py"),
            "--backend", cfg.get("backend", "rtlrepair"),
            "--temp", str(cfg.get("temp", 0.0)),
            "--prompt-style", cfg.get("prompt_style", "default"),
            "--out", os.path.join(OUT_DIR, f"rb_{exp['name']}.json")]
    if cfg.get("backend") == "claude":
        argv += ["--claude-model", cfg.get("claude_model", "claude-opus-4-8")]
    else:
        argv += ["--model", cfg.get("model", "base")]
    if cfg.get("anon"):
        argv.append("--anon")
    if "spec" in signal:
        argv.append("--spec")
    if "locate" in signal:
        argv.append("--locate")
    if bench:
        argv += ["--bench", bench]
    # Concurrency is a run-level knob, not a per-arm one: vLLM is not
    # batch-invariant, so every arm in a comparison must share this value.
    w = workers if workers is not None else cfg.get("workers")
    if w:
        argv += ["--workers", str(w)]

    # Per-experiment served endpoint (e.g. a separate OriGen vLLM server).
    env_over = {}
    ep_env = exp.get("endpoint_env")
    if ep_env:
        url = os.environ.get(ep_env)
        if url:
            env_over["RTLREPAIR_LLM_URL"] = url
    return argv, env_over


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=DEFAULT_CONFIG)
    ap.add_argument("--bench", default=None, help="override the benchmark jsonl for all runs")
    ap.add_argument("--only", nargs="*", help="run only these experiment names")
    ap.add_argument("--dry-run", action="store_true", help="print the plan; run nothing")
    ap.add_argument("--workers", type=int, default=None,
                    help="in-flight items per arm (passed to repairbench_eval). "
                         "Applied to EVERY arm in the run so the comparison stays valid.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = json.load(f)
    defaults = cfg.get("defaults", {})
    experiments = [e for e in cfg["experiments"] if not args.only or e["name"] in args.only]
    if not experiments:
        sys.exit(f"no experiments matched --only {args.only}")

    # Fail before burning GPU hours: two arms that bank to the same
    # {model}/{version}/{signal} directory would silently overwrite each other
    # (see WHERE_TO_STORE_LLM_RESULTS.md). Checked over the whole matrix, not
    # just --only, since the collision is with an arm that may already be banked.
    from storage import check_no_collisions
    check_no_collisions(cfg["experiments"])

    os.makedirs(OUT_DIR, exist_ok=True)
    print(f"# {len(experiments)} experiment(s) -> {OUT_DIR}\n")
    failures = []
    for exp in experiments:
        argv, env_over = build_command(exp, defaults, args.bench, args.workers)
        note = f"  (endpoint <- ${exp['endpoint_env']})" if exp.get("endpoint_env") else ""
        printable = " ".join(f'{k}={v}' for k, v in env_over.items()) + (" " if env_over else "") + " ".join(argv)
        print(f"## {exp['name']}{note}\n{printable}\n")
        if args.dry_run:
            continue
        env = {**os.environ, **env_over}
        rc = subprocess.run(argv, env=env).returncode
        if rc != 0:
            failures.append((exp["name"], rc))
            print(f"!! {exp['name']} exited {rc}", flush=True)

    if args.dry_run:
        print("# dry run — nothing executed. Analyze with:\n"
              f"#   RTLREPAIR_OUT={OUT_DIR} python {os.path.join(HERE, 'analyze_repairbench.py')}")
        return
    if failures:
        sys.exit(f"{len(failures)} experiment(s) failed: {failures}")
    print("# all experiments complete. Analyze with:\n"
          f"#   RTLREPAIR_OUT={OUT_DIR} python {os.path.join(HERE, 'analyze_repairbench.py')}")


if __name__ == "__main__":
    main()
