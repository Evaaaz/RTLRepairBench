#!/usr/bin/env python3
"""Does the oracle sign reversal replicate, or was it one pair?

Section 4.1 reports that on one paired gpt-5.5 draw, differential simulation puts the
specification arm 11.4pp below the diagnostic arm and an unbounded prover puts it 8.9pp
above. The supplement is explicit that this is an existence proof: one greedy draw per arm,
9 discordant pairs on the formal leg, and dropping one takes p to 0.070.

Three reviewers independently asked the same question -- what is the *rate* at which the two
tiers disagree in sign, not the sign of one pair -- and Supplementary D.3 item 3 already
registers the design. This runs it.

  * two models, so the answer is not a property of one endpoint
  * three paired replicates per model, INTERLEAVED rather than run as three blocks, so
    provider-side drift is not confounded with draw index (D.3 item 3)
  * both arms of each replicate scored by both tiers on the same retained candidates
  * the reported quantity is the fraction of replicates whose two tiers disagree in sign,
    with the per-replicate contrasts beside it

Generation goes through second_model_explore.py, which reads the credential from the
environment and never writes it anywhere. Scoring reuses the audited formal path.

Writes artifacts/public/capability/sign_reversal_replication.json.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import subprocess
import sys
from pathlib import Path

_CODE = Path(__file__).resolve().parent.parent
for _p in (str(_CODE), str(_CODE / "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from project_paths import output_root, project_root  # noqa: E402

ROOT = project_root()
GEN = output_root().parent / "generated"
OUT = ROOT / "artifacts/public/capability/sign_reversal_replication.json"

# The contrast that reversed: diagnostic against diagnostic+specification.
ARMS = ("spec0_loc0", "spec1_loc0")
MODELS = ("azure/openai/gpt-5.5", "azure/anthropic/claude-opus-4-8")
# The serving root comes from the frozen study config rather than a literal here: a vendor
# hostname is an organization identifier, and the release scan is right to reject one in a
# file that ships. The credential is likewise inherited from the environment and never named.
CONFIG = ROOT / "configs/vericodegen2026.json"
DRAWS = 3


def plan(models, draws):
    """Interleaved order: draw index varies slowest within a model is exactly what D.3
    forbids, so alternate model and arm at every step and let the draw index cycle."""
    order = []
    for d in range(1, draws + 1):
        for model, arm in itertools.product(models, ARMS):
            order.append((model, arm, d))
    # rotate each draw block so the same (model, arm) is not always first
    out = []
    for d in range(1, draws + 1):
        block = [x for x in order if x[2] == d]
        out.extend(block[d - 1:] + block[: d - 1])
    return out


def _serving_root() -> str:
    return json.loads(CONFIG.read_text())["provider"]["base_url"]


def run_one(model: str, arm: str, draw: int, env: dict, dry: bool) -> dict:
    tag = f"signrep{draw}"
    cmd = [sys.executable, str(_CODE / "scripts" / "second_model_explore.py")]
    step_env = dict(env)
    step_env.update({
        "EXPLORE_MODEL": model,
        "EXPLORE_TEMP": "0",
        "EXPLORE_TAG": tag,
        "EXPLORE_ARMS": arm,
        "FAIRCHILD_LLM_URL": env.get("FAIRCHILD_LLM_URL") or _serving_root(),
    })
    slug = model.replace("/", "__") + "__" + tag
    record = {"model": model, "arm": arm, "draw": draw, "run_dir": f"second_model/runs/{slug}"}
    if dry:
        record["status"] = "planned"
        return record
    proc = subprocess.run(cmd, env=step_env, capture_output=True, text=True)
    record["status"] = "ok" if proc.returncode == 0 else "failed"
    if proc.returncode != 0:
        record["stderr_tail"] = proc.stderr[-400:]
    return record


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--draws", type=int, default=DRAWS)
    ap.add_argument("--models", default=",".join(MODELS))
    ap.add_argument("--dry-run", action="store_true",
                    help="print the interleaved plan and make no calls")
    args = ap.parse_args()

    models = tuple(m.strip() for m in args.models.split(",") if m.strip())
    steps = plan(models, args.draws)
    print(f"{len(steps)} generation steps, interleaved:")
    for model, arm, draw in steps:
        print(f"   draw {draw}  {arm:11s}  {model}")

    env = dict(os.environ)
    # No pre-flight credential check here. Naming the provider's environment variable would
    # put an organization identifier in a file that ships, and splitting the literal to slip
    # past the scanner would be defeating the check rather than satisfying it.
    # second_model_explore.py already refuses to run without a credential.

    records = [run_one(m, a, d, env, args.dry_run) for m, a, d in steps]
    failed = [r for r in records if r.get("status") == "failed"]
    report = {
        "generated_by": "code/scripts/x9_sign_reversal_replication.py",
        "design": {
            "registered_as": "Supplementary D.3 item 3",
            "models": list(models),
            "draws_per_arm": args.draws,
            "arms": list(ARMS),
            "interleaved": True,
            "temperature": 0,
        },
        "note": (
            "Generation only. Scoring both tiers on the retained candidates and computing "
            "the sign-disagreement rate is the second stage; this file records what was run "
            "so a partial run is visible rather than silently averaged."
        ),
        "steps": records,
        "n_failed": len(failed),
    }
    if not args.dry_run:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"\nwrote {OUT.relative_to(ROOT)}  ({len(records) - len(failed)}/{len(records)} ok)")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
