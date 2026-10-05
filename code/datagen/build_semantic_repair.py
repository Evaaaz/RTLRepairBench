"""Build the semantic-class repair SFT set: data/repair_semantic.jsonl.

The lint bucket (build_repair_dataset.py) covers bugs Verilator can see. This
covers the other half: bugs that compile and lint clean but behave wrong. The
gate is differential simulation (diff_testbench.py) rather than lint:

    seed (clean, parseable, drivable module)
        -> inject_bugs.inject_semantic   (op_swap, flip_reset_polarity, ...)
        -> require broken LINTS CLEAN     (else it's a lint bug, not semantic)
        -> run_diff_test(original, broken) must DIVERGE (real behavioral bug)
        -> record { broken RTL + sim failure report -> unified-diff fix }

A seed only contributes if we can parse its ports and the golden module drives
cleanly through the harness (original-vs-original => DIFF_PASS); everything else
is skipped. Work is iverilog/vvp-bound, so seeds are processed in a thread pool.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

if __package__:
    from project_paths import project_root
    from .build_repair_dataset import make_diff
    from .diff_testbench import parse_module, require_simulator, run_diff_test
    from .inject_bugs import inject_semantic
    from .run_verilator_for_repairs import lint_source, require_verilator
else:  # compatibility: ``python datagen/build_semantic_repair.py``
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from project_paths import project_root
    from build_repair_dataset import make_diff
    from diff_testbench import parse_module, require_simulator, run_diff_test
    from inject_bugs import inject_semantic
    from run_verilator_for_repairs import lint_source, require_verilator

REPO_ROOT = project_root()

SYSTEM_PROMPT = (
    "You are an expert hardware repair assistant. You are given a SystemVerilog "
    "module that compiles and lints clean but fails a differential simulation "
    "against the reference behavior, together with the failure report. Return a "
    "unified diff (and nothing else) that fixes the behavioral bug."
)


def _record(rec_id, mutation, seed_id, fname, broken, fixed, error_text) -> dict:
    user = (
        f"Broken RTL (`{fname}`):\n```systemverilog\n{broken}\n```\n\n"
        f"Differential simulation reported:\n```\n{error_text}\n```\n\n"
        "Return a unified diff that fixes the behavioral bug."
    )
    return {
        "id": rec_id,
        "task": "repair",
        "bucket": "semantic",
        "mutation": mutation,
        "seed_id": seed_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": make_diff(broken, fixed, fname)},
        ],
    }


def _process_seed(fname: str, src: str) -> list[tuple]:
    """Return [(mutation, broken, error_text)] of verified semantic bugs for a seed."""
    return _process_seed_inner(fname, src)


def _process_seed_inner(fname: str, src: str) -> list[tuple]:
    info = parse_module(src)
    if info is None:
        return []
    # Harness sanity: the golden module must drive cleanly (identical => no diff).
    base = run_diff_test(src, src, info)
    if not base.compiled or base.diverged:
        return []
    found = []
    for mutation, broken in inject_semantic(src, seed=0):
        if lint_source(fname, broken).failed:
            continue  # caught by lint -> belongs to the lint bucket, not here
        result = run_diff_test(src, broken, info)
        if result.diverged:
            found.append((mutation, broken, src, result.error_text()))
    return [(fname, *f) for f in found]


class EmptySemanticDatasetError(RuntimeError):
    """The semantic gate completed without certifying any training pair."""


def build(
    seeds,
    out_path: Path,
    *,
    jobs: int = 8,
    cap_per_mutation: int = 0,
    allow_empty: bool = False,
) -> dict:
    require_verilator()
    require_simulator()
    seeds = list(seeds)
    if not seeds and not allow_empty:
        raise EmptySemanticDatasetError(
            "no .sv seeds were provided; refusing to write an empty semantic dataset"
        )
    random.Random(0).shuffle(seeds)
    stats = {"seeds": len(seeds), "kept": 0, "capped": 0}
    by_mutation: dict[str, int] = {}
    records: list[dict] = []

    with ThreadPoolExecutor(max_workers=jobs) as ex:
        # executor.map preserves the deterministic shuffled input order.  The
        # previous as_completed loop assigned row ids according to thread timing.
        results = ex.map(lambda pair: _process_seed(*pair), seeds)
        for seed_results in results:
            for fname, mutation, broken, fixed, error_text in seed_results:
                if cap_per_mutation and by_mutation.get(mutation, 0) >= cap_per_mutation:
                    stats["capped"] += 1
                    continue
                rec_id = f"repair_sem_{stats['kept']:04d}"
                records.append(
                    _record(rec_id, mutation, Path(fname).stem, fname, broken, fixed, error_text)
                )
                stats["kept"] += 1
                by_mutation[mutation] = by_mutation.get(mutation, 0) + 1

    if not records and not allow_empty:
        raise EmptySemanticDatasetError(
            "the semantic verifier certified zero pairs; check the seeds and toolchain "
            "instead of treating an empty file as a successful dataset"
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")
    stats["by_mutation"] = by_mutation
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seeds-dir", default=str(REPO_ROOT / "generated" / "datagen" / "mg_seeds")
    )
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "generated" / "datagen" / "repair_semantic.jsonl")
    )
    ap.add_argument("--jobs", type=int, default=8)
    ap.add_argument("--cap-per-mutation", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="cap seeds scanned (0=all)")
    ap.add_argument(
        "--allow-empty",
        action="store_true",
        help="permit an empty semantic output for a diagnostic run only",
    )
    args = ap.parse_args()

    paths = sorted(Path(args.seeds_dir).glob("*.sv"))
    if args.limit:
        paths = paths[: args.limit]
    seeds = [(p.name, p.read_text()) for p in paths]

    out_path = Path(args.out)
    try:
        stats = build(
            seeds,
            out_path,
            jobs=args.jobs,
            cap_per_mutation=args.cap_per_mutation,
            allow_empty=args.allow_empty,
        )
    except RuntimeError as exc:
        ap.error(str(exc))
    if args.allow_empty:
        print(
            "WARNING: --allow-empty was used; this diagnostic output is not a "
            "paper-valid semantic dataset."
        )
    print(
        f"\nseeds={stats['seeds']} kept={stats['kept']} capped={stats['capped']}\n"
        f"by_mutation={stats['by_mutation']}\n-> {out_path}"
    )


if __name__ == "__main__":
    main()
