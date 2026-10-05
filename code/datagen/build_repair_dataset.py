"""Build the lint-class repair SFT set: data/repair_train.jsonl.

Pipeline (phase 1):

    seed (.sv that lints clean)
        -> inject_bugs.inject_all  (one mutation per variant)
        -> run_verilator_for_repairs.lint_source  (broken must FAIL lint)
        -> record { broken RTL + captured Verilator error -> unified-diff fix }

The fix label is a unified diff from the broken source back to the original, so
it is exact by construction. Each kept row is a real, tool-verified broken->fixed
pair in the {"messages": [...]} chat shape (same shape as
tuning/mg_verilog_to_prime_sft.py), ready for the SFT mixture.

Seeds: phase 1 runs against the committed fixtures to prove the pipeline. Point
``--seeds-dir`` at exported MG-Verilog modules (one self-contained .sv per file)
for the full >=500-example run. Self-contained seeds need no companions; modules
that instantiate another (like systolic_tile -> mac_cell) declare it via the
``SEED_SET`` companion map.
"""

from __future__ import annotations

import argparse
import difflib
import json
import os
import random
import sys
from pathlib import Path

if __package__:
    from project_paths import project_root
    from .inject_bugs import inject_all
    from .run_verilator_for_repairs import lint_source
else:  # compatibility: ``python datagen/build_repair_dataset.py``
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from project_paths import project_root
    from inject_bugs import inject_all
    from run_verilator_for_repairs import lint_source

REPO_ROOT = project_root()

# The committed-fixture fallback is only reached when no --seeds-dir is passed.
# Override the fixtures location with RTLREPAIR_FIXTURE_RTL; otherwise use the
# repo's generated/fixtures/rtl. Always pass --seeds-dir for full runs
# (build_dataset.sh does).
FIXTURE_RTL = Path(os.environ.get("RTLREPAIR_FIXTURE_RTL", REPO_ROOT / "generated" / "fixtures" / "rtl"))

SYSTEM_PROMPT = (
    "You are an expert hardware repair assistant. You are given a SystemVerilog "
    "module that fails Verilator lint, together with the exact tool error. Return "
    "a unified diff (and nothing else) that fixes the error and lints clean."
)


def _read(name: str) -> str:
    p = FIXTURE_RTL / name
    if not p.exists():
        raise SystemExit(
            f"fixture seed not found: {p}\n"
            "This standalone copy ships no fixtures. Pass --seeds-dir <dir of .sv "
            "seeds>, or set RTLREPAIR_FIXTURE_RTL to a fixtures/rtl directory."
        )
    return p.read_text()


# A seed is (primary filename, primary source, {companion name: source}). The
# companion map lets a module be linted with the submodules it instantiates.
def default_seeds() -> list[tuple[str, str, dict[str, str]]]:
    return [
        ("mac_cell.sv", _read("mac_cell.sv"), {}),
        ("systolic_tile.sv", _read("systolic_tile.sv"), {"mac_cell.sv": _read("mac_cell.sv")}),
    ]


def seeds_from_dir(seeds_dir: Path) -> list[tuple[str, str, dict[str, str]]]:
    """Load every ``*.sv`` in a directory as a self-contained seed (no companions)."""
    return [(p.name, p.read_text(), {}) for p in sorted(seeds_dir.glob("*.sv"))]


def make_diff(broken: str, fixed: str, fname: str) -> str:
    """Unified diff turning the broken source back into the clean original."""
    diff = difflib.unified_diff(
        broken.splitlines(keepends=True),
        fixed.splitlines(keepends=True),
        fromfile=f"a/{fname}",
        tofile=f"b/{fname}",
    )
    return "".join(diff)


def build_record(
    rec_id: str,
    mutation: str,
    seed_id: str,
    fname: str,
    broken: str,
    fixed: str,
    error_text: str,
) -> dict:
    user = (
        f"Broken RTL (`{fname}`):\n```systemverilog\n{broken}\n```\n\n"
        f"Verilator reported:\n```\n{error_text}\n```\n\n"
        "Return a unified diff that fixes the error."
    )
    return {
        "id": rec_id,
        "task": "repair",
        "bucket": "lint",
        "mutation": mutation,
        "seed_id": seed_id,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": make_diff(broken, fixed, fname)},
        ],
    }


def build(
    seeds,
    out_path: Path,
    *,
    mutation_seed: int = 0,
    cap_per_mutation: int = 0,
    verbose: bool = True,
) -> dict:
    records: list[dict] = []
    stats = {"seeds": 0, "skipped_seeds": 0, "kept": 0, "dropped": 0, "capped": 0}
    by_mutation: dict[str, int] = {}

    # Shuffle so a per-mutation cap draws a diverse subset of seeds, not just the
    # alphabetical head. Seeded for reproducibility.
    seeds = list(seeds)
    random.Random(mutation_seed).shuffle(seeds)

    for fname, src, companions in seeds:
        stats["seeds"] += 1
        seed_id = Path(fname).stem
        # The original must lint clean, or every "fix" we'd emit is itself broken.
        clean = lint_source(fname, src, companions)
        if clean.failed:
            stats["skipped_seeds"] += 1
            if verbose:
                print(f"[skip seed] {fname}: original does not lint clean: {clean.first}")
            continue

        for mutation, broken in inject_all(src, seed=mutation_seed):
            if cap_per_mutation and by_mutation.get(mutation, 0) >= cap_per_mutation:
                stats["capped"] += 1
                continue
            result = lint_source(fname, broken, companions)
            if not result.failed:
                stats["dropped"] += 1
                if verbose:
                    print(f"[drop] {seed_id}/{mutation}: broken still lints clean")
                continue
            rec_id = f"repair_lint_{stats['kept']:04d}"
            error_text = "\n".join(result.diagnostics)
            records.append(
                build_record(rec_id, mutation, seed_id, fname, broken, src, error_text)
            )
            stats["kept"] += 1
            by_mutation[mutation] = by_mutation.get(mutation, 0) + 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    stats["by_mutation"] = by_mutation
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--seeds-dir",
        help="directory of self-contained .sv seeds (default: committed fixtures)",
    )
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "generated" / "datagen" / "repair_lint.jsonl")
    )
    ap.add_argument("--mutation-seed", type=int, default=0)
    ap.add_argument(
        "--cap-per-mutation",
        type=int,
        default=0,
        help="max pairs per mutation type (0=uncapped); balances the mix",
    )
    args = ap.parse_args()

    if args.seeds_dir:
        seeds = seeds_from_dir(Path(args.seeds_dir))
    else:
        seeds = default_seeds()

    out_path = Path(args.out)
    stats = build(
        seeds,
        out_path,
        mutation_seed=args.mutation_seed,
        cap_per_mutation=args.cap_per_mutation,
    )
    print(
        f"\nseeds={stats['seeds']} skipped_seeds={stats['skipped_seeds']} "
        f"kept={stats['kept']} dropped={stats['dropped']} capped={stats['capped']}"
    )
    print(f"by_mutation={stats['by_mutation']}")
    print(f"-> {out_path}")


if __name__ == "__main__":
    main()
