"""Extract the held-out benchmark RTL into data/benchmark_tasks.jsonl.

This is the reference set the training data must NOT overlap (EXECUTION_PLAN.md
§7.1). We pull every golden module from the pinned VerilogEval and RTLLM clones
so decontaminate.py can drop any training example that reproduces one of them.

Each record is { id, source, code } — `code` is the golden RTL (contains
`endmodule`, so decontaminate.py hashes it), and any `module <name>` inside it
feeds the module-family check. Conservative on purpose: we include both
VerilogEval datasets (spec-to-rtl + code-complete) so near-duplicates of either
are caught.

Inputs (clone first, see EXECUTION_PLAN.md §7.1):
  data/raw/verilog-eval/**/*_ref.sv
  data/raw/RTLLM/**/verified_*.v
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from pathlib import Path

if __package__:
    from project_paths import project_root
else:  # compatibility: ``python datagen/build_benchmark_tasks.py``
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from project_paths import project_root

REPO_ROOT = project_root()

# The held-out clones live under --raw (or RTLREPAIR_RAW), defaulting to the
# repo's data/raw.
_DEFAULT_RAW = os.environ.get("RTLREPAIR_RAW") or str(REPO_ROOT / "data" / "raw")


def collect(raw: Path) -> list[dict]:
    tasks: list[dict] = []
    ve = raw / "verilog-eval"
    for p in sorted(ve.rglob("*_ref.sv")):
        tasks.append(
            {"id": f"verilogeval:{p.stem}", "source": "verilogeval", "code": p.read_text()}
        )
    rtllm = raw / "RTLLM"
    for p in sorted(rtllm.rglob("verified_*.v")):
        rel = p.relative_to(rtllm).as_posix().replace("/", "_")
        tasks.append({"id": f"rtllm:{rel}", "source": "rtllm", "code": p.read_text()})
    return tasks


EXPECTED_SOURCES = frozenset({"verilogeval", "rtllm"})


class IncompleteHeldOutError(RuntimeError):
    """The decontamination reference set is absent or only partially present."""


def validate_collected_tasks(
    tasks: list[dict], *, allow_partial_sources: bool = False
) -> dict[str, int]:
    """Validate that every expected held-out source contributed records."""

    counts = collections.Counter(str(task.get("source", "")) for task in tasks)
    ids = [task.get("id") for task in tasks]
    if any(not isinstance(task_id, str) or not task_id.strip() for task_id in ids):
        raise IncompleteHeldOutError("a held-out task has no non-empty string id")
    if len(set(ids)) != len(ids):
        raise IncompleteHeldOutError("held-out task ids are not unique")
    invalid_rtl = [
        str(task.get("id", "<unknown>"))
        for task in tasks
        if not isinstance(task.get("code"), str)
        or "endmodule" not in task["code"]
    ]
    if invalid_rtl:
        raise IncompleteHeldOutError(
            "held-out tasks contain missing or non-module RTL: "
            + ", ".join(invalid_rtl[:8])
        )
    unexpected = sorted(set(counts) - EXPECTED_SOURCES)
    if unexpected:
        raise IncompleteHeldOutError(
            "held-out tasks contain unexpected sources: " + ", ".join(unexpected)
        )
    missing = sorted(source for source in EXPECTED_SOURCES if counts[source] == 0)
    if missing and not allow_partial_sources:
        raise IncompleteHeldOutError(
            "held-out collection is incomplete; no records found for: "
            + ", ".join(missing)
            + ". Check --raw, or explicitly pass --allow-partial-sources for a "
            "non-paper diagnostic run."
        )
    return {source: counts[source] for source in sorted(EXPECTED_SOURCES)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "generated" / "datagen" / "benchmark_tasks.jsonl")
    )
    ap.add_argument(
        "--raw",
        default=_DEFAULT_RAW,
        help=(
            "dir holding verilog-eval/ and RTLLM/ clones "
            "(default: $RTLREPAIR_RAW or <repo>/data/raw)"
        ),
    )
    ap.add_argument(
        "--allow-partial-sources",
        action="store_true",
        help="allow one or both held-out sources to contribute zero records; "
        "never use this for a decontaminated paper dataset",
    )
    args = ap.parse_args()

    tasks = collect(Path(args.raw))
    try:
        by_source = validate_collected_tasks(
            tasks, allow_partial_sources=args.allow_partial_sources
        )
    except IncompleteHeldOutError as exc:
        ap.error(str(exc))
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for t in tasks:
            f.write(json.dumps(t) + "\n")

    print(f"benchmark_tasks={len(tasks)} by_source={by_source} -> {out_path}")


if __name__ == "__main__":
    main()
