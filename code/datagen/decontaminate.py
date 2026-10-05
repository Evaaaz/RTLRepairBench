"""Drop training examples that overlap the held-out benchmark.

Leakage hygiene is non-negotiable for the Layer-1 claim (EXECUTION_PLAN.md §7.1):
training on (or near) the VerilogEval/RTLLM/HDLBits problems we benchmark on would
inflate the numbers and invalidate the whole result. This filter removes any
training record that matches a held-out item by:

    * **id**             — exact id collision
    * **module_family**  — same module name (catches the same problem reworded)
    * **source hash**    — comment/whitespace-insensitive hash of the RTL
                           (catches paraphrased near-duplicates exact match misses)

It is deliberately **schema-agnostic**: it scans each record's full text for module
names and for any RTL block (anything containing ``endmodule``), so it works on the
repair set, the spec2rtl set, and the benchmark file without per-format wiring.
Missing or empty held-out fingerprints are fatal by default: silently passing every
row would make the resulting dataset falsely appear decontaminated.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

if __package__:
    from project_paths import project_root
else:  # compatibility: ``python datagen/decontaminate.py``
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from project_paths import project_root

REPO_ROOT = project_root()

_MODULE_NAME_RE = re.compile(r"\bmodule\s+(\w+)")
_LINE_COMMENT_RE = re.compile(r"//.*")
_BLOCK_COMMENT_RE = re.compile(r"/\*.*?\*/", re.DOTALL)

# Generic module names carry no identity (benchmarks name the DUT `RefModule`,
# `top_module`, …). Matching on them would drop unrelated training modules
# wholesale, so the family check ignores them — source-hash + id still catch
# real overlap.
_GENERIC_NAMES = frozenset(
    {"refmodule", "top_module", "top", "dut", "tb", "testbench", "test", "module", "main"}
)


def _record_text(rec: dict) -> str:
    """All string content of a record, flattened — what we scan for leakage."""
    return json.dumps(rec)


def module_families(text: str) -> set[str]:
    """Every non-generic ``module <name>`` declared anywhere in the text."""
    names = {m.group(1).lower() for m in _MODULE_NAME_RE.finditer(text)}
    return names - _GENERIC_NAMES


def _normalize_rtl(src: str) -> str:
    src = _LINE_COMMENT_RE.sub("", src)
    src = _BLOCK_COMMENT_RE.sub("", src)
    src = re.sub(r"\s+", " ", src)
    return src.strip().lower()


def source_hashes(rec: dict) -> set[str]:
    """Normalized-source hashes of every RTL-looking string value in a record."""
    hashes: set[str] = set()

    def walk(v) -> None:
        if isinstance(v, str):
            if "endmodule" in v:
                norm = _normalize_rtl(v)
                if norm:
                    hashes.add(hashlib.sha1(norm.encode()).hexdigest())
        elif isinstance(v, dict):
            for x in v.values():
                walk(x)
        elif isinstance(v, list):
            for x in v:
                walk(x)

    walk(rec)
    return hashes


@dataclass
class HeldOut:
    """The held-out fingerprints a training record must not match."""

    ids: set[str] = field(default_factory=set)
    families: set[str] = field(default_factory=set)
    hashes: set[str] = field(default_factory=set)

    @classmethod
    def from_jsonl(cls, path: Path) -> "HeldOut":
        held = cls()
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if isinstance(rec.get("id"), str):
                held.ids.add(rec["id"])
            held.families |= module_families(_record_text(rec))
            held.hashes |= source_hashes(rec)
        return held


class EmptyBenchmarkError(RuntimeError):
    """No usable held-out fingerprints were available for decontamination."""


def load_held_out(path: Path, *, allow_empty: bool = False) -> HeldOut:
    """Load held-out fingerprints, rejecting a missing/empty reference by default."""

    if not path.exists():
        if allow_empty:
            return HeldOut()
        raise EmptyBenchmarkError(
            f"benchmark file {path} does not exist; refusing to label the output "
            "decontaminated"
        )
    held = HeldOut.from_jsonl(path)
    if not (held.ids or held.families or held.hashes) and not allow_empty:
        raise EmptyBenchmarkError(
            f"benchmark file {path} contains no usable held-out fingerprints; "
            "refusing to pass every training row through"
        )
    return held


def overlaps(rec: dict, held: HeldOut) -> list[str]:
    """Return the reasons ``rec`` leaks (empty list = clean)."""
    reasons: list[str] = []
    if isinstance(rec.get("id"), str) and rec["id"] in held.ids:
        reasons.append("id")
    if module_families(_record_text(rec)) & held.families:
        reasons.append("module_family")
    if source_hashes(rec) & held.hashes:
        reasons.append("source_hash")
    return reasons


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="inp", required=True, help="training JSONL to filter")
    output = ap.add_mutually_exclusive_group(required=True)
    output.add_argument("--out", help="fresh filtered output path")
    output.add_argument(
        "--in-place", action="store_true",
        help="explicitly replace --in after a complete successful filter",
    )
    ap.add_argument(
        "--benchmark",
        default=str(REPO_ROOT / "generated" / "datagen" / "benchmark_tasks.jsonl"),
    )
    ap.add_argument(
        "--allow-empty-benchmark",
        action="store_true",
        help="explicitly permit a missing/empty held-out set for a diagnostic, "
        "non-decontaminated run",
    )
    args = ap.parse_args()

    in_path = Path(args.inp)
    out_path = in_path if args.in_place else Path(args.out)
    bench_path = Path(args.benchmark)

    records = [json.loads(l) for l in in_path.read_text().splitlines() if l.strip()]

    try:
        held = load_held_out(
            bench_path, allow_empty=args.allow_empty_benchmark
        )
    except EmptyBenchmarkError as exc:
        ap.error(str(exc))

    if not (held.ids or held.families or held.hashes):
        print(
            "WARNING: --allow-empty-benchmark was used; output is NOT "
            f"decontaminated. Passing all {len(records)} records through."
        )
    else:
        print(
            f"held-out fingerprints: {len(held.ids)} ids, "
            f"{len(held.families)} families, {len(held.hashes)} source hashes"
        )

    kept, dropped = [], []
    reason_counts: dict[str, int] = {}
    for rec in records:
        reasons = overlaps(rec, held)
        if reasons:
            dropped.append((rec.get("id"), reasons))
            for r in reasons:
                reason_counts[r] = reason_counts.get(r, 0) + 1
        else:
            kept.append(rec)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=out_path.parent,
            prefix=f".{out_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            for rec in kept:
                handle.write(json.dumps(rec) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, out_path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)

    print(f"in={len(records)} kept={len(kept)} dropped={len(dropped)} -> {out_path}")
    if dropped:
        print(f"drop reasons: {reason_counts}")
        for rid, reasons in dropped[:10]:
            print(f"  dropped {rid}: {','.join(reasons)}")


if __name__ == "__main__":
    main()
