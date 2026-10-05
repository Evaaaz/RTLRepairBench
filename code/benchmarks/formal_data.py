"""Canonical data construction for the VeriCodeGen formal experiments.

The semantic benchmark stores the *broken* RTL and the committed repair as a
unified diff.  Those two fields are the source of truth.  In particular, this
module intentionally never joins ``data/eval_tasks.jsonl`` for golden RTL: that
file has drifted for at least two designs.

The builder is strict by design.  Every context/deletion line in a diff must
match the broken source, and every mutation derived from the same seed must
reconstruct byte-identical canonical RTL.  A mismatch aborts construction
before any experiment can run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from project_paths import project_root  # noqa: E402


ROOT = project_root()
DEFAULT_BENCHMARK = ROOT / "data" / "repairbench_sem85.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "data" / "formal" / "canonical_goldens"
DEFAULT_MANIFEST = ROOT / "data" / "formal" / "sem85_manifest.json"

_RTL_BLOCK = re.compile(
    r"Broken RTL \(`(?P<filename>[^`]+)`\):\n"
    r"```(?:systemverilog|verilog)\n(?P<rtl>.*?)```",
    re.DOTALL,
)
_HUNK_HEADER = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)


class CanonicalDataError(ValueError):
    """The committed broken-source/diff pair is inconsistent."""


@dataclass(frozen=True)
class SemanticCase:
    case_id: str
    seed_id: str
    mutation: str
    filename: str
    broken_rtl: str
    repair_diff: str
    canonical_rtl: str

    @property
    def broken_sha256(self) -> str:
        return sha256_text(self.broken_rtl)

    @property
    def diff_sha256(self) -> str:
        return sha256_text(self.repair_diff)

    @property
    def canonical_sha256(self) -> str:
        return sha256_text(self.canonical_rtl)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _portable_path(path: Path) -> str:
    """Use a repository-relative path when possible, absolute otherwise."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(ROOT))
    except ValueError:
        return str(resolved)


def _canonical_newline(text: str) -> str:
    """Normalize transport newlines and use exactly one final newline."""

    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n") + "\n"


def _transport_newlines(text: str) -> str:
    """Normalize line endings without deleting patch-significant blank lines."""

    return text.replace("\r\n", "\n").replace("\r", "\n")


def extract_broken_rtl(user_message: str) -> tuple[str, str]:
    match = _RTL_BLOCK.search(user_message)
    if not match:
        raise CanonicalDataError("benchmark prompt has no parseable Broken RTL block")
    return match.group("filename"), _transport_newlines(match.group("rtl"))


def apply_unified_diff(source: str, patch: str) -> str:
    """Apply a standard unified diff without consulting the filesystem.

    The implementation supports multiple hunks and validates both the declared
    old/new hunk lengths and all source context.  File headers and the standard
    ``No newline at end of file`` marker are ignored.
    """

    source_lines = _transport_newlines(source).splitlines()
    patch_lines = _transport_newlines(patch).splitlines()
    result: list[str] = []
    source_pos = 0
    hunk_count = 0
    index = 0

    while index < len(patch_lines):
        header = _HUNK_HEADER.match(patch_lines[index])
        if not header:
            index += 1
            continue

        hunk_count += 1
        old_start = int(header.group("old_start")) - 1
        old_expected = int(header.group("old_count") or "1")
        new_expected = int(header.group("new_count") or "1")
        if old_start < source_pos or old_start > len(source_lines):
            raise CanonicalDataError(
                f"hunk {hunk_count} starts at invalid/overlapping source line {old_start + 1}"
            )
        result.extend(source_lines[source_pos:old_start])
        source_pos = old_start
        old_seen = 0
        new_seen = 0
        index += 1

        while index < len(patch_lines) and not _HUNK_HEADER.match(patch_lines[index]):
            line = patch_lines[index]
            index += 1
            if line.startswith("\\ No newline at end of file"):
                continue
            if not line or line[0] not in " +-":
                # A new file header belongs to the next file, which this
                # single-file benchmark must never contain.
                if line.startswith(("--- ", "+++ ")):
                    raise CanonicalDataError("multi-file diff is not supported")
                continue

            marker, payload = line[0], line[1:]
            if marker in " -":
                if source_pos >= len(source_lines) or source_lines[source_pos] != payload:
                    actual = "<EOF>" if source_pos >= len(source_lines) else source_lines[source_pos]
                    raise CanonicalDataError(
                        f"hunk {hunk_count} source mismatch at line {source_pos + 1}: "
                        f"expected {payload!r}, found {actual!r}"
                    )
                source_pos += 1
                old_seen += 1
            if marker in " +":
                result.append(payload)
                new_seen += 1

        if old_seen != old_expected or new_seen != new_expected:
            raise CanonicalDataError(
                f"hunk {hunk_count} length mismatch: header old/new "
                f"{old_expected}/{new_expected}, body {old_seen}/{new_seen}"
            )

    if not hunk_count:
        raise CanonicalDataError("assistant repair contains no unified-diff hunk")
    result.extend(source_lines[source_pos:])
    return _canonical_newline("\n".join(result))


def load_semantic_cases(path: Path = DEFAULT_BENCHMARK) -> list[SemanticCase]:
    cases: list[SemanticCase] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
                messages = item["messages"]
                filename, broken = extract_broken_rtl(messages[1]["content"])
                repair_diff = _canonical_newline(messages[2]["content"])
                canonical = apply_unified_diff(broken, repair_diff)
                cases.append(
                    SemanticCase(
                        case_id=str(item["id"]),
                        seed_id=str(item["seed_id"]),
                        mutation=str(item["mutation"]),
                        filename=filename,
                        broken_rtl=broken,
                        repair_diff=repair_diff,
                        canonical_rtl=canonical,
                    )
                )
            except (KeyError, IndexError, TypeError, json.JSONDecodeError, CanonicalDataError) as exc:
                raise CanonicalDataError(f"{path}:{line_number}: {exc}") from exc
    if not cases:
        raise CanonicalDataError(f"no semantic cases found in {path}")
    return cases


def assert_seed_consistency(cases: Iterable[SemanticCase]) -> dict[str, SemanticCase]:
    canonical_by_seed: dict[str, SemanticCase] = {}
    for case in cases:
        previous = canonical_by_seed.setdefault(case.seed_id, case)
        if previous.canonical_sha256 != case.canonical_sha256:
            raise CanonicalDataError(
                "canonical reconstruction disagrees across mutations for "
                f"{case.seed_id}: {previous.case_id}={previous.canonical_sha256}, "
                f"{case.case_id}={case.canonical_sha256}"
            )
    return canonical_by_seed


def build_manifest(
    cases: list[SemanticCase],
    source_path: Path,
    output_dir: Path,
) -> dict:
    by_seed = assert_seed_consistency(cases)
    case_rows = []
    for case in cases:
        case_rows.append(
            {
                "case_id": case.case_id,
                "seed_id": case.seed_id,
                "mutation": case.mutation,
                "broken_sha256": case.broken_sha256,
                "repair_diff_sha256": case.diff_sha256,
                "canonical_sha256": case.canonical_sha256,
                "canonical_path": _portable_path(output_dir / f"{case.seed_id}.sv"),
            }
        )
    designs = []
    for seed_id, representative in sorted(by_seed.items()):
        members = [case for case in cases if case.seed_id == seed_id]
        designs.append(
            {
                "seed_id": seed_id,
                "canonical_sha256": representative.canonical_sha256,
                "canonical_path": _portable_path(output_dir / f"{seed_id}.sv"),
                "case_ids": [case.case_id for case in members],
                "mutations": [case.mutation for case in members],
            }
        )
    return {
        "schema_version": 1,
        "construction": "apply committed assistant unified diff to broken RTL",
        "source_path": _portable_path(source_path),
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "case_count": len(cases),
        "design_count": len(by_seed),
        "cases": case_rows,
        "designs": designs,
    }


def write_canonical_dataset(
    source_path: Path = DEFAULT_BENCHMARK,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest_path: Path = DEFAULT_MANIFEST,
) -> dict:
    cases = load_semantic_cases(source_path)
    by_seed = assert_seed_consistency(cases)
    output_dir.mkdir(parents=True, exist_ok=True)
    for seed_id, case in sorted(by_seed.items()):
        (output_dir / f"{seed_id}.sv").write_text(case.canonical_rtl, encoding="utf-8")
    manifest = build_manifest(cases, source_path, output_dir)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    parser.add_argument(
        "--output-dir", type=Path,
        default=ROOT / "generated/formal/canonical_goldens",
    )
    parser.add_argument(
        "--manifest", type=Path,
        default=ROOT / "generated/formal/sem85_manifest.json",
    )
    args = parser.parse_args()
    manifest = write_canonical_dataset(args.benchmark, args.output_dir, args.manifest)
    print(
        json.dumps(
            {
                "manifest": str(args.manifest),
                "cases": manifest["case_count"],
                "designs": manifest["design_count"],
                "source_sha256": manifest["source_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
