#!/usr/bin/env python3
"""Build and verify the allowlisted standalone release tree.

The exporter is deliberately standard-library-only and fail closed.  It copies
only paths named by ``artifact.lock.json`` and rejects symlinks, oversized
files, credential-shaped strings, workstation paths, and raw reasoning traces
in released data.  Every export receives a deterministic content manifest.

Typical use::

    python tools/export_standalone.py --check
    python tools/export_standalone.py --output /tmp/lint-vs-semantic-release
    python tools/export_standalone.py --smoke
"""

from __future__ import annotations

import argparse
import csv
import fnmatch
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Iterable, Mapping, Sequence


LOCK_NAME = "artifact.lock.json"
EXPORT_MANIFEST = "STANDALONE_MANIFEST.json"
SUPPORTED_SCHEMA_VERSION = 1


class ReleaseError(RuntimeError):
    """A release-policy or integrity check failed."""


SECRET_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("provider API key", re.compile(rb"(?<![A-Za-z0-9])sk-[A-Za-z0-9_-]{20,}")),
    ("NVIDIA API key", re.compile(rb"(?<![A-Za-z0-9])nvapi-[A-Za-z0-9_-]{16,}", re.I)),
    ("AWS access key", re.compile(rb"(?<![A-Z0-9])(?:AKIA|ASIA)[A-Z0-9]{16}(?![A-Z0-9])")),
    ("GitHub token", re.compile(rb"(?<![A-Za-z0-9])(?:gh[opsu]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,})")),
    ("Slack token", re.compile(rb"(?<![A-Za-z0-9])xox[baprs]-[A-Za-z0-9-]{20,}")),
    ("Google API key", re.compile(rb"(?<![A-Za-z0-9])AIza[0-9A-Za-z_-]{30,}")),
    ("private key", re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("credential-bearing URL", re.compile(rb"https?://[^/\s:@]+:[^/\s@]+@", re.I)),
    (
        "authorization credential",
        re.compile(
            rb"\bAuthorization\s*[:=]\s*(?:Bearer|Basic)\s+"
            rb"[A-Za-z0-9._~+/=-]{12,}",
            re.I,
        ),
    ),
    (
        "bearer credential",
        re.compile(rb"(?<![A-Za-z0-9])Bearer[ \t]+[A-Za-z0-9._~+/=-]{20,}", re.I),
    ),
)

# Require a concrete username and a following path component.  This avoids
# flagging source code that itself contains a generic detector such as
# ``/Users/|/home/`` while still rejecting real workstation paths.
ABSOLUTE_PATH_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "POSIX home path",
        re.compile(
            rb"(?<![A-Za-z0-9])/(?:Users|home)/[A-Za-z0-9._-]+/[^\s\x00\"'<>]+"
        ),
    ),
    (
        "Windows home path",
        re.compile(
            rb"(?i)(?<![A-Za-z0-9])[A-Z]:\\Users\\[A-Za-z0-9._-]+\\[^\s\x00\"'<>]+"
        ),
    ),
    (
        "ephemeral/workspace absolute path",
        re.compile(
            rb"(?<![A-Za-z0-9])/(?:"
            rb"private/(?:tmp|var/folders)|var/folders|Volumes/[A-Za-z0-9._-]+|"
            rb"root|(?:github/)?workspace|workspaces|runner/_work|mnt/(?:data|workspace)"
            rb")/[A-Za-z0-9._~-][^\s\x00\"'<>]*"
        ),
    ),
)

NETWORK_IDENTITY_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    (
        "literal IPv4 endpoint",
        re.compile(
            rb"(?<![0-9.])(?!0\.0\.0\.0(?:[^0-9.]|$))(?!127\.0\.0\.1(?:[^0-9.]|$))"
            rb"(?:[0-9]{1,3}\.){3}[0-9]{1,3}(?![0-9.])"
        ),
    ),
)

# Double-blind: the paper is anonymous but the supplement is shipped alongside it, so an
# employer name in a config, an endpoint, or a run-directory string can deanonymize the
# authors just as effectively. Author-specific names, handles, and email addresses are
# intentionally scanned with a private denylist outside this payload; embedding that list
# here would itself disclose the identities it is meant to protect.
#
# The routing namespace of a served model id is the one place the generic organization
# detector collides with scientific content, so it is allowlisted per file rather than
# silently stripped -- see ORG_IDENTITY_ALLOWED_PATHS.
ORG_IDENTITY_PATTERNS: tuple[tuple[str, re.Pattern[bytes]], ...] = (
    ("organization identifier", re.compile(rb"(?i)\bnvidia\b|\bnvcf\b|\bngc\b")),
    ("personal serving endpoint", re.compile(rb"(?<![<\w-])[a-z0-9]{3,}--[a-z0-9-]+\.modal\.run")),
    ("POSIX home path", re.compile(rb"/(?:Users|home)/(?!<)[A-Za-z0-9._-]+/")),
)

# Model ids carry their serving namespace and several analysis scripts join on the full
# string, so renaming them is a coordinated change across code, configs and the lock rather
# than a redaction. Until that is done these files are known exceptions, not clean.
ORG_IDENTITY_ALLOWED_PATHS = frozenset({
    # Model ids carry their serving namespace and several analysis scripts join on the full
    # string, so renaming them is a coordinated change across code, configs and the lock
    # rather than a redaction.
    "artifacts/public/capability/capability_curve.json",
    "artifacts/public/capability/counterexample_contrast.json",
    "artifacts/public/capability/curve68_cohort_audit.json",
    "artifacts/public/primary/analysis/e0_coverage.json",
    "artifacts/public/primary/ledger/public_events.jsonl",
    "configs/vericodegen2026.json",
    "code/scripts/e0_coverage_audit.py",
    "code/scripts/e6_curve_exhibits.py",
    "code/scripts/e6_decomposition_table.py",
    "code/scripts/repairbench_slot.py",
    "code/scripts/second_model_capability_curve.py",
    "code/scripts/second_model_explore.py",
    "code/scripts/self_reflect.py",
    "code/scripts/x1_counterexample_power.py",
    # The serving backend is named for its provider; renaming the module touches the frozen
    # preregistered harness and its tests, so it is tracked rather than done in place.
    "code/benchmarks/backend/nvidia_responses.py",
    "code/benchmarks/vericodegen_eval.py",
    "code/scripts/eval_endpoint.py",
    "code/benchmarks/README.md",
    "tests/test_nvidia_responses.py",
    "tests/test_vericodegen_eval.py",
    "tests/standalone/test_export_standalone.py",
    # Development history, shipped for provenance rather than for review.
    "docs/history/audits/E0_COVERAGE_AUDIT.md",
    "docs/history/audits/REVISION_PLAN.md",
    "docs/history/migration/MIGRATION_MANIFEST.md",
    "docs/history/migration/reports/R2b.md",
    "docs/history/migration/reports/R2c.md",
    "docs/history/migration/reports/R2d.md",
    "paper/E2_REALBUG.md",
    "paper/e0_checklist.json",
    "tools/export_standalone.py",
})

REASONING_TAG = re.compile(
    rb"<\s*/?\s*(?:think|analysis|reasoning)(?:\s+[^>]*)?\s*>", re.I
)
DATA_SUFFIXES = {".csv", ".json", ".jsonl", ".ndjson", ".tsv"}
STRUCTURED_DATA_SUFFIXES = {".json", ".jsonl", ".ndjson"}
RAW_PRIVATE_FIELD_KEYS = frozenset(
    {
        "analysis",
        "chain_of_thought",
        "chainofthought",
        "provider_response",
        "raw_completion",
        "raw_output",
        "raw_provider_response",
        "raw_response",
        "raw_response_body",
        "reasoning",
        "reasoning_content",
        "response_body",
        "scratchpad",
        "thinking",
    }
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def default_root() -> Path:
    return Path(__file__).resolve().parents[1]


def relative_posix(path: Path, root: Path) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as exc:
        raise ReleaseError(f"path escapes release root: {path}") from exc
    value = relative.as_posix()
    if value in {"", "."} or value.startswith("../"):
        raise ReleaseError(f"invalid release-relative path: {value!r}")
    return value


def _require_type(value: Any, expected: type, label: str) -> Any:
    if not isinstance(value, expected):
        raise ReleaseError(f"{label} must be {expected.__name__}")
    return value


def load_lock(root: Path) -> dict[str, Any]:
    path = root / LOCK_NAME
    if not path.is_file():
        raise ReleaseError(f"missing {LOCK_NAME}: {path}")
    try:
        lock = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot parse {path}: {exc}") from exc
    _require_type(lock, dict, LOCK_NAME)
    if lock.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ReleaseError(
            f"unsupported lock schema {lock.get('schema_version')!r}; "
            f"expected {SUPPORTED_SCHEMA_VERSION}"
        )
    _require_type(lock.get("artifact_name"), str, "artifact_name")
    _require_type(lock.get("include"), list, "include")
    _require_type(lock.get("exclude"), list, "exclude")
    _require_type(lock.get("required_paths"), list, "required_paths")
    _require_type(lock.get("paper_inputs"), list, "paper_inputs")
    _require_type(lock.get("scan_exceptions", []), list, "scan_exceptions")
    policy = _require_type(lock.get("policy"), dict, "policy")
    maximum = policy.get("max_file_bytes")
    if not isinstance(maximum, int) or maximum <= 0:
        raise ReleaseError("policy.max_file_bytes must be a positive integer")
    return lock


def path_matches(path: str, pattern: str) -> bool:
    """Match a POSIX path with ``**`` as zero or more complete components.

    ``fnmatch`` lets ``*`` consume ``/`` and ``PurePath.match`` treated ``**``
    inconsistently across supported Python versions.  Component-wise matching
    keeps a one-level pattern such as ``results/*.json`` from accidentally
    selecting or excluding nested release data.
    """

    path_parts = tuple(path.split("/"))
    pattern_parts = tuple(pattern.split("/"))
    memo: dict[tuple[int, int], bool] = {}

    def match(path_index: int, pattern_index: int) -> bool:
        key = (path_index, pattern_index)
        if key in memo:
            return memo[key]
        if pattern_index == len(pattern_parts):
            result = path_index == len(path_parts)
        elif pattern_parts[pattern_index] == "**":
            result = match(path_index, pattern_index + 1) or (
                path_index < len(path_parts)
                and match(path_index + 1, pattern_index)
            )
        else:
            result = (
                path_index < len(path_parts)
                and fnmatch.fnmatchcase(
                    path_parts[path_index], pattern_parts[pattern_index]
                )
                and match(path_index + 1, pattern_index + 1)
            )
        memo[key] = result
        return result

    return match(0, 0)


def is_excluded(relative: str, lock: Mapping[str, Any]) -> bool:
    patterns = lock["exclude"]
    return any(path_matches(relative, pattern) for pattern in patterns)


def selected_files(root: Path, lock: Mapping[str, Any]) -> list[Path]:
    selected: dict[str, Path] = {}
    empty_patterns: list[str] = []
    for pattern in lock["include"]:
        if not isinstance(pattern, str) or not pattern:
            raise ReleaseError("include entries must be non-empty strings")
        matches = 0
        for path in root.glob(pattern):
            if not path.is_file() and not path.is_symlink():
                continue
            relative = relative_posix(path, root)
            if is_excluded(relative, lock):
                continue
            selected[relative] = path
            matches += 1
        if matches == 0:
            empty_patterns.append(pattern)
    if empty_patterns:
        raise ReleaseError(
            "allowlist patterns matched no releasable files: "
            + ", ".join(empty_patterns)
        )

    required = lock["required_paths"]
    for relative in required:
        if not isinstance(relative, str) or not relative:
            raise ReleaseError("required_paths entries must be non-empty strings")
        if relative not in selected:
            raise ReleaseError(f"required path is missing or not allowlisted: {relative}")

    if lock["policy"].get("require_all_files_classified") is True:
        unclassified = []
        for path in root.rglob("*"):
            if not path.is_file() and not path.is_symlink():
                continue
            relative = relative_posix(path, root)
            if relative in selected or is_excluded(relative, lock):
                continue
            unclassified.append(relative)
        if unclassified:
            raise ReleaseError(
                "files are neither allowlisted nor explicitly excluded: "
                + ", ".join(sorted(unclassified))
            )

    return [selected[key] for key in sorted(selected)]


def _is_released_data(relative: str) -> bool:
    path = PurePosixPath(relative)
    data_area = any(part in {"artifacts", "data", "results"} for part in path.parts)
    return data_area and path.suffix.lower() in DATA_SUFFIXES


def _is_executable_config(relative: str) -> bool:
    """Return whether a JSON file is implementation-owned configuration.

    Files below the release's root ``configs/`` directory are scientific
    inputs and therefore receive the same private-field audit as result data.
    The narrow exception below is only for executable implementation config
    embedded below ``code/**/configs``.
    """

    path = PurePosixPath(relative)
    return (
        path.suffix.lower() == ".json"
        and bool(path.parts)
        and path.parts[0] == "code"
        and "configs" in path.parts
    )


def _requires_private_field_scan(relative: str) -> bool:
    path = PurePosixPath(relative)
    suffix = path.suffix.lower()
    if suffix not in DATA_SUFFIXES or _is_executable_config(relative):
        return False
    return _is_released_data(relative) or (
        bool(path.parts) and path.parts[0] == "configs"
    )


def _may_describe_reasoning_tags(relative: str) -> bool:
    """Allow tag literals only in executable/template implementation source."""

    path = PurePosixPath(relative)
    implementation = path.suffix.lower() in {".py", ".jinja", ".jinja2"} and (
        not path.parts or path.parts[0] in {"code", "src", "tests", "tools"}
    )
    return implementation or _is_executable_config(relative)


def _normalized_payload_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", key.strip().lower()).strip("_")


def _scan_private_fields(value: Any, relative: str, location: str = "$") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            if not isinstance(key, str):
                raise ReleaseError(f"{relative}: non-string JSON key at {location}")
            normalized = _normalized_payload_key(key)
            child_location = f"{location}.{key}"
            if normalized in RAW_PRIVATE_FIELD_KEYS:
                raise ReleaseError(
                    f"{relative}: forbidden raw reasoning/response field at "
                    f"{child_location}"
                )
            _scan_private_fields(child, relative, child_location)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _scan_private_fields(child, relative, f"{location}[{index}]")


def _scan_structured_release_data(relative: str, payload: bytes) -> None:
    suffix = PurePosixPath(relative).suffix.lower()
    if suffix not in STRUCTURED_DATA_SUFFIXES or not _requires_private_field_scan(relative):
        return
    try:
        text = payload.decode("utf-8")
        if suffix == ".json":
            documents = [json.loads(text)]
        else:
            documents = [
                json.loads(line)
                for line in text.splitlines()
                if line.strip()
            ]
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"{relative}: malformed released structured data: {exc}") from exc
    for index, document in enumerate(documents):
        location = "$" if suffix == ".json" else f"$line[{index + 1}]"
        _scan_private_fields(document, relative, location)


def _scan_tabular_release_data(relative: str, payload: bytes) -> None:
    suffix = PurePosixPath(relative).suffix.lower()
    if suffix not in {".csv", ".tsv"} or not _requires_private_field_scan(relative):
        return
    try:
        text = payload.decode("utf-8-sig")
        reader = csv.reader(
            io.StringIO(text, newline=""),
            delimiter="\t" if suffix == ".tsv" else ",",
            strict=True,
        )
        header = next(reader, [])
        # Consume the iterator so malformed quoting after the header cannot
        # escape validation merely because private fields are header-based.
        for _ in reader:
            pass
    except (UnicodeDecodeError, csv.Error) as exc:
        raise ReleaseError(f"{relative}: malformed released tabular data: {exc}") from exc
    for index, key in enumerate(header, start=1):
        if _normalized_payload_key(key) in RAW_PRIVATE_FIELD_KEYS:
            raise ReleaseError(
                f"{relative}: forbidden raw reasoning/response field in "
                f"column {index} ({key!r})"
            )


def scan_payload(
    relative: str,
    payload: bytes,
    allowed_detectors: Iterable[str] = (),
) -> None:
    allowed = set(allowed_detectors)
    for label, pattern in SECRET_PATTERNS:
        if label not in allowed and pattern.search(payload):
            raise ReleaseError(f"{relative}: detected {label}")
    for label, pattern in ABSOLUTE_PATH_PATTERNS:
        if label not in allowed and pattern.search(payload):
            raise ReleaseError(f"{relative}: detected {label}")
    for label, pattern in NETWORK_IDENTITY_PATTERNS:
        if pattern.search(payload):
            raise ReleaseError(f"{relative}: detected {label}")
    # only text payloads: compressed streams in a PDF hit a three-letter pattern by chance
    try:
        payload.decode("utf-8")
    except UnicodeDecodeError:
        pass
    else:
        for label, pattern in ORG_IDENTITY_PATTERNS:
            if label in allowed or relative in ORG_IDENTITY_ALLOWED_PATHS:
                continue
            if pattern.search(payload):
                raise ReleaseError(f"{relative}: detected {label}")
    if REASONING_TAG.search(payload) and not _may_describe_reasoning_tags(relative):
        raise ReleaseError(f"{relative}: raw reasoning tag is forbidden in release payloads")
    _scan_structured_release_data(relative, payload)
    _scan_tabular_release_data(relative, payload)


def scan_exception_detectors(
    relative: str,
    payload: bytes,
    lock: Mapping[str, Any],
) -> set[str]:
    """Return narrowly approved detector exceptions for an exact file hash."""

    known = {label for label, _ in SECRET_PATTERNS + ABSOLUTE_PATH_PATTERNS}
    matches = []
    for index, item in enumerate(lock.get("scan_exceptions", [])):
        if not isinstance(item, dict):
            raise ReleaseError(f"scan_exceptions[{index}] must be an object")
        if item.get("path") == relative:
            matches.append((index, item))
    if len(matches) > 1:
        raise ReleaseError(f"multiple scan exceptions target {relative}")
    if not matches:
        return set()

    index, item = matches[0]
    expected = item.get("sha256")
    if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ReleaseError(f"scan_exceptions[{index}].sha256 is not a SHA-256")
    actual = sha256_bytes(payload)
    if actual != expected:
        raise ReleaseError(
            f"scan exception hash mismatch for {relative}: expected {expected}, got {actual}"
        )
    detectors = item.get("detectors")
    if not isinstance(detectors, list) or not detectors:
        raise ReleaseError(f"scan_exceptions[{index}].detectors must be non-empty")
    if any(not isinstance(value, str) or value not in known for value in detectors):
        raise ReleaseError(f"scan_exceptions[{index}] names an unknown detector")
    reason = item.get("reason")
    if not isinstance(reason, str) or not reason.strip():
        raise ReleaseError(f"scan_exceptions[{index}].reason must be non-empty")
    return set(detectors)


def validate_file(path: Path, root: Path, lock: Mapping[str, Any]) -> dict[str, Any]:
    relative = relative_posix(path, root)
    if path.is_symlink():
        raise ReleaseError(f"symlinks are forbidden: {relative}")
    if not path.is_file():
        raise ReleaseError(f"not a regular file: {relative}")
    size = path.stat().st_size
    maximum = lock["policy"]["max_file_bytes"]
    if size > maximum:
        raise ReleaseError(
            f"{relative}: {size} bytes exceeds max_file_bytes={maximum}"
        )
    payload = path.read_bytes()
    allowed_detectors = scan_exception_detectors(relative, payload, lock)
    if lock["policy"].get("anonymous_manuscript", True) is not True:
        # The organization detector exists only for double-blind review; a named release
        # credits its authors' employer and upstream copyright holders by design.
        allowed_detectors = allowed_detectors | {"organization identifier"}
    scan_payload(relative, payload, allowed_detectors)
    if path.suffix == ".py":
        try:
            compile(payload, relative, "exec")
        except (SyntaxError, ValueError) as exc:
            raise ReleaseError(f"{relative}: Python syntax check failed: {exc}") from exc
    executable = bool(path.stat().st_mode & stat.S_IXUSR)
    return {
        "path": relative,
        "bytes": size,
        "sha256": sha256_bytes(payload),
        "executable": executable,
    }


def validate_paper_inputs(
    root: Path,
    lock: Mapping[str, Any],
    selected_relatives: set[str],
) -> list[dict[str, Any]]:
    verified: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, item in enumerate(lock["paper_inputs"]):
        if not isinstance(item, dict):
            raise ReleaseError(f"paper_inputs[{index}] must be an object")
        relative = item.get("path")
        expected = item.get("sha256")
        role = item.get("role")
        if not isinstance(relative, str) or not relative:
            raise ReleaseError(f"paper_inputs[{index}].path must be non-empty")
        if relative in seen:
            raise ReleaseError(f"duplicate paper input: {relative}")
        seen.add(relative)
        if not isinstance(role, str) or not role:
            raise ReleaseError(f"paper_inputs[{index}].role must be non-empty")
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ReleaseError(f"paper_inputs[{index}].sha256 is not a SHA-256")
        if relative not in selected_relatives:
            raise ReleaseError(f"paper input is not included in the release: {relative}")
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ReleaseError(f"paper input is missing or not regular: {relative}")
        actual = sha256_file(path)
        if actual != expected:
            raise ReleaseError(
                f"paper input hash mismatch for {relative}: expected {expected}, got {actual}"
            )
        verified.append({"path": relative, "role": role, "sha256": actual})
    if not verified:
        raise ReleaseError("paper_inputs must pin at least one released input")
    return verified


def validate_anonymous_manuscript(root: Path) -> None:
    source = root / "paper" / "paper.tex"
    if not source.is_file():
        raise ReleaseError("anonymous manuscript source is missing: paper/paper.tex")
    text = source.read_text(encoding="utf-8")
    author_blocks = re.findall(r"\\author\{([^}]*)\}", text)
    if author_blocks != ["Anonymous Author(s)"]:
        raise ReleaseError("paper/paper.tex must contain exactly one anonymous author block")
    if re.search(r"\\(?:thanks|affiliation|institute|email|orcid)\b", text, re.I):
        raise ReleaseError("paper/paper.tex contains identifying author metadata commands")
    style = re.search(
        r"\\usepackage\[([^]]*)\]\{neurips_2026\}", text
    )
    if style is None or "dblblindworkshop" not in {
        option.strip() for option in style.group(1).split(",")
    }:
        raise ReleaseError(
            "paper/paper.tex must use the neurips_2026 dblblindworkshop style"
        )
    if not re.search(
        r"\\workshoptitle\{\s*AI for Chip Design\s*\}", text
    ):
        raise ReleaseError(
            "paper/paper.tex must name the AI for Chip Design workshop"
        )
    if not re.search(r"\\input\{checklist\}", text):
        raise ReleaseError("paper/paper.tex must include paper/checklist.tex")

    checklist = root / "paper" / "checklist.tex"
    if not checklist.is_file():
        raise ReleaseError("official checklist source is missing: paper/checklist.tex")
    checklist_text = checklist.read_text(encoding="utf-8")
    if re.search(r"\\(?:answerTODO|justificationTODO)\b", checklist_text):
        raise ReleaseError("paper/checklist.tex contains unanswered TODO fields")


def validate_camera_ready_manuscript(root: Path) -> None:
    source = root / "paper" / "paper.tex"
    if not source.is_file():
        raise ReleaseError("camera-ready manuscript source is missing: paper/paper.tex")
    text = source.read_text(encoding="utf-8")
    style = re.search(r"\\usepackage\[([^]]*)\]\{neurips_2026\}", text)
    options = {option.strip() for option in style.group(1).split(",")} if style else set()
    if not {"final", "dblblindworkshop"} <= options:
        raise ReleaseError("paper/paper.tex must use the final neurips_2026 workshop style")
    if not re.search(r"\\workshoptitle\{\s*AI for Chip Design\s*\}", text):
        raise ReleaseError("paper/paper.tex must name the AI for Chip Design workshop")
    if "Anonymous Author" in text:
        raise ReleaseError("paper/paper.tex still carries the anonymous author block")


def check_source(root: Path) -> dict[str, Any]:
    root = root.resolve()
    lock = load_lock(root)
    files = selected_files(root, lock)
    entries = [validate_file(path, root, lock) for path in files]
    relatives = {entry["path"] for entry in entries}
    paper_inputs = validate_paper_inputs(root, lock, relatives)
    anonymous = lock["policy"].get("anonymous_manuscript", True) is True
    if anonymous:
        validate_anonymous_manuscript(root)
    else:
        validate_camera_ready_manuscript(root)
    return {
        "artifact_name": lock["artifact_name"],
        "files": entries,
        "paper_inputs": paper_inputs,
        "anonymous_manuscript": anonymous,
        "tree_sha256": sha256_bytes(canonical_json_bytes(entries)),
    }


def build_export_manifest(
    root: Path,
    source_report: Mapping[str, Any],
) -> dict[str, Any]:
    lock_hash = sha256_file(root / LOCK_NAME)
    return {
        "schema_version": SUPPORTED_SCHEMA_VERSION,
        "artifact_name": source_report["artifact_name"],
        "artifact_lock_sha256": lock_hash,
        "tree_sha256": source_report["tree_sha256"],
        "files": source_report["files"],
        "paper_inputs": source_report["paper_inputs"],
    }


def _write_manifest(path: Path, manifest: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def export_tree(source_root: Path, output: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    output = output.expanduser().resolve()
    if output == source_root or source_root in output.parents:
        raise ReleaseError("output must not be the source tree or a descendant of it")
    if output.exists():
        raise ReleaseError(f"output already exists; refusing to overwrite: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)

    source_report = check_source(source_root)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=str(output.parent))
    )
    try:
        for entry in source_report["files"]:
            relative = entry["path"]
            source = source_root / relative
            destination = staging / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
            destination.chmod(0o755 if entry["executable"] else 0o644)
        manifest = build_export_manifest(staging, source_report)
        _write_manifest(staging / EXPORT_MANIFEST, manifest)
        check_export(staging)
        staging.rename(output)
        return manifest
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _load_export_manifest(root: Path) -> dict[str, Any]:
    path = root / EXPORT_MANIFEST
    if not path.is_file():
        raise ReleaseError(f"missing export manifest: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ReleaseError(f"cannot parse {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ReleaseError(f"{EXPORT_MANIFEST} must contain an object")
    if value.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise ReleaseError("unsupported export manifest schema")
    if not isinstance(value.get("files"), list):
        raise ReleaseError("export manifest files must be a list")
    return value


def check_export(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest = _load_export_manifest(root)
    source_report = check_source(root)
    expected_entries = manifest["files"]
    if source_report["files"] != expected_entries:
        raise ReleaseError("export file metadata differs from STANDALONE_MANIFEST.json")
    if source_report["tree_sha256"] != manifest.get("tree_sha256"):
        raise ReleaseError("export tree hash mismatch")
    if sha256_file(root / LOCK_NAME) != manifest.get("artifact_lock_sha256"):
        raise ReleaseError("artifact lock hash mismatch")
    if source_report["paper_inputs"] != manifest.get("paper_inputs"):
        raise ReleaseError("paper input manifest mismatch")

    lock = load_lock(root)
    actual = set()
    for path in root.rglob("*"):
        if not path.is_file() and not path.is_symlink():
            continue
        relative = relative_posix(path, root)
        # A reviewer may run ``make reproduce`` before re-checking an exported
        # tree. Runtime products live only under explicitly excluded paths and
        # are not part of the immutable release manifest.
        if relative != EXPORT_MANIFEST and is_excluded(relative, lock):
            continue
        actual.add(relative)
    expected = {entry["path"] for entry in expected_entries} | {EXPORT_MANIFEST}
    extras = sorted(actual - expected)
    missing = sorted(expected - actual)
    if extras or missing:
        raise ReleaseError(f"export file-set mismatch; extra={extras}, missing={missing}")
    return source_report


def _sanitized_environment() -> dict[str, str]:
    # Start from an allowlist so unusual provider-specific credential names are
    # not inherited merely because they omit words such as KEY or TOKEN.
    inherited = (
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "TMPDIR",
        "TMP",
        "TEMP",
        "LANG",
        "LC_ALL",
        "TZ",
    )
    environment = {
        name: os.environ[name]
        for name in inherited
        if name in os.environ
    }
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONHASHSEED"] = "0"
    environment["PYTHONUTF8"] = "1"
    return environment


def fresh_temp_smoke(source_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    with tempfile.TemporaryDirectory(prefix="lint-vs-semantic-smoke-") as temp:
        output = Path(temp) / "release"
        manifest = export_tree(source_root, output)
        environment = _sanitized_environment()
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(output / "code"), str(output / "code" / "benchmarks"))
        )
        smoke_home = Path(temp) / "home"
        smoke_home.mkdir()
        environment["HOME"] = str(smoke_home)
        reproduced = Path(temp) / "reproduced"
        commands = [
            [
                sys.executable,
                "-c",
                (
                    "from pathlib import Path; "
                    "import bench_common, stats; "
                    "root=Path.cwd().resolve(); "
                    "assert Path(bench_common.DEFAULT_TASKS).resolve() == "
                    "root/'data'/'eval_tasks.jsonl'; "
                    "assert len(bench_common.load_tasks()) == 155; "
                    "assert Path(stats.DEFAULT_CLEAN).resolve() == "
                    "root/'data'/'eval_tasks_clean.jsonl'; "
                    "assert len(stats.load_clean_ids()) == 137"
                ),
            ],
            [
                sys.executable,
                "tools/export_standalone.py",
                "--check",
                "--root",
                str(output),
            ],
            [
                sys.executable,
                "tools/run_public_tests.py",
            ],
            [
                sys.executable,
                "tools/reproduce.py",
                "offline",
                "--output",
                str(reproduced),
            ],
        ]
        if sys.version.split()[0] == "3.9.6":
            commands.append(
                [
                    sys.executable,
                    "tools/reproduce.py",
                    "offline",
                    "--strict",
                    "--output",
                    str(Path(temp) / "reproduced-strict"),
                ]
            )
        commands.append(
            [
                sys.executable,
                "tools/export_standalone.py",
                "--check",
                "--root",
                str(output),
            ]
        )
        for command in commands:
            completed = subprocess.run(
                command,
                cwd=output,
                env=environment,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                timeout=180,
                check=False,
            )
            if completed.returncode != 0:
                raise ReleaseError(
                    "fresh-tree smoke command failed\n"
                    f"command: {' '.join(command)}\n"
                    f"output:\n{completed.stdout}"
                )
        return {
            "artifact_name": manifest["artifact_name"],
            "files": len(manifest["files"]),
            "tree_sha256": manifest["tree_sha256"],
            "fresh_temp_smoke": "passed",
        }


def _print_report(report: Mapping[str, Any], action: str) -> None:
    payload = {
        "action": action,
        "artifact_name": report["artifact_name"],
        "files": len(report.get("files", []))
        if isinstance(report.get("files"), list)
        else report.get("files"),
        "tree_sha256": report["tree_sha256"],
    }
    if "paper_inputs" in report:
        payload["paper_inputs"] = len(report["paper_inputs"])
    if "fresh_temp_smoke" in report:
        payload["fresh_temp_smoke"] = report["fresh_temp_smoke"]
    print(json.dumps(payload, indent=2, sort_keys=True))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=default_root(),
        help="source or exported tree root (default: parent of this tools directory)",
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--check",
        action="store_true",
        help="validate the source tree or an existing exported tree",
    )
    mode.add_argument(
        "--output",
        type=Path,
        help="create a new allowlisted export directory; existing paths are refused",
    )
    mode.add_argument(
        "--smoke",
        action="store_true",
        help="export to a fresh temporary directory and run release tests there",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        root = args.root.resolve()
        if args.check:
            if (root / EXPORT_MANIFEST).is_file():
                report = check_export(root)
                action = "check-export"
            else:
                report = check_source(root)
                action = "check-source"
        elif args.output is not None:
            report = export_tree(root, args.output)
            action = "export"
        else:
            report = fresh_temp_smoke(root)
            action = "smoke"
        _print_report(report, action)
        return 0
    except (OSError, ReleaseError, subprocess.SubprocessError) as exc:
        print(f"release check failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
