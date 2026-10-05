#!/usr/bin/env python3
"""Preregistered NVIDIA Responses runner for the VeriCodeGen study.

The command is dry-run by default.  Paid requests require the explicit
``--run`` flag and ``NVIDIA_API_KEY`` in the process environment.  The runner
does not import or use RTLRepair's fallback-capable IDE client.

Examples::

    python benchmarks/vericodegen_eval.py --mode preflight
    NVIDIA_API_KEY=... python benchmarks/vericodegen_eval.py \
        --mode preflight --run
    NVIDIA_API_KEY=... python benchmarks/vericodegen_eval.py \
        --mode main --run
    python benchmarks/vericodegen_eval.py \
        --export-validation-jobs generated/formal/validation_jobs.jsonl
    python -m benchmarks.formal_verify validate-batch \
        --jobs generated/formal/validation_jobs.jsonl \
        --results generated/formal/validation_results.jsonl \
        --artifacts generated/formal/candidates
    python benchmarks/vericodegen_eval.py \
        --ingest-verdicts generated/formal/validation_results.jsonl

Every wire attempt is reserved in an append-only JSONL ledger before it is
sent.  HTTP/gateway failures are retryable transport events and do not enter
the research denominator.  Every HTTP 200 response is terminal for that
preregistered call, including refusals, truncation, malformed output and
later compile failures.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from datetime import datetime, timezone
import difflib
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence


HERE = Path(__file__).resolve().parent
CODE = HERE.parent
for _p in (str(HERE), str(CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
ROOT = Path(os.environ.get("RTLREPAIR_ROOT", "")) if os.environ.get("RTLREPAIR_ROOT") else Path(__file__).resolve().parents[2]

from backend.nvidia_responses import (  # noqa: E402
    API_KEY_ENV,
    DEFAULT_BASE_URL,
    DEFAULT_MAX_OUTPUT_TOKENS,
    DEFAULT_MODEL,
    DEFAULT_REASONING_EFFORT,
    MissingNVIDIAAPIKey,
    NVIDIAResponse,
    NVIDIAResponsesClient,
    NVIDIAResponsesError,
)
from vericodegen_data import (  # noqa: E402
    validate_redaction_decision_manifest,
    verify_manifest as verify_data_manifest,
)


PROTOCOL_VERSION = "vericodegen-2026-v1"
FROZEN_STUDY_ID = "vericodegen-2026-verifier-guided-rtl-repair"
LEDGER_SCHEMA_VERSION = 1
HARD_ATTEMPT_CAP = 800
HARD_SUCCESS_CAP = 623
HARD_RETRY_ATTEMPT_CAP = 92
INDEPENDENT_SIMULATION_SEEDS = tuple(range(1001, 1011))
INDEPENDENT_SIMULATION_CYCLES = 200
EXPECTED_COUNTS = {
    "ping": 1,
    "preflight": 13,
    "main": 340,
    "shift": 100,
}
DEFAULT_CONFIG = ROOT / "configs" / "vericodegen2026.json"
DEFAULT_LEDGER = ROOT / "generated" / "logs" / "vericodegen2026_events.jsonl"
DEFAULT_MAIN_CASES = ROOT / "configs" / "vericodegen" / "main85_inputs.jsonl"
DEFAULT_SHIFT_CASES = ROOT / "configs" / "vericodegen" / "shift50_inputs.jsonl"
DEFAULT_CANONICAL_MANIFEST = ROOT / "data" / "formal" / "sem85_manifest.json"
DEFAULT_PROTOCOL_MANIFEST = ROOT / "data" / "formal" / "protocol_manifest.json"
DEFAULT_BENCHMARK_SOURCE = ROOT / "data" / "repairbench_sem85.jsonl"
DEFAULT_TOOLCHAIN_LOCK = ROOT / "docker" / "formal" / "toolchain.lock.json"
FORMAL_VERIFY_SOURCE = HERE / "formal_verify.py"
FORMAL_PROTOCOL_SOURCE = HERE / "formal_protocol.py"
DEFAULT_SHIFT_MANIFEST = ROOT / "configs" / "vericodegen" / "shift50_manifest.json"
DEFAULT_MAIN_INPUT_MANIFEST = ROOT / "configs" / "vericodegen" / "main85_inputs.manifest.json"
DEFAULT_REDACTION_TEMPLATE = (
    ROOT / "configs" / "vericodegen" / "redaction_annotations.template.json"
)
DEFAULT_REDACTION_MANIFEST = ROOT / "configs" / "vericodegen" / "redaction_manifest.json"
DEFAULT_PROTOCOL_AMENDMENT = (
    ROOT / "configs" / "vericodegen" / "protocol_amendment_2026-07-15.json"
)
DEFAULT_CALIBRATION_GATE = ROOT / "generated" / "formal" / "calibration_gate.json"


REPAIR_SYSTEM = """You are an expert RTL debugging engineer. Repair the supplied
compile-clean SystemVerilog module. Return the complete replacement module,
from `module` through `endmodule`; never return a patch, explanation, testbench,
or placeholder. Preserve the given module interface and emit synthesizable RTL."""

PING_SYSTEM = """This is a transport preflight. Follow the user's exact short
response instruction. Do not add explanation."""

_FENCE_RE = re.compile(
    r"```(?:systemverilog|verilog|sv|v)?[ \t]*\r?\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_MODULE_RE = re.compile(r"\bmodule\b.*?\bendmodule\b", re.DOTALL)
_MODULE_NAME_RE = re.compile(r"(\bmodule\s+)([A-Za-z_][A-Za-z0-9_$]*)")


class ProtocolError(RuntimeError):
    """The requested run would violate the frozen experiment protocol."""


class BudgetExhausted(ProtocolError):
    """A paid request would exceed a hard preregistered cap."""


@dataclass(frozen=True)
class ExperimentCase:
    case_id: str
    cluster_id: str
    broken_rtl: str
    full_spec: Optional[str] = None
    redacted_spec: Optional[str] = None
    redaction_label: Optional[str] = None
    location_line_number: Optional[int] = None
    location_line: Optional[str] = None
    previous_candidate: Optional[str] = None
    counterexample: Any = None
    mutation_family: Optional[str] = None
    # Validation-only identifiers never enter prompts or ledger events.
    source_case_id: Optional[str] = None
    validation_seed_id: Optional[str] = None
    validation_source_record_sha256: Optional[str] = None


@dataclass(frozen=True)
class ExperimentJob:
    study_id: str
    call_id: str
    mode: str
    arm: str
    case_id: str
    cluster_id: str
    prompt: str
    system: str
    nonresearch: bool = False
    parent_call_id: Optional[str] = None
    mutation_family: Optional[str] = None


@dataclass(frozen=True)
class RunSummary:
    scheduled: int
    skipped_completed: int
    http_200_recorded: int
    transport_errors: int
    total_attempts_after: int
    total_http_200_after: int
    total_retry_attempts_after: int


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _digest(*parts: str, length: int = 24) -> str:
    raw = "\0".join(parts).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:length]


def anonymous_id(kind: str, source_id: str) -> str:
    """Stable pseudonym; source benchmark names never enter prompts/ledgers."""

    return f"{kind}-{_digest(PROTOCOL_VERSION, kind, source_id, length=16)}"


def _normalized_source_id(value: str) -> str:
    key = str(value).split(":")[-1]
    return key[:-4] if key.endswith("_ref") else key


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ProtocolError(f"invalid JSONL at {path}:{line_number}") from exc
            if not isinstance(row, dict):
                raise ProtocolError(f"JSONL row is not an object at {path}:{line_number}")
            rows.append(row)
    return rows


def _load_spec_map(path: Optional[Path]) -> dict[str, str]:
    if path is None:
        return {}
    specs: dict[str, str] = {}
    for row in _read_jsonl(path):
        source_id = str(row.get("seed_id") or row.get("id") or "")
        spec = str(row.get("full_spec") or row.get("spec") or row.get("instruction") or "").strip()
        if source_id and spec:
            specs[_normalized_source_id(source_id)] = spec
    return specs


def _load_annotations(path: Optional[Path]) -> dict[str, dict[str, Any]]:
    if path is None:
        return {}
    annotations: dict[str, dict[str, Any]] = {}
    for row in _read_jsonl(path):
        source_id = str(row.get("source_id") or row.get("id") or row.get("seed_id") or "")
        if not source_id:
            raise ProtocolError(f"annotation row in {path} has no case identifier")
        annotations[_normalized_source_id(source_id)] = row
    return annotations


def _read_json_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ProtocolError(f"invalid JSON object: {path}") from exc
    if not isinstance(value, dict):
        raise ProtocolError(f"JSON document is not an object: {path}")
    return value


def _load_document_overlay(path: Optional[Path]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Load a frozen JSON manifest whose case list overlays source records."""

    if path is None:
        return {}, {}
    document = _read_json_object(path)
    rows = document.get("cases")
    if not isinstance(rows, list):
        raise ProtocolError(f"manifest has no case list: {path}")
    overlay: dict[str, dict[str, Any]] = {}
    for row in rows:
        if not isinstance(row, dict):
            raise ProtocolError(f"manifest case is not an object: {path}")
        source_id = str(
            row.get("source_id")
            or row.get("record_id")
            or row.get("case_id")
            or row.get("seed_id")
            or ""
        )
        if not source_id:
            raise ProtocolError(f"manifest case has no identifier: {path}")
        normalized = _normalized_source_id(source_id)
        mapped = dict(row)
        if mapped.get("intent_label") and not mapped.get("redaction_label"):
            mapped["redaction_label"] = mapped["intent_label"]
        overlay[normalized] = mapped
    return overlay, document


def _derive_location(broken: str, canonical: str) -> tuple[int, str]:
    """Find the sole broken-side line in a canonical single-edit mutation."""

    broken_lines = broken.strip().splitlines()
    canonical_lines = canonical.strip().splitlines()
    matcher = difflib.SequenceMatcher(a=canonical_lines, b=broken_lines, autojunk=False)
    candidates: list[tuple[int, str]] = []
    for tag, _a0, _a1, b0, b1 in matcher.get_opcodes():
        if tag in {"replace", "insert"}:
            candidates.extend((index + 1, broken_lines[index]) for index in range(b0, b1))
    if len(candidates) != 1:
        raise ProtocolError(
            f"oracle localization expected one broken source line, found {len(candidates)}"
        )
    return candidates[0]


def _load_canonical_locations(path: Optional[Path]) -> dict[str, tuple[Path, str]]:
    if path is None:
        return {}
    manifest = _read_json_object(path)
    rows = manifest.get("cases")
    if not isinstance(rows, list):
        raise ProtocolError(f"canonical manifest has no cases: {path}")
    locations: dict[str, tuple[Path, str]] = {}
    for row in rows:
        if not isinstance(row, Mapping):
            raise ProtocolError(f"canonical manifest case is invalid: {path}")
        source_id = str(row.get("case_id") or row.get("seed_id") or "")
        canonical_path = row.get("canonical_path")
        broken_sha = str(row.get("broken_sha256") or "")
        if not source_id or not canonical_path or not broken_sha:
            raise ProtocolError(f"canonical manifest case is incomplete: {path}")
        resolved = Path(str(canonical_path))
        if not resolved.is_absolute():
            resolved = ROOT / resolved
        locations[_normalized_source_id(source_id)] = (resolved, broken_sha)
    return locations


def _broken_from_messages(row: Mapping[str, Any]) -> str:
    messages = row.get("messages")
    if not isinstance(messages, list):
        return ""
    user_text = "\n".join(
        str(item.get("content") or "")
        for item in messages
        if isinstance(item, Mapping) and item.get("role") == "user"
    )
    fences = _FENCE_RE.findall(user_text)
    if fences:
        return str(fences[0])
    modules = _MODULE_RE.findall(user_text)
    return str(modules[0]) if modules else ""


def anonymize_module(source: str) -> tuple[str, Optional[str]]:
    """Rename the first module and all exact references to ``TopModule``."""

    match = _MODULE_NAME_RE.search(source)
    if not match:
        return source, None
    original = match.group(2)
    anonymized = re.sub(r"\b" + re.escape(original) + r"\b", "TopModule", source)
    return anonymized, original


def _location_fields(row: Mapping[str, Any]) -> tuple[Optional[int], Optional[str]]:
    location = row.get("location") or row.get("oracle_location")
    number: Any = row.get("location_line_number") or row.get("line_number")
    line: Any = row.get("location_line") or row.get("line_text") or row.get("buggy_line")
    if isinstance(location, Mapping):
        number = location.get("line_number") or location.get("line") or number
        line = (
            location.get("broken_line")
            or location.get("line_text")
            or location.get("text")
            or line
        )
    parsed_number: Optional[int] = None
    if number is not None:
        try:
            parsed_number = int(number)
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"invalid location line number: {number!r}") from exc
    return parsed_number, (str(line).strip() if line is not None else None)


def _manifest_sha(document: Mapping[str, Any]) -> str:
    without_sha = dict(document)
    without_sha.pop("manifest_sha256", None)
    return hashlib.sha256(_canonical_json(without_sha).encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _gate_manifest_sha(document: Mapping[str, Any]) -> str:
    unsigned = dict(document)
    unsigned.pop("gate_manifest_sha256", None)
    return hashlib.sha256(_canonical_json(unsigned).encode("utf-8")).hexdigest()


def _recorded_path(value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ProtocolError(f"calibration gate lacks {field}")
    path = Path(value)
    # Formal commands run with the repository mounted at /work.  Accept a gate
    # produced just before the schema-2 relative-path migration without making
    # host-specific absolute paths part of the trust contract.
    if path.is_absolute() and len(path.parts) >= 2 and path.parts[:2] == ("/", "work"):
        return ROOT.joinpath(*path.parts[2:])
    return path if path.is_absolute() else ROOT / path


def _current_formal_protocol_sha256(container_image_id: str) -> str:
    """Reproduce the formal worker's code/lock/image protocol fingerprint."""

    source = FORMAL_VERIFY_SOURCE.read_text(encoding="utf-8")
    match = re.search(
        r'^FORMAL_PROTOCOL_VERSION\s*=\s*["\']([^"\']+)["\']',
        source,
        re.MULTILINE,
    )
    if not match:
        raise ProtocolError("cannot read FORMAL_PROTOCOL_VERSION")
    digest = hashlib.sha256()
    digest.update(match.group(1).encode("utf-8"))
    for path in (
        FORMAL_VERIFY_SOURCE,
        FORMAL_PROTOCOL_SOURCE,
        DEFAULT_TOOLCHAIN_LOCK,
        DEFAULT_PROTOCOL_AMENDMENT,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(container_image_id.encode("utf-8"))
    return digest.hexdigest()


def _validate_materialized_inputs(
    cases_path: Path,
    rows: Sequence[Mapping[str, Any]],
    manifest_path: Path,
) -> None:
    manifest = _read_json_object(manifest_path)
    if manifest.get("manifest_sha256") != _manifest_sha(manifest):
        raise ProtocolError(f"materialized input manifest SHA is invalid: {manifest_path}")
    if manifest.get("case_count") != len(rows):
        raise ProtocolError("materialized input case count drift")
    if manifest.get("contains_golden_rtl") is not False:
        raise ProtocolError("runner input manifest must explicitly exclude golden RTL")
    jsonl = manifest.get("jsonl") or {}
    if not isinstance(jsonl, Mapping) or jsonl.get("sha256") != hashlib.sha256(
        cases_path.read_bytes()
    ).hexdigest():
        raise ProtocolError("materialized input JSONL SHA drift")
    sources = manifest.get("sources") or {}
    if not isinstance(sources, Mapping):
        raise ProtocolError("materialized input source manifest is invalid")
    for source in sources.values():
        if not isinstance(source, Mapping) or not source.get("path") or not source.get("sha256"):
            raise ProtocolError("materialized input source entry is incomplete")
        source_path = Path(str(source["path"]))
        if not source_path.is_absolute():
            source_path = ROOT / source_path
        if not source_path.exists():
            raise ProtocolError(f"materialized input source is missing: {source_path}")
        if hashlib.sha256(source_path.read_bytes()).hexdigest() != source["sha256"]:
            raise ProtocolError(f"materialized input source SHA drift: {source_path}")
    index = manifest.get("case_index")
    if not isinstance(index, list) or len(index) != len(rows):
        raise ProtocolError("materialized case index is missing or incomplete")
    expected_index = {
        (str(item.get("case_id")), str(item.get("anonymous_id"))): str(
            item.get("materialized_case_sha256")
        )
        for item in index
        if isinstance(item, Mapping)
    }
    for raw in rows:
        metadata = raw.get("internal_metadata")
        if not isinstance(metadata, Mapping):
            raise ProtocolError("materialized input lacks internal_metadata")
        row_without_sha = dict(raw)
        declared_row_sha = str(row_without_sha.pop("materialized_case_sha256", ""))
        actual_row_sha = hashlib.sha256(
            _canonical_json(row_without_sha).encode("utf-8")
        ).hexdigest()
        key = (str(metadata.get("case_id")), str(metadata.get("anonymous_id")))
        if not declared_row_sha or declared_row_sha != actual_row_sha:
            raise ProtocolError(f"materialized case SHA drift: {key[1]}")
        if expected_index.get(key) != declared_row_sha:
            raise ProtocolError(f"materialized case index drift: {key[1]}")


def _flatten_materialized_row(row: Mapping[str, Any]) -> dict[str, Any]:
    metadata = row.get("internal_metadata")
    model_input = row.get("model_input")
    if not isinstance(metadata, Mapping) or not isinstance(model_input, Mapping):
        return dict(row)
    if model_input.get("expected_module_name") != "TopModule":
        raise ProtocolError("materialized model input must use TopModule")
    validation_seed = metadata.get("source_seed_id") or metadata.get("source_task_id")
    if validation_seed and not str(validation_seed).endswith("_ref"):
        validation_seed = str(validation_seed) + "_ref"
    return {
        "source_id": metadata.get("case_id"),
        "source_case_id": metadata.get("case_id"),
        "anonymous_case_id": metadata.get("anonymous_id"),
        "cluster_id": metadata.get("cluster_id"),
        "broken_rtl": model_input.get("broken_rtl"),
        "full_spec": model_input.get("full_spec"),
        "oracle_location": model_input.get("oracle_location"),
        "mutation": metadata.get("mutation_family"),
        "validation_seed_id": validation_seed,
        "validation_source_record_sha256": metadata.get("source_record_sha256"),
    }


def load_cases(
    cases_path: Path,
    *,
    specs_path: Optional[Path] = None,
    annotations_path: Optional[Path] = None,
    overlay_manifest_path: Optional[Path] = None,
    canonical_manifest_path: Optional[Path] = None,
    selected_source_ids: Optional[set[str]] = None,
    materialized_manifest_path: Optional[Path] = None,
    require_materialized: bool = False,
) -> list[ExperimentCase]:
    """Load a frozen case manifest or adapt legacy RepairBench JSONL in memory."""

    raw_rows = _read_jsonl(cases_path)
    materialized = bool(raw_rows and "model_input" in raw_rows[0])
    if require_materialized and not materialized:
        raise ProtocolError(
            "paid study inputs must use the frozen {internal_metadata, model_input} materialization"
        )
    if materialized:
        if materialized_manifest_path is None:
            if cases_path.name.endswith("_inputs.jsonl"):
                materialized_manifest_path = cases_path.with_name(
                    cases_path.name.replace("_inputs.jsonl", "_inputs.manifest.json")
                )
        if materialized_manifest_path is None or not materialized_manifest_path.exists():
            raise ProtocolError("materialized runner inputs require their SHA/source manifest")
        _validate_materialized_inputs(cases_path, raw_rows, materialized_manifest_path)
    rows = [_flatten_materialized_row(row) for row in raw_rows]
    specs = _load_spec_map(specs_path)
    annotations = _load_annotations(annotations_path)
    document_overlay, _ = _load_document_overlay(overlay_manifest_path)
    canonical_locations = _load_canonical_locations(canonical_manifest_path)
    cases: list[ExperimentCase] = []
    seen: set[str] = set()
    for row in rows:
        # ``id`` is mutation-case identity; ``seed_id`` is the clustered
        # design identity and is intentionally not unique across sem85.
        source_id = str(row.get("source_id") or row.get("id") or row.get("seed_id") or "")
        if not source_id:
            raise ProtocolError("case row has no stable identifier")
        if selected_source_ids is not None and source_id not in selected_source_ids:
            continue
        normalized = _normalized_source_id(source_id)
        merged = dict(row)
        merged.update(annotations.get(normalized, {}))
        # Frozen document fields take precedence over legacy annotations.
        merged.update(document_overlay.get(normalized, {}))
        broken_raw = str(merged.get("broken_rtl") or "") or _broken_from_messages(merged)
        if not broken_raw.strip():
            raise ProtocolError(f"case {anonymous_id('case', source_id)} has no broken RTL")
        broken = broken_raw.strip()
        anonymized, original_module = anonymize_module(broken)
        case_id = str(merged.get("anonymous_case_id") or anonymous_id("case", source_id))
        cluster_source = str(
            merged.get("cluster_id") or merged.get("seed_id") or merged.get("task") or source_id
        )
        cluster_id = str(
            merged.get("anonymous_cluster_id") or anonymous_id("cluster", cluster_source)
        )
        if case_id in seen:
            raise ProtocolError(f"duplicate case id after anonymization: {case_id}")
        seen.add(case_id)

        full_spec = str(
            merged.get("full_spec")
            or merged.get("spec")
            or specs.get(normalized)
            or specs.get(_normalized_source_id(str(row.get("seed_id") or "")))
            or ""
        ).strip()
        redacted_spec_raw = str(merged.get("redacted_spec") or "")
        redacted_spec = redacted_spec_raw.strip()
        redaction_label = str(
            merged.get("redaction_label") or merged.get("spec_evidence") or ""
        ).strip().lower()
        expected_full_spec_sha = merged.get("full_spec_sha256")
        if expected_full_spec_sha and hashlib.sha256(
            _canonical_json(full_spec).encode("utf-8")
        ).hexdigest() != expected_full_spec_sha:
            raise ProtocolError(f"{case_id} full spec drifted after redaction freeze")
        expected_redacted_sha = merged.get("redacted_spec_sha256")
        if expected_redacted_sha and hashlib.sha256(
            _canonical_json(redacted_spec_raw).encode("utf-8")
        ).hexdigest() != expected_redacted_sha:
            raise ProtocolError(f"{case_id} redacted spec SHA mismatch")
        number, line = _location_fields(merged)
        if (number is None or not line) and canonical_locations:
            # Prefer a case-id entry, then the seed-id entry.  The formal
            # manifest pins the broken SHA and canonical source path.
            entry = canonical_locations.get(_normalized_source_id(str(row.get("id") or "")))
            entry = entry or canonical_locations.get(
                _normalized_source_id(str(row.get("seed_id") or ""))
            )
            if entry is None:
                raise ProtocolError(f"{case_id} is missing from canonical manifest")
            canonical_path, expected_broken_sha = entry
            actual_broken_sha = hashlib.sha256(broken_raw.encode("utf-8")).hexdigest()
            if actual_broken_sha != expected_broken_sha:
                # Legacy fenced text is whitespace-trimmed; formal manifests
                # hash the same stripped source in current schema.  Fail closed
                # on any other drift instead of localizing against stale RTL.
                raise ProtocolError(f"{case_id} broken RTL SHA does not match canonical manifest")
            canonical = canonical_path.read_text(encoding="utf-8")
            number, line = _derive_location(broken, canonical)
        if original_module and line:
            line = re.sub(r"\b" + re.escape(original_module) + r"\b", "TopModule", line)
        cases.append(
            ExperimentCase(
                case_id=case_id,
                cluster_id=cluster_id,
                broken_rtl=anonymized,
                full_spec=(full_spec or None),
                redacted_spec=(redacted_spec or None),
                redaction_label=(redaction_label or None),
                location_line_number=number,
                location_line=line,
                previous_candidate=(str(merged.get("previous_candidate") or "").strip() or None),
                counterexample=merged.get("counterexample") or merged.get("witness"),
                mutation_family=(str(row.get("mutation") or "").strip() or None),
                source_case_id=(str(merged.get("source_case_id") or source_id).strip() or None),
                validation_seed_id=(
                    str(
                        merged.get("validation_seed_id")
                        or row.get("seed_id")
                        or ""
                    ).strip()
                    or None
                ),
                validation_source_record_sha256=(
                    str(merged.get("validation_source_record_sha256") or "").strip()
                    or None
                ),
            )
        )
    return cases


def _repair_prompt(
    case: ExperimentCase,
    *,
    specification: Optional[str],
    include_location: bool,
) -> str:
    parts = [
        "Repair the following compile-clean SystemVerilog module, which fails functional verification.",
        "\nBuggy module:\n```systemverilog\n" + case.broken_rtl.strip() + "\n```",
    ]
    if specification:
        parts.append("\nBehavioral specification:\n" + specification.strip())
    if include_location:
        if case.location_line_number is None or not case.location_line:
            raise ProtocolError(f"{case.case_id} is missing frozen oracle location")
        parts.append(
            "\nOracle localization (this identifies the buggy source line but does not give the fix):\n"
            f"Line {case.location_line_number}: `{case.location_line}`"
        )
    parts.append(
        "\nReturn only the complete corrected `TopModule`, including its unchanged interface."
    )
    return "\n".join(parts)


def _format_witness(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2)


def _feedback_prompt(case: ExperimentCase, concrete: bool) -> str:
    if not case.previous_candidate:
        raise ProtocolError(f"feedback case {case.case_id} has no previous candidate")
    parts = [
        "Revise a previous repair attempt for this original problem.",
        "\nOriginal buggy module:\n```systemverilog\n" + case.broken_rtl.strip() + "\n```",
        "\nPrevious candidate:\n```systemverilog\n"
        + case.previous_candidate.strip()
        + "\n```",
    ]
    if concrete:
        witness = _format_witness(case.counterexample)
        if not witness:
            raise ProtocolError(f"feedback case {case.case_id} has no concrete witness")
        parts.append(
            "\nThe verifier produced this concrete trace from initialization through the first divergence. "
            "Values include the driven inputs, reference expected outputs, and candidate outputs:\n"
            "```text\n" + witness + "\n```"
        )
    else:
        parts.append("\nThe previous candidate still did not pass verification.")
    parts.append("\nReturn only the complete corrected `TopModule`.")
    return "\n".join(parts)


def _request_fingerprint(prompt: str, system: str) -> str:
    frozen_request = {
        "base_url": DEFAULT_BASE_URL,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
        "model": DEFAULT_MODEL,
        "prompt": prompt,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "system": system,
        "wire_api": "responses",
    }
    return hashlib.sha256(_canonical_json(frozen_request).encode("utf-8")).hexdigest()


def _call_id(
    study_id: str,
    mode: str,
    case_id: str,
    arm: str,
    *,
    prompt: str,
    system: str,
) -> str:
    # Including the full request fingerprint prevents a prompt/code change
    # from silently reusing an old result.  A mode-level schedule lock below
    # additionally refuses mixed old/new schedules on resume.
    return "call-" + _digest(
        PROTOCOL_VERSION,
        study_id,
        mode,
        case_id,
        arm,
        _request_fingerprint(prompt, system),
    )


def _job(
    *,
    study_id: str,
    mode: str,
    arm: str,
    case: ExperimentCase,
    prompt: str,
    nonresearch: bool = False,
    parent_call_id: Optional[str] = None,
) -> ExperimentJob:
    return ExperimentJob(
        study_id=study_id,
        call_id=_call_id(
            study_id,
            mode,
            case.case_id,
            arm,
            prompt=prompt,
            system=REPAIR_SYSTEM,
        ),
        mode=mode,
        arm=arm,
        case_id=case.case_id,
        cluster_id=case.cluster_id,
        prompt=prompt,
        system=REPAIR_SYSTEM,
        nonresearch=nonresearch,
        parent_call_id=parent_call_id,
        mutation_family=case.mutation_family,
    )


def _fixture_cases() -> list[ExperimentCase]:
    return [
        ExperimentCase(
            case_id="fixture-comb",
            cluster_id="fixture-comb",
            broken_rtl=(
                "module TopModule(input logic a, output logic y);\n"
                "  assign y = a;\nendmodule"
            ),
            full_spec="Output y is the logical inverse of input a.",
            location_line_number=2,
            location_line="assign y = a;",
        ),
        ExperimentCase(
            case_id="fixture-counter",
            cluster_id="fixture-counter",
            broken_rtl=(
                "module TopModule(input logic clk, input logic rst, output logic [3:0] q);\n"
                "  always_ff @(posedge clk) begin\n"
                "    if (rst) q <= 4'd0; else q <= q;\n"
                "  end\nendmodule"
            ),
            full_spec="On a rising clock edge, reset q to zero when rst is high; otherwise increment q by one.",
            location_line_number=3,
            location_line="if (rst) q <= 4'd0; else q <= q;",
        ),
        ExperimentCase(
            case_id="fixture-mux",
            cluster_id="fixture-mux",
            broken_rtl=(
                "module TopModule(input logic a, b, sel, output logic y);\n"
                "  assign y = sel ? a : b;\nendmodule"
            ),
            full_spec="Select b when sel is one and a when sel is zero.",
            location_line_number=2,
            location_line="assign y = sel ? a : b;",
        ),
    ]


def _ping_job(study_id: str, mode: str = "ping") -> ExperimentJob:
    prompt = "Reply with exactly: PONG"
    return ExperimentJob(
        study_id=study_id,
        # Standalone ping and the ping embedded in preflight intentionally use
        # one identity.  Running both therefore still spends at most 13 total
        # successful preflight calls, not 14.
        call_id=_call_id(
            study_id,
            "preflight",
            "fixture-ping",
            "ping",
            prompt=prompt,
            system=PING_SYSTEM,
        ),
        mode=mode,
        arm="ping",
        case_id="fixture-ping",
        cluster_id="fixture-ping",
        prompt=prompt,
        system=PING_SYSTEM,
        nonresearch=True,
    )


def hash_interleave(jobs: Iterable[ExperimentJob]) -> list[ExperimentJob]:
    """Deterministic protocol order independent of input-file row order."""

    return sorted(
        jobs,
        key=lambda job: hashlib.sha256(
            ("vericodegen-order-v1\0" + job.call_id).encode("utf-8")
        ).digest(),
    )


def _validate_main_cases(cases: Sequence[ExperimentCase]) -> None:
    for case in cases:
        if not case.full_spec:
            raise ProtocolError(f"{case.case_id} has no full specification")
        if case.location_line_number is None or not case.location_line:
            raise ProtocolError(f"{case.case_id} has no frozen oracle location")


def _validate_redactions(cases: Sequence[ExperimentCase]) -> None:
    allowed = {"explicit", "derivable", "absent"}
    invalid = [case.case_id for case in cases if case.redaction_label not in allowed]
    missing = [case.case_id for case in cases if not case.redacted_spec]
    if invalid:
        raise ProtocolError(
            f"redacted arm is not frozen: {len(invalid)} cases lack an explicit/derivable/absent label"
        )
    # An empty string cannot distinguish deliberate deletion from missing
    # annotation.  The frozen manifest should use an explicit neutral sentence
    # when removal empties the original description.
    if missing:
        raise ProtocolError(f"redacted arm is not frozen: {len(missing)} specs are missing")


def build_schedule(
    mode: str,
    *,
    study_id: str = PROTOCOL_VERSION,
    cases: Sequence[ExperimentCase] = (),
    ledger: Optional["ExperimentLedger"] = None,
    strict_counts: bool = True,
    redacted_arm_dropped: bool = False,
) -> list[ExperimentJob]:
    """Build one frozen experiment phase and hash-interleave its arm order."""

    if mode == "ping":
        return [_ping_job(study_id)]
    if mode == "preflight":
        fixture_jobs: list[ExperimentJob] = []
        for case in _fixture_cases():
            for has_spec, has_location in ((False, False), (True, False), (False, True), (True, True)):
                arm = f"spec{int(has_spec)}_loc{int(has_location)}"
                fixture_jobs.append(
                    _job(
                        study_id=study_id,
                        mode=mode,
                        arm=arm,
                        case=case,
                        prompt=_repair_prompt(
                            case,
                            specification=(case.full_spec if has_spec else None),
                            include_location=has_location,
                        ),
                        nonresearch=True,
                    )
                )
        # The transport ping is deliberately first; the remaining shapes use
        # fixed hash-interleaving.
        return [_ping_job(study_id, mode=mode)] + hash_interleave(fixture_jobs)

    if mode == "main":
        if strict_counts and len(cases) != 85:
            raise ProtocolError(f"main requires exactly 85 frozen cases, got {len(cases)}")
        _validate_main_cases(cases)
        jobs = []
        for case in cases:
            for has_spec, has_location in ((False, False), (True, False), (False, True), (True, True)):
                arm = f"spec{int(has_spec)}_loc{int(has_location)}"
                jobs.append(
                    _job(
                        study_id=study_id,
                        mode=mode,
                        arm=arm,
                        case=case,
                        prompt=_repair_prompt(
                            case,
                            specification=(case.full_spec if has_spec else None),
                            include_location=has_location,
                        ),
                    )
                )
        return hash_interleave(jobs)

    if mode == "redacted":
        if redacted_arm_dropped:
            if cases:
                raise ProtocolError("dropped redacted arm cannot contain scheduled cases")
            return []
        if not cases:
            raise ProtocolError("redacted arm has no frozen eligible cases")
        _validate_redactions(cases)
        return hash_interleave(
            _job(
                study_id=study_id,
                mode=mode,
                arm="redacted_spec_loc0",
                case=case,
                prompt=_repair_prompt(
                    case, specification=case.redacted_spec, include_location=False
                ),
            )
            for case in cases
        )

    if mode == "shift":
        if strict_counts and len(cases) != 50:
            raise ProtocolError(f"shift requires exactly 50 preregistered cases, got {len(cases)}")
        missing = [case.case_id for case in cases if not case.full_spec]
        if missing:
            raise ProtocolError(f"shift manifest has {len(missing)} cases without a full spec")
        return hash_interleave(
            _job(
                study_id=study_id,
                mode=mode,
                arm=f"spec{int(has_spec)}_loc0",
                case=case,
                prompt=_repair_prompt(
                    case,
                    specification=(case.full_spec if has_spec else None),
                    include_location=False,
                ),
            )
            for case in cases
            for has_spec in (False, True)
        )

    if mode == "feedback":
        if ledger is None:
            raise ProtocolError("feedback scheduling requires the first-pass ledger")
        if strict_counts:
            baseline = [
                response
                for response in ledger.latest_responses().values()
                if response.get("mode") == "main"
                and response.get("arm") == "spec0_loc0"
            ]
            baseline_case_ids = {str(row.get("case_id")) for row in baseline}
            expected_case_ids = {case.case_id for case in cases}
            if len(baseline) != 85 or baseline_case_ids != expected_case_ids:
                raise ProtocolError(
                    "feedback cohort cannot freeze until all 85 no-spec/no-location calls complete"
                )
            verdicts = ledger.latest_verdicts()
            unresolved = [
                str(row.get("call_id"))
                for row in baseline
                if row.get("model_outcome") == "accepted_pending_compile"
                and str(row.get("call_id")) not in verdicts
            ]
            if unresolved:
                raise ProtocolError(
                    "feedback cohort cannot freeze until every parsed baseline candidate is validated"
                )
        cohort = feedback_cohort(cases, ledger)
        if len(cohort) > 85:
            raise ProtocolError(f"feedback cohort F must be <=85, got {len(cohort)}")
        jobs = []
        for case, parent_call_id in cohort:
            for concrete in (False, True):
                arm = "concrete_counterexample" if concrete else "generic_failure"
                jobs.append(
                    _job(
                        study_id=study_id,
                        mode=mode,
                        arm=arm,
                        case=case,
                        prompt=_feedback_prompt(case, concrete),
                        parent_call_id=parent_call_id,
                    )
                )
        return hash_interleave(jobs)

    raise ProtocolError(f"unknown experiment mode: {mode}")


class ExperimentLedger:
    """Append-only event ledger with fsync after every state transition."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._events_cache: Optional[list[dict[str, Any]]] = None
        self._cache_size: Optional[int] = None

    def events(self) -> list[dict[str, Any]]:
        if not self.path.exists():
            self._events_cache = []
            self._cache_size = 0
            return []
        stat_size = self.path.stat().st_size
        if self._events_cache is not None and self._cache_size == stat_size:
            return list(self._events_cache)
        raw_bytes = self.path.read_bytes()
        if raw_bytes and not raw_bytes.endswith(b"\n"):
            # A killed write can leave bytes that never formed a committed
            # JSONL event.  Since request_attempt is fsynced before network I/O,
            # discarding only this unterminated tail cannot erase a sent wire
            # attempt.  A response-tail retry uses the same idempotency key.
            complete_end = raw_bytes.rfind(b"\n") + 1
            fd = os.open(str(self.path), os.O_RDWR)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX)
                if os.fstat(fd).st_size == len(raw_bytes):
                    os.ftruncate(fd, complete_end)
                    os.fsync(fd)
                    raw_bytes = raw_bytes[:complete_end]
            finally:
                try:
                    fcntl.flock(fd, fcntl.LOCK_UN)
                finally:
                    os.close(fd)
        current_size = len(raw_bytes)
        events: list[dict[str, Any]] = []
        lines = raw_bytes.decode("utf-8").splitlines()
        for index, line in enumerate(lines):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                raise ProtocolError(f"corrupt ledger event at line {index + 1}")
            if not isinstance(event, dict):
                raise ProtocolError(f"non-object ledger event at line {index + 1}")
            events.append(event)
        self._events_cache = events
        self._cache_size = current_size
        return list(events)

    def append(self, event: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "recorded_at": _utc_now(),
            **dict(event),
        }
        encoded = (_canonical_json(record) + "\n").encode("utf-8")
        fd = os.open(str(self.path), os.O_CREAT | os.O_WRONLY | os.O_APPEND, 0o600)
        prior_cache_size = self._cache_size
        try:
            fcntl.flock(fd, fcntl.LOCK_EX)
            view = memoryview(encoded)
            while view:
                written = os.write(fd, view)
                if written <= 0:
                    raise OSError("short write to experiment ledger")
                view = view[written:]
            os.fsync(fd)
            new_size = os.fstat(fd).st_size
            if (
                self._events_cache is not None
                and prior_cache_size is not None
                and prior_cache_size + len(encoded) == new_size
            ):
                self._events_cache.append(record)
                self._cache_size = new_size
            else:
                self._events_cache = None
                self._cache_size = None
        finally:
            try:
                fcntl.flock(fd, fcntl.LOCK_UN)
            finally:
                os.close(fd)

    def ensure_schedule(
        self,
        jobs: Sequence[ExperimentJob],
        *,
        config_fingerprint: str,
        protocol_amendment_sha256: Optional[str] = None,
    ) -> Optional[str]:
        """Write or verify the immutable mode schedule before paid calls."""

        if not jobs:
            return None
        study_ids = {job.study_id for job in jobs}
        modes = {job.mode for job in jobs}
        if len(study_ids) != 1 or len(modes) != 1:
            raise ProtocolError("a schedule lock may contain only one study and one mode")
        study_id = next(iter(study_ids))
        mode = next(iter(modes))
        amendment_sha = protocol_amendment_sha256 or _sha256_path(
            DEFAULT_PROTOCOL_AMENDMENT
        )
        descriptor = {
            "protocol_version": PROTOCOL_VERSION,
            "protocol_amendment_sha256": amendment_sha,
            "study_id": study_id,
            "mode": mode,
            "config_fingerprint": config_fingerprint,
            "jobs": [
                {
                    "call_id": job.call_id,
                    "arm": job.arm,
                    "case_id": job.case_id,
                    "cluster_id": job.cluster_id,
                    "nonresearch": job.nonresearch,
                    "parent_call_id": job.parent_call_id,
                    "prompt_sha256": hashlib.sha256(job.prompt.encode("utf-8")).hexdigest(),
                    "system_sha256": hashlib.sha256(job.system.encode("utf-8")).hexdigest(),
                    "request_fingerprint": _request_fingerprint(job.prompt, job.system),
                }
                for job in jobs
            ],
        }
        schedule_sha = hashlib.sha256(
            _canonical_json(descriptor).encode("utf-8")
        ).hexdigest()
        existing = [
            event
            for event in self.events()
            if event.get("event") == "schedule_lock"
            and event.get("study_id") == study_id
            and event.get("mode") == mode
        ]
        if existing:
            if any(event.get("schedule_sha256") != schedule_sha for event in existing):
                raise ProtocolError(
                    f"schedule/config drift for {study_id}/{mode}; use the frozen code and config"
                )
            return schedule_sha
        self.append(
            {
                "event": "schedule_lock",
                **descriptor,
                "schedule_sha256": schedule_sha,
            }
        )
        return schedule_sha

    def ensure_protocol_amendment_lock(self, protocol_amendment_sha256: str) -> None:
        """Bind an empty/pre-output ledger to exactly one checked-in amendment."""

        if not re.fullmatch(r"[0-9a-f]{64}", protocol_amendment_sha256):
            raise ProtocolError("protocol amendment SHA is invalid")
        events = self.events()
        locks = [event for event in events if event.get("event") == "protocol_amendment_lock"]
        if locks:
            if len(locks) != 1 or locks[0].get(
                "protocol_amendment_sha256"
            ) != protocol_amendment_sha256:
                raise ProtocolError("ledger protocol amendment lock is missing or conflicting")
            return
        if any(event.get("event") in {"request_attempt", "response"} for event in events):
            raise ProtocolError(
                "cannot add a protocol amendment lock after an API attempt/response exists"
            )
        self.append(
            {
                "event": "protocol_amendment_lock",
                "protocol_version": PROTOCOL_VERSION,
                "protocol_amendment_path": str(
                    DEFAULT_PROTOCOL_AMENDMENT.relative_to(ROOT)
                ),
                "protocol_amendment_sha256": protocol_amendment_sha256,
                "attempt_count_before_lock": 0,
                "response_count_before_lock": 0,
            }
        )

    def attempt_count(self) -> int:
        return sum(event.get("event") == "request_attempt" for event in self.events())

    def http_200_count(self) -> int:
        return sum(event.get("event") == "response" for event in self.events())

    def retry_attempt_count(self) -> int:
        """Count extra wire attempts beyond each call's reserved first try."""

        return sum(
            event.get("event") == "request_attempt"
            and isinstance(event.get("attempt_number"), int)
            and int(event["attempt_number"]) > 1
            for event in self.events()
        )

    def research_denominator_count(self) -> int:
        return sum(
            event.get("event") == "response" and not event.get("nonresearch", False)
            for event in self.events()
        )

    def completed_call_ids(self) -> set[str]:
        return {
            str(event.get("call_id"))
            for event in self.events()
            if event.get("event") == "response" and event.get("call_id")
        }

    def next_attempt_number(self, call_id: str) -> int:
        return 1 + sum(
            event.get("event") == "request_attempt" and event.get("call_id") == call_id
            for event in self.events()
        )

    def append_attempt(
        self,
        job: ExperimentJob,
        *,
        attempt_number: int,
        request_metadata: Mapping[str, Any],
    ) -> None:
        self.append(
            {
                "event": "request_attempt",
                "protocol_version": PROTOCOL_VERSION,
                "protocol_amendment_sha256": _sha256_path(DEFAULT_PROTOCOL_AMENDMENT),
                "call_id": job.call_id,
                "attempt_number": attempt_number,
                "mode": job.mode,
                "arm": job.arm,
                "case_id": job.case_id,
                "cluster_id": job.cluster_id,
                "nonresearch": job.nonresearch,
                "parent_call_id": job.parent_call_id,
                "mutation_family": job.mutation_family,
                "prompt": job.prompt,
                "system": job.system,
                "prompt_sha256": hashlib.sha256(job.prompt.encode("utf-8")).hexdigest(),
                "request_metadata": dict(request_metadata),
            }
        )

    def append_transport_error(
        self,
        job: ExperimentJob,
        *,
        attempt_number: int,
        error: NVIDIAResponsesError,
    ) -> None:
        self.append(
            {
                "event": "transport_error",
                "call_id": job.call_id,
                "attempt_number": attempt_number,
                "mode": job.mode,
                "arm": job.arm,
                "mutation_family": job.mutation_family,
                "error": error.safe_record(),
                "enters_research_denominator": False,
            }
        )

    def append_response(
        self,
        job: ExperimentJob,
        *,
        attempt_number: int,
        response: NVIDIAResponse,
        parsed_rtl: str,
        model_outcome: str,
    ) -> None:
        pending_validation = (
            model_outcome == "accepted_pending_compile" and not job.nonresearch
        )
        initial_verdict = (
            {"status": "PENDING"}
            if pending_validation
            else {
                "status": "NOT_RUN",
                "reason": (
                    "nonresearch_preflight"
                    if job.nonresearch
                    else f"model_outcome:{model_outcome}"
                ),
            }
        )
        self.append(
            {
                "event": "response",
                "protocol_version": PROTOCOL_VERSION,
                "protocol_amendment_sha256": _sha256_path(DEFAULT_PROTOCOL_AMENDMENT),
                "call_id": job.call_id,
                "attempt_number": attempt_number,
                "mode": job.mode,
                "arm": job.arm,
                "case_id": job.case_id,
                "cluster_id": job.cluster_id,
                "nonresearch": job.nonresearch,
                "parent_call_id": job.parent_call_id,
                "mutation_family": job.mutation_family,
                **response.storage_record(),
                "parsed_rtl": parsed_rtl,
                "parsed_rtl_sha256": (
                    hashlib.sha256(parsed_rtl.encode("utf-8")).hexdigest()
                    if parsed_rtl
                    else None
                ),
                "model_outcome": model_outcome,
                "compile_verdict": initial_verdict,
                "formal_verdict": initial_verdict,
                "simulation_verdict": initial_verdict,
                "enters_research_denominator": not job.nonresearch,
            }
        )

    def append_verdict(
        self,
        call_id: str,
        *,
        compile_verdict: Optional[Mapping[str, Any]] = None,
        formal_verdict: Optional[Mapping[str, Any]] = None,
        simulation_verdict: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Attach validation without mutating the original HTTP-200 event."""

        response = self.latest_responses().get(call_id)
        if response is None:
            raise ProtocolError(f"cannot attach verdict to unknown call: {call_id}")
        if (
            response.get("model_outcome") != "accepted_pending_compile"
            or response.get("nonresearch") is True
        ):
            raise ProtocolError(f"cannot validate a non-candidate response: {call_id}")
        compile_record = dict(compile_verdict or {})
        formal_record = dict(formal_verdict or {})
        simulation_record = dict(simulation_verdict or {})
        if not isinstance(compile_record.get("passed"), bool):
            raise ProtocolError("compile_verdict.passed must be boolean")
        allowed_formal = {
            "PROVED",
            "COUNTEREXAMPLE",
            "TIMEOUT",
            "UNSUPPORTED",
            "COMPILE_FAIL",
        }
        if str(formal_record.get("status") or "").upper() not in allowed_formal:
            raise ProtocolError("formal_verdict.status is not a FormalResult status")
        formal_status = str(formal_record.get("status") or "").upper()
        simulation_status = str(simulation_record.get("status") or "").upper()
        allowed_simulation = {
            "PASS",
            "COUNTEREXAMPLE",
            "TIMEOUT",
            "UNSUPPORTED",
            "COMPILE_FAIL",
        }
        if simulation_status not in allowed_simulation:
            raise ProtocolError("simulation_verdict.status is invalid")
        if compile_record["passed"] is False and formal_status != "COMPILE_FAIL":
            raise ProtocolError("a compile failure requires formal status COMPILE_FAIL")
        if compile_record["passed"] is True and formal_status == "COMPILE_FAIL":
            raise ProtocolError("compile/formal verdicts contradict each other")
        if formal_status == "COUNTEREXAMPLE" and formal_record.get(
            "counterexample_replayed"
        ) is not True:
            raise ProtocolError("formal counterexample was not replayed by Icarus")
        if simulation_status in {"PASS", "COUNTEREXAMPLE"}:
            if tuple(simulation_record.get("seeds") or ()) != INDEPENDENT_SIMULATION_SEEDS:
                raise ProtocolError("independent simulation seeds are not frozen 1001..1010")
            if simulation_record.get("cycles_per_seed") != INDEPENDENT_SIMULATION_CYCLES:
                raise ProtocolError("independent simulation must use 200 cycles per seed")
        override = "compile_fail" if compile_record.get("passed") is False else None
        proposed = {
            "compile_verdict": compile_record,
            "formal_verdict": formal_record,
            "simulation_verdict": simulation_record,
            "model_outcome_override": override,
        }
        existing = self.latest_verdicts().get(call_id)
        if existing:
            existing_payload = {key: existing.get(key) for key in proposed}
            if existing_payload == proposed:
                return
            raise ProtocolError(f"conflicting second validation for call: {call_id}")
        self.append(
            {
                "event": "verdict",
                "call_id": call_id,
                **proposed,
            }
        )

    def latest_responses(self) -> dict[str, dict[str, Any]]:
        return {
            str(event["call_id"]): event
            for event in self.events()
            if event.get("event") == "response" and event.get("call_id")
        }

    def latest_verdicts(self) -> dict[str, dict[str, Any]]:
        return {
            str(event["call_id"]): event
            for event in self.events()
            if event.get("event") == "verdict" and event.get("call_id")
        }

    def pending_validation_call_ids(self) -> list[str]:
        verdicts = self.latest_verdicts()
        return sorted(
            call_id
            for call_id, response in self.latest_responses().items()
            if response.get("model_outcome") == "accepted_pending_compile"
            and not response.get("nonresearch", False)
            and call_id not in verdicts
        )

    def validation_job_locks(self) -> dict[str, dict[str, str]]:
        """Return immutable candidate/golden/protocol provenance by call ID."""

        locks: dict[str, dict[str, str]] = {}
        for event in self.events():
            if event.get("event") != "validation_job_lock":
                continue
            call_id = str(event.get("call_id") or "")
            record = {
                "candidate_sha256": str(event.get("candidate_sha256") or ""),
                "golden_sha256": str(event.get("golden_sha256") or ""),
                "formal_protocol_sha256": str(
                    event.get("formal_protocol_sha256") or ""
                ),
            }
            if not call_id or any(
                not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in record.values()
            ):
                raise ProtocolError("corrupt validation-job provenance lock")
            previous = locks.get(call_id)
            if previous is not None and previous != record:
                raise ProtocolError(
                    f"conflicting validation-job provenance locks: {call_id}"
                )
            locks[call_id] = record
        return locks

    def ensure_validation_job_locks(
        self, jobs: Sequence[Mapping[str, Any]]
    ) -> None:
        """Persist the exact validation inputs before accepting worker output."""

        existing = self.validation_job_locks()
        seen: set[str] = set()
        to_append: list[dict[str, str]] = []
        for job in jobs:
            call_id = str(job.get("call_id") or "")
            record = {
                "candidate_sha256": str(job.get("candidate_sha256") or ""),
                "golden_sha256": str(job.get("golden_sha256") or ""),
                "formal_protocol_sha256": str(
                    job.get("formal_protocol_sha256") or ""
                ),
            }
            if not call_id or call_id in seen:
                raise ProtocolError("validation export has a missing or duplicate call_id")
            seen.add(call_id)
            if any(
                not re.fullmatch(r"[0-9a-f]{64}", value)
                for value in record.values()
            ):
                raise ProtocolError(
                    f"validation export has incomplete provenance: {call_id}"
                )
            if call_id in existing:
                if existing[call_id] != record:
                    raise ProtocolError(
                        f"validation export drift for locked call: {call_id}"
                    )
                continue
            to_append.append({"call_id": call_id, **record})
        for record in to_append:
            self.append(
                {
                    "event": "validation_job_lock",
                    "protocol_version": PROTOCOL_VERSION,
                    **record,
                }
            )

    def ingest_validation_jsonl(self, path: Path) -> int:
        """Ingest the formal/simulation worker's call-id keyed contract.

        Each row must contain ``call_id``, ``compile_verdict``,
        ``formal_verdict`` and ``simulation_verdict``.  Re-ingesting identical
        rows is idempotent; conflicting results fail closed.
        """

        ingested = 0
        validation_locks = self.validation_job_locks()
        for row in _read_jsonl(path):
            call_id = str(row.get("call_id") or "")
            if not call_id:
                raise ProtocolError("validation row has no call_id")
            response = self.latest_responses().get(call_id)
            if response is None:
                raise ProtocolError(f"validation row references unknown call: {call_id}")
            expected_candidate_sha = str(response.get("parsed_rtl_sha256") or "")
            declared_candidate_sha = str(row.get("candidate_sha256") or "")
            golden_sha = str(row.get("golden_sha256") or "")
            protocol_sha = str(row.get("formal_protocol_sha256") or "")
            sha_pattern = re.compile(r"^[0-9a-f]{64}$")
            if (
                not sha_pattern.fullmatch(declared_candidate_sha)
                or declared_candidate_sha != expected_candidate_sha
            ):
                raise ProtocolError(f"validation candidate SHA mismatch: {call_id}")
            if not sha_pattern.fullmatch(golden_sha):
                raise ProtocolError(f"validation golden SHA is missing or invalid: {call_id}")
            if not sha_pattern.fullmatch(protocol_sha):
                raise ProtocolError(f"validation protocol SHA is missing or invalid: {call_id}")
            locked = validation_locks.get(call_id)
            if locked is None:
                raise ProtocolError(
                    f"validation row has no frozen export provenance: {call_id}"
                )
            returned_provenance = {
                "candidate_sha256": declared_candidate_sha,
                "golden_sha256": golden_sha,
                "formal_protocol_sha256": protocol_sha,
            }
            if returned_provenance != locked:
                raise ProtocolError(
                    f"validation row drifted from frozen export provenance: {call_id}"
                )
            for field in ("formal_verdict", "simulation_verdict"):
                component = row.get(field)
                if isinstance(component, Mapping):
                    component_candidate_sha = component.get("candidate_sha256")
                    component_golden_sha = component.get("golden_sha256")
                    component_metadata = component.get("metadata")
                    component_protocol_sha = component.get("formal_protocol_sha256")
                    if (
                        component_protocol_sha is None
                        and isinstance(component_metadata, Mapping)
                    ):
                        component_protocol_sha = component_metadata.get(
                            "formal_protocol_sha256"
                        )
                    if (
                        component_candidate_sha is not None
                        and component_candidate_sha != declared_candidate_sha
                    ):
                        raise ProtocolError(
                            f"{field} candidate SHA mismatch: {call_id}"
                        )
                    if component_golden_sha is not None and component_golden_sha != golden_sha:
                        raise ProtocolError(
                            f"{field} golden SHA mismatch: {call_id}"
                        )
                    if (
                        component_protocol_sha is not None
                        and component_protocol_sha != protocol_sha
                    ):
                        raise ProtocolError(
                            f"{field} protocol SHA mismatch: {call_id}"
                        )
            before = call_id in self.latest_verdicts()
            self.append_verdict(
                call_id,
                compile_verdict=row.get("compile_verdict"),
                formal_verdict=row.get("formal_verdict"),
                simulation_verdict=row.get("simulation_verdict"),
            )
            if not before:
                ingested += 1
        return ingested


def feedback_cohort(
    cases: Sequence[ExperimentCase], ledger: ExperimentLedger
) -> list[tuple[ExperimentCase, str]]:
    """Freeze F from eligible no-spec/no-location first-pass outcomes."""

    by_case = {case.case_id: case for case in cases}
    verdicts = ledger.latest_verdicts()
    cohort: list[tuple[ExperimentCase, str]] = []
    for call_id, response in ledger.latest_responses().items():
        if response.get("mode") != "main" or response.get("arm") != "spec0_loc0":
            continue
        if response.get("model_outcome") != "accepted_pending_compile":
            continue
        parsed_rtl = str(response.get("parsed_rtl") or "").strip()
        case = by_case.get(str(response.get("case_id")))
        verdict = verdicts.get(call_id) or {}
        compile_verdict = verdict.get("compile_verdict") or {}
        formal_verdict = verdict.get("formal_verdict") or {}
        if not case or not parsed_rtl or compile_verdict.get("passed") is not True:
            continue
        if str(formal_verdict.get("status") or "").upper() != "COUNTEREXAMPLE":
            continue
        witness = (
            formal_verdict.get("witness")
            or formal_verdict.get("counterexample")
            or formal_verdict.get("trace")
        )
        if not witness:
            continue
        cohort.append(
            (
                replace(case, previous_candidate=parsed_rtl, counterexample=witness),
                call_id,
            )
        )
    cohort.sort(key=lambda pair: pair[0].case_id)
    return cohort


def export_validation_jobs(
    *,
    main_cases: Sequence[ExperimentCase],
    shift_cases: Sequence[ExperimentCase],
    ledger: ExperimentLedger,
    output_path: Path,
    formal_protocol_sha256: str,
    canonical_manifest_path: Path = DEFAULT_CANONICAL_MANIFEST,
    shift_source_path: Optional[Path] = None,
) -> int:
    """Materialize pending formal/simulation jobs without exposing goldens to prompts.

    This is the outbound half of the validation integration contract.  Run
    ``benchmarks.formal_verify validate-batch`` on the resulting JSONL, then
    feed that command's result JSONL back through ``--ingest-verdicts``.
    """

    if not re.fullmatch(r"[0-9a-f]{64}", formal_protocol_sha256):
        raise ProtocolError("validation export requires a calibrated formal protocol SHA")
    cases_by_id = {case.case_id: case for case in (*main_cases, *shift_cases)}
    if len(cases_by_id) != len(main_cases) + len(shift_cases):
        raise ProtocolError("anonymous main/shift case IDs collide")

    canonical_document = _read_json_object(canonical_manifest_path)
    canonical_rows = canonical_document.get("cases")
    if (
        canonical_document.get("schema_version") != 1
        or canonical_document.get("case_count") != 85
        or not isinstance(canonical_rows, list)
        or len(canonical_rows) != 85
    ):
        raise ProtocolError("canonical main-case manifest is incomplete")
    canonical_source = Path(str(canonical_document.get("source_path") or ""))
    if not canonical_source.is_absolute():
        canonical_source = ROOT / canonical_source
    expected_canonical_source_sha = str(
        canonical_document.get("source_sha256") or ""
    )
    if (
        not expected_canonical_source_sha
        or not canonical_source.exists()
        or hashlib.sha256(canonical_source.read_bytes()).hexdigest()
        != expected_canonical_source_sha
    ):
        raise ProtocolError("canonical main-case source SHA drift")
    canonical_by_case: dict[str, dict[str, Any]] = {}
    for row in canonical_rows:
        if isinstance(row, dict) and row.get("case_id"):
            source_case_id = str(row["case_id"])
            if source_case_id in canonical_by_case:
                raise ProtocolError("canonical main-case IDs are duplicated")
            canonical_by_case[source_case_id] = row

    shift_by_id: dict[str, dict[str, Any]] = {}
    if shift_source_path is not None:
        for source_row in _read_jsonl(shift_source_path):
            source_id = str(source_row.get("id") or "")
            if not source_id or source_id in shift_by_id:
                raise ProtocolError("shift validation source IDs are missing or duplicated")
            shift_by_id[source_id] = source_row

    jobs: list[dict[str, Any]] = []
    pending = set(ledger.pending_validation_call_ids())
    for call_id, response in sorted(ledger.latest_responses().items()):
        if call_id not in pending:
            continue
        case = cases_by_id.get(str(response.get("case_id")))
        if case is None or not case.source_case_id or not case.validation_seed_id:
            raise ProtocolError(f"validation mapping is missing for call {call_id}")
        candidate = str(response.get("parsed_rtl") or "")
        candidate_sha = hashlib.sha256(candidate.encode("utf-8")).hexdigest()
        if not candidate or candidate_sha != response.get("parsed_rtl_sha256"):
            raise ProtocolError(f"candidate SHA drift in ledger call {call_id}")
        row: dict[str, Any] = {
            "call_id": call_id,
            "seed_id": case.validation_seed_id,
            "candidate_rtl": candidate,
            "candidate_sha256": candidate_sha,
            "formal_protocol_sha256": formal_protocol_sha256,
        }
        if response.get("mode") == "shift":
            source = shift_by_id.get(case.source_case_id)
            expected_source_sha = case.validation_source_record_sha256
            actual_source_sha = (
                hashlib.sha256(_canonical_json(source).encode("utf-8")).hexdigest()
                if source is not None
                else None
            )
            if not expected_source_sha or actual_source_sha != expected_source_sha:
                raise ProtocolError(
                    f"shift validation source SHA drift for call {call_id}"
                )
            golden = source.get("golden_rtl") if source else None
            if not isinstance(golden, str) or not golden.strip():
                raise ProtocolError(
                    f"shift validation golden is missing for call {call_id}"
                )
            row["golden_rtl"] = golden.rstrip() + "\n"
            row["golden_sha256"] = hashlib.sha256(
                row["golden_rtl"].encode("utf-8")
            ).hexdigest()
        else:
            canonical = canonical_by_case.get(case.source_case_id)
            if canonical is None:
                raise ProtocolError(
                    f"canonical main-case mapping is missing for call {call_id}"
                )
            golden_path = Path(str(canonical.get("canonical_path") or ""))
            if not golden_path.is_absolute():
                golden_path = ROOT / golden_path
            if not golden_path.exists():
                raise ProtocolError(f"canonical golden is missing: {golden_path}")
            golden_sha = hashlib.sha256(golden_path.read_bytes()).hexdigest()
            if golden_sha != canonical.get("canonical_sha256"):
                raise ProtocolError(f"canonical golden SHA drift: {golden_path}")
            row["golden_path"] = str(golden_path)
            row["golden_sha256"] = golden_sha
        jobs.append(row)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{output_path.name}.", suffix=".tmp", dir=output_path.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for row in jobs:
                handle.write(_canonical_json(row) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output_path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    ledger.ensure_validation_job_locks(jobs)
    return len(jobs)


def parse_complete_rtl(text: str) -> str:
    """Extract only complete module-through-endmodule model output."""

    if not text:
        return ""
    fenced = _FENCE_RE.findall(text)
    body = "\n\n".join(fenced) if fenced else text
    modules = _MODULE_RE.findall(body)
    if not modules and fenced:
        modules = _MODULE_RE.findall(text)
    return "\n\n".join(module.strip() for module in modules).strip()


def classify_http_200(response: NVIDIAResponse) -> tuple[str, str]:
    """Return (model_outcome, parsed_rtl); every branch is denominator data."""

    if response.protocol_error:
        return "protocol_error", ""
    if response.is_refusal:
        return "refusal", ""
    if response.is_incomplete:
        return "truncated_or_incomplete", ""
    rtl = parse_complete_rtl(response.output_text)
    if not rtl:
        return "unparseable", ""
    return "accepted_pending_compile", rtl.rstrip() + "\n"


def _retry_delay(error: NVIDIAResponsesError, call_id: str, attempt_number: int) -> float:
    if error.retry_after_seconds is not None:
        return min(60.0, max(0.0, error.retry_after_seconds))
    base = min(30.0, 2.0 ** max(0, attempt_number - 1))
    jitter_bits = int(_digest(call_id, str(attempt_number), length=4), 16)
    return base * (0.75 + 0.5 * (jitter_bits / 65535.0))


def run_jobs(
    jobs: Sequence[ExperimentJob],
    *,
    ledger: ExperimentLedger,
    client: NVIDIAResponsesClient,
    max_retries: int = 4,
    attempt_cap: int = HARD_ATTEMPT_CAP,
    success_cap: int = HARD_SUCCESS_CAP,
    retry_attempt_cap: int = HARD_RETRY_ATTEMPT_CAP,
    config_fingerprint: str = "library-defaults",
    protocol_amendment_sha256: Optional[str] = None,
    sleep: Callable[[float], None] = time.sleep,
) -> RunSummary:
    """Run/resume jobs under the hard attempt and HTTP-200 response caps."""

    if not os.environ.get(API_KEY_ENV, "").strip():
        raise MissingNVIDIAAPIKey(f"{API_KEY_ENV} is required with --run")
    if (
        attempt_cap > HARD_ATTEMPT_CAP
        or success_cap > HARD_SUCCESS_CAP
        or retry_attempt_cap > HARD_RETRY_ATTEMPT_CAP
    ):
        raise ProtocolError("configured budgets may tighten but never raise the frozen hard caps")
    if attempt_cap < 0 or success_cap < 0 or retry_attempt_cap < 0:
        raise ProtocolError("configured budgets must be non-negative")
    if ledger.retry_attempt_count() > retry_attempt_cap:
        raise BudgetExhausted(
            f"transport-retry hard stop already exceeded ({retry_attempt_cap})"
        )
    amendment_sha = protocol_amendment_sha256 or _sha256_path(DEFAULT_PROTOCOL_AMENDMENT)
    ledger.ensure_protocol_amendment_lock(amendment_sha)
    ledger.ensure_schedule(
        jobs,
        config_fingerprint=config_fingerprint,
        protocol_amendment_sha256=amendment_sha,
    )
    completed = ledger.completed_call_ids()
    pending = [job for job in jobs if job.call_id not in completed]
    if ledger.http_200_count() + len(pending) > success_cap:
        raise BudgetExhausted(
            f"planned HTTP-200 total would exceed cap {success_cap}: "
            f"{ledger.http_200_count()} complete + {len(pending)} pending"
        )

    recorded = 0
    transport_errors = 0
    for job in pending:
        for retry_index in range(max_retries + 1):
            if ledger.attempt_count() >= attempt_cap:
                raise BudgetExhausted(f"wire-attempt hard stop reached ({attempt_cap})")
            attempt_number = ledger.next_attempt_number(job.call_id)
            if attempt_number > 1 and ledger.retry_attempt_count() >= retry_attempt_cap:
                raise BudgetExhausted(
                    f"transport-retry hard stop reached ({retry_attempt_cap})"
                )
            ledger.append_attempt(
                job,
                attempt_number=attempt_number,
                request_metadata=client.request_metadata(),
            )
            try:
                response = client.create(
                    prompt=job.prompt,
                    system=job.system,
                    idempotency_key=job.call_id,
                )
            except NVIDIAResponsesError as exc:
                ledger.append_transport_error(
                    job, attempt_number=attempt_number, error=exc
                )
                transport_errors += 1
                if not exc.retryable or retry_index >= max_retries:
                    raise
                sleep(_retry_delay(exc, job.call_id, attempt_number))
                continue
            if job.arm == "ping":
                # The ping establishes authenticated HTTP-200 transport.  Its
                # requested PONG is retained in raw/output_text but is not RTL
                # and must not be mislabeled as an unparseable research repair.
                outcome, parsed_rtl = "ping_http_200", ""
            else:
                outcome, parsed_rtl = classify_http_200(response)
            ledger.append_response(
                job,
                attempt_number=attempt_number,
                response=response,
                parsed_rtl=parsed_rtl,
                model_outcome=outcome,
            )
            recorded += 1
            break

    return RunSummary(
        scheduled=len(jobs),
        skipped_completed=len(jobs) - len(pending),
        http_200_recorded=recorded,
        transport_errors=transport_errors,
        total_attempts_after=ledger.attempt_count(),
        total_http_200_after=ledger.http_200_count(),
        total_retry_attempts_after=ledger.retry_attempt_count(),
    )


def _load_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ProtocolError(f"config is not an object: {path}")
    return value


def validate_protocol_amendment(
    path: Path = DEFAULT_PROTOCOL_AMENDMENT,
    *,
    config: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    """Validate the sole pre-output amendment and every artifact it binds."""

    if path.resolve() != DEFAULT_PROTOCOL_AMENDMENT.resolve() or not path.is_file():
        raise ProtocolError("paid calls require the checked-in 2026-07-15 protocol amendment")
    document = _read_json_object(path)
    if document.get("manifest_sha256") != _manifest_sha(document):
        raise ProtocolError("protocol amendment manifest SHA drift")
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "vericodegen_pre_output_protocol_amendment"
        or document.get("status") != "FROZEN"
        or document.get("effective_date") != "2026-07-15"
        or document.get("decision_timing") != "BEFORE_ANY_MODEL_OUTPUT"
        or document.get("model_outputs_seen_before_freeze") is not False
    ):
        raise ProtocolError("protocol amendment header is not the frozen pre-output decision")

    prestate = document.get("pre_amendment_state")
    if (
        not isinstance(prestate, Mapping)
        or prestate.get("config_sha256")
        != "c2298c393ac3733cf44387ce2850ac487aebb2a72df2d990c455f53e28005e81"
        or prestate.get("strict_v15_gate_sha256")
        != "bcb6482b384d0740ce5a528eccb42fe1aa136014c26e1ac35357f1247377011c"
        or prestate.get("strict_v15_passed") is not False
        or prestate.get("strict_v15_self_proved") != 85
        or prestate.get("strict_v15_formal_mutant_counterexamples") != 84
        or prestate.get("strict_v15_unsupported_mutants") != 1
        or prestate.get("ledger_existed") is not False
        or prestate.get("attempt_count") != 0
        or prestate.get("response_count") != 0
    ):
        raise ProtocolError("protocol amendment prestate is inconsistent")

    amendments = document.get("amendments")
    if not isinstance(amendments, list) or len(amendments) != 2:
        raise ProtocolError("protocol amendment must contain exactly two decisions")
    by_id = {
        str(item.get("id")): item for item in amendments if isinstance(item, Mapping)
    }
    composite = by_id.get("DAY3_EXACT_COMPOSITE_CALIBRATION_EXCEPTION")
    redaction = by_id.get("DROP_REDACTED_ARM_NO_INDEPENDENT_HUMAN_ANNOTATORS")
    if not isinstance(composite, Mapping) or not isinstance(redaction, Mapping):
        raise ProtocolError("protocol amendment decision IDs drifted")
    success_definition = composite.get("calibration_success_definition")
    allowlist = composite.get("allowlist")
    if (
        success_definition
        != {
            "golden_self_proofs": 85,
            "formal_replayable_mutant_counterexamples": 84,
            "independent_simulation_fallback_mutant_counterexamples": 1,
            "distinguishable_known_mutants": 85,
            "false_mutant_proofs": 0,
        }
        or not isinstance(allowlist, list)
        or len(allowlist) != 1
    ):
        raise ProtocolError("protocol amendment composite counts/allowlist drifted")
    exception = allowlist[0]
    if not isinstance(exception, Mapping) or exception != {
        "case_id": "repair_sem_0059",
        "seed_id": "Prob129_ece241_2013_q8_ref",
        "mutation": "flip_reset_polarity",
        "golden_sha256": "1ee2398da6c8170f1d7937eac091a160ec8bb037bc389ae0d66a8d19a5a49716",
        "candidate_sha256": "35c79ef6db0aec962710791bc4a3465979d011f1ae883379ea5fdc1d6f2b57c4",
        "required_formal_status": "UNSUPPORTED",
        "simulation_seeds": list(INDEPENDENT_SIMULATION_SEEDS),
        "cycles_per_seed": INDEPENDENT_SIMULATION_CYCLES,
        "required_status_per_seed": "COUNTEREXAMPLE",
        "required_counterexample_runs": 10,
        "require_concrete_post_initialization_counterexample": True,
    }:
        raise ProtocolError("protocol amendment exact fallback exception drifted")
    if (
        composite.get("forbid_any_other_fallback_case") is not True
        or composite.get("forbid_timeout_or_compile_fail_fallback") is not True
        or redaction.get("decision_basis") != "NO_INDEPENDENT_HUMAN_ANNOTATORS"
        or redaction.get("eligible_case_count") != 0
        or redaction.get("successful_call_count") != 0
        or redaction.get("annotations_performed") is not False
        or redaction.get("forbid_synthetic_or_model_generated_annotations") is not True
    ):
        raise ProtocolError("protocol amendment fail-closed policies drifted")

    if document.get("effective_call_budget") != {
        "successful_calls_before_revision": 453,
        "maximum_revision_calls": 170,
        "maximum_successful_calls": 623,
        "maximum_transport_retry_attempts": 92,
        "effective_maximum_wire_attempts_under_scheduled_cap": 715,
        "maximum_wire_attempts": 800,
    }:
        raise ProtocolError("protocol amendment call budget drifted")
    expected_contrasts = [
        ["what_full_spec", "main", "spec0_loc0", "spec1_loc0"],
        ["where_oracle_location", "main", "spec0_loc0", "spec0_loc1"],
        [
            "counterexample_revision",
            "feedback",
            "generic_failure",
            "concrete_counterexample",
        ],
    ]
    statistics = document.get("primary_statistics")
    if (
        not isinstance(statistics, Mapping)
        or statistics.get("ordered_contrast_definitions") != expected_contrasts
        or statistics.get("bootstrap_replicates") != 10_000
        or statistics.get("multiplicity_adjustment") != "Holm"
    ):
        raise ProtocolError("protocol amendment primary statistics drifted")

    fixed_bindings = {
        "redaction_decision": DEFAULT_REDACTION_MANIFEST,
        "redaction_template": DEFAULT_REDACTION_TEMPLATE,
        "main_input_manifest": DEFAULT_MAIN_INPUT_MANIFEST,
        "benchmark_source": DEFAULT_BENCHMARK_SOURCE,
        "strict_v15_results": ROOT / "generated/formal/calibration_full_v15_results.jsonl",
        "v15_fallback_simulation_report": ROOT
        / "generated/formal/calibration_full_v15/repair_sem_0059/independent_simulation_v3/independent_simulation.json",
    }
    bindings = document.get("bindings")
    if not isinstance(bindings, Mapping):
        raise ProtocolError("protocol amendment artifact bindings are missing")
    for name, expected_path in fixed_bindings.items():
        binding = bindings.get(name)
        if not isinstance(binding, Mapping):
            raise ProtocolError(f"protocol amendment lacks {name} binding")
        recorded_path = Path(str(binding.get("path") or ""))
        if not recorded_path.is_absolute():
            recorded_path = ROOT / recorded_path
        if recorded_path.resolve() != expected_path.resolve() or not expected_path.is_file():
            raise ProtocolError(f"protocol amendment {name} path is stale")
        if binding.get("sha256") != _sha256_path(expected_path):
            raise ProtocolError(f"protocol amendment {name} SHA drift")
    diagnostic = bindings.get("v15_fallback_simulation_report")
    if (
        not isinstance(diagnostic, Mapping)
        or diagnostic.get("purpose") != "PRE_AMENDMENT_DIAGNOSTIC_ONLY"
        or diagnostic.get("qualifies_for_composite_gate") is not False
    ):
        raise ProtocolError("v15 fallback report must remain diagnostic-only")

    validate_frozen_redaction_decision(DEFAULT_REDACTION_MANIFEST)
    if config is not None:
        configured = config.get("protocol_amendment")
        if not isinstance(configured, Mapping):
            raise ProtocolError("checked-in config lacks protocol_amendment binding")
        configured_path = Path(str(configured.get("path") or ""))
        if not configured_path.is_absolute():
            configured_path = ROOT / configured_path
        if (
            configured_path.resolve() != path.resolve()
            or configured.get("required_for_paid_calls") is not True
            or configured.get("file_sha256") != _sha256_path(path)
            or configured.get("manifest_sha256") != document.get("manifest_sha256")
        ):
            raise ProtocolError("checked-in config protocol amendment binding drifted")
        budgets = config.get("call_budget")
        if (
            not isinstance(budgets, Mapping)
            or budgets.get("successful_calls_before_revision") != 453
            or budgets.get("max_successful_research_calls") != 623
            or budgets.get("max_transport_retry_attempts_at_full_revision_cohort") != 92
            or budgets.get("hard_total_http_attempts") != 800
        ):
            raise ProtocolError("checked-in config does not implement amendment budgets")
    return document


def _validate_composite_fallback_artifact(
    row: Mapping[str, Any],
    evidence: Mapping[str, Any],
    *,
    formal_protocol_sha256: str,
    amendment_sha256: str,
) -> list[str]:
    """Re-open the sole allowlisted fallback report and validate raw provenance."""

    failures: list[str] = []
    result = row.get("result") if isinstance(row.get("result"), Mapping) else {}
    expected_identity = {
        "case_id": "repair_sem_0059",
        "seed_id": "Prob129_ece241_2013_q8_ref",
        "mutation": "flip_reset_polarity",
        "formal_status": "UNSUPPORTED",
        "golden_sha256": "1ee2398da6c8170f1d7937eac091a160ec8bb037bc389ae0d66a8d19a5a49716",
        "candidate_sha256": "35c79ef6db0aec962710791bc4a3465979d011f1ae883379ea5fdc1d6f2b57c4",
        "formal_protocol_sha256": formal_protocol_sha256,
        "protocol_amendment_sha256": amendment_sha256,
    }
    actual_identity = {
        "case_id": row.get("case_id"),
        "seed_id": row.get("seed_id"),
        "mutation": row.get("mutation"),
        "formal_status": result.get("status"),
        "golden_sha256": result.get("golden_sha256"),
        "candidate_sha256": result.get("candidate_sha256"),
        "formal_protocol_sha256": row.get("formal_protocol_sha256"),
        "protocol_amendment_sha256": row.get("protocol_amendment_sha256"),
    }
    if actual_identity != expected_identity:
        failures.append("repair_sem_0059 fallback row identity/hash drift")
    pointer = row.get("composite_fallback")
    if not isinstance(pointer, Mapping):
        return failures + ["repair_sem_0059 lacks a fallback evidence pointer"]
    for field, expected in {
        "schema_version": 1,
        "case_id": expected_identity["case_id"],
        "seed_id": expected_identity["seed_id"],
        "golden_sha256": expected_identity["golden_sha256"],
        "candidate_sha256": expected_identity["candidate_sha256"],
        "formal_protocol_sha256": formal_protocol_sha256,
        "protocol_amendment_sha256": amendment_sha256,
    }.items():
        if pointer.get(field) != expected:
            failures.append(f"fallback pointer {field} mismatch")
    report_path = _recorded_path(pointer.get("report_path"), field="fallback report_path")
    if "repair_sem_0059" not in report_path.parts or not report_path.is_file():
        failures.append("fallback report path is missing or not case-bound")
        return failures
    report_sha = str(pointer.get("report_sha256") or "")
    if not re.fullmatch(r"[0-9a-f]{64}", report_sha) or _sha256_path(report_path) != report_sha:
        failures.append("fallback report raw-byte SHA mismatch")
    report = _read_json_object(report_path)
    expected_seeds = list(INDEPENDENT_SIMULATION_SEEDS)
    for field, expected in {
        "status": "COUNTEREXAMPLE",
        "seeds": expected_seeds,
        "cycles_per_seed": INDEPENDENT_SIMULATION_CYCLES,
        "golden_sha256": expected_identity["golden_sha256"],
        "candidate_sha256": expected_identity["candidate_sha256"],
        "formal_protocol_sha256": formal_protocol_sha256,
        "protocol_amendment_sha256": amendment_sha256,
    }.items():
        if report.get(field) != expected:
            failures.append(f"fallback report {field} mismatch")
    runs = report.get("runs")
    if not isinstance(runs, list) or len(runs) != 10:
        failures.append("fallback report does not contain exactly ten runs")
        runs = []
    if [run.get("seed") for run in runs if isinstance(run, Mapping)] != expected_seeds:
        failures.append("fallback report seed order/set drift")
    if any(
        not isinstance(run, Mapping)
        or run.get("cycles") != INDEPENDENT_SIMULATION_CYCLES
        or run.get("status") != "COUNTEREXAMPLE"
        or run.get("passed") is not False
        for run in runs
    ):
        failures.append("fallback report requires 10/10 200-cycle counterexamples")
    summary = report.get("counterexample_summary")
    first = summary.get("first_post_initialization_concrete") if isinstance(summary, Mapping) else None
    concrete_count = summary.get("post_initialization_concrete_count") if isinstance(summary, Mapping) else None
    concrete = (
        isinstance(first, Mapping)
        and first.get("phase") == "post_initialization"
        and first.get("concrete_two_state") is True
        and re.fullmatch(r"[01]+", str(first.get("expected") or "")) is not None
        and re.fullmatch(r"[01]+", str(first.get("got") or "")) is not None
        and first.get("expected") != first.get("got")
    )
    if not isinstance(concrete_count, int) or isinstance(concrete_count, bool) or concrete_count < 1 or not concrete:
        failures.append("fallback report lacks a concrete post-initialization witness")
    expected_evidence = {
        **expected_identity,
        "report_path": pointer.get("report_path"),
        "report_sha256": report_sha,
        "simulation_status": "COUNTEREXAMPLE",
        "seeds": expected_seeds,
        "cycles_per_seed": INDEPENDENT_SIMULATION_CYCLES,
        "counterexample_seed_runs": 10,
        "post_initialization_concrete_count": concrete_count,
        "first_post_initialization_concrete": first,
        "qualified": True,
        "validation_errors": [],
    }
    if dict(evidence) != expected_evidence:
        failures.append("gate fallback evidence is not an exact report-derived summary")
    return failures


def validate_calibration_gate(path: Path) -> dict[str, Any]:
    """Fail closed unless the amended 85-case composite calibration passed."""

    if not path.exists():
        raise ProtocolError(
            f"paid research calls require a calibration gate, but it is missing: {path}"
        )
    gate = _read_json_object(path)
    gold_statuses = gate.get("golden_vs_golden_statuses") or {}
    mutant_statuses = gate.get("golden_vs_mutant_statuses") or {}
    false_equivalent = gate.get("mutants_incorrectly_proved")
    failures: list[str] = []
    if gate.get("schema_version") != 3:
        failures.append("schema_version != 3")
    if gate.get("gate") != "day3_harness_calibration":
        failures.append("wrong gate kind")
    if gate.get("expected_cases") != 85:
        failures.append("expected_cases != 85")
    if gate.get("golden_vs_golden_completed") != 85:
        failures.append("golden-vs-golden completion != 85")
    if not isinstance(gold_statuses, Mapping) or gold_statuses.get("PROVED") != 85:
        failures.append("golden-vs-golden PROVED != 85")
    if gate.get("golden_vs_mutant_completed") != 85:
        failures.append("golden-vs-mutant completion != 85")
    if not isinstance(mutant_statuses, Mapping) or dict(mutant_statuses) != {
        "COUNTEREXAMPLE": 84,
        "UNSUPPORTED": 1,
    }:
        failures.append("known-mutant statuses are not exact 84 CE + 1 UNSUPPORTED")
    if gate.get("replayable_mutant_counterexamples") != 84:
        failures.append("formal replayable known-mutant counterexamples != 84")
    if gate.get("fallback_mutant_counterexamples") != 1:
        failures.append("fallback known-mutant counterexamples != 1")
    if gate.get("distinguishable_mutants") != 85:
        failures.append("distinguishable known mutants != 85")
    if gate.get("fallback_validation_errors") != []:
        failures.append("fallback validation errors are nonempty")
    if gate.get("duplicate_run_ids") != [] or gate.get("unique_run_ids") is not True:
        failures.append("calibration run IDs are not unique")
    if gate.get("comparison_case_id_sets_match") is not True:
        failures.append("calibration comparison case-ID sets differ")
    if gate.get("amendment_bound_rows") != 170:
        failures.append("not all calibration rows bind the amendment")
    if gate.get("formal_protocol_rows_consistent") is not True:
        failures.append("calibration formal protocol rows are inconsistent")
    if false_equivalent != []:
        failures.append("at least one known mutant was incorrectly proved equivalent")
    if gate.get("halted_reason") not in (None, ""):
        failures.append("calibration halted")
    if gate.get("passed") is not True:
        failures.append("passed is not true")
    if gate.get("scope") != "full_sem85":
        failures.append("scope is not full_sem85")
    if gate.get("scope_case_ids") is not None:
        failures.append("full gate unexpectedly has subset case IDs")
    if gate.get("full_day3_gate_passed") is not True:
        failures.append("full_day3_gate_passed is not true")
    if gate.get("gate_manifest_sha256") != _gate_manifest_sha(gate):
        failures.append("gate manifest SHA mismatch")

    sha_pattern = re.compile(r"^[0-9a-f]{64}$")
    protocol_sha = str(gate.get("formal_protocol_sha256") or "")
    if not sha_pattern.fullmatch(protocol_sha):
        failures.append("formal protocol SHA is missing or invalid")
    amendment = validate_protocol_amendment()
    amendment_sha = _sha256_path(DEFAULT_PROTOCOL_AMENDMENT)
    if (
        _recorded_path(gate.get("amendment_path"), field="amendment_path").resolve()
        != DEFAULT_PROTOCOL_AMENDMENT.resolve()
        or gate.get("protocol_amendment_sha256") != amendment_sha
        or gate.get("amendment_manifest_sha256") != amendment.get("manifest_sha256")
    ):
        failures.append("calibration gate protocol amendment binding drift")
    expected_policy = {
        "name": "exact_allowlisted_unsupported_with_frozen_independent_simulation",
        "version": 1,
        "amendment_path": str(DEFAULT_PROTOCOL_AMENDMENT.relative_to(ROOT)),
        "protocol_amendment_sha256": amendment_sha,
        "amendment_manifest_sha256": amendment.get("manifest_sha256"),
        "allowlist": [
            {
                "case_id": "repair_sem_0059",
                "seed_id": "Prob129_ece241_2013_q8_ref",
                "mutation": "flip_reset_polarity",
                "golden_sha256": "1ee2398da6c8170f1d7937eac091a160ec8bb037bc389ae0d66a8d19a5a49716",
                "candidate_sha256": "35c79ef6db0aec962710791bc4a3465979d011f1ae883379ea5fdc1d6f2b57c4",
                "required_formal_status": "UNSUPPORTED",
                "seeds": list(INDEPENDENT_SIMULATION_SEEDS),
                "cycles_per_seed": INDEPENDENT_SIMULATION_CYCLES,
                "required_seed_status": "COUNTEREXAMPLE",
                "requires_post_initialization_concrete_two_state": True,
            }
        ],
    }
    if gate.get("composite_policy") != expected_policy:
        failures.append("calibration composite policy drift")
    fallback_evidence = gate.get("fallback_mutant_evidence")
    if not isinstance(fallback_evidence, list) or len(fallback_evidence) != 1:
        failures.append("calibration gate must contain exactly one fallback evidence row")
        fallback_evidence = []

    try:
        recorded_paths = {
            "benchmark_source": _recorded_path(
                gate.get("benchmark_source"), field="benchmark_source"
            ),
            "canonical_manifest": _recorded_path(
                gate.get("canonical_manifest"), field="canonical_manifest"
            ),
            "protocol_manifest": _recorded_path(
                gate.get("protocol_manifest"), field="protocol_manifest"
            ),
            "results_path": _recorded_path(
                gate.get("results_path"), field="results_path"
            ),
            "toolchain_manifest": _recorded_path(
                gate.get("toolchain_manifest"), field="toolchain_manifest"
            ),
        }
        expected_fixed_paths = {
            "benchmark_source": DEFAULT_BENCHMARK_SOURCE,
            "canonical_manifest": DEFAULT_CANONICAL_MANIFEST,
            "protocol_manifest": DEFAULT_PROTOCOL_MANIFEST,
        }
        for field, expected_path in expected_fixed_paths.items():
            if recorded_paths[field].resolve() != expected_path.resolve():
                failures.append(f"{field} is not the frozen repository artifact")

        hash_fields = {
            "benchmark_source": "benchmark_source_sha256",
            "canonical_manifest": "canonical_manifest_sha256",
            "protocol_manifest": "protocol_manifest_sha256",
            "results_path": "calibration_results_sha256",
            "toolchain_manifest": "toolchain_manifest_sha256",
        }
        for field, hash_field in hash_fields.items():
            artifact_path = recorded_paths[field]
            expected_sha = str(gate.get(hash_field) or "")
            if not artifact_path.is_file():
                failures.append(f"{field} artifact is missing")
            elif not sha_pattern.fullmatch(expected_sha):
                failures.append(f"{hash_field} is missing or invalid")
            elif _sha256_path(artifact_path) != expected_sha:
                failures.append(f"{hash_field} does not match artifact bytes")

        toolchain = _read_json_object(recorded_paths["toolchain_manifest"])
        container_image_id = str(toolchain.get("container_image_id") or "")
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", container_image_id):
            failures.append("toolchain container image ID is missing or invalid")
        lock_sha = str(toolchain.get("toolchain_lock_sha256") or "")
        if lock_sha != _sha256_path(DEFAULT_TOOLCHAIN_LOCK):
            failures.append("toolchain lock SHA does not match the repository lock")
        lock = _read_json_object(DEFAULT_TOOLCHAIN_LOCK)
        required_tools = (
            (lock.get("oss_cad_suite") or {}).get("required_tools")
            if isinstance(lock.get("oss_cad_suite"), Mapping)
            else None
        )
        tools = toolchain.get("tools")
        if not isinstance(required_tools, list) or not isinstance(tools, Mapping):
            failures.append("toolchain required-tool provenance is incomplete")
        else:
            unavailable = [
                str(tool)
                for tool in required_tools
                if not isinstance(tools.get(str(tool)), Mapping)
                or tools[str(tool)].get("available") is not True
            ]
            if unavailable:
                failures.append(f"toolchain tools unavailable: {unavailable}")
        if container_image_id and protocol_sha != _current_formal_protocol_sha256(
            container_image_id
        ):
            failures.append("formal protocol fingerprint is stale for current code/image")

        results = _read_jsonl(recorded_paths["results_path"])
        canonical = _read_json_object(DEFAULT_CANONICAL_MANIFEST)
        canonical_rows = canonical.get("cases")
        canonical_case_ids = {
            str(row.get("case_id"))
            for row in canonical_rows or []
            if isinstance(row, Mapping) and row.get("case_id")
        }
        expected_pairs = {
            (case_id, comparison)
            for case_id in canonical_case_ids
            for comparison in ("golden_vs_golden", "golden_vs_mutant")
        }
        actual_pairs = {
            (str(row.get("case_id") or ""), str(row.get("comparison") or ""))
            for row in results
        }
        run_ids = [str(row.get("run_id") or "") for row in results]
        if (
            len(canonical_case_ids) != 85
            or len(results) != 170
            or actual_pairs != expected_pairs
            or "" in run_ids
            or len(set(run_ids)) != 170
        ):
            failures.append("calibration results are not one exact 85x2 full-set run")
        if any(row.get("formal_protocol_sha256") != protocol_sha for row in results):
            failures.append("calibration result protocol fingerprint drift")
        if any(row.get("protocol_amendment_sha256") != amendment_sha for row in results):
            failures.append("calibration result amendment fingerprint drift")
        fallback_rows: list[Mapping[str, Any]] = []
        for row in results:
            result = row.get("result")
            if not isinstance(result, Mapping):
                failures.append("calibration result payload is missing")
                break
            comparison = row.get("comparison")
            if comparison == "golden_vs_golden" and result.get("status") != "PROVED":
                failures.append("calibration JSONL contains a non-PROVED golden")
                break
            if comparison == "golden_vs_mutant":
                if row.get("case_id") == "repair_sem_0059":
                    if result.get("status") != "UNSUPPORTED":
                        failures.append("repair_sem_0059 is not the sole UNSUPPORTED mutant")
                        break
                    fallback_rows.append(row)
                elif (
                    result.get("status") != "COUNTEREXAMPLE"
                    or result.get("counterexample_replayed") is not True
                    or row.get("composite_fallback") is not None
                ):
                    failures.append(
                        "calibration JSONL contains a non-replayable/non-allowlisted mutant"
                    )
                    break
        if len(fallback_rows) != 1 or len(fallback_evidence) != 1:
            failures.append("calibration JSONL does not contain exactly one fallback row")
        else:
            failures.extend(
                _validate_composite_fallback_artifact(
                    fallback_rows[0],
                    fallback_evidence[0],
                    formal_protocol_sha256=protocol_sha,
                    amendment_sha256=amendment_sha,
                )
            )
    except (OSError, ValueError, ProtocolError) as exc:
        failures.append(f"calibration provenance validation failed: {exc}")
    if failures:
        raise ProtocolError("calibration gate failed: " + "; ".join(failures))
    return gate


def validate_frozen_redaction_decision(path: Path) -> dict[str, Any]:
    """Require the blind human redaction decision before research outputs exist."""

    if not path.exists():
        raise ProtocolError(
            f"paid research calls require the blind redaction freeze: {path}"
        )
    document = _read_json_object(path)
    try:
        validate_redaction_decision_manifest(document)
        verify_data_manifest(document)
    except (OSError, ValueError) as exc:
        raise ProtocolError(f"invalid frozen redaction decision: {exc}") from exc
    if document.get("schema_version") == 2:
        sources = document.get("sources")
        expected_sources = {
            "annotation_template": DEFAULT_REDACTION_TEMPLATE,
            "main_input_manifest": DEFAULT_MAIN_INPUT_MANIFEST,
            "benchmark_source": DEFAULT_BENCHMARK_SOURCE,
        }
        if not isinstance(sources, Mapping):
            raise ProtocolError("schema-v2 redaction drop lacks source provenance")
        for label, expected_path in expected_sources.items():
            source = sources.get(label)
            if not isinstance(source, Mapping):
                raise ProtocolError(f"schema-v2 redaction drop lacks {label}")
            actual_path = Path(str(source.get("path") or ""))
            if not actual_path.is_absolute():
                actual_path = ROOT / actual_path
            if actual_path.resolve() != expected_path.resolve():
                raise ProtocolError(
                    f"schema-v2 redaction {label} is not the frozen repository artifact"
                )
        template = _read_json_object(DEFAULT_REDACTION_TEMPLATE)
        main_manifest = _read_json_object(DEFAULT_MAIN_INPUT_MANIFEST)
        if document.get("annotation_template_sha256") != template.get("manifest_sha256"):
            raise ProtocolError("schema-v2 redaction template manifest SHA mismatch")
        if document.get("main_input_manifest_sha256") != main_manifest.get("manifest_sha256"):
            raise ProtocolError("schema-v2 redaction main-input manifest SHA mismatch")
        if document.get("benchmark_source_sha256") != _sha256_path(DEFAULT_BENCHMARK_SOURCE):
            raise ProtocolError("schema-v2 redaction benchmark source SHA mismatch")
        template_cases = template.get("cases")
        main_index = main_manifest.get("case_index")
        template_ids = {
            str(item.get("case_id") or "")
            for item in template_cases or []
            if isinstance(item, Mapping)
        }
        main_ids = {
            str(item.get("case_id") or "")
            for item in main_index or []
            if isinstance(item, Mapping)
        }
        expected_case_ids_sha = hashlib.sha256(
            _canonical_json(sorted(template_ids)).encode("utf-8")
        ).hexdigest()
        if (
            len(template_ids) != 85
            or template_ids != main_ids
            or document.get("case_ids_sha256") != expected_case_ids_sha
        ):
            raise ProtocolError("schema-v2 redaction full-set case binding mismatch")
        return document
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "vericodegen_frozen_spec_redactions"
        or document.get("status") != "FROZEN"
        or document.get("case_count") != 85
    ):
        raise ProtocolError("redaction manifest is not a frozen 85-case decision")
    decision = document.get("arm_decision")
    fraction = document.get("unredactable_fraction")
    threshold = document.get("maximum_allowed_unredactable_fraction")
    eligible_ids = document.get("eligible_case_ids")
    eligible_count = document.get("eligible_case_count")
    frozen_cases = document.get("cases")
    unredactable_count = document.get("unredactable_count")
    if (
        decision not in {"RUN_REDACTED_ARM", "DROP_REDACTED_ARM"}
        or not isinstance(fraction, (int, float))
        or not isinstance(threshold, (int, float))
        or float(threshold) != 0.20
        or not isinstance(eligible_ids, list)
        or not isinstance(eligible_count, int)
        or eligible_count != len(eligible_ids)
        or len(set(map(str, eligible_ids))) != eligible_count
        or not isinstance(frozen_cases, list)
        or len(frozen_cases) != 85
        or not isinstance(unredactable_count, int)
    ):
        raise ProtocolError("frozen redaction decision fields are inconsistent")
    case_ids = [
        str(case.get("case_id") or "")
        for case in frozen_cases
        if isinstance(case, Mapping)
    ]
    derived_eligible = {
        str(case.get("case_id"))
        for case in frozen_cases
        if isinstance(case, Mapping) and case.get("redactable_without_hint") is True
    }
    if (
        len(case_ids) != 85
        or "" in case_ids
        or len(set(case_ids)) != 85
        or derived_eligible != set(map(str, eligible_ids))
        or eligible_count + unredactable_count != 85
        or abs(float(fraction) - unredactable_count / 85.0) > 1e-12
    ):
        raise ProtocolError("frozen redaction case list/counts are inconsistent")
    expected_decision = (
        "DROP_REDACTED_ARM" if float(fraction) > float(threshold) else "RUN_REDACTED_ARM"
    )
    if decision != expected_decision:
        raise ProtocolError("frozen redaction decision violates the 20% rule")
    return document


def validate_preflight_complete(
    ledger: ExperimentLedger,
    *,
    study_id: str,
    config_fingerprint: str,
) -> None:
    """Require the frozen 13-call preflight before any research call."""

    jobs = build_schedule("preflight", study_id=study_id)
    ledger.ensure_schedule(jobs, config_fingerprint=config_fingerprint)
    completed = ledger.completed_call_ids()
    missing = [job.call_id for job in jobs if job.call_id not in completed]
    if missing:
        raise ProtocolError(
            f"paid research calls require the complete 13-call preflight; {len(missing)} remain"
        )


def _nested(config: Mapping[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = config
    for key in keys:
        if not isinstance(cur, Mapping) or key not in cur:
            return default
        cur = cur[key]
    return cur


def _validate_provider_config(config: Mapping[str, Any]) -> None:
    provider = config.get("provider") or config.get("nvidia") or {}
    if not isinstance(provider, Mapping):
        raise ProtocolError("provider config must be an object")
    expected = {
        "base_url": DEFAULT_BASE_URL,
        "model": DEFAULT_MODEL,
        "reasoning_effort": DEFAULT_REASONING_EFFORT,
        "max_output_tokens": DEFAULT_MAX_OUTPUT_TOKENS,
    }
    for key, frozen in expected.items():
        configured = provider.get(key)
        if configured is not None and configured != frozen:
            raise ProtocolError(
                f"config tries to change frozen provider {key}: {configured!r} != {frozen!r}"
            )
    if provider.get("wire_api") not in (None, "responses"):
        raise ProtocolError("frozen provider wire_api must be responses")
    if provider.get("api_key_env") not in (None, API_KEY_ENV):
        raise ProtocolError(f"provider api_key_env must be {API_KEY_ENV}")
    if "fallback" in provider and provider.get("fallback") is not False:
        raise ProtocolError("provider fallback must be disabled")
    forbidden_secret_fields = {
        key
        for key in provider
        if "key" in str(key).lower() and str(key) != "api_key_env"
    }
    if forbidden_secret_fields:
        raise ProtocolError("API credentials are forbidden in config; use NVIDIA_API_KEY")


def _config_fingerprint(config: Mapping[str, Any]) -> str:
    """Hash the checked-in non-secret config into every paid schedule lock."""

    return hashlib.sha256(_canonical_json(config).encode("utf-8")).hexdigest()


def _config_path(config: Mapping[str, Any], key: str, fallback: Path) -> Path:
    paths = config.get("paths") or {}
    if isinstance(paths, Mapping) and paths.get(key):
        value = Path(str(paths[key]))
        return value if value.is_absolute() else ROOT / value
    return fallback


def _configured_dataset_path(
    config: Mapping[str, Any], key: str, fallback: Optional[Path] = None
) -> Optional[Path]:
    datasets = config.get("datasets") or {}
    if isinstance(datasets, Mapping) and datasets.get(key):
        value = Path(str(datasets[key]))
        return value if value.is_absolute() else ROOT / value
    return fallback


def load_shift_selection(path: Path, source_path: Path) -> set[str]:
    manifest = _read_json_object(path)
    if manifest.get("manifest_sha256") != _manifest_sha(manifest):
        raise ProtocolError("distribution-shift selection manifest SHA drift")
    if manifest.get("kind") != "vericodegen_distribution_shift_sample":
        raise ProtocolError(f"wrong distribution-shift manifest kind: {path}")
    rows = manifest.get("cases")
    if manifest.get("case_count") != 50 or not isinstance(rows, list) or len(rows) != 50:
        raise ProtocolError("distribution-shift manifest must freeze exactly 50 cases")
    selected = {str(row.get("record_id") or "") for row in rows if isinstance(row, Mapping)}
    if "" in selected or len(selected) != 50:
        raise ProtocolError("distribution-shift manifest record IDs are missing or duplicated")
    source = manifest.get("source") or {}
    if isinstance(source, Mapping) and source.get("path") and source.get("sha256"):
        declared_source = Path(str(source["path"]))
        if not declared_source.is_absolute():
            declared_source = ROOT / declared_source
        # Check the source hash when loading the original 274-record file.  A
        # separately materialized 50-case JSONL has its own row hashes.
        if source_path.resolve() == declared_source.resolve():
            actual = hashlib.sha256(source_path.read_bytes()).hexdigest()
            if actual != source["sha256"]:
                raise ProtocolError("distribution-shift source SHA drift")
    return selected


def _summary_for_dry_run(
    mode: str, jobs: Sequence[ExperimentJob], ledger: ExperimentLedger
) -> dict[str, Any]:
    completed = ledger.completed_call_ids()
    by_arm: dict[str, int] = {}
    for job in jobs:
        by_arm[job.arm] = by_arm.get(job.arm, 0) + 1
    return {
        "dry_run": True,
        "mode": mode,
        "scheduled": len(jobs),
        "pending": sum(job.call_id not in completed for job in jobs),
        "completed": sum(job.call_id in completed for job in jobs),
        "by_arm": dict(sorted(by_arm.items())),
        "ledger_attempts": ledger.attempt_count(),
        "ledger_http_200": ledger.http_200_count(),
        "hard_attempt_cap": HARD_ATTEMPT_CAP,
        "hard_http_200_cap": HARD_SUCCESS_CAP,
        "hard_transport_retry_cap": HARD_RETRY_ATTEMPT_CAP,
        "paid_requests_sent": False,
    }


def _resolve_cases_path(
    mode: str, cli_value: Optional[str], config: Mapping[str, Any]
) -> Optional[Path]:
    if mode in {"ping", "preflight"}:
        return None
    if cli_value:
        return Path(cli_value).resolve()
    fallback = DEFAULT_SHIFT_CASES if mode == "shift" else DEFAULT_MAIN_CASES
    config_key = "shift_cases" if mode == "shift" else "main_cases"
    return _config_path(config, config_key, fallback)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=["ping", "preflight", "main", "redacted", "feedback", "shift"],
    )
    parser.add_argument("--run", action="store_true", help="send paid API requests (default: dry-run)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--cases", help="frozen primary/shift JSONL manifest")
    parser.add_argument("--cases-manifest", help="SHA/source manifest for --cases")
    parser.add_argument("--specs", help="optional legacy spec JSONL join")
    parser.add_argument("--annotations", help="optional frozen annotation JSONL overlay")
    parser.add_argument("--canonical-manifest", help="canonical golden/SHA manifest for locations")
    parser.add_argument("--selection-manifest", help="frozen 50-case distribution-shift manifest")
    parser.add_argument("--redaction-manifest", help="frozen two-annotator redaction JSON manifest")
    parser.add_argument("--calibration", help="machine-readable Day-3 calibration gate")
    parser.add_argument(
        "--ingest-verdicts",
        help="append call-id keyed compile/formal/simulation verdict JSONL; sends no API request",
    )
    parser.add_argument(
        "--export-validation-jobs",
        help="write pending call-id keyed formal/simulation jobs; sends no API request",
    )
    parser.add_argument("--ledger", help="append-only event JSONL")
    parser.add_argument("--study-id", help="stable run namespace; defaults to config/protocol")
    parser.add_argument("--max-retries", type=int, default=4)
    args = parser.parse_args(argv)

    config_path = Path(args.config).resolve()
    config = _load_config(config_path)
    if args.run:
        frozen_config = _load_config(DEFAULT_CONFIG)
        if not frozen_config or _canonical_json(config) != _canonical_json(frozen_config):
            raise ProtocolError(
                "paid calls require the exact checked-in configs/vericodegen2026.json"
            )
    _validate_provider_config(config)
    configured_study_id = str(config.get("study_id") or PROTOCOL_VERSION)
    if configured_study_id != FROZEN_STUDY_ID:
        raise ProtocolError(
            f"study_id must remain frozen as {FROZEN_STUDY_ID!r}"
        )
    if args.study_id and args.study_id != configured_study_id:
        raise ProtocolError(
            f"--study-id cannot change the frozen study namespace {configured_study_id!r}"
        )
    study_id = configured_study_id
    ledger_path = (
        Path(args.ledger).resolve()
        if args.ledger
        else _config_path(config, "ledger", DEFAULT_LEDGER)
    )
    frozen_ledger_path = _config_path(config, "ledger", DEFAULT_LEDGER).resolve()
    if args.run and ledger_path.resolve() != frozen_ledger_path:
        raise ProtocolError(
            "paid calls must use the single frozen ledger so global caps cannot split"
        )
    ledger = ExperimentLedger(ledger_path)

    if args.export_validation_jobs:
        if args.run or args.mode or args.ingest_verdicts:
            raise ProtocolError(
                "--export-validation-jobs is offline; omit --run, --mode and --ingest-verdicts"
            )
        main_cases = load_cases(DEFAULT_MAIN_CASES, require_materialized=True)
        selected = load_shift_selection(DEFAULT_SHIFT_MANIFEST, DEFAULT_SHIFT_CASES)
        shift_cases = load_cases(
            DEFAULT_SHIFT_CASES,
            selected_source_ids=selected,
            require_materialized=True,
        )
        canonical_path = (
            Path(args.canonical_manifest).resolve()
            if args.canonical_manifest
            else DEFAULT_CANONICAL_MANIFEST
        )
        shift_source = _configured_dataset_path(
            config, "distribution_shift", ROOT / "data" / "repairbench_realbugs.jsonl"
        )
        calibration_path = (
            Path(args.calibration).resolve()
            if args.calibration
            else DEFAULT_CALIBRATION_GATE
        )
        calibration_gate = validate_calibration_gate(calibration_path)
        formal_protocol_sha256 = str(
            calibration_gate.get("formal_protocol_sha256") or ""
        )
        count = export_validation_jobs(
            main_cases=main_cases,
            shift_cases=shift_cases,
            ledger=ledger,
            output_path=Path(args.export_validation_jobs).resolve(),
            formal_protocol_sha256=formal_protocol_sha256,
            canonical_manifest_path=canonical_path,
            shift_source_path=shift_source,
        )
        print(
            json.dumps(
                {
                    "validation_jobs": count,
                    "output": str(Path(args.export_validation_jobs).resolve()),
                    "paid_requests_sent": False,
                },
                indent=2,
            )
        )
        return 0

    if args.ingest_verdicts:
        if args.run or args.mode:
            raise ProtocolError("--ingest-verdicts is an offline operation; omit --run and --mode")
        ingested = ledger.ingest_validation_jsonl(Path(args.ingest_verdicts).resolve())
        print(
            json.dumps(
                {
                    "ingested": ingested,
                    "pending_validation": len(ledger.pending_validation_call_ids()),
                    "paid_requests_sent": False,
                },
                indent=2,
            )
        )
        return 0
    if not args.mode:
        raise ProtocolError("--mode is required unless --ingest-verdicts is used")

    cases_path = _resolve_cases_path(args.mode, args.cases, config)
    cases: list[ExperimentCase] = []
    redaction_expected_count: Optional[int] = None
    redacted_arm_dropped = False
    if cases_path is not None:
        if not cases_path.exists():
            raise ProtocolError(
                f"frozen case manifest is missing: {cases_path}; no paid call was made"
            )
        specs_path = (
            Path(args.specs).resolve()
            if args.specs
            else _configured_dataset_path(config, "spec_source")
        )
        annotations_path = Path(args.annotations).resolve() if args.annotations else None
        canonical_manifest_path = (
            Path(args.canonical_manifest).resolve()
            if args.canonical_manifest
            else (DEFAULT_CANONICAL_MANIFEST if args.mode in {"main", "redacted", "feedback"} else None)
        )
        overlay_manifest_path: Optional[Path] = None
        selected_source_ids: Optional[set[str]] = None
        if args.mode == "redacted":
            overlay_manifest_path = (
                Path(args.redaction_manifest).resolve()
                if args.redaction_manifest
                else DEFAULT_REDACTION_MANIFEST
            )
            if not overlay_manifest_path.exists():
                raise ProtocolError(
                    f"redacted arm requires the frozen two-annotator manifest: {overlay_manifest_path}"
                )
            redaction_document = validate_frozen_redaction_decision(
                overlay_manifest_path
            )
            redacted_arm_dropped = (
                redaction_document.get("arm_decision") == "DROP_REDACTED_ARM"
            )
            declared_eligible = redaction_document.get("eligible_case_ids")
            redaction_expected_count = redaction_document.get("eligible_case_count")
            if (
                not isinstance(declared_eligible, list)
                or not isinstance(redaction_expected_count, int)
                or redaction_expected_count != len(declared_eligible)
            ):
                raise ProtocolError("frozen redaction eligible IDs/count are inconsistent")
            selected_source_ids = {str(case_id) for case_id in declared_eligible}
            if len(selected_source_ids) != redaction_expected_count:
                raise ProtocolError("frozen redaction eligible IDs are duplicated")
        if args.mode == "shift":
            selection_path = (
                Path(args.selection_manifest).resolve()
                if args.selection_manifest
                else DEFAULT_SHIFT_MANIFEST
            )
            if not selection_path.exists():
                raise ProtocolError(f"frozen shift selection is missing: {selection_path}")
            selected_source_ids = load_shift_selection(selection_path, cases_path)
        if not redacted_arm_dropped:
            cases = load_cases(
                cases_path,
                specs_path=specs_path,
                annotations_path=annotations_path,
                overlay_manifest_path=overlay_manifest_path,
                canonical_manifest_path=canonical_manifest_path,
                selected_source_ids=selected_source_ids,
                materialized_manifest_path=(
                    Path(args.cases_manifest).resolve() if args.cases_manifest else None
                ),
                require_materialized=True,
            )
    jobs = build_schedule(
        args.mode,
        study_id=study_id,
        cases=cases,
        ledger=ledger,
        strict_counts=True,
        redacted_arm_dropped=redacted_arm_dropped,
    )
    expected = EXPECTED_COUNTS.get(args.mode)
    if expected is not None and len(jobs) != expected:
        raise ProtocolError(
            f"{args.mode} schedule drift: expected {expected}, constructed {len(jobs)}"
        )
    if args.mode == "redacted" and len(jobs) != redaction_expected_count:
        raise ProtocolError(
            f"redacted schedule drift: expected {redaction_expected_count}, constructed {len(jobs)}"
        )
    if args.mode == "feedback" and len(jobs) > 170:
        raise ProtocolError(f"feedback schedule exceeds 2F<=170: {len(jobs)}")

    if not args.run:
        print(json.dumps(_summary_for_dry_run(args.mode, jobs, ledger), indent=2))
        return 0

    if args.mode == "redacted" and redacted_arm_dropped:
        raise ProtocolError(
            "the frozen no-annotator decision drops the redacted arm; paid redacted mode is forbidden"
        )
    if args.max_retries < 0:
        raise ProtocolError("--max-retries must be non-negative")
    amendment = validate_protocol_amendment(DEFAULT_PROTOCOL_AMENDMENT, config=config)
    amendment_sha = _sha256_path(DEFAULT_PROTOCOL_AMENDMENT)
    ledger.ensure_protocol_amendment_lock(amendment_sha)
    blindness_manifest_path = (
        Path(args.redaction_manifest).resolve()
        if args.redaction_manifest
        else DEFAULT_REDACTION_MANIFEST
    )
    validate_frozen_redaction_decision(blindness_manifest_path)
    calibration_path = (
        Path(args.calibration).resolve()
        if args.calibration
        else DEFAULT_CALIBRATION_GATE
    )
    validate_calibration_gate(calibration_path)
    if args.mode in {"main", "redacted", "feedback", "shift"}:
        validate_preflight_complete(
            ledger,
            study_id=study_id,
            config_fingerprint=_config_fingerprint(config),
        )
    budgets = config.get("call_budget") or config.get("budgets") or {}
    attempt_cap = min(
        HARD_ATTEMPT_CAP,
        int(
            budgets.get(
                "hard_total_http_attempts",
                budgets.get("hard_attempt_cap", HARD_ATTEMPT_CAP),
            )
        )
        if isinstance(budgets, Mapping)
        else HARD_ATTEMPT_CAP,
    )
    success_cap = min(
        HARD_SUCCESS_CAP,
        int(
            budgets.get(
                "max_successful_research_calls",
                budgets.get("hard_success_cap", HARD_SUCCESS_CAP),
            )
        )
        if isinstance(budgets, Mapping)
        else HARD_SUCCESS_CAP,
    )
    retry_attempt_cap = min(
        HARD_RETRY_ATTEMPT_CAP,
        int(
            budgets.get(
                "max_transport_retry_attempts_at_full_revision_cohort",
                HARD_RETRY_ATTEMPT_CAP,
            )
        )
        if isinstance(budgets, Mapping)
        else HARD_RETRY_ATTEMPT_CAP,
    )
    client = NVIDIAResponsesClient()
    summary = run_jobs(
        jobs,
        ledger=ledger,
        client=client,
        max_retries=args.max_retries,
        attempt_cap=attempt_cap,
        success_cap=success_cap,
        retry_attempt_cap=retry_attempt_cap,
        config_fingerprint=_config_fingerprint(config),
        protocol_amendment_sha256=amendment_sha,
    )
    print(json.dumps(summary.__dict__, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ProtocolError, NVIDIAResponsesError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise SystemExit(2)
