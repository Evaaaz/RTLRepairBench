#!/usr/bin/env python3
"""Build a fail-closed anonymous VeriCodeGen supplement.

The exporter reads only paths named on the command line (plus the frozen
repository defaults).  It never reads process credentials, Git configuration,
or repository remotes.  Internal ledgers remain untouched: the public ledger
retains exact prompts, request/response metadata, candidate hashes, and
formal/simulation verdicts, while raw response bodies and candidate text are
represented by hashes only.

By default the export is refused until the amended composite calibration gate, final formal
coverage gate, full85/clean75 statistics, normalized analysis rows, and formal
validation results all exist.  ``--draft`` is intentionally explicit and is
only for checking the anonymization machinery before the experiment freezes.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import tarfile
import tempfile
from typing import Any, Iterable, Mapping, Sequence


ROOT = Path(os.environ.get("RTLREPAIR_ROOT", "")) if os.environ.get("RTLREPAIR_ROOT") else Path(__file__).resolve().parents[2]
EXPORT_SCHEMA_VERSION = 1
EXPORTER_VERSION = "vericodegen-anonymous-supplement-v1"
PROTOCOL_VERSION = "vericodegen-2026-v1"

DEFAULT_CONFIG = ROOT / "configs" / "vericodegen2026.json"
DEFAULT_PROTOCOL_AMENDMENT = (
    ROOT / "configs" / "vericodegen" / "protocol_amendment_2026-07-15.json"
)
DEFAULT_REDACTION_DECISION = ROOT / "configs" / "vericodegen" / "redaction_manifest.json"
DEFAULT_LEDGER = ROOT / "generated" / "logs" / "vericodegen2026_events.jsonl"
DEFAULT_CALIBRATION_GATE = ROOT / "generated" / "formal" / "calibration_full_gate.json"
DEFAULT_FORMAL_PRIMARY_GATE = ROOT / "generated" / "formal" / "formal_primary_gate.json"
DEFAULT_VALIDATION_RESULTS = ROOT / "generated" / "formal" / "validation_results.jsonl"
DEFAULT_FULL_STATS = ROOT / "generated" / "reports" / "vericodegen_full85_stats.json"
DEFAULT_CLEAN_STATS = ROOT / "generated" / "reports" / "vericodegen_clean75_stats.json"
DEFAULT_FULL_ROWS = ROOT / "generated" / "reports" / "vericodegen_full85_rows.jsonl"
DEFAULT_CLEAN_ROWS = ROOT / "generated" / "reports" / "vericodegen_clean75_rows.jsonl"
DEFAULT_STAGING = ROOT / "generated" / "supplement" / "vericodegen_anonymous"
DEFAULT_MANIFESTS = (
    ROOT / "configs" / "vericodegen" / "main85_inputs.manifest.json",
    ROOT / "configs" / "vericodegen" / "clean75_manifest.json",
    ROOT / "configs" / "vericodegen" / "shift50_manifest.json",
    ROOT / "configs" / "vericodegen" / "shift50_inputs.manifest.json",
    ROOT / "data" / "formal" / "sem85_manifest.json",
    ROOT / "data" / "formal" / "protocol_manifest.json",
    ROOT / "data" / "formal" / "toolchain_manifest.json",
    ROOT / "docker" / "formal" / "toolchain.lock.json",
)
DEFAULT_IDENTIFIER_SOURCES = (
    ROOT / "configs" / "vericodegen" / "main85_inputs.jsonl",
    ROOT / "configs" / "vericodegen" / "shift50_inputs.jsonl",
)

_HEX64_RE = re.compile(r"^[0-9a-f]{64}$")
_SHIFT_ARMS = ("spec0_loc0", "spec1_loc0")
_TERMINAL_FORMAL_STATUSES = {
    "PROVED",
    "COUNTEREXAMPLE",
    "TIMEOUT",
    "UNSUPPORTED",
    "COMPILE_FAIL",
}
_TERMINAL_SIMULATION_STATUSES = {
    "PASS",
    "COUNTEREXAMPLE",
    "TIMEOUT",
    "UNSUPPORTED",
    "COMPILE_FAIL",
}
_SHIFT_EFFECT_FIELDS = {
    "risk_difference",
    "estimate",
    "effect",
    "effect_size",
    "ci",
    "ci95",
    "confidence_interval",
    "p",
    "p_value",
    "p_value_unadjusted",
    "p_value_holm",
    "bootstrap_samples",
    "bootstrap_rng_seed",
    "material_improvement",
    "case_count",
    "seed_count",
}
_CREDENTIAL_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{16,}\b"),
    re.compile(r"\bnvapi-[A-Za-z0-9_-]{12,}\b", re.IGNORECASE),
    re.compile(r"\bBearer\s+[A-Za-z0-9._~+/=-]{12,}\b", re.IGNORECASE),
    re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
)
_EMAIL_RE = re.compile(r"(?<![\w.+-])[\w.+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
_GIT_REMOTE_PATTERNS = (
    re.compile(r"(?:git@|ssh://git@)[^\s\"'<>]+", re.IGNORECASE),
    re.compile(
        r"https?://(?:[^/@\s]+@)?(?:github|gitlab|bitbucket)\.[^/\s]+/[^\s\"'<>]+\.git\b",
        re.IGNORECASE,
    ),
)
_INTERNAL_ID_PATTERNS = (
    re.compile(r"\brepair_sem_\d+\b", re.IGNORECASE),
    re.compile(r"\brealbug:[A-Za-z0-9_:.-]+", re.IGNORECASE),
    re.compile(r"\bProb\d{3}[A-Za-z0-9_]*\b", re.IGNORECASE),
)
_LOCAL_PATH_PATTERNS = (
    (re.compile(r"(?:file://)?/work(?=/|\b)"), "<REPO>"),
    (
        re.compile(r"(?:file://)?/(?:Users|home)/[^/\s\"'<>]+(?:/[^\s\"'<>]*)?"),
        "<LOCAL_PATH>",
    ),
    (re.compile(r"(?:file://)?/(?:private|tmp|var/folders)(?:/[^\s\"'<>]*)?"), "<TEMP_PATH>"),
    (re.compile(r"(?:file://)?/opt/oss-cad-suite(?=/|\b)"), "<TOOLCHAIN>"),
    (re.compile(r"[A-Za-z]:\\Users\\[^\s\"'<>]+", re.IGNORECASE), "<LOCAL_PATH>"),
)

_SECRET_KEYS = {
    "api_key",
    "apikey",
    "access_token",
    "auth_token",
    "authorization",
    "bearer_token",
    "client_secret",
    "cookie",
    "cookies",
    "credential",
    "credentials",
    "headers",
    "password",
    "private_key",
    "secret",
    "token",
}
_AUTHOR_KEYS = {
    "affiliation",
    "affiliations",
    "author",
    "authors",
    "created_by",
    "email",
    "emails",
    "institution",
    "institutions",
    "maintainer",
    "maintainers",
    "owner",
    "submitted_by",
    "user_name",
    "username",
}
_REMOTE_KEYS = {
    "git_remote",
    "git_remote_url",
    "git_url",
    "remote_url",
    "repository_remote",
}


class SupplementExportError(RuntimeError):
    """The requested public artifact is incomplete or not anonymous."""


@dataclass(frozen=True)
class NamedArtifact:
    name: str
    path: Path


@dataclass(frozen=True)
class SupplementInputs:
    config: Path = DEFAULT_CONFIG
    protocol_amendment: Path = DEFAULT_PROTOCOL_AMENDMENT
    redaction_decision: Path = DEFAULT_REDACTION_DECISION
    ledger: Path = DEFAULT_LEDGER
    calibration_gate: Path = DEFAULT_CALIBRATION_GATE
    formal_primary_gate: Path = DEFAULT_FORMAL_PRIMARY_GATE
    validation_results: Path = DEFAULT_VALIDATION_RESULTS
    full_stats: Path = DEFAULT_FULL_STATS
    clean_stats: Path = DEFAULT_CLEAN_STATS
    full_rows: Path = DEFAULT_FULL_ROWS
    clean_rows: Path = DEFAULT_CLEAN_ROWS
    manifests: tuple[Path, ...] = DEFAULT_MANIFESTS
    identifier_sources: tuple[Path, ...] = DEFAULT_IDENTIFIER_SOURCES
    calibration_results: Path | None = None
    additional_artifacts: tuple[NamedArtifact, ...] = field(default_factory=tuple)


def _canonical_bytes(value: Any) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SupplementExportError(f"invalid JSON object: {path.name}") from exc
    if not isinstance(value, dict):
        raise SupplementExportError(f"JSON document is not an object: {path.name}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeDecodeError) as exc:
        raise SupplementExportError(f"cannot read JSONL: {path.name}") from exc
    for line_number, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise SupplementExportError(
                f"invalid JSONL at {path.name}:{line_number}"
            ) from exc
        if not isinstance(row, dict):
            raise SupplementExportError(
                f"JSONL row is not an object at {path.name}:{line_number}"
            )
        rows.append(row)
    return rows


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_bytes(dict(row)).decode("utf-8") + "\n")


def _safe_name(value: str) -> str:
    result = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    if not result or result in {".", ".."}:
        raise SupplementExportError("artifact name is empty after normalization")
    return result


def _design_alias(source_id: str) -> str:
    digest = hashlib.sha256(
        ("vericodegen-public-design-v1\0" + source_id).encode("utf-8")
    ).hexdigest()[:16]
    return f"design_{digest}"


def _runner_cluster_id(source_id: str) -> str:
    digest = hashlib.sha256(
        "\0".join((PROTOCOL_VERSION, "cluster", source_id)).encode("utf-8")
    ).hexdigest()[:16]
    return f"cluster-{digest}"


def _identifier_map(paths: Sequence[Path]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    design_ids: set[str] = set()
    records: list[tuple[dict[str, Any], str]] = []
    for path in paths:
        if not path.exists():
            continue
        for row in _read_jsonl(path):
            metadata = row.get("internal_metadata")
            if not isinstance(metadata, dict):
                continue
            anonymous = str(metadata.get("anonymous_id") or "")
            if not anonymous:
                continue
            records.append((metadata, anonymous))
            for key in ("case_id", "record_id", "source_case_id"):
                value = str(metadata.get(key) or "")
                if value:
                    mapping[value] = anonymous
            for key in ("cluster_id", "source_seed_id", "source_task_id", "seed_id"):
                value = str(metadata.get(key) or "")
                if value:
                    design_ids.add(value)
    for source_id in sorted(design_ids):
        alias = _design_alias(source_id)
        variants = {source_id, _runner_cluster_id(source_id)}
        if source_id.endswith("_ref"):
            variants.add(source_id[:-4])
            variants.add(_runner_cluster_id(source_id[:-4]))
        else:
            variants.add(source_id + "_ref")
            variants.add(_runner_cluster_id(source_id + "_ref"))
        for value in variants:
            mapping.setdefault(value, alias)
    # Some case IDs embed a source task.  Replacing the complete case first
    # keeps the public row aligned with its frozen main_*/shift_* identity.
    for metadata, anonymous in records:
        value = str(metadata.get("case_id") or "")
        if value:
            mapping[value] = anonymous
    return mapping


class _Sanitizer:
    def __init__(
        self,
        identifier_map: Mapping[str, str],
        *,
        redact_tokens: Sequence[str] = (),
    ) -> None:
        self.identifier_pairs = sorted(
            ((str(source), str(target)) for source, target in identifier_map.items() if source),
            key=lambda pair: len(pair[0]),
            reverse=True,
        )
        automatic_tokens: list[str] = []
        root_parts = ROOT.parts
        for marker in ("Users", "home"):
            if marker in root_parts:
                index = root_parts.index(marker)
                if index + 1 < len(root_parts):
                    automatic_tokens.append(root_parts[index + 1])
        supplied = [str(token) for token in redact_tokens if str(token).strip()]
        if any(len(token.strip()) < 3 for token in supplied):
            raise SupplementExportError("redaction tokens must contain at least 3 characters")
        self.author_tokens = sorted(
            set(automatic_tokens + supplied), key=len, reverse=True
        )

    def text(self, value: str) -> str:
        result = value
        # Repository and container paths are replaced before the generic local
        # path expressions so useful repository-relative suffixes survive.
        root_text = str(ROOT)
        result = result.replace(root_text, "<REPO>")
        for pattern, replacement in _LOCAL_PATH_PATTERNS:
            result = pattern.sub(replacement, result)
        for source, target in self.identifier_pairs:
            result = result.replace(source, target)
        # Exact materialized-ID mappings preserve stable public case aliases.
        # This case-insensitive fallback catches protocol labels such as
        # ``prob151_normalized`` and any unindexed benchmark identifier.
        for pattern in _INTERNAL_ID_PATTERNS:
            result = pattern.sub("[ANONYMIZED_INTERNAL_ID]", result)
        for token in self.author_tokens:
            result = re.sub(re.escape(token), "[ANONYMIZED]", result, flags=re.IGNORECASE)
        for pattern in _CREDENTIAL_PATTERNS:
            result = pattern.sub("[REDACTED_CREDENTIAL]", result)
        for pattern in _GIT_REMOTE_PATTERNS:
            result = pattern.sub("[REDACTED_GIT_REMOTE]", result)
        result = _EMAIL_RE.sub("[REDACTED_EMAIL]", result)
        return result

    def value(self, value: Any, *, key: str | None = None) -> Any:
        normalized_key = (key or "").strip().lower().replace("-", "_")
        if normalized_key in _SECRET_KEYS and normalized_key != "api_key_env":
            return "[REDACTED]"
        if normalized_key in _AUTHOR_KEYS:
            return "[ANONYMIZED]"
        if normalized_key in _REMOTE_KEYS:
            return "[REDACTED_GIT_REMOTE]"
        if isinstance(value, dict):
            sanitized: dict[str, Any] = {}
            for raw_key, child in value.items():
                source_key = str(raw_key)
                public_key = self.text(source_key)
                if public_key in sanitized:
                    raise SupplementExportError(
                        "sanitization would collapse distinct JSON object keys"
                    )
                sanitized[public_key] = self.value(child, key=source_key)
            return sanitized
        if isinstance(value, list):
            return [self.value(item) for item in value]
        if isinstance(value, tuple):
            return [self.value(item) for item in value]
        if isinstance(value, str):
            return self.text(value)
        return value


def _hash_metadata(value: Any) -> str | None:
    if value is None or value == "":
        return None
    payload = value.encode("utf-8") if isinstance(value, str) else _canonical_bytes(value)
    return _sha256_bytes(payload)


def _public_ledger_event(event: Mapping[str, Any], sanitizer: _Sanitizer) -> dict[str, Any]:
    kind = str(event.get("event") or "")
    common_keys = (
        "schema_version",
        "recorded_at",
        "event",
        "protocol_version",
        "protocol_amendment_sha256",
        "study_id",
        "call_id",
        "attempt_number",
        "mode",
        "arm",
        "case_id",
        "cluster_id",
        "nonresearch",
        "parent_call_id",
        "mutation_family",
        "enters_research_denominator",
    )
    public = {key: event[key] for key in common_keys if key in event}
    if kind == "protocol_amendment_lock":
        for key in (
            "protocol_amendment_path",
            "attempt_count_before_lock",
            "response_count_before_lock",
        ):
            if key in event:
                public[key] = event[key]
    elif kind == "schedule_lock":
        for key in ("config_fingerprint", "jobs", "schedule_sha256"):
            if key in event:
                public[key] = event[key]
    elif kind == "request_attempt":
        for key in ("prompt", "system", "prompt_sha256", "request_metadata"):
            if key in event:
                public[key] = event[key]
    elif kind == "transport_error":
        error = event.get("error")
        if isinstance(error, Mapping):
            error_public = dict(error)
            request_id = error_public.pop("request_id", None)
            error_public["request_id_sha256"] = _hash_metadata(request_id)
            public["error"] = error_public
    elif kind == "response":
        for key in (
            "served_model",
            "response_status",
            "usage",
            "refusal",
            "incomplete_reason",
            "protocol_error",
            "elapsed_ms",
            "parsed_rtl_sha256",
            "model_outcome",
            "compile_verdict",
            "formal_verdict",
            "simulation_verdict",
        ):
            if key in event:
                public[key] = event[key]
        public["raw_response_sha256"] = _hash_metadata(event.get("raw_response"))
        public["output_text_sha256"] = _hash_metadata(event.get("output_text"))
        public["response_id_sha256"] = _hash_metadata(event.get("response_id"))
        public["request_id_sha256"] = _hash_metadata(event.get("request_id"))
    elif kind == "verdict":
        for key in (
            "compile_verdict",
            "formal_verdict",
            "simulation_verdict",
            "model_outcome_override",
        ):
            if key in event:
                public[key] = event[key]
    elif kind == "validation_job_lock":
        for key in (
            "candidate_sha256",
            "golden_sha256",
            "formal_protocol_sha256",
        ):
            if key in event:
                public[key] = event[key]
    else:
        raise SupplementExportError(f"unknown ledger event type: {kind!r}")
    result = sanitizer.value(public)
    assert isinstance(result, dict)
    return result


def _validate_analysis(path: Path, *, clean: bool) -> dict[str, Any]:
    document = _read_json(path)
    if document.get("kind") != "vericodegen_primary_clustered_analysis":
        raise SupplementExportError(f"not a final clustered analysis: {path.name}")
    expected_filter = 75 if clean else None
    if document.get("case_filter_count") != expected_filter:
        label = "clean75" if clean else "full85"
        raise SupplementExportError(f"{path.name} is not the final {label} analysis")
    contrasts = document.get("contrasts")
    expected = (
        ("what_full_spec", "main", "spec0_loc0", "spec1_loc0", 42),
        ("where_oracle_location", "main", "spec0_loc0", "spec0_loc1", 43),
        (
            "counterexample_revision",
            "feedback",
            "generic_failure",
            "concrete_counterexample",
            44,
        ),
    )
    expected_names = [item[0] for item in expected]
    if not isinstance(contrasts, dict) or set(contrasts) != set(expected_names):
        raise SupplementExportError(f"{path.name} lacks the frozen three-test Holm family")
    if document.get("holm_family") != expected_names:
        raise SupplementExportError(f"{path.name} changes the ordered Holm family")
    for name, mode, arm_a, arm_b, seed in expected:
        result = contrasts[name]
        if (
            not isinstance(result, dict)
            or result.get("mode") != mode
            or result.get("arm_a") != arm_a
            or result.get("arm_b") != arm_b
            or result.get("bootstrap_samples") != 10_000
            or result.get("bootstrap_rng_seed") != seed
        ):
            raise SupplementExportError(
                f"{path.name}/{name} changes its frozen mode/arms/bootstrap/seed"
            )
        case_count = result.get("case_count")
        if name in {"what_full_spec", "where_oracle_location"}:
            expected_cases = 75 if clean else 85
            if type(case_count) is not int or case_count != expected_cases:
                raise SupplementExportError(
                    f"{path.name}/{name} case_count is not the final "
                    f"{'clean75' if clean else 'full85'} cohort"
                )
        elif type(case_count) is not int or case_count <= 0:
            raise SupplementExportError(
                f"{path.name}/{name} has no nonempty frozen paired feedback cohort"
            )

    secondaries = document.get("secondaries")
    shift = secondaries.get("distribution_shift") if isinstance(secondaries, Mapping) else None
    if not isinstance(shift, Mapping):
        raise SupplementExportError(
            f"{path.name} lacks secondaries.distribution_shift coverage"
        )
    required_shift = {
        "status": "NO_ESTIMATE",
        "estimand": "full_frozen_50_task_shift_set",
        "reason": "post_output_validator_indeterminacy_precludes_complete_case_effect_estimation",
        "arm_a": _SHIFT_ARMS[0],
        "arm_b": _SHIFT_ARMS[1],
        "expected_cases": 50,
        "expected_calls": 100,
        "observed_calls": 100,
        "decision_timing": "POST_OUTPUT_CONSERVATIVE_DECISION",
        "complete_case_effect_reported": False,
        "unresolved_imputed": False,
        "secondary": True,
        "multiplicity_adjusted": False,
    }
    for key, expected_value in required_shift.items():
        if shift.get(key) != expected_value:
            raise SupplementExportError(
                f"{path.name}/distribution_shift has invalid {key!r} for NO_ESTIMATE"
            )
    definitive = shift.get("definitive_calls")
    unresolved = shift.get("unresolved_calls")
    if (
        type(definitive) is not int
        or type(unresolved) is not int
        or definitive < 0
        or unresolved < 0
        or definitive + unresolved != 100
    ):
        raise SupplementExportError(
            f"{path.name}/distribution_shift coverage does not partition 100 calls"
        )
    if shift.get("definitive_coverage") != definitive / 100:
        raise SupplementExportError(
            f"{path.name}/distribution_shift definitive_coverage is inconsistent"
        )
    pair_count = shift.get("fully_definitive_pair_count")
    if type(pair_count) is not int or not 0 <= pair_count <= 50:
        raise SupplementExportError(
            f"{path.name}/distribution_shift has invalid fully_definitive_pair_count"
        )
    per_arm = shift.get("per_arm")
    if not isinstance(per_arm, Mapping) or set(per_arm) != set(_SHIFT_ARMS):
        raise SupplementExportError(
            f"{path.name}/distribution_shift lacks exact per-arm coverage"
        )
    combined_reasons: dict[str, int] = {}
    combined_statuses: dict[str, int] = {}
    for arm in _SHIFT_ARMS:
        arm_coverage = per_arm[arm]
        if not isinstance(arm_coverage, Mapping):
            raise SupplementExportError(
                f"{path.name}/distribution_shift/{arm} coverage is invalid"
            )
        arm_definitive = arm_coverage.get("definitive_calls")
        arm_unresolved = arm_coverage.get("unresolved_calls")
        if (
            arm_coverage.get("expected_calls") != 50
            or arm_coverage.get("observed_calls") != 50
            or type(arm_definitive) is not int
            or type(arm_unresolved) is not int
            or arm_definitive < 0
            or arm_unresolved < 0
            or arm_definitive + arm_unresolved != 50
            or arm_coverage.get("definitive_coverage") != arm_definitive / 50
        ):
            raise SupplementExportError(
                f"{path.name}/distribution_shift/{arm} coverage is inconsistent"
            )
        reason_counts = arm_coverage.get("reason_counts")
        if (
            not isinstance(reason_counts, Mapping)
            or any(type(value) is not int or value <= 0 for value in reason_counts.values())
            or sum(reason_counts.values()) != arm_unresolved
        ):
            raise SupplementExportError(
                f"{path.name}/distribution_shift/{arm} reason counts are inconsistent"
            )
        for reason, count in reason_counts.items():
            combined_reasons[str(reason)] = combined_reasons.get(str(reason), 0) + count
        status_counts = arm_coverage.get("status_counts")
        if (
            not isinstance(status_counts, Mapping)
            or any(type(value) is not int or value <= 0 for value in status_counts.values())
            or sum(status_counts.values()) != arm_unresolved
        ):
            raise SupplementExportError(
                f"{path.name}/distribution_shift/{arm} status counts are inconsistent"
            )
        for status, count in status_counts.items():
            combined_statuses[str(status)] = combined_statuses.get(str(status), 0) + count
    if (
        sum(per_arm[arm]["definitive_calls"] for arm in _SHIFT_ARMS) != definitive
        or sum(per_arm[arm]["unresolved_calls"] for arm in _SHIFT_ARMS) != unresolved
        or shift.get("reason_counts") != dict(sorted(combined_reasons.items()))
        or shift.get("status_counts") != dict(sorted(combined_statuses.items()))
    ):
        raise SupplementExportError(
            f"{path.name}/distribution_shift aggregate coverage is inconsistent"
        )
    pending: list[tuple[str, Any]] = [("distribution_shift", shift)]
    while pending:
        prefix, value = pending.pop()
        if isinstance(value, Mapping):
            for raw_key, nested in value.items():
                key = str(raw_key).lower().replace("-", "_")
                nested_path = f"{prefix}.{raw_key}"
                forbidden = (
                    key in _SHIFT_EFFECT_FIELDS
                    or key.startswith(("bootstrap", "p_value", "effect_", "ci_"))
                    or key.endswith(("_p_value", "_confidence_interval"))
                )
                if forbidden and nested is not None:
                    raise SupplementExportError(
                        f"{path.name}/{nested_path} reports a non-null {raw_key!r} "
                        "despite NO_ESTIMATE"
                    )
                pending.append((nested_path, nested))
        elif isinstance(value, list):
            pending.extend((f"{prefix}[{index}]", nested) for index, nested in enumerate(value))
    recorded = document.get("analysis_sha256")
    unsigned = {key: value for key, value in document.items() if key != "analysis_sha256"}
    if not isinstance(recorded, str) or recorded != _sha256_bytes(_canonical_bytes(unsigned)):
        raise SupplementExportError(f"analysis SHA mismatch: {path.name}")
    return document


def _validate_manifest(path: Path) -> dict[str, Any]:
    document = _read_json(path)
    recorded = document.get("manifest_sha256")
    if recorded is not None:
        unsigned = {key: value for key, value in document.items() if key != "manifest_sha256"}
        if not isinstance(recorded, str) or recorded != _sha256_bytes(_canonical_bytes(unsigned)):
            raise SupplementExportError(f"manifest SHA mismatch: {path.name}")
    return document


def _validate_protocol_amendment(path: Path) -> dict[str, Any]:
    document = _validate_manifest(path)
    if (
        document.get("schema_version") != 1
        or document.get("kind") != "vericodegen_pre_output_protocol_amendment"
        or document.get("status") != "FROZEN"
        or document.get("effective_date") != "2026-07-15"
        or document.get("decision_timing") != "BEFORE_ANY_MODEL_OUTPUT"
        or document.get("model_outputs_seen_before_freeze") is not False
    ):
        raise SupplementExportError("protocol amendment is not the frozen pre-output decision")
    prestate = document.get("pre_amendment_state")
    if (
        not isinstance(prestate, dict)
        or prestate.get("ledger_existed") is not False
        or prestate.get("attempt_count") != 0
        or prestate.get("response_count") != 0
        or prestate.get("strict_v15_passed") is not False
        or prestate.get("strict_v15_self_proved") != 85
        or prestate.get("strict_v15_formal_mutant_counterexamples") != 84
        or prestate.get("strict_v15_unsupported_mutants") != 1
    ):
        raise SupplementExportError("protocol amendment pre-output state is inconsistent")
    amendments = document.get("amendments")
    by_id = {
        str(item.get("id")): item
        for item in amendments or []
        if isinstance(item, dict)
    }
    composite = by_id.get("DAY3_EXACT_COMPOSITE_CALIBRATION_EXCEPTION")
    redaction = by_id.get("DROP_REDACTED_ARM_NO_INDEPENDENT_HUMAN_ANNOTATORS")
    allowlist = composite.get("allowlist") if isinstance(composite, dict) else None
    if (
        not isinstance(composite, dict)
        or not isinstance(redaction, dict)
        or not isinstance(allowlist, list)
        or len(allowlist) != 1
        or composite.get("forbid_any_other_fallback_case") is not True
        or composite.get("forbid_timeout_or_compile_fail_fallback") is not True
        or redaction.get("decision_basis") != "NO_INDEPENDENT_HUMAN_ANNOTATORS"
        or redaction.get("eligible_case_count") != 0
        or redaction.get("successful_call_count") != 0
        or redaction.get("annotations_performed") is not False
        or redaction.get("forbid_synthetic_or_model_generated_annotations") is not True
    ):
        raise SupplementExportError("protocol amendment decisions are incomplete")
    exception = allowlist[0]
    expected_seeds = list(range(1001, 1011))
    if (
        not isinstance(exception, dict)
        or exception.get("case_id") != "repair_sem_0059"
        or exception.get("seed_id") != "Prob129_ece241_2013_q8_ref"
        or exception.get("golden_sha256")
        != "1ee2398da6c8170f1d7937eac091a160ec8bb037bc389ae0d66a8d19a5a49716"
        or exception.get("candidate_sha256")
        != "35c79ef6db0aec962710791bc4a3465979d011f1ae883379ea5fdc1d6f2b57c4"
        or exception.get("required_formal_status") != "UNSUPPORTED"
        or exception.get("simulation_seeds") != expected_seeds
        or exception.get("cycles_per_seed") != 200
        or exception.get("required_status_per_seed") != "COUNTEREXAMPLE"
        or exception.get("required_counterexample_runs") != 10
        or exception.get("require_concrete_post_initialization_counterexample") is not True
    ):
        raise SupplementExportError("protocol amendment exact fallback identity drifted")
    if document.get("effective_call_budget") != {
        "successful_calls_before_revision": 453,
        "maximum_revision_calls": 170,
        "maximum_successful_calls": 623,
        "maximum_transport_retry_attempts": 92,
        "effective_maximum_wire_attempts_under_scheduled_cap": 715,
        "maximum_wire_attempts": 800,
    }:
        raise SupplementExportError("protocol amendment call budget drifted")
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
        not isinstance(statistics, dict)
        or statistics.get("ordered_contrast_definitions") != expected_contrasts
        or statistics.get("bootstrap_replicates") != 10_000
        or statistics.get("multiplicity_adjustment") != "Holm"
    ):
        raise SupplementExportError("protocol amendment primary statistics drifted")
    return document


def _validate_redaction_cancellation(path: Path) -> dict[str, Any]:
    document = _validate_manifest(path)
    if (
        document.get("schema_version") != 2
        or document.get("kind") != "vericodegen_frozen_spec_redactions"
        or document.get("status") != "FROZEN"
        or document.get("arm_decision") != "DROP_REDACTED_ARM"
        or document.get("decision_basis") != "NO_INDEPENDENT_HUMAN_ANNOTATORS"
        or document.get("decision_timing") != "BEFORE_ANY_MODEL_OUTPUT"
        or document.get("model_outputs_seen_before_freeze") is not False
        or document.get("annotations_performed") is not False
        or document.get("eligible_case_count") != 0
        or document.get("eligible_case_ids") != []
        or document.get("cases") != []
    ):
        raise SupplementExportError("redaction arm is not the frozen pre-output cancellation")
    return document


def _validate_composite_calibration(
    calibration: Mapping[str, Any],
    calibration_results: Path,
    amendment_path: Path,
    amendment: Mapping[str, Any],
) -> Path:
    amendment_sha = _sha256_path(amendment_path)
    formal_protocol_sha = str(calibration.get("formal_protocol_sha256") or "")
    if not (
        calibration.get("schema_version") == 3
        and calibration.get("gate") == "day3_harness_calibration"
        and calibration.get("expected_cases") == 85
        and calibration.get("golden_vs_golden_completed") == 85
        and calibration.get("golden_vs_golden_statuses") == {"PROVED": 85}
        and calibration.get("golden_vs_mutant_completed") == 85
        and calibration.get("golden_vs_mutant_statuses")
        == {"COUNTEREXAMPLE": 84, "UNSUPPORTED": 1}
        and calibration.get("replayable_mutant_counterexamples") == 84
        and calibration.get("fallback_mutant_counterexamples") == 1
        and calibration.get("distinguishable_mutants") == 85
        and calibration.get("mutants_incorrectly_proved") == []
        and calibration.get("fallback_validation_errors") == []
        and calibration.get("duplicate_run_ids") == []
        and calibration.get("unique_run_ids") is True
        and calibration.get("comparison_case_id_sets_match") is True
        and calibration.get("amendment_bound_rows") == 170
        and calibration.get("formal_protocol_rows_consistent") is True
        and calibration.get("halted_reason") in (None, "")
        and calibration.get("passed") is True
        and calibration.get("full_day3_gate_passed") is True
        and calibration.get("scope") == "full_sem85"
        and calibration.get("scope_case_ids") is None
        and calibration.get("protocol_amendment_sha256") == amendment_sha
        and _HEX64_RE.fullmatch(formal_protocol_sha) is not None
    ):
        raise SupplementExportError("full composite Day-3 calibration gate is not green")
    policy = calibration.get("composite_policy")
    amendment_exception = next(
        item
        for item in amendment["amendments"]
        if item["id"] == "DAY3_EXACT_COMPOSITE_CALIBRATION_EXCEPTION"
    )["allowlist"][0]
    if (
        not isinstance(policy, dict)
        or policy.get("name")
        != "exact_allowlisted_unsupported_with_frozen_independent_simulation"
        or policy.get("version") != 1
        or policy.get("protocol_amendment_sha256") != amendment_sha
        or policy.get("allowlist") != [
            {
                "case_id": amendment_exception["case_id"],
                "seed_id": amendment_exception["seed_id"],
                "mutation": amendment_exception["mutation"],
                "golden_sha256": amendment_exception["golden_sha256"],
                "candidate_sha256": amendment_exception["candidate_sha256"],
                "required_formal_status": "UNSUPPORTED",
                "seeds": list(range(1001, 1011)),
                "cycles_per_seed": 200,
                "required_seed_status": "COUNTEREXAMPLE",
                "requires_post_initialization_concrete_two_state": True,
            }
        ]
    ):
        raise SupplementExportError("composite calibration policy does not match the amendment")
    evidence_rows = calibration.get("fallback_mutant_evidence")
    if not isinstance(evidence_rows, list) or len(evidence_rows) != 1:
        raise SupplementExportError("composite calibration must contain exactly one fallback")
    evidence = evidence_rows[0]
    if (
        not isinstance(evidence, dict)
        or evidence.get("case_id") != amendment_exception["case_id"]
        or evidence.get("seed_id") != amendment_exception["seed_id"]
        or evidence.get("formal_status") != "UNSUPPORTED"
        or evidence.get("golden_sha256") != amendment_exception["golden_sha256"]
        or evidence.get("candidate_sha256") != amendment_exception["candidate_sha256"]
        or evidence.get("protocol_amendment_sha256") != amendment_sha
        or evidence.get("formal_protocol_sha256") != formal_protocol_sha
        or evidence.get("simulation_status") != "COUNTEREXAMPLE"
        or evidence.get("seeds") != list(range(1001, 1011))
        or evidence.get("cycles_per_seed") != 200
        or evidence.get("counterexample_seed_runs") != 10
        or evidence.get("qualified") is not True
        or evidence.get("validation_errors") != []
    ):
        raise SupplementExportError("composite fallback summary is not the exact exception")

    result_rows = _read_jsonl(calibration_results)
    if len(result_rows) != 170:
        raise SupplementExportError("calibration results are not one exact 85x2 run")
    by_pair: dict[tuple[str, str], dict[str, Any]] = {}
    for row in result_rows:
        pair = (str(row.get("case_id") or ""), str(row.get("comparison") or ""))
        if not all(pair) or pair in by_pair:
            raise SupplementExportError("calibration results contain a duplicate/missing pair")
        if row.get("protocol_amendment_sha256") != amendment_sha:
            raise SupplementExportError("calibration row is not bound to the amendment")
        if row.get("formal_protocol_sha256") != formal_protocol_sha:
            raise SupplementExportError("calibration row formal protocol drifted")
        by_pair[pair] = row
    case_ids = {case_id for case_id, _comparison in by_pair}
    if len(case_ids) != 85 or set(by_pair) != {
        (case_id, comparison)
        for case_id in case_ids
        for comparison in ("golden_vs_golden", "golden_vs_mutant")
    }:
        raise SupplementExportError("calibration result case/comparison coverage drifted")
    fallback_row = by_pair[(amendment_exception["case_id"], "golden_vs_mutant")]
    fallback_result = fallback_row.get("result")
    pointer = fallback_row.get("composite_fallback")
    if (
        not isinstance(fallback_result, dict)
        or fallback_result.get("status") != "UNSUPPORTED"
        or fallback_result.get("golden_sha256") != amendment_exception["golden_sha256"]
        or fallback_result.get("candidate_sha256") != amendment_exception["candidate_sha256"]
        or not isinstance(pointer, dict)
        or pointer.get("report_sha256") != evidence.get("report_sha256")
        or pointer.get("protocol_amendment_sha256") != amendment_sha
        or pointer.get("formal_protocol_sha256") != formal_protocol_sha
    ):
        raise SupplementExportError("fallback result row/source identity drifted")
    for pair, row in by_pair.items():
        result = row.get("result")
        if not isinstance(result, dict):
            raise SupplementExportError("calibration result payload is missing")
        if pair[1] == "golden_vs_golden" and result.get("status") != "PROVED":
            raise SupplementExportError("calibration contains a non-PROVED golden")
        if pair[1] == "golden_vs_mutant" and pair[0] != amendment_exception["case_id"]:
            if result.get("status") != "COUNTEREXAMPLE" or result.get(
                "counterexample_replayed"
            ) is not True:
                raise SupplementExportError("calibration contains a non-replayed mutant")

    report_path = _resolve_recorded_path(evidence.get("report_path"))
    if report_path is None and isinstance(evidence.get("report_path"), str):
        candidate = Path(str(evidence["report_path"]))
        if candidate.is_absolute():
            try:
                candidate.resolve().relative_to(calibration_results.parent.resolve())
                report_path = candidate
            except ValueError:
                pass
    report_sha = str(evidence.get("report_sha256") or "")
    if (
        report_path is None
        or not report_path.is_file()
        or not _HEX64_RE.fullmatch(report_sha)
        or _sha256_path(report_path) != report_sha
    ):
        raise SupplementExportError("fallback simulation report byte SHA mismatch")
    report = _read_json(report_path)
    runs = report.get("runs")
    if not (
        report.get("status") == "COUNTEREXAMPLE"
        and report.get("golden_sha256") == amendment_exception["golden_sha256"]
        and report.get("candidate_sha256") == amendment_exception["candidate_sha256"]
        and report.get("protocol_amendment_sha256") == amendment_sha
        and report.get("formal_protocol_sha256") == formal_protocol_sha
        and report.get("seeds") == list(range(1001, 1011))
        and report.get("cycles_per_seed") == 200
        and isinstance(runs, list)
        and [run.get("seed") for run in runs if isinstance(run, dict)]
        == list(range(1001, 1011))
        and all(
            isinstance(run, dict)
            and run.get("cycles") == 200
            and run.get("status") == "COUNTEREXAMPLE"
            and run.get("passed") is False
            for run in runs
        )
    ):
        raise SupplementExportError("fallback report is not the raw-source frozen 10x200 run")
    summary = report.get("counterexample_summary")
    first = summary.get("first_post_initialization_concrete") if isinstance(summary, dict) else None
    if (
        not isinstance(summary, dict)
        or not isinstance(summary.get("post_initialization_concrete_count"), int)
        or summary.get("post_initialization_concrete_count", 0) < 1
        or not isinstance(first, dict)
        or first.get("phase") != "post_initialization"
        or first.get("concrete_two_state") is not True
        or not re.fullmatch(r"[01]+", str(first.get("expected") or ""))
        or not re.fullmatch(r"[01]+", str(first.get("got") or ""))
        or first.get("expected") == first.get("got")
    ):
        raise SupplementExportError("fallback report lacks a concrete post-initialization witness")
    return report_path


def _derive_terminal_success(
    call_id: str,
    response: Mapping[str, Any],
    verdict: Mapping[str, Any] | None,
) -> tuple[bool | None, dict[str, Any]]:
    """Derive an outcome from terminal evidence without trusting analysis rows."""

    if response.get("model_outcome") != "accepted_pending_compile":
        if verdict is not None:
            raise SupplementExportError(
                f"{call_id}: non-candidate response has a validation verdict"
            )
        return False, {
            "resolved": True,
            "evidence": "http_200_model_failure",
            "formal_status": None,
            "simulation_status": None,
            "reason": None,
            "formal_reason": None,
            "simulation_reason": None,
        }
    if verdict is None:
        raise SupplementExportError(f"{call_id}: accepted candidate lacks terminal verdict")
    compile_verdict = verdict.get("compile_verdict")
    formal = verdict.get("formal_verdict")
    simulation = verdict.get("simulation_verdict")
    if (
        not isinstance(compile_verdict, Mapping)
        or not isinstance(formal, Mapping)
        or not isinstance(simulation, Mapping)
    ):
        raise SupplementExportError(f"{call_id}: terminal verdict contract is incomplete")
    compile_passed = compile_verdict.get("passed")
    if type(compile_passed) is not bool:
        raise SupplementExportError(f"{call_id}: compile verdict is not boolean")
    formal_status = str(formal.get("status") or "").upper()
    simulation_status = str(simulation.get("status") or "").upper()
    formal_reason = str(formal["reason"]) if formal.get("reason") is not None else None
    simulation_reason = (
        str(simulation["reason"]) if simulation.get("reason") is not None else None
    )
    if formal_status not in _TERMINAL_FORMAL_STATUSES:
        raise SupplementExportError(f"{call_id}: formal verdict is not terminal")
    if simulation_status not in _TERMINAL_SIMULATION_STATUSES:
        raise SupplementExportError(f"{call_id}: simulation verdict is not terminal")
    if compile_passed is False and formal_status != "COMPILE_FAIL":
        raise SupplementExportError(f"{call_id}: compile/formal verdicts contradict")
    if compile_passed is True and formal_status == "COMPILE_FAIL":
        raise SupplementExportError(f"{call_id}: compile/formal verdicts contradict")
    metadata = {
        "resolved": True,
        "formal_status": formal_status,
        "simulation_status": simulation_status,
        "reason": None,
        "formal_reason": formal_reason,
        "simulation_reason": simulation_reason,
    }
    if compile_passed is False:
        return False, {**metadata, "evidence": "compile_fail"}
    if formal_status == "PROVED":
        if simulation_status == "COUNTEREXAMPLE":
            raise SupplementExportError(
                f"{call_id}: formal proof conflicts with independent simulation"
            )
        return True, {**metadata, "evidence": "unbounded_formal_proof"}
    if formal_status == "COUNTEREXAMPLE":
        if formal.get("counterexample_replayed") is not True:
            raise SupplementExportError(
                f"{call_id}: formal counterexample was not replayed"
            )
        return False, {**metadata, "evidence": "replayable_formal_counterexample"}
    if formal_status in {"TIMEOUT", "UNSUPPORTED"}:
        if simulation_status in {"PASS", "COUNTEREXAMPLE"}:
            return simulation_status == "PASS", {
                **metadata,
                "evidence": (
                    "independent_simulation_fallback_after_" + formal_status.lower()
                ),
            }
        reason = (
            simulation_reason
            or formal_reason
            or f"formal_{formal_status.lower()}__simulation_{simulation_status.lower()}"
        )
        return None, {
            **metadata,
            "resolved": False,
            "evidence": "inconclusive_shift_validation",
            "reason": reason,
        }
    raise SupplementExportError(f"{call_id}: invalid terminal outcome")


def _require_normalized_shift_row(
    *,
    label: str,
    row: Mapping[str, Any],
    response: Mapping[str, Any],
    expected_success: bool | None,
    metadata: Mapping[str, Any],
) -> None:
    call_id = str(response.get("call_id") or "")
    expected_identity = {
        "mode": response.get("mode"),
        "case_id": response.get("case_id"),
        "seed_id": response.get("cluster_id"),
        "arm": response.get("arm"),
    }
    if any(row.get(key) != value for key, value in expected_identity.items()):
        raise SupplementExportError(
            f"{label} normalized shift identity drift for {call_id}"
        )
    if "success" not in row or type(row.get("success")) is not type(expected_success):
        raise SupplementExportError(
            f"{label} normalized shift success type drift for {call_id}"
        )
    if row.get("success") != expected_success:
        raise SupplementExportError(
            f"{label} normalized shift outcome contradicts ledger for {call_id}"
        )
    for key in (
        "resolved",
        "evidence",
        "formal_status",
        "simulation_status",
        "reason",
        "formal_reason",
        "simulation_reason",
    ):
        if row.get(key) != metadata.get(key):
            raise SupplementExportError(
                f"{label} normalized shift {key} contradicts ledger for {call_id}"
            )


def _validate_final_inputs(inputs: SupplementInputs, *, strict: bool) -> dict[str, Any]:
    required = {
        "study config": inputs.config,
        "pre-output protocol amendment": inputs.protocol_amendment,
        "redaction cancellation decision": inputs.redaction_decision,
        "experiment ledger": inputs.ledger,
        "calibration gate": inputs.calibration_gate,
        "formal-primary gate": inputs.formal_primary_gate,
        "validation results": inputs.validation_results,
        "full85 statistics": inputs.full_stats,
        "clean75 statistics": inputs.clean_stats,
        "full85 normalized rows": inputs.full_rows,
        "clean75 normalized rows": inputs.clean_rows,
    }
    required.update({f"manifest {index}": path for index, path in enumerate(inputs.manifests)})
    if strict:
        missing = [label for label, path in required.items() if not path.is_file()]
        if missing:
            raise SupplementExportError("missing final artifact(s): " + ", ".join(missing))
    for label, path in required.items():
        if path.exists() and not path.is_file():
            raise SupplementExportError(f"{label} is not a regular file")

    calibration = _read_json(inputs.calibration_gate) if inputs.calibration_gate.exists() else {}
    formal_gate = _read_json(inputs.formal_primary_gate) if inputs.formal_primary_gate.exists() else {}
    amendment = (
        _validate_protocol_amendment(inputs.protocol_amendment)
        if inputs.protocol_amendment.exists()
        else {}
    )
    redaction_decision = (
        _validate_redaction_cancellation(inputs.redaction_decision)
        if inputs.redaction_decision.exists()
        else {}
    )
    calibration_results = inputs.calibration_results or _resolve_recorded_path(
        calibration.get("results_path")
    )
    fallback_report: Path | None = None
    full_analysis: dict[str, Any] = {}
    clean_analysis: dict[str, Any] = {}
    analysis_coverage: dict[str, Any] | None = None
    if strict:
        if not amendment or not redaction_decision:
            raise SupplementExportError("pre-output amendment/cancellation is missing")
        gate_unsigned = {
            key: value
            for key, value in calibration.items()
            if key != "gate_manifest_sha256"
        }
        if calibration.get("gate_manifest_sha256") != _sha256_bytes(
            _canonical_bytes(gate_unsigned)
        ):
            raise SupplementExportError("calibration gate self-digest is invalid")
        recorded_results_sha = str(calibration.get("calibration_results_sha256") or "")
        if not (
            calibration_results is not None
            and calibration_results.is_file()
            and _HEX64_RE.fullmatch(recorded_results_sha)
            and _sha256_path(calibration_results) == recorded_results_sha
        ):
            raise SupplementExportError("calibration results do not match the final gate")
        assert calibration_results is not None
        fallback_report = _validate_composite_calibration(
            calibration,
            calibration_results,
            inputs.protocol_amendment,
            amendment,
        )
        if not (
            formal_gate.get("gate") == "formal_primary_coverage"
            and isinstance(formal_gate.get("passed"), bool)
        ):
            raise SupplementExportError("formal-primary coverage gate is not final")
        required_arms = {
            "spec0_loc0",
            "spec1_loc0",
            "spec0_loc1",
            "spec1_loc1",
        }
        arm_reports = formal_gate.get("arms")
        final_arms = (
            isinstance(arm_reports, dict)
            and set(arm_reports) == required_arms
            and all(
                isinstance(report, dict) and report.get("records") == 85
                for report in arm_reports.values()
            )
        )
        reporting_mode = formal_gate.get("reporting_mode")
        expected_mode = (
            "formal_primary"
            if formal_gate.get("passed") is True
            else "formal_where_supported_plus_independent_simulation_fallback"
        )
        if not (
            formal_gate.get("expected_cases_per_arm") == 85
            and set(formal_gate.get("required_arms") or ()) == required_arms
            and final_arms
            and formal_gate.get("conflicts") == []
            and formal_gate.get("malformed_record_indexes") == []
            and reporting_mode == expected_mode
        ):
            raise SupplementExportError("formal-primary coverage gate is incomplete")
        full_analysis = _validate_analysis(inputs.full_stats, clean=False)
        clean_analysis = _validate_analysis(inputs.clean_stats, clean=True)
        for path in inputs.manifests:
            _validate_manifest(path)

    events = _read_jsonl(inputs.ledger) if inputs.ledger.exists() else []
    responses: dict[str, dict[str, Any]] = {}
    verdicts: dict[str, dict[str, Any]] = {}
    for event in events:
        if event.get("mode") == "redacted":
            raise SupplementExportError("cancelled redaction arm appears in the ledger")
        call_id = str(event.get("call_id") or "")
        if event.get("event") == "response" and call_id:
            if call_id in responses:
                raise SupplementExportError(f"duplicate terminal response: {call_id}")
            responses[call_id] = event
        elif event.get("event") == "verdict" and call_id:
            if call_id in verdicts:
                raise SupplementExportError(f"duplicate terminal verdict: {call_id}")
            verdicts[call_id] = event
    if strict:
        orphan_verdicts = sorted(set(verdicts) - set(responses))
        if orphan_verdicts:
            raise SupplementExportError(
                f"orphan verdict without response: {orphan_verdicts[0]}"
            )
        unresolved = []
        for call_id, response in responses.items():
            if response.get("nonresearch") is True:
                continue
            if response.get("model_outcome") == "accepted_pending_compile":
                candidate_sha = str(response.get("parsed_rtl_sha256") or "")
                if not _HEX64_RE.fullmatch(candidate_sha) or call_id not in verdicts:
                    unresolved.append(call_id)
        if unresolved:
            raise SupplementExportError(
                f"ledger has {len(unresolved)} candidate(s) without a final SHA/verdict"
            )
        accepted = {
            call_id: str(response.get("parsed_rtl_sha256"))
            for call_id, response in responses.items()
            if response.get("nonresearch") is not True
            and response.get("model_outcome") == "accepted_pending_compile"
        }
        validation_rows = _read_jsonl(inputs.validation_results)
        validation_by_call: dict[str, dict[str, Any]] = {}
        for row in validation_rows:
            call_id = str(row.get("call_id") or "")
            if not call_id or call_id in validation_by_call:
                raise SupplementExportError("validation results have a missing/duplicate call_id")
            validation_by_call[call_id] = row
        if set(validation_by_call) != set(accepted):
            raise SupplementExportError("validation results do not match accepted ledger candidates")
        for call_id, candidate_sha in accepted.items():
            if validation_by_call[call_id].get("candidate_sha256") != candidate_sha:
                raise SupplementExportError(f"candidate SHA drift in validation row: {call_id}")

        research_calls = {
            call_id
            for call_id, response in responses.items()
            if response.get("nonresearch") is not True
        }
        research_outcomes: dict[str, tuple[bool | None, dict[str, Any]]] = {}
        for call_id in sorted(research_calls):
            outcome = _derive_terminal_success(
                call_id, responses[call_id], verdicts.get(call_id)
            )
            if outcome[0] is None and responses[call_id].get("mode") != "shift":
                raise SupplementExportError(
                    f"{call_id}: inconclusive validation is only admissible in shift mode"
                )
            research_outcomes[call_id] = outcome
        normalized_by_label: dict[str, dict[str, dict[str, Any]]] = {}
        for label, path in (("full85", inputs.full_rows), ("clean75", inputs.clean_rows)):
            rows = _read_jsonl(path)
            row_ids = [str(row.get("call_id") or "") for row in rows]
            if len(row_ids) != len(set(row_ids)) or set(row_ids) != research_calls:
                raise SupplementExportError(
                    f"{label} normalized rows do not match the final research ledger"
                )
            normalized_by_label[label] = {
                str(row["call_id"]): row for row in rows
            }

        all_shift_responses = {
            call_id: response
            for call_id, response in responses.items()
            if response.get("mode") == "shift"
        }
        if any(
            response.get("nonresearch") is True
            for response in all_shift_responses.values()
        ):
            raise SupplementExportError(
                "distribution-shift responses cannot be excluded as nonresearch"
            )
        shift_responses = all_shift_responses
        if len(shift_responses) != 100:
            raise SupplementExportError(
                "distribution-shift ledger does not contain exactly 100 responses"
            )
        cases_by_arm: dict[str, set[str]] = {arm: set() for arm in _SHIFT_ARMS}
        shift_by_case_arm: dict[tuple[str, str], str] = {}
        for call_id, response in shift_responses.items():
            arm = str(response.get("arm") or "")
            case_id = str(response.get("case_id") or "")
            cluster_id = str(response.get("cluster_id") or "")
            if arm not in cases_by_arm or not case_id or not cluster_id:
                raise SupplementExportError(
                    f"{call_id}: invalid distribution-shift response identity"
                )
            key = (case_id, arm)
            if key in shift_by_case_arm:
                raise SupplementExportError(
                    f"duplicate distribution-shift case/arm response: {case_id}/{arm}"
                )
            shift_by_case_arm[key] = call_id
            cases_by_arm[arm].add(case_id)
        if (
            any(len(cases_by_arm[arm]) != 50 for arm in _SHIFT_ARMS)
            or cases_by_arm[_SHIFT_ARMS[0]] != cases_by_arm[_SHIFT_ARMS[1]]
        ):
            raise SupplementExportError(
                "distribution-shift responses are not the same 50 cases in both arms"
            )
        for case_id in cases_by_arm[_SHIFT_ARMS[0]]:
            clusters = {
                str(
                    shift_responses[shift_by_case_arm[(case_id, arm)]].get(
                        "cluster_id"
                    )
                )
                for arm in _SHIFT_ARMS
            }
            if len(clusters) != 1:
                raise SupplementExportError(
                    f"distribution-shift seed mismatch across arms: {case_id}"
                )

        shift_outcomes: dict[str, tuple[bool | None, dict[str, Any]]] = {}
        per_arm_counts: dict[str, dict[str, Any]] = {
            arm: {
                "definitive_calls": 0,
                "unresolved_calls": 0,
                "reason_counts": {},
                "status_counts": {},
            }
            for arm in _SHIFT_ARMS
        }
        for call_id, response in shift_responses.items():
            outcome, metadata = research_outcomes[call_id]
            shift_outcomes[call_id] = (outcome, metadata)
            arm_counts = per_arm_counts[str(response["arm"])]
            if type(outcome) is bool:
                arm_counts["definitive_calls"] += 1
            else:
                arm_counts["unresolved_calls"] += 1
                reason = str(metadata["reason"])
                reasons = arm_counts["reason_counts"]
                reasons[reason] = reasons.get(reason, 0) + 1
                status = (
                    f"formal_{str(metadata['formal_status']).lower()}__"
                    f"simulation_{str(metadata['simulation_status']).lower()}"
                )
                statuses = arm_counts["status_counts"]
                statuses[status] = statuses.get(status, 0) + 1
        definitive_calls = sum(
            values["definitive_calls"] for values in per_arm_counts.values()
        )
        unresolved_calls = sum(
            values["unresolved_calls"] for values in per_arm_counts.values()
        )
        if definitive_calls != 86 or unresolved_calls != 14:
            raise SupplementExportError(
                "distribution-shift terminal ledger coverage is not exactly 86 definitive "
                "+ 14 unresolved"
            )
        fully_definitive_pairs = sum(
            all(
                type(shift_outcomes[shift_by_case_arm[(case_id, arm)]][0]) is bool
                for arm in _SHIFT_ARMS
            )
            for case_id in cases_by_arm[_SHIFT_ARMS[0]]
        )
        for label, rows_by_call in normalized_by_label.items():
            for call_id, (outcome, metadata) in shift_outcomes.items():
                _require_normalized_shift_row(
                    label=label,
                    row=rows_by_call[call_id],
                    response=shift_responses[call_id],
                    expected_success=outcome,
                    metadata=metadata,
                )

        full_shift = full_analysis["secondaries"]["distribution_shift"]
        clean_shift = clean_analysis["secondaries"]["distribution_shift"]
        if full_shift != clean_shift:
            raise SupplementExportError(
                "full85/clean75 distribution-shift coverage summaries differ"
            )
        expected_coverage = {
            "expected_cases": 50,
            "expected_calls": 100,
            "observed_calls": 100,
            "definitive_calls": definitive_calls,
            "unresolved_calls": unresolved_calls,
            "definitive_coverage": definitive_calls / 100,
            "fully_definitive_pair_count": fully_definitive_pairs,
        }
        for key, value in expected_coverage.items():
            if full_shift.get(key) != value:
                raise SupplementExportError(
                    f"distribution-shift statistics contradict ledger coverage: {key}"
                )
        aggregate_reasons: dict[str, int] = {}
        aggregate_statuses: dict[str, int] = {}
        for arm in _SHIFT_ARMS:
            stats_arm = full_shift["per_arm"][arm]
            derived_arm = per_arm_counts[arm]
            expected_arm = {
                "expected_calls": 50,
                "observed_calls": 50,
                "definitive_calls": derived_arm["definitive_calls"],
                "unresolved_calls": derived_arm["unresolved_calls"],
                "definitive_coverage": derived_arm["definitive_calls"] / 50,
                "reason_counts": dict(sorted(derived_arm["reason_counts"].items())),
                "status_counts": dict(sorted(derived_arm["status_counts"].items())),
            }
            if any(stats_arm.get(key) != value for key, value in expected_arm.items()):
                raise SupplementExportError(
                    f"distribution-shift statistics contradict {arm} ledger coverage"
                )
            for reason, count in derived_arm["reason_counts"].items():
                aggregate_reasons[reason] = aggregate_reasons.get(reason, 0) + count
            for status, count in derived_arm["status_counts"].items():
                aggregate_statuses[status] = aggregate_statuses.get(status, 0) + count
        if full_shift.get("reason_counts") != dict(sorted(aggregate_reasons.items())):
            raise SupplementExportError(
                "distribution-shift statistics contradict ledger reason counts"
            )
        if full_shift.get("status_counts") != dict(sorted(aggregate_statuses.items())):
            raise SupplementExportError(
                "distribution-shift statistics contradict ledger status counts"
            )
        analysis_coverage = dict(full_shift)
    return {
        "calibration": calibration,
        "formal_gate": formal_gate,
        "calibration_results": calibration_results,
        "fallback_report": fallback_report,
        "amendment": amendment,
        "redaction_decision": redaction_decision,
        "events": events,
        "response_count": len(responses),
        "verdict_count": len(verdicts),
        "analysis_coverage": analysis_coverage,
    }


def _resolve_recorded_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.startswith("/work/"):
        return ROOT / text[len("/work/") :]
    candidate = Path(text)
    if candidate.is_absolute():
        try:
            candidate.relative_to(ROOT)
        except ValueError:
            return None
        return candidate
    return ROOT / candidate


def _record_source(
    records: list[dict[str, Any]],
    *,
    logical_name: str,
    source: Path,
    destination: Path,
) -> None:
    records.append(
        {
            "logical_name": logical_name,
            "public_path": destination.as_posix(),
            "source_sha256": _sha256_path(source),
        }
    )


def _copy_json_document(
    source: Path,
    destination: Path,
    public_root: Path,
    sanitizer: _Sanitizer,
    records: list[dict[str, Any]],
    logical_name: str,
) -> None:
    document = sanitizer.value(_read_json(source))
    assert isinstance(document, dict)
    # An anonymized document cannot truthfully retain a self-digest computed
    # over its private form. Preserve that frozen digest under an explicit
    # source_* name, then self-authenticate the public derivative separately.
    for signature in ("manifest_sha256", "analysis_sha256", "gate_manifest_sha256"):
        if signature in document:
            document[f"source_{signature}"] = document.pop(signature)
    document["public_document_sha256"] = _sha256_bytes(_canonical_bytes(document))
    _write_json(destination, document)
    _record_source(
        records,
        logical_name=logical_name,
        source=source,
        destination=destination.relative_to(public_root),
    )
    records[-1]["public_sha256"] = _sha256_path(destination)


def _copy_jsonl_document(
    source: Path,
    destination: Path,
    public_root: Path,
    sanitizer: _Sanitizer,
    records: list[dict[str, Any]],
    logical_name: str,
) -> None:
    rows = [sanitizer.value(row) for row in _read_jsonl(source)]
    _write_jsonl(destination, rows)
    _record_source(
        records,
        logical_name=logical_name,
        source=source,
        destination=destination.relative_to(public_root),
    )
    records[-1]["public_sha256"] = _sha256_path(destination)


def _copy_extra_text(
    artifact: NamedArtifact,
    staging: Path,
    sanitizer: _Sanitizer,
    records: list[dict[str, Any]],
) -> None:
    source = artifact.path
    base = staging / "artifacts" / _safe_name(artifact.name)
    files = [source] if source.is_file() else sorted(path for path in source.rglob("*") if path.is_file())
    if not source.exists() or (not source.is_file() and not source.is_dir()):
        raise SupplementExportError(f"additional artifact is missing: {artifact.name}")
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError) as exc:
            raise SupplementExportError(
                f"additional artifact must be UTF-8 text: {artifact.name}/{path.name}"
            ) from exc
        relative = Path(path.name) if source.is_file() else path.relative_to(source)
        if any(part in {"..", ".git"} for part in relative.parts):
            raise SupplementExportError(f"unsafe additional artifact path: {relative}")
        destination = base / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(sanitizer.text(text), encoding="utf-8")
        _record_source(
            records,
            logical_name=f"additional:{artifact.name}:{relative.as_posix()}",
            source=path,
            destination=destination.relative_to(staging),
        )
        records[-1]["public_sha256"] = _sha256_path(destination)


def _scan_staging(staging: Path, sanitizer: _Sanitizer) -> None:
    violations: list[str] = []
    for path in sorted(staging.rglob("*")):
        if path.is_symlink():
            violations.append(f"symlink:{path.relative_to(staging)}")
            continue
        if not path.is_file():
            continue
        relative = path.relative_to(staging)
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            violations.append(f"binary:{relative}")
            continue
        for pattern in _CREDENTIAL_PATTERNS:
            if pattern.search(text):
                violations.append(f"credential:{relative}")
                break
        if str(ROOT) in text or re.search(r"(?:/Users/|/home/|/work/|/private/tmp/)", text):
            violations.append(f"absolute-path:{relative}")
        if any(pattern.search(text) for pattern in _GIT_REMOTE_PATTERNS):
            violations.append(f"git-remote:{relative}")
        if _EMAIL_RE.search(text):
            violations.append(f"email:{relative}")
        for token in sanitizer.author_tokens:
            if re.search(re.escape(token), text, flags=re.IGNORECASE):
                violations.append(f"author-token:{relative}")
                break
        if any(pattern.search(text) for pattern in _INTERNAL_ID_PATTERNS):
            violations.append(f"internal-id:{relative}")
    if violations:
        raise SupplementExportError(
            "anonymous supplement scan failed: " + ", ".join(sorted(set(violations))[:12])
        )


def _deterministic_tar_gz(staging: Path, archive: Path) -> None:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.PAX_FORMAT) as tar:
        for path in sorted(staging.rglob("*"), key=lambda item: item.relative_to(staging).as_posix()):
            relative = Path("vericodegen_anonymous_supplement") / path.relative_to(staging)
            info = tarfile.TarInfo(relative.as_posix())
            info.mtime = 0
            info.uid = 0
            info.gid = 0
            info.uname = ""
            info.gname = ""
            if path.is_dir():
                info.type = tarfile.DIRTYPE
                info.mode = 0o755
                tar.addfile(info)
            elif path.is_file():
                payload = path.read_bytes()
                info.size = len(payload)
                info.mode = 0o644
                tar.addfile(info, io.BytesIO(payload))
    archive.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(prefix=archive.name + ".tmp-", dir=archive.parent)
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("wb") as raw:
            with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as compressed:
                compressed.write(buffer.getvalue())
        os.replace(temporary, archive)
    finally:
        if temporary.exists():
            temporary.unlink()


def export_supplement(
    inputs: SupplementInputs,
    staging: Path,
    *,
    archive: Path | None = None,
    strict: bool = True,
    force: bool = False,
    redact_tokens: Sequence[str] = (),
) -> dict[str, Any]:
    """Create an anonymous staging tree and optional deterministic tarball."""

    state = _validate_final_inputs(inputs, strict=strict)
    if staging.exists() and not force:
        raise SupplementExportError(f"staging path already exists: {staging}")
    if archive is not None and archive.exists() and not force:
        raise SupplementExportError(f"archive path already exists: {archive}")
    staging.parent.mkdir(parents=True, exist_ok=True)
    identifier_map = _identifier_map(inputs.identifier_sources)
    sanitizer = _Sanitizer(identifier_map, redact_tokens=redact_tokens)
    temporary = Path(
        tempfile.mkdtemp(prefix=staging.name + ".tmp-", dir=staging.parent)
    )
    source_records: list[dict[str, Any]] = []
    try:
        _copy_json_document(
            inputs.config,
            temporary / "config" / "study_config.json",
            temporary,
            sanitizer,
            source_records,
            "study_config",
        )
        _copy_json_document(
            inputs.protocol_amendment,
            temporary / "protocol" / "pre_output_amendment.json",
            temporary,
            sanitizer,
            source_records,
            "pre_output_protocol_amendment",
        )
        _copy_json_document(
            inputs.redaction_decision,
            temporary / "protocol" / "redaction_cancellation.json",
            temporary,
            sanitizer,
            source_records,
            "redaction_cancellation",
        )
        for index, manifest in enumerate(inputs.manifests):
            if not manifest.exists():
                continue
            destination = temporary / "manifests" / f"{index:02d}_{_safe_name(manifest.name)}"
            _copy_json_document(
                manifest,
                destination,
                temporary,
                sanitizer,
                source_records,
                f"manifest:{manifest.name}",
            )

        public_events = [
            _public_ledger_event(event, sanitizer) for event in state["events"]
        ]
        public_ledger = temporary / "ledger" / "public_events.jsonl"
        _write_jsonl(public_ledger, public_events)
        _record_source(
            source_records,
            logical_name="experiment_ledger",
            source=inputs.ledger,
            destination=public_ledger.relative_to(temporary),
        )
        source_records[-1]["public_sha256"] = _sha256_path(public_ledger)

        _copy_json_document(
            inputs.calibration_gate,
            temporary / "formal" / "calibration_gate.json",
            temporary,
            sanitizer,
            source_records,
            "calibration_gate",
        )
        _copy_json_document(
            inputs.formal_primary_gate,
            temporary / "formal" / "formal_primary_gate.json",
            temporary,
            sanitizer,
            source_records,
            "formal_primary_gate",
        )
        _copy_jsonl_document(
            inputs.validation_results,
            temporary / "formal" / "validation_results.jsonl",
            temporary,
            sanitizer,
            source_records,
            "validation_results",
        )
        calibration_results = state["calibration_results"]
        if calibration_results is not None and calibration_results.is_file():
            _copy_jsonl_document(
                calibration_results,
                temporary / "formal" / "calibration_results.jsonl",
                temporary,
                sanitizer,
                source_records,
                "calibration_results",
            )
        elif strict:
            raise SupplementExportError("final calibration results JSONL is missing")
        fallback_report = state["fallback_report"]
        if fallback_report is not None and fallback_report.is_file():
            _copy_json_document(
                fallback_report,
                temporary / "formal" / "calibration_fallback_simulation.json",
                temporary,
                sanitizer,
                source_records,
                "calibration_fallback_simulation",
            )
        elif strict:
            raise SupplementExportError("final calibration fallback report is missing")

        for source, destination, logical in (
            (inputs.full_stats, temporary / "analysis" / "full85_stats.json", "full85_stats"),
            (inputs.clean_stats, temporary / "analysis" / "clean75_stats.json", "clean75_stats"),
        ):
            _copy_json_document(
                source, destination, temporary, sanitizer, source_records, logical
            )
        for source, destination, logical in (
            (inputs.full_rows, temporary / "analysis" / "full85_rows.jsonl", "full85_rows"),
            (inputs.clean_rows, temporary / "analysis" / "clean75_rows.jsonl", "clean75_rows"),
        ):
            _copy_jsonl_document(
                source, destination, temporary, sanitizer, source_records, logical
            )

        for artifact in inputs.additional_artifacts:
            _copy_extra_text(artifact, temporary, sanitizer, source_records)

        inventory = {
            "schema_version": EXPORT_SCHEMA_VERSION,
            "kind": "vericodegen_anonymous_supplement_inventory",
            "exporter_version": EXPORTER_VERSION,
            "study_id": sanitizer.value(
                _read_json(inputs.config).get("study_id", "vericodegen-2026")
            ),
            "strict_final_export": strict,
            "privacy": {
                "environment_credentials_read": False,
                "git_configuration_read": False,
                "raw_response_bodies_included": False,
                "candidate_text_in_public_ledger": False,
                "candidate_sha256_preserved": True,
                "absolute_paths_removed": True,
                "internal_task_ids_pseudonymized": True,
            },
            "gates": {
                "strict_all_formal_calibration_passed": False,
                "composite_calibration_passed": state["calibration"].get("passed"),
                "formal_calibration_counterexamples": state["calibration"].get(
                    "replayable_mutant_counterexamples"
                ),
                "simulation_calibration_fallbacks": state["calibration"].get(
                    "fallback_mutant_counterexamples"
                ),
                "formal_primary_passed": state["formal_gate"].get("passed"),
                "formal_reporting_mode": state["formal_gate"].get("reporting_mode"),
            },
            "ledger": {
                "events": len(state["events"]),
                "responses": state["response_count"],
                "verdicts": state["verdict_count"],
            },
            "analysis_coverage": {
                "distribution_shift": sanitizer.value(state["analysis_coverage"]),
            },
            "source_artifacts": sorted(source_records, key=lambda row: row["public_path"]),
        }
        _write_json(temporary / "inventory.json", inventory)
        (temporary / "README.md").write_text(
            "# Anonymous VeriCodeGen supplement\n\n"
            "This tree was generated by `benchmarks.export_vericodegen_supplement`. "
            "`inventory.json` records the SHA-256 of every private source artifact and "
            "its sanitized public derivative. The public ledger contains exact prompts, "
            "request/response metadata, candidate SHA-256 values, and terminal formal and "
            "independent-simulation verdicts. Raw provider response bodies, candidate text, "
            "credentials, local paths, repository remotes, and author identifiers are not "
            "included.\n",
            encoding="utf-8",
        )
        _scan_staging(temporary, sanitizer)
        if staging.exists():
            shutil.rmtree(staging)
        os.replace(temporary, staging)
        if archive is not None:
            if archive.exists():
                archive.unlink()
            _deterministic_tar_gz(staging, archive)
        result = {
            "staging": str(staging),
            "inventory_sha256": _sha256_path(staging / "inventory.json"),
            "file_count": sum(path.is_file() for path in staging.rglob("*")),
            "archive": str(archive) if archive is not None else None,
            "archive_sha256": _sha256_path(archive) if archive is not None else None,
        }
        return result
    except Exception:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def _artifact_argument(value: str) -> NamedArtifact:
    if "=" not in value:
        raise argparse.ArgumentTypeError("artifact must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("artifact must be NAME=PATH")
    return NamedArtifact(name.strip(), Path(raw_path.strip()))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--protocol-amendment", type=Path, default=DEFAULT_PROTOCOL_AMENDMENT
    )
    parser.add_argument(
        "--redaction-decision", type=Path, default=DEFAULT_REDACTION_DECISION
    )
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--calibration-gate", type=Path, default=DEFAULT_CALIBRATION_GATE)
    parser.add_argument("--calibration-results", type=Path)
    parser.add_argument("--formal-primary-gate", type=Path, default=DEFAULT_FORMAL_PRIMARY_GATE)
    parser.add_argument("--validation-results", type=Path, default=DEFAULT_VALIDATION_RESULTS)
    parser.add_argument("--full-stats", type=Path, default=DEFAULT_FULL_STATS)
    parser.add_argument("--clean-stats", type=Path, default=DEFAULT_CLEAN_STATS)
    parser.add_argument("--full-rows", type=Path, default=DEFAULT_FULL_ROWS)
    parser.add_argument("--clean-rows", type=Path, default=DEFAULT_CLEAN_ROWS)
    parser.add_argument(
        "--manifest",
        type=Path,
        action="append",
        help="override the default frozen manifest set (repeatable)",
    )
    parser.add_argument(
        "--identifier-source",
        type=Path,
        action="append",
        help="materialized JSONL used only to pseudonymize internal IDs",
    )
    parser.add_argument("--artifact", type=_artifact_argument, action="append", default=[])
    parser.add_argument("--staging", type=Path, default=DEFAULT_STAGING)
    parser.add_argument("--archive", type=Path)
    parser.add_argument(
        "--redact-token",
        action="append",
        default=[],
        help="additional author/organization string to remove (repeatable)",
    )
    parser.add_argument("--draft", action="store_true", help="allow missing/nonfinal gates for a privacy dry run")
    parser.add_argument("--force", action="store_true", help="replace an existing staging tree/archive")
    args = parser.parse_args(argv)

    inputs = SupplementInputs(
        config=args.config,
        protocol_amendment=args.protocol_amendment,
        redaction_decision=args.redaction_decision,
        ledger=args.ledger,
        calibration_gate=args.calibration_gate,
        calibration_results=args.calibration_results,
        formal_primary_gate=args.formal_primary_gate,
        validation_results=args.validation_results,
        full_stats=args.full_stats,
        clean_stats=args.clean_stats,
        full_rows=args.full_rows,
        clean_rows=args.clean_rows,
        manifests=tuple(args.manifest) if args.manifest else DEFAULT_MANIFESTS,
        identifier_sources=(
            tuple(args.identifier_source)
            if args.identifier_source
            else DEFAULT_IDENTIFIER_SOURCES
        ),
        additional_artifacts=tuple(args.artifact),
    )
    try:
        result = export_supplement(
            inputs,
            args.staging,
            archive=args.archive,
            strict=not args.draft,
            force=args.force,
            redact_tokens=args.redact_token,
        )
    except SupplementExportError as exc:
        parser.error(str(exc))
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
