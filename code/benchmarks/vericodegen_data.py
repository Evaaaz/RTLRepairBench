"""Freeze the non-model data artifacts for the VeriCodeGen study.

This module deliberately performs no inference and no verification.  It turns
the existing, ignored JSONL data files into small, reviewable manifests that
pin the distribution-shift sample, the clean75 sensitivity set, and the manual
spec-redaction annotation form.

The three operations are deterministic:

* ``shift`` selects 50 model-generated compile-clean failures with RNG 42,
  one record per task, using the preregistered 20/20/5/5 strata and fallback
  order.
* ``clean75`` intersects the 85 semantic cases with the already-audited
  ``eval_tasks_clean.jsonl`` allowlist.  It never guesses overlap labels.
* ``redaction-template`` creates a blank two-annotator form.  ``redaction-freeze``
  refuses incomplete annotations and records the preregistered 20% go/no-go.

Every manifest records source hashes and has its own canonical SHA-256.  This
allows the runner to reject post-freeze drift without copying the source RTL.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import random
import re
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(os.environ.get("RTLREPAIR_ROOT", "")) if os.environ.get("RTLREPAIR_ROOT") else Path(__file__).resolve().parents[2]
DEFAULT_SEM85 = REPO_ROOT / "data" / "repairbench_sem85.jsonl"
DEFAULT_REALBUGS = REPO_ROOT / "data" / "repairbench_realbugs.jsonl"
DEFAULT_EVAL = REPO_ROOT / "data" / "eval_tasks.jsonl"
DEFAULT_CLEAN_EVAL = REPO_ROOT / "data" / "eval_tasks_clean.jsonl"
DEFAULT_CONFIG_DIR = REPO_ROOT / "configs" / "vericodegen"
DEFAULT_SHIFT_SELECTION = DEFAULT_CONFIG_DIR / "shift50_manifest.json"
DEFAULT_CLEAN75_MANIFEST = DEFAULT_CONFIG_DIR / "clean75_manifest.json"
DEFAULT_MAIN85_INPUT_MANIFEST = DEFAULT_CONFIG_DIR / "main85_inputs.manifest.json"
DEFAULT_REDACTION_TEMPLATE = DEFAULT_CONFIG_DIR / "redaction_annotations.template.json"
DEFAULT_REDACTION_MANIFEST = DEFAULT_CONFIG_DIR / "redaction_manifest.json"
DEFAULT_EXPERIMENT_LEDGER = (
    REPO_ROOT / "generated" / "logs" / "vericodegen2026_events.jsonl"
)

SHIFT_QUOTAS: dict[tuple[str, str], int] = {
    ("base", "combinational"): 20,
    ("base", "sequential"): 20,
    ("adapter", "combinational"): 5,
    ("adapter", "sequential"): 5,
}
REDACTION_LABELS = frozenset({"explicit", "derivable", "absent"})
ANONYMOUS_MODULE_NAME = "TopModule"
_BROKEN_RTL_BLOCK = re.compile(
    r"Broken RTL \(`(?P<filename>[^`]+)`\):\s*"
    r"```(?:systemverilog|verilog)\n(?P<rtl>.*?)```",
    re.DOTALL,
)
_DIFF_HUNK = re.compile(
    r"^@@ -(?P<old_start>\d+)(?:,(?P<old_count>\d+))? "
    r"\+(?P<new_start>\d+)(?:,(?P<new_count>\d+))? @@"
)
_MODULE_DECLARATION = re.compile(
    r"\bmodule\s+([A-Za-z_][A-Za-z0-9_$]*)(?=\s*(?:#\s*\(|\(|;))"
)


def read_jsonl(path: Path | str) -> list[dict[str, Any]]:
    """Read a JSONL file strictly, retaining no blank records."""
    records: list[dict[str, Any]] = []
    source = Path(path)
    with source.open("r", encoding="utf-8") as handle:
        for line_number, raw in enumerate(handle, 1):
            if not raw.strip():
                continue
            try:
                value = json.loads(raw)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{source}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{source}:{line_number}: expected a JSON object")
            records.append(value)
    return records


def sha256_file(path: Path | str) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _with_manifest_sha(manifest: Mapping[str, Any]) -> dict[str, Any]:
    result = dict(manifest)
    result["manifest_sha256"] = canonical_sha256(result)
    return result


def verify_manifest(
    manifest: Mapping[str, Any], *, verify_sources: bool = True
) -> None:
    """Reject manifest-content or recorded-source drift.

    Relative source paths are resolved from the repository root.  This function
    is intentionally side-effect free so experiment runners can call it before
    consuming a frozen ID list.
    """
    # Private/frozen manifests use ``manifest_sha256``.  Anonymous supplement
    # manifests are separately re-signed after identifiers are pseudonymized and
    # therefore use ``public_document_sha256``.  Both are content hashes over the
    # complete document with only the signature field removed; neither form is
    # allowed to bypass source-hash verification below.
    signature_key = (
        "manifest_sha256"
        if isinstance(manifest.get("manifest_sha256"), str)
        else "public_document_sha256"
    )
    expected = manifest.get(signature_key)
    if not isinstance(expected, str):
        raise ValueError("manifest_sha256 or public_document_sha256 is missing")
    unhashed = {key: value for key, value in manifest.items() if key != signature_key}
    actual = canonical_sha256(unhashed)
    if actual != expected:
        label = (
            "manifest SHA"
            if signature_key == "manifest_sha256"
            else "public document SHA"
        )
        raise ValueError(f"{label} mismatch: expected {expected}, computed {actual}")
    if not verify_sources:
        return
    recorded_sources: list[Mapping[str, Any]] = []
    source = manifest.get("source")
    if isinstance(source, Mapping):
        recorded_sources.append(source)
    sources = manifest.get("sources")
    if isinstance(sources, Mapping):
        recorded_sources.extend(
            value for value in sources.values() if isinstance(value, Mapping)
        )
    jsonl = manifest.get("jsonl")
    if isinstance(jsonl, Mapping):
        recorded_sources.append(jsonl)
    for item in recorded_sources:
        raw_path = item.get("path")
        expected_source_sha = item.get("sha256")
        if not isinstance(raw_path, str) or not isinstance(expected_source_sha, str):
            raise ValueError("recorded source requires path and sha256")
        path = Path(raw_path)
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.exists():
            raise ValueError(f"recorded manifest source is missing: {path}")
        actual_source_sha = sha256_file(path)
        if actual_source_sha != expected_source_sha:
            raise ValueError(
                f"source SHA mismatch for {path}: expected {expected_source_sha}, "
                f"computed {actual_source_sha}"
            )


def write_json_atomic(path: Path | str, value: Any) -> None:
    """Write generated manifests atomically so interruption cannot half-freeze one."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, ensure_ascii=False)
            handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_jsonl_atomic(path: Path | str, records: Sequence[Mapping[str, Any]]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
                )
                handle.write("\n")
        os.replace(temporary, destination)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _message_content(record: Mapping[str, Any], role: str) -> str:
    messages = record.get("messages")
    if not isinstance(messages, list):
        raise ValueError(f"record {record.get('id')!r} lacks messages")
    matches = [
        message.get("content")
        for message in messages
        if isinstance(message, Mapping) and message.get("role") == role
    ]
    if len(matches) != 1 or not isinstance(matches[0], str):
        raise ValueError(f"record {record.get('id')!r} needs exactly one {role} message")
    return matches[0]


def extract_broken_rtl(record: Mapping[str, Any]) -> str:
    match = _BROKEN_RTL_BLOCK.search(_message_content(record, "user"))
    if not match:
        raise ValueError(f"record {record.get('id')!r} has no Broken RTL block")
    return match.group("rtl").replace("\r\n", "\n").replace("\r", "\n")


def oracle_location_from_diff(broken_rtl: str, repair_diff: str) -> dict[str, Any]:
    """Derive the one broken source line and 1-based number from a strict diff."""
    broken_lines = broken_rtl.splitlines()
    patch_lines = repair_diff.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    deletions: list[tuple[int, str]] = []
    index = 0
    while index < len(patch_lines):
        match = _DIFF_HUNK.match(patch_lines[index])
        if not match:
            index += 1
            continue
        old_line = int(match.group("old_start"))
        old_seen = 0
        expected_old = int(match.group("old_count") or "1")
        index += 1
        while index < len(patch_lines) and not _DIFF_HUNK.match(patch_lines[index]):
            line = patch_lines[index]
            if line.startswith(("--- ", "+++ ")):
                break
            index += 1
            if line.startswith("\\ No newline at end of file"):
                continue
            if not line or line[0] not in " +-":
                continue
            marker, payload = line[0], line[1:]
            if marker == "-":
                deletions.append((old_line, payload))
            if marker in " -":
                old_line += 1
                old_seen += 1
        if old_seen != expected_old:
            raise ValueError(
                f"diff old-count mismatch: header {expected_old}, body {old_seen}"
            )
    if len(deletions) != 1:
        raise ValueError(f"expected exactly one deleted broken line, found {len(deletions)}")
    line_number, broken_line = deletions[0]
    if line_number < 1 or line_number > len(broken_lines):
        raise ValueError(f"diff points outside broken RTL at line {line_number}")
    if broken_lines[line_number - 1] != broken_line:
        raise ValueError(
            f"diff line {line_number} does not match broken RTL: "
            f"{broken_line!r} vs {broken_lines[line_number - 1]!r}"
        )
    return {"line_number": line_number, "broken_line": broken_line}


def mutation_target_from_diff(
    broken_rtl: str, repair_diff: str
) -> dict[str, Any]:
    """Return the exact one-line before/after target shown only to annotators.

    The behavior-redaction task is impossible to perform from the prose spec
    alone: annotators must know which behavior the committed mutation changed.
    This helper extracts that context from the same committed unified diff used
    to recover the canonical golden.  It is deliberately kept out of
    runner-facing ``model_input`` records.
    """

    broken_location = oracle_location_from_diff(broken_rtl, repair_diff)
    patch_lines = repair_diff.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    additions: list[tuple[int, str]] = []
    index = 0
    while index < len(patch_lines):
        match = _DIFF_HUNK.match(patch_lines[index])
        if not match:
            index += 1
            continue
        new_line = int(match.group("new_start"))
        new_seen = 0
        expected_new = int(match.group("new_count") or "1")
        index += 1
        while index < len(patch_lines) and not _DIFF_HUNK.match(patch_lines[index]):
            line = patch_lines[index]
            if line.startswith(("--- ", "+++ ")):
                break
            index += 1
            if line.startswith("\\ No newline at end of file"):
                continue
            if not line or line[0] not in " +-":
                continue
            marker, payload = line[0], line[1:]
            if marker == "+":
                additions.append((new_line, payload))
            if marker in " +":
                new_line += 1
                new_seen += 1
        if new_seen != expected_new:
            raise ValueError(
                f"diff new-count mismatch: header {expected_new}, body {new_seen}"
            )
    if len(additions) != 1:
        raise ValueError(f"expected exactly one added canonical line, found {len(additions)}")
    canonical_line_number, canonical_line = additions[0]
    return {
        "broken_line_number": broken_location["line_number"],
        "broken_line": broken_location["broken_line"],
        "canonical_line_number": canonical_line_number,
        "canonical_line": canonical_line,
    }


def _anonymize_module(
    source: str, identifiers: Sequence[str], *, infer_declaration: bool = True
) -> str:
    declaration = _MODULE_DECLARATION.search(source) if infer_declaration else None
    names = {identifier for identifier in identifiers if identifier}
    if declaration:
        names.add(declaration.group(1))
    anonymized = source
    for name in sorted(names, key=len, reverse=True):
        anonymized = re.sub(r"\b" + re.escape(name) + r"\b", ANONYMOUS_MODULE_NAME, anonymized)
    return anonymized


def _normalize_display_rtl(source: str) -> tuple[str, int]:
    """Remove transport-only edge blank lines and return leading-line offset."""
    lines = source.replace("\r\n", "\n").replace("\r", "\n").splitlines()
    leading = 0
    while lines and not lines[0].strip():
        lines.pop(0)
        leading += 1
    while lines and not lines[-1].strip():
        lines.pop()
    if not lines:
        raise ValueError("RTL is empty after display normalization")
    return "\n".join(lines) + "\n", leading


def _eval_specs(eval_records: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    specs: dict[str, str] = {}
    for record in eval_records:
        eval_id = str(record.get("id", "")).strip()
        instruction = str(record.get("instruction", "")).strip()
        if eval_id and instruction:
            specs[eval_id] = instruction
    return specs


def _materialized_case(
    *, internal_metadata: Mapping[str, Any], model_input: Mapping[str, Any]
) -> dict[str, Any]:
    record = {
        "schema_version": 1,
        "internal_metadata": dict(internal_metadata),
        "model_input": dict(model_input),
    }
    record["materialized_case_sha256"] = canonical_sha256(record)
    return record


def build_main85_inputs(
    sem85_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    clean75_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Build runner-ready main cases without embedding a golden or correct line."""
    verify_manifest(clean75_manifest, verify_sources=False)
    clean_ids = {
        str(item.get("case_id"))
        for item in clean75_manifest.get("cases", [])
        if isinstance(item, Mapping)
    }
    if len(sem85_records) != 85 or len(clean_ids) != 75:
        raise ValueError("main input build requires sem85 and the frozen clean75 manifest")
    specs = _eval_specs(eval_records)
    outputs: list[dict[str, Any]] = []
    for ordinal, record in enumerate(
        sorted(sem85_records, key=lambda item: str(item.get("id")))
    ):
        case_id = str(record.get("id", "")).strip()
        seed_id = str(record.get("seed_id", "")).strip()
        mutation = str(record.get("mutation", "")).strip()
        if not case_id or not seed_id or not mutation:
            raise ValueError("main case lacks id, seed_id, or mutation")
        full_spec = specs.get(seed_to_eval_id(seed_id), "").strip()
        if not full_spec:
            raise ValueError(f"{case_id}: no joined full spec")
        broken = extract_broken_rtl(record)
        location = oracle_location_from_diff(broken, _message_content(record, "assistant"))
        display_broken, leading_blank_lines = _normalize_display_rtl(broken)
        display_line_number = location["line_number"] - leading_blank_lines
        if display_line_number < 1:
            raise ValueError(f"{case_id}: oracle line was removed as transport whitespace")
        anonymized_broken = _anonymize_module(
            display_broken, (seed_id, seed_id.removesuffix("_ref"))
        )
        anonymized_line = _anonymize_module(
            location["broken_line"], (seed_id, seed_id.removesuffix("_ref"))
        )
        anonymized_spec = _anonymize_module(
            full_spec,
            (seed_id, seed_id.removesuffix("_ref")),
            infer_declaration=False,
        )
        leaked_tokens = (seed_id, seed_id.removesuffix("_ref"), case_id, mutation)
        combined_model_text = "\n".join((anonymized_broken, anonymized_spec, anonymized_line))
        if any(token and token in combined_model_text for token in leaked_tokens):
            raise ValueError(f"{case_id}: internal identifier leaked into model_input")
        displayed_lines = anonymized_broken.splitlines()
        if (
            display_line_number > len(displayed_lines)
            or displayed_lines[display_line_number - 1] != anonymized_line
        ):
            raise ValueError(f"{case_id}: displayed oracle line coordinate is inconsistent")
        outputs.append(
            _materialized_case(
                internal_metadata={
                    "case_id": case_id,
                    "anonymous_id": f"main_{ordinal:04d}",
                    "cluster_id": seed_id,
                    "source_seed_id": seed_id,
                    "mutation_family": mutation,
                    "clean75": case_id in clean_ids,
                },
                model_input={
                    "expected_module_name": ANONYMOUS_MODULE_NAME,
                    "broken_rtl": anonymized_broken,
                    "full_spec": anonymized_spec,
                    "oracle_location": {
                        "line_number": display_line_number,
                        "broken_line": anonymized_line,
                    },
                },
            )
        )
    if len({item["internal_metadata"]["case_id"] for item in outputs}) != 85:
        raise ValueError("main input cases are not unique")
    return outputs


def build_shift50_inputs(
    source_records: Sequence[Mapping[str, Any]],
    selection_manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Resolve the frozen sample and omit the pool's golden_rtl field entirely."""
    verify_manifest(selection_manifest, verify_sources=False)
    if selection_manifest.get("kind") != "vericodegen_distribution_shift_sample":
        raise ValueError("selection is not a shift50 manifest")
    rows_by_id = {str(record.get("id")): record for record in source_records}
    outputs: list[dict[str, Any]] = []
    for ordinal, selected in enumerate(selection_manifest.get("cases", [])):
        record_id = str(selected.get("record_id", ""))
        source_record = rows_by_id.get(record_id)
        if source_record is None:
            raise ValueError(f"selected shift record is missing: {record_id}")
        if canonical_sha256(source_record) != selected.get("source_record_sha256"):
            raise ValueError(f"selected shift source record drifted: {record_id}")
        task = str(source_record.get("task", "")).strip()
        broken = str(source_record.get("broken_rtl", ""))
        full_spec = str(source_record.get("spec", "")).strip()
        if not task or not broken or not full_spec:
            raise ValueError(f"{record_id}: shift source lacks task/broken_rtl/spec")
        display_broken, _ = _normalize_display_rtl(broken)
        anonymized_broken = _anonymize_module(display_broken, (task, record_id))
        anonymized_spec = _anonymize_module(
            full_spec, (task, record_id), infer_declaration=False
        )
        combined_model_text = anonymized_broken + "\n" + anonymized_spec
        if task in combined_model_text or record_id in combined_model_text:
            raise ValueError(f"{record_id}: internal shift identifier leaked into model_input")
        outputs.append(
            _materialized_case(
                internal_metadata={
                    "case_id": record_id,
                    "anonymous_id": f"shift_{ordinal:04d}",
                    "cluster_id": task,
                    "source_task_id": task,
                    "failure_origin": "model_generated_compile_clean",
                    "source_group": selected["source_group"],
                    "source_model": selected["source_model"],
                    "timing": selected["timing"],
                    "mutation_family": None,
                    "source_record_sha256": selected["source_record_sha256"],
                },
                model_input={
                    "expected_module_name": ANONYMOUS_MODULE_NAME,
                    "broken_rtl": anonymized_broken,
                    "full_spec": anonymized_spec,
                    "oracle_location": None,
                    "oracle_location_available": False,
                },
            )
        )
    if len(outputs) != 50 or len(
        {item["internal_metadata"]["cluster_id"] for item in outputs}
    ) != 50:
        raise ValueError("shift materialization requires 50 globally unique tasks")
    return outputs


def build_inputs_manifest(
    *,
    kind: str,
    jsonl_path: Path,
    records: Sequence[Mapping[str, Any]],
    sources: Mapping[str, Path],
) -> dict[str, Any]:
    manifest = {
        "schema_version": 1,
        "kind": kind,
        "case_count": len(records),
        "contains_golden_rtl": False,
        "jsonl": {
            "path": str(
                jsonl_path.relative_to(REPO_ROOT)
                if jsonl_path.is_relative_to(REPO_ROOT)
                else jsonl_path
            ),
            "sha256": sha256_file(jsonl_path),
        },
        "sources": {
            label: {
                "path": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
                "sha256": sha256_file(path),
            }
            for label, path in sources.items()
        },
        "case_index": [
            {
                "case_id": item["internal_metadata"]["case_id"],
                "anonymous_id": item["internal_metadata"]["anonymous_id"],
                "materialized_case_sha256": item["materialized_case_sha256"],
            }
            for item in records
        ],
    }
    return _with_manifest_sha(manifest)


def _source_group(source_model: Any) -> str:
    value = str(source_model).strip().lower()
    if value == "base":
        return "base"
    if value == "tuned" or value.startswith("adapter"):
        return "adapter"
    raise ValueError(f"unsupported source_model {source_model!r}; expected base/tuned/adapter")


def _timing(record: Mapping[str, Any]) -> str:
    taxonomy = record.get("taxonomy")
    if isinstance(taxonomy, str):
        try:
            taxonomy = ast.literal_eval(taxonomy)
        except (SyntaxError, ValueError) as exc:
            raise ValueError(
                f"record {record.get('id')!r} has an invalid string taxonomy"
            ) from exc
    if not isinstance(taxonomy, Mapping) or not isinstance(taxonomy.get("sequential"), bool):
        raise ValueError(f"record {record.get('id')!r} lacks taxonomy.sequential bool")
    return "sequential" if taxonomy["sequential"] else "combinational"


def _shift_candidates(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for record in records:
        record_id = str(record.get("id", "")).strip()
        task = str(record.get("task", "")).strip()
        if not record_id or not task:
            raise ValueError("every shift candidate needs non-empty id and task")
        if record_id in seen_ids:
            raise ValueError(f"duplicate shift candidate id: {record_id}")
        seen_ids.add(record_id)
        source = _source_group(record.get("source_model"))
        timing = _timing(record)
        candidates.append(
            {
                "record_id": record_id,
                "task": task,
                "source_model": str(record.get("source_model")),
                "source_group": source,
                "timing": timing,
                "source_record_sha256": canonical_sha256(record),
            }
        )
    return sorted(candidates, key=lambda item: item["record_id"])


def select_shift_cases(
    records: Sequence[Mapping[str, Any]],
    *,
    rng_seed: int = 42,
    quotas: Mapping[tuple[str, str], int] = SHIFT_QUOTAS,
) -> list[dict[str, Any]]:
    """Select a deterministic, globally task-unique stratified shift sample.

    Scarcer exact strata are allocated first so a task represented by both base
    and adapter outputs cannot accidentally consume an adapter slot.  Within
    each requested stratum the fallback order is exact stratum, same source and
    opposite timing, then any globally remaining task, as preregistered.
    """
    candidates = _shift_candidates(records)
    if len({candidate["task"] for candidate in candidates}) < sum(quotas.values()):
        raise ValueError("fewer unique tasks than requested shift sample size")

    rng = random.Random(rng_seed)
    selected_tasks: set[str] = set()
    selections: list[dict[str, Any]] = []
    quota_items = list(quotas.items())

    def exact_task_count(key: tuple[str, str]) -> int:
        source, timing = key
        return len(
            {
                candidate["task"]
                for candidate in candidates
                if candidate["source_group"] == source and candidate["timing"] == timing
            }
        )

    # Ratio first, then absolute supply, then the declared key: deterministic
    # and protective of the adapter/sequential stratum in the current corpus.
    allocation_order = sorted(
        quota_items,
        key=lambda item: (
            exact_task_count(item[0]) / item[1] if item[1] else float("inf"),
            exact_task_count(item[0]),
            item[0],
        ),
    )

    for (requested_source, requested_timing), quota in allocation_order:
        chosen_for_slot = 0
        stages = (
            (
                "requested_stratum",
                lambda candidate: candidate["source_group"] == requested_source
                and candidate["timing"] == requested_timing,
            ),
            (
                "same_source_other_timing",
                lambda candidate: candidate["source_group"] == requested_source
                and candidate["timing"] != requested_timing,
            ),
            ("global_remaining", lambda candidate: True),
        )
        for stage_name, predicate in stages:
            if chosen_for_slot >= quota:
                break
            by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
            for candidate in candidates:
                if candidate["task"] not in selected_tasks and predicate(candidate):
                    by_task[candidate["task"]].append(candidate)
            tasks = sorted(by_task)
            rng.shuffle(tasks)
            need = quota - chosen_for_slot
            for task in tasks[:need]:
                task_candidates = sorted(by_task[task], key=lambda item: item["record_id"])
                candidate = task_candidates[rng.randrange(len(task_candidates))]
                selected_tasks.add(task)
                chosen_for_slot += 1
                selections.append(
                    {
                        **candidate,
                        "requested_source_group": requested_source,
                        "requested_timing": requested_timing,
                        "selection_stage": stage_name,
                        "requested_slot_index": chosen_for_slot - 1,
                    }
                )
        if chosen_for_slot != quota:
            raise ValueError(
                f"could not fill {requested_source}/{requested_timing}: "
                f"selected {chosen_for_slot} of {quota}"
            )

    declared_order = {key: index for index, (key, _) in enumerate(quota_items)}
    selections.sort(
        key=lambda item: (
            declared_order[(item["requested_source_group"], item["requested_timing"])],
            item["requested_slot_index"],
        )
    )
    if len(selections) != sum(quotas.values()) or len(selected_tasks) != len(selections):
        raise AssertionError("shift sampler violated size or task-uniqueness invariant")
    return selections


def build_shift_manifest(
    records: Sequence[Mapping[str, Any]],
    *,
    input_path: Path | str | None = None,
    rng_seed: int = 42,
    quotas: Mapping[tuple[str, str], int] = SHIFT_QUOTAS,
) -> dict[str, Any]:
    selections = select_shift_cases(records, rng_seed=rng_seed, quotas=quotas)
    requested_counts = {
        f"{source}/{timing}": count for (source, timing), count in quotas.items()
    }
    actual_counts: dict[str, int] = defaultdict(int)
    fallback_counts: dict[str, int] = defaultdict(int)
    for selection in selections:
        actual_counts[f"{selection['source_group']}/{selection['timing']}"] += 1
        fallback_counts[selection["selection_stage"]] += 1
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "vericodegen_distribution_shift_sample",
        "description": "Model-generated compile-clean failures; not real-world bugs.",
        "rng": {"algorithm": "python.random.Random", "seed": rng_seed},
        "one_record_per_task": True,
        "fallback_order": [
            "requested_stratum",
            "same_source_other_timing",
            "global_remaining",
        ],
        "requested_counts": requested_counts,
        "actual_counts": dict(sorted(actual_counts.items())),
        "selection_stage_counts": dict(sorted(fallback_counts.items())),
        "case_count": len(selections),
        "cases": selections,
    }
    if input_path is not None:
        source = Path(input_path)
        manifest["source"] = {
            "path": str(source.relative_to(REPO_ROOT) if source.is_relative_to(REPO_ROOT) else source),
            "sha256": sha256_file(source),
        }
    return _with_manifest_sha(manifest)


def seed_to_eval_id(seed_id: str) -> str:
    suffix = seed_id[:-4] if seed_id.endswith("_ref") else seed_id
    return f"verilogeval:{suffix}"


def build_clean75_manifest(
    sem85_records: Sequence[Mapping[str, Any]],
    clean_eval_records: Sequence[Mapping[str, Any]],
    *,
    sem85_path: Path | str | None = None,
    clean_eval_path: Path | str | None = None,
    expected_full: int = 85,
    expected_clean: int = 75,
) -> dict[str, Any]:
    """Derive clean75 solely from an existing audited KEEP allowlist."""
    clean_eval_ids = {str(record.get("id")) for record in clean_eval_records}
    if not clean_eval_ids or "None" in clean_eval_ids:
        raise ValueError("clean eval allowlist is empty or contains a missing id")
    if len(sem85_records) != expected_full:
        raise ValueError(f"expected {expected_full} semantic cases, got {len(sem85_records)}")

    seen_cases: set[str] = set()
    kept: list[dict[str, str]] = []
    excluded: list[dict[str, str]] = []
    for record in sorted(sem85_records, key=lambda row: str(row.get("id"))):
        case_id = str(record.get("id", "")).strip()
        seed_id = str(record.get("seed_id", "")).strip()
        if not case_id or not seed_id:
            raise ValueError("semantic case requires id and seed_id")
        if case_id in seen_cases:
            raise ValueError(f"duplicate semantic case id: {case_id}")
        seen_cases.add(case_id)
        item = {"case_id": case_id, "seed_id": seed_id, "eval_id": seed_to_eval_id(seed_id)}
        (kept if item["eval_id"] in clean_eval_ids else excluded).append(item)

    if len(kept) != expected_clean:
        raise ValueError(
            f"audited allowlist produced {len(kept)} clean cases, expected {expected_clean}; "
            "do not guess exclusions—refresh or review the upstream audit"
        )
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "kind": "vericodegen_clean75_allowlist",
        "derivation": "intersection with the pre-existing audited eval_tasks_clean KEEP set",
        "full_case_count": len(sem85_records),
        "clean_case_count": len(kept),
        "excluded_case_count": len(excluded),
        "clean_seed_count": len({item["seed_id"] for item in kept}),
        "excluded_seed_count": len({item["seed_id"] for item in excluded}),
        "cases": kept,
        "excluded": excluded,
    }
    sources: dict[str, Any] = {}
    for label, path in (("sem85", sem85_path), ("audited_keep_set", clean_eval_path)):
        if path is not None:
            source = Path(path)
            sources[label] = {
                "path": str(source.relative_to(REPO_ROOT) if source.is_relative_to(REPO_ROOT) else source),
                "sha256": sha256_file(source),
            }
    if sources:
        manifest["sources"] = sources
    return _with_manifest_sha(manifest)


def build_redaction_template(
    sem85_records: Sequence[Mapping[str, Any]],
    eval_records: Sequence[Mapping[str, Any]],
    *,
    sem85_path: Path | str | None = None,
    eval_path: Path | str | None = None,
) -> dict[str, Any]:
    """Create a blank two-annotator form without inventing semantic labels."""
    specs = {str(record.get("id")): str(record.get("instruction", "")) for record in eval_records}
    cases: list[dict[str, Any]] = []
    for record in sorted(sem85_records, key=lambda row: str(row.get("id"))):
        case_id = str(record.get("id", "")).strip()
        seed_id = str(record.get("seed_id", "")).strip()
        eval_id = seed_to_eval_id(seed_id)
        full_spec = specs.get(eval_id, "").strip()
        if not full_spec:
            raise ValueError(f"no full spec found for {case_id} via {eval_id}")
        broken_rtl = extract_broken_rtl(record)
        repair_diff = _message_content(record, "assistant")
        mutation_target = mutation_target_from_diff(broken_rtl, repair_diff)
        blank_annotation = {
            "annotator_id": None,
            "intent_label": None,
            "redactable_without_hint": None,
            "minimal_removed_clause": None,
            "redacted_spec": None,
            "notes": None,
        }
        cases.append(
            {
                "case_id": case_id,
                "seed_id": seed_id,
                "eval_id": eval_id,
                "full_spec": full_spec,
                "full_spec_sha256": canonical_sha256(full_spec),
                "mutation_target": mutation_target,
                "mutation_target_sha256": canonical_sha256(mutation_target),
                "annotations": {
                    "annotator_a": dict(blank_annotation),
                    "annotator_b": dict(blank_annotation),
                },
                "adjudication": dict(blank_annotation),
            }
        )
    template: dict[str, Any] = {
        "schema_version": 1,
        "kind": "vericodegen_spec_redaction_annotations",
        "status": "BLANK_TEMPLATE_NOT_FROZEN",
        "instructions": {
            "blindness": "Both annotators must finish before any model output is inspected.",
            "task": (
                "Use mutation_target only to identify the changed behavior. For "
                "explicit/derivable behavior, copy full_spec and remove exactly one "
                "verbatim minimal determining clause without rewriting other text. "
                "For absent behavior, keep the already-redacted full spec unchanged."
            ),
            "privacy": (
                "mutation_target is human-only audit context and must never be copied into "
                "a model prompt."
            ),
            "labels": {
                "explicit": "The original spec directly states the mutated behavior.",
                "derivable": "The behavior follows from the spec but is not directly stated.",
                "absent": "The intended behavior is not determined by the spec.",
            },
            "redactable_without_hint": (
                "False means no meaningful behavior-redacted spec can be made without "
                "adding a new hint or destroying the task."
            ),
        },
        "case_count": len(cases),
        "cases": cases,
    }
    sources: dict[str, Any] = {}
    for label, path in (("sem85", sem85_path), ("eval_specs", eval_path)):
        if path is not None:
            source = Path(path)
            sources[label] = {
                "path": str(source.relative_to(REPO_ROOT) if source.is_relative_to(REPO_ROOT) else source),
                "sha256": sha256_file(source),
            }
    if sources:
        template["sources"] = sources
    return _with_manifest_sha(template)


def _validate_annotation(
    annotation: Any,
    *,
    full_spec: str,
    case_id: str,
    role: str,
) -> dict[str, Any]:
    if not isinstance(annotation, Mapping):
        raise ValueError(f"{case_id}: {role} must be an object")
    annotator_id = annotation.get("annotator_id")
    label = annotation.get("intent_label")
    redactable = annotation.get("redactable_without_hint")
    if not isinstance(annotator_id, str) or not annotator_id.strip():
        raise ValueError(f"{case_id}: {role}.annotator_id is incomplete")
    if label not in REDACTION_LABELS:
        raise ValueError(f"{case_id}: {role}.intent_label must be one of {sorted(REDACTION_LABELS)}")
    if not isinstance(redactable, bool):
        raise ValueError(f"{case_id}: {role}.redactable_without_hint must be boolean")
    redacted_spec = annotation.get("redacted_spec")
    removed = annotation.get("minimal_removed_clause")
    if redactable:
        if not isinstance(redacted_spec, str) or not redacted_spec.strip():
            raise ValueError(f"{case_id}: {role}.redacted_spec is required when redactable")
        if label == "absent":
            # The full spec already omits the mutated behavior; keeping it
            # unchanged is the correct behavior-redacted control.
            if removed not in (None, ""):
                raise ValueError(f"{case_id}: {role} labeled absent but removed a clause")
            if redacted_spec.strip() != full_spec.strip():
                raise ValueError(
                    f"{case_id}: {role} labeled absent but changed the full spec"
                )
        else:
            if redacted_spec.strip() == full_spec.strip():
                raise ValueError(f"{case_id}: {role}.redacted_spec did not change the full spec")
            if not isinstance(removed, str) or not removed.strip():
                raise ValueError(f"{case_id}: {role}.minimal_removed_clause is required")
            matches_single_deletion = False
            start = 0
            while True:
                position = full_spec.find(removed, start)
                if position < 0:
                    break
                deleted = full_spec[:position] + full_spec[position + len(removed) :]
                if deleted.strip() == redacted_spec.strip():
                    matches_single_deletion = True
                    break
                start = position + 1
            if not matches_single_deletion:
                raise ValueError(
                    f"{case_id}: {role}.redacted_spec must be one exact clause deletion"
                )
    else:
        if redacted_spec not in (None, "") or removed not in (None, ""):
            raise ValueError(f"{case_id}: {role} marked unredactable but supplied redaction text")
    return dict(annotation)


def freeze_redaction_annotations(
    annotation_document: Mapping[str, Any],
    *,
    template_document: Mapping[str, Any] | None = None,
    expected_cases: int = 85,
    max_unredactable_fraction: float = 0.20,
) -> dict[str, Any]:
    """Validate completed annotations and freeze the adjudicated redaction arm."""
    template_sha: str | None = None
    if template_document is not None:
        verify_manifest(template_document)
        template_cases = template_document.get("cases")
        annotation_cases = annotation_document.get("cases")
        if not isinstance(template_cases, list) or not isinstance(annotation_cases, list):
            raise ValueError("template and annotation documents require case lists")
        immutable_fields = (
            "case_id",
            "seed_id",
            "eval_id",
            "full_spec",
            "full_spec_sha256",
            "mutation_target",
            "mutation_target_sha256",
        )
        expected_by_id = {
            str(case.get("case_id")): case
            for case in template_cases
            if isinstance(case, Mapping)
        }
        actual_by_id = {
            str(case.get("case_id")): case
            for case in annotation_cases
            if isinstance(case, Mapping)
        }
        if set(expected_by_id) != set(actual_by_id):
            raise ValueError("annotation case IDs drifted from the blank template")
        for case_id, expected_case in expected_by_id.items():
            actual_case = actual_by_id[case_id]
            if any(actual_case.get(field) != expected_case.get(field) for field in immutable_fields):
                raise ValueError(f"{case_id}: immutable annotation source fields drifted")
        template_sha = str(template_document["manifest_sha256"])
    cases = annotation_document.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise ValueError(f"expected {expected_cases} annotation cases")
    frozen_cases: list[dict[str, Any]] = []
    seen: set[str] = set()
    for case in cases:
        if not isinstance(case, Mapping):
            raise ValueError("redaction case must be an object")
        case_id = str(case.get("case_id", "")).strip()
        full_spec = str(case.get("full_spec", ""))
        if not case_id or not full_spec:
            raise ValueError("redaction case requires case_id and full_spec")
        if case_id in seen:
            raise ValueError(f"duplicate redaction case: {case_id}")
        seen.add(case_id)
        if canonical_sha256(full_spec) != case.get("full_spec_sha256"):
            raise ValueError(f"{case_id}: full_spec SHA mismatch")
        mutation_target = case.get("mutation_target")
        if not isinstance(mutation_target, Mapping):
            raise ValueError(f"{case_id}: mutation_target context is missing")
        required_target_fields = {
            "broken_line_number",
            "broken_line",
            "canonical_line_number",
            "canonical_line",
        }
        if set(mutation_target) != required_target_fields:
            raise ValueError(f"{case_id}: mutation_target context is incomplete")
        if canonical_sha256(mutation_target) != case.get("mutation_target_sha256"):
            raise ValueError(f"{case_id}: mutation_target SHA mismatch")
        annotations = case.get("annotations")
        if not isinstance(annotations, Mapping):
            raise ValueError(f"{case_id}: annotations object is missing")
        a = _validate_annotation(
            annotations.get("annotator_a"), full_spec=full_spec, case_id=case_id, role="annotator_a"
        )
        b = _validate_annotation(
            annotations.get("annotator_b"), full_spec=full_spec, case_id=case_id, role="annotator_b"
        )
        if a["annotator_id"].strip() == b["annotator_id"].strip():
            raise ValueError(f"{case_id}: the two annotations must have distinct annotator_id values")
        adjudication = _validate_annotation(
            case.get("adjudication"), full_spec=full_spec, case_id=case_id, role="adjudication"
        )
        frozen_cases.append(
            {
                "case_id": case_id,
                "seed_id": str(case.get("seed_id")),
                "intent_label": adjudication["intent_label"],
                "redactable_without_hint": adjudication["redactable_without_hint"],
                "full_spec_sha256": case["full_spec_sha256"],
                "mutation_target": dict(mutation_target),
                "mutation_target_sha256": case["mutation_target_sha256"],
                "redacted_spec": adjudication["redacted_spec"],
                "redacted_spec_sha256": (
                    canonical_sha256(adjudication["redacted_spec"])
                    if adjudication["redactable_without_hint"]
                    else None
                ),
            }
        )
    unredactable = sum(not item["redactable_without_hint"] for item in frozen_cases)
    fraction = unredactable / len(frozen_cases)
    decision = "DROP_REDACTED_ARM" if fraction > max_unredactable_fraction else "RUN_REDACTED_ARM"
    eligible_case_ids = [
        item["case_id"] for item in frozen_cases if item["redactable_without_hint"]
    ]
    frozen = {
        "schema_version": 1,
        "kind": "vericodegen_frozen_spec_redactions",
        "status": "FROZEN",
        "case_count": len(frozen_cases),
        "unredactable_count": unredactable,
        "unredactable_fraction": fraction,
        "maximum_allowed_unredactable_fraction": max_unredactable_fraction,
        "arm_decision": decision,
        "eligible_case_count": len(eligible_case_ids),
        "eligible_case_ids": sorted(eligible_case_ids),
        "annotation_document_sha256": canonical_sha256(annotation_document),
        "annotation_template_sha256": template_sha,
        "cases": sorted(frozen_cases, key=lambda item: item["case_id"]),
    }
    return _with_manifest_sha(frozen)


def _case_ids_from_redaction_template(
    template_document: Mapping[str, Any], *, expected_cases: int
) -> set[str]:
    cases = template_document.get("cases")
    if not isinstance(cases, list) or len(cases) != expected_cases:
        raise ValueError(f"redaction template must contain exactly {expected_cases} cases")
    case_ids = {
        str(case.get("case_id") or "")
        for case in cases
        if isinstance(case, Mapping)
    }
    if "" in case_ids or len(case_ids) != expected_cases:
        raise ValueError("redaction template case IDs are missing or duplicated")
    return case_ids


def _case_ids_from_main_input_manifest(
    main_input_manifest: Mapping[str, Any], *, expected_cases: int
) -> set[str]:
    index = main_input_manifest.get("case_index")
    if not isinstance(index, list) or len(index) != expected_cases:
        raise ValueError(f"main input manifest must contain exactly {expected_cases} cases")
    case_ids = {
        str(item.get("case_id") or "")
        for item in index
        if isinstance(item, Mapping)
    }
    if "" in case_ids or len(case_ids) != expected_cases:
        raise ValueError("main input manifest case IDs are missing or duplicated")
    return case_ids


def freeze_redaction_drop_no_annotators(
    template_document: Mapping[str, Any],
    main_input_manifest: Mapping[str, Any],
    *,
    template_path: Path | str,
    main_input_manifest_path: Path | str,
    benchmark_source_path: Path | str,
    ledger_path: Path | str,
    additional_model_output_paths: Sequence[Path | str] = (),
    expected_cases: int = 85,
) -> dict[str, Any]:
    """Freeze a pre-output resource decision to omit the redacted-spec arm.

    This is deliberately *not* an annotation shortcut.  It records that two
    independent human annotators are unavailable, creates no semantic labels,
    and schedules no redacted prompts.  Any pre-existing experiment ledger or
    declared model-output artifact makes the blindness claim unauditable and
    therefore fails closed.
    """

    template_file = Path(template_path).resolve()
    main_manifest_file = Path(main_input_manifest_path).resolve()
    benchmark_file = Path(benchmark_source_path).resolve()
    ledger_file = Path(ledger_path).resolve()
    checked_output_paths = [ledger_file]
    checked_output_paths.extend(Path(path).resolve() for path in additional_model_output_paths)
    existing = sorted({str(path) for path in checked_output_paths if path.exists()})
    if existing:
        raise ValueError(
            "cannot freeze a pre-output redaction drop after a ledger/model output exists: "
            + ", ".join(existing)
        )

    verify_manifest(template_document)
    verify_manifest(main_input_manifest)
    if (
        template_document.get("schema_version") != 1
        or template_document.get("kind") != "vericodegen_spec_redaction_annotations"
        or template_document.get("status") != "BLANK_TEMPLATE_NOT_FROZEN"
        or template_document.get("case_count") != expected_cases
    ):
        raise ValueError("redaction drop requires the untouched blank annotation template")
    if (
        main_input_manifest.get("kind") != "vericodegen_main85_materialized_inputs"
        or main_input_manifest.get("case_count") != expected_cases
    ):
        raise ValueError("redaction drop requires the frozen main85 input manifest")
    template_case_ids = _case_ids_from_redaction_template(
        template_document, expected_cases=expected_cases
    )
    main_case_ids = _case_ids_from_main_input_manifest(
        main_input_manifest, expected_cases=expected_cases
    )
    if template_case_ids != main_case_ids:
        raise ValueError("redaction template and main input manifest case IDs differ")

    template_sources = template_document.get("sources")
    main_sources = main_input_manifest.get("sources")
    if not isinstance(template_sources, Mapping) or not isinstance(main_sources, Mapping):
        raise ValueError("redaction/main manifests lack source provenance")
    template_sem85 = template_sources.get("sem85")
    main_sem85 = main_sources.get("sem85")
    if not isinstance(template_sem85, Mapping) or not isinstance(main_sem85, Mapping):
        raise ValueError("redaction/main manifests lack sem85 provenance")
    benchmark_sha = sha256_file(benchmark_file)
    if (
        template_sem85.get("sha256") != benchmark_sha
        or main_sem85.get("sha256") != benchmark_sha
    ):
        raise ValueError("redaction/main manifests do not bind the current sem85 source")

    def source_record(path: Path) -> dict[str, str]:
        return {
            "path": str(path.relative_to(REPO_ROOT) if path.is_relative_to(REPO_ROOT) else path),
            "sha256": sha256_file(path),
        }

    manifest: dict[str, Any] = {
        "schema_version": 2,
        "kind": "vericodegen_frozen_spec_redactions",
        "status": "FROZEN",
        "arm_decision": "DROP_REDACTED_ARM",
        "decision_basis": "NO_INDEPENDENT_HUMAN_ANNOTATORS",
        "decision_timing": "BEFORE_ANY_MODEL_OUTPUT",
        "model_outputs_seen_before_freeze": False,
        "annotations_performed": False,
        "case_count": expected_cases,
        "eligible_case_count": 0,
        "eligible_case_ids": [],
        # An empty list is intentional: no case was semantically labeled.
        "cases": [],
        "annotation_template_sha256": str(template_document["manifest_sha256"]),
        "main_input_manifest_sha256": str(main_input_manifest["manifest_sha256"]),
        "benchmark_source_sha256": benchmark_sha,
        "case_ids_sha256": canonical_sha256(sorted(template_case_ids)),
        "pre_output_audit": {
            "ledger_path": str(
                ledger_file.relative_to(REPO_ROOT)
                if ledger_file.is_relative_to(REPO_ROOT)
                else ledger_file
            ),
            "ledger_existed": False,
            "response_count": 0,
            "additional_model_output_paths_checked": [
                str(
                    path.relative_to(REPO_ROOT)
                    if path.is_relative_to(REPO_ROOT)
                    else path
                )
                for path in checked_output_paths[1:]
            ],
        },
        "sources": {
            "annotation_template": source_record(template_file),
            "main_input_manifest": source_record(main_manifest_file),
            "benchmark_source": source_record(benchmark_file),
        },
    }
    return _with_manifest_sha(manifest)


def validate_redaction_decision_manifest(
    frozen_manifest: Mapping[str, Any], *, expected_cases: int = 85
) -> None:
    """Validate either a completed human freeze or a schema-v2 resource drop."""

    verify_manifest(frozen_manifest, verify_sources=False)
    if (
        frozen_manifest.get("kind") != "vericodegen_frozen_spec_redactions"
        or frozen_manifest.get("status") != "FROZEN"
        or frozen_manifest.get("case_count") != expected_cases
    ):
        raise ValueError("redaction manifest is not a frozen full-set decision")
    schema_version = frozen_manifest.get("schema_version")
    if schema_version == 2:
        exact = {
            "arm_decision": "DROP_REDACTED_ARM",
            "decision_basis": "NO_INDEPENDENT_HUMAN_ANNOTATORS",
            "decision_timing": "BEFORE_ANY_MODEL_OUTPUT",
            "model_outputs_seen_before_freeze": False,
            "annotations_performed": False,
            "eligible_case_count": 0,
            "eligible_case_ids": [],
            "cases": [],
        }
        if any(frozen_manifest.get(key) != value for key, value in exact.items()):
            raise ValueError("schema-v2 redaction resource-drop fields are inconsistent")
        for field in (
            "annotation_template_sha256",
            "main_input_manifest_sha256",
            "benchmark_source_sha256",
            "case_ids_sha256",
        ):
            if not re.fullmatch(r"[0-9a-f]{64}", str(frozen_manifest.get(field) or "")):
                raise ValueError(f"schema-v2 redaction resource drop lacks {field}")
        audit = frozen_manifest.get("pre_output_audit")
        if (
            not isinstance(audit, Mapping)
            or audit.get("ledger_existed") is not False
            or audit.get("response_count") != 0
        ):
            raise ValueError("schema-v2 redaction pre-output audit is inconsistent")
        return
    if schema_version != 1:
        raise ValueError("unsupported frozen redaction schema version")

    decision = frozen_manifest.get("arm_decision")
    fraction = frozen_manifest.get("unredactable_fraction")
    threshold = frozen_manifest.get("maximum_allowed_unredactable_fraction")
    eligible_ids = frozen_manifest.get("eligible_case_ids")
    eligible_count = frozen_manifest.get("eligible_case_count")
    cases = frozen_manifest.get("cases")
    unredactable_count = frozen_manifest.get("unredactable_count")
    if (
        decision not in {"RUN_REDACTED_ARM", "DROP_REDACTED_ARM"}
        or not isinstance(fraction, (int, float))
        or not isinstance(threshold, (int, float))
        or float(threshold) != 0.20
        or not isinstance(eligible_ids, list)
        or not isinstance(eligible_count, int)
        or eligible_count != len(eligible_ids)
        or len(set(map(str, eligible_ids))) != eligible_count
        or not isinstance(cases, list)
        or len(cases) != expected_cases
        or not isinstance(unredactable_count, int)
    ):
        raise ValueError("frozen human-redaction decision fields are inconsistent")
    case_ids = [
        str(case.get("case_id") or "") for case in cases if isinstance(case, Mapping)
    ]
    derived_eligible = {
        str(case.get("case_id"))
        for case in cases
        if isinstance(case, Mapping) and case.get("redactable_without_hint") is True
    }
    if (
        len(case_ids) != expected_cases
        or "" in case_ids
        or len(set(case_ids)) != expected_cases
        or derived_eligible != set(map(str, eligible_ids))
        or eligible_count + unredactable_count != expected_cases
        or abs(float(fraction) - unredactable_count / expected_cases) > 1e-12
    ):
        raise ValueError("frozen human-redaction case list/counts are inconsistent")
    expected_decision = (
        "DROP_REDACTED_ARM" if float(fraction) > float(threshold) else "RUN_REDACTED_ARM"
    )
    if decision != expected_decision:
        raise ValueError("frozen human-redaction decision violates the 20% rule")


def redacted_spec_overlay(frozen_manifest: Mapping[str, Any]) -> dict[str, str]:
    """Return the exact eligible case->spec mapping for the runner.

    A dropped arm returns an empty mapping. A runnable arm fails closed if any
    eligible case lacks text or if the declared eligible IDs drift from cases.
    """
    declared_case_count = frozen_manifest.get("case_count")
    if not isinstance(declared_case_count, int) or declared_case_count <= 0:
        raise ValueError("frozen redaction manifest lacks a positive case_count")
    validate_redaction_decision_manifest(
        frozen_manifest, expected_cases=declared_case_count
    )
    decision = frozen_manifest.get("arm_decision")
    if decision == "DROP_REDACTED_ARM":
        return {}
    if decision != "RUN_REDACTED_ARM":
        raise ValueError("redaction arm has no valid frozen decision")
    declared = {str(case_id) for case_id in frozen_manifest.get("eligible_case_ids", [])}
    overlay: dict[str, str] = {}
    for item in frozen_manifest.get("cases", []):
        if not isinstance(item, Mapping) or not item.get("redactable_without_hint"):
            continue
        case_id = str(item.get("case_id", ""))
        spec = item.get("redacted_spec")
        if not case_id or not isinstance(spec, str) or not spec.strip():
            raise ValueError("eligible redaction case lacks case_id or redacted_spec")
        if case_id in overlay:
            raise ValueError(f"duplicate redaction overlay case: {case_id}")
        overlay[case_id] = spec
    if set(overlay) != declared or len(overlay) != frozen_manifest.get("eligible_case_count"):
        raise ValueError("redaction overlay does not match frozen eligible IDs/count")
    return dict(sorted(overlay.items()))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    shift = subparsers.add_parser("shift", help="freeze the 50-case shift sample")
    shift.add_argument("--input", type=Path, default=DEFAULT_REALBUGS)
    shift.add_argument("--output", type=Path, default=DEFAULT_CONFIG_DIR / "shift50_manifest.json")
    shift.add_argument("--seed", type=int, default=42)

    clean = subparsers.add_parser("clean75", help="freeze the audited clean75 allowlist")
    clean.add_argument("--sem85", type=Path, default=DEFAULT_SEM85)
    clean.add_argument("--clean-eval", type=Path, default=DEFAULT_CLEAN_EVAL)
    clean.add_argument("--output", type=Path, default=DEFAULT_CONFIG_DIR / "clean75_manifest.json")

    main_inputs = subparsers.add_parser(
        "main-inputs", help="materialize anonymous runner inputs for the main 85 cases"
    )
    main_inputs.add_argument("--sem85", type=Path, default=DEFAULT_SEM85)
    main_inputs.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    main_inputs.add_argument("--clean75", type=Path, default=DEFAULT_CLEAN75_MANIFEST)
    main_inputs.add_argument(
        "--output", type=Path, default=DEFAULT_CONFIG_DIR / "main85_inputs.jsonl"
    )
    main_inputs.add_argument(
        "--manifest", type=Path, default=DEFAULT_CONFIG_DIR / "main85_inputs.manifest.json"
    )

    shift_inputs = subparsers.add_parser(
        "shift-inputs", help="materialize anonymous runner inputs for frozen shift50"
    )
    shift_inputs.add_argument("--input", type=Path, default=DEFAULT_REALBUGS)
    shift_inputs.add_argument("--selection", type=Path, default=DEFAULT_SHIFT_SELECTION)
    shift_inputs.add_argument(
        "--output", type=Path, default=DEFAULT_CONFIG_DIR / "shift50_inputs.jsonl"
    )
    shift_inputs.add_argument(
        "--manifest", type=Path, default=DEFAULT_CONFIG_DIR / "shift50_inputs.manifest.json"
    )

    template = subparsers.add_parser("redaction-template", help="write a blank annotation form")
    template.add_argument("--sem85", type=Path, default=DEFAULT_SEM85)
    template.add_argument("--eval", type=Path, default=DEFAULT_EVAL)
    template.add_argument(
        "--output", type=Path, default=DEFAULT_CONFIG_DIR / "redaction_annotations.template.json"
    )

    freeze = subparsers.add_parser("redaction-freeze", help="validate and freeze completed annotations")
    freeze.add_argument("--annotations", type=Path, required=True)
    freeze.add_argument(
        "--template",
        type=Path,
        default=DEFAULT_CONFIG_DIR / "redaction_annotations.template.json",
    )
    freeze.add_argument(
        "--output", type=Path, default=DEFAULT_CONFIG_DIR / "redaction_manifest.json"
    )

    drop = subparsers.add_parser(
        "redaction-drop-no-annotators",
        help="freeze a pre-output decision to omit redaction when two annotators are unavailable",
    )
    drop.add_argument("--template", type=Path, default=DEFAULT_REDACTION_TEMPLATE)
    drop.add_argument(
        "--main-input-manifest", type=Path, default=DEFAULT_MAIN85_INPUT_MANIFEST
    )
    drop.add_argument("--source", type=Path, default=DEFAULT_SEM85)
    drop.add_argument("--ledger", type=Path, default=DEFAULT_EXPERIMENT_LEDGER)
    drop.add_argument(
        "--model-output",
        action="append",
        type=Path,
        default=[],
        help="additional model-output path whose existence must block the freeze",
    )
    drop.add_argument("--output", type=Path, default=DEFAULT_REDACTION_MANIFEST)

    verify = subparsers.add_parser("verify", help="verify a frozen manifest and its sources")
    verify.add_argument("manifest", type=Path)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "shift":
        if args.seed != 42:
            raise SystemExit("the preregistered shift sample requires --seed 42")
        value = build_shift_manifest(read_jsonl(args.input), input_path=args.input, rng_seed=args.seed)
    elif args.command == "clean75":
        value = build_clean75_manifest(
            read_jsonl(args.sem85),
            read_jsonl(args.clean_eval),
            sem85_path=args.sem85,
            clean_eval_path=args.clean_eval,
        )
    elif args.command == "redaction-template":
        value = build_redaction_template(
            read_jsonl(args.sem85),
            read_jsonl(args.eval),
            sem85_path=args.sem85,
            eval_path=args.eval,
        )
    elif args.command == "redaction-freeze":
        with args.annotations.open("r", encoding="utf-8") as handle:
            document = json.load(handle)
        with args.template.open("r", encoding="utf-8") as handle:
            template_document = json.load(handle)
        value = freeze_redaction_annotations(
            document, template_document=template_document
        )
    elif args.command == "redaction-drop-no-annotators":
        if args.output.exists():
            raise SystemExit(
                f"refusing to overwrite an existing frozen redaction decision: {args.output}"
            )
        with args.template.open("r", encoding="utf-8") as handle:
            template_document = json.load(handle)
        with args.main_input_manifest.open("r", encoding="utf-8") as handle:
            main_input_manifest = json.load(handle)
        value = freeze_redaction_drop_no_annotators(
            template_document,
            main_input_manifest,
            template_path=args.template,
            main_input_manifest_path=args.main_input_manifest,
            benchmark_source_path=args.source,
            ledger_path=args.ledger,
            additional_model_output_paths=args.model_output,
        )
    elif args.command == "main-inputs":
        with args.clean75.open("r", encoding="utf-8") as handle:
            clean_manifest = json.load(handle)
        verify_manifest(clean_manifest)
        records = build_main85_inputs(
            read_jsonl(args.sem85), read_jsonl(args.eval), clean_manifest
        )
        write_jsonl_atomic(args.output, records)
        value = build_inputs_manifest(
            kind="vericodegen_main85_materialized_inputs",
            jsonl_path=args.output,
            records=records,
            sources={
                "sem85": args.sem85,
                "eval_specs": args.eval,
                "clean75_manifest": args.clean75,
            },
        )
        write_json_atomic(args.manifest, value)
        print(
            f"wrote {args.output} and {args.manifest} "
            f"({value['manifest_sha256']})"
        )
        return
    elif args.command == "shift-inputs":
        with args.selection.open("r", encoding="utf-8") as handle:
            selection = json.load(handle)
        verify_manifest(selection)
        records = build_shift50_inputs(read_jsonl(args.input), selection)
        write_jsonl_atomic(args.output, records)
        value = build_inputs_manifest(
            kind="vericodegen_shift50_materialized_inputs",
            jsonl_path=args.output,
            records=records,
            sources={"failure_pool": args.input, "selection_manifest": args.selection},
        )
        write_json_atomic(args.manifest, value)
        print(
            f"wrote {args.output} and {args.manifest} "
            f"({value['manifest_sha256']})"
        )
        return
    elif args.command == "verify":
        with args.manifest.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
        verify_manifest(value)
        print(f"verified {args.manifest} ({value['manifest_sha256']})")
        return
    else:  # pragma: no cover - argparse prevents this
        raise AssertionError(args.command)
    write_json_atomic(args.output, value)
    print(f"wrote {args.output} ({value['kind']}, sha256={value['manifest_sha256']})")


if __name__ == "__main__":
    main()
