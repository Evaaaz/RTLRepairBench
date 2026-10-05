#!/usr/bin/env python3
"""Build the five-model capability curve with an exact, hash-bound cohort.

The historical candidate/proof rows live outside the small standalone export.
This producer therefore fails closed unless every retained run contains exactly
the 68 case ids and four arms declared by ``curve68_manifest.json``.  Alongside
the aggregate curve it writes a source-binding audit with hashes for every
retained input.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable


SOURCE_ROOT = Path(__file__).resolve().parents[2]
ARMS = ("spec0_loc0", "spec1_loc0", "spec0_loc1", "spec1_loc1")
EXPECTED_COHORT = "retained_equivalence_harness_scheduled_cohort"
EXPECTED_CASE_COUNT = 68
FORMAL_VERDICTS = frozenset({"PROVED", "COUNTEREXAMPLE", "ERROR"})
RUN_SPECS = (
    (
        "nvcf/meta/llama-3.3-70b-instruct",
        Path("generated/second_model"),
    ),
    (
        "azure/openai/gpt-4.1",
        Path("generated/second_model/runs/azure__openai__gpt-4.1"),
    ),
    (
        "openai/openai/gpt-4o-mini",
        Path("generated/second_model/runs/openai__openai__gpt-4o-mini"),
    ),
    (
        "OriGen_Fix (RTL-specific 7B)",
        Path("generated/second_model/runs/origen_fix"),
    ),
)


def canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read cohort manifest {path}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise RuntimeError("curve cohort manifest has unsupported schema")
    if document.get("cohort") != EXPECTED_COHORT:
        raise RuntimeError(
            f"curve cohort must be {EXPECTED_COHORT!r}; got {document.get('cohort')!r}"
        )
    raw_ids = document.get("case_ids")
    if (
        not isinstance(raw_ids, list)
        or not raw_ids
        or any(not isinstance(value, str) or not value for value in raw_ids)
    ):
        raise RuntimeError("curve cohort manifest case_ids must be non-empty strings")
    case_ids = tuple(raw_ids)
    if list(case_ids) != sorted(case_ids) or len(case_ids) != len(set(case_ids)):
        raise RuntimeError("curve cohort case_ids must be sorted and unique")
    if document.get("count") != len(case_ids):
        raise RuntimeError("curve cohort count does not match case_ids")
    if len(case_ids) != EXPECTED_CASE_COUNT:
        raise RuntimeError(
            f"curve cohort must contain exactly {EXPECTED_CASE_COUNT} cases"
        )
    digest = sha256_bytes(canonical_json_bytes(list(case_ids)))
    if document.get("case_ids_sha256") != digest:
        raise RuntimeError("curve cohort case_ids_sha256 mismatch")
    return document, case_ids


def _load_json_list(path: Path) -> list[dict[str, Any]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc
    if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
        raise RuntimeError(f"{path} must contain a JSON list of objects")
    return value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
        rows = [json.loads(line) for line in lines if line.strip()]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read {path}: {exc}") from exc
    if any(not isinstance(row, dict) for row in rows):
        raise RuntimeError(f"{path} must contain one JSON object per line")
    return rows


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return path.name


def load_run(
    run_root: Path,
    relative: Path,
    model: str,
    expected_cases: tuple[str, ...],
    require_retained_harness: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    base = run_root / relative
    results_path = base / "results.json"
    verdicts_path = base / "verdicts.jsonl"
    results = _load_json_list(results_path)
    verdict_rows = _load_jsonl(verdicts_path)

    verdicts: dict[str, Any] = {}
    for index, row in enumerate(verdict_rows):
        directory = row.get("dir")
        if not isinstance(directory, str) or not directory:
            raise RuntimeError(f"{verdicts_path}: row {index} has invalid dir")
        if directory in verdicts:
            raise RuntimeError(f"{verdicts_path}: duplicate verdict dir {directory}")
        verdict = row.get("verdict")
        if verdict not in FORMAL_VERDICTS:
            raise RuntimeError(
                f"{verdicts_path}: row {index} has invalid verdict {verdict!r}"
            )
        verdicts[directory] = verdict

    expected_keys = {(case_id, arm) for case_id in expected_cases for arm in ARMS}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    source_dirs: dict[str, str] = {}
    for index, row in enumerate(results):
        case_id, arm = row.get("case_id"), row.get("arm")
        key = (case_id, arm)
        if key not in expected_keys:
            raise RuntimeError(
                f"{results_path}: row {index} is outside frozen cohort/arms: {key!r}"
            )
        if key in records:
            raise RuntimeError(f"{results_path}: duplicate result cell {key!r}")
        design = row.get("design")
        if not isinstance(design, str) or not design:
            raise RuntimeError(f"{results_path}: row {index} has invalid design")
        if require_retained_harness:
            source_dir = row.get("srcdir")
            if not isinstance(source_dir, str) or not source_dir:
                raise RuntimeError(f"{results_path}: row {index} has no retained srcdir")
            source_path = Path(source_dir)
            if source_path.is_absolute() or ".." in source_path.parts:
                raise RuntimeError(f"{results_path}: unsafe retained srcdir {source_dir!r}")
            previous = source_dirs.setdefault(str(case_id), source_path.as_posix())
            if previous != source_path.as_posix():
                raise RuntimeError(
                    f"{results_path}: inconsistent retained srcdir for {case_id}"
                )
        parsed_ok = row.get("parsed_ok")
        if not isinstance(parsed_ok, bool):
            raise RuntimeError(f"{results_path}: row {index} parsed_ok must be Boolean")
        directory = f"{arm}__{case_id}"
        if parsed_ok:
            if directory not in verdicts:
                raise RuntimeError(
                    f"{verdicts_path}: missing verdict for parsed result {directory}"
                )
            verdict = verdicts[directory]
        else:
            if directory in verdicts:
                raise RuntimeError(
                    f"{verdicts_path}: unparsed result unexpectedly has verdict {directory}"
                )
            verdict = "NO_PARSE"
        records[key] = {
            "success": verdict == "PROVED",
            "design": design,
            "verdict": verdict,
        }
    if set(records) != expected_keys:
        missing = sorted(expected_keys - set(records))
        raise RuntimeError(f"{results_path}: missing frozen curve cells: {missing[:5]}")
    unexpected_verdicts = sorted(set(verdicts) - {f"{a}__{c}" for c, a in expected_keys})
    if unexpected_verdicts:
        raise RuntimeError(
            f"{verdicts_path}: verdict rows outside frozen cohort: {unexpected_verdicts[:5]}"
        )

    case_digest = sha256_bytes(canonical_json_bytes(sorted(expected_cases)))
    audit: dict[str, Any] = {
        "model": model,
        "results": {
            "path": _relative(results_path, run_root),
            "sha256": sha256_file(results_path),
            "records": len(results),
        },
        "verdicts": {
            "path": _relative(verdicts_path, run_root),
            "sha256": sha256_file(verdicts_path),
            "records": len(verdict_rows),
        },
        "observed_case_count": len(expected_cases),
        "observed_case_ids_sha256": case_digest,
        "exact_case_arm_roster": True,
        "parsed_cells": sum(bool(row["parsed_ok"]) for row in results),
        "verdict_distribution": dict(sorted(Counter(verdicts.values()).items())),
        "scoring_rule": "PROVED=true; COUNTEREXAMPLE/ERROR/NO_PARSE=false",
    }
    if require_retained_harness:
        harnesses = []
        if set(source_dirs) != set(expected_cases):
            raise RuntimeError("retained harness inventory does not cover exact curve cohort")
        for case_id in expected_cases:
            relative_source = Path(source_dirs[case_id])
            source_path = run_root / relative_source
            equiv = source_path / "equiv_top.sv"
            golden = source_path / "golden.sv"
            if not equiv.is_file() or not golden.is_file():
                raise RuntimeError(f"retained harness is incomplete for {case_id}: {source_path}")
            harnesses.append(
                {
                    "case_id": case_id,
                    "srcdir": relative_source.as_posix(),
                    "equiv_top_sha256": sha256_file(equiv),
                    "golden_sha256": sha256_file(golden),
                }
            )
        audit["cohort_selection_evidence"] = {
            "rule": "case has a retained historical equivalence harness source directory",
            "case_harnesses": harnesses,
            "case_harnesses_sha256": sha256_bytes(canonical_json_bytes(harnesses)),
        }
    return {"model": model, "rec": records}, audit


def frozen_gpt55(
    rows_path: Path,
    run_root: Path,
    expected_cases: tuple[str, ...],
) -> tuple[dict[str, Any], dict[str, Any]]:
    expected_keys = {(case_id, arm) for case_id in expected_cases for arm in ARMS}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    selected_rows: list[dict[str, Any]] = []
    rows = _load_jsonl(rows_path)
    for index, row in enumerate(rows):
        if row.get("mode") != "main" or row.get("case_id") not in expected_cases:
            continue
        key = (row.get("case_id"), row.get("arm"))
        if key not in expected_keys:
            raise RuntimeError(f"{rows_path}: row {index} has unexpected arm")
        if key in records:
            raise RuntimeError(f"{rows_path}: duplicate frozen result cell {key!r}")
        design = row.get("seed_id")
        if not isinstance(design, str) or not design:
            raise RuntimeError(f"{rows_path}: row {index} has invalid seed_id")
        success = row.get("success")
        if not isinstance(success, bool):
            raise RuntimeError(f"{rows_path}: row {index} success must be Boolean")
        formal_status = row.get("formal_status")
        if formal_status not in {None, "PROVED", "COUNTEREXAMPLE", "TIMEOUT", "UNSUPPORTED"}:
            raise RuntimeError(
                f"{rows_path}: row {index} has invalid formal_status {formal_status!r}"
            )
        if formal_status == "PROVED" and not success:
            raise RuntimeError(f"{rows_path}: PROVED row {index} is not successful")
        if formal_status == "COUNTEREXAMPLE" and success:
            raise RuntimeError(f"{rows_path}: COUNTEREXAMPLE row {index} is successful")
        # Use one consistent cross-model curve rule.  Historical primary
        # ``success`` includes simulation fallback for two formal timeouts;
        # the exploratory curve instead credits only a formal PROVED verdict.
        records[key] = {"success": formal_status == "PROVED", "design": design}
        selected_rows.append(row)
    if set(records) != expected_keys:
        missing = sorted(expected_keys - set(records))
        raise RuntimeError(f"{rows_path}: missing frozen curve cells: {missing[:5]}")

    # Re-execute the historical selection rule from second_model_explore.py:
    # scan the full primary ledger in order and keep each case whose first
    # encountered call has a retained formal/PDR equivalence harness.
    derived_harnesses: dict[str, dict[str, str]] = {}
    for index, row in enumerate(rows):
        if row.get("mode") != "main":
            continue
        case_id = row.get("case_id")
        call_id = row.get("call_id")
        if not isinstance(case_id, str) or not isinstance(call_id, str):
            raise RuntimeError(f"{rows_path}: row {index} has invalid case/call identity")
        relative_source = Path(
            f"generated/formal/candidates/{call_id}/formal/pdr/src"
        )
        source = run_root / relative_source
        equiv = source / "equiv_top.sv"
        golden = source / "golden.sv"
        if case_id in derived_harnesses or not equiv.is_file():
            continue
        if not golden.is_file():
            raise RuntimeError(f"retained harness has no golden source: {source}")
        derived_harnesses[case_id] = {
            "case_id": case_id,
            "srcdir": relative_source.as_posix(),
            "equiv_top_sha256": sha256_file(equiv),
            "golden_sha256": sha256_file(golden),
        }
    derived_ids = sorted(derived_harnesses)
    if derived_ids != list(expected_cases):
        raise RuntimeError(
            "historical retained-harness selection does not equal curve cohort; "
            f"missing={sorted(set(expected_cases) - set(derived_ids))[:5]}, "
            f"unexpected={sorted(set(derived_ids) - set(expected_cases))[:5]}"
        )

    definitive_by_arm = []
    for arm in ARMS:
        definitive_by_arm.append(
            {
                row["case_id"]
                for row in rows
                if row.get("mode") == "main"
                and row.get("arm") == arm
                and row.get("formal_status") in {"PROVED", "COUNTEREXAMPLE"}
            }
        )
    formal_intersection = sorted(set.intersection(*definitive_by_arm))
    if len(formal_intersection) != EXPECTED_CASE_COUNT:
        raise RuntimeError("anchor four-arm formal intersection no longer contains 68 cases")
    cohort_overlap = sorted(set(expected_cases) & set(formal_intersection))
    case_digest = sha256_bytes(canonical_json_bytes(sorted(expected_cases)))
    statuses = Counter(
        row.get("formal_status") if row.get("formal_status") is not None else "MISSING"
        for row in selected_rows
    )
    evidence = Counter(str(row.get("evidence")) for row in selected_rows)
    definitive = statuses["PROVED"] + statuses["COUNTEREXAMPLE"]
    audit = {
        "model": "azure/openai/gpt-5.5",
        "rows": {
            "path": _relative(rows_path, run_root),
            "sha256": sha256_file(rows_path),
            "records": len(rows),
            "selected_records": len(records),
        },
        "observed_case_count": len(expected_cases),
        "observed_case_ids_sha256": case_digest,
        "exact_case_arm_roster": True,
        "formal_status_distribution": dict(sorted(statuses.items())),
        "evidence_distribution": dict(sorted(evidence.items())),
        "formal_definitive_cells": definitive,
        "non_formal_cells": len(selected_rows) - definitive,
        "historical_hybrid_success_cells": sum(bool(row["success"]) for row in selected_rows),
        "strict_formal_success_cells": statuses["PROVED"],
        "scoring_rule": "PROVED=true; every other formal status=false",
        "cohort_selection_evidence": {
            "historical_rule": (
                "first main-study call per case with a retained "
                "formal/pdr/src/equiv_top.sv harness"
            ),
            "derived_count": len(derived_ids),
            "derived_case_ids_sha256": sha256_bytes(canonical_json_bytes(derived_ids)),
            "exact_manifest_match": True,
            "case_harnesses": [derived_harnesses[case_id] for case_id in derived_ids],
            "case_harnesses_sha256": sha256_bytes(
                canonical_json_bytes(
                    [derived_harnesses[case_id] for case_id in derived_ids]
                )
            ),
        },
        "relationship_to_anchor_formal_intersection": {
            "formal_intersection_count": len(formal_intersection),
            "formal_intersection_case_ids_sha256": sha256_bytes(
                canonical_json_bytes(formal_intersection)
            ),
            "overlap_count": len(cohort_overlap),
            "identical": derived_ids == formal_intersection,
            "curve_only_case_ids": sorted(set(derived_ids) - set(formal_intersection)),
            "formal_only_case_ids": sorted(set(formal_intersection) - set(derived_ids)),
        },
    }
    return {"model": "azure/openai/gpt-5.5", "rec": records}, audit


def seed_macro(
    records: dict[tuple[str, str], dict[str, Any]],
    cases: Iterable[str],
    arm_a: str,
    arm_b: str,
    seed: int = 7,
) -> tuple[float, float, float]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for case_id in cases:
        left = records[(case_id, arm_a)]
        right = records[(case_id, arm_b)]
        if left["design"] != right["design"]:
            raise RuntimeError(f"design cluster drift for {case_id}: {arm_a} vs {arm_b}")
        clusters[left["design"]].append(int(right["success"]) - int(left["success"]))
    if not clusters:
        raise RuntimeError("cannot compute a seed-macro effect with no clusters")
    keys = sorted(clusters)
    means = {key: sum(clusters[key]) / len(clusters[key]) for key in keys}
    point = sum(means.values()) / len(means)
    rng = random.Random(seed)
    draws = sorted(
        sum(means[keys[rng.randrange(len(keys))]] for _ in keys) / len(keys)
        for _ in range(10000)
    )
    return point * 100, draws[250] * 100, draws[9750] * 100


def _finite(value: float, label: str) -> float:
    if not math.isfinite(value):
        raise RuntimeError(f"non-finite curve value: {label}")
    return value


def atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def build(
    run_root: Path,
    manifest_path: Path,
    frozen_rows_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    manifest, case_ids = load_manifest(manifest_path)
    runs: list[dict[str, Any]] = []
    bindings: list[dict[str, Any]] = []
    for index, (model, relative) in enumerate(RUN_SPECS):
        run, binding = load_run(
            run_root,
            relative,
            model,
            case_ids,
            require_retained_harness=index == 0,
        )
        runs.append(run)
        bindings.append(binding)
    frozen, frozen_binding = frozen_gpt55(frozen_rows_path, run_root, case_ids)
    runs.append(frozen)
    bindings.append(frozen_binding)

    scheduled_harnesses = {
        row["case_id"]: row["srcdir"]
        for row in bindings[0]["cohort_selection_evidence"]["case_harnesses"]
    }
    derived_harnesses = {
        row["case_id"]: row["srcdir"]
        for row in frozen_binding["cohort_selection_evidence"]["case_harnesses"]
    }
    if scheduled_harnesses != derived_harnesses:
        raise RuntimeError(
            "scheduled llama run is not bound to the re-derived retained harness cohort"
        )

    curve: list[dict[str, Any]] = []
    for run in runs:
        records = run["rec"]
        baseline = sum(
            bool(records[(case_id, "spec0_loc0")]["success"]) for case_id in case_ids
        ) / len(case_ids) * 100
        what = seed_macro(records, case_ids, "spec0_loc0", "spec1_loc0")
        where = seed_macro(records, case_ids, "spec0_loc0", "spec0_loc1")
        curve.append(
            {
                "model": run["model"],
                "baseline_pct": round(_finite(baseline, "baseline"), 1),
                "what_pp": round(what[0], 1),
                "what_ci": [round(what[1], 1), round(what[2], 1)],
                "where_pp": round(where[0], 1),
                "where_ci": [round(where[1], 1), round(where[2], 1)],
                "n_cases": len(case_ids),
                "cohort": manifest["cohort"],
                "case_ids_sha256": manifest["case_ids_sha256"],
                "scoring_rule": "strict_formal_proved_only",
            }
        )

    curve_payload = (json.dumps(curve, indent=2) + "\n").encode("utf-8")
    audit = {
        "schema_version": 1,
        "generated_by": "code/scripts/second_model_capability_curve.py",
        "status": "verified_exact_case_arm_roster",
        "scoring_rule": "strict_formal_proved_only_for_all_models",
        "cohort_manifest": {
            "path": "configs/vericodegen/curve68_manifest.json",
            "sha256": sha256_file(manifest_path),
            "cohort": manifest["cohort"],
            "count": len(case_ids),
            "case_ids_sha256": manifest["case_ids_sha256"],
        },
        "cohort_relationship": frozen_binding[
            "relationship_to_anchor_formal_intersection"
        ],
        "source_bindings": bindings,
        "curve_output": {
            "models": len(curve),
            "sha256": sha256_bytes(curve_payload),
        },
        "standalone_note": (
            "The hash-bound source rows are retained outside the small standalone tree. "
            "A clean recomputation requires those exact files from controlled shared storage."
        ),
    }
    return curve, audit


def parser() -> argparse.ArgumentParser:
    default_run_root = Path(os.environ.get("RTLREPAIR_ROOT", str(SOURCE_ROOT)))
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--run-root", type=Path, default=default_run_root)
    result.add_argument(
        "--manifest",
        type=Path,
        default=SOURCE_ROOT / "configs/vericodegen/curve68_manifest.json",
    )
    result.add_argument("--frozen-rows", type=Path)
    result.add_argument("--output", type=Path)
    result.add_argument("--audit-output", type=Path)
    return result


def main(argv: Iterable[str] | None = None) -> int:
    args = parser().parse_args(argv)
    run_root = args.run_root.resolve()
    frozen_rows = args.frozen_rows or run_root / "generated/reports/vericodegen_full85_rows.jsonl"
    output = args.output or run_root / "generated/second_model/capability_curve.json"
    audit_output = args.audit_output or output.with_name("capability_curve_cohort_audit.json")
    curve, audit = build(run_root, args.manifest.resolve(), frozen_rows.resolve())
    curve_payload = (json.dumps(curve, indent=2) + "\n").encode("utf-8")
    audit_payload = (json.dumps(audit, indent=2) + "\n").encode("utf-8")
    atomic_write(output.resolve(), curve_payload)
    atomic_write(audit_output.resolve(), audit_payload)
    print(
        f"wrote {output} and {audit_output}: {len(curve)} models, "
        f"{curve[0]['n_cases']} exact cohort cases"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
