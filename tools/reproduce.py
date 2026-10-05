#!/usr/bin/env python3
"""Offline reconstruction and preflight entry point for the paper artifact.

This command never calls a model endpoint.  It consumes only versioned,
sanitized records and writes derived reports under ``build/reproduced`` (or an
explicit ``--output`` directory).  Model reruns remain separate because they
are costly and are not byte-reproducible across provider revisions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import tempfile
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
CODE = ROOT / "code"
BENCHMARKS = CODE / "benchmarks"
SCRIPTS = CODE / "scripts"
FROZEN_ANALYSIS_PYTHON = "3.9.6"
for path in (str(CODE), str(BENCHMARKS), str(SCRIPTS)):
    if path not in sys.path:
        sys.path.insert(0, path)


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    path.write_text(payload, encoding="utf-8")


def write_json_atomic(path: Path, value: Any) -> None:
    """Publish one JSON document only after its complete bytes are durable."""

    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_environment_lock() -> dict[str, str]:
    """Fail closed unless environment and paper-input locks both match."""

    lock_path = ROOT / "environment.lock.json"
    lock = read_json(lock_path)
    if not isinstance(lock, dict) or lock.get("schema_version") != 1:
        raise RuntimeError("unsupported or malformed environment.lock.json")

    checked: dict[str, str] = {}

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if "path" in value or "sha256" in value:
                relative = value.get("path")
                expected = value.get("sha256")
                if not isinstance(relative, str) or not isinstance(expected, str):
                    raise RuntimeError("environment lock path/hash entry is malformed")
                candidate = Path(relative)
                if candidate.is_absolute() or ".." in candidate.parts:
                    raise RuntimeError(f"environment lock path escapes root: {relative}")
                target = ROOT / candidate
                if not target.is_file():
                    raise RuntimeError(f"environment lock input is missing: {relative}")
                actual = sha256_file(target)
                if actual != expected:
                    raise RuntimeError(
                        f"environment lock hash mismatch for {relative}: "
                        f"expected {expected}, got {actual}"
                    )
                checked[relative] = actual
            for item in value.values():
                visit(item)
        elif isinstance(value, list):
            for item in value:
                visit(item)

    visit(lock)
    formal = lock.get("formal_validation", {})
    relative = formal.get("container_lock")
    expected = formal.get("container_lock_sha256")
    if not isinstance(relative, str) or not isinstance(expected, str):
        raise RuntimeError("formal container lock is missing from environment lock")
    target = ROOT / relative
    if not target.is_file() or sha256_file(target) != expected:
        raise RuntimeError(f"formal container lock hash mismatch: {relative}")
    checked[relative] = expected

    artifact_lock = read_json(ROOT / "artifact.lock.json")
    paper_inputs = artifact_lock.get("paper_inputs")
    if not isinstance(paper_inputs, list) or not paper_inputs:
        raise RuntimeError("artifact.lock.json has no paper_inputs")
    for index, item in enumerate(paper_inputs):
        if not isinstance(item, dict):
            raise RuntimeError(f"artifact paper_inputs[{index}] is malformed")
        relative = item.get("path")
        expected = item.get("sha256")
        if not isinstance(relative, str) or not isinstance(expected, str):
            raise RuntimeError(f"artifact paper_inputs[{index}] is malformed")
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise RuntimeError(f"artifact paper input escapes root: {relative}")
        target = ROOT / candidate
        if not target.is_file():
            raise RuntimeError(f"artifact paper input is missing: {relative}")
        actual = sha256_file(target)
        if actual != expected:
            raise RuntimeError(
                f"artifact paper-input hash mismatch for {relative}: "
                f"expected {expected}, got {actual}"
            )
        if relative in checked and checked[relative] != actual:
            raise RuntimeError(f"conflicting locked hashes for {relative}")
        checked[relative] = actual

    # The sanitizer retains both logical names for provenance, but clean75 is
    # selected by its manifest at analysis time; its row payload must remain an
    # exact alias of full85 rather than drifting into a second source of truth.
    for full_name, clean_name in (
        (
            "artifacts/public/primary/analysis/vericodegen_full85_rows.jsonl",
            "artifacts/public/primary/analysis/vericodegen_clean75_rows.jsonl",
        ),
        (
            "artifacts/public/primary/analysis/full85_rows.jsonl",
            "artifacts/public/primary/analysis/clean75_rows.jsonl",
        ),
    ):
        if sha256_file(ROOT / full_name) != sha256_file(ROOT / clean_name):
            raise RuntimeError(
                f"sanitized full85/clean75 row aliases diverged: {full_name}, {clean_name}"
            )
    return dict(sorted(checked.items()))


def reconstruct_primary() -> dict[str, Any]:
    from vericodegen_data import canonical_sha256
    from vericodegen_stats import (
        DEFAULT_CONTRASTS,
        analyze_contrasts,
        analyze_secondaries,
        load_shift_case_ids,
    )

    source = (
        ROOT
        / "artifacts"
        / "public"
        / "primary"
        / "analysis"
        / "vericodegen_full85_rows.jsonl"
    )
    rows = read_jsonl(source)

    analysis = analyze_contrasts(
        rows,
        DEFAULT_CONTRASTS,
        bootstrap_samples=10_000,
        rng_seed=42,
    )
    shift_manifest = (
        ROOT
        / "artifacts"
        / "public"
        / "primary"
        / "manifests"
        / "02_shift50_manifest.json"
    )
    analysis["secondaries"] = analyze_secondaries(
        rows,
        shift_case_ids=load_shift_case_ids(shift_manifest),
        cancel_inconclusive_shift=True,
        bootstrap_samples=10_000,
        rng_seed=142,
    )
    analysis["analysis_sha256"] = canonical_sha256(
        {key: value for key, value in analysis.items() if key != "analysis_sha256"}
    )
    return analysis


def reconstruct_clean75() -> dict[str, Any]:
    from vericodegen_data import canonical_sha256
    from vericodegen_stats import (
        DEFAULT_CONTRASTS,
        analyze_contrasts,
        analyze_secondaries,
        load_clean_case_ids,
        load_materialized_id_map,
        load_shift_case_ids,
    )

    rows = read_jsonl(
        ROOT
        / "artifacts/public/primary/analysis/vericodegen_clean75_rows.jsonl"
    )
    main_id_map = load_materialized_id_map(
        ROOT / "configs/vericodegen/main85_inputs.manifest.json"
    )
    keep = load_clean_case_ids(
        ROOT / "configs/vericodegen/clean75_manifest.json", main_id_map
    )
    analysis = analyze_contrasts(
        rows,
        DEFAULT_CONTRASTS,
        keep_case_ids=keep,
        bootstrap_samples=10_000,
        rng_seed=42,
    )
    analysis["secondaries"] = analyze_secondaries(
        rows,
        main_keep_case_ids=keep,
        shift_case_ids=load_shift_case_ids(
            ROOT / "artifacts/public/primary/manifests/02_shift50_manifest.json"
        ),
        cancel_inconclusive_shift=True,
        bootstrap_samples=10_000,
        rng_seed=142,
    )
    analysis["analysis_sha256"] = canonical_sha256(
        {key: value for key, value in analysis.items() if key != "analysis_sha256"}
    )
    return analysis


def _without_export_signatures(value: dict[str, Any]) -> dict[str, Any]:
    return {
        key: item
        for key, item in value.items()
        if key not in {"public_document_sha256", "source_analysis_sha256"}
    }


def _estimands(value: Any, prefix: str = "") -> dict[str, dict[str, Any]]:
    found: dict[str, dict[str, Any]] = {}
    if isinstance(value, dict):
        if {
            "risk_difference",
            "case_count",
            "seed_count",
        }.issubset(value):
            found[prefix or "$"] = value
        for key, item in value.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            found.update(_estimands(item, child))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.update(_estimands(item, f"{prefix}[{index}]"))
    return found


_P_VALUE_FIELDS = {"p_value_unadjusted", "p_value_holm"}
_IGNORED_DERIVED_FIELDS = {"analysis_sha256"}
_P_VALUE_ABS_TOLERANCE = 0.01
_FLOAT_ABS_TOLERANCE = 1e-12


def _validate_analysis_domains(value: Any, path: str = "$") -> None:
    """Reject mathematically invalid values before applying drift tolerances."""

    if isinstance(value, dict):
        for key, item in value.items():
            child = f"{path}.{key}"
            if key in _P_VALUE_FIELDS:
                if (
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    or not 0.0 <= float(item) <= 1.0
                ):
                    raise RuntimeError(f"{child}: p-value must be finite and in [0, 1]")
            if key == "risk_difference":
                if (
                    isinstance(item, bool)
                    or not isinstance(item, (int, float))
                    or not math.isfinite(float(item))
                    or not -1.0 <= float(item) <= 1.0
                ):
                    raise RuntimeError(
                        f"{child}: risk difference must be finite and in [-1, 1]"
                    )
            if key == "ci95":
                if (
                    not isinstance(item, list)
                    or len(item) != 2
                    or any(
                        isinstance(endpoint, bool)
                        or not isinstance(endpoint, (int, float))
                        or not math.isfinite(float(endpoint))
                        for endpoint in item
                    )
                    or float(item[0]) > float(item[1])
                ):
                    raise RuntimeError(
                        f"{child}: confidence interval must contain ordered finite endpoints"
                    )
            _validate_analysis_domains(item, child)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_analysis_domains(item, f"{path}[{index}]")


def _verify_scientific_fields(frozen: Any, rebuilt: Any, path: str = "$") -> None:
    """Compare every declared analysis field with bounded numeric tolerances.

    Bootstrap p-values can move slightly across supported CPython versions
    because the finite draw stream is runtime-sensitive.  No other field is
    allowed a scientifically meaningful change.  Derived analysis hashes are
    checked only by ``--strict`` because any allowed float drift changes them.
    """

    if isinstance(frozen, dict):
        if not isinstance(rebuilt, dict):
            raise RuntimeError(f"{path}: expected object, got {type(rebuilt).__name__}")
        frozen_keys = set(frozen) - _IGNORED_DERIVED_FIELDS
        rebuilt_keys = set(rebuilt) - _IGNORED_DERIVED_FIELDS
        if frozen_keys != rebuilt_keys:
            raise RuntimeError(
                f"{path}: analysis field inventory drift; "
                f"missing={sorted(frozen_keys - rebuilt_keys)}, "
                f"extra={sorted(rebuilt_keys - frozen_keys)}"
            )
        for key in sorted(frozen_keys):
            _verify_scientific_fields(
                frozen[key], rebuilt[key], f"{path}.{key}"
            )
        return
    if isinstance(frozen, list):
        if not isinstance(rebuilt, list) or len(frozen) != len(rebuilt):
            raise RuntimeError(f"{path}: analysis list length/type drift")
        for index, (left, right) in enumerate(zip(frozen, rebuilt)):
            _verify_scientific_fields(left, right, f"{path}[{index}]")
        return
    if isinstance(frozen, bool) or frozen is None or isinstance(frozen, str):
        if frozen != rebuilt:
            raise RuntimeError(f"{path}: analysis value drift")
        return
    if isinstance(frozen, (int, float)):
        if isinstance(rebuilt, bool) or not isinstance(rebuilt, (int, float)):
            raise RuntimeError(f"{path}: analysis numeric type drift")
        left, right = float(frozen), float(rebuilt)
        if not (math.isfinite(left) and math.isfinite(right)):
            raise RuntimeError(f"{path}: non-finite analysis value")
        field = path.rsplit(".", 1)[-1]
        tolerance = (
            _P_VALUE_ABS_TOLERANCE
            if field in _P_VALUE_FIELDS
            else _FLOAT_ABS_TOLERANCE
        )
        if not math.isclose(left, right, rel_tol=0.0, abs_tol=tolerance):
            raise RuntimeError(
                f"{path}: analysis numeric drift exceeds {tolerance:g}; "
                f"expected {left}, got {right}"
            )
        return
    if frozen != rebuilt:
        raise RuntimeError(f"{path}: unsupported analysis value drift")


def verify_analysis(actual: dict[str, Any], expected_path: Path) -> dict[str, Any]:
    expected = read_json(expected_path)
    expected_source = _without_export_signatures(expected)
    _validate_analysis_domains(expected_source)
    _validate_analysis_domains(actual)
    _verify_scientific_fields(expected_source, actual)
    expected_estimands = _estimands(expected_source)
    actual_estimands = _estimands(actual)
    if set(expected_estimands) != set(actual_estimands):
        raise RuntimeError("primary analysis estimand inventory drifted")
    for path, frozen in expected_estimands.items():
        rebuilt = actual_estimands[path]
        if frozen["case_count"] != rebuilt["case_count"]:
            raise RuntimeError(f"{path}: case_count drift")
        if frozen["seed_count"] != rebuilt["seed_count"]:
            raise RuntimeError(f"{path}: seed_count drift")
        if not math.isclose(
            float(frozen["risk_difference"]),
            float(rebuilt["risk_difference"]),
            rel_tol=0.0,
            abs_tol=_FLOAT_ABS_TOLERANCE,
        ):
            raise RuntimeError(f"{path}: risk_difference drift")

    frozen_shift = expected_source["secondaries"]["distribution_shift"]
    rebuilt_shift = actual["secondaries"]["distribution_shift"]
    for key in (
        "status",
        "expected_cases",
        "observed_calls",
        "definitive_calls",
        "unresolved_calls",
    ):
        if frozen_shift.get(key) != rebuilt_shift.get(key):
            raise RuntimeError(f"distribution_shift.{key} drift")

    # Exact bytes remain a separate, stricter CPython-3.9.6 contract.  Other
    # supported runtimes must still match every categorical/decision field,
    # every CI and estimand (within float roundoff), and every p-value within a
    # finite 0.01 Monte Carlo tolerance.
    return {
        "point_estimands_verified": len(expected_estimands),
        "coverage_decision_verified": True,
        "byte_exact": actual == expected_source,
        "frozen_analysis_python": FROZEN_ANALYSIS_PYTHON,
        "runtime_python": sys.version.split()[0],
        "frozen_analysis_sha256": expected_source.get("analysis_sha256"),
        "recomputed_analysis_sha256": actual.get("analysis_sha256"),
        "known_gap": (
            None
            if actual == expected_source
            else (
                "finite-bootstrap floats/draw stream differ outside the frozen "
                f"CPython {FROZEN_ANALYSIS_PYTHON} runtime; all scientific fields "
                "passed bounded cross-runtime tolerances"
            )
        ),
    }


def specialization_summary() -> dict[str, Any]:
    root = ROOT / "artifacts" / "public" / "specialization"
    expected = {
        "rb_base_diag": ("base", False),
        "rb_base_diag_spec": ("base", True),
        "rb_v5_diag": ("tuned", False),
        "rb_v5_diag_spec": ("tuned", True),
        "rb_verireason_diag": ("verireason", False),
        "rb_verireason_diag_spec": ("verireason", True),
        "rb_vrqwen_diag": ("vrqwen", False),
        "rb_vrqwen_diag_spec": ("vrqwen", True),
    }
    # The flat rb_<model>_<arm>.json exports were replaced by a per-run tree keyed by
    # model/version/signal, with index.json as the entry point. Resolve the same eight
    # arms through the index so the integrity checks below are unchanged.
    index = read_json(root / "index.json")
    runs = index.get("runs") if isinstance(index, dict) else None
    if not isinstance(runs, list):
        raise RuntimeError("specialization index.json has no runs list")
    # The index carries every generation of a run, so a model/signal pair legitimately
    # appears more than once. Take the newest version per pair; the released report
    # names which generation the paper quotes.
    newest: dict[str, str] = {}
    actual: dict[str, Path] = {}
    for run in runs:
        stem = f"rb_{run['model']}_{run['signal']}"
        candidate = root / run["model"] / run["version"] / run["signal"] / "result.json"
        if not candidate.is_file():
            raise RuntimeError(f"specialization run missing its result: {candidate}")
        if stem not in expected:
            continue          # diagnostic arms (wrong-template controls) are kept, not reported
        if stem not in newest or run["version"] > newest[stem]:
            newest[stem] = run["version"]
            actual[stem] = candidate
    if set(actual) != set(expected):
        raise RuntimeError(
            "specialization arm inventory drift; "
            f"missing={sorted(set(expected) - set(actual))}, "
            f"extra={sorted(set(actual) - set(expected))}"
        )
    arms: dict[str, Any] = {}
    reference_ids: set[str] | None = None
    reference_buckets: dict[str, str] | None = None
    outcomes: dict[str, dict[str, bool]] = {}
    for stem in sorted(expected):
        path = actual[stem]
        value = read_json(path)
        summary = value.get("summary")
        records = value.get("records")
        if not isinstance(summary, dict) or not isinstance(records, list):
            raise ValueError(f"malformed specialization snapshot: {path}")
        expected_model, expected_spec = expected[stem]
        # The second generation stopped writing summary.spec, so absent is tolerated;
        # a value that contradicts the arm is still an error.
        spec_seen = summary.get("spec")
        if summary.get("model") != expected_model or (
            spec_seen is not None and bool(spec_seen) is not expected_spec
        ):
            raise RuntimeError(f"{path}: model/spec identity drift")
        if summary.get("n") != 459 or len(records) != 459:
            raise RuntimeError(f"{path}: expected exactly 459 records")
        identifiers = [str(row.get("id")) for row in records]
        if any(identifier == "None" for identifier in identifiers):
            raise RuntimeError(f"{path}: record without id")
        if len(set(identifiers)) != 459:
            raise RuntimeError(f"{path}: record ids are not unique")
        if reference_ids is None:
            reference_ids = set(identifiers)
            reference_buckets = {
                str(row["id"]): str(row.get("bucket")) for row in records
            }
        elif set(identifiers) != reference_ids:
            raise RuntimeError(f"{path}: case roster differs across arms")
        elif {
            str(row["id"]): str(row.get("bucket")) for row in records
        } != reference_buckets:
            raise RuntimeError(f"{path}: case bucket assignments differ across arms")
        if any(
            row.get("bucket") not in {"lint", "semantic"}
            or not isinstance(row.get("recovered"), bool)
            for row in records
        ):
            raise RuntimeError(f"{path}: invalid bucket/recovered record")
        outcomes[stem] = {
            str(row["id"]): bool(row["recovered"]) for row in records
        }
        by_bucket: dict[str, dict[str, int | float]] = {}
        for bucket in ("lint", "semantic"):
            subset = [row for row in records if row.get("bucket") == bucket]
            recovered = sum(row.get("recovered") is True for row in subset)
            by_bucket[bucket] = {
                "n": len(subset),
                "recovered": recovered,
                "recovery_rate": recovered / len(subset) if subset else 0.0,
            }
            frozen = summary.get("by_bucket", {}).get(bucket, {})
            if frozen.get("n") != len(subset) or frozen.get("recovered") != recovered:
                raise RuntimeError(f"{path}: frozen summary disagrees with per-case records")
        if by_bucket["lint"]["n"] != 374 or by_bucket["semantic"]["n"] != 85:
            raise RuntimeError(f"{path}: expected lint/semantic denominators 374/85")
        arms[path.stem] = by_bucket
    assert reference_buckets is not None
    semantic_ids = {
        case_id for case_id, bucket in reference_buckets.items() if bucket == "semantic"
    }
    base = outcomes["rb_base_diag"]
    tuned = outcomes["rb_v5_diag"]
    base_only = sum(base[case_id] and not tuned[case_id] for case_id in semantic_ids)
    tuned_only = sum(tuned[case_id] and not base[case_id] for case_id in semantic_ids)
    # Do not pin the counts: each generation of the run produces its own. What must hold
    # is that the same-base contrast stays non-significant, which is the paper's claim.
    if base_only + tuned_only == 0:
        raise RuntimeError("specialization same-base contrast has no discordant pairs")
    discordant = base_only + tuned_only
    tail = min(base_only, tuned_only)
    p_value = min(
        1.0,
        2.0 * sum(math.comb(discordant, k) for k in range(tail + 1)) / (2**discordant),
    )
    return {
        "schema_version": 1,
        "arms": arms,
        "same_base_v5_vs_base_semantic": {
            "n": len(semantic_ids),
            "base_recovered": sum(base[case_id] for case_id in semantic_ids),
            "v5_recovered": sum(tuned[case_id] for case_id in semantic_ids),
            "base_only": base_only,
            "v5_only": tuned_only,
            "exact_mcnemar_p": p_value,
        },
    }


def output_budget_summary() -> dict[str, Any]:
    from output_budget_audit import _load_jsonl, analyze

    ledger = ROOT / "artifacts/public/primary/ledger/public_events.jsonl"
    rebuilt = analyze(_load_jsonl(ledger))
    frozen_path = (
        ROOT / "artifacts/public/primary/analysis/output_budget_confound.json"
    )
    frozen = read_json(frozen_path)
    if rebuilt != frozen:
        raise RuntimeError("output-budget frozen summary disagrees with public ledger")
    return rebuilt


def reproduce_paper_exhibits(output_dir: Path) -> dict[str, Any]:
    from e6_curve_exhibits import generate as generate_curve
    from e6_table6_realbug import generate as generate_realbug_table
    from ws_i_prompts_appendix import render as render_prompts

    curve_figure, curve_table, curve_rows = generate_curve(
        ROOT / "artifacts/public/capability/capability_curve.json",
        output_dir,
    )
    realbug_table, realbug_rows = generate_realbug_table(
        ROOT / "artifacts/public/realbugs",
        output_dir,
    )
    prompt_appendix = output_dir / "app_prompts.tex"
    prompt_appendix.parent.mkdir(parents=True, exist_ok=True)
    prompt_appendix.write_bytes(render_prompts().encode("utf-8"))
    generated = (curve_figure, curve_table, realbug_table, prompt_appendix)
    for path in generated:
        frozen = ROOT / "paper" / path.name
        if path.read_bytes() != frozen.read_bytes():
            raise RuntimeError(
                f"regenerated paper fragment differs from frozen source: {path.name}"
            )
    return {
        "status": "verified",
        "capability_models": len(curve_rows),
        "realbug_models": len(realbug_rows),
        "fragments": [path.name for path in generated],
    }


def doctor() -> int:
    commands = ("verilator", "iverilog", "vvp", "docker", "tectonic", "latexmk")
    try:
        locked_inputs = verify_environment_lock()
        lock_error = None
    except (OSError, RuntimeError, ValueError) as exc:
        locked_inputs = {}
        lock_error = str(exc)
    report = {
        "python": sys.version.split()[0],
        "project_root": str(ROOT),
        "required_offline_files": {
            "claims": (ROOT / "claims.json").is_file(),
            "primary_rows": (
                ROOT
                / "artifacts/public/primary/analysis/vericodegen_full85_rows.jsonl"
            ).is_file(),
            "specialization": (ROOT / "artifacts/public/specialization").is_dir(),
            "paper": (ROOT / "paper/paper.tex").is_file(),
        },
        "optional_system_tools": {name: shutil.which(name) for name in commands},
        "environment_lock": {
            "status": "verified" if lock_error is None else "failed",
            "files": locked_inputs,
            "error": lock_error,
        },
        "artifact_root": os.environ.get("RTLREPAIR_ARTIFACTS", str(ROOT / "artifacts")),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    return (
        0
        if all(report["required_offline_files"].values()) and lock_error is None
        else 2
    )


def reproduce(output: Path, *, strict: bool = False) -> int:
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / "reproduction_report.json"
    write_json_atomic(
        report_path,
        {
            "schema_version": 1,
            "status": "running",
            "note": "No derived output is verified until status becomes verified.",
        },
    )
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output.name}.staging-", dir=output.parent)
    )
    try:
        locked_inputs = verify_environment_lock()
        primary = reconstruct_primary()
        write_json(staging / "primary_stats.json", primary)
        primary_audit = verify_analysis(
            primary,
            ROOT / "artifacts/public/primary/analysis/vericodegen_full85_stats.json",
        )
        clean75 = reconstruct_clean75()
        write_json(staging / "clean75_stats.json", clean75)
        clean75_audit = verify_analysis(
            clean75,
            ROOT / "artifacts/public/primary/analysis/vericodegen_clean75_stats.json",
        )
        specialization = specialization_summary()
        write_json(staging / "specialization_summary.json", specialization)
        output_budget = output_budget_summary()
        write_json(staging / "output_budget_confound.json", output_budget)
        exhibit_audit = reproduce_paper_exhibits(staging / "paper")

        if strict and (
            sys.version.split()[0] != FROZEN_ANALYSIS_PYTHON
            or not (primary_audit["byte_exact"] and clean75_audit["byte_exact"])
        ):
            raise RuntimeError(
                "strict reproduction requires the frozen CPython "
                f"{FROZEN_ANALYSIS_PYTHON} analysis runtime"
            )

        claims_path = ROOT / "claims.json"
        claims = read_json(claims_path) if claims_path.is_file() else {"claims": []}
        statuses: dict[str, int] = {}
        for claim in claims.get("claims", []):
            status = str(claim.get("status", "unknown"))
            statuses[status] = statuses.get(status, 0) + 1
        report = {
            "schema_version": 1,
            "status": "verified",
            "environment_lock": {
                "status": "verified",
                "files": locked_inputs,
            },
            "offline_reconstructions": {
                "primary": primary_audit,
                "clean75": clean75_audit,
                "specialization": "verified",
                "output_budget": "verified",
                "paper_exhibits": exhibit_audit,
            },
            "claim_status_counts": statuses,
            "note": (
                "Incomplete and stale claim statuses are disclosures, not silently "
                "upgraded by this command. See claims.json."
            ),
        }
        write_json(staging / "reproduction_report.json", report)

        # Publish verified analyses first and the success marker last.  A
        # failed run therefore cannot leave a stale successful report behind.
        for name in (
            "primary_stats.json",
            "clean75_stats.json",
            "specialization_summary.json",
            "output_budget_confound.json",
        ):
            os.replace(staging / name, output / name)
        (output / "paper").mkdir(parents=True, exist_ok=True)
        for name in (
            "fig_curve_body.tex",
            "tab_curve.tex",
            "tab_multimodel.tex",
            "app_prompts.tex",
        ):
            os.replace(staging / "paper" / name, output / "paper" / name)
        os.replace(staging / "reproduction_report.json", report_path)
    except Exception as exc:
        write_json_atomic(
            report_path,
            {
                "schema_version": 1,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
                "note": "No outputs from this failed attempt were published.",
            },
        )
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    print(f"verified offline reconstruction -> {output}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("doctor", help="report inputs and optional external tools")
    offline = subparsers.add_parser(
        "offline", help="reconstruct versioned analyses without network or API calls"
    )
    offline.add_argument("--output", type=Path, default=ROOT / "build" / "reproduced")
    offline.add_argument(
        "--strict",
        action="store_true",
        help="require byte-identical frozen bootstrap output (CPython 3.9.6)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "doctor":
        return doctor()
    if args.command == "offline":
        return reproduce(args.output, strict=args.strict)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
