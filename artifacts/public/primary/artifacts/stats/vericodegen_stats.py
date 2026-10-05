"""Preregistered clustered analysis for the VeriCodeGen repair study.

Input is normalized JSONL with exactly one record per ``(mode, case_id, arm)``
and five fields::

    {"mode": "main", "case_id": "main_0000", "seed_id": "cluster-...",
     "arm": "spec0_loc0", "success": false}

``success`` must reflect the frozen evaluation policy (formal proof where
supported, independent-simulation fallback otherwise). With ``--from-ledger``
the module constructs that Boolean itself using a fixed, fail-closed mapping;
otherwise the input must already contain it explicitly. The only nullable
exception is the explicit coverage-only cancellation of an inconclusive shift
effect.

The estimand is a seed/design-macro paired risk difference: first average the
paired case-level differences within each seed, then give each seed equal
weight.  Confidence intervals and p-values use 10,000 paired cluster bootstrap
resamples by default.  The three preregistered primary comparisons receive a
Holm family-wise correction.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.vericodegen_data import (  # noqa: E402
    canonical_sha256,
    read_jsonl,
    validate_redaction_decision_manifest,
    verify_manifest,
    write_json_atomic,
    write_jsonl_atomic,
)


DEFAULT_CONTRASTS: tuple[tuple[str, str, str, str], ...] = (
    ("what_full_spec", "main", "spec0_loc0", "spec1_loc0"),
    ("where_oracle_location", "main", "spec0_loc0", "spec0_loc1"),
    (
        "counterexample_revision",
        "feedback",
        "generic_failure",
        "concrete_counterexample",
    ),
)
SECONDARY_ARM_NAMES = {
    "no_spec_no_location": "spec0_loc0",
    "full_spec_no_location": "spec1_loc0",
    "no_spec_oracle_location": "spec0_loc1",
    "full_spec_oracle_location": "spec1_loc1",
    "redacted_spec_no_location": "redacted_spec_loc0",
    "shift_no_spec": "spec0_loc0",
    "shift_full_spec": "spec1_loc0",
}
DEFAULT_PROTOCOL_AMENDMENT = (
    REPO_ROOT / "configs" / "vericodegen" / "protocol_amendment_2026-07-15.json"
)


def protocol_amendment_binding(
    path: Path | str = DEFAULT_PROTOCOL_AMENDMENT,
) -> dict[str, str]:
    amendment_path = Path(path)
    with amendment_path.open("r", encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, Mapping):
        raise ValueError("protocol amendment must be an object")
    unsigned = {key: value for key, value in document.items() if key != "manifest_sha256"}
    if document.get("manifest_sha256") != canonical_sha256(unsigned):
        raise ValueError("protocol amendment manifest SHA drift")
    if (
        document.get("kind") != "vericodegen_pre_output_protocol_amendment"
        or document.get("status") != "FROZEN"
    ):
        raise ValueError("protocol amendment is not frozen")
    return {
        "path": str(
            amendment_path.relative_to(REPO_ROOT)
            if amendment_path.resolve().is_relative_to(REPO_ROOT.resolve())
            else amendment_path.resolve()
        ),
        "file_sha256": hashlib.sha256(amendment_path.read_bytes()).hexdigest(),
        "manifest_sha256": str(document["manifest_sha256"]),
    }


def _quantile(sorted_values: Sequence[float], probability: float) -> float:
    if not sorted_values:
        raise ValueError("quantile requires at least one value")
    if not 0.0 <= probability <= 1.0:
        raise ValueError("probability must be in [0, 1]")
    if len(sorted_values) == 1:
        return float(sorted_values[0])
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(sorted_values[lower] * (1.0 - weight) + sorted_values[upper] * weight)


def normalize_ledger_events(
    events: Sequence[Mapping[str, Any]],
    allow_inconclusive_shift: bool = False,
) -> list[dict[str, Any]]:
    """Convert terminal runner ledger events to the frozen binary success policy.

    Tool-inconclusive statuses are not model failures: TIMEOUT and UNSUPPORTED
    use the preregistered independent simulation. Any unresolved tool outcome or
    proof/simulation contradiction aborts analysis instead of being imputed.

    The sole opt-in exception retains terminal, tool-inconclusive ``shift`` rows
    as ``success=null`` coverage records.  It does not relax validation for any
    primary-study mode, a missing verdict, or a pending/unknown tool status.
    """
    responses: dict[str, Mapping[str, Any]] = {}
    verdicts: dict[str, Mapping[str, Any]] = {}
    for event in events:
        call_id = str(event.get("call_id", ""))
        if event.get("event") == "response" and call_id:
            if call_id in responses:
                raise ValueError(f"{call_id}: duplicate terminal response event")
            responses[call_id] = event
        elif event.get("event") == "verdict" and call_id:
            if call_id in verdicts:
                raise ValueError(f"{call_id}: duplicate terminal verdict event")
            verdicts[call_id] = event
    orphan_verdicts = sorted(set(verdicts) - set(responses))
    if orphan_verdicts:
        raise ValueError(f"orphan verdict without response: {orphan_verdicts[0]}")
    normalized: list[dict[str, Any]] = []
    for call_id, response in sorted(responses.items()):
        if response.get("nonresearch") is True:
            continue
        mode = str(response.get("mode", ""))
        arm = str(response.get("arm", ""))
        case_id = str(response.get("case_id", ""))
        cluster_id = str(response.get("cluster_id", ""))
        if not mode or not arm or not case_id or not cluster_id:
            raise ValueError(f"{call_id}: response lacks mode/arm/case/cluster identity")
        model_outcome = str(response.get("model_outcome", ""))
        evidence = "http_200_model_failure"
        resolved = True
        formal_status: str | None = None
        simulation_status: str | None = None
        formal_reason: str | None = None
        simulation_reason: str | None = None
        inconclusive_reason: str | None = None
        if model_outcome != "accepted_pending_compile":
            if call_id in verdicts:
                raise ValueError(f"{call_id}: non-candidate response has a validation verdict")
            success = False
        else:
            verdict = verdicts.get(call_id)
            if verdict is None:
                raise ValueError(f"{call_id}: accepted candidate has no terminal validation")
            compile_verdict = verdict.get("compile_verdict")
            formal = verdict.get("formal_verdict")
            simulation = verdict.get("simulation_verdict")
            if (
                not isinstance(compile_verdict, Mapping)
                or not isinstance(formal, Mapping)
                or not isinstance(simulation, Mapping)
            ):
                raise ValueError(f"{call_id}: validation contract is incomplete")
            compile_passed = compile_verdict.get("passed")
            if not isinstance(compile_passed, bool):
                raise ValueError(f"{call_id}: compile verdict is not boolean")
            formal_status = str(formal.get("status", "")).upper()
            simulation_status = str(simulation.get("status", "")).upper()
            formal_reason = (
                str(formal["reason"]) if formal.get("reason") is not None else None
            )
            simulation_reason = (
                str(simulation["reason"])
                if simulation.get("reason") is not None
                else None
            )
            terminal_simulation_statuses = {
                "PASS",
                "COUNTEREXAMPLE",
                "TIMEOUT",
                "UNSUPPORTED",
                "COMPILE_FAIL",
            }
            if simulation_status not in terminal_simulation_statuses:
                raise ValueError(
                    f"{call_id}: unknown or pending simulation status "
                    f"{simulation_status!r}"
                )
            if not compile_passed and formal_status != "COMPILE_FAIL":
                raise ValueError(f"{call_id}: compile and formal verdicts contradict")
            if compile_passed and formal_status == "COMPILE_FAIL":
                raise ValueError(f"{call_id}: compile and formal verdicts contradict")
            if not compile_passed:
                success = False
                evidence = "compile_fail"
            elif formal_status == "PROVED":
                if simulation_status == "COUNTEREXAMPLE":
                    raise ValueError(
                        f"{call_id}: formal proof conflicts with independent simulation"
                    )
                success = True
                evidence = "unbounded_formal_proof"
            elif formal_status == "COUNTEREXAMPLE":
                if formal.get("counterexample_replayed") is not True:
                    raise ValueError(f"{call_id}: formal counterexample was not replayed")
                success = False
                evidence = "replayable_formal_counterexample"
            elif formal_status in {"TIMEOUT", "UNSUPPORTED"}:
                if simulation_status in {"PASS", "COUNTEREXAMPLE"}:
                    success = simulation_status == "PASS"
                    evidence = (
                        "independent_simulation_fallback_after_"
                        f"{formal_status.lower()}"
                    )
                elif allow_inconclusive_shift and mode == "shift":
                    success = None
                    resolved = False
                    evidence = "inconclusive_shift_validation"
                    inconclusive_reason = (
                        simulation_reason
                        or formal_reason
                        or (
                            f"formal_{formal_status.lower()}__"
                            f"simulation_{simulation_status.lower()}"
                        )
                    )
                else:
                    raise ValueError(
                        f"{call_id}: {formal_status} lacks a definitive independent simulation"
                    )
            else:
                raise ValueError(f"{call_id}: unknown or pending formal status {formal_status!r}")
        normalized.append(
            {
                "call_id": call_id,
                "mode": mode,
                "case_id": case_id,
                "seed_id": cluster_id,
                "arm": arm,
                "success": success,
                "resolved": resolved,
                "evidence": evidence,
                "formal_status": formal_status,
                "simulation_status": simulation_status,
                "reason": inconclusive_reason,
                "formal_reason": formal_reason,
                "simulation_reason": simulation_reason,
                "mutation_family": response.get("mutation_family"),
            }
        )
    return normalized


def _validated_records(
    records: Iterable[Mapping[str, Any]],
    keep_case_ids: set[str] | None = None,
    mode: str | None = None,
) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for row in records:
        row_mode = str(row.get("mode", "")).strip()
        if mode is not None and row_mode != mode:
            continue
        case_id = str(row.get("case_id", "")).strip()
        if keep_case_ids is not None and case_id not in keep_case_ids:
            continue
        seed_id = str(row.get("seed_id") or row.get("cluster_id") or "").strip()
        arm = str(row.get("arm", "")).strip()
        success = row.get("success")
        if not case_id or not seed_id or not arm:
            raise ValueError("every result needs non-empty case_id, seed_id, and arm")
        if not isinstance(success, bool):
            raise ValueError(f"{case_id}/{arm}: success must be a JSON boolean")
        key = (case_id, arm)
        if key in seen:
            raise ValueError(f"duplicate result for {case_id}/{arm}")
        seen.add(key)
        normalized.append(
            {
                "mode": row_mode,
                "case_id": case_id,
                "seed_id": seed_id,
                "arm": arm,
                "success": success,
            }
        )
    return normalized


def paired_seed_differences(
    records: Iterable[Mapping[str, Any]],
    arm_a: str,
    arm_b: str,
    *,
    keep_case_ids: set[str] | None = None,
    mode: str | None = None,
) -> dict[str, list[float]]:
    """Return paired B-A case differences grouped by seed.

    Pairing is strict: both arms must contain the exact same case IDs and each
    case must map to the same seed in both arms.  Missing model outputs are
    analysis errors, not implicit failures or silently changed denominators.
    """
    rows = _validated_records(records, keep_case_ids, mode)
    by_arm: dict[str, dict[str, dict[str, Any]]] = {arm_a: {}, arm_b: {}}
    for row in rows:
        if row["arm"] in by_arm:
            by_arm[row["arm"]][row["case_id"]] = row
    ids_a = set(by_arm[arm_a])
    ids_b = set(by_arm[arm_b])
    if not ids_a or not ids_b:
        raise ValueError(f"contrast {arm_a} vs {arm_b} has an empty arm")
    if ids_a != ids_b:
        only_a = sorted(ids_a - ids_b)[:5]
        only_b = sorted(ids_b - ids_a)[:5]
        raise ValueError(
            f"contrast {arm_a} vs {arm_b} is not exactly paired; "
            f"only_a={only_a}, only_b={only_b}"
        )
    grouped: dict[str, list[float]] = defaultdict(list)
    for case_id in sorted(ids_a):
        a = by_arm[arm_a][case_id]
        b = by_arm[arm_b][case_id]
        if a["seed_id"] != b["seed_id"]:
            raise ValueError(f"{case_id}: seed mismatch between arms")
        grouped[a["seed_id"]].append(float(b["success"]) - float(a["success"]))
    return dict(sorted(grouped.items()))


def interaction_seed_differences(
    records: Iterable[Mapping[str, Any]],
    arm_00: str,
    arm_10: str,
    arm_01: str,
    arm_11: str,
    *,
    keep_case_ids: set[str] | None = None,
) -> dict[str, list[float]]:
    """Return per-seed case-level 2x2 differences-in-differences.

    The coding is ``(11 - 01) - (10 - 00)``: how the spec effect changes when
    oracle location is present. All four cells must be exactly case-paired.
    """
    rows = _validated_records(records, keep_case_ids)
    arms = (arm_00, arm_10, arm_01, arm_11)
    indexed: dict[str, dict[str, dict[str, Any]]] = {arm: {} for arm in arms}
    for row in rows:
        if row["arm"] in indexed:
            indexed[row["arm"]][row["case_id"]] = row
    id_sets = [set(indexed[arm]) for arm in arms]
    if any(not ids for ids in id_sets):
        raise ValueError("2x2 interaction contains an empty arm")
    if any(ids != id_sets[0] for ids in id_sets[1:]):
        raise ValueError("2x2 interaction arms are not exactly case-paired")
    grouped: dict[str, list[float]] = defaultdict(list)
    for case_id in sorted(id_sets[0]):
        values = [indexed[arm][case_id] for arm in arms]
        seed_ids = {value["seed_id"] for value in values}
        if len(seed_ids) != 1:
            raise ValueError(f"{case_id}: seed mismatch across 2x2 arms")
        y00, y10, y01, y11 = (float(value["success"]) for value in values)
        grouped[values[0]["seed_id"]].append((y11 - y01) - (y10 - y00))
    return dict(sorted(grouped.items()))


def seed_macro_risk_difference(seed_differences: Mapping[str, Sequence[float]]) -> float:
    if not seed_differences:
        raise ValueError("seed-macro risk difference needs at least one seed")
    seed_means = [sum(values) / len(values) for values in seed_differences.values() if values]
    if len(seed_means) != len(seed_differences):
        raise ValueError("every seed cluster must contain at least one paired case")
    return sum(seed_means) / len(seed_means)


def cluster_paired_bootstrap(
    seed_differences: Mapping[str, Sequence[float]],
    *,
    bootstrap_samples: int = 10_000,
    rng_seed: int = 0,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Bootstrap seed clusters and return point, percentile CI, and two-sided p."""
    if bootstrap_samples <= 0:
        raise ValueError("bootstrap_samples must be positive")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    seed_ids = sorted(seed_differences)
    if not seed_ids:
        raise ValueError("bootstrap needs at least one seed")
    seed_means = {
        seed: sum(seed_differences[seed]) / len(seed_differences[seed]) for seed in seed_ids
    }
    point = sum(seed_means.values()) / len(seed_means)
    rng = random.Random(rng_seed)
    draws: list[float] = []
    for _ in range(bootstrap_samples):
        draw = [seed_means[seed_ids[rng.randrange(len(seed_ids))]] for _ in seed_ids]
        draws.append(sum(draw) / len(draw))
    draws.sort()
    lower = _quantile(draws, alpha / 2.0)
    upper = _quantile(draws, 1.0 - alpha / 2.0)

    # Add-one correction keeps a finite Monte Carlo p-value even when every
    # bootstrap replicate is on one side of zero.
    less_equal_zero = sum(value <= 0.0 for value in draws)
    greater_equal_zero = sum(value >= 0.0 for value in draws)
    tail = min(
        (less_equal_zero + 1) / (bootstrap_samples + 1),
        (greater_equal_zero + 1) / (bootstrap_samples + 1),
    )
    p_value = min(1.0, 2.0 * tail)
    case_count = sum(len(values) for values in seed_differences.values())
    return {
        "risk_difference": point,
        "ci95": [lower, upper],
        "p_value_unadjusted": p_value,
        "case_count": case_count,
        "seed_count": len(seed_ids),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_rng_seed": rng_seed,
        "estimand": "mean_seed_mean_paired_success_difference_b_minus_a",
    }


def holm_adjust(p_values: Mapping[str, float]) -> dict[str, float]:
    """Return monotone Holm-adjusted p-values keyed like the input mapping."""
    if not p_values:
        return {}
    for name, value in p_values.items():
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name}: p-value outside [0, 1]")
    ordered = sorted(p_values.items(), key=lambda item: (item[1], item[0]))
    total = len(ordered)
    adjusted: dict[str, float] = {}
    running_max = 0.0
    for rank, (name, value) in enumerate(ordered):
        candidate = min(1.0, (total - rank) * value)
        running_max = max(running_max, candidate)
        adjusted[name] = running_max
    return {name: adjusted[name] for name in p_values}


def is_material_improvement(result: Mapping[str, Any], minimum_effect: float = 0.10) -> bool:
    """Apply the preregistered language rule: CI excludes zero and lift >=10pp."""
    interval = result.get("ci95")
    point = result.get("risk_difference")
    if (
        not isinstance(interval, Sequence)
        or isinstance(interval, (str, bytes))
        or len(interval) != 2
        or not isinstance(point, (int, float))
    ):
        raise ValueError("result must contain numeric risk_difference and two-value ci95")
    return float(point) >= minimum_effect and float(interval[0]) > 0.0


def _ids_for_arm(records: Iterable[Mapping[str, Any]], arm: str) -> set[str]:
    return {
        str(row.get("case_id"))
        for row in records
        if str(row.get("arm")) == arm and row.get("case_id") is not None
    }


def _strict_shift_index(
    records: Sequence[Mapping[str, Any]],
    shift_case_ids: set[str],
    arm_a: str,
    arm_b: str,
) -> dict[str, dict[str, Mapping[str, Any]]]:
    """Validate the frozen 50-task x two-arm shift result grid."""
    if len(shift_case_ids) != 50:
        raise ValueError(
            f"frozen shift manifest must contain exactly 50 IDs, got {len(shift_case_ids)}"
        )
    expected_arms = {arm_a, arm_b}
    indexed: dict[str, dict[str, Mapping[str, Any]]] = {arm_a: {}, arm_b: {}}
    for row in records:
        if str(row.get("mode", "")) != "shift":
            raise ValueError("internal error: non-shift row entered shift validation")
        arm = str(row.get("arm", "")).strip()
        case_id = str(row.get("case_id", "")).strip()
        seed_id = str(row.get("seed_id") or row.get("cluster_id") or "").strip()
        if arm not in expected_arms:
            raise ValueError(f"shift results contain unexpected arm {arm!r}")
        if not case_id or not seed_id:
            raise ValueError("every shift result needs non-empty case_id and seed_id")
        if case_id not in shift_case_ids:
            raise ValueError(f"shift result case {case_id!r} is absent from the frozen manifest")
        if case_id in indexed[arm]:
            raise ValueError(f"duplicate shift result for {case_id}/{arm}")

        success = row.get("success")
        resolved = row.get("resolved")
        if isinstance(success, bool):
            if resolved is False:
                raise ValueError(f"{case_id}/{arm}: boolean success conflicts with resolved=false")
            if resolved is not None and resolved is not True:
                raise ValueError(f"{case_id}/{arm}: resolved must be a JSON boolean")
        elif success is None:
            if resolved is not False:
                raise ValueError(
                    f"{case_id}/{arm}: null success requires explicit resolved=false"
                )
            formal_status = str(row.get("formal_status", "")).upper()
            simulation_status = str(row.get("simulation_status", "")).upper()
            reason = row.get("reason")
            if formal_status not in {"TIMEOUT", "UNSUPPORTED"}:
                raise ValueError(
                    f"{case_id}/{arm}: inconclusive shift row has invalid formal status"
                )
            if simulation_status not in {"TIMEOUT", "UNSUPPORTED", "COMPILE_FAIL"}:
                raise ValueError(
                    f"{case_id}/{arm}: inconclusive shift row has invalid simulation status"
                )
            if not isinstance(reason, str) or not reason.strip():
                raise ValueError(
                    f"{case_id}/{arm}: inconclusive shift row lacks a reason"
                )
        else:
            raise ValueError(f"{case_id}/{arm}: success must be boolean or null")
        indexed[arm][case_id] = row

    for arm in (arm_a, arm_b):
        ids = set(indexed[arm])
        if ids != shift_case_ids:
            missing = sorted(shift_case_ids - ids)[:5]
            extra = sorted(ids - shift_case_ids)[:5]
            raise ValueError(
                "shift result IDs do not match the frozen shift manifest for "
                f"{arm}; missing={missing}, extra={extra}"
            )
    if len(records) != 100:
        raise ValueError(f"shift result grid must contain exactly 100 rows, got {len(records)}")
    for case_id in sorted(shift_case_ids):
        seed_a = str(
            indexed[arm_a][case_id].get("seed_id")
            or indexed[arm_a][case_id].get("cluster_id")
            or ""
        )
        seed_b = str(
            indexed[arm_b][case_id].get("seed_id")
            or indexed[arm_b][case_id].get("cluster_id")
            or ""
        )
        if seed_a != seed_b:
            raise ValueError(f"{case_id}: seed mismatch between shift arms")
    return indexed


def _inconclusive_shift_coverage(
    indexed: Mapping[str, Mapping[str, Mapping[str, Any]]],
    arm_a: str,
    arm_b: str,
) -> dict[str, Any]:
    """Build a coverage-only object with no complete-case effect fields."""

    def summarize(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        materialized = list(rows)
        unresolved = [row for row in materialized if row.get("success") is None]
        reason_counts: dict[str, int] = defaultdict(int)
        status_counts: dict[str, int] = defaultdict(int)
        for row in unresolved:
            reason_counts[str(row["reason"])] += 1
            status_key = (
                f"formal_{str(row['formal_status']).lower()}__"
                f"simulation_{str(row['simulation_status']).lower()}"
            )
            status_counts[status_key] += 1
        definitive = len(materialized) - len(unresolved)
        return {
            "expected_calls": 50,
            "observed_calls": len(materialized),
            "definitive_calls": definitive,
            "unresolved_calls": len(unresolved),
            "definitive_coverage": definitive / 50,
            "reason_counts": dict(sorted(reason_counts.items())),
            "status_counts": dict(sorted(status_counts.items())),
        }

    per_arm = {
        arm_a: summarize(indexed[arm_a].values()),
        arm_b: summarize(indexed[arm_b].values()),
    }
    unresolved_reason_counts: dict[str, int] = defaultdict(int)
    unresolved_status_counts: dict[str, int] = defaultdict(int)
    for arm in (arm_a, arm_b):
        for reason, count in per_arm[arm]["reason_counts"].items():
            unresolved_reason_counts[reason] += count
        for status, count in per_arm[arm]["status_counts"].items():
            unresolved_status_counts[status] += count
    definitive_calls = sum(item["definitive_calls"] for item in per_arm.values())
    unresolved_calls = sum(item["unresolved_calls"] for item in per_arm.values())
    fully_definitive_pair_count = sum(
        isinstance(indexed[arm_a][case_id].get("success"), bool)
        and isinstance(indexed[arm_b][case_id].get("success"), bool)
        for case_id in indexed[arm_a]
    )
    return {
        "status": "NO_ESTIMATE",
        "decision_timing": "POST_OUTPUT_CONSERVATIVE_DECISION",
        "estimand": "full_frozen_50_task_shift_set",
        "reason": (
            "post_output_validator_indeterminacy_precludes_"
            "complete_case_effect_estimation"
        ),
        "arm_a": arm_a,
        "arm_b": arm_b,
        "expected_cases": 50,
        "expected_calls": 100,
        "observed_calls": sum(item["observed_calls"] for item in per_arm.values()),
        "definitive_calls": definitive_calls,
        "unresolved_calls": unresolved_calls,
        "definitive_coverage": definitive_calls / 100,
        "fully_definitive_pair_count": fully_definitive_pair_count,
        "per_arm": per_arm,
        "reason_counts": dict(sorted(unresolved_reason_counts.items())),
        "status_counts": dict(sorted(unresolved_status_counts.items())),
        "complete_case_effect_reported": False,
        "unresolved_imputed": False,
        "secondary": True,
        "multiplicity_adjusted": False,
    }


def mutation_family_estimates(
    records: Sequence[Mapping[str, Any]],
    arm_a: str,
    arm_b: str,
    *,
    bootstrap_samples: int = 10_000,
    rng_seed: int = 0,
) -> dict[str, Any]:
    """Compute descriptive clustered paired effects separately by mutation family."""
    relevant = [row for row in records if str(row.get("arm")) in {arm_a, arm_b}]
    families: set[str] = set()
    for row in relevant:
        family = row.get("mutation_family", row.get("mutation"))
        if not isinstance(family, str) or not family.strip():
            raise ValueError(
                f"{row.get('case_id')}/{row.get('arm')}: mutation family is missing"
            )
        families.add(family.strip())
    estimates: dict[str, Any] = {}
    for offset, family in enumerate(sorted(families)):
        subset = [
            row
            for row in relevant
            if str(row.get("mutation_family", row.get("mutation"))).strip() == family
        ]
        differences = paired_seed_differences(subset, arm_a, arm_b)
        result = cluster_paired_bootstrap(
            differences,
            bootstrap_samples=bootstrap_samples,
            rng_seed=rng_seed + offset,
        )
        result.update(
            {
                "arm_a": arm_a,
                "arm_b": arm_b,
                "descriptive_only": True,
                "multiplicity_adjusted": False,
            }
        )
        estimates[family] = result
    return estimates


def analyze_secondaries(
    records: Sequence[Mapping[str, Any]],
    *,
    arm_names: Mapping[str, str] = SECONDARY_ARM_NAMES,
    main_keep_case_ids: set[str] | None = None,
    redacted_eligible_case_ids: set[str] | None = None,
    shift_case_ids: set[str] | None = None,
    cancel_inconclusive_shift: bool = False,
    bootstrap_samples: int = 10_000,
    rng_seed: int = 100,
) -> dict[str, Any]:
    """Analyze preregistered secondary effects without adding them to Holm.

    Redacted cases are restricted to the arm's explicit eligible IDs, which
    allows a frozen <=20% unredactable subset without treating missing prompts
    as model failures. The caller must omit the redacted arm entirely when its
    frozen decision is ``DROP_REDACTED_ARM``. A shift grid containing explicit
    null outcomes remains an error unless the caller makes the post-output,
    coverage-only cancellation decision explicit.
    """
    required = set(SECONDARY_ARM_NAMES)
    if set(arm_names) != required:
        raise ValueError(f"arm_names must contain exactly {sorted(required)}")
    no_spec = arm_names["no_spec_no_location"]
    full_spec = arm_names["full_spec_no_location"]
    no_spec_location = arm_names["no_spec_oracle_location"]
    full_spec_location = arm_names["full_spec_oracle_location"]
    main_records = [
        row
        for row in records
        if str(row.get("mode")) == "main"
        and (
            main_keep_case_ids is None
            or str(row.get("case_id")) in main_keep_case_ids
        )
    ]
    redacted_records = [
        row
        for row in records
        if str(row.get("mode")) == "redacted"
        and (
            main_keep_case_ids is None
            or str(row.get("case_id")) in main_keep_case_ids
        )
    ]
    shift_records = [row for row in records if str(row.get("mode")) == "shift"]
    if not main_records:
        raise ValueError("secondary analysis requires normalized mode='main' rows")

    interaction = cluster_paired_bootstrap(
        interaction_seed_differences(
            main_records,
            no_spec,
            full_spec,
            no_spec_location,
            full_spec_location,
        ),
        bootstrap_samples=bootstrap_samples,
        rng_seed=rng_seed,
    )
    interaction.update(
        {
            "contrast": "(full_spec_location - no_spec_location) - "
            "(full_spec_no_location - no_spec_no_location)",
            "secondary": True,
            "multiplicity_adjusted": False,
        }
    )

    redacted_arm = arm_names["redacted_spec_no_location"]
    redacted_ids = _ids_for_arm(redacted_records, redacted_arm)
    expected_redacted_ids = (
        redacted_eligible_case_ids & main_keep_case_ids
        if redacted_eligible_case_ids is not None and main_keep_case_ids is not None
        else redacted_eligible_case_ids
    )
    redacted: dict[str, Any] | None
    if redacted_ids:
        if expected_redacted_ids is None:
            raise ValueError(
                "redacted results require the eligible IDs from a frozen redaction manifest"
            )
        if redacted_ids != expected_redacted_ids:
            raise ValueError(
                "redacted result IDs do not match the frozen eligible redaction IDs"
            )
        redacted = cluster_paired_bootstrap(
            paired_seed_differences(
                [*main_records, *redacted_records],
                no_spec,
                redacted_arm,
                keep_case_ids=expected_redacted_ids,
            ),
            bootstrap_samples=bootstrap_samples,
            rng_seed=rng_seed + 1,
        )
        redacted.update(
            {
                "arm_a": no_spec,
                "arm_b": redacted_arm,
                "eligible_case_count": len(redacted_ids),
                "secondary": True,
                "multiplicity_adjusted": False,
            }
        )
    else:
        if expected_redacted_ids:
            raise ValueError("frozen redaction arm is eligible but has no result rows")
        redacted = None

    shift_a = arm_names["shift_no_spec"]
    shift_b = arm_names["shift_full_spec"]
    shift: dict[str, Any] | None
    if shift_records:
        if shift_case_ids is None:
            raise ValueError("shift results require case IDs from the frozen shift manifest")
        indexed = _strict_shift_index(shift_records, shift_case_ids, shift_a, shift_b)
        has_inconclusive = any(
            row.get("success") is None
            for arm_rows in indexed.values()
            for row in arm_rows.values()
        )
        if has_inconclusive:
            if not cancel_inconclusive_shift:
                raise ValueError(
                    "shift results contain inconclusive outcomes; explicitly cancel "
                    "the inferential shift effect to emit coverage only"
                )
            shift = _inconclusive_shift_coverage(indexed, shift_a, shift_b)
        else:
            shift = cluster_paired_bootstrap(
                paired_seed_differences(
                    shift_records, shift_a, shift_b, keep_case_ids=shift_case_ids
                ),
                bootstrap_samples=bootstrap_samples,
                rng_seed=rng_seed + 2,
            )
            shift.update(
                {
                    "arm_a": shift_a,
                    "arm_b": shift_b,
                    "secondary": True,
                    "multiplicity_adjusted": False,
                }
            )
    else:
        if shift_case_ids:
            raise ValueError("frozen shift sample has no result rows")
        shift = None

    families = {
        "what_full_spec": mutation_family_estimates(
            main_records,
            no_spec,
            full_spec,
            bootstrap_samples=bootstrap_samples,
            rng_seed=rng_seed + 10,
        ),
        "where_oracle_location": mutation_family_estimates(
            main_records,
            no_spec,
            no_spec_location,
            bootstrap_samples=bootstrap_samples,
            rng_seed=rng_seed + 20,
        ),
    }
    result = {
        "schema_version": 1,
        "kind": "vericodegen_secondary_clustered_analysis",
        "excluded_from_primary_holm_family": True,
        "interaction": interaction,
        "redacted_spec": redacted,
        "distribution_shift": shift,
        "mutation_family": families,
    }
    result["analysis_sha256"] = canonical_sha256(result)
    return result


def _validate_frozen_primary_coverage(
    records: Sequence[Mapping[str, Any]],
    keep_case_ids: set[str] | None,
) -> None:
    """Require the complete frozen main grid and a paired feedback cohort."""
    main_arms = ("spec0_loc0", "spec1_loc0", "spec0_loc1", "spec1_loc1")
    feedback_arms = ("generic_failure", "concrete_counterexample")
    expected_main_count = 75 if keep_case_ids is not None else 85
    if keep_case_ids is not None and len(keep_case_ids) != 75:
        raise ValueError(
            f"clean primary analysis requires exactly 75 case IDs, got {len(keep_case_ids)}"
        )

    main_rows = [
        row
        for row in records
        if str(row.get("mode", "")) == "main"
        and (
            keep_case_ids is None
            or str(row.get("case_id", "")) in keep_case_ids
        )
    ]
    indexed_main: dict[str, dict[str, Mapping[str, Any]]] = {
        arm: {} for arm in main_arms
    }
    for row in main_rows:
        arm = str(row.get("arm", ""))
        case_id = str(row.get("case_id", "")).strip()
        seed_id = str(row.get("seed_id") or row.get("cluster_id") or "").strip()
        if arm not in indexed_main:
            raise ValueError(f"main results contain unexpected arm {arm!r}")
        if not case_id or not seed_id:
            raise ValueError("every primary main row needs non-empty case_id and seed_id")
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"{case_id}/{arm}: primary main success must be boolean")
        if case_id in indexed_main[arm]:
            raise ValueError(f"duplicate primary main result for {case_id}/{arm}")
        indexed_main[arm][case_id] = row

    main_id_sets = [set(indexed_main[arm]) for arm in main_arms]
    if any(len(ids) != expected_main_count for ids in main_id_sets):
        counts = {arm: len(indexed_main[arm]) for arm in main_arms}
        raise ValueError(
            f"primary main coverage requires {expected_main_count} rows per arm; "
            f"got {counts}"
        )
    if any(ids != main_id_sets[0] for ids in main_id_sets[1:]):
        raise ValueError("primary main four-arm grids do not share the exact case set")
    if keep_case_ids is not None and main_id_sets[0] != keep_case_ids:
        raise ValueError("clean primary main IDs do not match the frozen clean75 set")
    for case_id in sorted(main_id_sets[0]):
        seed_ids = {
            str(indexed_main[arm][case_id].get("seed_id") or "") for arm in main_arms
        }
        if len(seed_ids) != 1:
            raise ValueError(f"{case_id}: seed mismatch across primary main arms")

    # clean75 is a case sensitivity filter: apply it to the corresponding
    # feedback cohort, never to unrelated shift-mode rows.
    feedback_rows = [
        row
        for row in records
        if str(row.get("mode", "")) == "feedback"
        and (
            keep_case_ids is None
            or str(row.get("case_id", "")) in keep_case_ids
        )
    ]
    indexed_feedback: dict[str, dict[str, Mapping[str, Any]]] = {
        arm: {} for arm in feedback_arms
    }
    for row in feedback_rows:
        arm = str(row.get("arm", ""))
        case_id = str(row.get("case_id", "")).strip()
        seed_id = str(row.get("seed_id") or row.get("cluster_id") or "").strip()
        if arm not in indexed_feedback:
            raise ValueError(f"feedback results contain unexpected arm {arm!r}")
        if not case_id or not seed_id:
            raise ValueError("every primary feedback row needs non-empty case_id and seed_id")
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"{case_id}/{arm}: primary feedback success must be boolean")
        if case_id in indexed_feedback[arm]:
            raise ValueError(f"duplicate primary feedback result for {case_id}/{arm}")
        indexed_feedback[arm][case_id] = row
    ids_generic = set(indexed_feedback[feedback_arms[0]])
    ids_concrete = set(indexed_feedback[feedback_arms[1]])
    if not ids_generic or not ids_concrete:
        raise ValueError("primary feedback comparison requires a non-empty two-arm cohort")
    if ids_generic != ids_concrete:
        raise ValueError("primary feedback arms do not share the exact cohort")
    if not ids_generic.issubset(main_id_sets[0]):
        raise ValueError("primary feedback cohort is not contained in the analyzed main cases")
    for case_id in sorted(ids_generic):
        seed_a = str(indexed_feedback[feedback_arms[0]][case_id].get("seed_id") or "")
        seed_b = str(indexed_feedback[feedback_arms[1]][case_id].get("seed_id") or "")
        if seed_a != seed_b:
            raise ValueError(f"{case_id}: seed mismatch between primary feedback arms")
        main_seed = str(indexed_main[main_arms[0]][case_id].get("seed_id") or "")
        if seed_a != main_seed:
            raise ValueError(f"{case_id}: feedback seed does not match primary main seed")


def analyze_contrasts(
    records: Sequence[Mapping[str, Any]],
    contrasts: Sequence[tuple[str, ...]],
    *,
    keep_case_ids: set[str] | None = None,
    bootstrap_samples: int = 10_000,
    rng_seed: int = 0,
) -> dict[str, Any]:
    if tuple(tuple(contrast) for contrast in contrasts) == DEFAULT_CONTRASTS:
        _validate_frozen_primary_coverage(records, keep_case_ids)
    results: dict[str, dict[str, Any]] = {}
    for offset, contrast in enumerate(contrasts):
        if len(contrast) == 3:
            name, arm_a, arm_b = contrast
            mode = None
        elif len(contrast) == 4:
            name, mode, arm_a, arm_b = contrast
        else:
            raise ValueError("contrast must be (name,a,b) or (name,mode,a,b)")
        differences = paired_seed_differences(
            records, arm_a, arm_b, keep_case_ids=keep_case_ids, mode=mode
        )
        result = cluster_paired_bootstrap(
            differences,
            bootstrap_samples=bootstrap_samples,
            rng_seed=rng_seed + offset,
        )
        result.update({"arm_a": arm_a, "arm_b": arm_b, "mode": mode})
        results[name] = result
    adjusted = holm_adjust(
        {name: result["p_value_unadjusted"] for name, result in results.items()}
    )
    for name, result in results.items():
        result["p_value_holm"] = adjusted[name]
        result["material_improvement"] = is_material_improvement(result)
    analysis = {
        "schema_version": 1,
        "kind": "vericodegen_primary_clustered_analysis",
        "success_definition": "pre-normalized frozen evaluation-policy boolean",
        "case_filter_count": len(keep_case_ids) if keep_case_ids is not None else None,
        "holm_family": list(results),
        "contrast_definitions": [list(contrast) for contrast in contrasts],
        "protocol_amendment": protocol_amendment_binding(),
        "contrasts": results,
    }
    analysis["analysis_sha256"] = canonical_sha256(analysis)
    return analysis


def load_materialized_id_map(path: Path | str) -> dict[str, str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_manifest(manifest)
    if manifest.get("kind") not in {
        "vericodegen_main85_materialized_inputs",
        "vericodegen_shift50_materialized_inputs",
    }:
        raise ValueError("not a materialized runner-input manifest")
    index = manifest.get("case_index")
    if not isinstance(index, list):
        raise ValueError("materialized input manifest lacks case_index")
    mapping: dict[str, str] = {}
    for item in index:
        if not isinstance(item, Mapping):
            raise ValueError("invalid materialized case index row")
        internal = str(item.get("case_id", ""))
        anonymous = str(item.get("anonymous_id", ""))
        if not internal or not anonymous or internal in mapping:
            raise ValueError("invalid or duplicate materialized ID mapping")
        mapping[internal] = anonymous
    return mapping


def load_clean_case_ids(
    path: Path | str, id_map: Mapping[str, str] | None = None
) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_manifest(manifest)
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("clean manifest lacks cases")
    internal_ids = {str(item.get("case_id")) for item in cases if isinstance(item, Mapping)}
    if id_map is not None and not internal_ids.issubset(id_map):
        missing = sorted(internal_ids - set(id_map))[:5]
        raise ValueError(f"clean75 cases missing from materialized ID map: {missing}")
    ids = (
        {str(id_map[case_id]) for case_id in internal_ids}
        if id_map is not None
        else internal_ids
    )
    if len(ids) != 75:
        raise ValueError(f"clean75 manifest contains {len(ids)} unique case IDs")
    return ids


def load_redacted_eligible_case_ids(
    path: Path | str, id_map: Mapping[str, str] | None = None
) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    validate_redaction_decision_manifest(manifest)
    verify_manifest(manifest)
    decision = manifest.get("arm_decision")
    ids = manifest.get("eligible_case_ids")
    if decision == "DROP_REDACTED_ARM":
        return set()
    if decision != "RUN_REDACTED_ARM" or not isinstance(ids, list):
        raise ValueError("redaction manifest lacks a valid frozen arm decision")
    internal = {str(case_id) for case_id in ids}
    if id_map is not None and not internal.issubset(id_map):
        missing = sorted(internal - set(id_map))[:5]
        raise ValueError(f"redaction cases missing from materialized ID map: {missing}")
    result = {str(id_map[case_id]) for case_id in internal} if id_map else internal
    if len(result) != manifest.get("eligible_case_count"):
        raise ValueError("redaction eligible IDs/count mismatch")
    return result


def load_shift_case_ids(path: Path | str) -> set[str]:
    with Path(path).open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    verify_manifest(manifest)
    if manifest.get("kind") == "vericodegen_shift50_materialized_inputs":
        ids = set(load_materialized_id_map(path).values())
        if len(ids) != 50:
            raise ValueError("materialized shift manifest must contain 50 anonymous IDs")
        return ids
    if manifest.get("kind") != "vericodegen_distribution_shift_sample":
        raise ValueError("not a frozen shift selection or materialized-input manifest")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        raise ValueError("shift manifest lacks cases")
    ids = {
        str(item.get("record_id"))
        for item in cases
        if isinstance(item, Mapping) and item.get("record_id") is not None
    }
    if len(ids) != 50 or len(ids) != manifest.get("case_count"):
        raise ValueError("shift manifest must contain 50 unique record IDs")
    return ids


def _parse_contrast(value: str) -> tuple[str, ...]:
    pieces = value.split(":")
    if len(pieces) not in {3, 4} or any(not piece.strip() for piece in pieces):
        raise argparse.ArgumentTypeError(
            "contrast must be NAME:ARM_A:ARM_B or NAME:MODE:ARM_A:ARM_B"
        )
    return tuple(piece.strip() for piece in pieces)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "records", type=Path, help="normalized result JSONL, or runner ledger with --from-ledger"
    )
    parser.add_argument(
        "--from-ledger",
        action="store_true",
        help="normalize response/verdict events from the append-only runner ledger first",
    )
    parser.add_argument(
        "--normalized-output",
        type=Path,
        help="optionally save the frozen normalized rows used for analysis",
    )
    parser.add_argument("--contrast", action="append", type=_parse_contrast)
    parser.add_argument("--clean75", type=Path, help="optional clean75 manifest")
    parser.add_argument(
        "--main-input-manifest",
        type=Path,
        help="map internal clean/redaction IDs to the runner's anonymous IDs",
    )
    parser.add_argument(
        "--redaction-manifest",
        type=Path,
        help="frozen redaction manifest, required when redacted-arm rows are present",
    )
    parser.add_argument(
        "--shift-manifest",
        type=Path,
        help="frozen shift50 manifest, required when shift-arm rows are present",
    )
    parser.add_argument("--bootstrap", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--include-secondaries",
        action="store_true",
        help=(
            "also require/analyze the 2x2 cells and mutation metadata; "
            "optional redacted/shift arms"
        ),
    )
    parser.add_argument(
        "--cancel-inconclusive-shift",
        action="store_true",
        help=(
            "retain terminal inconclusive shift rows and replace the shift effect "
            "with a post-output coverage-only NO_ESTIMATE object"
        ),
    )
    args = parser.parse_args()

    if args.contrast:
        parser.error(
            "the final analysis CLI uses the three frozen primary contrasts; "
            "custom --contrast definitions are not permitted"
        )
    if args.cancel_inconclusive_shift:
        missing = []
        if not args.from_ledger:
            missing.append("--from-ledger")
        if not args.include_secondaries:
            missing.append("--include-secondaries")
        if args.shift_manifest is None:
            missing.append("--shift-manifest")
        if missing:
            parser.error(
                "--cancel-inconclusive-shift requires " + ", ".join(missing)
            )

    source_records = read_jsonl(args.records)
    records = (
        normalize_ledger_events(
            source_records,
            allow_inconclusive_shift=args.cancel_inconclusive_shift,
        )
        if args.from_ledger
        else source_records
    )
    if args.normalized_output:
        write_jsonl_atomic(args.normalized_output, records)
    contrasts = DEFAULT_CONTRASTS
    main_id_map = (
        load_materialized_id_map(args.main_input_manifest)
        if args.main_input_manifest
        else None
    )
    keep = load_clean_case_ids(args.clean75, main_id_map) if args.clean75 else None
    analysis = analyze_contrasts(
        records,
        contrasts,
        keep_case_ids=keep,
        bootstrap_samples=args.bootstrap,
        rng_seed=args.seed,
    )
    if args.include_secondaries:
        analysis["secondaries"] = analyze_secondaries(
            records,
            main_keep_case_ids=keep,
            redacted_eligible_case_ids=(
                load_redacted_eligible_case_ids(args.redaction_manifest, main_id_map)
                if args.redaction_manifest
                else None
            ),
            shift_case_ids=(
                load_shift_case_ids(args.shift_manifest) if args.shift_manifest else None
            ),
            cancel_inconclusive_shift=args.cancel_inconclusive_shift,
            bootstrap_samples=args.bootstrap,
            rng_seed=args.seed + 100,
        )
        analysis["analysis_sha256"] = canonical_sha256(
            {key: value for key, value in analysis.items() if key != "analysis_sha256"}
        )
    if args.output:
        write_json_atomic(args.output, analysis)
        print(f"wrote {args.output} ({analysis['analysis_sha256']})")
    else:
        print(json.dumps(analysis, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
