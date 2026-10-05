from __future__ import annotations

import json
import unittest
from unittest.mock import patch

from benchmarks.vericodegen_stats import (
    DEFAULT_CONTRASTS,
    analyze_contrasts,
    analyze_secondaries,
    cluster_paired_bootstrap,
    holm_adjust,
    interaction_seed_differences,
    is_material_improvement,
    mutation_family_estimates,
    normalize_ledger_events,
    paired_seed_differences,
    seed_macro_risk_difference,
    main as stats_main,
)


def _row(
    case: str,
    seed: str,
    arm: str,
    success: bool | None,
    mode: str = "",
):
    return {
        "mode": mode,
        "case_id": case,
        "seed_id": seed,
        "arm": arm,
        "success": success,
    }


def _candidate_events(
    call_id: str,
    *,
    mode: str = "shift",
    arm: str = "spec0_loc0",
    formal: str = "UNSUPPORTED",
    simulation: str = "UNSUPPORTED",
):
    return [
        {
            "event": "response",
            "call_id": call_id,
            "mode": mode,
            "arm": arm,
            "case_id": call_id,
            "cluster_id": call_id,
            "model_outcome": "accepted_pending_compile",
        },
        {
            "event": "verdict",
            "call_id": call_id,
            "compile_verdict": {"passed": True},
            "formal_verdict": {"status": formal, "reason": "formal reason"},
            "simulation_verdict": {
                "status": simulation,
                "reason": "simulation reason",
            },
        },
    ]


def _secondary_main_records():
    records = []
    main_arms = {
        "spec0_loc0": False,
        "spec1_loc0": True,
        "spec0_loc1": False,
        "spec1_loc1": True,
    }
    for case, seed, family in (("m1", "s1", "op"), ("m2", "s2", "mux")):
        for arm, success in main_arms.items():
            row = _row(case, seed, arm, success, "main")
            row["mutation_family"] = family
            records.append(row)
    return records


def _shift_grid(
    *,
    unresolved_a: set[int] | None = None,
    unresolved_b: set[int] | None = None,
):
    unresolved_a = unresolved_a or set()
    unresolved_b = unresolved_b or set()
    records = []
    for index in range(50):
        case = f"x{index:02d}"
        for arm, unresolved, success in (
            ("spec0_loc0", index in unresolved_a, False),
            ("spec1_loc0", index in unresolved_b, index % 2 == 0),
        ):
            row = _row(case, case, arm, None if unresolved else success, "shift")
            if unresolved:
                if arm == "spec0_loc0":
                    simulation_reason = "ansi_or_interface_parse_failure"
                elif index < 7:
                    simulation_reason = "no_reset_or_legal_preamble"
                else:
                    simulation_reason = "inconsistent_clock_edge"
                row.update(
                    {
                        "resolved": False,
                        "formal_status": "UNSUPPORTED",
                        "simulation_status": "TIMEOUT",
                        "reason": simulation_reason,
                        "formal_reason": "unsupported construct",
                        "simulation_reason": simulation_reason,
                    }
                )
            else:
                row["resolved"] = True
            records.append(row)
    return records


def _full_primary_records(feedback_cases: tuple[int, ...] = (0, 1, 2)):
    records = []
    for index in range(85):
        case = f"m{index:03d}"
        seed = f"s{index // 2:03d}"
        for arm, success in (
            ("spec0_loc0", False),
            ("spec1_loc0", index % 2 == 0),
            ("spec0_loc1", True),
            ("spec1_loc1", True),
        ):
            records.append(_row(case, seed, arm, success, "main"))
    for index in feedback_cases:
        case = f"m{index:03d}"
        seed = f"s{index // 2:03d}"
        records.append(_row(case, seed, "generic_failure", False, "feedback"))
        records.append(
            _row(case, seed, "concrete_counterexample", True, "feedback")
        )
    return records


class ClusteredStatisticsTests(unittest.TestCase):
    def test_ledger_normalization_uses_formal_then_simulation_fallback(self):
        events = []
        cases = (
            ("proof", "PROVED", "PASS", True),
            ("ce", "COUNTEREXAMPLE", "COUNTEREXAMPLE", False),
            ("unsupported", "UNSUPPORTED", "PASS", True),
            ("timeout", "TIMEOUT", "COUNTEREXAMPLE", False),
        )
        for case, formal, simulation, _ in cases:
            events.append(
                {
                    "event": "response",
                    "call_id": case,
                    "mode": "main",
                    "arm": "spec0_loc0",
                    "case_id": case,
                    "cluster_id": case,
                    "model_outcome": "accepted_pending_compile",
                }
            )
            events.append(
                {
                    "event": "verdict",
                    "call_id": case,
                    "compile_verdict": {"passed": True},
                    "formal_verdict": {
                        "status": formal,
                        "counterexample_replayed": formal == "COUNTEREXAMPLE",
                    },
                    "simulation_verdict": {"status": simulation},
                }
            )
        events.append(
            {
                "event": "response",
                "call_id": "refusal",
                "mode": "main",
                "arm": "spec0_loc0",
                "case_id": "refusal",
                "cluster_id": "refusal",
                "model_outcome": "refusal",
            }
        )
        normalized = {row["case_id"]: row for row in normalize_ledger_events(events)}
        for case, _, _, expected in cases:
            self.assertEqual(normalized[case]["success"], expected)
        self.assertFalse(normalized["refusal"]["success"])

    def test_ledger_normalization_rejects_formal_simulation_conflict(self):
        events = [
            {
                "event": "response",
                "call_id": "c",
                "mode": "main",
                "arm": "spec0_loc0",
                "case_id": "c",
                "cluster_id": "s",
                "model_outcome": "accepted_pending_compile",
            },
            {
                "event": "verdict",
                "call_id": "c",
                "compile_verdict": {"passed": True},
                "formal_verdict": {"status": "PROVED"},
                "simulation_verdict": {"status": "COUNTEREXAMPLE"},
            },
        ]
        with self.assertRaisesRegex(ValueError, "conflicts"):
            normalize_ledger_events(events)

    def test_ledger_normalization_rejects_duplicate_or_orphan_terminals(self):
        response = {
            "event": "response",
            "call_id": "duplicate",
            "mode": "main",
            "arm": "spec0_loc0",
            "case_id": "c",
            "cluster_id": "s",
            "model_outcome": "refusal",
        }
        with self.assertRaisesRegex(ValueError, "duplicate terminal response"):
            normalize_ledger_events([response, dict(response)])
        with self.assertRaisesRegex(ValueError, "orphan verdict"):
            normalize_ledger_events(
                [{"event": "verdict", "call_id": "orphan"}]
            )

    def test_ledger_normalization_rejects_compile_formal_contradiction(self):
        events = [
            {
                "event": "response",
                "call_id": "c",
                "mode": "main",
                "arm": "spec0_loc0",
                "case_id": "c",
                "cluster_id": "s",
                "model_outcome": "accepted_pending_compile",
            },
            {
                "event": "verdict",
                "call_id": "c",
                "compile_verdict": {"passed": True},
                "formal_verdict": {"status": "COMPILE_FAIL"},
                "simulation_verdict": {"status": "COMPILE_FAIL"},
            },
        ]
        with self.assertRaisesRegex(ValueError, "contradict"):
            normalize_ledger_events(events)

    def test_inconclusive_shift_normalization_is_explicit_and_fail_closed(self):
        events = _candidate_events("shift-null")
        with self.assertRaisesRegex(ValueError, "definitive independent simulation"):
            normalize_ledger_events(events)

        normalized = normalize_ledger_events(
            events, allow_inconclusive_shift=True
        )
        self.assertEqual(len(normalized), 1)
        row = normalized[0]
        self.assertIsNone(row["success"])
        self.assertFalse(row["resolved"])
        self.assertEqual(row["evidence"], "inconclusive_shift_validation")
        self.assertEqual(row["formal_status"], "UNSUPPORTED")
        self.assertEqual(row["simulation_status"], "UNSUPPORTED")
        self.assertEqual(row["reason"], "simulation reason")
        self.assertEqual(row["formal_reason"], "formal reason")
        self.assertEqual(row["simulation_reason"], "simulation reason")

    def test_inconclusive_reason_prefers_simulation_then_formal_then_status(self):
        formal_fallback = _candidate_events("formal-fallback")
        formal_fallback[1]["simulation_verdict"].pop("reason")
        row = normalize_ledger_events(
            formal_fallback, allow_inconclusive_shift=True
        )[0]
        self.assertEqual(row["reason"], "formal reason")

        status_fallback = _candidate_events("status-fallback")
        status_fallback[1]["simulation_verdict"].pop("reason")
        status_fallback[1]["formal_verdict"].pop("reason")
        row = normalize_ledger_events(
            status_fallback, allow_inconclusive_shift=True
        )[0]
        self.assertEqual(
            row["reason"], "formal_unsupported__simulation_unsupported"
        )

    def test_inconclusive_opt_in_never_relaxes_primary_or_pending_validation(self):
        for mode in ("main", "feedback"):
            with self.subTest(mode=mode):
                with self.assertRaisesRegex(
                    ValueError, "definitive independent simulation"
                ):
                    normalize_ledger_events(
                        _candidate_events(f"{mode}-null", mode=mode),
                        allow_inconclusive_shift=True,
                    )

        missing_verdict = _candidate_events("missing-verdict")[:1]
        with self.assertRaisesRegex(ValueError, "no terminal validation"):
            normalize_ledger_events(
                missing_verdict, allow_inconclusive_shift=True
            )
        for status in ("PENDING", "MYSTERY"):
            with self.subTest(simulation_status=status):
                with self.assertRaisesRegex(
                    ValueError, "unknown or pending simulation status"
                ):
                    normalize_ledger_events(
                        _candidate_events("bad-simulation", simulation=status),
                        allow_inconclusive_shift=True,
                    )

    def test_normalized_output_keeps_all_450_research_rows_and_14_nulls(self):
        events = []
        main_arms = ("spec0_loc0", "spec1_loc0", "spec0_loc1", "spec1_loc1")
        for index in range(340):
            events.append(
                {
                    "event": "response",
                    "call_id": f"main-{index}",
                    "mode": "main",
                    "arm": main_arms[index % 4],
                    "case_id": f"m{index // 4:03d}",
                    "cluster_id": f"s{index // 8:03d}",
                    "model_outcome": "refusal",
                }
            )
        for arm_index, arm in enumerate(("spec0_loc0", "spec1_loc0")):
            unresolved_count = 6 if arm_index == 0 else 8
            for index in range(50):
                call_id = f"shift-{arm_index}-{index}"
                if index < unresolved_count:
                    events.extend(
                        _candidate_events(call_id, mode="shift", arm=arm)
                    )
                else:
                    events.append(
                        {
                            "event": "response",
                            "call_id": call_id,
                            "mode": "shift",
                            "arm": arm,
                            "case_id": f"x{index:02d}",
                            "cluster_id": f"x{index:02d}",
                            "model_outcome": "refusal",
                        }
                    )
        for index in range(5):
            for arm in ("generic_failure", "concrete_counterexample"):
                events.append(
                    {
                        "event": "response",
                        "call_id": f"feedback-{index}-{arm}",
                        "mode": "feedback",
                        "arm": arm,
                        "case_id": f"m{index:03d}",
                        "cluster_id": f"s{index // 2:03d}",
                        "model_outcome": "refusal",
                    }
                )
        normalized = normalize_ledger_events(
            events, allow_inconclusive_shift=True
        )
        self.assertEqual(len(normalized), 450)
        self.assertEqual(sum(row["success"] is None for row in normalized), 14)
        self.assertEqual(
            sum(row["mode"] == "shift" for row in normalized), 100
        )

    def test_seed_macro_weights_designs_not_mutation_rows(self):
        records = [
            _row("a1", "seed-a", "a", False),
            _row("a1", "seed-a", "b", True),
            _row("a2", "seed-a", "a", False),
            _row("a2", "seed-a", "b", True),
            _row("b1", "seed-b", "a", True),
            _row("b1", "seed-b", "b", False),
        ]
        differences = paired_seed_differences(records, "a", "b")
        self.assertEqual(differences, {"seed-a": [1.0, 1.0], "seed-b": [-1.0]})
        self.assertEqual(seed_macro_risk_difference(differences), 0.0)

    def test_pairing_is_strict(self):
        records = [
            _row("case-1", "seed", "a", False),
            _row("case-2", "seed", "b", True),
        ]
        with self.assertRaisesRegex(ValueError, "not exactly paired"):
            paired_seed_differences(records, "a", "b")

    def test_bootstrap_is_reproducible(self):
        differences = {"s1": [1.0], "s2": [0.0], "s3": [1.0]}
        first = cluster_paired_bootstrap(
            differences, bootstrap_samples=1_000, rng_seed=42
        )
        second = cluster_paired_bootstrap(
            differences, bootstrap_samples=1_000, rng_seed=42
        )
        self.assertEqual(first, second)
        self.assertAlmostEqual(first["risk_difference"], 2 / 3)
        self.assertGreater(first["ci95"][0], -1e-12)

    def test_holm_is_monotone_in_sorted_order(self):
        adjusted = holm_adjust({"first": 0.01, "third": 0.04, "second": 0.03})
        self.assertAlmostEqual(adjusted["first"], 0.03)
        self.assertAlmostEqual(adjusted["second"], 0.06)
        self.assertAlmostEqual(adjusted["third"], 0.06)

    def test_material_language_requires_ci_and_ten_points(self):
        self.assertTrue(
            is_material_improvement({"risk_difference": 0.10, "ci95": [0.01, 0.20]})
        )
        self.assertFalse(
            is_material_improvement({"risk_difference": 0.09, "ci95": [0.01, 0.20]})
        )
        self.assertFalse(
            is_material_improvement({"risk_difference": 0.20, "ci95": [0.0, 0.30]})
        )

    def test_analysis_applies_one_holm_family(self):
        records = []
        for case, seed in (("c1", "s1"), ("c2", "s2"), ("c3", "s3")):
            for arm, value in (("a", False), ("b", True), ("c", False), ("d", True)):
                records.append(_row(case, seed, arm, value))
        analysis = analyze_contrasts(
            records,
            (("one", "a", "b"), ("two", "c", "d")),
            bootstrap_samples=100,
            rng_seed=3,
        )
        self.assertEqual(analysis["holm_family"], ["one", "two"])
        self.assertEqual(
            analysis["contrast_definitions"],
            [["one", "a", "b"], ["two", "c", "d"]],
        )
        self.assertEqual(len(analysis["protocol_amendment"]["file_sha256"]), 64)
        self.assertIn("p_value_holm", analysis["contrasts"]["one"])

    def test_mode_scope_prevents_shift_arm_name_collision(self):
        records = [
            _row("main", "s1", "spec0_loc0", False, "main"),
            _row("main", "s1", "spec1_loc0", True, "main"),
            _row("shift", "x1", "spec0_loc0", None, "shift"),
            _row("shift", "x1", "spec1_loc0", False, "shift"),
        ]
        analysis = analyze_contrasts(
            records,
            (("what", "main", "spec0_loc0", "spec1_loc0"),),
            bootstrap_samples=100,
        )
        self.assertEqual(analysis["contrasts"]["what"]["risk_difference"], 1.0)
        self.assertEqual(analysis["contrasts"]["what"]["case_count"], 1)

    def test_keep_filter_precedes_boolean_validation(self):
        records = [
            _row("kept", "s1", "a", False, "main"),
            _row("kept", "s1", "b", True, "main"),
            _row("outside", "s2", "a", None, "main"),
            _row("outside", "s2", "b", None, "main"),
        ]
        analysis = analyze_contrasts(
            records,
            (("what", "main", "a", "b"),),
            keep_case_ids={"kept"},
            bootstrap_samples=20,
        )
        self.assertEqual(analysis["contrasts"]["what"]["case_count"], 1)

    def test_interaction_is_clustered_difference_in_differences(self):
        records = [
            _row("c1", "s1", "00", False),
            _row("c1", "s1", "10", True),
            _row("c1", "s1", "01", True),
            _row("c1", "s1", "11", True),
        ]
        differences = interaction_seed_differences(records, "00", "10", "01", "11")
        self.assertEqual(differences, {"s1": [-1.0]})

    def test_mutation_family_outputs_are_descriptive(self):
        records = []
        for case, seed, family in (("c1", "s1", "op"), ("c2", "s2", "mux")):
            a = _row(case, seed, "a", False)
            b = _row(case, seed, "b", True)
            a["mutation"] = family
            b["mutation"] = family
            records.extend((a, b))
        estimates = mutation_family_estimates(
            records, "a", "b", bootstrap_samples=100
        )
        self.assertEqual(set(estimates), {"mux", "op"})
        self.assertTrue(estimates["mux"]["descriptive_only"])
        self.assertFalse(estimates["op"]["multiplicity_adjusted"])

    def test_secondary_bundle_keeps_redacted_and_shift_outside_holm(self):
        records = []
        main_arms = {
            "spec0_loc0": False,
            "spec1_loc0": True,
            "spec0_loc1": False,
            "spec1_loc1": True,
        }
        for case, seed, family in (("m1", "s1", "op"), ("m2", "s2", "mux")):
            for arm, success in main_arms.items():
                row = _row(case, seed, arm, success, "main")
                row["mutation"] = family
                records.append(row)
            redacted = _row(case, seed, "redacted_spec_loc0", False, "redacted")
            redacted["mutation"] = family
            records.append(redacted)
        shift_ids = {f"x{index:02d}" for index in range(50)}
        for index in range(50):
            case = f"x{index:02d}"
            records.append(_row(case, case, "spec0_loc0", False, "shift"))
            records.append(
                _row(case, case, "spec1_loc0", index % 2 == 0, "shift")
            )
        secondary = analyze_secondaries(
            records,
            redacted_eligible_case_ids={"m1", "m2"},
            shift_case_ids=shift_ids,
            bootstrap_samples=100,
        )
        self.assertTrue(secondary["excluded_from_primary_holm_family"])
        self.assertEqual(secondary["redacted_spec"]["eligible_case_count"], 2)
        self.assertEqual(secondary["distribution_shift"]["case_count"], 50)
        self.assertIn("op", secondary["mutation_family"]["what_full_spec"])

    def test_inconclusive_shift_requires_explicit_cancellation(self):
        records = [
            *_secondary_main_records(),
            *_shift_grid(unresolved_a=set(range(6)), unresolved_b=set(range(8))),
        ]
        with self.assertRaisesRegex(ValueError, "explicitly cancel"):
            analyze_secondaries(
                records,
                redacted_eligible_case_ids=set(),
                shift_case_ids={f"x{index:02d}" for index in range(50)},
                bootstrap_samples=20,
            )

    def test_cancelled_shift_returns_coverage_only_no_estimate(self):
        records = [
            *_secondary_main_records(),
            *_shift_grid(unresolved_a=set(range(6)), unresolved_b=set(range(8))),
        ]
        secondary = analyze_secondaries(
            records,
            redacted_eligible_case_ids=set(),
            shift_case_ids={f"x{index:02d}" for index in range(50)},
            cancel_inconclusive_shift=True,
            bootstrap_samples=20,
        )
        shift = secondary["distribution_shift"]
        self.assertEqual(shift["status"], "NO_ESTIMATE")
        self.assertEqual(
            shift["decision_timing"], "POST_OUTPUT_CONSERVATIVE_DECISION"
        )
        self.assertEqual(shift["estimand"], "full_frozen_50_task_shift_set")
        self.assertEqual(shift["expected_cases"], 50)
        self.assertEqual(shift["expected_calls"], 100)
        self.assertEqual(shift["observed_calls"], 100)
        self.assertEqual(shift["definitive_calls"], 86)
        self.assertEqual(shift["unresolved_calls"], 14)
        self.assertEqual(shift["fully_definitive_pair_count"], 42)
        self.assertEqual(
            shift["definitive_calls"] + shift["unresolved_calls"],
            shift["observed_calls"],
        )
        self.assertEqual(
            shift["per_arm"]["spec0_loc0"]["definitive_calls"], 44
        )
        self.assertEqual(
            shift["per_arm"]["spec0_loc0"]["unresolved_calls"], 6
        )
        self.assertEqual(
            shift["per_arm"]["spec1_loc0"]["definitive_calls"], 42
        )
        self.assertEqual(
            shift["per_arm"]["spec1_loc0"]["unresolved_calls"], 8
        )
        self.assertEqual(
            shift["reason_counts"],
            {
                "ansi_or_interface_parse_failure": 6,
                "inconsistent_clock_edge": 1,
                "no_reset_or_legal_preamble": 7,
            },
        )
        self.assertEqual(
            shift["status_counts"],
            {"formal_unsupported__simulation_timeout": 14},
        )
        self.assertFalse(shift["complete_case_effect_reported"])
        self.assertFalse(shift["unresolved_imputed"])

        forbidden = {
            "risk_difference",
            "ci95",
            "p_value_unadjusted",
            "p_value_holm",
            "bootstrap_samples",
            "bootstrap_rng_seed",
        }

        def all_keys(value):
            if isinstance(value, dict):
                for key, nested in value.items():
                    yield key
                    yield from all_keys(nested)
            elif isinstance(value, list):
                for nested in value:
                    yield from all_keys(nested)

        self.assertTrue(forbidden.isdisjoint(set(all_keys(shift))))
        self.assertNotIn("NaN", json.dumps(shift, allow_nan=False))

    def test_shift_grid_and_inconclusive_row_validation_are_strict(self):
        records = [*_secondary_main_records(), *_shift_grid()]
        with self.assertRaisesRegex(ValueError, "exactly 50 IDs"):
            analyze_secondaries(
                records,
                redacted_eligible_case_ids=set(),
                shift_case_ids={f"x{index:02d}" for index in range(49)},
                bootstrap_samples=20,
            )

        malformed = _shift_grid(unresolved_a={0})
        for row in malformed:
            if row["success"] is None:
                row["simulation_status"] = "PENDING"
        with self.assertRaisesRegex(ValueError, "invalid simulation status"):
            analyze_secondaries(
                [*_secondary_main_records(), *malformed],
                redacted_eligible_case_ids=set(),
                shift_case_ids={f"x{index:02d}" for index in range(50)},
                cancel_inconclusive_shift=True,
                bootstrap_samples=20,
            )

    def test_cancel_flag_with_fully_resolved_shift_keeps_bootstrap_estimate(self):
        secondary = analyze_secondaries(
            [*_secondary_main_records(), *_shift_grid()],
            redacted_eligible_case_ids=set(),
            shift_case_ids={f"x{index:02d}" for index in range(50)},
            cancel_inconclusive_shift=True,
            bootstrap_samples=20,
        )
        shift = secondary["distribution_shift"]
        self.assertIn("risk_difference", shift)
        self.assertIn("ci95", shift)
        self.assertNotIn("status", shift)

    def test_frozen_primary_requires_full_85_grid_and_paired_feedback(self):
        records = _full_primary_records()
        analysis = analyze_contrasts(
            records,
            DEFAULT_CONTRASTS,
            bootstrap_samples=20,
        )
        self.assertEqual(
            analysis["contrasts"]["what_full_spec"]["case_count"], 85
        )
        self.assertEqual(
            analysis["contrasts"]["where_oracle_location"]["case_count"], 85
        )
        self.assertEqual(
            analysis["contrasts"]["counterexample_revision"]["case_count"], 3
        )

        missing_cell = [
            row
            for row in records
            if not (
                row["mode"] == "main"
                and row["arm"] == "spec1_loc1"
                and row["case_id"] == "m084"
            )
        ]
        with self.assertRaisesRegex(ValueError, "85 rows per arm"):
            analyze_contrasts(
                missing_cell,
                DEFAULT_CONTRASTS,
                bootstrap_samples=20,
            )

        unpaired_feedback = [
            row
            for row in records
            if not (
                row["mode"] == "feedback"
                and row["arm"] == "concrete_counterexample"
                and row["case_id"] == "m002"
            )
        ]
        with self.assertRaisesRegex(ValueError, "exact cohort"):
            analyze_contrasts(
                unpaired_feedback,
                DEFAULT_CONTRASTS,
                bootstrap_samples=20,
            )

    def test_clean_primary_requires_75_main_cases_and_filters_feedback_only(self):
        records = _full_primary_records(feedback_cases=(0, 1, 80))
        # Inconclusive rows outside clean75 must not poison clean sensitivity,
        # including a main cell and an unrelated shift-mode record.
        for row in records:
            if row["mode"] == "main" and row["case_id"] == "m080":
                row["success"] = None
        records.append(_row("shift-only", "shift", "spec0_loc0", None, "shift"))
        clean = {f"m{index:03d}" for index in range(75)}
        analysis = analyze_contrasts(
            records,
            DEFAULT_CONTRASTS,
            keep_case_ids=clean,
            bootstrap_samples=20,
        )
        self.assertEqual(
            analysis["contrasts"]["what_full_spec"]["case_count"], 75
        )
        self.assertEqual(
            analysis["contrasts"]["counterexample_revision"]["case_count"], 2
        )
        with self.assertRaisesRegex(ValueError, "exactly 75"):
            analyze_contrasts(
                records,
                DEFAULT_CONTRASTS,
                keep_case_ids=set(sorted(clean)[:-1]),
                bootstrap_samples=20,
            )

        no_clean_feedback = _full_primary_records(feedback_cases=(80, 81))
        with self.assertRaisesRegex(ValueError, "non-empty two-arm cohort"):
            analyze_contrasts(
                no_clean_feedback,
                DEFAULT_CONTRASTS,
                keep_case_ids=clean,
                bootstrap_samples=20,
            )

    def test_cancel_shift_cli_flag_requires_all_safety_switches(self):
        invocations = (
            [
                "vericodegen_stats.py",
                "missing.jsonl",
                "--cancel-inconclusive-shift",
                "--include-secondaries",
                "--shift-manifest",
                "shift.json",
            ],
            [
                "vericodegen_stats.py",
                "missing.jsonl",
                "--cancel-inconclusive-shift",
                "--from-ledger",
                "--shift-manifest",
                "shift.json",
            ],
            [
                "vericodegen_stats.py",
                "missing.jsonl",
                "--cancel-inconclusive-shift",
                "--from-ledger",
                "--include-secondaries",
            ],
        )
        for argv in invocations:
            with self.subTest(argv=argv), patch("sys.argv", argv):
                with self.assertRaises(SystemExit) as error:
                    stats_main()
                self.assertEqual(error.exception.code, 2)

    def test_dropped_redacted_arm_with_no_rows_produces_no_pseudo_estimate(self):
        records = []
        for case, seed, family in (("m1", "s1", "op"), ("m2", "s2", "mux")):
            for arm, success in {
                "spec0_loc0": False,
                "spec1_loc0": True,
                "spec0_loc1": False,
                "spec1_loc1": True,
            }.items():
                row = _row(case, seed, arm, success, "main")
                row["mutation"] = family
                records.append(row)
        secondary = analyze_secondaries(
            records,
            redacted_eligible_case_ids=set(),
            shift_case_ids=set(),
            bootstrap_samples=100,
        )
        self.assertIsNone(secondary["redacted_spec"])


if __name__ == "__main__":
    unittest.main()
