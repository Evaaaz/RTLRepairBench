from __future__ import annotations

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.export_vericodegen_supplement import (
    DEFAULT_PROTOCOL_AMENDMENT,
    DEFAULT_REDACTION_DECISION,
    NamedArtifact,
    SupplementExportError,
    SupplementInputs,
    export_supplement,
)


def _canonical_sha(value):
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _resign_analysis(document):
    document.pop("analysis_sha256", None)
    document["analysis_sha256"] = _canonical_sha(document)
    return document


def _analysis(clean: bool):
    definitions = (
        ("what_full_spec", "main", "spec0_loc0", "spec1_loc0"),
        ("where_oracle_location", "main", "spec0_loc0", "spec0_loc1"),
        (
            "counterexample_revision",
            "feedback",
            "generic_failure",
            "concrete_counterexample",
        ),
    )
    contrasts = {
        name: {
            "mode": mode,
            "arm_a": arm_a,
            "arm_b": arm_b,
            "bootstrap_samples": 10_000,
            "bootstrap_rng_seed": 42 + index,
            "case_count": (
                1
                if name == "counterexample_revision"
                else (75 if clean else 85)
            ),
        }
        for index, (name, mode, arm_a, arm_b) in enumerate(definitions)
    }
    shift = {
        "status": "NO_ESTIMATE",
        "decision_timing": "POST_OUTPUT_CONSERVATIVE_DECISION",
        "estimand": "full_frozen_50_task_shift_set",
        "reason": "post_output_validator_indeterminacy_precludes_complete_case_effect_estimation",
        "arm_a": "spec0_loc0",
        "arm_b": "spec1_loc0",
        "expected_cases": 50,
        "expected_calls": 100,
        "observed_calls": 100,
        "definitive_calls": 86,
        "unresolved_calls": 14,
        "definitive_coverage": 0.86,
        "fully_definitive_pair_count": 42,
        "per_arm": {
            "spec0_loc0": {
                "expected_calls": 50,
                "observed_calls": 50,
                "definitive_calls": 44,
                "unresolved_calls": 6,
                "definitive_coverage": 0.88,
                "reason_counts": {
                    "ansi_or_interface_parse_failure": 6,
                },
                "status_counts": {
                    "formal_unsupported__simulation_timeout": 6,
                },
            },
            "spec1_loc0": {
                "expected_calls": 50,
                "observed_calls": 50,
                "definitive_calls": 42,
                "unresolved_calls": 8,
                "definitive_coverage": 0.84,
                "reason_counts": {
                    "inconsistent_clock_edge": 1,
                    "no_reset_or_legal_preamble": 7,
                },
                "status_counts": {
                    "formal_unsupported__simulation_timeout": 8,
                },
            },
        },
        "reason_counts": {
            "ansi_or_interface_parse_failure": 6,
            "inconsistent_clock_edge": 1,
            "no_reset_or_legal_preamble": 7,
        },
        "status_counts": {
            "formal_unsupported__simulation_timeout": 14,
        },
        "complete_case_effect_reported": False,
        "unresolved_imputed": False,
        "secondary": True,
        "multiplicity_adjusted": False,
    }
    document = {
        "schema_version": 1,
        "kind": "vericodegen_primary_clustered_analysis",
        "case_filter_count": 75 if clean else None,
        "contrasts": contrasts,
        "holm_family": list(contrasts),
        "secondaries": {"distribution_shift": shift},
    }
    document["analysis_sha256"] = _canonical_sha(document)
    return document


class SupplementFixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.config = root / "study.json"
        self.amendment = root / "protocol_amendment.json"
        self.redaction_decision = root / "redaction_decision.json"
        self.ledger = root / "events.jsonl"
        self.calibration_gate = root / "calibration_gate.json"
        self.calibration_results = root / "calibration_results.jsonl"
        self.fallback_report = root / "fallback_simulation.json"
        self.formal_gate = root / "formal_gate.json"
        self.validation = root / "validation.jsonl"
        self.full_stats = root / "full_stats.json"
        self.clean_stats = root / "clean_stats.json"
        self.full_rows = root / "full_rows.jsonl"
        self.clean_rows = root / "clean_rows.jsonl"
        self.manifest = root / "manifest.json"
        self.identifiers = root / "materialized.jsonl"
        self.candidate_sha = "a" * 64
        self.formal_protocol_sha = "f" * 64
        credential_marker = "sk-TESTCREDENTIAL123456789"

        _write_json(
            self.config,
            {
                "study_id": "vericodegen-test",
                "provider": {
                    "api_key_env": "NVIDIA_API_KEY",
                    "api_key": credential_marker,
                },
                "author": "Alice Example",
                "git_remote": "git@github.com:alice/private.git",
                "scratch": "/Users/alice/private/run",
                "prob151_normalized": False,
            },
        )
        self.amendment.write_bytes(DEFAULT_PROTOCOL_AMENDMENT.read_bytes())
        self.redaction_decision.write_bytes(DEFAULT_REDACTION_DECISION.read_bytes())
        self.amendment_sha = hashlib.sha256(self.amendment.read_bytes()).hexdigest()
        manifest = {
            "schema_version": 1,
            "kind": "test_manifest",
            "cases": [
                {
                    "case_id": "repair_sem_0000",
                    "seed_id": "Prob013_m2014_q4e_ref",
                }
            ],
        }
        manifest["manifest_sha256"] = _canonical_sha(manifest)
        _write_json(self.manifest, manifest)
        _write_jsonl(
            self.identifiers,
            [
                {
                    "internal_metadata": {
                        "anonymous_id": "main_0000",
                        "case_id": "repair_sem_0000",
                        "cluster_id": "Prob013_m2014_q4e_ref",
                        "source_seed_id": "Prob013_m2014_q4e_ref",
                    }
                },
                {
                    "internal_metadata": {
                        "anonymous_id": "main_0059",
                        "case_id": "repair_sem_0059",
                        "cluster_id": "Prob129_ece241_2013_q8_ref",
                        "source_seed_id": "Prob129_ece241_2013_q8_ref",
                    }
                }
            ],
        )
        events = [
                {
                    "schema_version": 1,
                    "recorded_at": "2026-07-13T23:59:58Z",
                    "event": "protocol_amendment_lock",
                    "protocol_version": "vericodegen-2026-v1",
                    "protocol_amendment_path": "configs/vericodegen/protocol_amendment_2026-07-15.json",
                    "protocol_amendment_sha256": self.amendment_sha,
                    "attempt_count_before_lock": 0,
                    "response_count_before_lock": 0,
                },
                {
                    "schema_version": 1,
                    "recorded_at": "2026-07-14T00:00:00Z",
                    "event": "request_attempt",
                    "call_id": "call-one",
                    "attempt_number": 1,
                    "mode": "main",
                    "arm": "spec0_loc0",
                    "case_id": "main_0000",
                    "cluster_id": "Prob013_m2014_q4e_ref",
                    "prompt": (
                        "Repair Prob013_m2014_q4e_ref with prob151_normalized "
                        "under /Users/alice/private"
                    ),
                    "system": "Return a module.",
                    "prompt_sha256": "b" * 64,
                    "request_metadata": {"model": "model-test"},
                },
                {
                    "schema_version": 1,
                    "recorded_at": "2026-07-14T00:00:01Z",
                    "event": "response",
                    "call_id": "call-one",
                    "attempt_number": 1,
                    "mode": "main",
                    "arm": "spec0_loc0",
                    "case_id": "main_0000",
                    "cluster_id": "Prob013_m2014_q4e_ref",
                    "nonresearch": False,
                    "raw_response": {"debug": credential_marker},
                    "output_text": "module TopModule; endmodule",
                    "parsed_rtl": "module TopModule; endmodule",
                    "parsed_rtl_sha256": self.candidate_sha,
                    "served_model": "model-test",
                    "response_status": "completed",
                    "usage": {"input_tokens": 10, "output_tokens": 5},
                    "response_id": "provider-response-id",
                    "request_id": "provider-request-id",
                    "elapsed_ms": 2.0,
                    "model_outcome": "accepted_pending_compile",
                    "compile_verdict": {"status": "PENDING"},
                    "formal_verdict": {"status": "PENDING"},
                    "simulation_verdict": {"status": "PENDING"},
                },
                {
                    "schema_version": 1,
                    "recorded_at": "2026-07-14T00:00:01.5Z",
                    "event": "validation_job_lock",
                    "protocol_version": "vericodegen-2026-v1",
                    "call_id": "call-one",
                    "candidate_sha256": self.candidate_sha,
                    "golden_sha256": "c" * 64,
                    "formal_protocol_sha256": self.formal_protocol_sha,
                },
                {
                    "schema_version": 1,
                    "recorded_at": "2026-07-14T00:00:02Z",
                    "event": "verdict",
                    "call_id": "call-one",
                    "compile_verdict": {"passed": True},
                    "formal_verdict": {
                        "status": "PROVED",
                        "candidate_sha256": self.candidate_sha,
                        "vcd_path": "/work/generated/formal/private.vcd",
                    },
                    "simulation_verdict": {
                        "status": "PASS",
                        "candidate_sha256": self.candidate_sha,
                    },
                },
            ]
        validation_rows = [
                {
                    "call_id": "call-one",
                    "seed_id": "Prob013_m2014_q4e_ref",
                    "candidate_sha256": self.candidate_sha,
                    "formal_verdict": {"status": "PROVED"},
                    "simulation_verdict": {"status": "PASS"},
                }
            ]
        normalized_rows = [
            {
                "call_id": "call-one",
                "mode": "main",
                "case_id": "main_0000",
                "seed_id": "Prob013_m2014_q4e_ref",
                "arm": "spec0_loc0",
                "success": True,
            }
        ]
        for arm_index, arm in enumerate(("spec0_loc0", "spec1_loc0")):
            unresolved_limit = (6, 8)[arm_index]
            for case_index in range(50):
                call_id = f"shift-{arm_index}-{case_index:02d}"
                case_id = f"shift_{case_index:04d}"
                cluster_id = f"shift_seed_{case_index:04d}"
                candidate_sha = hashlib.sha256(call_id.encode("utf-8")).hexdigest()
                unresolved = case_index < unresolved_limit
                formal_status = "UNSUPPORTED" if unresolved else "PROVED"
                simulation_status = "TIMEOUT" if unresolved else "PASS"
                formal_reason = "formal frontend inconclusive" if unresolved else None
                simulation_reason = None
                if unresolved and arm_index == 0:
                    simulation_reason = "ansi_or_interface_parse_failure"
                elif unresolved and case_index < 7:
                    simulation_reason = "no_reset_or_legal_preamble"
                elif unresolved:
                    simulation_reason = "inconsistent_clock_edge"
                reason = simulation_reason or formal_reason
                formal_verdict = {"status": formal_status}
                simulation_verdict = {"status": simulation_status}
                if formal_reason is not None:
                    formal_verdict["reason"] = formal_reason
                if simulation_reason is not None:
                    simulation_verdict["reason"] = simulation_reason
                events.extend(
                    [
                        {
                            "schema_version": 1,
                            "recorded_at": "2026-07-14T00:01:00Z",
                            "event": "response",
                            "call_id": call_id,
                            "attempt_number": 1,
                            "mode": "shift",
                            "arm": arm,
                            "case_id": case_id,
                            "cluster_id": cluster_id,
                            "nonresearch": False,
                            "parsed_rtl_sha256": candidate_sha,
                            "served_model": "model-test",
                            "response_status": "completed",
                            "model_outcome": "accepted_pending_compile",
                        },
                        {
                            "schema_version": 1,
                            "recorded_at": "2026-07-14T00:01:01Z",
                            "event": "verdict",
                            "call_id": call_id,
                            "compile_verdict": {"passed": True},
                            "formal_verdict": formal_verdict,
                            "simulation_verdict": simulation_verdict,
                        },
                    ]
                )
                validation_rows.append(
                    {
                        "call_id": call_id,
                        "seed_id": cluster_id,
                        "candidate_sha256": candidate_sha,
                        "formal_verdict": formal_verdict,
                        "simulation_verdict": simulation_verdict,
                    }
                )
                normalized_rows.append(
                    {
                        "call_id": call_id,
                        "mode": "shift",
                        "case_id": case_id,
                        "seed_id": cluster_id,
                        "arm": arm,
                        "success": None if unresolved else True,
                        "resolved": not unresolved,
                        "evidence": (
                            "inconclusive_shift_validation"
                            if unresolved
                            else "unbounded_formal_proof"
                        ),
                        "formal_status": formal_status,
                        "simulation_status": simulation_status,
                        "reason": reason,
                        "formal_reason": formal_reason,
                        "simulation_reason": simulation_reason,
                    }
                )
        _write_jsonl(self.ledger, events)
        _write_jsonl(self.validation, validation_rows)
        self.normalized_rows = normalized_rows
        golden_sha = "1ee2398da6c8170f1d7937eac091a160ec8bb037bc389ae0d66a8d19a5a49716"
        mutant_sha = "35c79ef6db0aec962710791bc4a3465979d011f1ae883379ea5fdc1d6f2b57c4"
        fallback_runs = [
            {
                "seed": seed,
                "cycles": 200,
                "status": "COUNTEREXAMPLE",
                "passed": False,
            }
            for seed in range(1001, 1011)
        ]
        _write_json(
            self.fallback_report,
            {
                "status": "COUNTEREXAMPLE",
                "seeds": list(range(1001, 1011)),
                "cycles_per_seed": 200,
                "golden_sha256": golden_sha,
                "candidate_sha256": mutant_sha,
                "formal_protocol_sha256": self.formal_protocol_sha,
                "protocol_amendment_sha256": self.amendment_sha,
                "runs": fallback_runs,
                "counterexample_summary": {
                    "post_initialization_concrete_count": 10,
                    "first_post_initialization_concrete": {
                        "phase": "post_initialization",
                        "concrete_two_state": True,
                        "expected": "1",
                        "got": "0",
                    },
                },
            },
        )
        fallback_report_sha = hashlib.sha256(self.fallback_report.read_bytes()).hexdigest()
        calibration_rows = []
        case_ids = [f"case-{index:03d}" for index in range(84)] + ["repair_sem_0059"]
        for case_id in case_ids:
            calibration_rows.append(
                {
                    "run_id": f"{case_id}:golden:self",
                    "case_id": case_id,
                    "comparison": "golden_vs_golden",
                    "formal_protocol_sha256": self.formal_protocol_sha,
                    "protocol_amendment_sha256": self.amendment_sha,
                    "result": {"status": "PROVED"},
                }
            )
            if case_id == "repair_sem_0059":
                calibration_rows.append(
                    {
                        "run_id": f"{case_id}:mutant:fallback",
                        "case_id": case_id,
                        "seed_id": "Prob129_ece241_2013_q8_ref",
                        "mutation": "flip_reset_polarity",
                        "comparison": "golden_vs_mutant",
                        "formal_protocol_sha256": self.formal_protocol_sha,
                        "protocol_amendment_sha256": self.amendment_sha,
                        "result": {
                            "status": "UNSUPPORTED",
                            "golden_sha256": golden_sha,
                            "candidate_sha256": mutant_sha,
                        },
                        "composite_fallback": {
                            "report_path": str(self.fallback_report),
                            "report_sha256": fallback_report_sha,
                            "formal_protocol_sha256": self.formal_protocol_sha,
                            "protocol_amendment_sha256": self.amendment_sha,
                        },
                    }
                )
            else:
                calibration_rows.append(
                    {
                        "run_id": f"{case_id}:mutant:formal",
                        "case_id": case_id,
                        "comparison": "golden_vs_mutant",
                        "formal_protocol_sha256": self.formal_protocol_sha,
                        "protocol_amendment_sha256": self.amendment_sha,
                        "result": {
                            "status": "COUNTEREXAMPLE",
                            "counterexample_replayed": True,
                        },
                    }
                )
        _write_jsonl(self.calibration_results, calibration_rows)
        allowlist = [
            {
                "case_id": "repair_sem_0059",
                "seed_id": "Prob129_ece241_2013_q8_ref",
                "mutation": "flip_reset_polarity",
                "golden_sha256": golden_sha,
                "candidate_sha256": mutant_sha,
                "required_formal_status": "UNSUPPORTED",
                "seeds": list(range(1001, 1011)),
                "cycles_per_seed": 200,
                "required_seed_status": "COUNTEREXAMPLE",
                "requires_post_initialization_concrete_two_state": True,
            }
        ]
        evidence = {
            "case_id": "repair_sem_0059",
            "seed_id": "Prob129_ece241_2013_q8_ref",
            "formal_status": "UNSUPPORTED",
            "golden_sha256": golden_sha,
            "candidate_sha256": mutant_sha,
            "formal_protocol_sha256": self.formal_protocol_sha,
            "protocol_amendment_sha256": self.amendment_sha,
            "report_path": str(self.fallback_report),
            "report_sha256": fallback_report_sha,
            "simulation_status": "COUNTEREXAMPLE",
            "seeds": list(range(1001, 1011)),
            "cycles_per_seed": 200,
            "counterexample_seed_runs": 10,
            "qualified": True,
            "validation_errors": [],
        }
        calibration_gate = {
            "schema_version": 3,
            "gate": "day3_harness_calibration",
            "expected_cases": 85,
            "golden_vs_golden_completed": 85,
            "golden_vs_golden_statuses": {"PROVED": 85},
            "golden_vs_mutant_completed": 85,
            "golden_vs_mutant_statuses": {"COUNTEREXAMPLE": 84, "UNSUPPORTED": 1},
            "replayable_mutant_counterexamples": 84,
            "fallback_mutant_counterexamples": 1,
            "distinguishable_mutants": 85,
            "composite_policy": {
                "name": "exact_allowlisted_unsupported_with_frozen_independent_simulation",
                "version": 1,
                "protocol_amendment_sha256": self.amendment_sha,
                "allowlist": allowlist,
            },
            "fallback_mutant_evidence": [evidence],
            "fallback_validation_errors": [],
            "duplicate_run_ids": [],
            "unique_run_ids": True,
            "comparison_case_id_sets_match": True,
            "amendment_bound_rows": 170,
            "formal_protocol_rows_consistent": True,
            "mutants_incorrectly_proved": [],
            "halted_reason": "",
            "passed": True,
            "full_day3_gate_passed": True,
            "scope": "full_sem85",
            "scope_case_ids": None,
            "protocol_amendment_sha256": self.amendment_sha,
            "formal_protocol_sha256": self.formal_protocol_sha,
            "results_path": str(self.calibration_results),
            "calibration_results_sha256": hashlib.sha256(
                self.calibration_results.read_bytes()
            ).hexdigest(),
        }
        calibration_gate["gate_manifest_sha256"] = _canonical_sha(calibration_gate)
        _write_json(self.calibration_gate, calibration_gate)
        arms = {
            arm: {"records": 85, "definitive_formal_results": 70, "passed": False}
            for arm in ("spec0_loc0", "spec1_loc0", "spec0_loc1", "spec1_loc1")
        }
        _write_json(
            self.formal_gate,
            {
                "schema_version": 1,
                "gate": "formal_primary_coverage",
                "expected_cases_per_arm": 85,
                "required_arms": list(arms),
                "arms": arms,
                "conflicts": [],
                "malformed_record_indexes": [],
                "passed": False,
                "reporting_mode": "formal_where_supported_plus_independent_simulation_fallback",
            },
        )
        _write_json(self.full_stats, _analysis(clean=False))
        _write_json(self.clean_stats, _analysis(clean=True))
        _write_jsonl(self.full_rows, self.normalized_rows)
        _write_jsonl(self.clean_rows, self.normalized_rows)

    def inputs(self, **changes) -> SupplementInputs:
        values = {
            "config": self.config,
            "protocol_amendment": self.amendment,
            "redaction_decision": self.redaction_decision,
            "ledger": self.ledger,
            "calibration_gate": self.calibration_gate,
            "calibration_results": self.calibration_results,
            "formal_primary_gate": self.formal_gate,
            "validation_results": self.validation,
            "full_stats": self.full_stats,
            "clean_stats": self.clean_stats,
            "full_rows": self.full_rows,
            "clean_rows": self.clean_rows,
            "manifests": (self.manifest,),
            "identifier_sources": (self.identifiers,),
        }
        values.update(changes)
        return SupplementInputs(**values)


class SupplementExportTests(unittest.TestCase):
    def test_strict_export_refuses_missing_final_stats(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            missing = Path(tmp) / "missing_stats.json"
            with self.assertRaisesRegex(SupplementExportError, "missing final artifact"):
                export_supplement(
                    fixture.inputs(full_stats=missing),
                    Path(tmp) / "stage",
                )

    def test_export_keeps_evidence_but_removes_private_material(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            fixture = SupplementFixture(root)
            staging = root / "stage"
            result = export_supplement(
                fixture.inputs(),
                staging,
                redact_tokens=("Alice Example", "alice"),
            )
            self.assertEqual(result["file_count"], 16)
            combined = "\n".join(
                path.read_text(encoding="utf-8")
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            )
            self.assertNotIn("TESTCREDENTIAL", combined)
            self.assertNotIn("/Users/", combined)
            self.assertNotIn("git@github.com", combined)
            self.assertNotIn("Alice Example", combined)
            self.assertNotIn("repair_sem_0000", combined)
            self.assertNotIn("Prob013", combined)
            self.assertNotIn("prob151", combined.lower())
            self.assertIn(fixture.candidate_sha, combined)
            self.assertIn("request_metadata", combined)
            self.assertIn("formal_verdict", combined)
            self.assertIn("simulation_verdict", combined)
            inventory = json.loads((staging / "inventory.json").read_text())
            self.assertFalse(inventory["gates"]["strict_all_formal_calibration_passed"])
            self.assertTrue(inventory["gates"]["composite_calibration_passed"])
            self.assertEqual(inventory["gates"]["formal_calibration_counterexamples"], 84)
            self.assertEqual(inventory["gates"]["simulation_calibration_fallbacks"], 1)
            shift_coverage = inventory["analysis_coverage"]["distribution_shift"]
            self.assertEqual(shift_coverage["status"], "NO_ESTIMATE")
            self.assertEqual(shift_coverage["definitive_calls"], 86)
            self.assertEqual(shift_coverage["unresolved_calls"], 14)
            self.assertEqual(
                shift_coverage["decision_timing"],
                "POST_OUTPUT_CONSERVATIVE_DECISION",
            )
            self.assertFalse(shift_coverage["complete_case_effect_reported"])
            self.assertFalse(shift_coverage["unresolved_imputed"])
            self.assertTrue(
                (staging / "formal" / "calibration_fallback_simulation.json").is_file()
            )

            events = [
                json.loads(line)
                for line in (staging / "ledger" / "public_events.jsonl")
                .read_text(encoding="utf-8")
                .splitlines()
            ]
            response = next(row for row in events if row["event"] == "response")
            self.assertNotIn("raw_response", response)
            self.assertNotIn("output_text", response)
            self.assertNotIn("parsed_rtl", response)
            self.assertEqual(response["parsed_rtl_sha256"], fixture.candidate_sha)
            self.assertRegex(response["raw_response_sha256"], r"^[0-9a-f]{64}$")

    def test_additional_text_artifact_is_exported_and_sanitized(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            fixture = SupplementFixture(root)
            source = root / "analysis.py"
            source.write_text("label = 'prob151_normalized'\n", encoding="utf-8")
            staging = root / "stage"
            export_supplement(
                fixture.inputs(
                    additional_artifacts=(NamedArtifact("analysis", source),)
                ),
                staging,
            )
            public = (staging / "artifacts" / "analysis" / "analysis.py").read_text(
                encoding="utf-8"
            )
            self.assertNotIn("prob151", public.lower())
            self.assertIn("ANONYMIZED_INTERNAL_ID", public)

    def test_strict_export_rejects_fallback_report_byte_drift(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            fixture.fallback_report.write_text(
                fixture.fallback_report.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(SupplementExportError, "byte SHA"):
                export_supplement(fixture.inputs(), Path(tmp) / "stage")

    def test_strict_export_rejects_primary_name_with_wrong_arm_or_seed(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            analysis = json.loads(fixture.full_stats.read_text(encoding="utf-8"))
            analysis["contrasts"]["what_full_spec"]["arm_b"] = "redacted_spec_loc0"
            analysis["contrasts"]["what_full_spec"]["bootstrap_rng_seed"] = 99
            analysis.pop("analysis_sha256")
            analysis["analysis_sha256"] = _canonical_sha(analysis)
            _write_json(fixture.full_stats, analysis)
            with self.assertRaisesRegex(SupplementExportError, "mode/arms/bootstrap/seed"):
                export_supplement(fixture.inputs(), Path(tmp) / "stage")

    def test_strict_export_rejects_tampered_shift_coverage(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            analysis = json.loads(fixture.full_stats.read_text(encoding="utf-8"))
            analysis["secondaries"]["distribution_shift"]["observed_calls"] = 99
            _write_json(fixture.full_stats, _resign_analysis(analysis))
            with self.assertRaisesRegex(SupplementExportError, "observed_calls"):
                export_supplement(fixture.inputs(), Path(tmp) / "stage")

    def test_strict_export_rejects_unresolved_null_changed_to_false(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            rows = [
                json.loads(line)
                for line in fixture.full_rows.read_text(encoding="utf-8").splitlines()
            ]
            unresolved = next(row for row in rows if row.get("success") is None)
            unresolved["success"] = False
            unresolved["resolved"] = True
            _write_jsonl(fixture.full_rows, rows)
            with self.assertRaisesRegex(SupplementExportError, "success type drift"):
                export_supplement(fixture.inputs(), Path(tmp) / "stage")

    def test_strict_export_rejects_deleted_shift_row(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            rows = [
                json.loads(line)
                for line in fixture.clean_rows.read_text(encoding="utf-8").splitlines()
            ]
            _write_jsonl(fixture.clean_rows, rows[:-1])
            with self.assertRaisesRegex(SupplementExportError, "normalized rows"):
                export_supplement(fixture.inputs(), Path(tmp) / "stage")

    def test_strict_export_rejects_deleted_candidate_verdict(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            events = [
                json.loads(line)
                for line in fixture.ledger.read_text(encoding="utf-8").splitlines()
            ]
            events = [
                row
                for row in events
                if not (
                    row.get("event") == "verdict"
                    and row.get("call_id") == "shift-0-00"
                )
            ]
            _write_jsonl(fixture.ledger, events)
            with self.assertRaisesRegex(SupplementExportError, "without a final SHA/verdict"):
                export_supplement(fixture.inputs(), Path(tmp) / "stage")

    def test_strict_export_rejects_full_clean_shift_coverage_mismatch(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            analysis = json.loads(fixture.clean_stats.read_text(encoding="utf-8"))
            shift = analysis["secondaries"]["distribution_shift"]
            shift["per_arm"]["spec0_loc0"].update(
                {
                    "definitive_calls": 43,
                    "unresolved_calls": 7,
                    "definitive_coverage": 0.86,
                    "reason_counts": {"formal_timeout__simulation_timeout": 7},
                    "status_counts": {
                        "formal_unsupported__simulation_timeout": 7
                    },
                }
            )
            shift["per_arm"]["spec1_loc0"].update(
                {
                    "definitive_calls": 43,
                    "unresolved_calls": 7,
                    "definitive_coverage": 0.86,
                    "reason_counts": {
                        "formal_unsupported__simulation_unsupported": 7
                    },
                    "status_counts": {
                        "formal_unsupported__simulation_timeout": 7
                    },
                }
            )
            shift["reason_counts"] = {
                "formal_timeout__simulation_timeout": 7,
                "formal_unsupported__simulation_unsupported": 7,
            }
            _write_json(fixture.clean_stats, _resign_analysis(analysis))
            with self.assertRaisesRegex(SupplementExportError, "summaries differ"):
                export_supplement(fixture.inputs(), Path(tmp) / "stage")

    def test_strict_export_rejects_numeric_shift_effect_injection(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            analysis = json.loads(fixture.full_stats.read_text(encoding="utf-8"))
            analysis["secondaries"]["distribution_shift"]["risk_difference"] = 0.25
            _write_json(fixture.full_stats, _resign_analysis(analysis))
            with self.assertRaisesRegex(SupplementExportError, "non-null 'risk_difference'"):
                export_supplement(fixture.inputs(), Path(tmp) / "stage")

    def test_strict_export_rejects_partial_main_case_count(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            analysis = json.loads(fixture.full_stats.read_text(encoding="utf-8"))
            analysis["contrasts"]["what_full_spec"]["case_count"] = 84
            _write_json(fixture.full_stats, _resign_analysis(analysis))
            with self.assertRaisesRegex(SupplementExportError, "case_count"):
                export_supplement(fixture.inputs(), Path(tmp) / "stage")

    def test_strict_export_rejects_preregistered_shift_downgrade_claim(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            analysis = json.loads(fixture.full_stats.read_text(encoding="utf-8"))
            analysis["secondaries"]["distribution_shift"]["decision_timing"] = (
                "PRE_REGISTERED"
            )
            _write_json(fixture.full_stats, _resign_analysis(analysis))
            with self.assertRaisesRegex(SupplementExportError, "decision_timing"):
                export_supplement(fixture.inputs(), Path(tmp) / "stage")

    def test_strict_export_rejects_any_cancelled_redaction_ledger_event(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            with fixture.ledger.open("a", encoding="utf-8") as handle:
                handle.write(
                    json.dumps(
                        {
                            "event": "schedule_lock",
                            "mode": "redacted",
                            "protocol_amendment_sha256": fixture.amendment_sha,
                        }
                    )
                    + "\n"
                )
            with self.assertRaisesRegex(SupplementExportError, "cancelled redaction"):
                export_supplement(fixture.inputs(), Path(tmp) / "stage")

    def test_strict_export_rejects_legacy_all_formal_gate_shape(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            fixture = SupplementFixture(Path(tmp))
            gate = json.loads(fixture.calibration_gate.read_text(encoding="utf-8"))
            gate["schema_version"] = 2
            gate.pop("gate_manifest_sha256")
            gate["gate_manifest_sha256"] = _canonical_sha(gate)
            _write_json(fixture.calibration_gate, gate)
            with self.assertRaisesRegex(SupplementExportError, "composite Day-3"):
                export_supplement(fixture.inputs(), Path(tmp) / "stage")

    def test_archive_is_byte_deterministic(self):
        with tempfile.TemporaryDirectory(dir=Path.cwd()) as tmp:
            root = Path(tmp)
            fixture = SupplementFixture(root)
            first = root / "first.tar.gz"
            second = root / "second.tar.gz"
            export_supplement(
                fixture.inputs(),
                root / "stage-one",
                archive=first,
                redact_tokens=("Alice Example", "alice"),
            )
            export_supplement(
                fixture.inputs(),
                root / "stage-two",
                archive=second,
                redact_tokens=("Alice Example", "alice"),
            )
            self.assertEqual(first.read_bytes(), second.read_bytes())


if __name__ == "__main__":
    unittest.main()
