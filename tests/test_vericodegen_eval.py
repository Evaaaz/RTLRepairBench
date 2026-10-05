from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from backend.nvidia_responses import NVIDIAResponse, NVIDIATransportError
from benchmarks.vericodegen_eval import (
    BudgetExhausted,
    DEFAULT_MAIN_CASES,
    DEFAULT_PROTOCOL_AMENDMENT,
    DEFAULT_BENCHMARK_SOURCE,
    DEFAULT_CANONICAL_MANIFEST,
    DEFAULT_PROTOCOL_MANIFEST,
    DEFAULT_TOOLCHAIN_LOCK,
    DEFAULT_SHIFT_CASES,
    DEFAULT_SHIFT_MANIFEST,
    ExperimentCase,
    ExperimentLedger,
    ProtocolError,
    _validate_provider_config,
    _current_formal_protocol_sha256,
    _gate_manifest_sha,
    _sha256_path,
    build_schedule,
    classify_http_200,
    export_validation_jobs,
    feedback_cohort,
    load_cases,
    load_shift_selection,
    main,
    run_jobs,
    validate_calibration_gate,
    validate_frozen_redaction_decision,
    validate_protocol_amendment,
)


def _case(index: int = 0) -> ExperimentCase:
    return ExperimentCase(
        case_id=f"case-{index}",
        cluster_id=f"cluster-{index}",
        broken_rtl=(
            "module TopModule(input logic a, output logic y);\n"
            "  assign y = a;\n"
            "endmodule"
        ),
        full_spec="Output y is the inverse of a.",
        redacted_spec="Output y is a Boolean function of a.",
        redaction_label="explicit",
        location_line_number=2,
        location_line="assign y = a;",
        mutation_family="op_swap",
    )


def _response(
    text: str = "module TopModule(input logic a, output logic y); assign y = ~a; endmodule",
    *,
    status: str = "completed",
    refusal: str | None = None,
) -> NVIDIAResponse:
    return NVIDIAResponse(
        raw_response={"status": status, "output_text": text},
        output_text=text,
        response_id="resp-test",
        model="azure/openai/gpt-5.5",
        status=status,
        usage={"input_tokens": 10, "output_tokens": 10},
        refusal=refusal,
        incomplete_reason=(status if status != "completed" else None),
        protocol_error=None,
        request_id="req-test",
        elapsed_ms=1.0,
    )


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = []

    def request_metadata(self):
        return {
            "provider": "nvidia",
            "wire_api": "responses",
            "model": "azure/openai/gpt-5.5",
            "fallback": False,
        }

    def create(self, *, prompt, system, idempotency_key):
        self.calls.append(idempotency_key)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class ScheduleTests(unittest.TestCase):
    def test_dropped_redacted_arm_schedules_exactly_zero(self):
        self.assertEqual(
            build_schedule("redacted", cases=[], redacted_arm_dropped=True), []
        )
        with self.assertRaisesRegex(ProtocolError, "cannot contain"):
            build_schedule(
                "redacted", cases=[_case()], redacted_arm_dropped=True
            )
    def test_preflight_is_13_and_reuses_standalone_ping_identity(self):
        ping = build_schedule("ping")
        preflight = build_schedule("preflight")
        self.assertEqual(len(ping), 1)
        self.assertEqual(len(preflight), 13)
        self.assertEqual(ping[0].call_id, preflight[0].call_id)
        self.assertEqual(
            {job.arm for job in preflight},
            {"ping", "spec0_loc0", "spec1_loc0", "spec0_loc1", "spec1_loc1"},
        )

    def test_main_and_shift_counts_and_hash_order_are_stable(self):
        cases = [_case(index) for index in range(3)]
        forward = build_schedule("main", cases=cases, strict_counts=False)
        reverse = build_schedule("main", cases=list(reversed(cases)), strict_counts=False)
        self.assertEqual([job.call_id for job in forward], [job.call_id for job in reverse])
        self.assertEqual(len(forward), 12)
        shift = build_schedule("shift", cases=cases, strict_counts=False)
        self.assertEqual(len(shift), 6)

    def test_internal_identifiers_and_mutation_never_enter_prompts(self):
        case = replace(
            _case(),
            case_id="repair_sem_0000",
            cluster_id="Prob013_secret_ref",
            mutation_family="op_swap",
        )
        jobs = build_schedule("main", cases=[case], strict_counts=False)
        combined = "\n".join(job.prompt for job in jobs)
        self.assertNotIn("repair_sem", combined)
        self.assertNotIn("Prob013", combined)
        self.assertNotIn("op_swap", combined)
        self.assertTrue(all(job.mutation_family == "op_swap" for job in jobs))

    def test_committed_materialized_inputs_and_source_hashes_load(self):
        main = load_cases(DEFAULT_MAIN_CASES)
        selected = load_shift_selection(DEFAULT_SHIFT_MANIFEST, DEFAULT_SHIFT_CASES)
        shift = load_cases(DEFAULT_SHIFT_CASES, selected_source_ids=selected)
        self.assertEqual((len(main), len({case.cluster_id for case in main})), (85, 52))
        self.assertEqual((len(shift), len({case.cluster_id for case in shift})), (50, 50))
        self.assertTrue(all(case.case_id.startswith("main_") for case in main))
        self.assertTrue(all("Prob" not in case.case_id for case in main))

    def test_http_200_failure_classes_are_terminal_model_outcomes(self):
        self.assertEqual(classify_http_200(_response("plain prose")), ("unparseable", ""))
        self.assertEqual(
            classify_http_200(_response("", refusal="no")), ("refusal", "")
        )
        self.assertEqual(
            classify_http_200(_response("module TopModule;", status="incomplete")),
            ("truncated_or_incomplete", ""),
        )


class LedgerAndBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.ledger = ExperimentLedger(Path(self.temp.name) / "events.jsonl")
        self.job = build_schedule("main", cases=[_case()], strict_counts=False)[0]

    def tearDown(self):
        self.temp.cleanup()

    def test_protocol_amendment_lock_precedes_attempts_and_is_immutable(self):
        amendment_sha = _sha256_path(DEFAULT_PROTOCOL_AMENDMENT)
        self.ledger.ensure_protocol_amendment_lock(amendment_sha)
        self.ledger.ensure_protocol_amendment_lock(amendment_sha)
        locks = [
            event
            for event in self.ledger.events()
            if event.get("event") == "protocol_amendment_lock"
        ]
        self.assertEqual(len(locks), 1)
        with self.assertRaisesRegex(ProtocolError, "conflicting"):
            self.ledger.ensure_protocol_amendment_lock("0" * 64)

        late = ExperimentLedger(Path(self.temp.name) / "late.jsonl")
        late.append({"event": "request_attempt", "call_id": "already-sent"})
        with self.assertRaisesRegex(ProtocolError, "after an API attempt"):
            late.ensure_protocol_amendment_lock(amendment_sha)

    def _lock_validation_row(self, row):
        self.ledger.ensure_validation_job_locks(
            [
                {
                    "call_id": row["call_id"],
                    "candidate_sha256": row["candidate_sha256"],
                    "golden_sha256": row["golden_sha256"],
                    "formal_protocol_sha256": row["formal_protocol_sha256"],
                }
            ]
        )

    def test_transport_retry_is_not_denominator_and_resume_is_idempotent(self):
        client = FakeClient([NVIDIATransportError("test"), _response()])
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "ledger-secret"}, clear=True):
            summary = run_jobs(
                [self.job], ledger=self.ledger, client=client, sleep=lambda _: None
            )
        self.assertEqual(summary.transport_errors, 1)
        self.assertEqual(self.ledger.attempt_count(), 2)
        self.assertEqual(self.ledger.http_200_count(), 1)
        self.assertEqual(self.ledger.research_denominator_count(), 1)
        self.assertNotIn("ledger-secret", self.ledger.path.read_text())

        never = FakeClient([])
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "another"}, clear=True):
            resumed = run_jobs([self.job], ledger=self.ledger, client=never)
        self.assertEqual(resumed.skipped_completed, 1)
        self.assertEqual(never.calls, [])

    def test_schedule_lock_rejects_prompt_or_config_drift(self):
        client = FakeClient([_response()])
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            run_jobs(
                [self.job],
                ledger=self.ledger,
                client=client,
                config_fingerprint="config-a",
            )
            with self.assertRaisesRegex(ProtocolError, "schedule/config drift"):
                run_jobs(
                    [replace(self.job, prompt=self.job.prompt + " drift")],
                    ledger=self.ledger,
                    client=FakeClient([]),
                    config_fingerprint="config-a",
                )
            with self.assertRaisesRegex(ProtocolError, "schedule/config drift"):
                run_jobs(
                    [self.job],
                    ledger=self.ledger,
                    client=FakeClient([]),
                    config_fingerprint="config-b",
                )

    def test_attempt_cap_is_a_hard_stop(self):
        client = FakeClient(
            [NVIDIATransportError("first"), NVIDIATransportError("second")]
        )
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            with self.assertRaises(BudgetExhausted):
                run_jobs(
                    [self.job],
                    ledger=self.ledger,
                    client=client,
                    max_retries=2,
                    attempt_cap=1,
                    sleep=lambda _: None,
                )
        self.assertEqual(self.ledger.attempt_count(), 1)
        self.assertEqual(self.ledger.http_200_count(), 0)

    def test_retry_cap_is_independent_of_unused_success_budget(self):
        client = FakeClient(
            [
                NVIDIATransportError("initial"),
                NVIDIATransportError("retry-one"),
                _response(),
            ]
        )
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            with self.assertRaisesRegex(BudgetExhausted, "transport-retry"):
                run_jobs(
                    [self.job],
                    ledger=self.ledger,
                    client=client,
                    max_retries=3,
                    retry_attempt_cap=1,
                    sleep=lambda _: None,
                )
        self.assertEqual(self.ledger.attempt_count(), 2)
        self.assertEqual(self.ledger.retry_attempt_count(), 1)
        self.assertEqual(self.ledger.http_200_count(), 0)

    def test_uncommitted_partial_tail_is_recovered_for_resume(self):
        self.ledger.append({"event": "marker", "value": 1})
        with self.ledger.path.open("ab") as handle:
            handle.write(b'{"event":"request_attempt"')
        events = self.ledger.events()
        self.assertEqual(len(events), 1)
        self.assertTrue(self.ledger.path.read_bytes().endswith(b"\n"))
        self.assertNotIn(b"request_attempt", self.ledger.path.read_bytes())

    def test_success_cap_is_checked_before_any_attempt(self):
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            with self.assertRaises(BudgetExhausted):
                run_jobs(
                    [self.job],
                    ledger=self.ledger,
                    client=FakeClient([]),
                    success_cap=0,
                )
        self.assertEqual(self.ledger.attempt_count(), 0)

    def test_validation_contract_builds_only_eligible_feedback_cohort(self):
        # Locate the baseline arm irrespective of hash-interleaved order.
        job = next(
            job
            for job in build_schedule("main", cases=[_case()], strict_counts=False)
            if job.arm == "spec0_loc0"
        )
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            run_jobs([job], ledger=self.ledger, client=FakeClient([_response()]))
        self.assertEqual(self.ledger.pending_validation_call_ids(), [job.call_id])
        self.ledger.append_verdict(
            job.call_id,
            compile_verdict={"passed": True, "tool": "iverilog"},
            formal_verdict={
                "status": "COUNTEREXAMPLE",
                "counterexample_replayed": True,
                "witness": [{"cycle": 0, "inputs": {"a": 0}, "expected": 1, "got": 0}],
            },
            simulation_verdict={
                "status": "COUNTEREXAMPLE",
                "seeds": list(range(1001, 1011)),
                "cycles_per_seed": 200,
            },
        )
        cohort = feedback_cohort([_case()], self.ledger)
        self.assertEqual(len(cohort), 1)
        self.assertEqual(cohort[0][1], job.call_id)
        feedback = build_schedule(
            "feedback", cases=[_case()], ledger=self.ledger, strict_counts=False
        )
        self.assertEqual(len(feedback), 2)
        generic = next(item for item in feedback if item.arm == "generic_failure")
        concrete = next(
            item for item in feedback if item.arm == "concrete_counterexample"
        )
        self.assertNotIn("expected", generic.prompt)
        self.assertIn("expected", concrete.prompt)

    def test_verdict_ingest_is_idempotent_and_conflicts_fail(self):
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            run_jobs([self.job], ledger=self.ledger, client=FakeClient([_response()]))
        row = {
            "call_id": self.job.call_id,
            "candidate_sha256": self.ledger.latest_responses()[self.job.call_id][
                "parsed_rtl_sha256"
            ],
            "golden_sha256": "a" * 64,
            "formal_protocol_sha256": "b" * 64,
            "compile_verdict": {"passed": True},
            "formal_verdict": {"status": "PROVED"},
            "simulation_verdict": {
                "status": "PASS",
                "seeds": list(range(1001, 1011)),
                "cycles_per_seed": 200,
            },
        }
        self._lock_validation_row(row)
        path = Path(self.temp.name) / "verdicts.jsonl"
        path.write_text(json.dumps(row) + "\n")
        self.assertEqual(self.ledger.ingest_validation_jsonl(path), 1)
        self.assertEqual(self.ledger.ingest_validation_jsonl(path), 0)
        row["formal_verdict"] = {"status": "TIMEOUT"}
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ProtocolError, "conflicting"):
            self.ledger.ingest_validation_jsonl(path)

    def test_verdict_ingest_rejects_candidate_sha_mismatch(self):
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            run_jobs([self.job], ledger=self.ledger, client=FakeClient([_response()]))
        row = {
            "call_id": self.job.call_id,
            "candidate_sha256": "0" * 64,
            "golden_sha256": "a" * 64,
            "formal_protocol_sha256": "b" * 64,
            "compile_verdict": {"passed": True},
            "formal_verdict": {"status": "PROVED"},
            "simulation_verdict": {
                "status": "PASS",
                "seeds": list(range(1001, 1011)),
                "cycles_per_seed": 200,
            },
        }
        self._lock_validation_row(
            {
                **row,
                "candidate_sha256": self.ledger.latest_responses()[self.job.call_id][
                    "parsed_rtl_sha256"
                ],
            }
        )
        path = Path(self.temp.name) / "wrong_candidate.jsonl"
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ProtocolError, "candidate SHA mismatch"):
            self.ledger.ingest_validation_jsonl(path)

    def test_verdict_ingest_rejects_golden_or_protocol_drift_from_export(self):
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            run_jobs([self.job], ledger=self.ledger, client=FakeClient([_response()]))
        row = {
            "call_id": self.job.call_id,
            "candidate_sha256": self.ledger.latest_responses()[self.job.call_id][
                "parsed_rtl_sha256"
            ],
            "golden_sha256": "a" * 64,
            "formal_protocol_sha256": "b" * 64,
            "compile_verdict": {"passed": True},
            "formal_verdict": {"status": "PROVED"},
            "simulation_verdict": {
                "status": "PASS",
                "seeds": list(range(1001, 1011)),
                "cycles_per_seed": 200,
            },
        }
        self._lock_validation_row(row)
        row["golden_sha256"] = "c" * 64
        path = Path(self.temp.name) / "wrong_golden.jsonl"
        path.write_text(json.dumps(row) + "\n")
        with self.assertRaisesRegex(ProtocolError, "frozen export provenance"):
            self.ledger.ingest_validation_jsonl(path)

    def test_pending_candidate_exports_to_formal_batch_contract(self):
        main_cases = load_cases(DEFAULT_MAIN_CASES)
        selected = load_shift_selection(DEFAULT_SHIFT_MANIFEST, DEFAULT_SHIFT_CASES)
        shift_cases = load_cases(DEFAULT_SHIFT_CASES, selected_source_ids=selected)
        job = build_schedule(
            "main", cases=[main_cases[0]], strict_counts=False
        )[0]
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            run_jobs([job], ledger=self.ledger, client=FakeClient([_response()]))
        output = Path(self.temp.name) / "validation_jobs.jsonl"
        count = export_validation_jobs(
            main_cases=main_cases,
            shift_cases=shift_cases,
            ledger=self.ledger,
            output_path=output,
            formal_protocol_sha256="b" * 64,
        )
        self.assertEqual(count, 1)
        row = json.loads(output.read_text())
        self.assertEqual(row["call_id"], job.call_id)
        self.assertTrue(row["seed_id"].endswith("_ref"))
        self.assertIn("candidate_rtl", row)
        self.assertEqual(row["candidate_sha256"], self.ledger.latest_responses()[job.call_id]["parsed_rtl_sha256"])
        self.assertEqual(row["formal_protocol_sha256"], "b" * 64)
        self.assertRegex(row["golden_sha256"], r"^[0-9a-f]{64}$")
        self.assertIn("golden_path", row)
        self.assertNotIn("golden_rtl", row)
        self.assertEqual(
            self.ledger.validation_job_locks()[job.call_id]["golden_sha256"],
            row["golden_sha256"],
        )

    def test_shift_validation_rejects_source_record_sha_drift(self):
        main_cases = load_cases(DEFAULT_MAIN_CASES)
        selected = load_shift_selection(DEFAULT_SHIFT_MANIFEST, DEFAULT_SHIFT_CASES)
        shift_cases = load_cases(DEFAULT_SHIFT_CASES, selected_source_ids=selected)
        case = shift_cases[0]
        job = build_schedule("shift", cases=[case], strict_counts=False)[0]
        with mock.patch.dict(os.environ, {"NVIDIA_API_KEY": "x"}, clear=True):
            run_jobs([job], ledger=self.ledger, client=FakeClient([_response()]))

        source_path = Path(__file__).resolve().parents[1] / "data" / "repairbench_realbugs.jsonl"
        source_rows = [
            json.loads(line) for line in source_path.read_text().splitlines() if line.strip()
        ]
        selected_row = next(row for row in source_rows if row["id"] == case.source_case_id)
        selected_row["golden_rtl"] += "\n// drift"
        drifted_path = Path(self.temp.name) / "drifted_shift.jsonl"
        drifted_path.write_text(json.dumps(selected_row) + "\n")
        with self.assertRaisesRegex(ProtocolError, "source SHA drift"):
            export_validation_jobs(
                main_cases=main_cases,
                shift_cases=shift_cases,
                ledger=self.ledger,
                output_path=Path(self.temp.name) / "shift_jobs.jsonl",
                formal_protocol_sha256="b" * 64,
                shift_source_path=drifted_path,
            )


class GateAndConfigTests(unittest.TestCase):
    @staticmethod
    def _passing_gate(directory):
        base = Path(directory)
        lock = json.loads(DEFAULT_TOOLCHAIN_LOCK.read_text())
        required_tools = lock["oss_cad_suite"]["required_tools"]
        toolchain = {
            "schema_version": 1,
            "container_image": "rtlrepair-formal:test",
            "container_image_id": "sha256:" + "2" * 64,
            "toolchain_lock_path": "docker/formal/toolchain.lock.json",
            "toolchain_lock_sha256": _sha256_path(DEFAULT_TOOLCHAIN_LOCK),
            "tools": {
                tool: {"available": True, "path": f"/opt/bin/{tool}", "version": "test"}
                for tool in required_tools
            },
        }
        toolchain_path = base / "toolchain_manifest.json"
        toolchain_path.write_text(json.dumps(toolchain, sort_keys=True))
        protocol_sha = _current_formal_protocol_sha256(toolchain["container_image_id"])
        amendment_sha = _sha256_path(DEFAULT_PROTOCOL_AMENDMENT)
        amendment = json.loads(DEFAULT_PROTOCOL_AMENDMENT.read_text())
        canonical = json.loads(DEFAULT_CANONICAL_MANIFEST.read_text())
        results = []
        for case in canonical["cases"]:
            for comparison in ("golden_vs_golden", "golden_vs_mutant"):
                is_fallback = (
                    case["case_id"] == "repair_sem_0059"
                    and comparison == "golden_vs_mutant"
                )
                result = {
                    "status": (
                        "PROVED"
                        if comparison == "golden_vs_golden"
                        else ("UNSUPPORTED" if is_fallback else "COUNTEREXAMPLE")
                    ),
                    "counterexample_replayed": (
                        True
                        if comparison == "golden_vs_mutant" and not is_fallback
                        else None
                    ),
                }
                if is_fallback:
                    result.update(
                        {
                            "golden_sha256": "1ee2398da6c8170f1d7937eac091a160ec8bb037bc389ae0d66a8d19a5a49716",
                            "candidate_sha256": "35c79ef6db0aec962710791bc4a3465979d011f1ae883379ea5fdc1d6f2b57c4",
                        }
                    )
                row = {
                    "run_id": f"{case['case_id']}:{comparison}:{protocol_sha}",
                    "case_id": case["case_id"],
                    "seed_id": (
                        "Prob129_ece241_2013_q8_ref" if is_fallback else case["seed_id"]
                    ),
                    "mutation": "flip_reset_polarity" if is_fallback else case["mutation"],
                    "comparison": comparison,
                    "formal_protocol_sha256": protocol_sha,
                    "protocol_amendment_sha256": amendment_sha,
                    "result": result,
                }
                results.append(row)
        fallback_report = {
            "status": "COUNTEREXAMPLE",
            "seeds": list(range(1001, 1011)),
            "cycles_per_seed": 200,
            "golden_sha256": "1ee2398da6c8170f1d7937eac091a160ec8bb037bc389ae0d66a8d19a5a49716",
            "candidate_sha256": "35c79ef6db0aec962710791bc4a3465979d011f1ae883379ea5fdc1d6f2b57c4",
            "formal_protocol_sha256": protocol_sha,
            "protocol_amendment_sha256": amendment_sha,
            "runs": [
                {"seed": seed, "cycles": 200, "status": "COUNTEREXAMPLE", "passed": False}
                for seed in range(1001, 1011)
            ],
            "counterexample_summary": {
                "post_initialization_concrete_count": 1,
                "first_post_initialization_concrete": {
                    "phase": "post_initialization",
                    "concrete_two_state": True,
                    "cycle": 4,
                    "expected": "1",
                    "got": "0",
                    "output": "z",
                    "seed": 1001,
                },
            },
        }
        report_path = base / "repair_sem_0059" / "independent_simulation.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(fallback_report, sort_keys=True))
        report_sha = _sha256_path(report_path)
        fallback_row = next(
            row
            for row in results
            if row["case_id"] == "repair_sem_0059"
            and row["comparison"] == "golden_vs_mutant"
        )
        fallback_row["composite_fallback"] = {
            "schema_version": 1,
            "policy_name": "exact_allowlisted_unsupported_with_frozen_independent_simulation",
            "policy_version": 1,
            "case_id": "repair_sem_0059",
            "seed_id": "Prob129_ece241_2013_q8_ref",
            "golden_sha256": fallback_row["result"]["golden_sha256"],
            "candidate_sha256": fallback_row["result"]["candidate_sha256"],
            "formal_protocol_sha256": protocol_sha,
            "protocol_amendment_sha256": amendment_sha,
            "report_path": str(report_path),
            "report_sha256": report_sha,
        }
        results_path = base / "calibration_results.jsonl"
        results_path.write_text(
            "".join(json.dumps(row, sort_keys=True) + "\n" for row in results)
        )
        gate = {
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
            "fallback_mutant_evidence": [
                {
                    "case_id": "repair_sem_0059",
                    "seed_id": "Prob129_ece241_2013_q8_ref",
                    "mutation": "flip_reset_polarity",
                    "formal_status": "UNSUPPORTED",
                    "golden_sha256": fallback_row["result"]["golden_sha256"],
                    "candidate_sha256": fallback_row["result"]["candidate_sha256"],
                    "formal_protocol_sha256": protocol_sha,
                    "protocol_amendment_sha256": amendment_sha,
                    "report_path": str(report_path),
                    "report_sha256": report_sha,
                    "simulation_status": "COUNTEREXAMPLE",
                    "seeds": list(range(1001, 1011)),
                    "cycles_per_seed": 200,
                    "counterexample_seed_runs": 10,
                    "post_initialization_concrete_count": 1,
                    "first_post_initialization_concrete": fallback_report["counterexample_summary"]["first_post_initialization_concrete"],
                    "qualified": True,
                    "validation_errors": [],
                }
            ],
            "fallback_validation_errors": [],
            "duplicate_run_ids": [],
            "unique_run_ids": True,
            "comparison_case_id_sets_match": True,
            "amendment_bound_rows": 170,
            "formal_protocol_rows_consistent": True,
            "composite_policy": {
                "name": "exact_allowlisted_unsupported_with_frozen_independent_simulation",
                "version": 1,
                "amendment_path": "configs/vericodegen/protocol_amendment_2026-07-15.json",
                "protocol_amendment_sha256": amendment_sha,
                "amendment_manifest_sha256": amendment["manifest_sha256"],
                "allowlist": [
                    {
                        "case_id": "repair_sem_0059",
                        "seed_id": "Prob129_ece241_2013_q8_ref",
                        "mutation": "flip_reset_polarity",
                        "golden_sha256": fallback_row["result"]["golden_sha256"],
                        "candidate_sha256": fallback_row["result"]["candidate_sha256"],
                        "required_formal_status": "UNSUPPORTED",
                        "seeds": list(range(1001, 1011)),
                        "cycles_per_seed": 200,
                        "required_seed_status": "COUNTEREXAMPLE",
                        "requires_post_initialization_concrete_two_state": True,
                    }
                ],
            },
            "mutants_incorrectly_proved": [],
            "halted_reason": "",
            "passed": True,
            "benchmark_source": str(DEFAULT_BENCHMARK_SOURCE),
            "benchmark_source_sha256": _sha256_path(DEFAULT_BENCHMARK_SOURCE),
            "canonical_manifest": str(DEFAULT_CANONICAL_MANIFEST),
            "canonical_manifest_sha256": _sha256_path(DEFAULT_CANONICAL_MANIFEST),
            "protocol_manifest": str(DEFAULT_PROTOCOL_MANIFEST),
            "protocol_manifest_sha256": _sha256_path(DEFAULT_PROTOCOL_MANIFEST),
            "results_path": str(results_path),
            "calibration_results_sha256": _sha256_path(results_path),
            "toolchain_manifest": str(toolchain_path),
            "toolchain_manifest_sha256": _sha256_path(toolchain_path),
            "formal_protocol_sha256": protocol_sha,
            "amendment_path": str(DEFAULT_PROTOCOL_AMENDMENT),
            "protocol_amendment_sha256": amendment_sha,
            "amendment_manifest_sha256": amendment["manifest_sha256"],
            "scope": "full_sem85",
            "scope_case_ids": None,
            "full_day3_gate_passed": True,
        }
        gate["gate_manifest_sha256"] = _gate_manifest_sha(gate)
        return gate

    def test_calibration_gate_is_exact_and_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            path.write_text(json.dumps(self._passing_gate(directory)))
            validate_calibration_gate(path)
            failed = self._passing_gate(directory)
            failed["replayable_mutant_counterexamples"] = 83
            failed["gate_manifest_sha256"] = _gate_manifest_sha(failed)
            path.write_text(json.dumps(failed))
            with self.assertRaisesRegex(ProtocolError, "replayable"):
                validate_calibration_gate(path)

    def test_calibration_gate_rejects_subset_and_results_byte_drift(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            gate = self._passing_gate(directory)
            gate["scope"] = "subset_smoke"
            gate["scope_case_ids"] = ["repair_sem_0000"]
            gate["full_day3_gate_passed"] = False
            gate["gate_manifest_sha256"] = _gate_manifest_sha(gate)
            path.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ProtocolError, "scope"):
                validate_calibration_gate(path)

            gate = self._passing_gate(directory)
            path.write_text(json.dumps(gate))
            Path(gate["results_path"]).write_text(
                Path(gate["results_path"]).read_text() + "\n"
            )
            with self.assertRaisesRegex(ProtocolError, "artifact bytes"):
                validate_calibration_gate(path)

    def test_calibration_gate_rejects_stale_protocol_or_image_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gate.json"
            gate = self._passing_gate(directory)
            toolchain = json.loads(Path(gate["toolchain_manifest"]).read_text())
            toolchain["container_image_id"] = "sha256:" + "1" * 64
            changed_toolchain = Path(directory) / "changed_toolchain.json"
            changed_toolchain.write_text(json.dumps(toolchain))
            gate["toolchain_manifest"] = str(changed_toolchain)
            gate["toolchain_manifest_sha256"] = _sha256_path(changed_toolchain)
            gate["gate_manifest_sha256"] = _gate_manifest_sha(gate)
            path.write_text(json.dumps(gate))
            with self.assertRaisesRegex(ProtocolError, "fingerprint is stale"):
                validate_calibration_gate(path)

    def test_safe_api_key_env_config_is_allowed_but_secret_field_is_not(self):
        _validate_provider_config(
            {
                "provider": {
                    "api_key_env": "NVIDIA_API_KEY",
                    "wire_api": "responses",
                    "fallback": False,
                }
            }
        )
        with self.assertRaisesRegex(ProtocolError, "fallback"):
            _validate_provider_config({"provider": {"fallback": None}})
        with self.assertRaisesRegex(ProtocolError, "credentials"):
            _validate_provider_config({"provider": {"api_key": "secret"}})

    def test_checked_in_protocol_amendment_and_redaction_drop_are_valid(self):
        config = json.loads((Path(__file__).parents[1] / "configs/vericodegen2026.json").read_text())
        amendment = validate_protocol_amendment(config=config)
        self.assertEqual(amendment["effective_call_budget"]["maximum_successful_calls"], 623)
        decision = validate_frozen_redaction_decision(
            Path(__file__).parents[1] / "configs/vericodegen/redaction_manifest.json"
        )
        self.assertEqual(decision["decision_basis"], "NO_INDEPENDENT_HUMAN_ANNOTATORS")
        self.assertEqual(decision["eligible_case_count"], 0)

    def test_paid_redacted_mode_is_explicitly_forbidden_after_drop(self):
        with self.assertRaisesRegex(ProtocolError, "paid redacted mode is forbidden"):
            main(["--mode", "redacted", "--run"])

    def test_redaction_blindness_gate_enforces_frozen_twenty_percent_decision(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "redaction.json"
            document = {
                "schema_version": 1,
                "kind": "vericodegen_frozen_spec_redactions",
                "status": "FROZEN",
                "case_count": 85,
                "unredactable_fraction": 0.2,
                "unredactable_count": 17,
                "maximum_allowed_unredactable_fraction": 0.2,
                "arm_decision": "RUN_REDACTED_ARM",
                "eligible_case_count": 68,
                "eligible_case_ids": [f"case-{index}" for index in range(68)],
                "cases": [
                    {
                        "case_id": f"case-{index}",
                        "redactable_without_hint": index < 68,
                    }
                    for index in range(85)
                ],
            }
            from benchmarks.vericodegen_eval import _manifest_sha

            document["manifest_sha256"] = _manifest_sha(document)
            path.write_text(json.dumps(document))
            validate_frozen_redaction_decision(path)
            document["arm_decision"] = "DROP_REDACTED_ARM"
            document["manifest_sha256"] = _manifest_sha(document)
            path.write_text(json.dumps(document))
            with self.assertRaisesRegex(ProtocolError, "20% rule"):
                validate_frozen_redaction_decision(path)

    def test_paid_cli_rejects_config_or_ledger_budget_splitting(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "config.json"
            config_path.write_text(
                json.dumps(
                    {
                        "study_id": "vericodegen-2026-verifier-guided-rtl-repair",
                        "provider": {"fallback": False},
                    }
                )
            )
            with self.assertRaisesRegex(ProtocolError, "exact checked-in"):
                main(["--mode", "preflight", "--run", "--config", str(config_path)])
            with self.assertRaisesRegex(ProtocolError, "single frozen ledger"):
                main(
                    [
                        "--mode",
                        "preflight",
                        "--run",
                        "--ledger",
                        str(Path(directory) / "second-ledger.jsonl"),
                    ]
                )


if __name__ == "__main__":
    unittest.main()
