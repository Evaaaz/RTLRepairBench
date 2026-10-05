from __future__ import annotations

import tempfile
import unittest
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import call, patch

from benchmarks.formal_data import load_semantic_cases
from benchmarks.formal_protocol import (
    FormalResult,
    FormalStatus,
    InitializationContract,
    sha256_text,
)
from benchmarks.formal_verify import (
    CommandResult,
    DEFAULT_PROTOCOL_AMENDMENT,
    FormalVerifier,
    _candidate_security_reason,
    _generate_independent_tb,
    _generate_replay_tb,
    _formal_gate,
    _canonical_json_sha256,
    _normalize_sby_status,
    _parse_independent_verdict,
    _parse_vcd_states,
    _parse_smtbmc_internal_init,
    _prepare_simulation_pair,
    _sha256_file,
    _summarize_sim_counterexamples,
    _prepare_pair,
    _merge_event_ledger,
    calibration_gate,
    formal_primary_gate,
    formal_protocol_fingerprint,
    run_independent_simulation,
    run_calibration,
)


COMB = """module Tiny(input a, output y);
assign y = a;
endmodule
"""

SEQ = """module Seq(input clk, input reset, input d, output reg q);
always @(posedge clk) if (reset) q <= 0; else q <= d;
endmodule
"""


class VerifierFailClosedTest(unittest.TestCase):
    def test_candidate_cannot_drive_verifier_owned_input_ports(self):
        candidates = {
            "continuous": COMB.replace(
                "assign y = a;", "assign a = 1'b0; assign y = a;"
            ),
            "blocking procedural": (
                "module Tiny(input logic a, output logic y); "
                "always @(*) begin a = 1'b0; y = a; end endmodule\n"
            ),
            "nonblocking procedural": (
                "module Tiny(input logic a, output logic y); "
                "always @(*) begin a <= 1'b0; y = a; end endmodule\n"
            ),
            "concatenation": COMB.replace(
                "assign y = a;", "assign {y, a} = 2'b00;"
            ),
            "alias": COMB.replace("assign y = a;", "alias a = y;"),
            "bidirectional primitive": COMB.replace(
                "assign y = a;", "tran link(a, y);"
            ),
            "gate output": COMB.replace(
                "assign y = a;", "buf drive_input(a, y);"
            ),
            "multi-output gate": COMB.replace(
                "assign y = a;", "buf drive_two(y, a, a);"
            ),
            "later primitive instance": COMB.replace(
                "assign y = a;", "and legal(y, a, a), escape(a, y, y);"
            ),
            "strength-qualified primitive": COMB.replace(
                "assign y = a;", "and (strong1, pull0) escape(a, y, y);"
            ),
            "body direction redeclaration": COMB.replace(
                "assign y = a;", "output a; assign y = a;"
            ),
        }
        for label, candidate in candidates.items():
            with self.subTest(label=label):
                formal_pair, formal_reason = _prepare_pair("Tiny", COMB, candidate)
                simulation_pair, simulation_reason = _prepare_simulation_pair(
                    "Tiny", COMB, candidate
                )
                self.assertIsNone(formal_pair)
                self.assertIsNone(simulation_pair)
                self.assertIn("security preflight", formal_reason)
                self.assertTrue(
                    "input port" in formal_reason
                    or "input direction" in formal_reason,
                    formal_reason,
                )
                self.assertEqual(formal_reason, simulation_reason)

    def test_candidate_inout_port_is_explicitly_security_rejected(self):
        candidate = "module Tiny(inout a, output y); assign y = a; endmodule\n"
        formal_pair, formal_reason = _prepare_pair("Tiny", COMB, candidate)
        simulation_pair, simulation_reason = _prepare_simulation_pair(
            "Tiny", COMB, candidate
        )
        self.assertIsNone(formal_pair)
        self.assertIsNone(simulation_pair)
        self.assertIn("rejected inout port", formal_reason)
        self.assertEqual(formal_reason, simulation_reason)

    def test_input_read_and_output_or_internal_drives_remain_supported(self):
        legal_candidates = (
            COMB,
            "module Tiny(input a, output y); buf legal_gate(y, a); endmodule\n",
            (
                "module Tiny(input a, output logic y); logic internal; "
                "always @(*) begin internal = (a <= 1'b0); y = internal; end "
                "endmodule\n"
            ),
        )
        for candidate in legal_candidates:
            with self.subTest(candidate=candidate):
                formal_pair, formal_reason = _prepare_pair("Tiny", COMB, candidate)
                simulation_pair, simulation_reason = _prepare_simulation_pair(
                    "Tiny", COMB, candidate
                )
                self.assertIsNotNone(formal_pair, formal_reason)
                self.assertIsNotNone(simulation_pair, simulation_reason)

    def test_all_formal_and_simulation_harnesses_isolate_candidate_inputs(self):
        pair, reason = _prepare_pair("Tiny", COMB, COMB)
        self.assertFalse(reason)
        isolated_connection = ".a({rtlrepair_candidate_input_a})"
        for harness in (
            pair.wrapper_source,
            _generate_independent_tb(pair, 1),
            _generate_replay_tb(pair, [{"a": "0"}]),
        ):
            with self.subTest(harness=harness.splitlines()[1]):
                self.assertIn("wire rtlrepair_candidate_input_a;", harness)
                self.assertIn(
                    "assign rtlrepair_candidate_input_a = a;", harness
                )
                self.assertIn(isolated_connection, harness)
                self.assertIn("design_golden golden (.a(a),", harness)

    def test_candidate_security_preflight_rejects_verifier_injection(self):
        payloads = {
            "formal constraint": "always @* assume(a);",
            "formal system function": "wire injected = $anyconst;",
            "formal signal attribute": "(* anyconst *) reg injected;",
            "early termination": "initial $finish;",
            "protocol output spoof": 'initial $display("SIM_PASS");',
            "plusarg access": (
                'initial if ($test$plusargs("RTLREPAIR_VERDICT_NONCE")) begin end'
            ),
            "reserved hierarchy": "initial force independent_tb.mismatches = 0;",
            "embedded checker": "bind Tiny TinyChecker checker_i();",
        }
        for label, payload in payloads.items():
            with self.subTest(label=label):
                candidate = COMB.replace("endmodule", f"  {payload}\nendmodule")
                formal_pair, formal_reason = _prepare_pair("Tiny", COMB, candidate)
                simulation_pair, simulation_reason = _prepare_simulation_pair(
                    "Tiny", COMB, candidate
                )
                self.assertIsNone(formal_pair)
                self.assertIsNone(simulation_pair)
                self.assertIn("security preflight", formal_reason)
                self.assertEqual(formal_reason, simulation_reason)

    def test_candidate_security_preflight_rejects_extra_module(self):
        candidate = COMB + "module Spoof; initial $finish; endmodule\n"
        self.assertIn("exactly one module", _candidate_security_reason(candidate))

    def test_candidate_security_scanner_ignores_comments_and_strings(self):
        source = (
            "// assert assume $finish module endmodule\n"
            "module Tiny(input a, output y);\n"
            '  parameter [8*24-1:0] NOTE = "assert $finish module";\n'
            "  assign y = a; /* cover $display endmodule */\n"
            "endmodule\n"
        )
        self.assertEqual(_candidate_security_reason(source), "")

    def test_sby_064_status_suffix_is_not_part_of_semantic_status(self):
        self.assertEqual(_normalize_sby_status("PASS 0 0\n"), "PASS")
        self.assertEqual(_normalize_sby_status("ERROR 16 0\n"), "ERROR")

    def test_missing_yosys_is_unsupported_not_proved(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmarks.formal_verify.shutil.which", return_value=None
        ):
            result = FormalVerifier().verify("Tiny", COMB, COMB, Path(tmp))
        self.assertEqual(result.status, FormalStatus.UNSUPPORTED)

    def test_legal_unparsed_candidate_is_unsupported(self):
        non_ansi = "module Tiny(a,y); input a; output y; assign y=a; endmodule\n"
        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmarks.formal_verify._source_compiles_with_icarus", return_value=True
        ):
            result = FormalVerifier().verify("Tiny", COMB, non_ansi, Path(tmp))
        self.assertEqual(result.status, FormalStatus.UNSUPPORTED)

    def test_unparsed_candidate_that_fails_icarus_is_compile_fail(self):
        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmarks.formal_verify._source_compiles_with_icarus", return_value=False
        ), patch(
            "benchmarks.formal_verify._source_compiles_with_verilator", return_value=False
        ):
            result = FormalVerifier().verify("Tiny", COMB, "not verilog", Path(tmp))
        self.assertEqual(result.status, FormalStatus.COMPILE_FAIL)

    def test_prob151_unaudited_cast_guard_candidate_is_unsupported_before_tools(self):
        case = next(
            case for case in load_semantic_cases()
            if case.seed_id == "Prob151_review2015_fsm_ref"
        )
        drifted = case.canonical_rtl.replace("B0: next = B1;", "B0: next = B2;")
        with tempfile.TemporaryDirectory() as tmp:
            result = FormalVerifier().verify(
                case.seed_id, case.canonical_rtl, drifted, Path(tmp)
            )
        self.assertEqual(result.status, FormalStatus.UNSUPPORTED)
        self.assertIn("normalization rejected source drift", result.detail)

    def test_bmc_pass_cannot_become_proof(self):
        pair, reason = _prepare_pair("Seq", SEQ, SEQ)
        self.assertFalse(reason)
        command = CommandResult(["sby"], 0, "", "", 0.01)
        stages = iter(
            [
                ("TIMEOUT", command, None),
                ("PASS", command, None),
                ("UNKNOWN", command, None),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmarks.formal_verify.shutil.which", return_value="/bin/tool"
        ), patch.object(FormalVerifier, "_frontend_preflight", return_value=None), patch.object(
            FormalVerifier, "_run_sby", side_effect=lambda *args, **kwargs: next(stages)
        ):
            result = FormalVerifier(1, 1)._verify_sequential(pair, Path(tmp), 0.0)
        self.assertEqual(result.status, FormalStatus.TIMEOUT)
        self.assertFalse(result.metadata["bmc_pass_used_as_proof"])

    def test_pdr_and_induction_failures_are_not_reportable_witnesses(self):
        pair, _ = _prepare_pair("Seq", SEQ, SEQ)
        command = CommandResult(["sby"], 1, "", "", 0.01)
        fake_trace = Path("unreachable_induction_trace.vcd")
        stages = iter(
            [
                ("FAIL", command, fake_trace),
                ("PASS", command, None),
                ("FAIL", command, fake_trace),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmarks.formal_verify.shutil.which", return_value="/bin/tool"
        ), patch.object(FormalVerifier, "_frontend_preflight", return_value=None), patch.object(
            FormalVerifier, "_run_sby", side_effect=lambda *args, **kwargs: next(stages)
        ):
            result = FormalVerifier(1, 1)._verify_sequential(pair, Path(tmp), 0.0)
        self.assertEqual(result.status, FormalStatus.TIMEOUT)
        self.assertIsNone(result.witness)

    def test_backend_error_without_wallclock_timeout_is_unsupported(self):
        pair, _ = _prepare_pair("Seq", SEQ, SEQ)
        command = CommandResult(["sby"], 16, "backend rejected lowering", "", 0.01)
        stages = iter(
            [
                ("ERROR", command, None),
                ("ERROR", command, None),
                ("ERROR", command, None),
            ]
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmarks.formal_verify.shutil.which", return_value="/bin/tool"
        ), patch.object(FormalVerifier, "_frontend_preflight", return_value=None), patch.object(
            FormalVerifier, "_run_sby", side_effect=lambda *args, **kwargs: next(stages)
        ):
            result = FormalVerifier(1, 1)._verify_sequential(pair, Path(tmp), 0.0)
        self.assertEqual(result.status, FormalStatus.UNSUPPORTED)
        self.assertIn("backend error", result.detail)

    def test_single_clock_sby_config_lowers_checks_without_multiclock(self):
        config = FormalVerifier()._sby_config("prove", "abc pdr")
        self.assertIn("chformal -lower", config)
        self.assertIn("read_verilog -formal -sv", config)
        self.assertNotIn("\nread -formal -sv", config)
        self.assertIn("setattr -unset always_comb p:*", config)
        self.assertIn("select -clear", config)
        self.assertLess(
            config.index("setattr -unset always_comb p:*"),
            config.index("prep -top equiv_top"),
        )
        self.assertNotIn("multiclock", config)

    def test_combinational_sat_defines_formal_nondeterminism(self):
        pair, _ = _prepare_pair("Tiny", COMB, COMB)
        def completed(command, _cwd, _timeout):
            return CommandResult(command, 0, "", "", 0.01)

        with tempfile.TemporaryDirectory() as tmp, patch.object(
            FormalVerifier, "_frontend_preflight", return_value=None
        ), patch("benchmarks.formal_verify._run", side_effect=completed):
            result = FormalVerifier()._verify_combinational(pair, Path(tmp), 0.0)
        self.assertEqual(result.status, FormalStatus.PROVED)
        self.assertIn("-set-def-formal", result.command[-1])
        self.assertIn("memory_map", result.command[-1])
        self.assertLess(
            result.command[-1].index("setattr -unset always_comb p:*"),
            result.command[-1].index("prep -top equiv_top"),
        )

    def test_calibration_case_filter_rejects_unknown_id_before_tools(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            with self.assertRaisesRegex(ValueError, "unknown calibration"):
                run_calibration(
                    Path("data/repairbench_sem85.jsonl"),
                    base / "artifacts",
                    base / "results.jsonl",
                    base / "gate.json",
                    1,
                    1,
                    {"not-a-case"},
                )

    def test_protocol_fingerprint_binds_container_identity(self):
        with patch.dict(os.environ, {"RTLREPAIR_FORMAL_IMAGE_ID": "image-a"}):
            first = formal_protocol_fingerprint()
        with patch.dict(os.environ, {"RTLREPAIR_FORMAL_IMAGE_ID": "image-b"}):
            second = formal_protocol_fingerprint()
        self.assertNotEqual(first, second)

    def test_calibration_run_id_binds_protocol_fingerprint(self):
        contract = InitializationContract("combinational", "none")
        timeout = FormalResult(
            status=FormalStatus.TIMEOUT,
            engine="mock",
            initialization_contract=contract,
            elapsed_seconds=0.0,
            candidate_sha256="c",
            golden_sha256="g",
        )
        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmarks.formal_verify.write_toolchain_manifest", return_value={}
        ), patch.object(FormalVerifier, "verify", return_value=timeout), patch(
            "builtins.print"
        ):
            base = Path(tmp)
            protocol_sha = formal_protocol_fingerprint()
            run_calibration(
                Path("data/repairbench_sem85.jsonl"),
                base / "artifacts",
                base / "results.jsonl",
                base / "gate.json",
                1,
                1,
                {"repair_sem_0000"},
            )
            rows = [json.loads(line) for line in (base / "results.jsonl").read_text().splitlines()]
            gate = json.loads((base / "gate.json").read_text())
            results_sha = _sha256_file(base / "results.jsonl")
        self.assertEqual(len(rows), 2)
        self.assertTrue(all(row["formal_protocol_sha256"] == protocol_sha for row in rows))
        self.assertTrue(all(protocol_sha in row["run_id"] for row in rows))
        self.assertEqual(
            gate["calibration_results_sha256"], results_sha
        )
        gate_digest = gate.pop("gate_manifest_sha256")
        self.assertEqual(gate_digest, _canonical_json_sha256(gate))

    def test_allowlisted_fallback_is_attached_before_single_append_and_resumes(self):
        contract = InitializationContract("reset", "test reset")

        def verify(_self, seed_id, golden, candidate, _artifact_dir):
            status = FormalStatus.PROVED if golden == candidate else FormalStatus.UNSUPPORTED
            return FormalResult(
                status=status,
                engine="mock",
                initialization_contract=contract,
                elapsed_seconds=0.0,
                candidate_sha256=sha256_text(candidate),
                golden_sha256=sha256_text(golden),
            )

        def simulate(seed_id, golden, candidate, artifact_dir):
            protocol_sha = formal_protocol_fingerprint()
            report = {
                "status": "COUNTEREXAMPLE",
                "seeds": list(range(1001, 1011)),
                "cycles_per_seed": 200,
                "golden_sha256": sha256_text(golden),
                "candidate_sha256": sha256_text(candidate),
                "formal_protocol_sha256": protocol_sha,
                "counterexample_summary": {
                    "post_initialization_concrete_count": 10,
                    "first_post_initialization_concrete": {
                        "phase": "post_initialization",
                        "sample_point": "pre_edge",
                        "cycle": 4,
                        "output": "z",
                        "expected": "1",
                        "got": "0",
                        "concrete_two_state": True,
                        "seed": 1001,
                    },
                },
                "runs": [
                    {
                        "seed": seed,
                        "cycles": 200,
                        "status": "COUNTEREXAMPLE",
                        "passed": False,
                    }
                    for seed in range(1001, 1011)
                ],
            }
            artifact_dir.mkdir(parents=True, exist_ok=True)
            (artifact_dir / "independent_simulation.json").write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            return report

        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmarks.formal_verify.write_toolchain_manifest", return_value={}
        ), patch.object(FormalVerifier, "verify", autospec=True, side_effect=verify), patch(
            "benchmarks.formal_verify.run_independent_simulation", side_effect=simulate
        ) as simulation, patch("builtins.print"):
            base = Path(tmp)
            arguments = (
                Path("data/repairbench_sem85.jsonl"),
                base / "artifacts",
                base / "results.jsonl",
                base / "gate.json",
                1,
                1,
                {"repair_sem_0059"},
            )
            run_calibration(*arguments)
            rows = [
                json.loads(line)
                for line in (base / "results.jsonl").read_text().splitlines()
            ]
            self.assertEqual(len(rows), 2)
            self.assertEqual(len({row["run_id"] for row in rows}), 2)
            mutant = next(row for row in rows if row["comparison"] == "golden_vs_mutant")
            self.assertEqual(mutant["result"]["status"], "UNSUPPORTED")
            self.assertIn("composite_fallback", mutant)
            self.assertEqual(simulation.call_count, 1)

            # A complete, byte-identical sidecar resumes without another
            # simulation and without appending a duplicate comparison row.
            run_calibration(*arguments)
            self.assertEqual(simulation.call_count, 1)
            self.assertEqual(
                len((base / "results.jsonl").read_text().splitlines()), 2
            )


class WitnessTest(unittest.TestCase):
    def test_smtbmc_concrete_internal_initialization_is_replayable_but_x_is_not(self):
        trace_tb = """
    UUT.candidate.q = 4'b0000;
    UUT.candidate.hidden = 1'bx;
    UUT.candidate._witness_.anyinit_state = 4'b0011;
    UUT.formal_step = 2'b00;
    UUT.golden.bank[0] = 8'b10100101;
    UUT.golden.\\escaped = 1'b0;
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace_tb.v"
            path.write_text(trace_tb, encoding="utf-8")
            assignments = _parse_smtbmc_internal_init(path)
        self.assertEqual(
            assignments,
            [
                {"instance": "candidate", "path": "q", "value": "4'b0000"},
                {
                    "instance": "golden",
                    "path": "bank[0]",
                    "value": "8'b10100101",
                },
            ],
        )
        pair, _ = _prepare_pair("Seq", SEQ, SEQ)
        tb = _generate_replay_tb(
            pair,
            [{"clk": "1", "reset": "1", "d": "0"}],
            assignments[:1],
        )
        self.assertIn("candidate.q = 4'b0000;", tb)
        self.assertNotIn("candidate.hidden", tb)

    def test_vcd_input_trace_and_replay_tb_are_structured(self):
        vcd = """$scope module equiv_top $end
$var wire 1 ! a $end
$enddefinitions $end
#0
0!
#1
1!
"""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "trace.vcd"
            path.write_text(vcd, encoding="utf-8")
            states = _parse_vcd_states(path, ["a"], None)
        self.assertEqual(states, [{"a": "1"}])
        pair, _ = _prepare_pair("Tiny", COMB, COMB.replace("= a", "= ~a"))
        tb = _generate_replay_tb(pair, states)
        self.assertIn("TRACE_INPUT cycle=0 name=a", tb)
        self.assertIn("TRACE_OUTPUT cycle=0 name=y expected=%h got=%h", tb)

    def test_preamble_x_values_cannot_be_counted_as_counterexample(self):
        case = next(
            case for case in load_semantic_cases() if case.seed_id == "Prob054_edgedetect_ref"
        )
        pair, _ = _prepare_pair(case.seed_id, case.canonical_rtl, case.broken_rtl)
        states = [
            {"clk": "1", "in": "00000000"},
            {"clk": "1", "in": "00000001"},
        ]
        tb = _generate_replay_tb(pair, states)
        self.assertNotIn("REPLAY_MISMATCH cycle=0", tb)
        self.assertIn("REPLAY_MISMATCH cycle=1", tb)
        self.assertIn("!== 1'bx", tb)

    def test_negedge_replay_drives_one_to_zero_before_compare(self):
        case = next(
            case for case in load_semantic_cases() if case.seed_id == "Prob046_dff8p_ref"
        )
        pair, _ = _prepare_pair(case.seed_id, case.canonical_rtl, case.broken_rtl)
        tb = _generate_replay_tb(
            pair, [{"clk": "0", "reset": "1", "d": "00000000"}]
        )
        self.assertIn("clk = 1; #1;", tb)
        self.assertIn("clk = 0; #1;", tb)


class IndependentSimulationSamplingTest(unittest.TestCase):
    def test_nonce_verdict_requires_one_anchored_terminal_line(self):
        nonce = "0123456789abcdef0123456789abcdef"
        valid = f"VCD info\nRTLREPAIR_SIM_VERDICT nonce={nonce} status=PASS\n"
        self.assertEqual(_parse_independent_verdict(valid, nonce), ("PASS", ""))
        attacks = {
            "spoofed stale nonce": (
                "RTLREPAIR_SIM_VERDICT nonce=ffffffffffffffffffffffffffffffff "
                "status=PASS\n"
            ),
            "dual verdict": valid + (
                f"RTLREPAIR_SIM_VERDICT nonce={nonce} status=COUNTEREXAMPLE\n"
            ),
            "no verdict": "VCD info only\n",
            "nonterminal verdict": valid + "candidate trailing output\n",
            "unanchored verdict": (
                f"spoof RTLREPAIR_SIM_VERDICT nonce={nonce} status=PASS\n"
            ),
        }
        for label, log in attacks.items():
            with self.subTest(label=label):
                verdict, reason = _parse_independent_verdict(log, nonce)
                self.assertIsNone(verdict)
                self.assertTrue(reason)

    def test_independent_runs_use_fresh_unpredictable_nonces(self):
        nonces = [f"{index:032x}" for index in range(1, 11)]
        observed = []

        def run(command, _cwd, _timeout):
            if command[0] == "iverilog":
                return CommandResult(command, 0, "", "", 0.01)
            nonce_arg = next(
                value for value in command
                if value.startswith("+RTLREPAIR_VERDICT_NONCE=")
            )
            nonce = nonce_arg.split("=", 1)[1]
            observed.append(nonce)
            return CommandResult(
                command,
                0,
                f"RTLREPAIR_SIM_VERDICT nonce={nonce} status=PASS\n",
                "",
                0.01,
            )

        with tempfile.TemporaryDirectory() as tmp, patch(
            "benchmarks.formal_verify.shutil.which", return_value="/bin/tool"
        ), patch(
            "benchmarks.formal_verify.secrets.token_hex", side_effect=nonces
        ) as token_hex, patch(
            "benchmarks.formal_verify._run", side_effect=run
        ):
            report = run_independent_simulation("Tiny", COMB, COMB, Path(tmp))
        self.assertEqual(report["status"], "PASS")
        self.assertEqual(observed, nonces)
        self.assertEqual(token_hex.call_args_list, [call(16)] * 10)

    def test_independent_tb_nonce_register_holds_all_32_ascii_hex_digits(self):
        pair, reason = _prepare_pair("Tiny", COMB, COMB)
        self.assertFalse(reason)
        testbench = _generate_independent_tb(pair, 1)
        self.assertIn("reg [255:0] verdict_nonce;", testbench)
        self.assertIn(
            '$value$plusargs("RTLREPAIR_VERDICT_NONCE=%s", verdict_nonce)',
            testbench,
        )

    def test_prob151_simulation_preparation_keeps_original_four_state_rtl(self):
        case = next(
            case for case in load_semantic_cases()
            if case.seed_id == "Prob151_review2015_fsm_ref"
        )
        formal_pair, formal_reason = _prepare_pair(
            case.seed_id, case.canonical_rtl, case.canonical_rtl
        )
        simulation_pair, simulation_reason = _prepare_simulation_pair(
            case.seed_id, case.canonical_rtl, case.canonical_rtl
        )
        self.assertFalse(formal_reason)
        self.assertFalse(simulation_reason)
        self.assertNotIn("States'(", formal_pair.golden_source)
        self.assertNotIn("|state === 1'bx", formal_pair.golden_source)
        self.assertIn("States'(", simulation_pair.golden_source)
        self.assertIn("|state === 1'bx", simulation_pair.golden_source)
        self.assertFalse(simulation_pair.normalized_prob151)

    def test_prob151_format_drift_is_formal_unsupported_but_simulation_prepares(self):
        case = next(
            case for case in load_semantic_cases()
            if case.seed_id == "Prob151_review2015_fsm_ref"
        )
        drifted = case.canonical_rtl.replace("B0: next = B1;", "B0:  next = B1;")
        self.assertNotEqual(drifted, case.canonical_rtl)
        formal_pair, formal_reason = _prepare_pair(
            case.seed_id, case.canonical_rtl, drifted
        )
        simulation_pair, simulation_reason = _prepare_simulation_pair(
            case.seed_id, case.canonical_rtl, drifted
        )
        self.assertIsNone(formal_pair)
        self.assertIn("outside the audited", formal_reason)
        self.assertFalse(simulation_reason)
        self.assertIsNotNone(simulation_pair)
        self.assertIn("B0:  next = B1;", simulation_pair.candidate_source)
        self.assertIn("States'(", simulation_pair.candidate_source)

    def test_sequential_cycles_sample_before_and_after_active_edge(self):
        pair, _ = _prepare_pair("Seq", SEQ, SEQ.replace("q <= d", "q <= ~d"))
        tb = _generate_independent_tb(pair, 2)
        pre = tb.index("sample_point=pre_edge")
        edge = tb.index("clk = 1; #1;", pre)
        post = tb.index("sample_point=post_edge", edge)
        self.assertLess(pre, edge)
        self.assertLess(edge, post)
        self.assertIn("phase=post_initialization sample_point=pre_edge", tb)
        self.assertIn("phase=post_initialization sample_point=post_edge", tb)

    def test_prob154_ignores_invalid_payload_but_always_checks_done(self):
        case = next(
            case for case in load_semantic_cases()
            if case.seed_id == "Prob154_fsm_ps2data_ref"
        )
        pair, _ = _prepare_pair(case.seed_id, case.canonical_rtl, case.broken_rtl)
        tb = _generate_independent_tb(pair, 2)
        self.assertIn(
            "(done_golden === 1'b1) && (out_bytes_golden !== out_bytes_candidate)",
            tb,
        )
        self.assertNotIn("if (out_bytes_golden !== out_bytes_candidate)", tb)
        self.assertIn("if (done_golden !== done_candidate)", tb)

    def test_counterexample_summary_separates_reset_x_from_concrete_mealy_sample(self):
        summary = _summarize_sim_counterexamples(
            "SIM_COUNTEREXAMPLE phase=initialization sample_point=post_edge "
            "cycle=0 output=z expected=0 got=x\n"
            "SIM_COUNTEREXAMPLE phase=post_initialization sample_point=pre_edge "
            "cycle=3 output=z expected=1 got=0\n"
        )
        self.assertEqual(summary["total"], 2)
        self.assertEqual(summary["post_initialization_concrete_count"], 1)
        self.assertEqual(
            summary["first_post_initialization_concrete"]["sample_point"],
            "pre_edge",
        )
        self.assertEqual(
            summary["by_phase_and_sample_point"]["initialization:post_edge"], 1
        )


class GateTest(unittest.TestCase):
    @staticmethod
    def _composite_records(report_path: Path) -> list[dict]:
        protocol_sha = formal_protocol_fingerprint()
        amendment_sha = _sha256_file(DEFAULT_PROTOCOL_AMENDMENT)
        golden_sha = (
            "1ee2398da6c8170f1d7937eac091a160ec8bb037bc389ae0d66a8d19a5a49716"
        )
        candidate_sha = (
            "35c79ef6db0aec962710791bc4a3465979d011f1ae883379ea5fdc1d6f2b57c4"
        )
        report = {
            "status": "COUNTEREXAMPLE",
            "seeds": list(range(1001, 1011)),
            "cycles_per_seed": 200,
            "golden_sha256": golden_sha,
            "candidate_sha256": candidate_sha,
            "formal_protocol_sha256": protocol_sha,
            "protocol_amendment_sha256": amendment_sha,
            "counterexample_summary": {
                "post_initialization_concrete_count": 10,
                "first_post_initialization_concrete": {
                    "phase": "post_initialization",
                    "sample_point": "pre_edge",
                    "cycle": 4,
                    "output": "z",
                    "expected": "1",
                    "got": "0",
                    "concrete_two_state": True,
                    "seed": 1001,
                },
            },
            "runs": [
                {
                    "seed": seed,
                    "cycles": 200,
                    "status": "COUNTEREXAMPLE",
                    "passed": False,
                }
                for seed in range(1001, 1011)
            ],
        }
        report_path.write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        records = []
        for index in range(85):
            case_id = f"repair_sem_{index:04d}"
            records.append(
                {
                    "run_id": f"{case_id}:golden",
                    "case_id": case_id,
                    "comparison": "golden_vs_golden",
                    "formal_protocol_sha256": protocol_sha,
                    "protocol_amendment_sha256": amendment_sha,
                    "result": {"status": "PROVED"},
                }
            )
            mutant = {
                "run_id": f"{case_id}:mutant",
                "case_id": case_id,
                "seed_id": f"seed-{index}",
                "mutation": "test",
                "comparison": "golden_vs_mutant",
                "formal_protocol_sha256": protocol_sha,
                "protocol_amendment_sha256": amendment_sha,
                "result": {
                    "status": "COUNTEREXAMPLE",
                    "counterexample_replayed": True,
                },
            }
            if index == 59:
                mutant.update(
                    {
                        "seed_id": "Prob129_ece241_2013_q8_ref",
                        "mutation": "flip_reset_polarity",
                        "result": {
                            "status": "UNSUPPORTED",
                            "counterexample_replayed": None,
                            "golden_sha256": golden_sha,
                            "candidate_sha256": candidate_sha,
                        },
                        "composite_fallback": {
                            "schema_version": 1,
                            "policy_name": (
                                "exact_allowlisted_unsupported_with_frozen_"
                                "independent_simulation"
                            ),
                            "policy_version": 1,
                            "case_id": case_id,
                            "seed_id": "Prob129_ece241_2013_q8_ref",
                            "golden_sha256": golden_sha,
                            "candidate_sha256": candidate_sha,
                            "formal_protocol_sha256": protocol_sha,
                            "protocol_amendment_sha256": amendment_sha,
                            "report_path": str(report_path),
                            "report_sha256": _sha256_file(report_path),
                        },
                    }
                )
            records.append(mutant)
        return records

    def test_calibration_halts_on_false_mutant_proof(self):
        rows = [
            {
                "case_id": "x",
                "comparison": "golden_vs_mutant",
                "result": {"status": "PROVED"},
            }
        ]
        gate = calibration_gate(rows, expected_cases=1)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["mutants_incorrectly_proved"], ["x"])

    def test_composite_calibration_gate_accepts_only_exact_84_plus_1(self):
        with tempfile.TemporaryDirectory() as tmp:
            records = self._composite_records(Path(tmp) / "simulation.json")
            gate = calibration_gate(records)
        self.assertTrue(gate["passed"])
        self.assertEqual(gate["schema_version"], 3)
        self.assertEqual(gate["replayable_mutant_counterexamples"], 84)
        self.assertEqual(gate["fallback_mutant_counterexamples"], 1)
        self.assertEqual(gate["distinguishable_mutants"], 85)
        self.assertEqual(
            gate["fallback_mutant_evidence"][0]["counterexample_seed_runs"], 10
        )

    def test_composite_calibration_gate_rejects_report_byte_drift(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "simulation.json"
            records = self._composite_records(report_path)
            report_path.write_text(
                report_path.read_text(encoding="utf-8") + " ", encoding="utf-8"
            )
            gate = calibration_gate(records)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["fallback_mutant_counterexamples"], 0)
        self.assertIn(
            "byte SHA",
            " ".join(gate["fallback_validation_errors"][0]["errors"]),
        )

    def test_composite_calibration_gate_rejects_one_non_ce_seed(self):
        with tempfile.TemporaryDirectory() as tmp:
            report_path = Path(tmp) / "simulation.json"
            records = self._composite_records(report_path)
            report = json.loads(report_path.read_text(encoding="utf-8"))
            report["runs"][-1]["status"] = "PASS"
            report["runs"][-1]["passed"] = True
            report_path.write_text(
                json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            fallback = next(
                row for row in records if row["case_id"] == "repair_sem_0059"
                and row["comparison"] == "golden_vs_mutant"
            )
            fallback["composite_fallback"]["report_sha256"] = _sha256_file(report_path)
            gate = calibration_gate(records)
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["fallback_mutant_counterexamples"], 0)

    def test_composite_calibration_gate_rejects_every_other_mutant_outcome(self):
        for status in ("UNSUPPORTED", "TIMEOUT", "COMPILE_FAIL", "PROVED"):
            with self.subTest(status=status), tempfile.TemporaryDirectory() as tmp:
                records = self._composite_records(Path(tmp) / "simulation.json")
                ordinary = next(
                    row for row in records if row["case_id"] == "repair_sem_0000"
                    and row["comparison"] == "golden_vs_mutant"
                )
                ordinary["result"] = {
                    "status": status,
                    "counterexample_replayed": None,
                }
                gate = calibration_gate(records)
            self.assertFalse(gate["passed"])

    def test_formal_primary_gate_requires_coverage_and_zero_conflicts(self):
        rows = []
        for index in range(3):
            rows.append(
                {
                    "case_id": str(index),
                    "arm": "a",
                    "formal_result": {"status": "PROVED"},
                    "simulation_result": {"status": "PASS"},
                }
            )
        rows.extend(
            [
                {
                    "case_id": "fixture",
                    "mode": "preflight",
                    "nonresearch": True,
                    "arm": "a",
                    "formal_result": {"status": "PROVED"},
                    "simulation_result": {"status": "PASS"},
                },
                {
                    "case_id": "redacted",
                    "mode": "redacted",
                    "arm": "redacted_spec_loc0",
                    "formal_result": {"status": "PROVED"},
                    "simulation_result": {"status": "PASS"},
                },
            ]
        )
        gate = formal_primary_gate(
            rows, expected_cases_per_arm=3, minimum_definitive=2, required_arms=("a",)
        )
        self.assertTrue(gate["passed"])
        rows[0]["simulation_result"]["status"] = "COUNTEREXAMPLE"
        gate = formal_primary_gate(
            rows, expected_cases_per_arm=3, minimum_definitive=2, required_arms=("a",)
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["conflicts"]), 1)

    def test_formal_primary_gate_rejects_inconclusive_independent_simulation(self):
        rows = [
            {
                "case_id": "x",
                "arm": "a",
                "formal_result": {"status": "PROVED"},
                "simulation_result": {"status": "TIMEOUT"},
            }
        ]
        gate = formal_primary_gate(
            rows, expected_cases_per_arm=1, minimum_definitive=1, required_arms=("a",)
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(
            gate["inconclusive_simulation"],
            [{"arm": "a", "case_id": "x", "simulation_status": "TIMEOUT"}],
        )

    def test_formal_primary_gate_requires_unique_nonempty_shared_roster(self):
        def row(case_id, arm):
            return {
                "case_id": case_id,
                "arm": arm,
                "formal_result": {"status": "PROVED"},
                "simulation_result": {"status": "PASS"},
            }

        duplicate = [row("x", "a"), row("x", "a")]
        gate = formal_primary_gate(
            duplicate, expected_cases_per_arm=2, minimum_definitive=2,
            required_arms=("a",),
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(gate["arms"]["a"]["duplicate_case_ids"], ["x"])

        missing = [row("x", "a"), row("", "a")]
        gate = formal_primary_gate(
            missing, expected_cases_per_arm=2, minimum_definitive=2,
            required_arms=("a",),
        )
        self.assertFalse(gate["passed"])
        self.assertEqual(len(gate["arms"]["a"]["invalid_case_id_record_indexes"]), 1)

        swapped = [
            row("x", "a"), row("y", "a"),
            row("x", "b"), row("z", "b"),
        ]
        gate = formal_primary_gate(
            swapped, expected_cases_per_arm=2, minimum_definitive=2,
            required_arms=("a", "b"),
        )
        self.assertFalse(gate["passed"])
        self.assertFalse(gate["arm_case_id_sets_match"])
        self.assertEqual(
            gate["case_id_roster_mismatches"]["b"],
            {"missing_from_reference_arm": ["y"], "extra_vs_reference_arm": ["z"]},
        )

    def test_event_ledger_rejects_duplicate_response_and_verdict_rows(self):
        base_response = {"event": "response", "call_id": "c1", "arm": "a"}
        base_verdict = {"event": "verdict", "call_id": "c1"}
        for event_name, rows in (
            ("response", [base_response, dict(base_response), base_verdict]),
            ("verdict", [base_response, base_verdict, dict(base_verdict)]),
        ):
            with self.subTest(event=event_name):
                _records, errors = _merge_event_ledger(rows)
                self.assertEqual(len(errors), 1)
                self.assertIn(f"duplicate {event_name}", errors[0])

    def test_formal_gate_fails_closed_on_duplicate_event_ledger_row(self):
        response = {
            "event": "response",
            "call_id": "c1",
            "case_id": "x",
            "arm": "a",
        }
        verdict = {
            "event": "verdict",
            "call_id": "c1",
            "formal_result": {"status": "PROVED"},
            "simulation_result": {"status": "PASS"},
        }
        with tempfile.TemporaryDirectory() as tmp, patch("builtins.print"):
            base = Path(tmp)
            records_path = base / "events.jsonl"
            records_path.write_text(
                "\n".join(
                    json.dumps(row)
                    for row in (response, verdict, dict(verdict))
                )
                + "\n",
                encoding="utf-8",
            )
            output_path = base / "gate.json"
            returncode = _formal_gate(
                SimpleNamespace(
                    records=records_path,
                    output=output_path,
                    expected_cases=1,
                    minimum_definitive=1,
                    required_arm=["a"],
                )
            )
            report = json.loads(output_path.read_text(encoding="utf-8"))
        self.assertEqual(returncode, 3)
        self.assertFalse(report["passed"])
        self.assertIn("duplicate verdict", report["event_ledger_errors"][0])


if __name__ == "__main__":
    unittest.main()
