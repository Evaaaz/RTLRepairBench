from __future__ import annotations

import hashlib
import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from candidate_security import candidate_security_reason, input_port_drive_reason
from benchmarks import formal_data, formal_protocol, formal_verify, repairbench_eval
from datagen import build_benchmark_tasks
from datagen import build_semantic_repair
from datagen import decontaminate
from datagen import diff_testbench
from datagen import run_verilator_for_repairs
from project_paths import data_root, discover_project_root, output_root, project_root


class ProjectPathTests(unittest.TestCase):
    def test_nearest_marked_bundle_and_explicit_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            outer = Path(tmp) / "outer"
            bundle = outer / "workshop" / "lint-vs-semantic"
            code = bundle / "code"
            code.mkdir(parents=True)
            (outer / "data").mkdir()
            (bundle / "configs").mkdir()

            self.assertEqual(
                discover_project_root(code_root=code, environ={}), bundle.resolve()
            )
            override = Path(tmp) / "explicit"
            self.assertEqual(
                discover_project_root(
                    code_root=code, environ={"RTLREPAIR_ROOT": str(override)}
                ),
                override.resolve(),
            )

    def test_bare_extraction_falls_back_to_parent_of_code(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = Path(tmp) / "release" / "code"
            code.mkdir(parents=True)
            self.assertEqual(
                discover_project_root(code_root=code, environ={}), code.parent.resolve()
            )

    def test_data_and_output_overrides_are_resolved(self):
        with tempfile.TemporaryDirectory() as tmp, mock.patch.dict(
            "os.environ",
            {
                "RTLREPAIR_DATA": str(Path(tmp) / "data"),
                "RTLREPAIR_OUT": str(Path(tmp) / "output"),
            },
            clear=True,
        ):
            self.assertEqual(data_root(), (Path(tmp) / "data").resolve())
            self.assertEqual(output_root(), (Path(tmp) / "output").resolve())


class DecontaminationPreflightTests(unittest.TestCase):
    def test_collection_requires_every_held_out_source(self):
        with self.assertRaises(build_benchmark_tasks.IncompleteHeldOutError):
            build_benchmark_tasks.validate_collected_tasks([])
        with self.assertRaisesRegex(
            build_benchmark_tasks.IncompleteHeldOutError, "rtllm"
        ):
            build_benchmark_tasks.validate_collected_tasks(
                [
                    {
                        "id": "ve:tiny",
                        "source": "verilogeval",
                        "code": "module Tiny; endmodule\n",
                    }
                ]
            )

        counts = build_benchmark_tasks.validate_collected_tasks(
            [
                {
                    "id": "ve:tiny",
                    "source": "verilogeval",
                    "code": "module Tiny; endmodule\n",
                },
                {
                    "id": "rtllm:tiny",
                    "source": "rtllm",
                    "code": "module Tiny; endmodule\n",
                },
            ]
        )
        self.assertEqual(counts, {"rtllm": 1, "verilogeval": 1})
        with self.assertRaisesRegex(
            build_benchmark_tasks.IncompleteHeldOutError, "non-module RTL"
        ):
            build_benchmark_tasks.validate_collected_tasks(
                [
                    {"id": "ve:bad", "source": "verilogeval", "code": ""},
                    {
                        "id": "rtllm:tiny",
                        "source": "rtllm",
                        "code": "module Tiny; endmodule\n",
                    },
                ]
            )

    def test_missing_or_empty_benchmark_is_fatal_by_default(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "held-out.jsonl"
            with self.assertRaises(decontaminate.EmptyBenchmarkError):
                decontaminate.load_held_out(path)
            path.write_text("\n", encoding="utf-8")
            with self.assertRaises(decontaminate.EmptyBenchmarkError):
                decontaminate.load_held_out(path)
            self.assertEqual(
                decontaminate.load_held_out(path, allow_empty=True),
                decontaminate.HeldOut(),
            )


class SemanticGatePreflightTests(unittest.TestCase):
    def test_missing_simulator_and_verilator_are_loud(self):
        with mock.patch.object(diff_testbench.shutil, "which", return_value=None):
            with self.assertRaises(diff_testbench.SimulatorUnavailableError):
                diff_testbench.require_simulator()
        with mock.patch.object(
            run_verilator_for_repairs.shutil, "which", return_value=None
        ):
            with self.assertRaises(
                run_verilator_for_repairs.VerilatorUnavailableError
            ):
                run_verilator_for_repairs.require_verilator()

    def test_simulator_nonzero_or_missing_verdict_cannot_count_as_pass(self):
        source = "module Tiny(input a, output y); assign y = a; endmodule\n"
        info = diff_testbench.parse_module(source)
        self.assertIsNotNone(info)

        compile_ok = subprocess.CompletedProcess([], 0, "", "")
        compile_failed = subprocess.CompletedProcess([], 1, "", "compile error")
        runtime_failed = subprocess.CompletedProcess([], 1, "", "runtime error")
        no_verdict = subprocess.CompletedProcess([], 0, "ordinary output", "")
        with mock.patch.object(diff_testbench, "require_simulator"), mock.patch.object(
            diff_testbench.subprocess,
            "run",
            side_effect=[compile_failed],
        ):
            result = diff_testbench.run_diff_test(source, source, info)
            self.assertFalse(result.compiled)
            self.assertEqual(result.failure_kind, "compile_failure")
        with mock.patch.object(diff_testbench, "require_simulator"), mock.patch.object(
            diff_testbench.subprocess,
            "run",
            side_effect=[compile_ok, runtime_failed],
        ):
            with self.assertRaises(diff_testbench.SimulatorProtocolError):
                diff_testbench.run_diff_test(source, source, info)
        with mock.patch.object(diff_testbench, "require_simulator"), mock.patch.object(
            diff_testbench.subprocess,
            "run",
            side_effect=[compile_ok, no_verdict],
        ):
            with self.assertRaises(diff_testbench.SimulatorProtocolError):
                diff_testbench.run_diff_test(source, source, info)

    def test_simulator_verdict_is_nonce_authenticated_unique_and_terminal(self):
        source = "module Tiny(input a, output y); assign y = a; endmodule\n"
        info = diff_testbench.parse_module(source)
        self.assertIsNotNone(info)
        nonce = "ab" * 16
        compile_ok = subprocess.CompletedProcess([], 0, "", "")

        def run_with(stdout):
            runtime = subprocess.CompletedProcess([], 0, stdout, "")
            patches = (
                mock.patch.object(diff_testbench, "require_simulator"),
                mock.patch.object(diff_testbench.secrets, "token_hex", return_value=nonce),
                mock.patch.object(
                    diff_testbench.subprocess,
                    "run",
                    side_effect=[compile_ok, runtime],
                ),
            )
            return patches

        for spoofed in (
            "DIFF_PASS\n",
            f"RTLREPAIR_DIFF_{nonce} PASS\nRTLREPAIR_DIFF_{nonce} FAIL first_cycle=1\n",
            f"RTLREPAIR_DIFF_{nonce} PASS\nlate output\n",
        ):
            first, second, third = run_with(spoofed)
            with first, second, third:
                with self.assertRaises(diff_testbench.SimulatorProtocolError):
                    diff_testbench.run_diff_test(source, source, info)

        valid = f"RTLREPAIR_DIFF_{nonce} PASS\n"
        first, second, third = run_with(valid)
        with first, second, third:
            result = diff_testbench.run_diff_test(source, source, info)
        self.assertTrue(result.compiled)
        self.assertFalse(result.diverged)

        failed = (
            f"RTLREPAIR_DIFF_DETAIL_{nonce} OUTDIFF y 2\n"
            f"RTLREPAIR_DIFF_{nonce} FAIL first_cycle=3\n"
        )
        first, second, third = run_with(failed)
        with first, second, third:
            result = diff_testbench.run_diff_test(source, source, info)
        self.assertTrue(result.diverged)
        self.assertEqual(result.first_cycle, 3)
        self.assertEqual(result.out_diffs, {"y": 2})

    def test_diff_harness_names_are_unpredictable_and_candidate_cannot_escape(self):
        source = "module Tiny(input a, output y); assign y = a; endmodule\n"
        info = diff_testbench.parse_module(source)
        self.assertIsNotNone(info)
        nonce = "cd" * 16
        tb = diff_testbench.generate_tb(info, protocol_nonce=nonce)
        self.assertIsNotNone(tb)
        self.assertIn(f"module rtlrepair_diff_tb_{nonce};", tb)
        self.assertIn(f"dut_good_{nonce}", tb)
        self.assertIn(f"fc_good_in_a_{nonce}", tb)
        self.assertIn(f"fc_bad_in_a_{nonce}", tb)
        self.assertRegex(
            tb,
            rf"assign fc_bad_in_a_{nonce} = fc_in_a_{nonce};",
        )
        self.assertNotIn("module tb;", tb)

        payloads = (
            "assign y = tb.y_good;",
            "initial force dut_good.y = 1'b0;",
            "initial $display(\"RTLREPAIR_DIFF_spoof PASS\");",
            "initial $finish;",
            "always @* assume(a);",
            "assign y = $root.tb.y_good;",
            "`include \"payload.sv\"",
            "assign a = 1'b0; assign y = a;",
            "always @(*) a = 1'b0;",
            "always @(posedge a) a <= 1'b0;",
            "tran (a, y);",
            "buf (y, a, a);",
            "not (y, a, a);",
            "and g1(y, a, a), g2(a, y, y);",
            "and (strong1, pull0) g1(a, y, y);",
        )
        with mock.patch.object(diff_testbench, "require_simulator"):
            for payload in payloads:
                with self.subTest(payload=payload):
                    candidate = source.replace("assign y = a;", payload)
                    result = diff_testbench.run_diff_test(source, candidate, info)
                    self.assertFalse(result.compiled)
                    self.assertEqual(result.failure_kind, "security_rejection")

        extra_module = source + "module tb; initial $finish; endmodule\n"
        with mock.patch.object(diff_testbench, "require_simulator"):
            result = diff_testbench.run_diff_test(source, extra_module, info)
        self.assertEqual(result.failure_kind, "security_rejection")

        changed_direction = source.replace("input a", "output a")
        with mock.patch.object(diff_testbench, "require_simulator"):
            result = diff_testbench.run_diff_test(source, changed_direction, info)
        self.assertEqual(result.failure_kind, "security_rejection")

        body_redeclaration = source.replace(
            "assign y = a;", "inout a; assign y = a;"
        )
        with mock.patch.object(diff_testbench, "require_simulator"):
            result = diff_testbench.run_diff_test(source, body_redeclaration, info)
        self.assertEqual(result.failure_kind, "security_rejection")

    def test_security_scanner_distinguishes_wildcard_event_from_attribute(self):
        source = (
            "module Tiny(input a, input b, output reg y);\n"
            "always @(*) y = (a <= b);\n"
            "endmodule\n"
        )
        info = diff_testbench.parse_module(source)
        self.assertIsNotNone(info)
        compile_ok = subprocess.CompletedProcess([], 0, "", "")
        nonce = "ef" * 16
        runtime_ok = subprocess.CompletedProcess(
            [], 0, f"RTLREPAIR_DIFF_{nonce} PASS\n", ""
        )
        with mock.patch.object(diff_testbench, "require_simulator"), mock.patch.object(
            diff_testbench.secrets, "token_hex", return_value=nonce
        ), mock.patch.object(
            diff_testbench.subprocess, "run", side_effect=[compile_ok, runtime_ok]
        ):
            result = diff_testbench.run_diff_test(source, source, info)
        self.assertTrue(result.compiled)

    def test_released_realbug_rtl_remains_inside_security_subset(self):
        rows = [
            json.loads(line)
            for line in (data_root() / "repairbench_realbugs.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        self.assertEqual(len(rows), 274)
        for row in rows:
            for role in ("golden_rtl", "broken_rtl"):
                source = row[role]
                with self.subTest(case_id=row["id"], role=role):
                    self.assertEqual(candidate_security_reason(source), "")
                    parsed = diff_testbench.parse_module(source)
                    if parsed is not None:
                        self.assertEqual(
                            input_port_drive_reason(
                                source, (port.name for port in parsed.inputs)
                            ),
                            "",
                        )

    def test_zero_certified_pairs_does_not_create_success_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp) / "semantic.jsonl"
            with mock.patch.object(build_semantic_repair, "require_verilator"), mock.patch.object(
                build_semantic_repair, "require_simulator"
            ), mock.patch.object(build_semantic_repair, "_process_seed", return_value=[]):
                with self.assertRaises(build_semantic_repair.EmptySemanticDatasetError):
                    build_semantic_repair.build(
                        [("Tiny.sv", "module Tiny; endmodule\n")], output, jobs=1
                    )
            self.assertFalse(output.exists())


class RepairBenchCompletenessTests(unittest.TestCase):
    def test_malformed_benchmark_is_rejected_before_inference(self):
        errors = repairbench_eval.benchmark_structure_errors(
            [
                {
                    "id": "duplicate",
                    "bucket": "unknown",
                    "seed_id": "",
                    "mutation": "",
                    "messages": [],
                },
                {
                    "id": "duplicate",
                    "bucket": "lint",
                    "seed_id": "Tiny",
                    "mutation": "flip",
                    "messages": [],
                },
            ]
        )
        self.assertTrue(any("not unique" in error for error in errors))
        self.assertTrue(any("bucket" in error for error in errors))
        self.assertTrue(any("broken RTL" in error for error in errors))

    def test_semantic_goldens_require_complete_hash_verified_set(self):
        items = [
            {"id": "case-0", "bucket": "semantic", "seed_id": "Tiny"},
            {"id": "case-1", "bucket": "semantic", "seed_id": "Other"},
        ]
        with tempfile.TemporaryDirectory() as tmp:
            seeds = Path(tmp)
            tiny = b"module Tiny; endmodule\n"
            other = b"module Other; endmodule\n"
            (seeds / "Tiny.sv").write_bytes(tiny)
            (seeds / "Other.sv").write_bytes(other)
            manifest = {
                "Tiny": hashlib.sha256(tiny).hexdigest(),
                "Other": hashlib.sha256(other).hexdigest(),
            }
            (seeds / "_manifest.sha256.json").write_text(
                json.dumps(manifest), encoding="utf-8"
            )
            self.assertEqual(
                repairbench_eval.semantic_golden_errors(items, seeds_dir=seeds), []
            )

            (seeds / "Other.sv").write_text(
                "module Other; wire drift; endmodule\n", encoding="utf-8"
            )
            self.assertTrue(
                any(
                    "mismatch for Other" in error
                    for error in repairbench_eval.semantic_golden_errors(
                        items, seeds_dir=seeds
                    )
                )
            )

    def test_arm_requires_exact_ids_model_responses_and_specs(self):
        items = [
            {"id": "case-0"},
            {"id": "case-1"},
        ]
        records = [
            {"id": "case-0", "used_llm": True, "had_spec": True},
            {"id": "wrong", "used_llm": False, "had_spec": False},
        ]
        errors = repairbench_eval.run_completeness_errors(
            items, records, require_specs=True
        )
        self.assertTrue(any("result ids" in error for error in errors))
        self.assertTrue(any("model response" in error for error in errors))
        self.assertTrue(any("spec" in error for error in errors))
        self.assertIn(
            "benchmark contains zero cases",
            repairbench_eval.run_completeness_errors(
                [], [], require_specs=False
            ),
        )


class FormalStandaloneImportTests(unittest.TestCase):
    def test_formal_modules_share_the_extraction_safe_root(self):
        expected = project_root()
        self.assertEqual(formal_data.ROOT, expected)
        self.assertEqual(formal_protocol.ROOT, expected)
        self.assertEqual(formal_verify.ROOT, expected)


if __name__ == "__main__":
    unittest.main()
