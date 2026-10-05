from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from datagen.diff_testbench import DiffResult


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "code" / "scripts"


def load_script(name):
    module_name = f"test_loaded_{name}"
    spec = importlib.util.spec_from_file_location(module_name, SCRIPTS / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


class ExploratoryFailClosedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.decomp = load_script("decomp_second_anchor")
        cls.reflect = load_script("self_reflect")
        cls.contract = load_script("contract_robustness_experiment")
        cls.realbug = load_script("realbug_frontier")
        cls.general_score = load_script("general_score")
        cls.e4 = load_script("e4_tier_s_rescore")
        cls.curve = load_script("second_model_capability_curve")
        cls.e5 = load_script("e5_formal_coverage")

    def test_iterative_exact_mcnemar_matches_reported_six_to_zero(self):
        self.assertEqual(self.reflect.exact_mcnemar_p(6, 0), 0.03125)
        self.assertEqual(self.reflect.exact_mcnemar_p(0, 0), 1.0)

    def test_curve68_manifest_matches_public_frozen_roster(self):
        manifest = json.loads(
            (ROOT / "configs/vericodegen/curve68_manifest.json").read_text()
        )
        public_e4 = json.loads(
            (ROOT / "artifacts/public/primary/analysis/e4_tier_s.json").read_text()
        )
        observed = sorted({row["case_id"] for row in public_e4["rows"]})
        self.assertEqual(manifest["case_ids"], observed)
        canonical = json.dumps(observed, separators=(",", ":")).encode()
        self.assertEqual(
            manifest["case_ids_sha256"], hashlib.sha256(canonical).hexdigest()
        )
        self.assertEqual(
            manifest["cohort"], "retained_equivalence_harness_scheduled_cohort"
        )
        formal = json.loads(
            (ROOT / "configs/vericodegen/anchor_formal68_manifest.json").read_text()
        )
        self.assertEqual(formal["count"], 68)
        self.assertEqual(len(set(formal["case_ids"]) & set(manifest["case_ids"])), 52)
        self.assertNotEqual(formal["case_ids"], manifest["case_ids"])

        curve = json.loads(
            (ROOT / "artifacts/public/capability/capability_curve.json").read_text()
        )
        self.assertEqual(len(curve), 5)
        for row in curve:
            self.assertEqual(row["case_ids_sha256"], manifest["case_ids_sha256"])
            self.assertEqual(row["scoring_rule"], "strict_formal_proved_only")
        anchor = next(row for row in curve if row["model"] == "azure/openai/gpt-5.5")
        self.assertEqual(
            (anchor["baseline_pct"], anchor["what_pp"], anchor["where_pp"]),
            (82.4, 4.5, 3.6),
        )

    def test_capability_run_loader_rejects_missing_duplicate_and_unknown_verdicts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "generated" / "run"
            run.mkdir(parents=True)
            results = [
                {
                    "case_id": "case_1",
                    "arm": arm,
                    "parsed_ok": True,
                    "design": "cluster_1",
                }
                for arm in self.curve.ARMS
            ]
            (run / "results.json").write_text(json.dumps(results), encoding="utf-8")
            verdict_rows = [
                {"dir": f"{arm}__case_1", "verdict": "PROVED"}
                for arm in self.curve.ARMS[:-1]
            ]
            (run / "verdicts.jsonl").write_text(
                "\n".join(json.dumps(row) for row in verdict_rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "missing verdict"):
                self.curve.load_run(
                    root,
                    Path("generated/run"),
                    "synthetic",
                    ("case_1",),
                )
            with self.assertRaisesRegex(RuntimeError, "missing verdict"):
                with mock.patch.object(self.e5, "ROOT", root):
                    self.e5.run_arms(
                        "generated/run", "synthetic", {"case_1"}
                    )

            verdict_rows.append(
                {"dir": f"{self.curve.ARMS[-1]}__case_1", "verdict": "BANANA"}
            )
            (run / "verdicts.jsonl").write_text(
                "\n".join(json.dumps(row) for row in verdict_rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "invalid verdict"):
                self.curve.load_run(
                    root,
                    Path("generated/run"),
                    "synthetic",
                    ("case_1",),
                )

            verdict_rows[-1]["verdict"] = "ERROR"
            verdict_rows.append(dict(verdict_rows[-1]))
            (run / "verdicts.jsonl").write_text(
                "\n".join(json.dumps(row) for row in verdict_rows) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate verdict"):
                self.curve.load_run(
                    root,
                    Path("generated/run"),
                    "synthetic",
                    ("case_1",),
                )

    def test_capability_manifest_rejects_same_named_but_partial_cohort(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "curve68_manifest.json"
            case_ids = ["case_1"]
            path.write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "cohort": self.curve.EXPECTED_COHORT,
                        "count": 1,
                        "case_ids_sha256": hashlib.sha256(
                            json.dumps(case_ids, separators=(",", ":")).encode()
                        ).hexdigest(),
                        "case_ids": case_ids,
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "exactly 68"):
                self.curve.load_manifest(path)

    def test_capability_run_loader_allows_explicit_no_parse_without_verdict(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            run = root / "generated" / "run"
            run.mkdir(parents=True)
            results = []
            verdicts = []
            for index, arm in enumerate(self.curve.ARMS):
                parsed = index != 0
                results.append(
                    {
                        "case_id": "case_1",
                        "arm": arm,
                        "parsed_ok": parsed,
                        "design": "cluster_1",
                    }
                )
                if parsed:
                    verdicts.append(
                        {"dir": f"{arm}__case_1", "verdict": "COUNTEREXAMPLE"}
                    )
            (run / "results.json").write_text(json.dumps(results), encoding="utf-8")
            (run / "verdicts.jsonl").write_text(
                "\n".join(json.dumps(row) for row in verdicts) + "\n",
                encoding="utf-8",
            )
            loaded, audit = self.curve.load_run(
                root,
                Path("generated/run"),
                "synthetic",
                ("case_1",),
            )
            self.assertEqual(audit["parsed_cells"], 3)
            self.assertEqual(
                loaded["rec"][("case_1", self.curve.ARMS[0])]["verdict"],
                "NO_PARSE",
            )

    def test_iterative_run_refuses_existing_output_before_endpoint_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "immutable"
            out.mkdir()
            (out / "marker").write_text("prior run", encoding="utf-8")
            with mock.patch.object(self.reflect, "OUT", out), mock.patch.object(
                self.reflect, "require_simulator"
            ), mock.patch.object(self.reflect, "load_inputs") as load_inputs:
                with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
                    self.reflect.main()
                load_inputs.assert_not_called()

    def test_second_anchor_distinguishes_model_compile_failure_from_protocol_error(self):
        source = "module Tiny(input a, output y); assign y = a; endmodule"
        with mock.patch.object(
            self.decomp,
            "run_diff_test",
            return_value=DiffResult(compiled=False, failure_kind="compile_failure"),
        ):
            self.assertFalse(self.decomp.sim_ok(source, source))
        with mock.patch.object(
            self.decomp,
            "run_diff_test",
            return_value=DiffResult(compiled=False, failure_kind="unsupported"),
        ):
            with self.assertRaisesRegex(RuntimeError, "could not classify"):
                self.decomp.sim_ok(source, source)

    def test_contract_analyzer_rejects_partial_roster_without_summary(self):
        with tempfile.TemporaryDirectory() as tmp:
            work = Path(tmp)
            (work / "manifest.json").write_text(
                json.dumps([{"variant": "repro", "call_id": "one"}]),
                encoding="utf-8",
            )
            (work / "results.jsonl").write_text(
                json.dumps({"variant": "repro", "call_id": "one", "verdict": "PASS"})
                + "\n",
                encoding="utf-8",
            )
            with mock.patch.object(self.contract, "WORK", str(work)):
                with self.assertRaisesRegex(RuntimeError, "368 unique jobs"):
                    self.contract.analyze()
            self.assertFalse((work / "SUMMARY.json").exists())

    def test_realbug_loader_rejects_duplicate_normalized_ids(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "duplicates.jsonl"
            path.write_text(
                json.dumps({"id": "same"}) + "\n" + json.dumps({"id": "same"}) + "\n",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "duplicate normalized IDs"):
                self.realbug.load_records(path)

    def test_realbug_scorer_mcnemar_is_exact(self):
        self.assertEqual(self.general_score.mcnemar(6, 0), 0.03125)

    def test_realbug_scorer_rejects_endpoint_error_field_regardless_of_value(self):
        for endpoint_error in ("", False, None):
            with self.subTest(endpoint_error=endpoint_error), tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                input_path = root / "input.jsonl"
                input_path.write_text("fixture\n", encoding="utf-8")
                ids = [f"case_{index:03d}" for index in range(274)]
                rows = [
                    {
                        "candidate_id": cid,
                        "arm": arm,
                        "parsed_ok": False,
                        "finish_reason": "stop",
                        "response_sha256": "0" * 64,
                    }
                    for cid in ids
                    for arm in self.general_score.ARMS
                ]
                rows[0]["endpoint_error"] = endpoint_error
                rows_path = root / "generation_rows.json"
                rows_path.write_text(
                    json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
                )
                (root / "generation_manifest.json").write_text(
                    json.dumps(
                        {
                            "schema_version": 1,
                            "status": "complete",
                            "n_cases": 274,
                            "scheduled": 822,
                            "arms": list(self.general_score.ARMS),
                            "input_sha256": hashlib.sha256(
                                input_path.read_bytes()
                            ).hexdigest(),
                            "rows": "generation_rows.json",
                            "rows_sha256": hashlib.sha256(
                                rows_path.read_bytes()
                            ).hexdigest(),
                        }
                    ),
                    encoding="utf-8",
                )
                with self.assertRaisesRegex(RuntimeError, "endpoint failures"):
                    self.general_score.load_generation(root, input_path, ids)

    def test_realbug_scorer_rejects_nonboolean_parse_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            input_path = root / "input.jsonl"
            input_path.write_text("fixture\n", encoding="utf-8")
            ids = [f"case_{index:03d}" for index in range(274)]
            rows = [
                {
                    "candidate_id": cid,
                    "arm": arm,
                    "parsed_ok": False,
                    "finish_reason": "stop",
                    "response_sha256": "0" * 64,
                }
                for cid in ids
                for arm in self.general_score.ARMS
            ]
            rows[0].pop("parsed_ok")
            rows_path = root / "generation_rows.json"
            rows_path.write_text(
                json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8"
            )
            (root / "generation_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_version": 1,
                        "status": "complete",
                        "n_cases": 274,
                        "scheduled": 822,
                        "arms": list(self.general_score.ARMS),
                        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
                        "rows": "generation_rows.json",
                        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(RuntimeError, "parsed_ok is not Boolean"):
                self.general_score.load_generation(root, input_path, ids)

    def test_e4_rejects_malformed_verdict_ledger(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "verdicts.jsonl").write_text("not-json\n", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "malformed verdict row"):
                self.e4.load_verdicts(run)

    def test_e4_verdict_keys_must_equal_the_exact_frozen_cell_roster(self):
        cases = ["main_0001", "main_0002"]
        exact = {
            f"{arm}__{case_id}": "PROVED"
            for case_id in cases
            for arm in self.e4.ARMS
        }
        self.e4.validate_verdict_roster("fixture", exact, cases)

        missing = dict(exact)
        missing.pop("spec0_loc0__main_0001")
        with self.assertRaisesRegex(RuntimeError, "verdict key roster mismatch"):
            self.e4.validate_verdict_roster("fixture", missing, cases)

        extra = {**exact, "spec0_loc0__main_9999": "PROVED"}
        with self.assertRaisesRegex(RuntimeError, "verdict key roster mismatch"):
            self.e4.validate_verdict_roster("fixture", extra, cases)

        swapped = dict(exact)
        swapped.pop("spec0_loc0__main_0001")
        swapped["main_0001__spec0_loc0"] = "PROVED"
        with self.assertRaisesRegex(RuntimeError, "verdict key roster mismatch"):
            self.e4.validate_verdict_roster("fixture", swapped, cases)

    @staticmethod
    def _write_e4_cell(root, rel, case_id, arm, candidate="candidate-v1"):
        work = Path(rel) / "work" / f"{arm}__{case_id}"
        directory = root / work
        directory.mkdir(parents=True)
        (directory / "golden.sv").write_text("golden-v1", encoding="utf-8")
        (directory / "candidate.sv").write_text(candidate, encoding="utf-8")
        return {"case_id": case_id, "arm": arm, "work": work.as_posix()}

    def test_e4_work_binding_is_canonical_unique_and_content_auditable(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            rel = "generated/fixture"
            first = self._write_e4_cell(root, rel, "main_0001", "spec0_loc0")
            second = self._write_e4_cell(root, rel, "main_0002", "spec1_loc1")
            provenance = self.e4.validate_work_bindings(
                rel, [first, second], root=root
            )
            first_identity = provenance["spec0_loc0__main_0001"]
            self.assertEqual(first_identity["work"], first["work"])
            self.assertEqual(len(first_identity["candidate_sha256"]), 64)
            self.assertEqual(len(first_identity["golden_sha256"]), 64)

            with self.assertRaisesRegex(RuntimeError, "duplicate work path"):
                self.e4.validate_work_bindings(rel, [first, dict(first)], root=root)

            swapped = dict(first)
            swapped["work"] = second["work"]
            with self.assertRaisesRegex(RuntimeError, "not canonically bound"):
                self.e4.validate_work_bindings(rel, [swapped], root=root)

            traversal = dict(first)
            traversal["work"] = "../outside/spec0_loc0__main_0001"
            with self.assertRaisesRegex(RuntimeError, "not canonically bound"):
                self.e4.validate_work_bindings(rel, [traversal], root=root)

            declared_drift = dict(first)
            declared_drift["candidate_sha256"] = "0" * 64
            with self.assertRaisesRegex(RuntimeError, "candidate SHA256 mismatch"):
                self.e4.validate_work_bindings(
                    rel, [declared_drift], root=root
                )

            candidate_path = root / first["work"] / "candidate.sv"
            candidate_path.write_text("candidate-v2", encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "changed after provenance"):
                self.e4.tier_s(
                    root / first["work"],
                    expected_candidate_sha256=first_identity["candidate_sha256"],
                    expected_golden_sha256=first_identity["golden_sha256"],
                )

    def test_e4_frozen_public_work_paths_match_the_strict_binding_contract(self):
        public_e4 = json.loads(
            (ROOT / "artifacts/public/primary/analysis/e4_tier_s.json").read_text()
        )
        observed = []
        for row in public_e4["rows"]:
            expected = (
                Path(row["run"])
                / "work"
                / f"{row['arm']}__{row['case_id']}"
            ).as_posix()
            self.assertEqual(row["work"], expected)
            observed.append(row["work"])
        self.assertEqual(len(observed), 1088)
        self.assertEqual(len(set(observed)), 1088)


if __name__ == "__main__":
    unittest.main()
