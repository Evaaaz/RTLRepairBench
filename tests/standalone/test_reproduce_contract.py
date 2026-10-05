from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "reproduce.py"
SPEC = importlib.util.spec_from_file_location("reproduce_contract", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {MODULE_PATH}")
REPRODUCE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPRODUCE)
EXPECTED = ROOT / "artifacts/public/primary/analysis/vericodegen_full85_stats.json"


class AnalysisVerificationContractTests(unittest.TestCase):
    def frozen_analysis(self):
        return REPRODUCE._without_export_signatures(REPRODUCE.read_json(EXPECTED))

    def test_rejects_p_ci_and_decision_tampering(self) -> None:
        mutations = (
            ("p_value_holm", 999.0),
            ("ci95", [-0.5, 0.5]),
            ("material_improvement", True),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                actual = copy.deepcopy(self.frozen_analysis())
                actual["contrasts"]["what_full_spec"][field] = value
                with self.assertRaises(RuntimeError):
                    REPRODUCE.verify_analysis(actual, EXPECTED)

    def test_accepts_only_bounded_cross_runtime_float_drift(self) -> None:
        actual = copy.deepcopy(self.frozen_analysis())
        contrast = actual["contrasts"]["what_full_spec"]
        contrast["p_value_holm"] += 0.004
        contrast["p_value_unadjusted"] += 0.004
        contrast["risk_difference"] += 1e-14
        contrast["ci95"][0] += 1e-14
        audit = REPRODUCE.verify_analysis(actual, EXPECTED)
        self.assertFalse(audit["byte_exact"])

    def test_rejects_invalid_probability_interval_and_effect_domains(self) -> None:
        mutations = (
            ("p_value_holm", 1.009),
            ("ci95", [0.2, -0.2]),
            ("risk_difference", 1.01),
        )
        for field, value in mutations:
            with self.subTest(field=field):
                actual = copy.deepcopy(self.frozen_analysis())
                actual["contrasts"]["where_oracle_location"][field] = value
                with self.assertRaises(RuntimeError):
                    REPRODUCE.verify_analysis(actual, EXPECTED)

    def test_failed_reproduction_invalidates_stale_success_atomically(self) -> None:
        with tempfile.TemporaryDirectory(prefix="reproduce-transaction-") as temp:
            output = Path(temp) / "result"
            output.mkdir()
            REPRODUCE.write_json(
                output / "reproduction_report.json",
                {"schema_version": 1, "status": "verified", "old": True},
            )
            with mock.patch.object(
                REPRODUCE,
                "verify_environment_lock",
                side_effect=RuntimeError("forced verification failure"),
            ):
                with self.assertRaises(RuntimeError):
                    REPRODUCE.reproduce(output)
            report = REPRODUCE.read_json(output / "reproduction_report.json")
            self.assertEqual(report["status"], "failed")
            self.assertNotIn("old", report)
            self.assertFalse((output / "primary_stats.json").exists())


if __name__ == "__main__":
    unittest.main()
