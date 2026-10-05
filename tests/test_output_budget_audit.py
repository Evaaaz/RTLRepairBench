from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "code" / "scripts" / "output_budget_audit.py"
SPEC = importlib.util.spec_from_file_location("output_budget_audit", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover
    raise RuntimeError(f"cannot load {MODULE_PATH}")
AUDIT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(AUDIT)
LEDGER = ROOT / "artifacts/public/primary/ledger/public_events.jsonl"


class OutputBudgetAuditTests(unittest.TestCase):
    def rows(self):
        return AUDIT._load_jsonl(LEDGER)

    def test_reconstructs_paper_counts_and_sign_tests(self) -> None:
        result = AUDIT.analyze(self.rows())
        self.assertEqual(result["n_cases"], 85)
        self.assertEqual(result["arms"]["spec0_loc0"]["truncations"], 7)
        self.assertEqual(result["arms"]["spec1_loc1"]["truncations"], 2)
        self.assertEqual(result["baseline_to_joint_truncation_reduction"], 5)
        self.assertEqual(result["minimum_reasoning_tokens_in_truncated_cell"], 4093)
        tests = result["paired_reasoning_sign_tests_vs_spec0_loc0"]
        self.assertAlmostEqual(tests["spec0_loc1"]["two_sided_p"], 6.083249930070788e-08)
        self.assertAlmostEqual(tests["spec1_loc0"]["two_sided_p"], 0.00021680385841747665)
        self.assertAlmostEqual(tests["spec1_loc1"]["two_sided_p"], 3.761512780416608e-05)

    def test_rejects_missing_cell_and_invalid_token_accounting(self) -> None:
        rows = self.rows()
        missing = [
            row for row in rows
            if not (
                row.get("event") == "response"
                and row.get("mode") == "main"
                and row.get("case_id") == "main_0001"
                and row.get("arm") == "spec0_loc0"
            )
        ]
        with self.assertRaises(ValueError):
            AUDIT.analyze(missing)

        invalid = copy.deepcopy(rows)
        target = next(
            row for row in invalid
            if row.get("event") == "response" and row.get("mode") == "main"
        )
        target["usage"]["output_tokens_details"]["reasoning_tokens"] = 5000
        with self.assertRaises(ValueError):
            AUDIT.analyze(invalid)


if __name__ == "__main__":
    unittest.main()
