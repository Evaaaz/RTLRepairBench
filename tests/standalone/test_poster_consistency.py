from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


class PosterConsistencyTests(unittest.TestCase):
    def test_repairbench_and_mined_cohort_counts(self) -> None:
        heldout = load_jsonl(ROOT / "data" / "repairbench_heldout.jsonl")
        lint = load_jsonl(ROOT / "data" / "repairbench_lint.jsonl")
        semantic = load_jsonl(ROOT / "data" / "repairbench_sem85.jsonl")
        mined = load_jsonl(ROOT / "data" / "repairbench_realbugs.jsonl")

        self.assertEqual(len(heldout), 459)
        self.assertEqual(len(lint), 374)
        self.assertEqual(len(semantic), 85)
        self.assertEqual(Counter(row["bucket"] for row in heldout), {"lint": 374, "semantic": 85})
        self.assertEqual(len(mined), 274)
        self.assertEqual(len({row["task"] for row in mined}), 112)

    def test_oracle_table_and_attribution_match_frozen_evidence(self) -> None:
        report = json.loads(
            (ROOT / "artifacts/public/capability/frontier_oracle_flip.json").read_text(
                encoding="utf-8"
            )
        )
        self.assertEqual(report["n_cases"], 85)
        self.assertEqual(report["n_definitive_both_arms"], 79)
        self.assertEqual(report["arms"]["diag"]["tier_s_pct"], 75.9)
        self.assertEqual(report["arms"]["spec"]["tier_s_pct"], 64.6)
        self.assertEqual(report["arms"]["diag"]["tier_f_pct"], 89.9)
        self.assertEqual(report["arms"]["spec"]["tier_f_pct"], 98.7)
        self.assertEqual(report["arms"]["diag"]["tier_s_hits"], 60)
        self.assertEqual(report["arms"]["spec"]["tier_s_hits"], 51)
        self.assertEqual(report["arms"]["diag"]["tier_f_hits"], 71)
        self.assertEqual(report["arms"]["spec"]["tier_f_hits"], 78)
        self.assertEqual(report["tier_s_contrast"]["spec_minus_diag_pp"], -11.4)
        self.assertEqual(report["tier_f_contrast"]["spec_minus_diag_pp"], 8.9)
        rescored = report["contract_rescores"]
        self.assertEqual(
            (
                rescored["tier_s_cycle1"]["arms"]["diag"]["hits"],
                rescored["tier_s_cycle1"]["arms"]["spec"]["hits"],
                rescored["tier_s_cycle1"]["paired"]["spec_minus_diag_pp"],
            ),
            (72, 78, 7.6),
        )
        self.assertEqual(
            (
                rescored["tier_s_reset_aware"]["arms"]["diag"]["hits"],
                rescored["tier_s_reset_aware"]["arms"]["spec"]["hits"],
                rescored["tier_s_reset_aware"]["paired"]["spec_minus_diag_pp"],
            ),
            (71, 78, 8.9),
        )
        self.assertEqual(
            report["contract_rescore_checks"]["reset_aware_vs_tier_f"],
            {"matching_cells": 158, "total_cells": 158},
        )
        self.assertEqual(
            report["contract_rescore_checks"]["cycle1_vs_cycle0"],
            {
                "fail_to_pass": {"diag": 12, "spec": 27},
                "pass_to_fail": {"diag": 0, "spec": 0},
            },
        )
        self.assertEqual(report["reset_contract_attribution"]["diag"], {
            "cleared_by_reset_aware_tb": 12,
            "false_divergences": 12,
        })
        self.assertEqual(report["reset_contract_attribution"]["spec"], {
            "cleared_by_reset_aware_tb": 27,
            "false_divergences": 30,
        })

        paper = (ROOT / "paper" / "paper.tex").read_text(encoding="utf-8")
        for value in (
            "60/79 (75.9\\%)",
            "51/79 (64.6\\%)",
            "72/79 (91.1\\%)",
            "78/79 (98.7\\%)",
            "71/79 (89.9\\%)",
            "$-11.4$ pp",
            "$+7.6$ pp",
            "$+8.9$ pp",
        ):
            self.assertIn(value, paper)
        self.assertIn("reset-aware initialization", paper)
        self.assertIn("does not make bounded simulation equivalent to formal proof", paper)
        self.assertIn("27/30 in the two arms, respectively", paper)
        self.assertNotIn("removed each of these false divergences", paper)

    def test_cycle1_panel_matches_frozen_evidence(self) -> None:
        report = json.loads(
            (ROOT / "artifacts/public/realbugs/cycle1_rescore.json").read_text(
                encoding="utf-8"
            )
        )
        expected = {
            "gpt55": (260, 57.3, 81.5, 76.9, -4.6),
            "claude_opus_4_8": (271, 34.3, 71.6, 69.0, -2.6),
            "gpt54mini": (274, 29.6, 51.5, 56.9, 5.5),
            "gpt41": (271, 27.3, 46.5, 38.0, -8.5),
            "gpt4omini": (274, 19.0, 39.1, 41.2, 2.2),
            "claude_opus_4_6": (273, 9.5, 51.6, 70.0, 18.3),
        }
        for model, values in expected.items():
            cycle1 = report["models"][model]["cycle1"]
            observed = (
                cycle1["n"],
                cycle1["pct"]["nospec"],
                cycle1["pct"]["spec"],
                cycle1["pct"]["speconly"],
                cycle1["speconly_minus_spec_pp"],
            )
            self.assertEqual(observed, values)

    def test_current_submission_points_only_to_the_poster_source(self) -> None:
        submission = json.loads(
            (ROOT / "configs" / "submission.json").read_text(encoding="utf-8")
        )
        self.assertEqual(submission["paper_source"], "paper/paper.tex")
        self.assertEqual(submission["venue"]["main_text_min_pages"], 3)
        self.assertEqual(submission["venue"]["main_text_max_pages"], 4)
        self.assertTrue(submission["venue"]["double_blind"])
        camera_ready = submission["camera_ready"]
        self.assertFalse(camera_ready["checklist_included"])
        self.assertEqual(camera_ready["public_artifact"], "https://github.com/Evaaaz/RTLRepairBench")
        paper = (ROOT / "paper" / "paper.tex").read_text(encoding="utf-8")
        self.assertNotIn("\\input{checklist}", paper)
        self.assertIn(camera_ready["public_artifact"], paper)
        self.assertFalse((ROOT / "paper" / "paper_poster_sketch.tex").exists())

    def test_released_training_file_matches_its_documented_composition(self) -> None:
        rows = load_jsonl(ROOT / "data" / "rtlrepairdataset_train.jsonl")
        self.assertEqual(len(rows), 5775)
        lint = [r for r in rows if "fails Verilator lint" in r["messages"][0]["content"]]
        self.assertEqual(len(lint), 5394)
        self.assertEqual(len(rows) - len(lint), 381)
        self.assertEqual({r["source"] for r in rows}, {"repair"})


if __name__ == "__main__":
    unittest.main()
