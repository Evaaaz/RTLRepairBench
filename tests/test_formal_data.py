from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from benchmarks.formal_data import (
    CanonicalDataError,
    apply_unified_diff,
    assert_seed_consistency,
    load_semantic_cases,
    write_canonical_dataset,
)


class UnifiedDiffTest(unittest.TestCase):
    def test_applies_multiple_hunks_strictly(self):
        source = "a\nb\nc\nd\ne\n"
        patch = """--- a/x
+++ b/x
@@ -1,2 +1,2 @@
 a
-b
+B
@@ -4,2 +4,3 @@
 d
+D2
 e
"""
        self.assertEqual(apply_unified_diff(source, patch), "a\nB\nc\nd\nD2\ne\n")

    def test_rejects_context_drift(self):
        patch = "@@ -1 +1 @@\n-wrong\n+right\n"
        with self.assertRaisesRegex(CanonicalDataError, "source mismatch"):
            apply_unified_diff("actual\n", patch)


class CanonicalDatasetTest(unittest.TestCase):
    def test_real_sem85_is_85_cases_52_consistent_designs(self):
        cases = load_semantic_cases()
        self.assertEqual(len(cases), 85)
        self.assertEqual(len(assert_seed_consistency(cases)), 52)

    def test_manifest_supports_paths_outside_repository(self):
        row = {
            "id": "case-0",
            "seed_id": "Tiny",
            "mutation": "op_swap",
            "messages": [
                {"role": "system", "content": "repair"},
                {
                    "role": "user",
                    "content": (
                        "Broken RTL (`Tiny.sv`):\n```systemverilog\n"
                        "module Tiny(input a, output y);\nassign y = ~a;\nendmodule\n```"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        "--- a/Tiny.sv\n+++ b/Tiny.sv\n@@ -1,3 +1,3 @@\n"
                        " module Tiny(input a, output y);\n-assign y = ~a;\n"
                        "+assign y = a;\n endmodule\n"
                    ),
                },
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            benchmark = tmp_path / "bench.jsonl"
            benchmark.write_text(json.dumps(row) + "\n", encoding="utf-8")
            output = tmp_path / "goldens"
            manifest_path = tmp_path / "manifest.json"
            manifest = write_canonical_dataset(benchmark, output, manifest_path)
            self.assertEqual(manifest["case_count"], 1)
            self.assertEqual(manifest["design_count"], 1)
            self.assertTrue(Path(manifest["source_path"]).is_absolute())
            self.assertEqual(
                (output / "Tiny.sv").read_text(encoding="utf-8"),
                "module Tiny(input a, output y);\nassign y = a;\nendmodule\n",
            )


if __name__ == "__main__":
    unittest.main()
