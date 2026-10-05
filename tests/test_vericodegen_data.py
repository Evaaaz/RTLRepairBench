from __future__ import annotations

import copy
import json
from pathlib import Path
import tempfile
import unittest

from benchmarks.vericodegen_data import (
    build_clean75_manifest,
    build_main85_inputs,
    build_shift50_inputs,
    build_shift_manifest,
    build_redaction_template,
    DEFAULT_MAIN85_INPUT_MANIFEST,
    DEFAULT_REDACTION_TEMPLATE,
    DEFAULT_SEM85,
    freeze_redaction_annotations,
    freeze_redaction_drop_no_annotators,
    mutation_target_from_diff,
    redacted_spec_overlay,
    select_shift_cases,
    verify_manifest,
)


def _shift_row(index: int, source: str, sequential: bool, task: str | None = None):
    return {
        "id": f"failure-{index:03d}",
        "task": task or f"task-{index:03d}",
        "source_model": source,
        "taxonomy": {"sequential": sequential},
    }


class ShiftSamplerTests(unittest.TestCase):
    def test_requested_counts_uniqueness_and_input_order_invariance(self):
        rows = []
        index = 0
        for source, sequential, count in (
            ("base", False, 30),
            ("base", True, 30),
            ("tuned", False, 10),
            ("tuned", True, 10),
        ):
            for _ in range(count):
                rows.append(_shift_row(index, source, sequential))
                index += 1
        first = select_shift_cases(rows)
        second = select_shift_cases(list(reversed(rows)))
        self.assertEqual(
            [row["record_id"] for row in first],
            [row["record_id"] for row in second],
        )
        self.assertEqual(len(first), 50)
        self.assertEqual(len({row["task"] for row in first}), 50)
        requested = {}
        for row in first:
            key = (row["requested_source_group"], row["requested_timing"])
            requested[key] = requested.get(key, 0) + 1
            self.assertEqual(row["selection_stage"], "requested_stratum")
        self.assertEqual(
            requested,
            {
                ("base", "combinational"): 20,
                ("base", "sequential"): 20,
                ("adapter", "combinational"): 5,
                ("adapter", "sequential"): 5,
            },
        )

    def test_fallback_uses_same_source_other_timing_before_global(self):
        rows = [
            _shift_row(0, "tuned", True),
            _shift_row(1, "tuned", False),
            _shift_row(2, "tuned", False),
            _shift_row(3, "base", True),
        ]
        selected = select_shift_cases(
            rows,
            quotas={("adapter", "sequential"): 3},
        )
        self.assertEqual(
            [row["selection_stage"] for row in selected].count("requested_stratum"), 1
        )
        self.assertEqual(
            [row["selection_stage"] for row in selected].count("same_source_other_timing"), 2
        )

    def test_string_taxonomy_is_parsed_strictly(self):
        rows = [
            {
                "id": "failure-0",
                "task": "task-0",
                "source_model": "base",
                "taxonomy": "{'sequential': False, 'code_lines': 3}",
            }
        ]
        selected = select_shift_cases(
            rows, quotas={("base", "combinational"): 1}
        )
        self.assertEqual(selected[0]["timing"], "combinational")

    def test_manifest_hash_detects_content_drift(self):
        rows = [_shift_row(0, "base", False)]
        manifest = build_shift_manifest(
            rows, quotas={("base", "combinational"): 1}
        )
        verify_manifest(manifest, verify_sources=False)
        manifest["case_count"] = 2
        with self.assertRaisesRegex(ValueError, "manifest SHA mismatch"):
            verify_manifest(manifest, verify_sources=False)


class Clean75Tests(unittest.TestCase):
    def test_allowlist_intersection_is_the_only_exclusion_authority(self):
        sem = [
            {"id": "case-0", "seed_id": "Prob000_ref"},
            {"id": "case-1", "seed_id": "Prob001_ref"},
            {"id": "case-2", "seed_id": "Prob002_ref"},
        ]
        clean = [{"id": "verilogeval:Prob000"}, {"id": "verilogeval:Prob002"}]
        manifest = build_clean75_manifest(
            sem, clean, expected_full=3, expected_clean=2
        )
        self.assertEqual(
            [item["case_id"] for item in manifest["cases"]], ["case-0", "case-2"]
        )
        self.assertEqual(
            [item["case_id"] for item in manifest["excluded"]], ["case-1"]
        )


class MaterializedInputTests(unittest.TestCase):
    @staticmethod
    def _semantic_case(index: int):
        seed = f"Prob{index:03d}_ref"
        broken = (
            "\n"
            f"module {seed};\n"
            "  wire x;\n"
            "  assign x = 1'b0;\n"
            "endmodule\n"
        )
        diff = (
            f"--- a/{seed}.sv\n"
            f"+++ b/{seed}.sv\n"
            "@@ -1,5 +1,5 @@\n"
            " \n"
            f" module {seed};\n"
            "   wire x;\n"
            "-  assign x = 1'b0;\n"
            "+  assign x = 1'b1;\n"
            " endmodule\n"
        )
        return {
            "id": f"case-{index:03d}",
            "seed_id": seed,
            "mutation": "flip_literal",
            "messages": [
                {"role": "system", "content": "repair"},
                {
                    "role": "user",
                    "content": (
                        f"Broken RTL (`{seed}.sv`):\n"
                        f"```systemverilog\n{broken}```\n"
                    ),
                },
                {"role": "assistant", "content": diff},
            ],
        }

    def test_main_inputs_anonymize_and_derive_location_without_golden(self):
        sem = [self._semantic_case(index) for index in range(85)]
        eval_rows = [
            {
                "id": f"verilogeval:Prob{index:03d}",
                "instruction": (
                    f"Implement a module named TopModule like Prob{index:03d}_ref "
                    "with output low."
                ),
            }
            for index in range(85)
        ]
        clean_rows = eval_rows[:75]
        clean = build_clean75_manifest(sem, clean_rows)
        materialized = build_main85_inputs(sem, eval_rows, clean)
        self.assertEqual(len(materialized), 85)
        first = materialized[0]
        self.assertEqual(first["model_input"]["oracle_location"]["line_number"], 3)
        self.assertEqual(first["model_input"]["expected_module_name"], "TopModule")
        self.assertNotIn("Prob000_ref", str(first["model_input"]))
        self.assertIn("module named TopModule", first["model_input"]["full_spec"])
        self.assertNotIn("module TopModule TopModule", first["model_input"]["full_spec"])
        self.assertNotIn("golden", str(first).lower())
        self.assertEqual(sum(row["internal_metadata"]["clean75"] for row in materialized), 75)

    def test_shift_inputs_resolve_frozen_record_sha_and_drop_golden(self):
        rows = []
        for index in range(50):
            row = _shift_row(index, "base", False)
            row.update(
                {
                    "broken_rtl": f"module task_{index:03d}; endmodule",
                    "spec": f"Implement task-{index:03d}.",
                    "golden_rtl": "SHOULD NEVER MATERIALIZE",
                }
            )
            rows.append(row)
        selection = build_shift_manifest(
            rows, quotas={("base", "combinational"): 50}
        )
        materialized = build_shift50_inputs(rows, selection)
        self.assertEqual(len(materialized), 50)
        self.assertTrue(
            all(row["model_input"]["oracle_location"] is None for row in materialized)
        )
        self.assertNotIn("golden", str(materialized).lower())
        drifted = copy.deepcopy(rows)
        selected_id = selection["cases"][0]["record_id"]
        next(row for row in drifted if row["id"] == selected_id)["broken_rtl"] += " // drift"
        with self.assertRaisesRegex(ValueError, "drifted"):
            build_shift50_inputs(drifted, selection)


def _complete_annotation(annotation, *, annotator: str, label: str = "explicit"):
    annotation.update(
        {
            "annotator_id": annotator,
            "intent_label": label,
            "redactable_without_hint": True,
            "minimal_removed_clause": " Target clause." if label != "absent" else None,
            "redacted_spec": "Full spec." if label != "absent" else "Full spec. Target clause.",
            "notes": "",
        }
    )


class RedactionTests(unittest.TestCase):
    def _template(self, count: int = 5):
        sem = [MaterializedInputTests._semantic_case(index) for index in range(count)]
        eval_rows = [
            {
                "id": f"verilogeval:Prob{index:03d}",
                "instruction": "Full spec. Target clause.",
            }
            for index in range(count)
        ]
        return build_redaction_template(sem, eval_rows)

    def test_template_contains_sha_pinned_human_only_mutation_target(self):
        template = self._template()
        target = template["cases"][0]["mutation_target"]
        self.assertEqual(target["broken_line"], "  assign x = 1'b0;")
        self.assertEqual(target["canonical_line"], "  assign x = 1'b1;")
        self.assertEqual(target["broken_line_number"], 4)
        self.assertEqual(target["canonical_line_number"], 4)
        self.assertNotIn("mutation_target", str(template["cases"][0]["annotations"]))

    def test_mutation_target_requires_exactly_one_committed_replacement(self):
        case = MaterializedInputTests._semantic_case(0)
        broken = next(
            message["content"]
            for message in case["messages"]
            if message["role"] == "user"
        )
        rtl = broken.split("```systemverilog\n", 1)[1].split("```", 1)[0]
        diff = next(
            message["content"]
            for message in case["messages"]
            if message["role"] == "assistant"
        )
        expanded_diff = diff.replace("@@ -1,5 +1,5 @@", "@@ -1,5 +1,6 @@")
        with self.assertRaisesRegex(ValueError, "exactly one added"):
            mutation_target_from_diff(rtl, expanded_diff + "+  assign x = 1'b0;\n")

    def _complete(self, document):
        result = copy.deepcopy(document)
        for case in result["cases"]:
            _complete_annotation(case["annotations"]["annotator_a"], annotator="ann-a")
            _complete_annotation(case["annotations"]["annotator_b"], annotator="ann-b")
            _complete_annotation(case["adjudication"], annotator="adjudicator")
        return result

    def test_blank_template_cannot_be_frozen(self):
        with self.assertRaisesRegex(ValueError, "incomplete"):
            freeze_redaction_annotations(self._template(), expected_cases=5)

    def test_absent_behavior_may_keep_full_spec_unchanged(self):
        document = self._complete(self._template())
        for role in (
            document["cases"][0]["annotations"]["annotator_a"],
            document["cases"][0]["annotations"]["annotator_b"],
            document["cases"][0]["adjudication"],
        ):
            role.update(
                {
                    "intent_label": "absent",
                    "minimal_removed_clause": None,
                    "redacted_spec": "Full spec. Target clause.",
                }
            )
        frozen = freeze_redaction_annotations(document, expected_cases=5)
        self.assertEqual(frozen["arm_decision"], "RUN_REDACTED_ARM")
        self.assertIn("case-000", frozen["eligible_case_ids"])
        overlay = redacted_spec_overlay(frozen)
        self.assertEqual(overlay["case-000"], "Full spec. Target clause.")

    def test_freeze_rejects_rewrite_instead_of_exact_clause_deletion(self):
        document = self._complete(self._template())
        document["cases"][0]["adjudication"]["redacted_spec"] = "A rewritten spec."
        with self.assertRaisesRegex(ValueError, "exact clause deletion"):
            freeze_redaction_annotations(document, expected_cases=5)

    def test_absent_label_cannot_change_specification(self):
        document = self._complete(self._template())
        document["cases"][0]["adjudication"].update(
            {
                "intent_label": "absent",
                "minimal_removed_clause": None,
                "redacted_spec": "Full spec.",
            }
        )
        with self.assertRaisesRegex(ValueError, "absent but changed"):
            freeze_redaction_annotations(document, expected_cases=5)

    def test_more_than_twenty_percent_unredactable_drops_arm(self):
        document = self._complete(self._template())
        for case in document["cases"][:2]:
            for role in (
                case["annotations"]["annotator_a"],
                case["annotations"]["annotator_b"],
                case["adjudication"],
            ):
                role.update(
                    {
                        "redactable_without_hint": False,
                        "minimal_removed_clause": None,
                        "redacted_spec": None,
                    }
                )
        frozen = freeze_redaction_annotations(document, expected_cases=5)
        self.assertEqual(frozen["arm_decision"], "DROP_REDACTED_ARM")
        self.assertEqual(frozen["eligible_case_count"], 3)
        self.assertEqual(len(frozen["eligible_case_ids"]), 3)
        self.assertEqual(redacted_spec_overlay(frozen), {})

    def test_freeze_rejects_immutable_source_drift_from_template(self):
        template = self._template()
        document = self._complete(template)
        document["cases"][0]["seed_id"] = "different-seed"
        with self.assertRaisesRegex(ValueError, "immutable"):
            freeze_redaction_annotations(
                document,
                template_document=template,
                expected_cases=5,
            )

    def test_freeze_rejects_mutation_target_sha_drift_without_template(self):
        document = self._complete(self._template())
        document["cases"][0]["mutation_target"]["canonical_line"] = "  assign x = 1'bx;"
        with self.assertRaisesRegex(ValueError, "mutation_target SHA"):
            freeze_redaction_annotations(document, expected_cases=5)

    def test_no_annotator_drop_freezes_zero_cases_without_fabricating_labels(self):
        template = json.loads(DEFAULT_REDACTION_TEMPLATE.read_text())
        main_manifest = json.loads(DEFAULT_MAIN85_INPUT_MANIFEST.read_text())
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "never-created-ledger.jsonl"
            frozen = freeze_redaction_drop_no_annotators(
                template,
                main_manifest,
                template_path=DEFAULT_REDACTION_TEMPLATE,
                main_input_manifest_path=DEFAULT_MAIN85_INPUT_MANIFEST,
                benchmark_source_path=DEFAULT_SEM85,
                ledger_path=ledger,
            )
            self.assertEqual(frozen["schema_version"], 2)
            self.assertEqual(frozen["decision_basis"], "NO_INDEPENDENT_HUMAN_ANNOTATORS")
            self.assertFalse(frozen["model_outputs_seen_before_freeze"])
            self.assertFalse(frozen["annotations_performed"])
            self.assertEqual(frozen["case_count"], 85)
            self.assertEqual(frozen["eligible_case_count"], 0)
            self.assertEqual(frozen["eligible_case_ids"], [])
            self.assertEqual(frozen["cases"], [])
            self.assertEqual(redacted_spec_overlay(frozen), {})

    def test_no_annotator_drop_rejects_any_existing_ledger_or_output(self):
        template = json.loads(DEFAULT_REDACTION_TEMPLATE.read_text())
        main_manifest = json.loads(DEFAULT_MAIN85_INPUT_MANIFEST.read_text())
        with tempfile.TemporaryDirectory() as directory:
            ledger = Path(directory) / "events.jsonl"
            ledger.touch()
            with self.assertRaisesRegex(ValueError, "ledger/model output exists"):
                freeze_redaction_drop_no_annotators(
                    template,
                    main_manifest,
                    template_path=DEFAULT_REDACTION_TEMPLATE,
                    main_input_manifest_path=DEFAULT_MAIN85_INPUT_MANIFEST,
                    benchmark_source_path=DEFAULT_SEM85,
                    ledger_path=ledger,
                )


if __name__ == "__main__":
    unittest.main()
