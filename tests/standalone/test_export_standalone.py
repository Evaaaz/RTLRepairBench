from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = ROOT / "tools" / "export_standalone.py"
SPEC = importlib.util.spec_from_file_location("export_standalone", MODULE_PATH)
if SPEC is None or SPEC.loader is None:  # pragma: no cover - import bootstrap guard
    raise RuntimeError(f"cannot load {MODULE_PATH}")
EXPORTER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORTER)


class StandaloneReleaseTests(unittest.TestCase):
    def test_globs_are_component_aware_and_recursive(self) -> None:
        self.assertTrue(
            EXPORTER.path_matches(
                "docs/history/migration/reports/R2d.md", "docs/history/**"
            )
        )
        self.assertTrue(
            EXPORTER.path_matches("tests/test_vericodegen_data.py", "**/*.py")
        )
        self.assertFalse(
            EXPORTER.path_matches(
                "code/benchmarks/results/batched/result.json",
                "code/benchmarks/results/*.json",
            )
        )

    def test_source_tree_passes_policy_and_paper_hash_checks(self) -> None:
        report = EXPORTER.check_source(ROOT)
        self.assertEqual(report["artifact_name"], "lint-vs-semantic-standalone")
        self.assertGreater(len(report["files"]), 20)
        self.assertGreater(len(report["paper_inputs"]), 1)
        self.assertRegex(report["tree_sha256"], r"^[0-9a-f]{64}$")
        self.assertIs(report["anonymous_manuscript"], False)
        EXPORTER.validate_camera_ready_manuscript(ROOT)

    def test_manuscript_author_block_is_exact_and_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paper = root / "paper"
            paper.mkdir()
            (paper / "paper.tex").write_text(
                r"\author{Named Author}" + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(EXPORTER.ReleaseError, "anonymous author block"):
                EXPORTER.validate_anonymous_manuscript(root)

    def test_manuscript_requires_workshop_style_and_completed_checklist(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paper = root / "paper"
            paper.mkdir()
            source = (
                r"\usepackage[dblblindworkshop,nonatbib]{neurips_2026}"
                "\n"
                r"\workshoptitle{AI for Chip Design}"
                "\n"
                r"\author{Anonymous Author(s)}"
                "\n"
                r"\input{checklist}"
                "\n"
            )
            (paper / "paper.tex").write_text(source, encoding="utf-8")
            (paper / "checklist.tex").write_text(
                r"\answerTODO{}" + "\n", encoding="utf-8"
            )
            with self.assertRaisesRegex(EXPORTER.ReleaseError, "unanswered TODO"):
                EXPORTER.validate_anonymous_manuscript(root)

            (paper / "checklist.tex").write_text(
                r"\answerYes{}" + "\n", encoding="utf-8"
            )
            EXPORTER.validate_anonymous_manuscript(root)

    def test_canonical_config_and_specialization_locations_are_unique(self) -> None:
        self.assertTrue((ROOT / "configs" / "vericodegen2026.json").is_file())
        self.assertFalse((ROOT / "code" / "configs").exists())
        self.assertFalse((ROOT / "code" / "benchmarks" / "results").exists())

        specialization = ROOT / "artifacts" / "public" / "specialization"
        # Upstream replaced the flat rb_<model>_<arm>.json exports with a per-run tree
        # keyed by model/version/signal, with index.json as the entry point. Assert the
        # tree and the index agree rather than pinning the retired filenames.
        index = json.loads((specialization / "index.json").read_text(encoding="utf-8"))
        runs = index["runs"]
        # The four reported models must each have both arms. The index may also carry
        # diagnostic arms that are not paper rows -- verireason-default is the wrong
        # chat template run kept as a control, parsed_rate 0.02 -- so require the
        # reported set as a subset rather than pinning the index exactly.
        self.assertLessEqual(
            {(model, signal)
             for model in ("base", "v5", "verireason", "vrqwen")
             for signal in ("diag", "diag_spec")},
            {(run["model"], run["signal"]) for run in runs},
        )
        for run in runs:
            result = specialization / run["model"] / run["version"] / run["signal"] / "result.json"
            self.assertTrue(result.is_file(), result)
        self.assertFalse(any(specialization.glob("rb_*.json")))
        self.assertTrue((specialization / "REPORT.txt").is_file())

    def test_allowlist_excludes_raw_history_cache_and_validation(self) -> None:
        lock = EXPORTER.load_lock(ROOT)
        selected = {
            EXPORTER.relative_posix(path, ROOT)
            for path in EXPORTER.selected_files(ROOT, lock)
        }
        forbidden_parts = {"__pycache__", "raw", "migration_reports"}
        for relative in selected:
            self.assertTrue(forbidden_parts.isdisjoint(Path(relative).parts), relative)
            self.assertFalse(relative.endswith((".pyc", ".pyo")), relative)
        self.assertNotIn("MIGRATION_MANIFEST.md", selected)
        self.assertNotIn("code/DEPRECATED_DUPLICATES.md", selected)
        self.assertNotIn("code/benchmarks/make_base_v5.py", selected)
        self.assertNotIn("code/benchmarks/make_comparison.py", selected)
        self.assertNotIn("code/scripts/ws_g_consistency.py", selected)
        self.assertNotIn(
            "artifacts/public/specialization/validation/semantic_fixture.json",
            selected,
        )
        self.assertIn("artifacts/public/specialization/REPORT.txt", selected)
        self.assertFalse(
            any(
                EXPORTER.path_matches("data/processed/unexpected.jsonl", pattern)
                for pattern in lock["include"]
            )
        )
        self.assertFalse(
            EXPORTER.is_excluded("data/processed/unexpected.jsonl", lock)
        )
        self.assertTrue(EXPORTER.is_excluded("code/datagen/out/private.jsonl", lock))

    def test_secret_absolute_path_and_reasoning_checks_fail_closed(self) -> None:
        secret = b'credential = "' + b"sk-" + (b"A" * 32) + b'"\n'
        with self.assertRaises(EXPORTER.ReleaseError):
            EXPORTER.scan_payload("config.json", secret)

        workstation_path = (
            b'root = "' + b"/" + b"Users" + b"/example/project/output.json" + b'"\n'
        )
        with self.assertRaises(EXPORTER.ReleaseError):
            EXPORTER.scan_payload("config.json", workstation_path)

        for relative, workstation_path in (
            (
                "artifacts/public/private.json",
                b'{"path":"/' + b'private/tmp/private/run.json"}',
            ),
            (
                "configs/private.json",
                b'{"path":"/' + b'var/folders/ab/private/run.json"}',
            ),
            (
                "data/private.csv",
                b"path\n/" + b"Volumes/private-disk/run.json\n",
            ),
            (
                "data/private.jsonl",
                b'{"path":"/' + b'root/private/run.json"}\n',
            ),
            (
                "artifacts/public/runner.json",
                b'{"path":"/' + b'github/workspace/private/run.json"}',
            ),
        ):
            with self.subTest(relative=relative), self.assertRaises(
                EXPORTER.ReleaseError
            ):
                EXPORTER.scan_payload(relative, workstation_path)

        synthetic_endpoint = b"https://" + b"192" + b".0.2.42/v1"
        with self.assertRaises(EXPORTER.ReleaseError):
            EXPORTER.scan_payload("code/deploy.py", synthetic_endpoint)

        reasoning = b'{"raw": "<' + b"think" + b'>private trace"}\n'
        with self.assertRaises(EXPORTER.ReleaseError):
            EXPORTER.scan_payload("artifacts/public/rows.jsonl", reasoning)
        with self.assertRaises(EXPORTER.ReleaseError):
            EXPORTER.scan_payload("artifacts/public/trace.txt", reasoning)
        with self.assertRaises(EXPORTER.ReleaseError):
            EXPORTER.scan_payload("paper/raw.log", reasoning)

        closing_reasoning = b'{"raw": "<' + b"/think" + b'>private trace"}\n'
        with self.assertRaises(EXPORTER.ReleaseError):
            EXPORTER.scan_payload(
                "artifacts/public/closing.jsonl", closing_reasoning
            )

        authorization = (
            b"Authorization" + b": " + b"Bearer " + b"long-secret-value-1234567890"
        )
        with self.assertRaises(EXPORTER.ReleaseError):
            EXPORTER.scan_payload("artifacts/public/header.txt", authorization)

        for key, value in (
            ("reasoning_content", "private chain of thought"),
            ("analysis", "private analysis"),
            ("scratchpad", "private notes"),
            ("raw_response", {"body": "provider payload"}),
        ):
            with self.subTest(key=key), self.assertRaises(EXPORTER.ReleaseError):
                EXPORTER.scan_payload(
                    "artifacts/public/private.json",
                    json.dumps({key: value}).encode("utf-8"),
                )

        for relative, payload in (
            (
                "configs/private.json",
                b'{"reasoning_content":"private chain of thought"}',
            ),
            (
                "configs/private.jsonl",
                b'{"raw_response":{"body":"provider payload"}}\n',
            ),
            (
                "configs/private.ndjson",
                b'{"scratchpad":"private notes"}\n',
            ),
            (
                "data/private.csv",
                b"case_id,reasoning_content,score\ncase-1,private trace,1\n",
            ),
            (
                "artifacts/public/private.tsv",
                b"case_id\traw_response\tscore\ncase-1\tprovider payload\t1\n",
            ),
        ):
            with self.subTest(relative=relative), self.assertRaises(
                EXPORTER.ReleaseError
            ):
                EXPORTER.scan_payload(relative, payload)

        # Implementation-owned JSON may document provider fields, but the
        # root scientific configs directory never receives this exception.
        EXPORTER.scan_payload(
            "code/benchmarks/configs/provider.json",
            b'{"raw_response":"implementation field mapping"}',
        )

        # Aggregate usage metadata is public evidence, not a reasoning trace.
        EXPORTER.scan_payload(
            "artifacts/public/usage.json",
            b'{"usage":{"reasoning_tokens":4093}}',
        )

        # Prompt-building source is allowed to describe the tag.  The ban is on
        # released result/data payloads, not on implementation source.
        EXPORTER.scan_payload("src/prompts.py", reasoning)

    def test_benchmark_defaults_resolve_inside_fresh_release(self) -> None:
        environment = {
            name: os.environ[name]
            for name in ("PATH", "SYSTEMROOT", "WINDIR")
            if name in os.environ
        }
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        environment["PYTHONPATH"] = os.pathsep.join(
            (str(ROOT / "code"), str(ROOT / "code" / "benchmarks"))
        )
        program = (
            "from pathlib import Path; import bench_common, stats; "
            "root=Path.cwd().resolve(); "
            "assert Path(bench_common.DEFAULT_TASKS).resolve() == "
            "root/'data'/'eval_tasks.jsonl'; "
            "assert len(bench_common.load_tasks()) == 155; "
            "assert Path(stats.DEFAULT_CLEAN).resolve() == "
            "root/'data'/'eval_tasks_clean.jsonl'; "
            "assert len(stats.load_clean_ids()) == 137"
        )
        completed = subprocess.run(
            [sys.executable, "-c", program],
            cwd=ROOT,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=30,
            check=False,
        )
        self.assertEqual(completed.returncode, 0, completed.stdout)

    def test_negative_fixture_exception_is_bound_to_exact_file_hash(self) -> None:
        lock = EXPORTER.load_lock(ROOT)
        relative = "tests/test_supplement_export.py"
        payload = (ROOT / relative).read_bytes()
        allowed = EXPORTER.scan_exception_detectors(relative, payload, lock)
        self.assertEqual(allowed, {"provider API key", "POSIX home path"})
        with self.assertRaises(EXPORTER.ReleaseError):
            EXPORTER.scan_exception_detectors(relative, payload + b"\n", lock)

    def test_export_round_trip_is_exact_and_contains_no_extra_files(self) -> None:
        with tempfile.TemporaryDirectory(prefix="standalone-export-test-") as temp:
            output = Path(temp) / "release"
            manifest = EXPORTER.export_tree(ROOT, output)
            checked = EXPORTER.check_export(output)
            self.assertEqual(manifest["tree_sha256"], checked["tree_sha256"])
            self.assertTrue((output / EXPORTER.EXPORT_MANIFEST).is_file())

            exported = {
                path.relative_to(output).as_posix()
                for path in output.rglob("*")
                if path.is_file()
            }
            expected = {entry["path"] for entry in manifest["files"]}
            expected.add(EXPORTER.EXPORT_MANIFEST)
            self.assertEqual(exported, expected)

    def test_excluded_runtime_outputs_do_not_invalidate_export(self) -> None:
        with tempfile.TemporaryDirectory(prefix="standalone-runtime-test-") as temp:
            output = Path(temp) / "release"
            EXPORTER.export_tree(ROOT, output)
            runtime = output / "build" / "reproduced" / "runtime.json"
            runtime.parent.mkdir(parents=True)
            runtime.write_text('{"generated": true}\n', encoding="utf-8")
            EXPORTER.check_export(output)

    def test_tampering_is_detected(self) -> None:
        with tempfile.TemporaryDirectory(prefix="standalone-tamper-test-") as temp:
            output = Path(temp) / "release"
            EXPORTER.export_tree(ROOT, output)
            target = output / "README.md"
            target.write_bytes(target.read_bytes() + b"\nchanged\n")
            with self.assertRaises(EXPORTER.ReleaseError):
                EXPORTER.check_export(output)

    def test_lock_is_json_and_contains_no_unpinned_paper_input(self) -> None:
        lock = json.loads((ROOT / EXPORTER.LOCK_NAME).read_text(encoding="utf-8"))
        for item in lock["paper_inputs"]:
            self.assertRegex(item["sha256"], r"^[0-9a-f]{64}$")
            self.assertTrue((ROOT / item["path"]).is_file())
        pinned = {item["path"] for item in lock["paper_inputs"]}
        prompt_chain = {
            "code/scripts/ws_i_prompts_appendix.py",
            "code/benchmarks/vericodegen_eval.py",
            "code/benchmarks/repair_agent.py",
            "code/scripts/realbug_frontier.py",
            "code/scripts/second_model_redact.py",
            "code/scripts/second_model_explore.py",
        }
        self.assertTrue(prompt_chain.issubset(set(lock["required_paths"])))
        self.assertTrue(prompt_chain.issubset(pinned))


if __name__ == "__main__":
    unittest.main()
