from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class StandaloneExhibitProducerTests(unittest.TestCase):
    def run_command(
        self,
        command: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            command,
            cwd=cwd,
            env=environment,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            timeout=120,
            check=False,
        )

    def run_checked(
        self,
        command: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
    ) -> None:
        completed = self.run_command(
            command,
            cwd=cwd,
            environment=environment,
        )
        self.assertEqual(
            completed.returncode,
            0,
            msg=f"command failed: {' '.join(command)}\n{completed.stdout}",
        )

    def run_rejected(
        self,
        command: list[str],
        *,
        cwd: Path,
        environment: dict[str, str],
    ) -> None:
        completed = self.run_command(
            command,
            cwd=cwd,
            environment=environment,
        )
        self.assertNotEqual(
            completed.returncode,
            0,
            msg=f"invalid evidence was accepted: {' '.join(command)}\n{completed.stdout}",
        )

    def test_fresh_export_rebuilds_all_four_fragments_from_public_evidence(self) -> None:
        with tempfile.TemporaryDirectory(prefix="e6-standalone-test-") as temporary:
            temporary_root = Path(temporary)
            release = temporary_root / "release"
            inherited = ("PATH", "SYSTEMROOT", "WINDIR", "TMPDIR", "TMP", "TEMP")
            environment = {
                name: os.environ[name]
                for name in inherited
                if name in os.environ
            }
            environment.update(
                {
                    "PYTHONDONTWRITEBYTECODE": "1",
                    "PYTHONHASHSEED": "0",
                    "PYTHONUTF8": "1",
                }
            )
            for name in ("RTLREPAIR_ROOT", "RTLREPAIR_DATA", "RTLREPAIR_OUT"):
                environment.pop(name, None)
            self.run_checked(
                [
                    sys.executable,
                    str(ROOT / "tools" / "export_standalone.py"),
                    "--root",
                    str(ROOT),
                    "--output",
                    str(release),
                ],
                cwd=ROOT,
                environment=environment,
            )
            self.assertFalse((release / "generated").exists())

            output_dir = temporary_root / "fragments"
            self.run_checked(
                [
                    sys.executable,
                    "code/scripts/e6_curve_exhibits.py",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=release,
                environment=environment,
            )
            self.run_checked(
                [
                    sys.executable,
                    "code/scripts/e6_table6_realbug.py",
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=release,
                environment=environment,
            )
            self.run_checked(
                [
                    sys.executable,
                    "code/scripts/ws_i_prompts_appendix.py",
                    "--output",
                    str(output_dir / "app_prompts.tex"),
                ],
                cwd=release,
                environment=environment,
            )

            for name in (
                "fig_curve_body.tex",
                "tab_curve.tex",
                "tab_multimodel.tex",
                "app_prompts.tex",
            ):
                self.assertEqual(
                    (output_dir / name).read_bytes(),
                    (release / "paper" / name).read_bytes(),
                    msg=f"regenerated fragment differs from frozen paper input: {name}",
                )

            frozen_inputs = {
                name: (release / "paper" / name).read_bytes()
                for name in (
                    "fig_curve_body.tex",
                    "tab_curve.tex",
                    "tab_multimodel.tex",
                    "app_prompts.tex",
                )
            }
            for command in (
                [sys.executable, "code/scripts/e6_curve_exhibits.py"],
                [sys.executable, "code/scripts/e6_table6_realbug.py"],
                [sys.executable, "code/scripts/ws_i_prompts_appendix.py"],
                [sys.executable, "code/scripts/ws_i_prompts_appendix.py", "--check"],
                [sys.executable, "code/scripts/output_budget_audit.py"],
            ):
                self.run_checked(command, cwd=release, environment=environment)
            for name, payload in frozen_inputs.items():
                self.assertEqual(
                    (release / "paper" / name).read_bytes(), payload,
                    msg=f"ordinary producer invocation overwrote frozen input: {name}",
                )
            self.assertTrue((release / "build/paper-inputs/fig_curve_body.tex").is_file())
            self.assertTrue((release / "build/paper-inputs/tab_curve.tex").is_file())
            self.assertTrue((release / "build/paper-inputs/tab_multimodel.tex").is_file())
            self.assertTrue((release / "build/paper-inputs/app_prompts.tex").is_file())
            self.assertTrue((release / "build/analysis/output_budget_confound.json").is_file())

            frozen_realbugs = release / "data/repairbench_realbugs.jsonl"
            frozen_realbugs_bytes = frozen_realbugs.read_bytes()
            self.run_rejected(
                [sys.executable, "code/datagen/build_realbug_repairset.py"],
                cwd=release,
                environment=environment,
            )
            self.run_rejected(
                [
                    sys.executable,
                    "code/datagen/build_realbug_repairset.py",
                    "--output",
                    str(frozen_realbugs),
                    "--overwrite",
                ],
                cwd=release,
                environment=environment,
            )
            self.assertEqual(frozen_realbugs.read_bytes(), frozen_realbugs_bytes)
            self.assertFalse(
                (release / "generated/realbugs/repairbench_realbugs.rebuilt.jsonl").exists()
            )

            evidence = release / "artifacts" / "public" / "realbugs"
            missing_evidence = temporary_root / "missing-evidence"
            shutil.copytree(evidence, missing_evidence)
            (missing_evidence / "gpt41" / "score.json").unlink()
            missing_output = temporary_root / "missing-output"
            self.run_rejected(
                [
                    sys.executable,
                    "code/scripts/e6_table6_realbug.py",
                    "--evidence-dir",
                    str(missing_evidence),
                    "--output-dir",
                    str(missing_output),
                ],
                cwd=release,
                environment=environment,
            )
            self.assertFalse((missing_output / "tab_multimodel.tex").exists())

            mutations = (
                ("denominator", ("n",), 273),
                ("rate", ("nospec", "pct"), 101.0),
                ("p_value", ("regen_vs_repair", "mcnemar_p"), 1.5),
            )
            for name, key_path, replacement in mutations:
                with self.subTest(tamper=name):
                    tampered_evidence = temporary_root / f"tampered-{name}"
                    shutil.copytree(evidence, tampered_evidence)
                    score_path = tampered_evidence / "gpt55" / "score.json"
                    document = json.loads(score_path.read_text(encoding="utf-8"))
                    target = document
                    for key in key_path[:-1]:
                        target = target[key]
                    target[key_path[-1]] = replacement
                    score_path.write_text(
                        json.dumps(document, indent=2, sort_keys=True) + "\n",
                        encoding="utf-8",
                    )
                    tampered_output = temporary_root / f"tampered-output-{name}"
                    table_path = tampered_output / "tab_multimodel.tex"
                    if name == "p_value":
                        tampered_output.mkdir()
                        table_path.write_bytes(b"existing-good-table\n")
                    self.run_rejected(
                        [
                            sys.executable,
                            "code/scripts/e6_table6_realbug.py",
                            "--evidence-dir",
                            str(tampered_evidence),
                            "--output-dir",
                            str(tampered_output),
                        ],
                        cwd=release,
                        environment=environment,
                    )
                    if name == "p_value":
                        self.assertEqual(table_path.read_bytes(), b"existing-good-table\n")
                    else:
                        self.assertFalse(table_path.exists())


if __name__ == "__main__":
    unittest.main()
