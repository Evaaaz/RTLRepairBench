"""Verilator gate for the repair-dataset pipeline.

This is the quality guarantee for the lint-class bucket: we mutate clean code,
then *actually run Verilator* and keep a pair only if

    * the original source LINTS CLEAN, and
    * the broken variant FAILS lint,

capturing the real ``%Error``/gating-``%Warning`` text Verilator prints so the
training input contains a genuine tool message, never a hand-written one.

"Fails lint" follows the project's waiver policy (EXECUTION_PLAN.md §8 Verifier):
any ``%Error`` counts, and so do ``%Warning`` lines in the *gating* classes that
correspond to real bug families (WIDTH, IMPLICIT, CASEINCOMPLETE, LATCH, ...).
Cosmetic classes (DECLFILENAME, UNUSED, ...) are ignored so "fails lint" stays
meaningful rather than tripping on style noise.

Verilator must be on PATH; unlike backend/verifier.py there is no stub fallback
here — a missing tool means we cannot certify the dataset, which should be loud.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

# %Warning classes that gate "lint pass" because each maps to a real bug family
# we want the repair model to learn. Everything else is treated as cosmetic.
GATING_WARNINGS: frozenset[str] = frozenset(
    {
        "WIDTH",
        "WIDTHCONCAT",
        "WIDTHTRUNC",
        "WIDTHEXPAND",
        "IMPLICIT",
        "UNDRIVEN",
        "SELRANGE",
        "CASEINCOMPLETE",
        "CASEOVERLAP",
        "LATCH",
        "BLKSEQ",
        "PINMISSING",
        "MULTIDRIVEN",
    }
)

_ERROR_RE = re.compile(r"^%Error")
_WARNING_RE = re.compile(r"^%Warning-([A-Z0-9]+)")


class VerilatorUnavailableError(RuntimeError):
    """The lint gate cannot run because Verilator is unavailable."""


def require_verilator() -> None:
    if shutil.which("verilator") is None:
        raise VerilatorUnavailableError(
            "verilator not found on PATH; the repair gate cannot certify pairs "
            "without it (install Verilator or use the pinned verifier container)."
        )


@dataclass
class LintResult:
    """Outcome of linting one file set."""

    failed: bool
    diagnostics: list[str] = field(default_factory=list)  # gating lines only
    log: str = ""  # full combined stdout+stderr

    @property
    def first(self) -> str:
        return self.diagnostics[0] if self.diagnostics else ""


def _classify(combined: str) -> list[str]:
    """Pull the gating diagnostics (errors + gating warnings) out of the log."""
    diags: list[str] = []
    for raw in combined.splitlines():
        line = raw.strip()
        if _ERROR_RE.match(line):
            diags.append(line)
            continue
        m = _WARNING_RE.match(line)
        if m and m.group(1) in GATING_WARNINGS:
            diags.append(line)
    return diags


def lint(files: list[str]) -> LintResult:
    """Run ``verilator --lint-only`` over ``files`` and classify the result.

    ``files`` should be a self-contained set: a module that instantiates another
    must be passed together with its companion(s), or Verilator errors on the
    missing definition and the gate would wrongly reject the clean original.
    """
    require_verilator()
    cmd = ["verilator", "--lint-only", "-Wall", "-sv", *files]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
    combined = proc.stdout + ("\n" if proc.stdout and proc.stderr else "") + proc.stderr
    diags = _classify(combined)
    failed = bool(diags) or proc.returncode != 0
    return LintResult(failed=failed, diagnostics=diags, log=combined)


def lint_source(
    primary_name: str,
    primary_src: str,
    companions: dict[str, str] | None = None,
) -> LintResult:
    """Lint in-memory source by materializing it to a temp dir first.

    Files are written under their real basenames (so module/file names match and
    Verilator does not raise spurious DECLFILENAME noise). ``companions`` are
    extra modules the primary depends on, written clean alongside it.
    """
    with tempfile.TemporaryDirectory(prefix="repair_lint_") as tmp:
        tmp_path = Path(tmp)
        for name, text in (companions or {}).items():
            (tmp_path / name).write_text(text)
        primary_path = tmp_path / primary_name
        primary_path.write_text(primary_src)
        files = [str(primary_path)] + [
            str(tmp_path / n) for n in (companions or {})
        ]
        result = lint(files)
        # Scrub the temp dir prefix so diagnostics read `mac_cell.sv:52:..`,
        # not a throwaway absolute path that would leak into training data.
        prefix = str(tmp_path) + "/"
        result.diagnostics = [d.replace(prefix, "") for d in result.diagnostics]
        result.log = result.log.replace(prefix, "")
        return result
