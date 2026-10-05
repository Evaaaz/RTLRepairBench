#!/usr/bin/env python3
"""Regenerate the multi-model real-bug table from public frozen evidence.

The default evidence directory is ``artifacts/public/realbugs`` in the
standalone package.  The generated ``tab_multimodel.tex`` defaults to
``build/paper-inputs/``; release maintainers may explicitly select ``paper/``
when promoting a reviewed snapshot.  No model-run tree is required.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


def project_root() -> Path:
    override = os.environ.get("RTLREPAIR_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


# tag -> (display label, is anchor).  The anchor bit remains metadata for the
# table contract even though sorting is determined by unaided capability.
LABELS = {
    "gpt55": ("gpt-5.5 (OpenAI, frozen anchor)", True),
    "claude_opus_4_8": ("Claude Opus 4.8 (Anthropic, anchor)", True),
    "gpt54mini": ("gpt-5.4-mini (OpenAI)", False),
    "gpt41": ("gpt-4.1 (OpenAI)", False),
    "gpt4omini": ("gpt-4o-mini (OpenAI)", False),
    "claude_opus_4_6": ("Claude Opus 4.6 (Anthropic)", False),
}

EXPECTED_CASES = 274
RealBugRow = tuple[float, str, float, float, float, float, float]
FROZEN_EVIDENCE_SHA256 = {
    ("gpt55", "score.json"): "54b1192a9033fb8965ab9a0533a30d3c28ba274a30c297d4fce8b7a45608735c",
    (
        "claude_opus_4_8",
        "score.json",
    ): "ae7d6e35bb84ba84903b97cc37cc3c8b5cfc641a8a444b8e4ae3c46a24f9f6b7",
    ("gpt54mini", "score.json"): "1d7b7a81ca9ff08e0936e964e6eaa95df8bc30c88add962c375efbb0116e5551",
    ("gpt41", "score.json"): "360f2d143e9c71caf0cf825f0bc3f8ec275824e5a8e4ff22a5b574781f863907",
    ("gpt4omini", "score.json"): "de4c61cc24b2f622e05df423b5c428ca09bea6e9151d212d3d24aa86cd73cdc8",
    (
        "claude_opus_4_6",
        "score.json",
    ): "71059970eba443dc9920f8bdb240512e7a0bf2b3714abc27ffc88413b52d13a4",
    (
        "claude_opus_4_6",
        "speconly_compare.json",
    ): "484e3831a425d30dc5f04592b078b7ec80084a992114a6faf84a574fd98c7e24",
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _verify_frozen_file(path: Path, tag: str, filename: str) -> None:
    expected = FROZEN_EVIDENCE_SHA256[(tag, filename)]
    try:
        actual = _sha256_file(path)
    except OSError as exc:
        raise ValueError(f"cannot read real-bug evidence {path}: {exc}") from exc
    if actual != expected:
        raise ValueError(
            f"real-bug evidence hash mismatch for {tag}/{filename}: "
            f"expected {expected}, got {actual}"
        )


def _load_object(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read real-bug evidence {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"real-bug evidence must be an object: {path}")
    return value


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _percentage(value: Any, label: str) -> float:
    result = _number(value, label)
    if not 0.0 <= result <= 100.0:
        raise ValueError(f"{label} must be between 0 and 100")
    return result


def _probability(value: Any, label: str) -> float:
    result = _number(value, label)
    if not 0.0 <= result <= 1.0:
        raise ValueError(f"{label} must be between 0 and 1")
    return result


def _case_count(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _paired_count(value: Any, label: str, n_cases: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    if not 0 <= value <= n_cases:
        raise ValueError(f"{label} must be between 0 and {n_cases}")
    return value


def read_score(evidence_dir: Path, tag: str) -> tuple[float, float, float, float, int]:
    score_path = evidence_dir / tag / "score.json"
    _verify_frozen_file(score_path, tag, "score.json")
    score = _load_object(score_path)
    if score.get("tag") != tag:
        raise ValueError(f"{score_path}: tag does not match directory name")
    n_cases = _case_count(score.get("n"), f"{score_path}: n")
    if n_cases != EXPECTED_CASES:
        raise ValueError(
            f"{score_path}: expected exactly {EXPECTED_CASES} cases, got {n_cases}"
        )

    if "regen_vs_repair" in score:
        percentages: list[float] = []
        for arm in ("nospec", "spec", "speconly"):
            value = score.get(arm)
            if not isinstance(value, dict):
                raise ValueError(f"{score_path}: missing {arm} arm")
            arm_n = _case_count(value.get("n"), f"{score_path}: {arm}.n")
            if arm_n != n_cases:
                raise ValueError(f"{score_path}: {arm}.n disagrees with n")
            recovered = value.get("recovered")
            if isinstance(recovered, bool) or not isinstance(recovered, int):
                raise ValueError(f"{score_path}: {arm}.recovered must be an integer")
            if not 0 <= recovered <= arm_n:
                raise ValueError(f"{score_path}: {arm}.recovered is out of range")
            percentage = _percentage(value.get("pct"), f"{score_path}: {arm}.pct")
            if round(100.0 * recovered / arm_n, 1) != percentage:
                raise ValueError(f"{score_path}: {arm}.pct disagrees with counts")
            percentages.append(percentage)
        comparison = score.get("regen_vs_repair")
        if not isinstance(comparison, dict):
            raise ValueError(f"{score_path}: malformed regen_vs_repair comparison")
        speconly_fixes = _paired_count(
            comparison.get("speconly_fixes_spec_misses"),
            f"{score_path}: regen_vs_repair.speconly_fixes_spec_misses",
            n_cases,
        )
        spec_fixes = _paired_count(
            comparison.get("spec_fixes_speconly_misses"),
            f"{score_path}: regen_vs_repair.spec_fixes_speconly_misses",
            n_cases,
        )
        if speconly_fixes + spec_fixes > n_cases:
            raise ValueError(f"{score_path}: regen_vs_repair counts exceed n")
        p_value = _probability(
            comparison.get("mcnemar_p"),
            f"{score_path}: regen_vs_repair.mcnemar_p",
        )
        intent = score.get("intent_effect")
        if not isinstance(intent, dict):
            raise ValueError(f"{score_path}: missing intent_effect comparison")
        intent_fixed = _paired_count(
            intent.get("fixed"),
            f"{score_path}: intent_effect.fixed",
            n_cases,
        )
        intent_lost = _paired_count(
            intent.get("lost"),
            f"{score_path}: intent_effect.lost",
            n_cases,
        )
        if intent_fixed + intent_lost > n_cases:
            raise ValueError(f"{score_path}: intent_effect counts exceed n")
        _probability(
            intent.get("mcnemar_p"),
            f"{score_path}: intent_effect.mcnemar_p",
        )
        return (*percentages, p_value, n_cases)

    compare_path = evidence_dir / tag / "speconly_compare.json"
    _verify_frozen_file(compare_path, tag, "speconly_compare.json")
    comparison = _load_object(compare_path)
    compare_n = _case_count(comparison.get("n"), f"{compare_path}: n")
    if compare_n != n_cases:
        raise ValueError(f"{compare_path}: n disagrees with {score_path}")
    spec_pct = _percentage(score.get("spec_pct"), f"{score_path}: spec_pct")
    if _percentage(comparison.get("spec_pct"), f"{compare_path}: spec_pct") != spec_pct:
        raise ValueError(f"{compare_path}: spec_pct disagrees with {score_path}")
    fixed_by_spec = _paired_count(
        score.get("fixed_by_spec"),
        f"{score_path}: fixed_by_spec",
        n_cases,
    )
    lost_by_spec = _paired_count(
        score.get("lost_by_spec"),
        f"{score_path}: lost_by_spec",
        n_cases,
    )
    if fixed_by_spec + lost_by_spec > n_cases:
        raise ValueError(f"{score_path}: intent-effect counts exceed n")
    _probability(score.get("mcnemar_p"), f"{score_path}: mcnemar_p")
    broken_adds = _paired_count(
        comparison.get("broken_adds_b"),
        f"{compare_path}: broken_adds_b",
        n_cases,
    )
    broken_removes = _paired_count(
        comparison.get("broken_removes_c"),
        f"{compare_path}: broken_removes_c",
        n_cases,
    )
    if broken_adds + broken_removes > n_cases:
        raise ValueError(f"{compare_path}: regen-vs-repair counts exceed n")
    return (
        _percentage(score.get("nospec_pct"), f"{score_path}: nospec_pct"),
        spec_pct,
        _percentage(comparison.get("speconly_pct"), f"{compare_path}: speconly_pct"),
        _probability(comparison.get("mcnemar_p"), f"{compare_path}: mcnemar_p"),
        n_cases,
    )


def verdict(delta: float, p_value: float) -> str:
    """Report a descriptive difference; all six tests are exploratory and uncorrected."""

    if p_value < 1e-5:
        p_text = "p{<}10^{-5}"
    elif p_value < 0.1:
        p_text = f"p{{=}}{p_value:.3f}"
    else:
        p_text = f"p{{=}}{p_value:.2f}"
    return f"spec-only ${delta:+.1f}$pp, ${{{p_text}}}$"


def load_rows(evidence_dir: Path) -> tuple[list[RealBugRow], int]:
    rows: list[RealBugRow] = []
    case_counts: set[int] = set()
    pinned_score_tags = {
        tag
        for tag, filename in FROZEN_EVIDENCE_SHA256
        if filename == "score.json"
    }
    if pinned_score_tags != set(LABELS):
        raise ValueError(
            "Table 6 code/lock inventory mismatch: "
            f"labels={sorted(LABELS)}, pinned={sorted(pinned_score_tags)}"
        )
    missing = [
        tag
        for tag in LABELS
        if not (evidence_dir / tag / "score.json").is_file()
    ]
    if missing:
        raise ValueError(f"Table 6 evidence inventory is incomplete: {missing}")
    for tag, (label, _) in LABELS.items():
        nospec, spec, speconly, p_value, n_cases = read_score(evidence_dir, tag)
        case_counts.add(n_cases)
        rows.append(
            (
                nospec,
                label,
                nospec,
                spec,
                speconly,
                speconly - spec,
                p_value,
            )
        )
    if len(case_counts) != 1:
        raise ValueError(f"real-bug model case counts disagree: {sorted(case_counts)}")
    if case_counts != {EXPECTED_CASES}:
        raise ValueError(
            f"Table 6 requires denominator {EXPECTED_CASES}, got {sorted(case_counts)}"
        )
    rows.sort(key=lambda row: row[0], reverse=True)
    return rows, next(iter(case_counts))


# tag in the rescore artifact for each label in this table
RESCORE_TAGS = {
    "gpt-5.5 (OpenAI, frozen anchor)": "gpt55",
    "Claude Opus 4.8 (Anthropic, anchor)": "claude_opus_4_8",
    "gpt-5.4-mini (OpenAI)": "gpt54mini",
    "gpt-4.1 (OpenAI)": "gpt41",
    "gpt-4o-mini (OpenAI)": "gpt4omini",
    "Claude Opus 4.6 (Anthropic)": "claude_opus_4_6",
}


def _corrected(field: str) -> dict[str, float]:
    """Delta under the comparison guard, from the rescore of the same candidates.

    The panel was scored before the pre-initialization compare was identified, so the
    shipped column and the corrected one are both reported: a reader should be able to see
    what the fix moves rather than take it on trust.
    """
    path = project_root() / "artifacts/public/realbugs/cycle1_rescore.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text(encoding="utf-8"))["models"]
    out = {}
    for label, tag in RESCORE_TAGS.items():
        entry = data.get(tag, {}).get("cycle1", {})
        if field in entry:
            out[label] = entry[field]
    return out


def render(
    rows: list[RealBugRow],
    n_cases: int,
) -> str:
    corrected = _corrected("speconly_minus_spec_pp")
    corrected_p = _corrected("mcnemar_p")

    def _corr(label: str) -> str:
        v = corrected.get(label)
        return f"${v:+.1f}$" if v is not None else "---"

    def _corr_p(label: str) -> str:
        v = corrected_p.get(label)
        if v is None:
            return "---"
        return "$p{<}10^{-5}$" if v < 1e-5 else f"$p{{=}}{v:.3f}$"

    body = "\n".join(
        f"{label} & {nospec:.1f}\\% & {spec:.1f}\\% & {speconly:.1f}\\% & "
        f"${delta:+.1f}$ & {_corr(label)} & {_corr_p(label)} \\\\"
        for _, label, nospec, spec, speconly, delta, p_value in rows
    )
    caption_start = (
        f"\\caption{{Multi-model mined-model-failure ablation ({len(rows)} models, "
        f"{n_cases} bugs each, differential-sim scored),"
    )
    caption_body = (
        "ordered by unaided recovery. $\\Delta$ is spec-only minus spec-with-the-bug, so positive means\n"
        "showing the model its own broken RTL \\emph{hurt}. $\\Delta_{\\text{c1}}$ re-scores the same\n"
        "candidates with the pre-initialization compare guarded (Supplementary\n"
        "Section~\\ref{app:oracleflip}) and is the column to read; the shipped $\\Delta$ stays beside it so\n"
        "the correction is visible rather than asserted. $p$ is the exact McNemar test on the guarded\n"
        "scorer, descriptive and uncorrected. The 274 bugs come from 112 tasks, so a task-clustered\n"
        "interval on the anchor is wider than the case-level test ($-3.3$pp, $[-9.9, +3.3]$).}"
    )
    return f"""% GENERATED by scripts/e6_table6_realbug.py -- do not edit by hand.
\\begin{{table}}[h]\\centering\\footnotesize
{caption_start}
{caption_body}
\\label{{tab:multimodel}}
\\begin{{tabular}}{{lrrrrrl}}\\toprule
model (family) & no-spec & $+$spec & spec-only & $\\Delta$ & $\\Delta_{{\\text{{c1}}}}$ & $p$ (c1) \\\\\\midrule
{body}\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""


def atomic_write(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(payload)
        os.replace(temporary_path, path)
    except Exception:
        temporary_path.unlink(missing_ok=True)
        raise


def generate(evidence_dir: Path, output_dir: Path) -> tuple[Path, list[RealBugRow]]:
    rows, n_cases = load_rows(evidence_dir)
    output_path = output_dir / "tab_multimodel.tex"
    atomic_write(output_path, render(rows, n_cases))
    return output_path, rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--evidence-dir",
        type=Path,
        help="real-bug evidence directory (default: artifacts/public/realbugs)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for tab_multimodel.tex (default: build/paper-inputs)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root()
    evidence_dir = args.evidence_dir or root / "artifacts/public/realbugs"
    output_dir = args.output_dir or root / "build" / "paper-inputs"
    output_path, rows = generate(evidence_dir, output_dir)
    print(f"wrote {os.path.relpath(output_path, root)} with {len(rows)} models\n")
    print(f"{'model':<34}{'no-spec':>9}{'+spec':>8}{'spec-only':>11}{'delta':>8}{'p':>10}")
    for _, label, nospec, spec, speconly, delta, p_value in rows:
        print(
            f"{label:<34}{nospec:>8.1f}%{spec:>7.1f}%{speconly:>10.1f}%"
            f"{delta:>+8.1f}{p_value:>10.2g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
