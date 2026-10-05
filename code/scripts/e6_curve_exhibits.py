#!/usr/bin/env python3
"""Regenerate the capability-curve paper fragments from public evidence.

The default input is the frozen standalone snapshot at
``artifacts/public/capability/capability_curve.json``.  Generated LaTeX
fragments default to ``build/paper-inputs/``; release maintainers may explicitly
select ``paper/`` when promoting a reviewed snapshot.  No model-run tree is
required.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
import pathlib
from pathlib import Path
import random
import tempfile
from typing import Any, Iterable


def project_root() -> Path:
    override = os.environ.get("RTLREPAIR_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return Path(__file__).resolve().parents[2]


# Display names and where each label sits relative to its point.  The two
# low-baseline models are 1.5 points apart on x, so they are separated by hand;
# everything else is centred above.  Values are (label, anchor, x shift, y shift).
STYLE = {
    "azure/openai/gpt-5.5": ("gpt-5.5",), "claude-opus-4-8": ("Claude 4.8",),
    "azure/openai/gpt-4.1": ("gpt-4.1",), "openai/openai/gpt-4o-mini": ("gpt-4o-mini",),
    "v5": ("qwen2.5-coder-7b-repair-sft",), "nvcf/meta/llama-3.3-70b-instruct": ("llama-3.3-70b",),
    "base": ("Qwen-Coder base",), "OriGen_Fix (RTL-specific 7B)": ("OriGen\_Fix",),
    "verireason": ("VeriReason-CodeLlama",), "vrqwen": ("VeriReason-Qwen",),
}

N_CURVE_CASES = 68
# tab_curve stays on the five models of the frozen snapshot: its numbers are what
# claims.json and the lock verify, and the wider roster in the figure comes from
# artifacts the snapshot does not cover. tab_decomposition is the ten-model table.
FROZEN_CURVE_MODELS = ("azure/openai/gpt-5.5", "azure/openai/gpt-4.1",
                       "openai/openai/gpt-4o-mini", "nvcf/meta/llama-3.3-70b-instruct",
                       "OriGen_Fix (RTL-specific 7B)")
FROZEN_CURVE_SHA256 = "a1ec9d5f7b3133c410b2fbb49124e80c81093b83b3ab1c1da07d369f2299e053"
FROZEN_AUDIT_SHA256 = "38ddec524d62593ee72bf117243cf28e08182fe6dc993dc5f784c591562c7209"
EXPECTED_COHORT = "retained_equivalence_harness_scheduled_cohort"
EXPECTED_CASE_IDS_SHA256 = "7db08fda17e8a5cd0aeecca6e040faed37da9816ad2e048c02c13cf72ac2b65e"
EXPECTED_FORMAL_INTERSECTION_SHA256 = (
    "f23d72278fc1b9174c47eadc41613390bb26d84d4ff49cd4bbba9e51878c6f57"
)
EXPECTED_SCORING_RULE = "strict_formal_proved_only"
ARMS = ("spec0_loc0", "spec1_loc0", "spec0_loc1", "spec1_loc1")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{label} must be finite")
    return result


def _interval(value: Any, label: str) -> tuple[float, float]:
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{label} must be a two-element list")
    lower = _number(value[0], f"{label}[0]")
    upper = _number(value[1], f"{label}[1]")
    if lower > upper:
        raise ValueError(f"{label} lower endpoint exceeds upper endpoint")
    return lower, upper


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _load_object(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _verify_cohort_manifest(path: Path) -> tuple[dict[str, Any], list[str]]:
    document = _load_object(path, "curve cohort manifest")
    case_ids = document.get("case_ids")
    if (
        document.get("schema_version") != 1
        or document.get("cohort") != EXPECTED_COHORT
        or document.get("count") != 68
        or not isinstance(case_ids, list)
        or len(case_ids) != 68
        or any(not isinstance(value, str) or not value for value in case_ids)
        or case_ids != sorted(set(case_ids))
    ):
        raise ValueError("curve cohort manifest identity/count is invalid")
    digest = hashlib.sha256(_canonical_json_bytes(case_ids)).hexdigest()
    if digest != EXPECTED_CASE_IDS_SHA256 or document.get("case_ids_sha256") != digest:
        raise ValueError("curve cohort manifest case-id hash mismatch")
    return document, case_ids


def _verify_formal_manifest(
    path: Path,
    curve_ids: list[str],
) -> tuple[dict[str, Any], list[str]]:
    document = _load_object(path, "anchor formal-intersection manifest")
    case_ids = document.get("case_ids")
    if (
        document.get("schema_version") != 1
        or document.get("cohort")
        != "frozen_gpt55_four_arm_formal_definitive_intersection"
        or document.get("count") != 68
        or not isinstance(case_ids, list)
        or len(case_ids) != 68
        or any(not isinstance(value, str) or not value for value in case_ids)
        or case_ids != sorted(set(case_ids))
    ):
        raise ValueError("anchor formal-intersection manifest identity/count is invalid")
    digest = hashlib.sha256(_canonical_json_bytes(case_ids)).hexdigest()
    if (
        digest != EXPECTED_FORMAL_INTERSECTION_SHA256
        or document.get("case_ids_sha256") != digest
    ):
        raise ValueError("anchor formal-intersection case-id hash mismatch")
    relationship = document.get("relationship_to_curve68")
    if not isinstance(relationship, dict) or (
        relationship.get("curve_cohort") != EXPECTED_COHORT
        or relationship.get("curve_count") != 68
        or relationship.get("overlap_count") != 52
        or relationship.get("identical") is not False
        or len(set(case_ids) & set(curve_ids)) != 52
    ):
        raise ValueError("anchor/curve manifest relationship is invalid")
    return document, case_ids


def _seed_macro(
    records: dict[tuple[str, str], dict[str, Any]],
    cases: list[str],
    arm_a: str,
    arm_b: str,
) -> tuple[float, list[float]]:
    clusters: dict[str, list[int]] = defaultdict(list)
    for case_id in cases:
        left, right = records[(case_id, arm_a)], records[(case_id, arm_b)]
        if left["seed_id"] != right["seed_id"]:
            raise ValueError(f"anchor cluster drift for {case_id}")
        clusters[left["seed_id"]].append(
            int(right["formal_status"] == "PROVED")
            - int(left["formal_status"] == "PROVED")
        )
    keys = sorted(clusters)
    means = {key: sum(clusters[key]) / len(clusters[key]) for key in keys}
    point = sum(means.values()) / len(means) * 100
    rng = random.Random(7)
    draws = sorted(
        sum(means[keys[rng.randrange(len(keys))]] for _ in keys) / len(keys) * 100
        for _ in range(10000)
    )
    return round(point, 1), [round(draws[250], 1), round(draws[9750], 1)]


def _verify_anchor_from_public_rows(
    root: Path,
    case_ids: list[str],
    anchor: dict[str, Any],
) -> None:
    rows_path = (
        root
        / "artifacts/public/primary/analysis/vericodegen_full85_rows.jsonl"
    )
    try:
        rows = [
            json.loads(line)
            for line in rows_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read public anchor rows: {exc}") from exc
    expected = {(case_id, arm) for case_id in case_ids for arm in ARMS}
    records: dict[tuple[str, str], dict[str, Any]] = {}
    for index, row in enumerate(rows):
        if row.get("mode") != "main" or row.get("case_id") not in case_ids:
            continue
        key = (row.get("case_id"), row.get("arm"))
        if key not in expected or key in records:
            raise ValueError(f"public anchor row roster error at row {index}: {key!r}")
        if not isinstance(row.get("success"), bool):
            raise ValueError(f"public anchor row {index} has non-Boolean success")
        status = row.get("formal_status")
        if status not in {None, "PROVED", "COUNTEREXAMPLE", "TIMEOUT", "UNSUPPORTED"}:
            raise ValueError(f"public anchor row {index} has invalid formal status")
        seed_id = row.get("seed_id")
        if not isinstance(seed_id, str) or not seed_id:
            raise ValueError(f"public anchor row {index} has invalid seed id")
        records[key] = {"formal_status": status, "seed_id": seed_id}
    if set(records) != expected:
        raise ValueError("public anchor rows do not cover the exact curve cohort")

    baseline = round(
        sum(records[(case_id, "spec0_loc0")]["formal_status"] == "PROVED" for case_id in case_ids)
        / len(case_ids)
        * 100,
        1,
    )
    what, what_ci = _seed_macro(records, case_ids, "spec0_loc0", "spec1_loc0")
    where, where_ci = _seed_macro(records, case_ids, "spec0_loc0", "spec0_loc1")
    expected_values = {
        "baseline_pct": baseline,
        "what_pp": what,
        "what_ci": tuple(what_ci),
        "where_pp": where,
        "where_ci": tuple(where_ci),
    }
    for key, value in expected_values.items():
        if anchor.get(key) != value:
            raise ValueError(
                f"public anchor replay mismatch for {key}: expected {value}, got {anchor.get(key)}"
            )


def _verify_cohort_audit(
    path: Path,
    curve_sha256: str,
    manifest_path: Path,
    manifest: dict[str, Any],
    formal_case_ids: list[str],
    curve_case_ids: list[str],
) -> None:
    if _sha256_file(path) != FROZEN_AUDIT_SHA256:
        raise ValueError("capability cohort audit hash mismatch")
    audit = _load_object(path, "capability cohort audit")
    if (
        audit.get("schema_version") != 1
        or audit.get("status") != "verified_exact_case_arm_roster"
        or audit.get("scoring_rule") != "strict_formal_proved_only_for_all_models"
    ):
        raise ValueError("capability cohort audit contract is invalid")
    bound_manifest = audit.get("cohort_manifest")
    if not isinstance(bound_manifest, dict) or (
        bound_manifest.get("sha256") != _sha256_file(manifest_path)
        or bound_manifest.get("cohort") != EXPECTED_COHORT
        or bound_manifest.get("count") != 68
        or bound_manifest.get("case_ids_sha256") != manifest["case_ids_sha256"]
    ):
        raise ValueError("capability cohort audit does not bind the manifest")
    curve_output = audit.get("curve_output")
    if not isinstance(curve_output, dict) or (
        curve_output.get("models") != 5 or curve_output.get("sha256") != curve_sha256
    ):
        raise ValueError("capability cohort audit does not bind the curve output")
    bindings = audit.get("source_bindings")
    if not isinstance(bindings, list) or len(bindings) != 5:
        raise ValueError("capability cohort audit must bind five model sources")
    models = set()
    for binding in bindings:
        if not isinstance(binding, dict):
            raise ValueError("capability source binding must be an object")
        model = binding.get("model")
        if not isinstance(model, str) or model in models:
            raise ValueError("capability source bindings have duplicate/invalid models")
        models.add(model)
        if (
            binding.get("observed_case_count") != 68
            or binding.get("observed_case_ids_sha256") != EXPECTED_CASE_IDS_SHA256
            or binding.get("exact_case_arm_roster") is not True
        ):
            raise ValueError(f"capability source binding roster mismatch: {model}")
    anchor = next((item for item in bindings if item.get("model") == "azure/openai/gpt-5.5"), None)
    if not isinstance(anchor, dict) or (
        anchor.get("formal_definitive_cells") != 249
        or anchor.get("non_formal_cells") != 23
        or anchor.get("scoring_rule") != "PROVED=true; every other formal status=false"
    ):
        raise ValueError("capability anchor evidence mix is not the audited 249/23 split")
    relationship = audit.get("cohort_relationship")
    if not isinstance(relationship, dict) or (
        relationship.get("formal_intersection_count") != 68
        or relationship.get("formal_intersection_case_ids_sha256")
        != EXPECTED_FORMAL_INTERSECTION_SHA256
        or relationship.get("overlap_count") != 52
        or relationship.get("identical") is not False
    ):
        raise ValueError("capability/formal cohort relationship audit is invalid")
    if (
        relationship.get("curve_only_case_ids")
        != sorted(set(curve_case_ids) - set(formal_case_ids))
        or relationship.get("formal_only_case_ids")
        != sorted(set(formal_case_ids) - set(curve_case_ids))
    ):
        raise ValueError("capability/formal cohort difference sets are invalid")
    selection = anchor.get("cohort_selection_evidence")
    if not isinstance(selection, dict) or (
        selection.get("derived_count") != 68
        or selection.get("derived_case_ids_sha256") != EXPECTED_CASE_IDS_SHA256
        or selection.get("exact_manifest_match") is not True
    ):
        raise ValueError("historical retained-harness selection was not re-derived")


def load_curve(path: Path) -> list[dict[str, Any]]:
    try:
        actual_sha256 = _sha256_file(path)
    except OSError as exc:
        raise ValueError(f"cannot read capability curve {path}: {exc}") from exc
    if actual_sha256 != FROZEN_CURVE_SHA256:
        raise ValueError(
            f"capability curve hash mismatch: expected {FROZEN_CURVE_SHA256}, "
            f"got {actual_sha256}"
        )
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read capability curve {path}: {exc}") from exc
    if not isinstance(value, list) or not value:
        raise ValueError("capability curve must be a non-empty JSON list")

    root = project_root()
    manifest_path = root / "configs/vericodegen/curve68_manifest.json"
    manifest, case_ids = _verify_cohort_manifest(manifest_path)
    _, formal_case_ids = _verify_formal_manifest(
        root / "configs/vericodegen/anchor_formal68_manifest.json",
        case_ids,
    )
    _verify_cohort_audit(
        path.parent / "curve68_cohort_audit.json",
        actual_sha256,
        manifest_path,
        manifest,
        formal_case_ids,
        case_ids,
    )

    rows: list[dict[str, Any]] = []
    models: list[str] = []
    case_counts: set[int] = set()
    for index, raw in enumerate(value):
        if not isinstance(raw, dict):
            raise ValueError(f"curve row {index} must be an object")
        model = raw.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError(f"curve row {index} has no model identifier")
        if model in models:
            raise ValueError(f"duplicate capability-curve model: {model}")
        models.append(model)

        n_cases = raw.get("n_cases")
        if isinstance(n_cases, bool) or not isinstance(n_cases, int) or n_cases <= 0:
            raise ValueError(f"curve row {index} has invalid n_cases")
        case_counts.add(n_cases)
        if (
            n_cases != 68
            or raw.get("cohort") != EXPECTED_COHORT
            or raw.get("case_ids_sha256") != EXPECTED_CASE_IDS_SHA256
            or raw.get("scoring_rule") != EXPECTED_SCORING_RULE
        ):
            raise ValueError(f"curve row {index} is not bound to the audited cohort/scorer")
        row = dict(raw)
        row["baseline_pct"] = _number(raw.get("baseline_pct"), f"row {index} baseline_pct")
        row["what_pp"] = _number(raw.get("what_pp"), f"row {index} what_pp")
        row["where_pp"] = _number(raw.get("where_pp"), f"row {index} where_pp")
        row["what_ci"] = _interval(raw.get("what_ci"), f"row {index} what_ci")
        row["where_ci"] = _interval(raw.get("where_ci"), f"row {index} where_ci")
        if not row["what_ci"][0] <= row["what_pp"] <= row["what_ci"][1]:
            raise ValueError(f"curve row {index} What estimate falls outside its CI")
        if not row["where_ci"][0] <= row["where_pp"] <= row["where_ci"][1]:
            raise ValueError(f"curve row {index} Where estimate falls outside its CI")
        rows.append(row)

    missing = sorted(set(STYLE) - set(models))
    unexpected = sorted(set(models) - set(STYLE))
    if missing or unexpected:
        raise ValueError(
            f"capability-curve model inventory drifted; missing={missing}, unexpected={unexpected}"
        )
    if len(case_counts) != 1:
        raise ValueError(f"capability-curve n_cases values disagree: {sorted(case_counts)}")
    anchor = next(row for row in rows if row["model"] == "azure/openai/gpt-5.5")
    _verify_anchor_from_public_rows(root, case_ids, anchor)
    return sorted(rows, key=lambda row: row["baseline_pct"])


def half(ci: tuple[float, float]) -> float:
    """Return the symmetric half-width consumed by pgfplots."""

    return round((ci[1] - ci[0]) / 2, 2)



def full_roster(source_root: str) -> list[dict]:
    """Every model with a What effect, ordered by baseline.

    All four inputs are public artifacts inside the bundle, so a fresh export
    rebuilds this figure without the run tree. Where is read wherever the artifact
    carries it and stays None otherwise, so a missing triangle always means an arm
    that was not run rather than a null that was measured.
    """
    import json as _json
    import os as _os
    cap = f"{source_root}/artifacts/public/capability"
    rows = []
    for r in _json.load(open(f"{cap}/capability_curve.json")):
        rows.append({"model": r["model"], "baseline_pct": r["baseline_pct"],
                     "what_pp": r["what_pp"], "what_ci": r["what_ci"],
                     "where_pp": r["where_pp"], "where_ci": r["where_ci"]})
    # The historical Claude Opus 4.8 aggregate is intentionally excluded: its
    # candidate-level rows were not retained, so its interval is not recomputable.
    # the second anchor belongs on the curve for the same reason it belongs in the
    # decomposition table: it is Tier F scored on the same estimand, and it is the only
    # point between 41% and 85% baseline. Dropping it from one exhibit and not the other
    # left the prose describing eight models below the anchor while the table showed nine.
    a2p = f"{cap}/second_anchor_claude_opus_4_8_tierf.json"
    if _os.path.isfile(a2p):
        a2 = _json.load(open(a2p))
        sm = a2["seed_macro"]
        rows.append({"model": "claude-opus-4-8", "baseline_pct": a2["where"]["baseline_pct"],
                     "what_pp": sm["what"]["pp"], "what_ci": sm["what"]["ci"],
                     "where_pp": sm["where"]["pp"], "where_ci": sm["where"]["ci"]})
    ow = _json.load(open(f"{cap}/openweight_what_effects.json"))
    of = _json.load(open(f"{cap}/openweight_tier_f.json"))
    for k in sorted(ow):
        rows.append({"model": k, "baseline_pct": of["arms"][f"{k}__diag"]["tier_f_pct"],
                     "what_pp": ow[k]["what_pp"], "what_ci": ow[k]["ci"],
                     "where_pp": ow[k].get("where_pp"), "where_ci": ow[k].get("where_ci")})
    return sorted(rows, key=lambda r: r["baseline_pct"])


def render(curve: list[dict[str, Any]]) -> tuple[str, str]:
    # Categorical axis: ten models evenly spaced and ordered by baseline, which is
    # printed under each name. On a true baseline scale the crowded pairs are under
    # 3mm apart and no labelling survives it. Where is plotted only for the models
    # whose localization arm ran; the rest carry What alone.
    ordered = sorted(curve, key=lambda r: r["baseline_pct"])
    DODGE = 0.16
    what = " ".join(
        f"({i - DODGE},{row['what_pp']}) +- (0,{half(row['what_ci'])})"
        for i, row in enumerate(ordered)
    )
    where = " ".join(
        f"({i + DODGE},{row['where_pp']}) +- (0,{half(row['where_ci'])})"
        for i, row in enumerate(ordered) if row.get("where_pp") is not None
    )
    ticks = ",".join(str(i) for i in range(len(ordered)))
    ticklabels = ",".join(
        "{" + STYLE[row["model"]][0] + r"\\[-1pt]{\tiny " + f"{row['baseline_pct']:.1f}" + r"\%}" + "}"
        for row in ordered
    )
    ceiling = len(ordered) - 1
    # The band marks the models with no headroom left, so its edge follows the baselines
    # rather than the roster size: hardcoding a width silently swallowed a 41.2%-baseline
    # model the moment the roster changed length.
    CEILING_BASELINE_PCT = 70.0
    n_ceiling = sum(1 for r in ordered if r["baseline_pct"] >= CEILING_BASELINE_PCT)
    band_left = ceiling - n_ceiling + 0.5 if n_ceiling else ceiling + 1.0
    n_where = sum(1 for r in ordered if r.get("where_pp") is not None)

    figure = f"""% GENERATED by scripts/e6_curve_exhibits.py -- do not edit by hand.
\\begin{{axis}}[
  width=\\linewidth, height=5.8cm,
  ylabel={{effect (pp)}},
  xmin=-0.7, xmax={ceiling + 0.7}, ymin=-14, ymax=62,
  xtick={{{ticks}}}, xticklabels={{{ticklabels}}},
  x tick label style={{font=\\tiny, align=center, rotate=34, anchor=north east, inner sep=1.5pt}},
  ytick={{0,20,40,60}}, ymajorgrids, grid style={{gray!16, line width=0.3pt}},
  tick align=outside, tick style={{gray!55}}, axis line style={{gray!55}},
  every axis plot/.append style={{thick}},
  legend style={{at={{(0.5,-0.42)}}, anchor=north, legend columns=2, draw=none,
                font=\\small, column sep=1.2em}},
  legend cell align=left,
]
\\addplot[draw=none, fill=gray!8, forget plot]
  coordinates {{({band_left},-14) ({ceiling + 0.7},-14) ({ceiling + 0.7},62) ({band_left},62)}} \\closedcycle;
\\node[gray!60, font=\\scriptsize, anchor=north east] at (axis cs:{ceiling + 0.6},60) {{near the ceiling}};
\\addplot[gray!70, dashed, forget plot, domain=-0.7:{ceiling + 0.7}] {{0}};
\\addplot[gray!45, densely dotted, forget plot, domain=-0.7:{ceiling + 0.7}] {{10}};
\\addplot[oiblue, only marks, mark=*, mark size=2pt,
  error bars/.cd, y dir=both, y explicit, error bar style={{oiblue!65, line width=0.6pt}}]
  coordinates {{{what}}};
\\addlegendentry{{What (specification), all {len(ordered)}}}
\\addplot[oiorange, only marks, mark=triangle*, mark size=2.5pt,
  error bars/.cd, y dir=both, y explicit, error bar style={{oiorange!65, line width=0.6pt}}]
  coordinates {{{where}}};
\\addlegendentry{{Where (localization), {'all ' + str(n_where) if n_where == len(ordered) else 'the ' + str(n_where) + ' with a loc arm'}}}
\\end{{axis}}
"""
    rows = "\n".join(
        f"{STYLE[row['model']][0]} & {row['baseline_pct']:.1f} & "
        f"${row['what_pp']:+.1f}$ & "
        f"$[{row['what_ci'][0]:+.1f}, {row['what_ci'][1]:+.1f}]$ & "
        f"${row['where_pp']:+.1f}$ & "
        f"$[{row['where_ci'][0]:+.1f}, {row['where_ci'][1]:+.1f}]$ \\\\"
        for row in curve if row.get("where_pp") is not None and row["model"] in FROZEN_CURVE_MODELS
    )
    caption_start = (
        "\\caption{Capability curve behind Figure~\\ref{fig:curve} "
        f"(exploratory strict-formal score, retained-harness {N_CURVE_CASES}-case cohort;"
    )
    column_header = (
        "model & baseline (\\%) & \\multicolumn{2}{c}{What (spec)} & "
        "\\multicolumn{2}{c}{Where (loc)} \\\\"
    )
    table = f"""% GENERATED by scripts/e6_curve_exhibits.py -- do not edit by hand.
\\begin{{table}}[t]
{caption_start}
seed-macro risk differences with 95\\% cluster-bootstrap CIs). Both aids are large where the model
has headroom and small-to-null at the gpt-5.5 ceiling.}}
\\label{{tab:curve}}
\\centering\\small
\\begin{{tabular}}{{lrrlrl}}
\\toprule
{column_header}
\\cmidrule(lr){{3-4}}\\cmidrule(lr){{5-6}}
 & & pp & 95\\% CI & pp & 95\\% CI \\\\
\\midrule
{rows}
\\bottomrule
\\end{{tabular}}
\\end{{table}}
"""
    return figure, table


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


def generate(curve_path: Path, output_dir: Path) -> tuple[Path, Path, list[dict[str, Any]]]:
    # The figure now carries the same roster as the decomposition table, so the two
    # exhibits cannot show different sets of models.
    # curve_path is <bundle>/artifacts/public/capability/capability_curve.json
    source_root = str(curve_path.resolve().parents[3])
    curve = full_roster(source_root)
    if not curve:
        curve = load_curve(curve_path)
    figure, table = render(curve)
    figure_path = output_dir / "fig_curve_body.tex"
    table_path = output_dir / "tab_curve.tex"
    atomic_write(figure_path, figure)
    atomic_write(table_path, table)
    return figure_path, table_path, curve


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--curve",
        type=Path,
        help="capability curve JSON (default: artifacts/public/capability snapshot)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="directory for fig_curve_body.tex and tab_curve.tex (default: build/paper-inputs)",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = project_root()
    curve_path = args.curve or root / "artifacts/public/capability/capability_curve.json"
    output_dir = args.output_dir or root / "build" / "paper-inputs"
    figure_path, table_path, curve = generate(curve_path, output_dir)
    print(
        f"wrote {os.path.relpath(figure_path, root)} and {os.path.relpath(table_path, root)} "
        f"from {len(curve)} models\n"
    )
    print(f"{'model':<16}{'baseline':>10}{'What':>9}{'Where':>9}")
    for row in curve:
        print(
            f"{STYLE[row['model']][0]:<16}{row['baseline_pct']:>9.1f}%"
            f"{row['what_pp']:>+9.1f}"
            + (f"{row['where_pp']:>+9.1f}" if row.get("where_pp") is not None
               else f"{'--':>9}")
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
