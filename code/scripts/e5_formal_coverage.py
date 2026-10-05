#!/usr/bin/env python3
"""E5 -- formal (Tier F) coverage recompute (REVISION_PLAN Part 1).

The plan's E5 has two parts. The proving half runs the formal pipeline over new candidate
sets; as of E0 there are none (E1's arms do not exist yet, and Claude's 2x2 candidates were
never retained), and only 34 of 3,060 banked candidates lack a definitive verdict -- all of
them prover ERROR or COMPILE_FAIL, not unscored work. This script is the other half: it
recomputes the definitive-verdict coverage that the paper hardcodes as "68", across whatever
models are currently in the Figure 1 roster, and audits how non-definitive verdicts are being
scored.

Reads retained historical rows and writes a cohort-explicit coverage audit.
"""
import hashlib, json, os, sys
from collections import defaultdict, Counter
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(SOURCE_ROOT / "code/scripts"))
from second_model_capability_curve import load_run as load_curve_run
ROOT = Path(os.environ.get("RTLREPAIR_ROOT", str(SOURCE_ROOT))).resolve()
OUT = Path(os.environ.get("E5_OUTPUT", str(ROOT / "generated/reports/e5_formal_coverage.json")))
CURVE_MANIFEST = Path(
    os.environ.get(
        "CURVE68_MANIFEST",
        str(SOURCE_ROOT / "configs/vericodegen/curve68_manifest.json"),
    )
)
FORMAL_MANIFEST = Path(
    os.environ.get(
        "ANCHOR_FORMAL68_MANIFEST",
        str(SOURCE_ROOT / "configs/vericodegen/anchor_formal68_manifest.json"),
    )
)
DEFINITIVE = ("PROVED", "COUNTEREXAMPLE")
ARMS = ["spec0_loc0", "spec1_loc0", "spec0_loc1", "spec1_loc1"]
FORMAL_STATUSES = {None, "PROVED", "COUNTEREXAMPLE", "TIMEOUT", "UNSUPPORTED"}
MAIN_INPUTS = SOURCE_ROOT / "configs/vericodegen/main85_inputs.jsonl"

# Figure 1 roster: model -> banked run. gpt-5.5 is the frozen ledger, handled separately.
RUNS = {
    "llama-3.3-70b": "generated/second_model",
    "gpt-4.1":       "generated/second_model/runs/azure__openai__gpt-4.1",
    "gpt-4o-mini":   "generated/second_model/runs/openai__openai__gpt-4o-mini",
    "OriGen_Fix":    "generated/second_model/runs/origen_fix",
}


def frozen_arms():
    """gpt-5.5: per-arm definitive case sets from the frozen preregistered ledger."""
    expected_cases = set()
    try:
        for index, line in enumerate(MAIN_INPUTS.read_text(encoding="utf-8").splitlines()):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = (row.get("internal_metadata") or {}).get("anonymous_id")
            if not isinstance(case_id, str) or not case_id:
                raise RuntimeError(f"main input row {index} has invalid anonymous id")
            if case_id in expected_cases:
                raise RuntimeError(f"duplicate main input case id: {case_id}")
            expected_cases.add(case_id)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read exact primary inputs: {exc}") from exc
    if len(expected_cases) != 85:
        raise RuntimeError("primary input roster must contain exactly 85 cases")

    ledger_path = ROOT / "generated/reports/vericodegen_full85_rows.jsonl"
    try:
        rows = [
            json.loads(line)
            for line in ledger_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot read frozen primary ledger: {exc}") from exc
    expected_keys = {(case_id, arm) for case_id in expected_cases for arm in ARMS}
    observed = {}
    per, verdicts = defaultdict(set), Counter()
    for index, row in enumerate(rows):
        if row.get("mode") != "main":
            continue
        key = (row.get("case_id"), row.get("arm"))
        if key not in expected_keys:
            raise RuntimeError(f"frozen primary row {index} is outside exact roster: {key!r}")
        if key in observed:
            raise RuntimeError(f"duplicate frozen primary cell: {key!r}")
        status = row.get("formal_status")
        if status not in FORMAL_STATUSES:
            raise RuntimeError(f"frozen primary row {index} has invalid status {status!r}")
        if not isinstance(row.get("success"), bool):
            raise RuntimeError(f"frozen primary row {index} success must be Boolean")
        observed[key] = status
        verdicts[status] += 1
        if status in DEFINITIVE:
            per[key[1]].add(key[0])
    if set(observed) != expected_keys:
        missing = sorted(expected_keys - set(observed))
        raise RuntimeError(f"frozen primary ledger is incomplete: {missing[:5]}")
    return per, verdicts


def run_arms(run, model, curve_ids):
    """A banked run: per-arm definitive case sets, plus the full verdict distribution."""
    loaded, _audit = load_curve_run(
        ROOT,
        Path(run),
        model,
        tuple(sorted(curve_ids)),
    )
    per, verdicts = defaultdict(set), Counter()
    for (case_id, arm), record in loaded["rec"].items():
        verdict = record["verdict"]
        verdicts[verdict] += 1
        if verdict in DEFINITIVE:
            per[arm].add(case_id)
    return per, verdicts


def load_manifest(path, expected_cohort):
    document = json.load(open(path))
    ids = document.get("case_ids")
    if (
        document.get("schema_version") != 1
        or document.get("cohort") != expected_cohort
        or document.get("count") != 68
        or not isinstance(ids, list)
        or ids != sorted(set(ids))
    ):
        raise RuntimeError(f"invalid 68-case manifest: {path}")
    digest = hashlib.sha256(
        json.dumps(ids, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if document.get("case_ids_sha256") != digest:
        raise RuntimeError(f"case-id digest mismatch: {path}")
    return document, set(ids)


def main():
    curve_manifest, curve_ids = load_manifest(
        CURVE_MANIFEST, "retained_equivalence_harness_scheduled_cohort"
    )
    formal_manifest, formal_ids = load_manifest(
        FORMAL_MANIFEST, "frozen_gpt55_four_arm_formal_definitive_intersection"
    )
    models = {}
    per, verdicts = frozen_arms()
    primary_intersection = set.intersection(*per.values())
    if primary_intersection != formal_ids:
        raise RuntimeError("frozen rows disagree with anchor formal68 manifest")
    models["gpt-5.5 (frozen primary)"] = (
        per,
        verdicts,
        "generated/reports/vericodegen_full85_rows.jsonl",
    )
    for m, run in RUNS.items():
        p, v = run_arms(run, m, curve_ids)
        models[m] = (p, v, run)

    report = {
        "generated_by": "e5_formal_coverage.py",
        "cohorts": {
            "primary_four_arm_formal_intersection": {
                "manifest": "configs/vericodegen/anchor_formal68_manifest.json",
                "count": len(formal_ids),
                "case_ids_sha256": formal_manifest["case_ids_sha256"],
            },
            "capability_curve_scheduled_cohort": {
                "manifest": "configs/vericodegen/curve68_manifest.json",
                "selection": "retained historical equivalence-harness availability",
                "count": len(curve_ids),
                "case_ids_sha256": curve_manifest["case_ids_sha256"],
            },
            "relationship": {
                "overlap_count": len(formal_ids & curve_ids),
                "identical": formal_ids == curve_ids,
            },
        },
        "models": {},
    }
    own = {}
    for m, (per, verdicts, src) in models.items():
        inter = set.intersection(*per.values()) if per else set()
        own[m] = inter
        # Figure 1's scorer treats any non-PROVED verdict as "not recovered" (see
        # second_model_capability_curve.py). COMPILE_FAIL is a genuine repair failure; a
        # prover ERROR is not -- it is missing evidence scored as a negative.
        report["models"][m] = {
            "source": src,
            "per_arm_definitive": {a: len(per.get(a, ())) for a in ARMS},
            "own_4arm_intersection": len(inter),
            "verdict_distribution": {str(k): v for k, v in verdicts.items()},
            "prover_errors_scored_as_failure": verdicts.get("ERROR", 0),
            "compile_fails_scored_as_failure": verdicts.get("COMPILE_FAIL", 0),
        }

    later_sets = [value for key, value in own.items() if key != "gpt-5.5 (frozen primary)"]
    later_intersection = set.intersection(*later_sets) if later_sets else set()
    report["intersections"] = {
        "per_model": {m: len(s) for m, s in own.items()},
        "primary_four_arm_formal_intersection": len(primary_intersection),
        "common_definitive_cases_across_four_later_models": len(later_intersection),
        "note": (
            "The frozen primary four-arm formal intersection and the capability-curve "
            "retained-harness cohort both contain 68 cases but are different sets: they "
            "overlap on 52. The four later models ran only on the retained-harness cohort "
            "and each has 1-3 non-definitive cells, conservatively scored as failures."
        ),
    }

    os.makedirs(OUT.parent, exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2)

    print(f"\nE5 FORMAL COVERAGE -> {os.path.relpath(OUT, ROOT)}\n")
    print(f"{'model':<20}{'spec0_loc0':>11}{'spec1_loc0':>11}{'spec0_loc1':>11}{'spec1_loc1':>11}{'4-arm':>8}{'ERROR':>7}{'CFAIL':>7}")
    print("-" * 86)
    for m, info in report["models"].items():
        pa = info["per_arm_definitive"]
        print(f"{m:<20}" + "".join(f"{pa[a]:>11}" for a in ARMS) +
              f"{info['own_4arm_intersection']:>8}"
              f"{info['prover_errors_scored_as_failure']:>7}"
              f"{info['compile_fails_scored_as_failure']:>7}")
    print(
        f"\nfrozen primary formal intersection: "
        f"{report['intersections']['primary_four_arm_formal_intersection']}"
    )
    print(
        "primary/curve cohort overlap: "
        f"{report['cohorts']['relationship']['overlap_count']}/68"
    )
    print(
        "intersection common to the four later scheduled models: "
        f"{report['intersections']['common_definitive_cases_across_four_later_models']}"
    )
    tot_err = sum(i["prover_errors_scored_as_failure"] for i in report["models"].values())
    print(f"\nprover ERRORs currently scored as repair failures: {tot_err} cells across the roster")


if __name__ == "__main__":
    main()
