#!/usr/bin/env python3
"""How much of the preregistered primary is decided by repair, and how much by silence.

The frozen contrasts score an empty model response as a failed repair, which is the
conservative choice and is what the preregistration specified. It also means a
discordant pair can be a case where one arm produced a module and the other produced
nothing at all. This script separates the two so the paper can report which kind of
event its null rests on:

  * the baseline's failures, split by what the model actually did
  * the largest seed-macro effect any aid could have shown given that baseline
  * per contrast: discordant pairs, how many involve an empty response, and the
    complete-case estimate over the pairs where both arms emitted a module

Reads only artifacts/public/primary/analysis/full85_rows.jsonl, the same rows the
frozen statistics were computed from, and reproduces the frozen point estimates as a
self-check before reporting anything derived from them.

Writes artifacts/public/primary/analysis/primary_informativeness.json.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from project_paths import project_root  # noqa: E402

ROOT = project_root()
ROWS = ROOT / "artifacts/public/primary/analysis/full85_rows.jsonl"
STATS = ROOT / "artifacts/public/primary/analysis/full85_stats.json"
OUT = ROOT / "artifacts/public/primary/analysis/primary_informativeness.json"

BASE = "spec0_loc0"
CONTRASTS = (("what_full_spec", "spec1_loc0"), ("where_oracle_location", "spec0_loc1"))
# an HTTP 200 that carries no usable module: the model spent its budget and said nothing
EMPTY = "http_200_model_failure"


def seed_macro(rows, cases, seeds, arm_a, arm_b):
    clusters = defaultdict(list)
    for case in cases:
        clusters[seeds[case]].append(
            int(bool(rows[(case, arm_b)]["success"]))
            - int(bool(rows[(case, arm_a)]["success"]))
        )
    return sum(sum(v) / len(v) for v in clusters.values()) / len(clusters) * 100


def main():
    records = [json.loads(line) for line in ROWS.read_text().splitlines() if line.strip()]
    main_rows = [r for r in records if r.get("mode") == "main"]
    rows = {(r["case_id"], r["arm"]): r for r in main_rows}
    cases = sorted({r["case_id"] for r in main_rows})
    seeds = {r["case_id"]: r["seed_id"] for r in main_rows}
    frozen = json.loads(STATS.read_text())["contrasts"]

    passed = lambda case, arm: bool(rows[(case, arm)]["success"])  # noqa: E731
    silent = lambda case, arm: rows[(case, arm)]["evidence"] == EMPTY  # noqa: E731

    # self-check: recompute the frozen point estimates before deriving anything
    for name, arm in CONTRASTS:
        got = seed_macro(rows, cases, seeds, BASE, arm)
        want = frozen[name]["risk_difference"] * 100
        if abs(got - want) > 0.05:
            raise SystemExit(f"{name}: recomputed {got:+.2f}pp, frozen says {want:+.2f}pp")

    failures = [c for c in cases if not passed(c, BASE)]
    evidence = defaultdict(int)
    for case in failures:
        evidence[rows[(case, BASE)]["evidence"]] += 1

    clusters = defaultdict(list)
    for case in cases:
        clusters[seeds[case]].append(0 if passed(case, BASE) else 1)
    ceiling = sum(sum(v) / len(v) for v in clusters.values()) / len(clusters) * 100

    report = {
        "generated_by": "code/scripts/ws_primary_informativeness.py",
        "source": "artifacts/public/primary/analysis/full85_rows.jsonl",
        "note": (
            "An empty response is scored as a failed repair, per the preregistration. "
            "These counts separate that from a repair the oracle refuted."
        ),
        "n_cases": len(cases),
        "n_seed_clusters": len(clusters),
        "baseline": {
            "arm": BASE,
            "passed": len(cases) - len(failures),
            "failed": len(failures),
            "failure_evidence": dict(sorted(evidence.items())),
            "silent": evidence[EMPTY],
        },
        "attainable_ceiling_pp": round(ceiling, 1),
        "materiality_bar_pp": 10.0,
        # An absolute-points threshold against an 84.7% baseline is cramped by arithmetic
        # rather than by the aid: no aid could have shown more than the ceiling. Reporting
        # each effect as the share of the ceiling it recovered puts the headroom inside the
        # measurement instead of leaving it to a caveat.
        "share_of_ceiling_pct": {
            name: round(frozen[name]["risk_difference"] * 100 / ceiling * 100)
            for name, _ in CONTRASTS
        },
        # The ratio inherits the numerator's uncertainty, and the numerator is a
        # non-significant difference. Reported without this interval the point value
        # reads as a recovery rate; What's 43% carries a CI that spans zero twice over.
        "share_of_ceiling_ci95_pct": {
            name: [round(frozen[name]["ci95"][0] * 100 / ceiling * 100),
                   round(frozen[name]["ci95"][1] * 100 / ceiling * 100)]
            for name, _ in CONTRASTS
        },
        "contrasts": {},
    }

    for name, arm in CONTRASTS:
        discordant = [c for c in cases if passed(c, BASE) != passed(c, arm)]
        involving_silence = [c for c in discordant if silent(c, BASE) or silent(c, arm)]
        informative = [c for c in cases if not silent(c, BASE) and not silent(c, arm)]
        report["contrasts"][name] = {
            "arm_b": arm,
            "risk_diff_pp": round(frozen[name]["risk_difference"] * 100, 1),
            "discordant_pairs": len(discordant),
            "discordant_involving_empty_response": len(involving_silence),
            "discordant_both_arms_emitted": len(discordant) - len(involving_silence),
            "complete_case_n": len(informative),
            "complete_case_pp": round(
                seed_macro(rows, informative, seeds, BASE, arm), 1
            ),
            "silent_cells": {
                BASE: sum(1 for c in cases if silent(c, BASE)),
                arm: sum(1 for c in cases if silent(c, arm)),
            },
        }

    # The 2x2's fourth cell. It is not one of the three prespecified contrasts, so it stays
    # outside the Holm family, but the paper calls the design complete and a reader can compute
    # this from the released rows either way.
    import random
    joint = "spec1_loc1"
    clusters = defaultdict(list)
    for case in cases:
        clusters[seeds[case]].append(int(passed(case, joint)) - int(passed(case, BASE)))
    keys = sorted(clusters)
    means = {k: sum(v) / len(v) for k, v in clusters.items()}
    point = sum(means.values()) / len(means) * 100
    rng = random.Random(44)
    draws = sorted(
        sum(means[keys[rng.randrange(len(keys))]] for _ in keys) / len(keys) * 100
        for _ in range(10000))
    report["joint_cell"] = {
        "arm": joint,
        "prespecified": False,
        "passed": sum(1 for c in cases if passed(c, joint)),
        "seed_macro_pp": round(point, 1),
        "ci95": [round(draws[250], 1), round(draws[9750], 1)],
        "share_of_ceiling_pct": round(point / ceiling * 100),
        "baseline_failures_fixed": sum(
            1 for c in cases if not passed(c, BASE) and passed(c, joint)),
        "baseline_successes_broken": sum(
            1 for c in cases if passed(c, BASE) and not passed(c, joint)),
    }

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    b = report["baseline"]
    print(f"  baseline {b['passed']}/{report['n_cases']} pass; of {b['failed']} failures, "
          f"{b['silent']} are empty responses")
    print(f"  attainable seed-macro ceiling +{report['attainable_ceiling_pp']}pp "
          f"vs the +{report['materiality_bar_pp']}pp bar")
    for name, c in report["contrasts"].items():
        print(f"  {name}: {c['risk_diff_pp']:+.1f}pp | {c['discordant_pairs']} discordant "
              f"({c['discordant_involving_empty_response']} involve an empty response) | "
              f"complete-case {c['complete_case_pp']:+.1f}pp on n={c['complete_case_n']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
