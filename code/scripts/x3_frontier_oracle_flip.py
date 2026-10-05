#!/usr/bin/env python3
"""X3 -- the simulation and formal oracles reverse a descriptive contrast.

Section 4.1 reports that bounded differential simulation disagrees with unbounded formal
and that the shortfall is about four times larger on specification-conditioned candidates.
This uses a retained historical endpoint-output snapshot. The provider alias is not an
immutable model revision, so the script describes the frozen outputs rather than a frozen
model.

Both arms are run at a matched 4096-token budget, greedy, anonymized, on the same 85
semantic cases, and every candidate is retained so it can be scored twice: once by the
differential testbench the field uses, once by unbounded sequential equivalence checking.

  Tier S, cycles 0--199       : diagnostic 60/79, diagnostic+spec 51/79 -> -11.4 pp
  Tier S, cycles 1--199       : diagnostic 72/79, diagnostic+spec 78/79 ->  +7.6 pp
  Tier S, reset-aware init.   : diagnostic 71/79, diagnostic+spec 78/79 ->  +8.9 pp
  Tier F, per-design contract : diagnostic 71/79, diagnostic+spec 78/79 ->  +8.9 pp

The gap contains false divergences and is not evenly distributed. Re-scoring those cases
under ``reset_aware_sim`` clears almost all of them, supporting reset and first-cycle
semantics as the main measured source without claiming an internal model mechanism.

Reads the retained candidates and the formal verdicts; writes
artifacts/public/capability/frontier_oracle_flip.json.
"""
import json
import sys
from collections import Counter
from math import comb
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
for _p in (str(_CODE_DIR), str(_CODE_DIR / "datagen"), str(Path(__file__).resolve().parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from project_paths import output_root, project_root  # noqa: E402

SOURCE_ROOT = project_root()
PROBE = output_root().parent / "generated/gpt55_tierf_probe"
SEEDS = output_root().parent / "generated/repairbench/seeds"
OUT = SOURCE_ROOT / "artifacts/public/capability/frontier_oracle_flip.json"
ARMS = ("diag", "spec")
DEFINITIVE = ("PROVED", "COUNTEREXAMPLE")


def _status(value):
    if isinstance(value, dict):
        return value.get("status") or value.get("verdict")
    return value


def _mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(0, min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def _paired_summary(hit: dict[tuple[str, str], bool], cases: list[str]) -> dict:
    """Summarize two paired arms without subtracting rounded percentages."""

    n = len(cases)
    diag_hits = sum(bool(hit[("diag", case)]) for case in cases)
    spec_hits = sum(bool(hit[("spec", case)]) for case in cases)
    both_pass = sum(
        bool(hit[("diag", case)]) and bool(hit[("spec", case)]) for case in cases
    )
    diag_only = sum(
        bool(hit[("diag", case)]) and not bool(hit[("spec", case)]) for case in cases
    )
    spec_only = sum(
        not bool(hit[("diag", case)]) and bool(hit[("spec", case)]) for case in cases
    )
    both_fail = n - both_pass - diag_only - spec_only
    return {
        "n": n,
        "arms": {
            "diag": {"hits": diag_hits, "pct": round(diag_hits / n * 100, 1)},
            "spec": {"hits": spec_hits, "pct": round(spec_hits / n * 100, 1)},
        },
        "paired": {
            "both_pass": both_pass,
            "diag_only_wins": diag_only,
            "spec_only_wins": spec_only,
            "both_fail": both_fail,
            "spec_minus_diag_pp": round((spec_hits - diag_hits) / n * 100, 1),
            "mcnemar_p": _mcnemar(diag_only, spec_only),
        },
    }


def main(check_reset_contract: bool = True) -> int:
    formal, tier_s, seed_of = {}, {}, {}
    for line in (PROBE / "results.jsonl").read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        _, arm, case = row["call_id"].split("__", 2)
        formal[(arm, case)] = _status(row["formal_verdict"])
    for arm in ARMS:
        for row in json.loads((PROBE / f"rb_{arm}.json").read_text())["records"]:
            tier_s[(arm, row["id"])] = bool(row["recovered"])
            seed_of[row["id"]] = row["seed_id"]

    cases = sorted(seed_of)
    both = [c for c in cases if all(formal.get((a, c)) in DEFINITIVE for a in ARMS)]

    report = {
        "generated_by": "code/scripts/x3_frontier_oracle_flip.py",
        "model": "azure/openai/gpt-5.5",
        "budget_tokens": 4096,
        "note": (
            "Both arms greedy, anonymized, matched budget, same 85 cases; every candidate "
            "scored by both tiers. An empty response never occurred here (used_llm 1.0)."
        ),
        "n_cases": len(cases),
        "n_definitive_both_arms": len(both),
        "arms": {},
    }
    for arm in ARMS:
        s_hits = sum(1 for c in both if tier_s[(arm, c)])
        f_hits = sum(1 for c in both if formal[(arm, c)] == "PROVED")
        report["arms"][arm] = {
            "tier_s_hits": s_hits,
            "tier_s_pct": round(s_hits / len(both) * 100, 1),
            "tier_f_hits": f_hits,
            "tier_f_pct": round(f_hits / len(both) * 100, 1),
            "false_divergences": sum(
                1 for c in both if not tier_s[(arm, c)] and formal[(arm, c)] == "PROVED"),
            "over_credits": sum(
                1 for c in both if tier_s[(arm, c)] and formal[(arm, c)] == "COUNTEREXAMPLE"),
            "formal_verdicts_all_85": dict(
                Counter(formal.get((arm, c)) for c in cases)),
        }

    for tier, hit in (("tier_s", lambda a, c: tier_s[(a, c)]),
                      ("tier_f", lambda a, c: formal[(a, c)] == "PROVED")):
        b = sum(1 for c in both if hit("diag", c) and not hit("spec", c))
        d = sum(1 for c in both if not hit("diag", c) and hit("spec", c))
        report[f"{tier}_contrast"] = {
            # Compute the contrast from paired counts, not from already-rounded
            # marginal percentages.  This gives -9/79=-11.4 pp and
            # +7/79=+8.9 pp for the two tiers.
            "spec_minus_diag_pp": round((d - b) / len(both) * 100, 1),
            "diag_only_wins": b,
            "spec_only_wins": d,
            "mcnemar_p": _mcnemar(b, d),
        }
    report["verdict_reverses_sign"] = (
        report["tier_s_contrast"]["spec_minus_diag_pp"] < 0
        < report["tier_f_contrast"]["spec_minus_diag_pp"]
    )

    if check_reset_contract:
        from diff_testbench import _rename_module, parse_module, run_diff_test
        import reset_aware_sim  # local simulation only, no API calls

        cycle0 = {(arm, case): tier_s[(arm, case)] for arm in ARMS for case in both}
        tier_f = {
            (arm, case): formal[(arm, case)] == "PROVED"
            for arm in ARMS
            for case in both
        }
        cycle1: dict[tuple[str, str], bool] = {}
        reset_aware: dict[tuple[str, str], bool] = {}
        for arm in ARMS:
            for case in both:
                golden = (SEEDS / f"{seed_of[case]}.sv").read_text()
                candidate = (PROBE / arm / f"{case}.sv").read_text()
                info = parse_module(golden)
                if info is None:
                    raise RuntimeError(f"could not parse golden for {case}")
                candidate_info = parse_module(candidate)
                if candidate_info and candidate_info.name != info.name:
                    candidate = _rename_module(candidate, candidate_info.name, info.name)

                cycle1_result = run_diff_test(
                    golden, candidate, info, compare_from_cycle=1
                )
                if not cycle1_result.compiled:
                    raise RuntimeError(
                        f"cycle-1 simulation failed for {arm}/{case}: "
                        f"{cycle1_result.failure_kind}: {cycle1_result.log[:200]}"
                    )
                cycle1[(arm, case)] = not cycle1_result.diverged

                reset_result = reset_aware_sim.run(golden, candidate)
                if not reset_result.compiled:
                    raise RuntimeError(
                        f"reset-aware simulation failed for {arm}/{case}: "
                        f"{reset_result.log[:200]}"
                    )
                reset_aware[(arm, case)] = not reset_result.diverged

        report["contract_rescores"] = {
            "tier_s_cycle0": _paired_summary(cycle0, both),
            "tier_s_cycle1": _paired_summary(cycle1, both),
            "tier_s_reset_aware": {
                **_paired_summary(reset_aware, both),
                "scorer": "code/scripts/reset_aware_sim.py",
            },
            "tier_f": _paired_summary(tier_f, both),
        }
        report["contract_rescore_checks"] = {
            "cycle1_vs_cycle0": {
                "fail_to_pass": {
                    arm: sum(
                        not cycle0[(arm, case)] and cycle1[(arm, case)]
                        for case in both
                    )
                    for arm in ARMS
                },
                "pass_to_fail": {
                    arm: sum(
                        cycle0[(arm, case)] and not cycle1[(arm, case)]
                        for case in both
                    )
                    for arm in ARMS
                },
            },
            "reset_aware_vs_tier_f": {
                "matching_cells": sum(
                    reset_aware[(arm, case)] == tier_f[(arm, case)]
                    for arm in ARMS
                    for case in both
                ),
                "total_cells": len(ARMS) * len(both),
            },
            "cycle1_vs_reset_aware": {
                f"{arm}_disagreements": sum(
                    cycle1[(arm, case)] != reset_aware[(arm, case)] for case in both
                )
                for arm in ARMS
            },
        }

        cleared = {}
        for arm in ARMS:
            wrong = [c for c in cases
                     if formal.get((arm, c)) == "PROVED" and not tier_s.get((arm, c))]
            ok = 0
            for case in wrong:
                if (arm, case) in reset_aware:
                    ok += bool(reset_aware[(arm, case)])
                    continue
                golden = (SEEDS / f"{seed_of[case]}.sv").read_text()
                candidate = (PROBE / arm / f"{case}.sv").read_text()
                reset_result = reset_aware_sim.run(golden, candidate)
                ok += bool(reset_result.compiled and not reset_result.diverged)
            cleared[arm] = {"false_divergences": len(wrong), "cleared_by_reset_aware_tb": ok}
        report["reset_contract_attribution"] = cleared

    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(SOURCE_ROOT)}")
    for arm in ARMS:
        a = report["arms"][arm]
        print(f"  {arm:5s} Tier S {a['tier_s_pct']:5.1f}%  Tier F {a['tier_f_pct']:5.1f}%  "
              f"false divergences {a['false_divergences']}")
    for tier in ("tier_s", "tier_f"):
        c = report[f"{tier}_contrast"]
        print(f"  {tier}: {c['spec_minus_diag_pp']:+.1f}pp  "
              f"(diag-only {c['diag_only_wins']}, spec-only {c['spec_only_wins']}, "
              f"p={c['mcnemar_p']:.4f})")
    print(f"  sign reverses between tiers: {report['verdict_reverses_sign']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
