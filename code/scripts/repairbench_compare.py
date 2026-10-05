#!/usr/bin/env python3
"""Preregistered base-vs-candidate comparison on the honest no-spec RepairBench arm.
Same estimand as the paper's primary: seed-macro paired risk difference (clustered by
design seed), 10k-replicate cluster bootstrap CI, exact McNemar, and the +-10-point
material-improvement rule with a TOST equivalence check.

  python3 scripts/repairbench_compare.py <BASE_TAG> <NEW_TAG>

A candidate model "genuinely improves semantic repair" only if it clears +10pp with a
CI excluding 0 on THIS arm (no spec, anonymized, formal-scored). Anything else is a
null (bounded, if TOST passes) or inconclusive -- reported honestly either way.
"""
import json, os, sys, math, random
from collections import defaultdict
from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
MARGIN = 10.0   # preregistered material-improvement margin (percentage points)


def load(tag):
    d = json.load(open(f"{ROOT}/generated/repairbench_slot/{tag}/score.json"))
    return {c: (v["recovered"], v["cluster_id"]) for c, v in d["per_case"].items()}


def mcnemar_exact(b, c):
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) * 0.5 ** n)


def seed_macro(base, new, seed=42, reps=10000):
    g = defaultdict(list)
    for c in base:
        if c in new:
            g[base[c][1]].append(int(new[c][0]) - int(base[c][0]))
    ks = sorted(g)
    sm = {k: sum(g[k]) / len(g[k]) for k in ks}
    pt = sum(sm.values()) / len(sm)
    rng = random.Random(seed)
    draws = sorted(sum(sm[ks[rng.randrange(len(ks))]] for _ in ks) / len(ks) for _ in range(reps))
    return pt * 100, draws[int(reps * 0.025)] * 100, draws[int(reps * 0.975)] * 100


def main(base_tag, new_tag):
    base, new = load(base_tag), load(new_tag)
    ids = [c for c in base if c in new]
    br = sum(1 for c in ids if base[c][0]); nr = sum(1 for c in ids if new[c][0])
    b = sum(1 for c in ids if not base[c][0] and new[c][0])   # new fixes base missed
    cc = sum(1 for c in ids if base[c][0] and not new[c][0])  # base had, new lost
    p = mcnemar_exact(b, cc)
    rd, lo, hi = seed_macro(base, new)
    # verdict
    if rd >= MARGIN and lo > 0:
        verdict = "MATERIAL IMPROVEMENT (clears +10pp, CI excludes 0)"
    elif -MARGIN < lo and hi < MARGIN:
        verdict = "BOUNDED NULL (TOST passes; effect within +-10pp)"
    else:
        verdict = "INCONCLUSIVE (CI admits a material effect but not significant)"
    out = {"base": base_tag, "candidate": new_tag, "n": len(ids),
           "base_recovery_pct": round(100 * br / len(ids), 1),
           "candidate_recovery_pct": round(100 * nr / len(ids), 1),
           "seed_macro_rd_pp": round(rd, 1), "ci95_pp": [round(lo, 1), round(hi, 1)],
           "mcnemar_b_new_fixed": b, "mcnemar_c_new_lost": cc, "mcnemar_p": p,
           "margin_pp": MARGIN, "verdict": verdict, "arm": "no_spec_anonymized_formal"}
    print(json.dumps(out, indent=2))
    outdir = f"{ROOT}/generated/repairbench_slot"
    json.dump(out, open(f"{outdir}/compare_{base_tag}_vs_{new_tag}.json", "w"), indent=2)
    print(f"\n[saved compare_{base_tag}_vs_{new_tag}.json]")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        raise SystemExit("usage: repairbench_compare.py <BASE_TAG> <NEW_TAG>")
    main(sys.argv[1], sys.argv[2])
