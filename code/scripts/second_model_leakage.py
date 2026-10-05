#!/usr/bin/env python3
"""Leakage control: on llama-3.3-70b, compare the specification effect with the
FULL spec vs the behavior-REDACTED spec (baseline shared). The drop from full to
redacted estimates how much of the +42pp spec effect is answer disclosure."""
import json, os, random
from collections import defaultdict

from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
MAIN = f"{ROOT}/generated/second_model"                         # baseline + full-spec
RED = f"{ROOT}/generated/second_model/runs/nvcf__meta__llama-3.3-70b-instruct__redacted"

def load(base, arms):
    res = json.load(open(f"{base}/results.json"))
    verd = {json.loads(l)["dir"]: json.loads(l)["verdict"] for l in open(f"{base}/verdicts.jsonl")}
    rec = {}
    for r in res:
        if r["arm"] not in arms:
            continue
        v = verd.get(f"{r['arm']}__{r['case_id']}", "MISS") if r.get("parsed_ok") else "NO_PARSE"
        rec[(r["case_id"], r["arm"])] = {"success": v == "PROVED", "design": r["design"]}
    return rec

main = load(MAIN, {"spec0_loc0", "spec1_loc0"})
red = load(RED, {"spec1_loc0"})

def effect(base_rec, treat_rec, seed=42):
    # base_rec keyed spec0_loc0, treat_rec keyed spec1_loc0
    cases = {c for c, a in base_rec if a == "spec0_loc0"} & {c for c, a in treat_rec if a == "spec1_loc0"}
    g = defaultdict(list)
    for c in cases:
        b = base_rec[(c, "spec0_loc0")]; t = treat_rec[(c, "spec1_loc0")]
        g[b["design"]].append(int(t["success"]) - int(b["success"]))
    ks = sorted(g); sm = {k: sum(g[k]) / len(g[k]) for k in ks}
    pt = sum(sm.values()) / len(sm)
    rng = random.Random(seed)
    dr = sorted(sum(sm[ks[rng.randrange(len(ks))]] for _ in ks) / len(ks) for _ in range(10000))
    return pt * 100, dr[250] * 100, dr[9750] * 100, len(cases)

full = effect(main, main)
redacted = effect(main, red)
# per-arm rates
def rate(rec, arm):
    xs = [rec[(c, arm)]["success"] for c in {cc for cc, a in rec if a == arm}]
    return sum(xs), len(xs)
b_s, b_n = rate(main, "spec0_loc0")
f_s, f_n = rate(main, "spec1_loc0")
r_s, r_n = rate(red, "spec1_loc0")
print(f"llama-3.3-70b leakage control (n={full[3]} cases):")
print(f"  baseline (no spec)     {b_s}/{b_n} = {b_s/b_n*100:.1f}%")
print(f"  full spec              {f_s}/{f_n} = {f_s/f_n*100:.1f}%   effect {full[0]:+.1f} [{full[1]:+.1f},{full[2]:+.1f}]")
print(f"  redacted spec          {r_s}/{r_n} = {r_s/r_n*100:.1f}%   effect {redacted[0]:+.1f} [{redacted[1]:+.1f},{redacted[2]:+.1f}]")
drop = full[0] - redacted[0]
print(f"\n  full - redacted = {drop:+.1f}pp  ({drop/full[0]*100:.0f}% of the spec effect lost to redaction)")
print(f"  => the residual redacted-spec effect ({redacted[0]:+.1f}pp) bounds the non-disclosure component.")
json.dump({"model": "llama-3.3-70b", "n_cases": full[3],
           "full_spec_effect_pp": round(full[0], 1), "full_ci": [round(full[1], 1), round(full[2], 1)],
           "redacted_spec_effect_pp": round(redacted[0], 1), "redacted_ci": [round(redacted[1], 1), round(redacted[2], 1)],
           "drop_pp": round(drop, 1), "pct_lost": round(drop / full[0] * 100)},
          open(f"{ROOT}/generated/second_model/leakage/leakage_result.json", "w"), indent=2)
print("[saved leakage_result.json]")
