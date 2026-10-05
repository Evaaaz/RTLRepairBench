#!/usr/bin/env python3
"""Multi-sample full-2x2 on llama-3.3-70b: per-draw What/Where/Joint/interaction
seed-macro effects across 5 temperature-0.7 draws (msf1-5), reporting mean and
range to show the whole curve's flagship effects are stable across draws."""
import json, os
from collections import defaultdict

from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
rd = f"{ROOT}/generated/second_model/runs"
draws = sorted(d for d in os.listdir(rd) if "__msf" in d)

def load(base):
    res = json.load(open(f"{base}/results.json"))
    verd = {json.loads(l)["dir"]: json.loads(l)["verdict"] for l in open(f"{base}/verdicts.jsonl")}
    rec = {}
    for r in res:
        v = verd.get(f"{r['arm']}__{r['case_id']}", "MISS") if r.get("parsed_ok") else "NO_PARSE"
        rec[(r["case_id"], r["arm"])] = {"success": v == "PROVED", "design": r["design"]}
    return rec

def macro(rec, a, b):
    cases = {c for c, ar in rec if ar == a} & {c for c, ar in rec if ar == b}
    g = defaultdict(list)
    for c in cases:
        g[rec[(c, a)]["design"]].append(int(rec[(c, b)]["success"]) - int(rec[(c, a)]["success"]))
    return sum(sum(v)/len(v) for v in g.values()) / len(g) * 100

def inter(rec):
    arms = ["spec0_loc0","spec1_loc0","spec0_loc1","spec1_loc1"]
    cases = set.intersection(*[{c for c,a in rec if a==ar} for ar in arms])
    g = defaultdict(list)
    for c in cases:
        y = {ar:int(rec[(c,ar)]["success"]) for ar in arms}
        g[rec[(c,"spec0_loc0")]["design"]].append((y["spec1_loc1"]-y["spec0_loc1"])-(y["spec1_loc0"]-y["spec0_loc0"]))
    return sum(sum(v)/len(v) for v in g.values())/len(g)*100

rows = {"What": [], "Where": [], "Joint": [], "Interaction": []}
print(f"{'draw':8s} {'base%':>6} {'What':>7} {'Where':>7} {'Joint':>7} {'Inter':>7}")
for d in draws:
    rec = load(f"{rd}/{d}")
    base = [c for c,a in rec if a=="spec0_loc0"]
    br = sum(1 for c in base if rec[(c,"spec0_loc0")]["success"])/len(base)*100
    w = macro(rec,"spec0_loc0","spec1_loc0"); wh = macro(rec,"spec0_loc0","spec0_loc1")
    j = macro(rec,"spec0_loc0","spec1_loc1"); it = inter(rec)
    rows["What"].append(w); rows["Where"].append(wh); rows["Joint"].append(j); rows["Interaction"].append(it)
    print(f"{d.split('__')[-1]:8s} {br:6.1f} {w:+7.1f} {wh:+7.1f} {j:+7.1f} {it:+7.1f}")
import statistics
print("\nacross draws (mean [min, max], sd):")
summ = {}
for k, vs in rows.items():
    print(f"  {k:12s} {statistics.mean(vs):+6.1f} [{min(vs):+.1f}, {max(vs):+.1f}], sd {statistics.pstdev(vs):.1f}")
    summ[k] = {"mean": round(statistics.mean(vs),1), "min": round(min(vs),1), "max": round(max(vs),1), "sd": round(statistics.pstdev(vs),1), "per_draw": [round(x,1) for x in vs]}
json.dump(summ, open(f"{ROOT}/generated/second_model/msfull_summary.json","w"), indent=2)
print("[saved msfull_summary.json]")
