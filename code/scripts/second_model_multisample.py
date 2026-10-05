#!/usr/bin/env python3
"""Location-effect stability across draws for llama-3.3-70b:
1 temperature-0 draw (main run) + 4 temperature-0.7 draws. Reports the seed-macro
location risk difference per draw to show the +21pp effect is not a single-draw fluke."""
import json, os, random
from collections import defaultdict

from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])

def load(base):
    rp, vp = f"{base}/results.json", f"{base}/verdicts.jsonl"
    if not (os.path.isfile(rp) and os.path.isfile(vp)):
        return None
    res = json.load(open(rp))
    verd = {json.loads(l)["dir"]: json.loads(l)["verdict"] for l in open(vp)}
    rec = {}
    for r in res:
        if r["arm"] not in ("spec0_loc0", "spec0_loc1"):
            continue
        v = verd.get(f"{r['arm']}__{r['case_id']}", "MISS") if r.get("parsed_ok") else "NO_PARSE"
        rec[(r["case_id"], r["arm"])] = {"success": v == "PROVED", "design": r["design"]}
    return rec

def loc_rd(rec, seed=7):
    cases = {c for c, a in rec if a == "spec0_loc0"} & {c for c, a in rec if a == "spec0_loc1"}
    g = defaultdict(list)
    for c in cases:
        a, b = rec[(c, "spec0_loc0")], rec[(c, "spec0_loc1")]
        g[a["design"]].append(int(b["success"]) - int(a["success"]))
    ks = sorted(g); sm = {k: sum(g[k]) / len(g[k]) for k in ks}
    pt = sum(sm.values()) / len(sm)
    base_rate = sum(1 for c in cases if rec[(c, "spec0_loc0")]["success"]) / len(cases) * 100
    return pt * 100, base_rate, len(cases)

draws = [("temp0 (main)", f"{ROOT}/generated/second_model")]
rd = f"{ROOT}/generated/second_model/runs"
for slug in sorted(os.listdir(rd)):
    if "temp07" in slug:
        draws.append((slug.split("__")[-1], f"{rd}/{slug}"))

print(f"{'draw':16s} {'baseline%':>9} {'location RD':>12} {'n':>4}")
rds = []
for name, base in draws:
    rec = load(base)
    if not rec:
        print(f"  {name}: (not verified yet)"); continue
    pt, br, n = loc_rd(rec)
    rds.append(pt)
    print(f"{name:16s} {br:9.1f} {pt:+11.1f} {n:>4}")
if len(rds) >= 2:
    import statistics
    print(f"\nlocation effect across {len(rds)} draws: mean {statistics.mean(rds):+.1f}pp, "
          f"range [{min(rds):+.1f}, {max(rds):+.1f}], sd {statistics.pstdev(rds):.1f}")
    json.dump({"draws": len(rds), "location_rd_per_draw": [round(x,1) for x in rds],
               "mean": round(statistics.mean(rds),1), "min": round(min(rds),1), "max": round(max(rds),1)},
              open(f"{ROOT}/generated/second_model/multisample_location.json","w"), indent=2)
    print("[saved multisample_location.json]")
