#!/usr/bin/env python3
"""Score a RepairBench eval-slot run on the honest no-spec arm: Tier F (unbounded
formal equivalence via sby/abc-pdr) with an independent differential-simulation
fallback where formal is inconclusive. Emits per-case verdicts + recovery over all 85
semantic cases (a missing/unparsed candidate counts as not-recovered). Needs the pinned
oss-cad-suite on PATH.

  python3 scripts/repairbench_slot_score.py <TAG>

Then compare two scored runs with repairbench_compare.py.
"""
import json, os, re, subprocess, sys
from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "datagen"))
from diff_testbench import parse_module, run_diff_test, _rename_module
INPUTS = f"{ROOT}/configs/vericodegen/main85_inputs.jsonl"


def sby_verdict(d):
    try:
        p = subprocess.run(["sby", "-f", "run.sby"], cwd=d, capture_output=True, text=True, timeout=180)
        out = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "INCONCLUSIVE"
    if re.search(r"DONE \(PASS", out) or "status: PASS" in out:
        return "PROVED"
    if re.search(r"DONE \(FAIL", out) or "status: FAIL" in out:
        return "DISPROVED"
    return "INCONCLUSIVE"


def sim_pass(d):
    g, c = f"{d}/golden.sv", f"{d}/candidate.sv"
    if not (os.path.isfile(g) and os.path.isfile(c)):
        return None
    golden, cand = open(g).read(), open(c).read()
    info = parse_module(golden)
    if info is None:
        return None
    ci = parse_module(cand)
    if ci and ci.name != info.name:
        cand = _rename_module(cand, ci.name, info.name)
    r = run_diff_test(golden, cand, info)
    return bool(r.compiled and not r.diverged)


def main(tag):
    OUT = f"{ROOT}/generated/repairbench_slot/{tag}"
    gen = {r["case_id"]: r for r in json.load(open(f"{OUT}/gen_results.json"))}
    meta = {json.loads(l)["internal_metadata"]["case_id"]: json.loads(l)["internal_metadata"]
            for l in open(INPUTS)}
    all_cases = sorted(meta)
    per = {}
    for cid in all_cases:
        g = gen.get(cid)
        if not g or not g.get("parsed_ok") or "work" not in g:
            per[cid] = {"tier": "none", "verdict": "NO_CANDIDATE", "recovered": False,
                        "cluster_id": meta[cid]["cluster_id"]}
            continue
        d = f"{ROOT}/{g['work']}"
        v = sby_verdict(d)
        if v == "PROVED":
            rec, tier = True, "F"
        elif v == "DISPROVED":
            rec, tier = False, "F"
        else:
            sp = sim_pass(d)                      # formal inconclusive -> sim fallback
            rec, tier = (bool(sp), "S")
        per[cid] = {"tier": tier, "verdict": v, "recovered": rec,
                    "cluster_id": meta[cid]["cluster_id"]}
        n = list(all_cases).index(cid) + 1
        if n % 10 == 0:
            print(f"  {n}/{len(all_cases)} scored", flush=True)
    n = len(all_cases)
    rec = sum(1 for c in per if per[c]["recovered"])
    ntf = sum(1 for c in per if per[c]["tier"] == "F")
    print(f"\n{tag}: no-spec recovery {rec}/{n} = {round(100*rec/n,1)}%  "
          f"(Tier-F on {ntf}/{n}, sim fallback on {n-ntf-sum(1 for c in per if per[c]['tier']=='none')})",
          flush=True)
    json.dump({"tag": tag, "n": n, "recovered": rec, "recovery_pct": round(100*rec/n, 1),
               "per_case": per}, open(f"{OUT}/score.json", "w"), indent=2)
    print(f"[saved {OUT}/score.json]")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "candidate_model")
