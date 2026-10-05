#!/usr/bin/env python3
"""Larger-scale probe, formal coverage: attempt unbounded SEC (golden vs the injected
mutant) on each RTLLM-scale design. A definitive verdict (DISPROVED -- the proof finds the
injected bug) means the formal oracle is applicable; parse/contract/timeout failures are
coverage gaps. Reports formal coverage on larger designs vs the 68/85 (80%) on small ones.
Needs yosys/sby/abc on PATH.
"""
import json, os, re, subprocess, sys
from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "datagen"))
from diff_testbench import parse_module
from benchmarks.formal_protocol import initialization_contract, generate_equivalence_wrapper, rename_top_module
WORK = (os.environ.get("RTLREPAIR_SCRATCH") or f"{ROOT}/generated/largescale/scratch") + "/ls_formal"
SBY = """[options]
mode prove
depth 20
timeout 90
[engines]
abc pdr
[script]
read_verilog -formal -sv golden.sv candidate.sv equiv_top.sv
setattr -unset always_comb p:*; select -clear
prep -top equiv_top
chformal -lower
[files]
golden.sv
candidate.sv
equiv_top.sv
"""


def verdict(d):
    try:
        p = subprocess.run(["sby", "-f", "run.sby"], cwd=d, capture_output=True, text=True, timeout=140)
        o = p.stdout + p.stderr
    except subprocess.TimeoutExpired:
        return "TIMEOUT"
    if re.search(r"DONE \(FAIL", o) or "status: FAIL" in o:
        return "DISPROVED"      # proof found the injected bug -> formal applicable
    if re.search(r"DONE \(PASS", o) or "status: PASS" in o:
        return "PROVED"
    return "INCONCLUSIVE"


def main():
    bugs = [json.loads(l) for l in open(f"{ROOT}/generated/largescale/bugs.jsonl")]
    os.makedirs(WORK, exist_ok=True)
    counts = {}
    rows = []
    for i, b in enumerate(bugs):
        name = f"{b['design']}_{b['mutation']}_{i}"
        try:
            info = parse_module(b["golden_rtl"])
            if info is None:
                v = "NO_PARSE"
            else:
                contract = initialization_contract(b["design"], b["golden_rtl"])
                g, _ = rename_top_module(b["golden_rtl"], "design_golden")
                c, _ = rename_top_module(b["broken_rtl"], "design_candidate")
                etop = generate_equivalence_wrapper(info, contract, seed_id=b["design"])
                d = f"{WORK}/{name}"
                os.makedirs(d, exist_ok=True)
                open(f"{d}/golden.sv", "w").write(g)
                open(f"{d}/candidate.sv", "w").write(c)
                open(f"{d}/equiv_top.sv", "w").write(etop)
                open(f"{d}/run.sby", "w").write(SBY)
                v = verdict(d)
        except Exception as e:
            v = "WRAP_FAIL:" + type(e).__name__
        counts[v.split(':')[0]] = counts.get(v.split(':')[0], 0) + 1
        rows.append({"design": b["design"], "lines": b["lines"], "verdict": v})
        print(f"  {name:34s} {b['lines']:3d}L -> {v}", flush=True)
    defin = counts.get("DISPROVED", 0) + counts.get("PROVED", 0)
    n = len(bugs)
    out = {"n": n, "definitive": defin, "coverage_pct": round(100 * defin / n, 1), "counts": counts, "rows": rows}
    json.dump(out, open(f"{ROOT}/generated/largescale/formal_coverage.json", "w"), indent=2)
    print(f"\nFORMAL COVERAGE on larger designs: {defin}/{n} = {out['coverage_pct']}% definitive "
          f"(vs 68/85=80% on HDLBits-scale). counts={counts}")


if __name__ == "__main__":
    main()
