#!/usr/bin/env python3
"""Larger-scale transfer probe, step 1: build a semantic-bug set on RTLLM designs
(bigger, more realistic than the HDLBits-scale seeds). For each design, inject semantic
mutations, keep only those that a differential simulation confirms diverge, and attach
the design's natural-language description (for the spec arm). Needs iverilog on PATH.
"""
import json, os, sys, urllib.request, urllib.parse
from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "datagen"))
from inject_bugs import inject_semantic
from diff_testbench import parse_module, run_diff_test, _rename_module
SC = os.environ.get("RTLREPAIR_SCRATCH") or f"{ROOT}/generated/largescale/scratch"
RTLLM = f"{SC}/rtllm"

# name -> RTLLM folder (for the design_description.txt)
FOLDER = {
 "adder_64bit": "Arithmetic/Adder/adder_pipe_64bit",
 "div_16bit": "Arithmetic/Divider/div_16bit",
 "radix2_div": "Arithmetic/Divider/radix2_div",
 "multi_16bit": "Arithmetic/Multiplier/multi_16bit",
 "fixed_point_adder": "Arithmetic/Other/fixed_point_adder",
 "JC_counter": "Control/Counter/JC_counter",
 "up_down_counter": "Control/Counter/up_down_counter",
 "sequence_detector": "Control/Finite State Machine/sequence_detector",
 "LIFObuffer": "Memory/LIFO/LIFObuffer",
 "edge_detect": "Miscellaneous/Others/edge_detect",
 "parallel2serial": "Miscellaneous/Others/parallel2serial",
 "pulse_detect": "Miscellaneous/Others/pulse_detect",
 "serial2parallel": "Miscellaneous/Others/serial2parallel",
}


def fetch_desc(folder):
    for fn in ("design_description.txt", "design_description.md"):
        url = "https://raw.githubusercontent.com/hkust-zhiyao/RTLLM/main/" + urllib.parse.quote(f"{folder}/{fn}")
        try:
            return urllib.request.urlopen(url, timeout=20).read().decode("utf-8", "replace").strip()
        except Exception:
            continue
    return ""


def validated(golden, mutant):
    """True if mutant compiles and diverges from golden under differential sim."""
    info = parse_module(golden)
    if info is None:
        return False
    m = mutant
    ci = parse_module(mutant)
    if ci and ci.name != info.name:
        m = _rename_module(mutant, ci.name, info.name)
    r = run_diff_test(golden, m, info)
    return r.compiled and r.diverged


def main(per_design=2):
    bugs = []
    for name, folder in FOLDER.items():
        gp = f"{RTLLM}/{name}.v"
        if not os.path.isfile(gp):
            continue
        golden = open(gp).read()
        desc = fetch_desc(folder)
        # sanity: golden vs golden must NOT diverge
        info = parse_module(golden)
        if info is None or run_diff_test(golden, golden, info).diverged:
            print(f"  [skip {name}: golden self-diverges or unparsable]"); continue
        kept = 0
        for mut_name, mutant in inject_semantic(golden, seed=7):
            if kept >= per_design:
                break
            if validated(golden, mutant):
                bugs.append({"design": name, "folder": folder, "mutation": mut_name,
                             "lines": len(golden.splitlines()),
                             "golden_rtl": golden, "broken_rtl": mutant, "spec": desc})
                kept += 1
        print(f"  {name:20s} {len(golden.splitlines()):3d}L  desc={len(desc):4d}ch  validated_bugs={kept}", flush=True)
    os.makedirs(f"{ROOT}/generated/largescale", exist_ok=True)
    with open(f"{ROOT}/generated/largescale/bugs.jsonl", "w") as f:
        for b in bugs:
            f.write(json.dumps(b) + "\n")
    print(f"\ntotal larger-scale semantic bugs: {len(bugs)} over {len(set(b['design'] for b in bugs))} designs "
          f"(median {sorted(b['lines'] for b in bugs)[len(bugs)//2]} lines)")


if __name__ == "__main__":
    main()
