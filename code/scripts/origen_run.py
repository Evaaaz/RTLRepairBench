#!/usr/bin/env python3
"""EXPLORATORY: evaluate the RTL-specific repair model OriGen_Fix (a LoRA on
deepseek-coder-7b-instruct-v1.5) on the semantic-bug set, local MPS inference.

NOTE ON FIT: OriGen_Fix is trained to fix SYNTAX/compile errors given a compiler
`error` field. Our bugs are compile-clean (no compiler error), so this is an
off-distribution test -- exactly the question "can an RTL syntax-repair model fix
semantic bugs?". We give it its native fix format and, on the location arms, an
error-pointer at the buggy line (its most favorable mode). Run under the venv
/tmp/origen_env. Same post hoc 68-case retained-harness scheduled cohort and
strict PDR-verdict scoring as the exploratory capability curve.
"""
import json, os, re, sys, time
from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))
from benchmarks.formal_protocol import rename_top_module

OUT = f"{ROOT}/generated/second_model/runs/origen_fix"
CR = f"{ROOT}/generated/formal/candidates"
ROWS = f"{ROOT}/generated/reports/vericodegen_full85_rows.jsonl"
INPUTS = f"{ROOT}/configs/vericodegen/main85_inputs.jsonl"
BASE_MODEL = "deepseek-ai/deepseek-coder-7b-instruct-v1.5"
LORA = "henryen/OriGen_Fix"
ARMS = [("spec0_loc0", False, False), ("spec1_loc0", True, False),
        ("spec0_loc1", False, True), ("spec1_loc1", True, True)]

SBY = ("[options]\nmode prove\ndepth 20\ntimeout 120\n[engines]\nabc pdr\n[script]\n"
       "read_verilog -formal -sv golden.sv candidate.sv equiv_top.sv\n"
       "setattr -unset always_comb p:*; select -clear\nprep -top equiv_top\n"
       "chformal -lower\n[files]\ngolden.sv\ncandidate.sv\nequiv_top.sv\n")


def build_prompt(spec, broken, loc, use_spec, use_loc):
    parts = ["You are an expert Verilog engineer. The following module compiles but "
             "fails functional verification. Return the complete corrected module "
             "(module ... endmodule), preserving the interface."]
    if use_spec:
        parts.append("\nSpecification:\n" + spec.strip())
    parts.append("\nBuggy module:\n```verilog\n" + broken.strip() + "\n```")
    if use_loc:
        parts.append(f"\nError: line {loc['line_number']} `{loc['broken_line'].strip()}` "
                     "is functionally incorrect.")
    else:
        parts.append("\nError: the module is functionally incorrect (no compile error).")
    parts.append("\nReturn only the corrected module.")
    return "\n".join(parts)


def extract(text):
    m = re.search(r"(module\b.*?endmodule)", text or "", re.S)
    return m.group(1).strip() if m else None


def reusable():
    rows = [json.loads(l) for l in open(ROWS)]
    by = {}
    for r in rows:
        if r["mode"] != "main":
            continue
        d = f"{CR}/{r['call_id']}/formal/pdr/src"
        if os.path.isfile(d + "/equiv_top.sv") and r["case_id"] not in by:
            by[r["case_id"]] = d
    return by


def main(limit=None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
    dev = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"loading {BASE_MODEL} + LoRA {LORA} on {dev} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype=torch.float16,
                                                 low_cpu_mem_usage=True)
    model = PeftModel.from_pretrained(model, LORA)
    model = model.to(dev).eval()
    print("model loaded", flush=True)

    inputs = {json.loads(l)["internal_metadata"]["anonymous_id"]: json.loads(l) for l in open(INPUTS)}
    by = reusable()
    cases = sorted(c for c in inputs if c in by)
    if limit:
        cases = cases[:limit]
    os.makedirs(OUT, exist_ok=True)
    results = []
    t0 = time.time()
    n = 0
    for cid in cases:
        mi = inputs[cid]["model_input"]; im = inputs[cid]["internal_metadata"]
        for arm, us, ul in ARMS:
            n += 1
            prompt = build_prompt(mi["full_spec"], mi["broken_rtl"], mi["oracle_location"], us, ul)
            msgs = [{"role": "user", "content": prompt}]
            enc = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True,
                                          return_dict=True)
            enc = {k: v.to(dev) for k, v in enc.items()}
            plen = enc["input_ids"].shape[1]
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=1024, do_sample=False,
                                     pad_token_id=tok.eos_token_id)
            text = tok.decode(out[0][plen:], skip_special_tokens=True)
            mod = extract(text)
            rec = {"case_id": cid, "anon_id": im["anonymous_id"], "mutation_family": im["mutation_family"],
                   "design": im["cluster_id"], "arm": arm, "parsed_ok": bool(mod)}
            if mod:
                d = f"{OUT}/work/{arm}__{cid}"; os.makedirs(d, exist_ok=True)
                renamed, _ = rename_top_module(mod, "design_candidate")
                open(f"{d}/candidate.sv", "w").write(renamed)
                for f in ("golden.sv", "equiv_top.sv"):
                    open(f"{d}/{f}", "w").write(open(f"{by[cid]}/{f}").read())
                open(f"{d}/run.sby", "w").write(SBY)
                rec["work"] = os.path.relpath(d, ROOT)
            results.append(rec)
            if n % 10 == 0:
                print(f"  {n}/{len(cases)*4}  ({(time.time()-t0)/n:.1f}s/gen)", flush=True)
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=0)
    ok = sum(1 for r in results if r["parsed_ok"])
    print(f"done: {len(results)} gens, {ok} parsed, {(time.time()-t0):.0f}s total", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
