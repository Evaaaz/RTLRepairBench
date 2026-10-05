#!/usr/bin/env python3
"""RepairBench eval slot (preregistered honest-repair evaluation).

Drop-in evaluation for a candidate repair model on the HONEST target: the no-spec,
anonymized semantic-repair arm, scored by the formal oracle (Tier F, unbounded
equivalence) with an independent-simulation fallback. This is the arm our paper
argues is the only unconfounded one -- repair from the broken RTL plus the tool
diagnostic, with NO specification (so a gain is repair, not spec-regeneration) and
anonymized prompts (no problem-identity leakage).

Two backends, same protocol:
  * API   :  SLOT_BACKEND=api   SLOT_MODEL=<model-id served by the endpoint>
             any OpenAI-compatible server -- the internal NVIDIA API by default, or a
             self-hosted vLLM / Modal deployment via EVAL_BASE_URL (eval_endpoint.py):
               EVAL_BASE_URL=https://<user>--rtlrepair.modal.run \
               SLOT_BACKEND=api SLOT_MODEL=<served-name> SLOT_TAG=candidate \
               python3 code/scripts/repairbench_slot.py
  * local :  SLOT_BACKEND=local SLOT_BASE=<hf-base>  SLOT_LORA=<hf-or-path lora>

Generation only; scoring is repairbench_slot_score.py. Candidate RTL is renamed to
`design_candidate` and dropped into a per-case work dir alongside the frozen
golden.sv / equiv_top.sv and a run.sby, so the scorer is model-agnostic.

Credential (API mode) read only from EVAL_API_KEY / NVIDIA_API_KEY; never written
anywhere.
"""
import json, os, re, sys, urllib.request
from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))
_SD = str(Path(__file__).resolve().parent)
sys.path.insert(0, _SD)
from eval_endpoint import chat_url, auth_headers
from benchmarks.formal_protocol import rename_top_module

INPUTS = f"{ROOT}/configs/vericodegen/main85_inputs.jsonl"
ROWS = f"{ROOT}/generated/reports/vericodegen_full85_rows.jsonl"
CR = f"{ROOT}/generated/formal/candidates"
TAG = os.environ.get("SLOT_TAG", "candidate_model")
OUT = f"{ROOT}/generated/repairbench_slot/{TAG}"
BACKEND = os.environ.get("SLOT_BACKEND", "api")

# Frozen proof protocol (matches contract_robustness_experiment.py / the reproduced 40/40).
SBY = """[options]
mode prove
depth 20
timeout 120
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

SYSTEM = ("You are an expert Verilog engineer. The module compiles but fails functional "
          "verification. Return the complete corrected module (module ... endmodule), "
          "preserving the interface. Return only the module.")


def no_spec_prompt(broken):
    # HONEST arm: broken RTL + tool diagnostic only. No specification. No oracle line.
    return ("The following SystemVerilog module compiles and lints clean but is "
            "functionally incorrect (it diverges from the intended behavior in "
            "simulation). Fix the functional bug.\n\n```systemverilog\n"
            + broken.strip() + "\n```\n\nReturn only the corrected module.")


def extract(text):
    m = re.search(r"(module\b.*?endmodule)", text or "", re.S)
    return m.group(1).strip() if m else None


def case_reuse_map():
    """anonymous_id (rows use it as case_id, e.g. main_0004) -> dir with reusable
    golden.sv + equiv_top.sv (identical across arms)."""
    rows = [json.loads(l) for l in open(ROWS)]
    by = {}
    for r in rows:
        if r["mode"] != "main":
            continue
        d = f"{CR}/{r['call_id']}/formal/pdr/src"
        if r["case_id"] not in by and os.path.isfile(f"{d}/equiv_top.sv") and os.path.isfile(f"{d}/golden.sv"):
            by[r["case_id"]] = d           # r["case_id"] here is the anonymous_id (main_XXXX)
    return by


# ---- backends -------------------------------------------------------------
def api_call(prompt, max_tokens=2048):
    body = json.dumps({"model": os.environ["SLOT_MODEL"], "temperature": 0,
                       "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": SYSTEM},
                                    {"role": "user", "content": prompt}]}).encode()
    req = urllib.request.Request(chat_url(), data=body, headers=auth_headers())
    with urllib.request.urlopen(req, timeout=180) as r:
        ch = json.load(r)["choices"][0]["message"]
    return ch.get("content") or ch.get("reasoning_content") or ""


def local_loader():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    base = os.environ["SLOT_BASE"]
    dev = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(base)
    model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, low_cpu_mem_usage=True)
    lora = os.environ.get("SLOT_LORA")
    if lora:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, lora)
    model = model.to(dev).eval()

    def gen(prompt, max_tokens=1024):
        msgs = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]
        enc = tok.apply_chat_template(msgs, return_tensors="pt", add_generation_prompt=True, return_dict=True)
        enc = {k: v.to(dev) for k, v in enc.items()}
        plen = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=max_tokens, do_sample=False, pad_token_id=tok.eos_token_id)
        return tok.decode(out[0][plen:], skip_special_tokens=True)
    return gen


def main(limit=None):
    inputs = {json.loads(l)["internal_metadata"]["anonymous_id"]: json.loads(l)
              for l in open(INPUTS)}
    reuse = case_reuse_map()               # keyed by anonymous_id (main_XXXX)
    cases = [aid for aid in inputs if aid in reuse]
    cases.sort()
    if limit:
        cases = cases[:limit]
    gen = local_loader() if BACKEND == "local" else (lambda p: api_call(p))
    os.makedirs(OUT, exist_ok=True)
    results = []
    for i, aid in enumerate(cases):
        rec = inputs[aid]; mi = rec["model_input"]; im = rec["internal_metadata"]
        cid = im["case_id"]
        try:
            mod = extract(gen(no_spec_prompt(mi["broken_rtl"])))
        except Exception as e:
            results.append({"case_id": cid, "anon_id": aid, "parsed_ok": False, "error": str(e)[:120]})
            continue
        r = {"case_id": cid, "anon_id": aid, "cluster_id": im["cluster_id"],
             "mutation_family": im["mutation_family"], "parsed_ok": bool(mod)}
        if mod:
            d = f"{OUT}/work/{cid}"; os.makedirs(d, exist_ok=True)
            renamed, _ = rename_top_module(mod, "design_candidate")
            open(f"{d}/candidate.sv", "w").write(renamed)
            for f in ("golden.sv", "equiv_top.sv"):
                open(f"{d}/{f}", "w").write(open(f"{reuse[aid]}/{f}").read())
            open(f"{d}/run.sby", "w").write(SBY)
            r["work"] = os.path.relpath(d, ROOT)
        results.append(r)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{len(cases)}  parsed {sum(1 for x in results if x['parsed_ok'])}", flush=True)
    json.dump(results, open(f"{OUT}/gen_results.json", "w"), indent=0)
    ok = sum(1 for r in results if r["parsed_ok"])
    print(f"done: {len(results)} cases, {ok} parsed -> {OUT}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
