#!/usr/bin/env python3
"""
EXPLORATORY (non-preregistered) second-model replication of the What x Where 2x2.

Runs a mid-capability model (default llama-3.3-70b-instruct) on the same frozen
85-case inputs and the same repair prompts as the main study, to test whether the
specification / oracle-location effects appear when the ceiling is not binding.
This is explicitly NOT part of the frozen 463-call ledger: it writes to a separate
output tree and never touches generated/logs/vericodegen2026_events.jsonl.

Serving: NVCF by default; set RTLREPAIR_LLM_URL to a /v1 root to drive any
OpenAI-compatible endpoint instead (the Modal vLLM apps under
benchmarks/backend/modal_inference/ are what the open-weight curve models use).
The credential is read only from RTLREPAIR_LLM_API_KEY / NVIDIA_API_KEY in the
environment and is never written to disk, logs, or results.

The per-case formal sources (golden.sv, equiv_top.sv) come from the frozen dirs
in generated/formal/candidates/ when present; when they are not -- they live only
on the retired run machine -- they are rebuilt through the audited
formal_verify._prepare_pair path, and each record says which was used.

  build   : construct prompts + API-call all arms, write candidate RTL + results
  verify  : (run inside the pinned container) PDR-check each candidate
  analyze : per-arm success + seed-macro effects
"""
import glob, json, os, re, sys, urllib.request, hashlib
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[4])
_CODE = Path(__file__).resolve().parent.parent
# benchmarks/ itself, not just code/: the formal modules import each other flatly
# (formal_verify does `from formal_data import ...`), so the package dir alone
# is not enough to import them.
for _p in (str(_CODE), str(_CODE / "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from benchmarks.formal_protocol import rename_top_module

MODEL = os.environ.get("EXPLORE_MODEL", "nvcf/meta/llama-3.3-70b-instruct")
TEMP = float(os.environ.get("EXPLORE_TEMP", "0"))
TAG = os.environ.get("EXPLORE_TAG", "")            # e.g. "s1" for a repeated sample
ARMS_ENV = os.environ.get("EXPLORE_ARMS", "")      # comma-sep subset, e.g. "spec0_loc0,spec0_loc1"

# Endpoint. The original curve ran against NVCF, which stays the default so the
# banked llama runs remain reproducible; any OpenAI-compatible server (Modal
# vLLM, in particular) is selected by setting RTLREPAIR_LLM_URL to its /v1 root.
_NVCF = "https://provider-a.invalid/v1/chat/completions"


def _chat_url() -> str:
    url = os.environ.get("RTLREPAIR_LLM_URL", "").strip().rstrip("/")
    if not url:
        return _NVCF
    return url if url.endswith("/chat/completions") else url + "/chat/completions"


BASE = _chat_url()
# per-model output dir so a capability curve can be built from several runs;
# the original llama-3.3-70b run stays at the top level for backward-compat.
_SLUG = MODEL.replace("/", "__") + (("__" + TAG) if TAG else "")
OUT = (f"{ROOT}/generated/second_model"
       if MODEL == "nvcf/meta/llama-3.3-70b-instruct" and not TAG and not ARMS_ENV
       else f"{ROOT}/generated/second_model/runs/{_SLUG}")
CR = f"{ROOT}/generated/formal/candidates"
ROWS = f"{ROOT}/generated/reports/vericodegen_full85_rows.jsonl"
INPUTS = f"{ROOT}/configs/vericodegen/main85_inputs.jsonl"

# The prompt grammar is frozen in benchmarks/prompts.py so this script and the
# ledger writer (benchmarks/generate_ledger.py) cannot drift; the text is
# unchanged from what the banked curve runs were generated with.
from benchmarks.prompts import (  # noqa: E402
    SPEC_LOCATE_ARMS as ARMS,
    SPEC_LOCATE_SYSTEM as REPAIR_SYSTEM,
    spec_locate_prompt as repair_prompt,
)


def call_model(system, user, max_tokens=4096):
    # NVIDIA_API_KEY stays the key for the NVCF default; RTLREPAIR_LLM_API_KEY is
    # the bearer for a self-served endpoint (the Modal apps' VLLM_API_KEY).
    key = (os.environ.get("RTLREPAIR_LLM_API_KEY", "").strip()
           or os.environ.get("NVIDIA_API_KEY", "").strip())
    if not key:
        raise SystemExit("set RTLREPAIR_LLM_API_KEY (served endpoint) or NVIDIA_API_KEY (NVCF)")
    body = json.dumps({"model": MODEL, "temperature": TEMP, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(BASE, data=body, method="POST", headers={
        "Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=180) as r:
        d = json.loads(r.read())
    ch = d["choices"][0]
    return ch["message"].get("content") or "", ch.get("finish_reason"), d.get("usage", {})


def extract_module(text):
    if not text:
        return None
    m = re.search(r"(module\b.*?endmodule)", text, re.S)
    return m.group(1).strip() if m else None


GOLDENS = os.environ.get("RTLREPAIR_SEEDS", f"{ROOT}/data/formal/canonical_goldens")


def reusable_dirs():
    """The frozen per-case formal src dirs from the original run machine.

    Empty when `generated/reports/vericodegen_full85_rows.jsonl` or
    `generated/formal/candidates/` are absent -- they are not in the repo, and
    the machine that wrote them is retired. Callers fall back to rebuilding.
    """
    if not os.path.isfile(ROWS):
        return {}
    rows = [json.loads(l) for l in open(ROWS)]
    bycase = {}
    for r in rows:
        if r["mode"] != "main":
            continue
        d = f"{CR}/{r['call_id']}/formal/pdr/src"
        if os.path.isfile(d + "/equiv_top.sv") and r["case_id"] not in bycase:
            bycase[r["case_id"]] = d
    return bycase


def banked_cases():
    """The formally-tractable case set, recovered from any banked curve run.

    The frozen dirs above were the *only* record of which cases carry a usable
    equivalence wrapper -- the "68" the paper quotes. Every banked run under
    generated/second_model/runs/ carries that same set in its results.json
    (verified: 13 runs, all 68, identical), so the set survives the loss of the
    machine. Set EXPLORE_CASES_JSON to pin an explicit list instead.
    """
    pin = os.environ.get("EXPLORE_CASES_JSON")
    if pin:
        return sorted(json.load(open(pin)))
    sets = []
    for p in sorted(glob.glob(f"{ROOT}/generated/second_model/runs/*/results.json")):
        try:
            sets.append({r["case_id"] for r in json.load(open(p)) if r.get("case_id")})
        except Exception:
            continue
    if not sets:
        raise SystemExit(
            "no case set available: stage generated/reports/vericodegen_full85_rows.jsonl\n"
            "+ generated/formal/candidates/, or point EXPLORE_CASES_JSON at a case-id list.")
    # Intersection, not union: a case missing from any banked run is one we
    # cannot show was tractable, and silently widening the set would change the
    # denominator the published curve is computed on.
    inter = set.intersection(*sets)
    if len(set.union(*sets)) != len(inter):
        print(f"warning: banked runs disagree on the case set "
              f"(union {len(set.union(*sets))} vs intersection {len(inter)}); using intersection")
    return sorted(inter)


def golden_for(case):
    """Canonical golden RTL for a case, keyed by its design (cluster_id)."""
    path = f"{GOLDENS}/{case['internal_metadata']['cluster_id']}.sv"
    return open(path).read() if os.path.isfile(path) else None


def build_formal_src(seed_id, golden, candidate):
    """Rebuild golden.sv / candidate.sv / equiv_top.sv for one case.

    Delegates to formal_verify._prepare_pair -- the same audited path the
    verifier itself uses -- so a rebuilt wrapper is byte-for-byte what the
    frozen one was: identical initialization contract, Prob151 normalization,
    top-module renaming, and wrapper emission. Reimplementing any of that here
    would fork the protocol the paper's verdicts rest on.
    """
    from benchmarks.formal_verify import _prepare_pair
    pair, reason = _prepare_pair(seed_id, golden, candidate)
    if pair is None:
        return None, reason
    return {"golden.sv": pair.golden_source,
            "candidate.sv": pair.candidate_source,
            "equiv_top.sv": pair.wrapper_source}, ""


def build(limit=None):
    inputs = {json.loads(l)["internal_metadata"]["anonymous_id"]: json.loads(l)
              for l in open(INPUTS)}
    bycase = reusable_dirs()
    if bycase:
        cases = sorted(c for c in inputs if c in bycase)
        print(f"formal src: reusing {len(cases)} frozen dirs under {CR}")
    else:
        cases = [c for c in banked_cases() if c in inputs]
        print(f"formal src: frozen dirs absent -- rebuilding from {GOLDENS} "
              f"for {len(cases)} cases recovered from banked runs")
    if limit:
        cases = cases[:limit]
    os.makedirs(OUT, exist_ok=True)
    # optional spec override (e.g. behavior-redacted specs for the leakage control)
    spec_override = None
    if os.environ.get("EXPLORE_SPEC_JSON"):
        spec_override = json.load(open(os.environ["EXPLORE_SPEC_JSON"]))
    jobs = []
    arm_filter = set(ARMS_ENV.split(",")) if ARMS_ENV else None
    for cid in cases:
        d = inputs[cid]; mi = d["model_input"]; im = d["internal_metadata"]
        spec_text = mi["full_spec"]
        if spec_override and cid in spec_override:
            spec_text = spec_override[cid]
        for arm, us, ul in ARMS:
            if arm_filter and arm not in arm_filter:
                continue
            jobs.append((cid, im["anonymous_id"], im["mutation_family"], im["cluster_id"],
                         arm, repair_prompt(mi["broken_rtl"], spec_text,
                                            mi["oracle_location"], us, ul), bycase.get(cid)))
    print(f"{len(cases)} cases x 4 arms = {len(jobs)} calls on {MODEL}")
    results = []

    def run(job):
        cid, aid, fam, design, arm, prompt, srcdir = job
        try:
            text, finish, usage = call_model(REPAIR_SYSTEM, prompt)
        except Exception as e:
            return {"case_id": cid, "arm": arm, "error": str(e)[:200]}
        mod = extract_module(text)
        rec = {"case_id": cid, "anon_id": aid, "mutation_family": fam, "design": design,
               "arm": arm, "finish_reason": finish, "usage": usage,
               "parsed_ok": bool(mod),
               "srcdir": os.path.relpath(srcdir, ROOT) if srcdir else None,
               "formal_src": "frozen" if srcdir else "rebuilt",  # provenance of the proof obligation
               "resp_sha": hashlib.sha256((text or "").encode()).hexdigest()[:16]}
        if mod:
            renamed, _ = rename_top_module(mod, "design_candidate")
            wd = f"{OUT}/work/{arm}__{cid}"
            os.makedirs(wd, exist_ok=True)
            if srcdir:
                open(f"{wd}/candidate.sv", "w").write(renamed)
                # reuse frozen golden + equiv_top for this design
                for f in ("golden.sv", "equiv_top.sv"):
                    open(f"{wd}/{f}", "w").write(open(f"{srcdir}/{f}").read())
            else:
                golden = golden_for(inputs[cid])
                if golden is None:
                    rec["error"] = f"no canonical golden for design {design}"
                    return rec
                src, why = build_formal_src(design, golden, mod)
                if src is None:
                    rec["error"] = f"formal prep rejected: {why}"
                    return rec
                for name, text_ in src.items():
                    open(f"{wd}/{name}", "w").write(text_)
            open(f"{wd}/run.sby", "w").write(
                "[options]\nmode prove\ndepth 20\ntimeout 120\n[engines]\nabc pdr\n"
                "[script]\nread_verilog -formal -sv golden.sv candidate.sv equiv_top.sv\n"
                "setattr -unset always_comb p:*; select -clear\nprep -top equiv_top\n"
                "chformal -lower\n[files]\ngolden.sv\ncandidate.sv\nequiv_top.sv\n")
            rec["work"] = os.path.relpath(wd, ROOT)
        return rec

    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = [ex.submit(run, j) for j in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            results.append(f.result())
            if i % 25 == 0:
                print(f"  {i}/{len(jobs)}")
    json.dump(results, open(f"{OUT}/results.json", "w"), indent=0)
    ok = sum(1 for r in results if r.get("parsed_ok"))
    err = sum(1 for r in results if r.get("error"))
    print(f"done: {len(results)} responses, {ok} parsed modules, {err} errors")


if __name__ == "__main__":
    cmd = sys.argv[1] if len(sys.argv) > 1 else "build"
    lim = int(sys.argv[2]) if len(sys.argv) > 2 else None
    if cmd == "build":
        build(lim)
