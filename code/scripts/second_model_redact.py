#!/usr/bin/env python3
"""EXPLORATORY leakage control: produce behavior-redacted specs with a capable
model + a fixed rubric, so a spec arm on the redacted prose measures how much of
the specification effect is answer disclosure. The redacted text remains in the
external run tree; the standalone release contains aggregate summaries only.
Not part of the frozen study; the frozen 'no LLM annotation' commitment is about
the cancelled preregistered redaction ARM, not this exploratory probe."""
import json, os, re, sys, urllib.request

from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
_SD = str(Path(__file__).resolve().parent)
sys.path.insert(0, _SD)
from eval_endpoint import chat_url, auth_headers
OUT = f"{ROOT}/generated/second_model/leakage"
REDACTOR = "azure/openai/gpt-4.1"

RUBRIC = ("You are redacting a hardware module specification for a controlled experiment. "
  "Rewrite the specification so it still states the module's PURPOSE and its INPUT/OUTPUT "
  "INTERFACE, but REMOVES or GENERALIZES every detail that reveals the exact correct logic: "
  "specific boolean operators, comparisons, equalities, polarities (active-high/low), literal "
  "constants and reset values, exact state-transition tables, truth-table rows, and any phrase "
  "that names the precise function (e.g. replace 'a NOR gate' or 'z = (x^y) & x' with 'the "
  "specified combinational function', replace a full state table with 'the state machine "
  "described by the following interface'). Keep it a plausible, well-formed spec a designer "
  "could receive, just without the giveaway details. Do NOT add new information or the answer. "
  "Return ONLY the redacted specification text, no preamble.")


def call(system, user, max_tokens=1200):
    body = json.dumps({"model": REDACTOR, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": system},
                                    {"role": "user", "content": user}]}).encode()
    req = urllib.request.Request(chat_url(), data=body, method="POST", headers=auth_headers())
    with urllib.request.urlopen(req, timeout=120) as r:
        d = json.loads(r.read())
    return d["choices"][0]["message"].get("content") or ""


def reusable():
    rows = [json.loads(l) for l in open(f"{ROOT}/generated/reports/vericodegen_full85_rows.jsonl")]
    bycase = set()
    for r in rows:
        if r["mode"] == "main" and os.path.isfile(
                f"{ROOT}/generated/formal/candidates/{r['call_id']}/formal/pdr/src/equiv_top.sv"):
            bycase.add(r["case_id"])
    return bycase


def run(limit=None):
    os.makedirs(OUT, exist_ok=True)
    inputs = {json.loads(l)["internal_metadata"]["anonymous_id"]: json.loads(l)
              for l in open(f"{ROOT}/configs/vericodegen/main85_inputs.jsonl")}
    keep = reusable()
    cases = sorted(c for c in inputs if c in keep)
    if limit:
        cases = cases[:limit]
    red = {}
    for i, cid in enumerate(cases, 1):
        spec = inputs[cid]["model_input"]["full_spec"]
        try:
            red[cid] = call(RUBRIC, "Specification to redact:\n\n" + spec.strip())
        except Exception as e:
            red[cid] = f"[REDACTION_ERROR: {e}]"
        if i % 20 == 0:
            print(f"  redacted {i}/{len(cases)}")
    json.dump(red, open(f"{OUT}/redacted_specs.json", "w"), indent=2)
    print(f"wrote {len(red)} redacted specs")
    # spot check: print 3
    for cid in cases[:3]:
        print(f"\n=== {cid} [{inputs[cid]['internal_metadata']['mutation_family']}] ===")
        print("ORIG:", " ".join(inputs[cid]["model_input"]["full_spec"].split())[-260:])
        print("RED :", " ".join(red[cid].split())[:300])


if __name__ == "__main__":
    run(int(sys.argv[1]) if len(sys.argv) > 1 else None)
