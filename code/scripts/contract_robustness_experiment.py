#!/usr/bin/env python3
"""
Contract-robustness re-verification of the frozen VeriCodeGen candidate RTL.

This is a container-only audit (pinned formal image, NO model API calls, the
463-call ledger stays frozen). It supports the soundness argument in Section 8 of
the paper by testing whether the frozen reset/initialization contracts could have
produced a spurious PROVED verdict.

Three checks, all on the frozen candidate/golden RTL under generated/formal/candidates/:
  1. Reproducibility: re-prove every accepted PROVED candidate whose proof source
     was retained (244 of 306) under the exact frozen contract. Expect 244/244 PASS.
  2. Weakened-reset probe: for the 119 whose contract releases a synchronous reset,
     re-verify under a strictly weaker contract that keeps the 2-cycle reset
     synchronization but leaves `reset` free (anyseq) thereafter, so any legal
     re-assertion of reset must also preserve equivalence. A flip to FAIL would
     reveal that the frozen "reset-low-forever" assumption masked a reachable
     divergence. Result: 119/119 PASS, zero flips.
  3. Non-vacuity: re-verify every retained COUNTEREXAMPLE candidate and report
     the observed verdict counts without substituting a hard-coded result.

Usage:
  # 1. build SBY work dirs from the frozen candidate RTL
  python3 scripts/contract_robustness_experiment.py build
  # 2. run all jobs inside the pinned container
  bash code/scripts/run_formal_container.sh  # (or: docker run ... run_batch.sh)
  # 3. summarize
  python3 scripts/contract_robustness_experiment.py analyze
"""
import json, os, re, shutil, sys

from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
CR = f"{ROOT}/generated/formal/candidates"
TAG = os.environ.get("ROBUSTNESS_TAG", "audit_v1")
WORK = f"{ROOT}/generated/formal/robustness/{TAG}"
ROWS = f"{ROOT}/generated/reports/vericodegen_full85_rows.jsonl"

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
# strictly-weaker contract: drop the "else assume(reset==0)" reset-low-forever clause
RELEASE = re.compile(r"[ \t]*else\s+assume\s*\(\s*reset\s*==\s*1'b0\s*\)\s*;\s*\n")


def _src(cid):
    return f"{CR}/{cid}/formal/pdr/src"


def build():
    rows = [
        json.loads(line)
        for line in Path(ROWS).read_text(encoding="utf-8").splitlines()
        if line
    ]
    main = [r for r in rows if r["mode"] == "main"]
    if len(main) != 340 or len({r["call_id"] for r in main}) != 340:
        raise RuntimeError(
            f"contract audit requires 340 unique primary cells; found {len(main)}"
        )
    if sum(r["formal_status"] == "PROVED" for r in main) != 306:
        raise RuntimeError("contract audit requires the frozen 306-PROVED inventory")
    if os.path.exists(WORK):
        raise RuntimeError(f"refusing to overwrite an existing immutable audit: {WORK}")
    os.makedirs(WORK)
    manifest = []

    def mk(r, variant, etop):
        cid = r["call_id"]
        d = f"{WORK}/{variant}__{cid}"
        os.makedirs(d, exist_ok=True)
        shutil.copy(f"{_src(cid)}/golden.sv", f"{d}/golden.sv")
        shutil.copy(f"{_src(cid)}/candidate.sv", f"{d}/candidate.sv")
        Path(d, "equiv_top.sv").write_text(etop, encoding="utf-8")
        Path(d, "run.sby").write_text(SBY, encoding="utf-8")
        manifest.append({"dir": os.path.relpath(d, ROOT), "variant": variant,
                         "call_id": cid, "case_id": r["case_id"], "arm": r["arm"],
                         "frozen_status": r["formal_status"]})

    n_weak = 0
    proved = [
        r
        for r in main
        if r["formal_status"] == "PROVED"
        and all(
            os.path.isfile(f"{_src(r['call_id'])}/{name}")
            for name in ("golden.sv", "candidate.sv", "equiv_top.sv")
        )
    ]
    ce = [
        r
        for r in main
        if r["formal_status"] == "COUNTEREXAMPLE"
        and all(
            os.path.isfile(f"{_src(r['call_id'])}/{name}")
            for name in ("golden.sv", "candidate.sv", "equiv_top.sv")
        )
    ]
    if len(proved) != 244 or len(ce) != 5:
        raise RuntimeError(
            "contract audit requires exactly 244 retained PROVED sources and "
            f"5 retained COUNTEREXAMPLE sources; found {len(proved)} and {len(ce)}"
        )
    for r in proved:
        etop = Path(f"{_src(r['call_id'])}/equiv_top.sv").read_text(encoding="utf-8")
        mk(r, "repro", etop)
        if RELEASE.search(etop):
            mk(r, "weak", RELEASE.sub("", etop))
            n_weak += 1
    for r in ce:
        mk(
            r,
            "repro_ce",
            Path(f"{_src(r['call_id'])}/equiv_top.sv").read_text(encoding="utf-8"),
        )
    if n_weak != 119 or len(manifest) != 368:
        raise RuntimeError(
            f"contract audit requires 119 weakened jobs and 368 total jobs; "
            f"found {n_weak} and {len(manifest)}"
        )
    _write_json_atomic(f"{WORK}/manifest.json", manifest)
    print(f"built repro={len(proved)}, weakened={n_weak}, ce={len(ce)}, total={len(manifest)}")


def _write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    tmp.replace(path)


def analyze():
    from collections import Counter
    summary_path = Path(WORK) / "SUMMARY.json"
    if summary_path.exists():
        raise RuntimeError(f"refusing to overwrite frozen summary: {summary_path}")
    manifest = json.loads(Path(WORK, "manifest.json").read_text(encoding="utf-8"))
    res = [
        json.loads(line)
        for line in Path(WORK, "results.jsonl").read_text(encoding="utf-8").splitlines()
        if line
    ]
    expected = {(x["variant"], x["call_id"]) for x in manifest}
    observed = [(x.get("variant"), x.get("call_id")) for x in res]
    if len(manifest) != 368 or len(expected) != 368:
        raise RuntimeError("contract manifest must contain 368 unique jobs")
    if len(observed) != len(set(observed)):
        raise RuntimeError("contract results contain duplicate jobs")
    missing = expected - set(observed)
    extra = set(observed) - expected
    if missing or extra:
        raise RuntimeError(
            f"contract result roster mismatch: missing={len(missing)}, extra={len(extra)}"
        )
    if any(x.get("verdict") not in {"PASS", "FAIL"} for x in res):
        raise RuntimeError("contract results contain a nonterminal or tool-error verdict")
    c = Counter(f"{x['variant']}:{x['verdict']}" for x in res)
    weak_flips = [x for x in res if x["variant"] == "weak" and x["verdict"] != "PASS"]
    counterexample_rechecks = {
        key.split(":", 1)[1]: count
        for key, count in sorted(c.items())
        if key.startswith("repro_ce:")
    }
    if (
        c.get("repro:PASS", 0) != 244
        or c.get("weak:PASS", 0) != 119
        or len(weak_flips) != 0
        or c.get("repro_ce:FAIL", 0) != 5
    ):
        raise RuntimeError(f"contract audit acceptance gate failed: {dict(c)}")
    summary = {
        "experiment": "contract_robustness_reverification",
        "repro_proved_pass": c.get("repro:PASS", 0),
        "weakened_pass": c.get("weak:PASS", 0),
        "weakened_flips_to_cex": len(weak_flips),
        "counterexample_rechecks_by_verdict": counterexample_rechecks,
        "weakening": "removed assume(reset==0) release; reset free (anyseq) after the frozen 2-cycle sync",
        "engine": "abc pdr (unbounded)", "image": "rtlrepair-formal:20260508",
    }
    _write_json_atomic(summary_path, summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    {"build": build, "analyze": analyze}.get(sys.argv[1] if len(sys.argv) > 1 else "analyze", analyze)()
