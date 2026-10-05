#!/usr/bin/env python3
"""Exploratory SECOND-anchor decomposition: repeat the What/Where 2x2 on
a second frontier model (Claude Opus 4.8) to probe whether the gpt-5.5 primary result is
single-model. Four arms over the post hoc 68-case retained-harness cohort (spec x location), scored by differential
simulation (sim-scored; the union analysis showed 81-99% sim/formal agreement, 100% on the
frozen gpt-5.5 primary). Reports seed-macro What and Where risk differences with a cluster
bootstrap CI. Generation + scoring; needs iverilog on PATH; endpoint/credential from
EVAL_BASE_URL / EVAL_API_KEY (see eval_endpoint.py).
"""
import json, os, re, sys, time, urllib.request, urllib.error, concurrent.futures, random, hashlib
from collections import defaultdict
from pathlib import Path
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "datagen"))
_SD = str(Path(__file__).resolve().parent)
sys.path.insert(0, _SD)
from eval_endpoint import chat_url, auth_headers
from diff_testbench import (
    SimulatorProtocolError,
    SimulatorUnavailableError,
    _rename_module,
    parse_module,
    require_simulator,
    run_diff_test,
)
INPUTS = f"{ROOT}/configs/vericodegen/main85_inputs.jsonl"
COHORT_MANIFEST = f"{ROOT}/configs/vericodegen/curve68_manifest.json"
ROWS = f"{ROOT}/generated/reports/vericodegen_full85_rows.jsonl"
CR = f"{ROOT}/generated/formal/candidates"
MODEL = os.environ.get("DEC_MODEL", "azure/anthropic/claude-opus-4-8")
TAG = os.environ.get("DEC_TAG", "claude_opus_4_8_decomp")
OUT = f"{ROOT}/generated/second_anchor/{TAG}"
ARMS = [("spec0_loc0", False, False), ("spec1_loc0", True, False),
        ("spec0_loc1", False, True), ("spec1_loc1", True, True)]


def call(user, max_tokens=2048):
    body = json.dumps({"model": MODEL, "temperature": 0, "max_tokens": max_tokens,
                       "messages": [{"role": "system", "content": "You repair broken SystemVerilog. Return only the corrected module."},
                                    {"role": "user", "content": user}]}).encode()
    for attempt in range(6):
        try:
            req = urllib.request.Request(chat_url(), data=body, headers=auth_headers())
            with urllib.request.urlopen(req, timeout=180) as r:
                ch = json.load(r)["choices"][0]["message"]
            return ch.get("content") or ch.get("reasoning_content") or ""
        except urllib.error.HTTPError as e:
            if e.code == 429 and attempt < 5:
                time.sleep(2 ** attempt); continue
            raise


def extract(t):
    m = re.search(r"(module\b.*?endmodule)", t or "", re.S)
    return m.group(1).strip() if m else None


def prompt(mi, use_spec, use_loc):
    p = ["The following SystemVerilog module compiles and lints clean but is functionally incorrect. Fix the functional bug.\n\n```systemverilog\n" + mi["broken_rtl"].strip() + "\n```"]
    if use_spec:
        p.append("\nSpecification:\n" + mi["full_spec"].strip())
    if use_loc:
        loc = mi["oracle_location"]
        p.append(f"\nThe bug is on line {loc['line_number']}: `{loc['broken_line'].strip()}` is functionally incorrect.")
    p.append("\nReturn only the corrected module.")
    return "\n".join(p)


def golden_map():
    by = {}
    sha_by = {}
    for l in open(ROWS):
        r = json.loads(l)
        if r["mode"] == "main":
            g = f"{CR}/{r['call_id']}/formal/pdr/src/golden.sv"
            if os.path.isfile(g):
                digest = hashlib.sha256(Path(g).read_bytes()).hexdigest()
                previous = sha_by.get(r["case_id"])
                if previous is not None and previous != digest:
                    raise RuntimeError(
                        f"inconsistent retained goldens for case {r['case_id']}"
                    )
                sha_by[r["case_id"]] = digest
                by.setdefault(r["case_id"], g)
    return by


def load_cohort():
    manifest = json.loads(Path(COHORT_MANIFEST).read_text(encoding="utf-8"))
    case_ids = manifest.get("case_ids")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("count") != 68
        or not isinstance(case_ids, list)
        or len(case_ids) != 68
        or len(set(case_ids)) != 68
        or case_ids != sorted(case_ids)
    ):
        raise RuntimeError("curve68 cohort manifest is malformed")
    digest = hashlib.sha256(
        json.dumps(case_ids, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != manifest.get("case_ids_sha256"):
        raise RuntimeError("curve68 cohort manifest hash mismatch")
    return case_ids, digest


def sim_ok(golden, cand):
    info = parse_module(golden)
    if info is None:
        raise RuntimeError("retained golden is not parseable by the scoring harness")
    if not cand:
        return False
    ci = parse_module(cand)
    if ci is None:
        return False
    if ci.name != info.name:
        cand = _rename_module(cand, ci.name, info.name)
    r = run_diff_test(golden, cand, info)
    if not r.compiled:
        if r.failure_kind == "compile_failure":
            return False
        raise RuntimeError(
            f"differential scorer could not classify candidate: {r.failure_kind}"
        )
    return not r.diverged


def preflight_goldens(cases, golden_paths):
    """Require every retained reference to self-compare before endpoint spend."""
    sources = {}
    for aid in cases:
        source = Path(golden_paths[aid]).read_text(encoding="utf-8")
        info = parse_module(source)
        if info is None:
            raise RuntimeError(f"golden preflight parse failure: {aid}")
        result = run_diff_test(source, source, info)
        if not result.compiled or result.diverged:
            raise RuntimeError(
                f"golden preflight differential failure: {aid} "
                f"({result.failure_kind or 'unexpected divergence'})"
            )
        sources[aid] = source
    return sources


def one(job):
    aid, arm, us, ul, mi, im = job
    try:
        response = call(prompt(mi, us, ul))
    except Exception as e:
        return (aid, arm, None, None, f"{type(e).__name__}: {e}"[:500])
    return (
        aid,
        arm,
        extract(response),
        hashlib.sha256(response.encode()).hexdigest(),
        None,
    )


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def write_jsonl_atomic(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    tmp.replace(path)


def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)


def seed_macro(rec, a, b, cid_of, seed=42):
    g = defaultdict(list)
    for aid in rec:
        if a in rec[aid] and b in rec[aid] and rec[aid][a] is not None and rec[aid][b] is not None:
            g[cid_of[aid]].append(int(rec[aid][b]) - int(rec[aid][a]))
    ks = sorted(g); sm = {k: sum(g[k]) / len(g[k]) for k in ks}
    pt = sum(sm.values()) / len(sm)
    rng = random.Random(seed)
    draws = sorted(sum(sm[ks[rng.randrange(len(ks))]] for _ in ks) / len(ks) for _ in range(10000))
    return round(pt * 100, 1), round(draws[250] * 100, 1), round(draws[9750] * 100, 1)


def main():
    # Do not spend endpoint calls when the scientific oracle is unavailable.
    require_simulator()
    out_path = Path(OUT)
    if out_path.exists() and any(out_path.iterdir()):
        raise RuntimeError(
            f"refusing to mix a new run with existing output: {out_path}"
        )
    input_records = [
        json.loads(line)
        for line in Path(INPUTS).read_text(encoding="utf-8").splitlines()
        if line
    ]
    inputs = {}
    for record in input_records:
        aid = record["internal_metadata"]["anonymous_id"]
        if aid in inputs:
            raise RuntimeError(f"duplicate second-anchor anonymous_id: {aid}")
        inputs[aid] = record
    gm = golden_map()
    cid_of = {a: inputs[a]["internal_metadata"]["cluster_id"] for a in inputs}
    cases, cohort_sha256 = load_cohort()
    if len(inputs) != 85 or not set(cases).issubset(inputs) or set(gm) != set(cases):
        raise RuntimeError(
            "second-anchor inputs/goldens do not match the 85-input retained-harness 68-case cohort"
        )
    golden_sources = preflight_goldens(cases, gm)
    jobs = [(aid, arm, us, ul, inputs[aid]["model_input"], inputs[aid]["internal_metadata"])
            for aid in cases for (arm, us, ul) in ARMS]
    out_path.mkdir(parents=True, exist_ok=True)
    rec = defaultdict(dict)
    rows, endpoint_errors, verifier_errors = [], [], []
    candidate_root = out_path / "candidates"
    done = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
        for aid, arm, mod, response_sha256, err in ex.map(one, jobs):
            row = {"anonymous_id": aid, "arm": arm, "endpoint_error": err}
            if err:
                endpoint_errors.append({"anonymous_id": aid, "arm": arm, "error": err})
                rows.append(row)
                done += 1
                continue
            row.update({
                "response_sha256": response_sha256,
                "parsed_ok": bool(mod),
            })
            if not mod:
                rec[aid][arm] = False
                row["recovered"] = False
                rows.append(row)
                done += 1
                continue
            candidate_path = candidate_root / aid / f"{arm}.sv"
            write_text_atomic(candidate_path, mod + "\n")
            row.update({
                "candidate_path": str(candidate_path.relative_to(out_path)),
                "candidate_sha256": hashlib.sha256((mod + "\n").encode()).hexdigest(),
            })
            try:
                recovered = sim_ok(golden_sources[aid], mod)
            except (SimulatorProtocolError, SimulatorUnavailableError, RuntimeError) as exc:
                message = str(exc)[:500]
                row["verifier_error"] = message
                verifier_errors.append({
                    "anonymous_id": aid,
                    "arm": arm,
                    "error": message,
                })
                rows.append(row)
                done += 1
                continue
            rec[aid][arm] = recovered
            row["recovered"] = recovered
            rows.append(row)
            done += 1
            if done % 40 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    write_jsonl_atomic(out_path / "rows.jsonl", rows)
    if endpoint_errors or verifier_errors:
        write_json_atomic(out_path / "FAILED.json", {
            "status": "incomplete",
            "scheduled": len(jobs),
            "completed_candidates": len(jobs) - len(endpoint_errors),
            "endpoint_errors": endpoint_errors,
            "verifier_errors": verifier_errors,
        })
        raise RuntimeError(
            f"second-anchor run incomplete: {len(endpoint_errors)} endpoint failures and "
            f"{len(verifier_errors)} verifier failures; "
            "no scientific summary was written"
        )
    missing = [
        (aid, arm)
        for aid in cases
        for arm, _, _ in ARMS
        if arm not in rec[aid]
    ]
    if missing or len(rows) != len(jobs):
        raise RuntimeError(f"second-anchor completeness gate failed: missing={missing[:5]}")
    base = [a for a in cases if rec[a].get("spec0_loc0") is not None]
    brate = round(100 * sum(1 for a in base if rec[a]["spec0_loc0"]) / len(base), 1)
    what = seed_macro(rec, "spec0_loc0", "spec1_loc0", cid_of)
    where = seed_macro(rec, "spec0_loc0", "spec0_loc1", cid_of)
    out = {"model": MODEL, "n_cases": len(base), "baseline_pct": brate,
           "what_pp": what[0], "what_ci": [what[1], what[2]],
           "where_pp": where[0], "where_ci": [where[1], where[2]],
           "scored_by": "differential_simulation", "scheduled": len(jobs),
           "row_ledger": "rows.jsonl", "candidates_retained": True}
    out["cohort_manifest"] = "configs/vericodegen/curve68_manifest.json"
    out["case_ids_sha256"] = cohort_sha256
    write_json_atomic(out_path / "decomp.json", out)
    print(f"\nSECOND-ANCHOR DECOMPOSITION ({MODEL}, sim-scored, n={len(base)}):")
    print(f"  baseline {brate}%  What {what[0]}pp {what[1:]}  Where {where[0]}pp {where[1:]}")


if __name__ == "__main__":
    main()
