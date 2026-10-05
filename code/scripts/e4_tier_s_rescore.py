#!/usr/bin/env python3
"""E4 -- Tier S rescore of banked decomposition candidates (REVISION_PLAN Part 1).

Local and deterministic: no inference, no new generation. Re-runs differential simulation
(golden vs. candidate, identical $random stimulus) over the candidate RTL already banked in
each run's work/ dirs, and reports:

  * per-arm Tier S recovery, so the four curve models get their Table 3 semantic row
    (the row IS the spec0_loc0 arm -- no spec, no location; see E0 finding F4); and
  * per-model sim-vs-formal agreement on identical candidates, which is the input E6 needs
    to recompute the oracle-miscalibration denominators (the paper's 2.6% / 9.7%).

Needs iverilog+vvp on PATH (pinned oss-cad-suite). Writes generated/reports/e4_tier_s.json.
Existing artifacts are read, never modified.
"""
import concurrent.futures
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from project_paths import output_root, project_root  # noqa: E402

# The split the rest of the bundle uses: configs and inputs come from the bundle, the
# heavy run tree is reached through RTLREPAIR_OUT. Neither encodes a nesting depth.
SOURCE_ROOT = str(project_root())
ROOT = str(output_root().parent)
_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "datagen"))
from diff_testbench import _rename_module, parse_module, require_simulator, run_diff_test

OUT = os.environ.get("E4_OUTPUT", f"{ROOT}/generated/reports/e4_tier_s_recheck.json")
COHORT_MANIFEST = f"{SOURCE_ROOT}/configs/vericodegen/curve68_manifest.json"
ARMS = ["spec0_loc0", "spec1_loc0", "spec0_loc1", "spec1_loc1"]

# Run dir -> roster model key. The canonical llama run is the second_model root; the msf*/
# temp07* subdirs are the multi-sample and temperature probes, not the Figure 1 point.
TARGETS = {
    "generated/second_model":                        "llama-3.3-70b",
    "generated/second_model/runs/openai__openai__gpt-4o-mini": "gpt-4o-mini",
    "generated/second_model/runs/azure__openai__gpt-4.1":      "gpt-4.1",
    "generated/second_model/runs/origen_fix":        "origen_fix",
}


def _sha256_bytes(payload):
    return hashlib.sha256(payload).hexdigest()


def tier_s(work_dir, expected_candidate_sha256=None, expected_golden_sha256=None):
    """Differential-sim verdict for one complete banked candidate."""
    g, c = Path(work_dir) / "golden.sv", Path(work_dir) / "candidate.sv"
    if not (g.is_file() and c.is_file()):
        raise RuntimeError(f"missing banked source under {work_dir}")
    golden_bytes, candidate_bytes = g.read_bytes(), c.read_bytes()
    golden_sha256 = _sha256_bytes(golden_bytes)
    candidate_sha256 = _sha256_bytes(candidate_bytes)
    if expected_golden_sha256 and golden_sha256 != expected_golden_sha256:
        raise RuntimeError(f"banked golden changed after provenance preflight: {work_dir}")
    if expected_candidate_sha256 and candidate_sha256 != expected_candidate_sha256:
        raise RuntimeError(f"banked candidate changed after provenance preflight: {work_dir}")
    try:
        golden = golden_bytes.decode("utf-8")
        cand = candidate_bytes.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise RuntimeError(f"banked source is not UTF-8 under {work_dir}") from exc
    info = parse_module(golden)
    if info is None:
        raise RuntimeError(f"golden preflight parse failure under {work_dir}")
    ci = parse_module(cand)
    if ci and ci.name != info.name:
        cand = _rename_module(cand, ci.name, info.name)
    r = run_diff_test(golden, cand, info)
    if not r.compiled:
        if r.failure_kind != "compile_failure":
            raise RuntimeError(
                f"differential scorer could not classify {work_dir}: {r.failure_kind}"
            )
        return False, "compile_fail", golden_sha256, candidate_sha256
    return (
        not r.diverged,
        "pass" if not r.diverged else "diverged",
        golden_sha256,
        candidate_sha256,
    )


def one(job):
    rel, rec, provenance = job
    work = rec.get("work")
    if not work:
        raise RuntimeError(f"{rel} {rec.get('case_id')} {rec.get('arm')}: no work field")
    ok, detail, golden_sha256, candidate_sha256 = tier_s(
        f"{ROOT}/{work}",
        expected_candidate_sha256=provenance["candidate_sha256"],
        expected_golden_sha256=provenance["golden_sha256"],
    )
    return {"case_id": rec.get("case_id"), "arm": rec.get("arm"), "run": rel,
            "tier_s": ok, "detail": detail, "work": work,
            "golden_sha256": golden_sha256, "candidate_sha256": candidate_sha256}


def load_verdicts(run_dir):
    p = Path(run_dir) / "verdicts.jsonl"
    out = {}
    if not p.is_file():
        raise RuntimeError(f"missing verdict ledger: {p}")
    for line_number, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
        try:
            verdict = json.loads(line)
            key = verdict["dir"]
        except Exception as exc:
            raise RuntimeError(f"malformed verdict row {p}:{line_number}: {exc}") from exc
        if key in out:
            raise RuntimeError(f"duplicate verdict key in {p}: {key}")
        if verdict.get("verdict") not in ("PROVED", "COUNTEREXAMPLE", "ERROR"):
            raise RuntimeError(f"invalid formal verdict in {p}:{line_number}")
        out[key] = verdict["verdict"]
    return out


def validate_verdict_roster(rel, verdicts, cases):
    """Require the formal ledger to close over the retained-harness 68x4 cells."""
    expected = {f"{arm}__{case_id}" for case_id in cases for arm in ARMS}
    observed = set(verdicts)
    if observed != expected:
        missing = sorted(expected - observed)
        extra = sorted(observed - expected)
        raise RuntimeError(
            f"{rel} verdict key roster mismatch: "
            f"missing={missing[:8]} ({len(missing)} total), "
            f"extra={extra[:8]} ({len(extra)} total)"
        )


def _reject_symlink_components(root, relative):
    cursor = Path(root)
    for component in Path(relative).parts:
        cursor = cursor / component
        if cursor.is_symlink():
            raise RuntimeError(f"E4 work provenance contains a symlink: {cursor}")


def validate_work_bindings(rel, records, root=None):
    """Bind each row to one canonical, immutable-on-read candidate directory.

    The exact layout is the layout already used by every row in the frozen
    public E4 artifact.  Content digests are captured here and checked again
    in the scoring worker, closing the path/content TOCTOU window.
    """
    root = Path(root or ROOT)
    provenance = {}
    seen_work = set()
    for index, record in enumerate(records):
        case_id, arm = record.get("case_id"), record.get("arm")
        if not isinstance(case_id, str) or not case_id or arm not in ARMS:
            raise RuntimeError(f"{rel} row {index} has invalid cell identity")
        cell = f"{arm}__{case_id}"
        expected = (Path(rel) / "work" / cell).as_posix()
        work = record.get("work")
        if work != expected:
            raise RuntimeError(
                f"{rel} {cell}: work path is not canonically bound to its row "
                f"(expected {expected!r}, found {work!r})"
            )
        if work in seen_work:
            raise RuntimeError(f"{rel}: duplicate work path for E4 cell: {work}")
        seen_work.add(work)
        _reject_symlink_components(root, work)
        work_dir = root / work
        if not work_dir.is_dir():
            raise RuntimeError(f"{rel} {cell}: missing banked work directory: {work_dir}")
        golden_path = work_dir / "golden.sv"
        candidate_path = work_dir / "candidate.sv"
        if not golden_path.is_file() or not candidate_path.is_file():
            raise RuntimeError(f"{rel} {cell}: incomplete banked source under {work_dir}")
        if golden_path.is_symlink() or candidate_path.is_symlink():
            raise RuntimeError(f"{rel} {cell}: banked source must not be a symlink")
        candidate_sha256 = _sha256_bytes(candidate_path.read_bytes())
        golden_sha256 = _sha256_bytes(golden_path.read_bytes())
        declared = record.get("candidate_sha256")
        if declared is not None and declared != candidate_sha256:
            raise RuntimeError(f"{rel} {cell}: declared candidate SHA256 mismatch")
        provenance[cell] = {
            "work": work,
            "golden_sha256": golden_sha256,
            "candidate_sha256": candidate_sha256,
        }
    return provenance


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


def preflight_goldens(jobs):
    """Self-compare each distinct golden before any candidate is classified."""
    checked = set()
    case_sha256 = {}
    for rel, record, provenance in jobs:
        work = record.get("work")
        if not work:
            raise RuntimeError(f"{rel} {record.get('case_id')} {record.get('arm')}: no work field")
        path = Path(ROOT) / work / "golden.sv"
        if not path.is_file():
            raise RuntimeError(f"missing banked golden: {path}")
        golden = path.read_text(encoding="utf-8")
        digest = hashlib.sha256(golden.encode()).hexdigest()
        if digest != provenance["golden_sha256"]:
            raise RuntimeError(f"banked golden changed after provenance preflight: {path}")
        case_id = record.get("case_id")
        previous = case_sha256.get(case_id)
        if previous is not None and previous != digest:
            raise RuntimeError(f"inconsistent golden mapping for case {case_id}")
        case_sha256[case_id] = digest
        if digest in checked:
            continue
        info = parse_module(golden)
        if info is None:
            raise RuntimeError(f"golden preflight parse failure: {path}")
        result = run_diff_test(golden, golden, info)
        if not result.compiled or result.diverged:
            raise RuntimeError(
                f"golden preflight differential failure: {path} "
                f"({result.failure_kind or 'unexpected divergence'})"
            )
        checked.add(digest)
    return case_sha256


def main():
    require_simulator()
    if os.path.exists(OUT):
        raise RuntimeError(f"refusing to overwrite an existing E4 result: {OUT}")
    cases, cohort_sha256 = load_cohort()
    expected_roster = {(case_id, arm) for case_id in cases for arm in ARMS}
    jobs, meta = [], {}
    for rel, key in TARGETS.items():
        rp = f"{ROOT}/{rel}/results.json"
        if not os.path.isfile(rp):
            raise RuntimeError(f"missing required E4 run: {rp}")
        recs = json.load(open(rp))
        roster = [(r.get("case_id"), r.get("arm")) for r in recs]
        if len(recs) != 272 or len(roster) != len(set(roster)) or set(roster) != expected_roster:
            raise RuntimeError(f"{rel} does not contain the exact 68x4 candidate roster")
        verdicts = load_verdicts(f"{ROOT}/{rel}")
        validate_verdict_roster(rel, verdicts, cases)
        provenance = validate_work_bindings(rel, recs)
        meta[rel] = {"model": key, "n": len(recs), "verdicts": verdicts}
        jobs += [
            (rel, r, provenance[f"{r['arm']}__{r['case_id']}"])
            for r in recs
        ]

    golden_map = preflight_goldens(jobs)
    if set(golden_map) != set(cases):
        raise RuntimeError("E4 golden map does not match the retained-harness 68-case cohort")
    print(f"E4: rescoring {len(jobs)} banked candidates across {len(meta)} runs\n", flush=True)
    results, done = [], 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get("E4_WORKERS", "8"))) as ex:
        for r in ex.map(one, jobs):
            results.append(r); done += 1
            if done % 100 == 0:
                print(f"  {done}/{len(jobs)}", flush=True)
    if len(results) != 1088:
        raise RuntimeError(f"E4 completeness gate expected 1088 cells; found {len(results)}")

    # ---- aggregate per run x arm, and cross-check against the banked formal verdicts
    per = defaultdict(lambda: defaultdict(list))
    for r in results:
        per[r["run"]][r["arm"]].append(r)

    # Verdict vocabulary in the banked verdicts.jsonl is PROVED / COUNTEREXAMPLE / ERROR.
    # PROVED = formally equivalent, COUNTEREXAMPLE = formally refuted, ERROR = no verdict.
    def direction(arms, v, only=None):
        c = Counter()
        for arm, rs in arms.items():
            if only and arm != only:
                continue
            for r in rs:
                fv = v.get(f"{arm}__{r['case_id']}")
                if fv not in ("PROVED", "COUNTEREXAMPLE") or r["tier_s"] is None:
                    c["no_formal"] += 1
                    continue
                formal_pass = (fv == "PROVED")
                if formal_pass == r["tier_s"]:
                    c["agree_pass" if formal_pass else "agree_fail"] += 1
                elif r["tier_s"]:
                    c["sim_false_pass"] += 1        # sim over-credits a repair the proof refutes
                else:
                    c["sim_false_diverge"] += 1     # sim flags divergence the proof certifies away
        return c

    report = {
        "generated_by": "e4_tier_s_rescore.py",
        "cohort_manifest": "configs/vericodegen/curve68_manifest.json",
        "case_ids_sha256": cohort_sha256,
        "runs": {},
    }
    for rel, arms in per.items():
        v = meta[rel]["verdicts"]
        allarms, base = direction(arms, v), direction(arms, v, only="spec0_loc0")
        def pack(c):
            n = c["agree_pass"] + c["agree_fail"] + c["sim_false_pass"] + c["sim_false_diverge"]
            return {**dict(c), "compared": n,
                    "agreement_pct": round(100 * (c["agree_pass"] + c["agree_fail"]) / n, 1) if n else None}
        report["runs"][rel] = {
            "model": meta[rel]["model"],
            "arms": {a: {"n": len(rs),
                         "recovered": sum(1 for r in rs if r["tier_s"]),
                         "rate_pct": round(100 * sum(1 for r in rs if r["tier_s"]) / len(rs), 1),
                         "detail": dict(Counter(r["detail"] for r in rs))}
                     for a, rs in sorted(arms.items())},
            # baseline arm only: the pool the paper's 2.6% / 9.7% is computed over
            "sim_vs_formal_baseline_arm": pack(base),
            # all four arms: the enlarged pool E6 may report instead
            "sim_vs_formal_all_arms": pack(allarms),
        }
    if set(report["runs"]) != set(TARGETS):
        raise RuntimeError("E4 aggregation omitted a required run")
    expected_all = {
        "llama-3.3-70b": 270,
        "gpt-4o-mini": 271,
        "gpt-4.1": 270,
        "origen_fix": 269,
    }
    for info in report["runs"].values():
        if set(info["arms"]) != set(ARMS) or any(a["n"] != 68 for a in info["arms"].values()):
            raise RuntimeError(f"E4 arm denominator drift for {info['model']}")
        if info["sim_vs_formal_baseline_arm"]["compared"] != 67:
            raise RuntimeError(f"E4 baseline comparison denominator drift for {info['model']}")
        if info["sim_vs_formal_all_arms"]["compared"] != expected_all[info["model"]]:
            raise RuntimeError(f"E4 all-arm comparison denominator drift for {info['model']}")

    print(f"\nE4 TIER S RESCORE -> {os.path.relpath(OUT, ROOT)}\n")
    print(f"{'model':<16}{'arm':<13}{'n':>5}{'recovered':>11}{'rate':>8}")
    print("-" * 55)
    for rel, info in report["runs"].items():
        for arm, a in info["arms"].items():
            mark = "  <- Table 3 semantic row" if arm == "spec0_loc0" else ""
            print(f"{info['model']:<16}{arm:<13}{a['n']:>5}{a['recovered']:>11}{a['rate_pct']:>7}%{mark}")
    for pool, label in (("sim_vs_formal_baseline_arm", "baseline arm only (the paper's published pool)"),
                        ("sim_vs_formal_all_arms", "all four arms (enlarged pool)")):
        print(f"\nsim vs. formal, {label}:")
        tot = Counter()
        for info in report["runs"].values():
            s = info[pool]
            for k in ("compared", "agree_pass", "agree_fail", "sim_false_pass", "sim_false_diverge"):
                tot[k] += s.get(k, 0)
            print(f"  {info['model']:<16} n={s['compared']:<5} agree={s['agreement_pct']}%"
                  f"  over-credits={s.get('sim_false_pass', 0)}  false-diverges={s.get('sim_false_diverge', 0)}")
        if tot["compared"]:
            n = tot["compared"]
            print(f"  {'POOLED':<16} n={n:<5} "
                  f"agree={round(100 * (tot['agree_pass'] + tot['agree_fail']) / n, 1)}%"
                  f"  over-credits={tot['sim_false_pass']} ({round(100 * tot['sim_false_pass'] / n, 1)}%)"
                  f"  false-diverges={tot['sim_false_diverge']} ({round(100 * tot['sim_false_diverge'] / n, 1)}%)")

    # ---- parity: reproduce the banked union analysis exactly
    print("\nparity vs. banked union analysis:")
    ok = True
    ts = f"{ROOT}/generated/second_model/union_tier_s.json"
    dr = f"{ROOT}/generated/second_model/union_direction.json"
    name = {"gpt-4.1": "gpt-4.1", "gpt-4o-mini": "gpt-4o-mini",
            "origen_fix": "OriGen_Fix", "llama-3.3-70b": "llama-3.3-70b"}
    if os.path.isfile(ts):
        banked = json.load(open(ts))
        for info in report["runs"].values():
            b = banked.get(name[info["model"]], {}).get("arms", {})
            for arm, a in info["arms"].items():
                exp = b.get(arm, {}).get("tier_s_recovered")
                if exp is not None and exp != a["recovered"]:
                    print(f"  MISMATCH {info['model']} {arm}: banked {exp} vs rescored {a['recovered']}")
                    ok = False
        print(f"  tier_s_recovered: {'all arms reproduce' if ok else 'MISMATCHES ABOVE'}")
    if os.path.isfile(dr):
        banked = json.load(open(dr))
        for info in report["runs"].values():
            b, m = banked.get(name[info["model"]], {}), info["sim_vs_formal_baseline_arm"]
            for k in ("agree_pass", "agree_fail", "sim_false_pass", "sim_false_diverge"):
                if k in b and b[k] != m.get(k, 0):
                    print(f"  MISMATCH {info['model']} {k}: banked {b[k]} vs rescored {m.get(k, 0)}")
                    ok = False
        print(f"  union_direction: {'reproduces' if ok else 'MISMATCHES ABOVE'}")
    if not ok:
        raise RuntimeError("E4 parity gate failed; no report was written")

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    tmp = OUT + ".tmp"
    with open(tmp, "w") as handle:
        json.dump({**report, "rows": results}, handle, indent=1)
        handle.write("\n")
    os.replace(tmp, OUT)


if __name__ == "__main__":
    main()
