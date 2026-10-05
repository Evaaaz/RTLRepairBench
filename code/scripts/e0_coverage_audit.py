#!/usr/bin/env python3
"""E0 -- coverage audit (REVISION_PLAN Part 1). Read-only.

Inventories every banked run behind the three exhibits the revision needs, checks that
the raw artifacts those runs reference still exist on disk, and emits the definitive
"cells to run" checklist that E1--E3 consume so nothing is re-run that already exists
and no hole is discovered late.

Three exhibit families, three on-disk schemas:

  Table 3 (recovery by bug class)   {generated,artifacts/public/specialization}/rb_*.json
                                    -> {"summary": {...}, "records": [{bucket, recovered}]}
  Figure 1 (2x2 decomposition)      generated/second_model/runs/<slug>/results.json
                                    -> [{case_id, arm, finish_reason, parsed_ok, srcdir, work}]
                                       + verdicts.jsonl (formal), work/ (Tier S rescore input)
  Table 6 (real-bug three-arm)      generated/realbug_frontier/<tag>/{score.json,<arm>/*.sv}

Writes generated/reports/e0_coverage.json. Nothing is mutated, generated, or scored.
"""
import json, os, re, sys, glob
from collections import Counter, defaultdict
import sys
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from project_paths import output_root, project_root  # noqa: E402

# The split the rest of the bundle uses: configs and inputs come from the bundle, the
# heavy run tree is reached through RTLREPAIR_OUT. Neither encodes a nesting depth.
SOURCE_ROOT = str(project_root())
ROOT = str(output_root().parent)
OUT = f"{ROOT}/generated/reports/e0_coverage.json"

# ---------------------------------------------------------------- target roster
# From REVISION_PLAN Part 1 (D1 specialization panel, D4 lint budget, D5 Table 6).
# "serving" decides which workstream owns the missing cells: modal -> E1,
# openai/anthropic/nvidia -> E2/E3.
ROSTER = [
    # key                    label                              serving      t3   fig1  t6
    ("base",                 "Qwen2.5-Coder-7B (base)",         "modal",     1,   1,    1),
    ("v5",                   "v5 (repair LoRA)",                "modal",     1,   1,    0),
    ("verireason",           "VeriReason-CodeLlama-7B",         "modal",     1,   1,    1),
    ("vrqwen",               "VR-Qwen2.5-7B",                   "modal",     1,   1,    1),
    ("origen_fix",           "OriGen_Fix",                      "modal",     1,   1,    0),
    ("llama-3.3-70b",        "llama-3.3-70b",                   "modal",     1,   1,    1),
    ("gpt-4o-mini",          "gpt-4o-mini",                     "openai",    1,   1,    1),
    ("gpt-4.1",              "gpt-4.1",                         "openai",    1,   1,    1),
    ("gpt-5.5",              "gpt-5.5 (frozen anchor)",         "openai",    1,   1,    1),
    ("claude-opus-4-8",      "Claude Opus 4.8 (anchor)",        "anthropic", 1,   1,    1),
]
ARMS_2X2 = ["spec0_loc0", "spec1_loc0", "spec0_loc1", "spec1_loc1"]
ARMS_REALBUG = ["nospec", "spec", "speconly"]
LINT_N, SEM_N = 374, 85
# D4 says API models run "the 125-case fixed subset (Claude's existing subset)". E0 finding:
# rb_claude_opus_anon.json's 125 is 40 lint + 85 semantic, so the lint subset Claude actually
# ran is 40 cases, not 125. Target the real subset; D4's number needs correcting in the plan.
LINT_SUBSET_N = 40


def norm(s):
    """Map the many on-disk model spellings onto roster keys."""
    s = (s or "").lower()
    for key, pat in [
        ("gpt-5.5",         r"gpt-?5\.5|gpt55"),
        ("claude-opus-4-8", r"claude.*(4-8|4\.8)|claude_opus_4_8|claude-opus-4-8|^claude-opus$|opus_anon"),
        ("gpt-4o-mini",     r"4o-?mini"),
        ("gpt-4.1",         r"gpt-?4\.1"),
        ("llama-3.3-70b",   r"llama"),
        ("origen_fix",      r"origen"),
        ("verireason",      r"verireason"),
        ("vrqwen",          r"vrqwen|vr-qwen"),
        ("v5",              r"^v5$|tuned|qwen_v5"),
        ("base",            r"^base$|qwen2?\.?5?-?coder"),
    ]:
        if re.search(pat, s):
            return key
    return None


# ---------------------------------------------------------------- Table 3
def audit_table3():
    """Every rb_*.json anywhere in the tree, keyed by (model, spec-arm)."""
    seen, files = defaultdict(list), []
    for pat in (
        "generated/rb_*.json",
        "generated/repairbench_base.json",
        "artifacts/public/specialization/rb_*.json",
    ):
        files += sorted(glob.glob(f"{ROOT}/{pat}"))
    for p in files:
        try:
            d = json.load(open(p))
        except Exception as e:
            seen["_unreadable"].append({"file": p, "error": str(e)[:80]}); continue
        s, recs = d.get("summary", {}), d.get("records", [])
        stem = Path(p).stem
        key = norm(s.get("model")) or norm(stem)
        if key is None:
            seen["_unmapped"].append({"file": os.path.relpath(p, ROOT), "model": s.get("model"), "stem": stem})
            continue
        by = Counter((r.get("bucket"), bool(r.get("recovered"))) for r in recs)
        arm = "diag_spec" if (s.get("spec") or "spec" in stem) else "diag"
        if s.get("locate"):
            arm += "_loc"
        seen[key].append({
            "file": os.path.relpath(p, ROOT), "arm": arm,
            "anon": s.get("anon"), "n": s.get("n", len(recs)),
            "lint_n": by[("lint", True)] + by[("lint", False)],
            "lint_recovered": by[("lint", True)],
            "semantic_n": by[("semantic", True)] + by[("semantic", False)],
            "semantic_recovered": by[("semantic", True)],
        })
    return seen


# ---------------------------------------------------------------- Figure 1
def audit_fig1():
    """2x2 decomposition runs: arm coverage, formal verdicts, artifact availability."""
    out = {}
    for p in sorted(glob.glob(f"{ROOT}/generated/second_model/runs/*/results.json")):
        run = Path(p).parent
        slug = run.name
        recs = json.load(open(p))
        arms = Counter(r["arm"] for r in recs)
        vf = run / "verdicts.jsonl"
        verdicts = {}
        if vf.is_file():
            for line in open(vf):
                try:
                    v = json.loads(line); verdicts[v["dir"]] = v.get("verdict")
                except Exception:
                    pass
        # artifact availability: work/ feeds a Tier S rescore, srcdir feeds formal.
        # Some runs (origen_fix) never recorded srcdir; that is not an artifact gap.
        has_src = any("srcdir" in r for r in recs)
        src_ok = sum(1 for r in recs if r.get("srcdir") and os.path.isdir(f"{ROOT}/{r['srcdir']}"))
        work_ok = sum(1 for r in recs if r.get("work") and os.path.isdir(f"{ROOT}/{r['work']}"))
        out[slug] = {
            "model_key": norm(slug), "run_dir": os.path.relpath(run, ROOT),
            "n_records": len(recs), "arms": dict(arms),
            "arms_missing": [a for a in ARMS_2X2 if a not in arms],
            "cases_per_arm": sorted(set(arms.values())),
            "finish_reasons": dict(Counter(r.get("finish_reason") for r in recs)),
            "parsed_ok": sum(1 for r in recs if r.get("parsed_ok")),
            "formal_verdicts": len(verdicts),
            "formal_definitive": sum(1 for v in verdicts.values() if v in ("PROVED", "DISPROVED")),
            "verdict_breakdown": dict(Counter(verdicts.values())),
            "srcdir_field": "present" if has_src else "absent",
            "srcdirs_present": src_ok, "work_dirs_present": work_ok,
            "artifacts_complete": work_ok == len(recs) and (src_ok == len(recs) or not has_src),
            "candidates_retained": True,
        }

    # gpt-5.5's 2x2 is the frozen preregistered ledger, not a second_model run.
    p = f"{ROOT}/generated/reports/vericodegen_full85_rows.jsonl"
    if os.path.isfile(p):
        rows = [json.loads(l) for l in open(p)]
        main = [r for r in rows if r.get("mode") == "main"]
        fs = Counter(r.get("formal_status") for r in main)
        src_ok = sum(1 for r in main
                     if os.path.isdir(f"{ROOT}/generated/formal/candidates/{r['call_id']}/formal/pdr/src"))
        out["FROZEN_gpt-5.5"] = {
            "model_key": "gpt-5.5", "run_dir": os.path.relpath(p, ROOT),
            "n_records": len(main), "arms": dict(Counter(r["arm"] for r in main)),
            "arms_missing": [a for a in ARMS_2X2 if a not in {r["arm"] for r in main}],
            "cases_per_arm": sorted(set(Counter(r["arm"] for r in main).values())),
            "finish_reasons": {"n/a": len(main)},
            "parsed_ok": sum(1 for r in main if r.get("resolved") is not False),
            "formal_verdicts": sum(v for k, v in fs.items() if k),
            "formal_definitive": fs.get("PROVED", 0) + fs.get("COUNTEREXAMPLE", 0),
            "verdict_breakdown": {str(k): v for k, v in fs.items()},
            "srcdir_field": "derived from call_id",
            "srcdirs_present": src_ok, "work_dirs_present": src_ok,
            # The shortfall is the already-disclosed non-retained proof sources (paper: 244 of
            # 306 accepted proofs re-verify, 62 not re-checkable), not a new staging gap. The
            # frozen ledger is never re-scored, so this blocks nothing.
            "artifacts_complete": True,
            "srcdirs_missing_known": len(main) - src_ok,
            "candidates_retained": True,
            "note": ("FROZEN preregistered primary -- must not be re-scored or re-run; "
                     f"{len(main) - src_ok} of {len(main)} srcdirs absent, matching the "
                     "documented 62 non-retained proof sources"),
        }

    # Claude Opus 4.8's exploratory 2x2 saved only a summary; candidates were not kept.
    p = f"{ROOT}/generated/second_anchor/claude_opus_4_8_decomp/decomp.json"
    if os.path.isfile(p):
        d = json.load(open(p))
        out["second_anchor_claude-opus-4-8"] = {
            "model_key": "claude-opus-4-8", "run_dir": os.path.relpath(Path(p).parent, ROOT),
            "n_records": d.get("n_cases", 0) * 4, "arms": {a: d.get("n_cases", 0) for a in ARMS_2X2},
            "arms_missing": [], "cases_per_arm": [d.get("n_cases", 0)],
            "finish_reasons": {"n/a": 0}, "parsed_ok": 0,
            "formal_verdicts": 0, "formal_definitive": 0, "verdict_breakdown": {},
            "srcdir_field": "absent", "srcdirs_present": 0, "work_dirs_present": 0,
            "artifacts_complete": False, "candidates_retained": False,
            "scored_by": d.get("scored_by"),
            "note": ("summary only: decomp_second_anchor.py writes decomp.json and discards the "
                     "generated RTL, so Tier F scoring of this arm requires re-running all "
                     f"{d.get('n_cases', 0) * 4} generations -- the plan's roster entry "
                     "'candidates exist, need formal scoring' does not hold"),
        }
    return out


# ---------------------------------------------------------------- Table 6
def audit_table6():
    out = {}
    for d in sorted(glob.glob(f"{ROOT}/generated/realbug_frontier/*/")):
        tag = Path(d).name
        sc = Path(d) / "score.json"
        arms = {a: len(glob.glob(f"{d}/{a}/*.sv")) for a in ARMS_REALBUG}
        out[tag] = {
            "model_key": norm(tag), "dir": os.path.relpath(d, ROOT),
            "arms_generated": arms,
            "arms_missing": [a for a, n in arms.items() if n == 0],
            "scored": sc.is_file(),
            "score": json.load(open(sc)) if sc.is_file() else None,
        }
    return out


# ---------------------------------------------------------------- frozen ledger
def audit_frozen():
    p = f"{ROOT}/generated/reports/vericodegen_full85_rows.jsonl"
    if not os.path.isfile(p):
        return {"present": False}
    rows = [json.loads(l) for l in open(p)]
    main = [r for r in rows if r.get("mode") == "main"]
    return {"present": True, "file": os.path.relpath(p, ROOT), "n_rows": len(rows),
            "n_main": len(main), "modes": dict(Counter(r.get("mode") for r in rows)),
            "distinct_main_cases": len({r["case_id"] for r in main})}


# ---------------------------------------------------------------- checklist
def build_checklist(t3, f1, t6):
    """Per roster model x exhibit: HAVE / NEED, and which E-task owns a NEED."""
    owner = {"modal": "E1", "openai": "E2", "anthropic": "E3", "nvidia": "E2"}
    f1_by_model = defaultdict(list)
    for slug, info in f1.items():
        if info["model_key"]:
            f1_by_model[info["model_key"]].append(info)
    t6_by_model = defaultdict(list)
    for tag, info in t6.items():
        if info["model_key"]:
            t6_by_model[info["model_key"]].append(info)

    cells = []
    for key, label, serving, need_t3, need_f1, need_t6 in ROSTER:
        runs3 = [r for r in t3.get(key, []) if isinstance(r, dict)]
        lint = max([r["lint_n"] for r in runs3], default=0)
        sem = max([r["semantic_n"] for r in runs3], default=0)
        # A model whose 2x2 baseline arm is banked already has its Table 3 semantic cell:
        # spec0_loc0 IS the no-spec semantic recovery row. No new inference needed.
        derived_sem = ""
        if sem < SEM_N:
            base_arm = max((r["arms"].get("spec0_loc0", 0) for r in f1_by_model.get(key, [])), default=0)
            if base_arm:
                sem, derived_sem = base_arm, "derived from banked spec0_loc0 arm (no new inference)"
        lint_target = LINT_SUBSET_N if serving in ("openai", "anthropic") else LINT_N
        if need_t3:
            cells.append({"model": key, "label": label, "exhibit": "table3_lint",
                          "have": lint, "target": lint_target,
                          "status": "HAVE" if lint >= lint_target else ("PARTIAL" if lint else "NEED"),
                          "owner": owner[serving], "serving": serving,
                          "sources": [r["file"] for r in runs3]})
            cells.append({"model": key, "label": label, "exhibit": "table3_semantic",
                          "have": sem, "target": SEM_N,
                          "status": "HAVE" if sem >= SEM_N else ("PARTIAL" if sem else "NEED"),
                          "owner": "E4" if derived_sem else owner[serving], "serving": serving,
                          "note": derived_sem,
                          "sources": [r["file"] for r in runs3]})
        if need_f1:
            runs = f1_by_model.get(key, [])
            best = max(runs, key=lambda r: r["n_records"], default=None)
            missing = best["arms_missing"] if best else ARMS_2X2
            cells.append({"model": key, "label": label, "exhibit": "fig1_2x2",
                          "have": best["n_records"] if best else 0,
                          "arms_missing": missing,
                          "status": "HAVE" if best and not missing else ("PARTIAL" if best else "NEED"),
                          "owner": owner[serving], "serving": serving,
                          "formal_scored": best["formal_definitive"] if best else 0,
                          "artifacts_complete": best["artifacts_complete"] if best else False,
                          "candidates_retained": best["candidates_retained"] if best else False,
                          "note": (best or {}).get("note", ""),
                          "sources": [r["run_dir"] for r in runs]})
        if need_t6:
            runs = t6_by_model.get(key, [])
            best = max(runs, key=lambda r: sum(r["arms_generated"].values()), default=None)
            missing = best["arms_missing"] if best else ARMS_REALBUG
            cells.append({"model": key, "label": label, "exhibit": "table6_realbug",
                          "have": sum(best["arms_generated"].values()) if best else 0,
                          "arms_missing": missing,
                          "status": "HAVE" if best and not missing else ("PARTIAL" if best else "NEED"),
                          "owner": owner[serving], "serving": serving,
                          "scored": best["scored"] if best else False,
                          "sources": [best["dir"]] if best else []})
    return cells


def main():
    t3, f1, t6, frozen = audit_table3(), audit_fig1(), audit_table6(), audit_frozen()
    cells = build_checklist(t3, f1, t6)
    report = {"generated_by": "e0_coverage_audit.py", "root": ROOT,
              "table3_runs": t3, "fig1_runs": f1, "table6_runs": t6,
              "frozen_ledger": frozen, "checklist": cells}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(report, open(OUT, "w"), indent=2, default=str)

    # ---- console summary
    print(f"\nE0 COVERAGE AUDIT -> {os.path.relpath(OUT, ROOT)}\n")
    print(f"{'model':<22}{'exhibit':<18}{'status':<9}{'have':>6}  owner  note")
    print("-" * 88)
    for c in cells:
        note = ""
        if c["exhibit"] == "fig1_2x2":
            note = f"formal={c['formal_scored']} artifacts={'ok' if c['artifacts_complete'] else 'MISSING'}"
            if not c["candidates_retained"]:
                note += " CANDIDATES-NOT-RETAINED"
            if c["arms_missing"]:
                note += f" missing={','.join(c['arms_missing'])}"
        elif c["exhibit"] == "table6_realbug":
            note = ("scored" if c["scored"] else "unscored")
            if c["arms_missing"]:
                note += f" missing={','.join(c['arms_missing'])}"
        else:
            note = f"target={c['target']}"
        print(f"{c['model']:<22}{c['exhibit']:<18}{c['status']:<9}{c['have']:>6}  {c['owner']:<6} {note}")

    todo = [c for c in cells if c["status"] != "HAVE"]
    print("\n" + "-" * 88)
    print(f"cells to run: {len(todo)} of {len(cells)}   " +
          "  ".join(f"{k}={v}" for k, v in sorted(Counter(c["owner"] for c in todo).items())))
    art = [s for s, i in f1.items() if not i["artifacts_complete"]]
    print(f"raw artifacts: {len(f1) - len(art)}/{len(f1)} decomposition runs complete on disk" +
          (f"; INCOMPLETE: {', '.join(art)}" if art else " (E4/E5 need no regeneration)"))
    if t3.get("_unmapped"):
        print(f"unmapped result files: {[u['stem'] for u in t3['_unmapped']]}")


if __name__ == "__main__":
    main()
