#!/usr/bin/env python3
"""Score the four open-weight RepairBench rows under Tier~F.

Table~3 shipped scored by Tier~S alone, and its caption said the open-weight
candidates had not been retained and so could not be re-scored. That was wrong.
``artifacts/public/specialization/{base,v5,verireason,vrqwen}/*/{diag,diag_spec}/
candidates.jsonl`` each carry all 85 semantic candidates with populated
``extracted_rtl``, the 52 goldens are vendored, and the prover is pinned. This
script does the work the caption claimed was impossible.

The candidates were generated against an anonymized ``TopModule``; that needs no
special handling here because ``_prepare_pair`` renames both sides to
``design_golden``/``design_candidate`` before building the miter.

Emits, per model and arm, the Tier~F recovery rate beside the shipped Tier~S rate
and the per-cell verdict distribution, so a cell that the prover cannot decide is
reported as coverage rather than folded into a failure.

Writes artifacts/public/specialization/openweight_tierf_rescore.json.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

_CODE = Path(__file__).resolve().parent.parent
for _p in (str(_CODE), str(_CODE / "benchmarks")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from project_paths import project_root  # noqa: E402
from benchmarks.formal_verify import run_validation_batch  # noqa: E402

ROOT = project_root()
SPEC = ROOT / "artifacts/public/specialization"
GOLDENS = ROOT / "data/formal/canonical_goldens"
OUT = SPEC / "openweight_tierf_rescore.json"

MODELS = ("base", "v5", "verireason", "vrqwen")
ARMS = ("diag", "diag_spec")
# a formal verdict that settles the cell; anything else is coverage loss, not a failure
PASSING = {"PROVED"}
REFUTING = {"COUNTEREXAMPLE"}
# Tier S records a pass as the absence of an observed divergence
SIM_PASSING = {"EQUIVALENT", "PASS", "PASSED", "NO_DIVERGENCE"}


def candidate_run(model: str) -> Path | None:
    """The run whose candidates were retained.

    Each model has two batches and only one of them kept candidate RTL, so this
    selects on the artifact rather than on the newest directory name. Choosing by
    name picks the wrong batch for three of the four models.
    """
    runs = [p for p in sorted((SPEC / model).glob("*")) if p.is_dir()
            and any((p / arm / "candidates.jsonl").is_file() for arm in ARMS)]
    if len(runs) > 1:
        raise SystemExit(f"{model}: more than one batch retained candidates: {runs}")
    return runs[0] if runs else None


def build_jobs(jobs_path: Path) -> int:
    rows = []
    for model in MODELS:
        run = candidate_run(model)
        if run is None:
            print(f"  skip {model}: no run directory", flush=True)
            continue
        for arm in ARMS:
            cand_file = run / arm / "candidates.jsonl"
            if not cand_file.is_file():
                print(f"  skip {model}/{arm}: no candidates.jsonl", flush=True)
                continue
            for line in cand_file.read_text().splitlines():
                if not line.strip():
                    continue
                rec = json.loads(line)
                if rec.get("bucket") != "semantic":
                    continue
                rtl = rec.get("extracted_rtl")
                seed = rec.get("seed_id")
                if not rtl or not seed:
                    continue
                golden = GOLDENS / f"{seed}.sv"
                if not golden.is_file():
                    continue
                rows.append({
                    "call_id": f"{model}|{run.name}|{arm}|{rec['case_id']}",
                    "seed_id": seed,
                    "golden_rtl": golden.read_text(),
                    "candidate_rtl": rtl,
                })
    jobs_path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return len(rows)


def summarize(results_path: Path) -> dict:
    # Tier S is carried alongside so the two tiers are compared on identical
    # candidates rather than across the shipped table and this rescore.
    cells: dict[tuple[str, str, str], tuple[str, str]] = {}
    for line in results_path.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        model, _run, arm, case = str(row["call_id"]).split("|", 3)
        formal = str((row.get("formal_verdict") or {}).get("status") or "MISSING").upper()
        sim = str((row.get("simulation_verdict") or {}).get("status") or "MISSING").upper()
        cells[(model, arm, case)] = (formal, sim)

    report: dict = {
        "generated_by": "code/scripts/x5_openweight_tierf_rescore.py",
        "note": (
            "Tier F on the retained open-weight semantic candidates. Recovery is "
            "PROVED / definitive, so cells the prover cannot decide are reported as "
            "coverage loss instead of being scored as failed repairs."
        ),
        "models": {},
    }
    for model in MODELS:
        entry = {}
        for arm in ARMS:
            got = {c: v for (m, a, c), v in cells.items() if m == model and a == arm}
            if not got:
                continue
            dist = Counter(f for f, _ in got.values())
            definitive = sum(v for k, v in dist.items() if k in PASSING | REFUTING)
            proved = sum(v for k, v in dist.items() if k in PASSING)
            sim_pass = sum(1 for _, s in got.values() if s in SIM_PASSING)
            # the two tiers on the same candidates: where they disagree and in which direction
            both = {c: (f, s) for c, (f, s) in got.items() if f in PASSING | REFUTING}
            sim_only = sum(1 for f, s in both.values() if s in SIM_PASSING and f in REFUTING)
            formal_only = sum(1 for f, s in both.values() if s not in SIM_PASSING and f in PASSING)
            entry[arm] = {
                "n_scored": len(got),
                "n_definitive": definitive,
                "proved": proved,
                "recovery_pct_of_definitive": (
                    round(100 * proved / definitive, 1) if definitive else None),
                "recovery_pct_of_all": round(100 * proved / len(got), 1),
                "tier_s_pass_same_candidates": sim_pass,
                "tier_s_pct_of_all": round(100 * sim_pass / len(got), 1),
                "sim_over_credited": sim_only,
                "sim_false_diverged": formal_only,
                "n_both_tiers_definitive": len(both),
                "verdict_distribution": dict(sorted(dist.items())),
            }
        if entry:
            report["models"][model] = entry
    return report



# Tier S levels as printed in Table 3, so the fragment cannot drift from the paper
# without this producer failing its own comparison.
TIER_S = {
    "base": ("Qwen2.5-Coder-7B (base)", 25.9, 38.8),
    "v5": ("qwen2.5-coder-7b-repair-sft", 28.2, 44.7),
    "verireason": ("VeriReason-CodeLlama-7B", 15.3, 28.2),
    "vrqwen": ("VeriReason-Qwen2.5-7B", 10.6, 32.9),
}


def render_table(report: dict) -> str:
    rows = []
    spec_deltas = []
    for key, (label, ts_diag, ts_spec) in TIER_S.items():
        arms = report["models"].get(key)
        if not arms:
            continue
        d, sp = arms["diag"], arms["diag_spec"]
        spec_deltas.append(sp["recovery_pct_of_definitive"] - ts_spec)
        rows.append(
            f"{label} & {ts_diag:.1f}\\% & {d['recovery_pct_of_definitive']:.1f}\\% & "
            f"${d['recovery_pct_of_definitive'] - ts_diag:+.1f}$ & {d['n_definitive']}/{d['n_scored']} & "
            f"{ts_spec:.1f}\\% & {sp['recovery_pct_of_definitive']:.1f}\\% & "
            f"${sp['recovery_pct_of_definitive'] - ts_spec:+.1f}$ & {sp['n_definitive']}/{sp['n_scored']} \\\\"
        )
    mean = sum(spec_deltas) / len(spec_deltas) if spec_deltas else 0.0
    lo, hi = (min(spec_deltas), max(spec_deltas)) if spec_deltas else (0.0, 0.0)
    return (
        "% GENERATED by scripts/x5_openweight_tierf_rescore.py -- do not edit by hand.\n"
        "\\begin{table}[h]\\centering\\footnotesize\\setlength{\\tabcolsep}{3.5pt}\n"
        "\\caption{Table~\\ref{tab:recovery}'s open-weight rows re-scored under Tier~F on the same retained\n"
        "candidates. \\emph{def.} is the cases the prover decides; a cell it cannot decide is coverage loss and\n"
        f"is excluded rather than scored as a failed repair. The specification-arm correction averages\n"
        f"${mean:+.1f}$pp but runs from ${lo:+.1f}$ to ${hi:+.1f}$pp, so it is not a per-cell bound and one\n"
        "model rises under the prover. OriGen\\_Fix, gpt-5.5 and Claude Opus 4.8 retained no candidates and are\n"
        "not re-scorable here.}\n"
        "\\label{tab:tierfrescore}\n"
        "\\begin{tabular}{lrrrc rrrc}\\toprule\n"
        " & \\multicolumn{4}{c}{tool diagnostic} & \\multicolumn{4}{c}{$+$ specification} \\\\\n"
        "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
        "model & Tier~S & Tier~F & $\\Delta$ & def. & Tier~S & Tier~F & $\\Delta$ & def. \\\\\\midrule\n"
        + "\n".join(rows) + "\\bottomrule\n"
        "\\end{tabular}\n\\end{table}\n"
    )

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", default=None, help="where to stage the job file")
    ap.add_argument("--results", default=None, help="where to accumulate verdicts (resumable)")
    ap.add_argument("--fast-timeout", type=int, default=60)
    ap.add_argument("--fallback-timeout", type=int, default=300)
    args = ap.parse_args()

    work = Path(args.jobs).parent if args.jobs else (ROOT / "build/tierf_rescore")
    work.mkdir(parents=True, exist_ok=True)
    jobs = Path(args.jobs) if args.jobs else work / "jobs.jsonl"
    results = Path(args.results) if args.results else work / "results.jsonl"

    n = build_jobs(jobs)
    print(f"staged {n} candidate pairs -> {jobs}", flush=True)
    if not n:
        return 1

    # per-case proof trees go under the work directory, not into the release tree:
    # 680 pairs is ~13k files and artifacts/ is allowlist-governed
    proofs = work / "proofs"
    proofs.mkdir(parents=True, exist_ok=True)
    run_validation_batch(jobs, results, proofs,
                         fast_timeout=args.fast_timeout,
                         fallback_timeout=args.fallback_timeout)

    report = summarize(results)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    frag = ROOT / "paper/tab_tierf_rescore.tex"
    frag.write_text(render_table(report))
    print(f"wrote {frag.relative_to(ROOT)}")
    for model, arms in report["models"].items():
        for arm, r in arms.items():
            print(f"  {model:12s} {arm:10s} proved {r['proved']:>2}/{r['n_definitive']:>2} "
                  f"definitive ({r['recovery_pct_of_definitive']}%), "
                  f"{r['n_scored'] - r['n_definitive']} undecided")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
