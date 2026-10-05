"""score_clean -- recompute the benchmark table on the CLEAN (0-leakage) subset.

This does **no** model inference and runs **no** harness. It re-scores an already
completed benchmark run by filtering each model's per-task records down to a
clean id allowlist, then re-emits the headline table with
``benchmarks.make_benchmark_table.make_table`` (so pass@k is the *real*,
unbiased estimator -- never reimplemented here).

Why a clean subset: ``docs/decontamination_note.md`` found 0 exact-duplicate /
source-hash leakage, but ~12% (18/155) of eval specs share a 12+ consecutive-word
spec-paraphrase with a single resyn27k *training* record. Dropping those 18 eval
tasks yields a genuinely 0-leakage held-out subset. The clean ids to KEEP live in
``data/eval_tasks_clean.jsonl`` (one JSON record per line, each with an ``id``),
produced by the sibling decontamination step.

Inputs (READ ONLY):
  data/eval_tasks_clean.jsonl                          -- clean id allowlist (KEEP)
  generated/benchmark_results/base/result.json         -- per-task {id,n,c,...}
  generated/benchmark_results/tuned/result.json
  generated/benchmark_results/repair/result.json        -- optional; merged onto tuned

Outputs (the ONLY things written):
  generated/benchmark_results/table_clean.md
  generated/benchmark_results/table_clean.csv

It deliberately does NOT overwrite ``table.md`` / ``table.csv`` (the full-suite
table) and writes nothing else under ``generated/benchmark_results/``.

The clean row metrics:
  * pass@1 / pass@3 -- recomputed by ``make_table`` from the *filtered* ``problems``
    (each problem keeps its own n,c; make_table averages pass@k over the kept set).
  * compile_pass / testbench_pass -- recomputed as the mean over filtered problems.
  * repair_fail_to_pass (tuned row only) -- recomputed from the repair ``records``
    filtered to clean ids, as ``repaired / initially_failing`` over the kept set,
    exactly like ``benchmarks/run_repair_eval.run_repair`` / ``run_all.py``.

Run (after the live run under generated/benchmark_results/ has finished):

    .venv/bin/python benchmarks/score_clean.py
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
from typing import Dict, List, Optional, Set

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pathlib as _pl  # noqa: E402
_REPO_ROOT = str(_pl.Path(os.environ.get("RTLREPAIR_ROOT", "")) if os.environ.get("RTLREPAIR_ROOT") else _pl.Path(__file__).resolve().parents[2])

# The real table writer -- pass@k is its unbiased estimator, NOT reimplemented here.
import make_benchmark_table as mbt  # noqa: E402

RESULTS_DIR = os.path.join(_REPO_ROOT, "generated", "benchmark_results")
CLEAN_TASKS = os.path.join(_REPO_ROOT, "data", "eval_tasks_clean.jsonl")
BASE_RESULT = os.path.join(RESULTS_DIR, "base", "result.json")
TUNED_RESULT = os.path.join(RESULTS_DIR, "tuned", "result.json")
REPAIR_RESULT = os.path.join(RESULTS_DIR, "repair", "result.json")

OUT_MD = os.path.join(RESULTS_DIR, "table_clean.md")
OUT_CSV = os.path.join(RESULTS_DIR, "table_clean.csv")


def _load_json(path: str) -> Optional[Dict]:
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            obj = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("[score_clean] could not parse {0}: {1}".format(path, exc))
    if not isinstance(obj, dict):
        raise SystemExit("[score_clean] expected a JSON object in {0}".format(path))
    return obj


def load_clean_ids(path: str = CLEAN_TASKS) -> Set[str]:
    """Read the KEEP allowlist: ids from data/eval_tasks_clean.jsonl (one rec/line).

    Accepts either full task records ({"id": ...}) or bare id strings per line.
    """
    if not os.path.exists(path):
        raise SystemExit(
            "[score_clean] clean allowlist not found: {0}\n"
            "  (expected the decontaminated KEEP set, one JSON record per line "
            "with an 'id' field).".format(path)
        )
    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                # tolerate a plain id-per-line file
                ids.add(line)
                continue
            if isinstance(obj, str):
                ids.add(obj)
            elif isinstance(obj, dict) and obj.get("id") is not None:
                ids.add(str(obj["id"]))
            else:
                raise SystemExit(
                    "[score_clean] {0}:{1}: record has no 'id'".format(path, lineno)
                )
    if not ids:
        raise SystemExit("[score_clean] clean allowlist is empty: {0}".format(path))
    return ids


def _filter_problems(problems, clean_ids: Set[str]) -> List[Dict]:
    """Keep only per-task records whose id is in the clean allowlist."""
    kept: List[Dict] = []
    for p in problems or []:
        if isinstance(p, dict) and str(p.get("id")) in clean_ids:
            kept.append(p)
    return kept


def build_clean_row(result: Dict, clean_ids: Set[str]) -> Dict:
    """Filter one model's result to clean ids and recompute the scalar means.

    pass@1/pass@3 are left for make_table to derive from the filtered ``problems``
    (each keeps its own n,c). We strip any precomputed full-suite scalars so the
    table reflects ONLY the clean subset.
    """
    problems = _filter_problems(result.get("problems"), clean_ids)
    row = {
        "model": result.get("model", "unknown"),
        "problems": problems,
        # Recompute compile/testbench means over the kept subset (override the
        # full-suite scalars that were stored in result.json).
        "compile_pass": mbt._mean_field(problems, "compile_pass"),
        "testbench_pass": mbt._mean_field(problems, "test_pass"),
        "is_placeholder": bool(result.get("is_placeholder", False)),
    }
    return row, len(problems)


def merge_repair(tuned_row: Dict, repair: Optional[Dict], clean_ids: Set[str]):
    """Recompute repair fail->pass on the clean subset and merge onto tuned row.

    Mirrors run_repair_eval.run_repair: rate = repaired / initially_failing over
    the kept records, where an initially-failing record has pre_pass False and a
    repaired one additionally has post_pass True. Also OR-in the repair
    placeholder flag, matching run_all.py.
    """
    if not repair:
        return None
    kept = [
        r for r in (repair.get("records") or [])
        if isinstance(r, dict) and str(r.get("id")) in clean_ids
    ]
    initially_failing = sum(1 for r in kept if not r.get("pre_pass"))
    repaired = sum(1 for r in kept if (not r.get("pre_pass")) and r.get("post_pass"))
    rate = (repaired / initially_failing) if initially_failing else None
    tuned_row["repair_fail_to_pass"] = rate
    tuned_row["is_placeholder"] = bool(
        tuned_row.get("is_placeholder") or repair.get("is_placeholder")
    )
    return {
        "kept_records": len(kept),
        "initially_failing": initially_failing,
        "repaired": repaired,
        "repair_fail_to_pass": rate,
    }


def _write_clean_table(rows: List[Dict]) -> Dict[str, str]:
    """Call the real make_table, then place its output as table_clean.{md,csv}.

    make_table hardcodes the filenames table.md/table.csv, so we render into a
    throwaway temp dir and move the files to table_clean.* -- guaranteeing we
    never touch the full-suite table.md / table.csv.
    """
    os.makedirs(RESULTS_DIR, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="score_clean_")
    try:
        paths = mbt.make_table(rows, out_dir=tmp)
        shutil.move(paths["md"], OUT_MD)
        shutil.move(paths["csv"], OUT_CSV)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return {"md": OUT_MD, "csv": OUT_CSV}


def _passk_from_row(row: Dict) -> Dict[str, Optional[float]]:
    """Recompute pass@1/pass@3 the same way make_table will, for the printout."""
    problems = row.get("problems")
    return {
        "pass@1": mbt.mean_pass_at_k(problems, 1) if problems else None,
        "pass@3": mbt.mean_pass_at_k(problems, 3) if problems else None,
    }


def _fmt(v) -> str:
    return "-" if v is None else "{0:.4f}".format(v)


def main() -> int:
    clean_ids = load_clean_ids()

    base_res = _load_json(BASE_RESULT)
    tuned_res = _load_json(TUNED_RESULT)
    if base_res is None or tuned_res is None:
        missing = [p for p, o in ((BASE_RESULT, base_res), (TUNED_RESULT, tuned_res)) if o is None]
        raise SystemExit(
            "[score_clean] required result file(s) missing: {0}\n"
            "  (run the benchmark first; this script only re-scores a finished run)."
            .format(", ".join(missing))
        )
    repair_res = _load_json(REPAIR_RESULT)  # optional

    base_row, base_kept = build_clean_row(base_res, clean_ids)
    tuned_row, tuned_kept = build_clean_row(tuned_res, clean_ids)
    repair_info = merge_repair(tuned_row, repair_res, clean_ids)

    table_rows = [base_row, tuned_row]
    paths = _write_clean_table(table_rows)

    # ---- report ----
    print("[score_clean] clean allowlist: {0} ids ({1})".format(
        len(clean_ids), CLEAN_TASKS))
    for name, res, row, kept in (
        ("base", base_res, base_row, base_kept),
        ("tuned", tuned_res, tuned_row, tuned_kept),
    ):
        total = len(res.get("problems") or [])
        pk = _passk_from_row(row)
        line = "[score_clean] {0:<5} kept {1}/{2} tasks  pass@1={3}  pass@3={4}".format(
            name, kept, total, _fmt(pk["pass@1"]), _fmt(pk["pass@3"]))
        if res.get("is_placeholder"):
            line += "  [PLACEHOLDER]"
        print(line)
    if repair_info is not None:
        print("[score_clean] repair kept {0} records  initially_failing={1}  "
              "repaired={2}  fail_to_pass={3}".format(
                  repair_info["kept_records"], repair_info["initially_failing"],
                  repair_info["repaired"], _fmt(repair_info["repair_fail_to_pass"])))
    else:
        print("[score_clean] repair result.json absent -> repair_fail_to_pass left blank")

    print("[score_clean] wrote {0}".format(paths["md"]))
    print("[score_clean] wrote {0}".format(paths["csv"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
