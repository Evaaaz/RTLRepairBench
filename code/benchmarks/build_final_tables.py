"""build_final_tables -- assemble the base / v5 / v6 headline tables.

Reads the per-model result.json (after stub-repair), and writes:
  generated/benchmark_results/table.md / .csv         -- full 155-task set
  generated/benchmark_results/table_clean.md / .csv   -- decontaminated 137-task subset

The clean subset keeps only the ids in data/eval_tasks_clean.jsonl (the
paraphrase-overlapping tasks removed), so it is a genuinely 0-leakage held-out
number. pass@k is recomputed by make_benchmark_table from the filtered problems.
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile

_THIS = os.path.dirname(os.path.abspath(__file__))
_CODE = os.path.dirname(_THIS)
for _p in (_THIS, _CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)
import pathlib as _pl  # noqa: E402
_ROOT = str(_pl.Path(os.environ.get("RTLREPAIR_ROOT", "")) if os.environ.get("RTLREPAIR_ROOT") else _pl.Path(__file__).resolve().parents[2])

import bench_common  # noqa: E402
import make_benchmark_table as mbt  # noqa: E402

RD = bench_common.RESULTS_DIR
MODELS = [("base", "base"), ("tuned", "v5 (tuned)"), ("tunedv6", "v6 (tunedv6)")]
CLEAN = os.path.join(_ROOT, "data", "eval_tasks_clean.jsonl")


def _row(model_dir, label, keep=None):
    j = json.load(open(os.path.join(RD, model_dir, "result.json")))
    probs = j.get("problems", [])
    if keep is not None:
        probs = [p for p in probs if str(p.get("id")) in keep]
    return {"model": label, "problems": probs, "is_placeholder": j.get("is_placeholder", False)}


def _clean_ids():
    ids = set()
    with open(CLEAN, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                ids.add(str(json.loads(line).get("id")))
    return ids


def _write(rows, md_name):
    tmp = tempfile.mkdtemp()
    paths = mbt.make_table(rows, out_dir=tmp)
    out_md = os.path.join(RD, md_name)
    out_csv = os.path.join(RD, md_name.replace(".md", ".csv"))
    shutil.move(paths["md"], out_md)
    shutil.move(paths["csv"], out_csv)
    shutil.rmtree(tmp, ignore_errors=True)
    return out_md


def main():
    full = [_row(d, lab) for d, lab in MODELS]
    full_md = _write(full, "table.md")
    print("=== FULL (155 tasks) -> {0} ===".format(full_md))
    print(open(full_md).read())

    keep = _clean_ids()
    clean = [_row(d, lab, keep=keep) for d, lab in MODELS]
    n_clean = len(clean[0]["problems"])
    clean_md = _write(clean, "table_clean.md")
    print("=== CLEAN / 0-leakage ({0} tasks) -> {1} ===".format(n_clean, clean_md))
    print(open(clean_md).read())


if __name__ == "__main__":
    main()
