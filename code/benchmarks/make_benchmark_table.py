"""make_benchmark_table -- aggregate per-model benchmark results into a table.

Reads one or more per-model result dicts (each summarising a model's run over
the benchmark suite) and writes:

    generated/benchmark_results/table.md
    generated/benchmark_results/table.csv

with columns:
    model, compile_pass, testbench_pass, pass_at_1, pass_at_3, repair_fail_to_pass

pass@k is the unbiased estimator from Chen et al. (HumanEval / Codex):

    pass@k = 1 - C(n - c, k) / C(n, k)

averaged over problems, where for a problem we draw ``n`` samples and ``c`` of
them pass. This is numerically stable and exact for small n.

A model row is a dict, e.g.::

    {
      "model": "tuned",
      "compile_pass": 0.74,          # optional precomputed scalars ...
      "testbench_pass": 0.59,
      "repair_fail_to_pass": 0.22,
      "is_placeholder": False,
      # ... OR raw per-problem records to derive pass@k from:
      "problems": [{"n": 20, "c": 7, "compile_pass": 0.9, ...}, ...],
      "pass_at_k": {1: 0.59, 3: 0.73},  # optional precomputed
    }

If a row carries ``problems`` we recompute pass@1/pass@3 (and compile/testbench
means) from them; otherwise we fall back to the scalar fields. Missing fields
render as ``-``.
"""
from __future__ import annotations

import csv
import json
import math
import os
from typing import Dict, List, Optional, Sequence

import pathlib as _pl  # noqa: E402
_REPO_ROOT = str(_pl.Path(os.environ.get("RTLREPAIR_ROOT", "")) if os.environ.get("RTLREPAIR_ROOT") else _pl.Path(__file__).resolve().parents[2])
DEFAULT_OUT_DIR = os.path.join(_REPO_ROOT, "generated", "benchmark_results")

COLUMNS = [
    "model",
    "compile_pass",
    "testbench_pass",
    "pass_at_1",
    "pass_at_3",
    "repair_fail_to_pass",
]


def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k).

    Args:
        n: total samples drawn for the problem.
        c: number of those samples that passed.
        k: cutoff.

    Returns:
        Probability that at least one of k samples (drawn without replacement
        from the n) passes. Returns 1.0 when n - c < k (cannot draw k failures),
        0.0 when c == 0, and clamps degenerate inputs.
    """
    if n <= 0 or k <= 0 or c < 0:
        return 0.0
    if c >= n:
        return 1.0
    # Zero passes => pass@k is 0 (check before the n-c<k shortcut, else a low-n
    # run with c==0 and n<k would wrongly report 1.0).
    if c == 0:
        return 0.0
    # If there aren't k failing samples to draw, every k-subset hits a pass.
    if n - c < k:
        return 1.0
    # 1 - prod_{i=0}^{k-1} (n-c-i)/(n-i)  -- the stable product form.
    prob_all_fail = 1.0
    for i in range(k):
        prob_all_fail *= (n - c - i) / (n - i)
    return 1.0 - prob_all_fail


def mean_pass_at_k(problems: Sequence[Dict], k: int) -> Optional[float]:
    """Mean of pass@k over problems. Each problem needs ``n`` and ``c``.

    Returns None if no usable problems.
    """
    vals = []
    for p in problems or []:
        n = p.get("n")
        c = p.get("c")
        if n is None or c is None:
            continue
        vals.append(pass_at_k(int(n), int(c), k))
    if not vals:
        return None
    return sum(vals) / len(vals)


def _mean_field(problems: Sequence[Dict], field: str) -> Optional[float]:
    vals = [p[field] for p in (problems or []) if isinstance(p.get(field), (int, float))]
    if not vals:
        return None
    return sum(vals) / len(vals)


def _derive_row(row: Dict) -> Dict:
    """Resolve one model row to the canonical column set (values or None)."""
    out = {"model": str(row.get("model", row.get("name", "unknown")))}
    problems = row.get("problems")

    # compile_pass / testbench_pass: prefer scalar, else mean over problems.
    out["compile_pass"] = _coalesce(
        row.get("compile_pass"), _mean_field(problems, "compile_pass")
    )
    out["testbench_pass"] = _coalesce(
        row.get("testbench_pass"),
        row.get("test_pass"),
        _mean_field(problems, "test_pass"),
    )

    # pass@k: prefer derived from problems, else precomputed dict/scalar.
    precomp = row.get("pass_at_k") or {}
    out["pass_at_1"] = _coalesce(
        mean_pass_at_k(problems, 1) if problems else None,
        _lookup_k(precomp, 1),
        row.get("pass_at_1"),
    )
    out["pass_at_3"] = _coalesce(
        mean_pass_at_k(problems, 3) if problems else None,
        _lookup_k(precomp, 3),
        row.get("pass_at_3"),
    )

    out["repair_fail_to_pass"] = _coalesce(
        row.get("repair_fail_to_pass"), row.get("repair_success_rate")
    )
    out["is_placeholder"] = bool(row.get("is_placeholder", False))
    return out


def _coalesce(*vals):
    for v in vals:
        if v is not None:
            return v
    return None


def _lookup_k(d: Dict, k: int):
    if not isinstance(d, dict):
        return None
    if k in d:
        return d[k]
    if str(k) in d:
        return d[str(k)]
    return None


def _fmt(v) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return "{0:.2f}".format(v)
    return str(v)


def make_table(rows: List[Dict], out_dir: str = DEFAULT_OUT_DIR) -> Dict[str, str]:
    """Aggregate ``rows`` into table.md + table.csv under ``out_dir``.

    Returns {"md": <path>, "csv": <path>}.
    """
    os.makedirs(out_dir, exist_ok=True)
    derived = [_derive_row(r) for r in rows]
    any_placeholder = any(d.get("is_placeholder") for d in derived)

    md_path = os.path.join(out_dir, "table.md")
    csv_path = os.path.join(out_dir, "table.csv")

    # ---- Markdown ----
    title = "# RTL Generation Benchmark"
    if any_placeholder:
        title += " (contains PLACEHOLDER rows)"
    lines = [title, ""]
    if any_placeholder:
        lines += [
            "> **PLACEHOLDER:** rows marked PLACEHOLDER were produced offline "
            "(serve_stub fallback) with no real model server. Replace once a "
            "model is served.",
            "",
        ]
    header = "| " + " | ".join(COLUMNS + ["note"]) + " |"
    sep = "|" + "|".join(["---"] * (len(COLUMNS) + 1)) + "|"
    lines += [header, sep]
    for d in derived:
        note = "PLACEHOLDER" if d.get("is_placeholder") else "measured"
        cells = [_fmt(d[c]) for c in COLUMNS] + [note]
        lines.append("| " + " | ".join(cells) + " |")
    lines.append("")
    lines.append("_pass@k uses the unbiased estimator 1 - C(n-c,k)/C(n,k)._")
    lines.append("_repair_fail_to_pass is measured on greedy (temp=0) first attempts — "
                 "a separate metric from the temp-sampled pass@k columns._")
    lines.append("")
    with open(md_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))

    # ---- CSV ----
    with open(csv_path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(COLUMNS + ["note"])
        for d in derived:
            note = "PLACEHOLDER" if d.get("is_placeholder") else "measured"
            writer.writerow([_csv_cell(d[c]) for c in COLUMNS] + [note])

    return {"md": md_path, "csv": csv_path}


def _csv_cell(v) -> str:
    if v is None:
        return ""
    if isinstance(v, float):
        return "{0:.4f}".format(v)
    return str(v)


def load_rows(paths: Sequence[str]) -> List[Dict]:
    """Load model-result rows from JSON files. Skips missing/bad files."""
    rows: List[Dict] = []
    for p in paths or []:
        if not p or not os.path.exists(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as fh:
                obj = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if isinstance(obj, list):
            rows.extend(x for x in obj if isinstance(x, dict))
        elif isinstance(obj, dict):
            # Allow {"rows": [...]} or a single model dict.
            if isinstance(obj.get("rows"), list):
                rows.extend(x for x in obj["rows"] if isinstance(x, dict))
            else:
                rows.append(obj)
    return rows


def _cli():
    import argparse

    ap = argparse.ArgumentParser(description="Aggregate per-model benchmark results.")
    ap.add_argument("results", nargs="*", help="per-model result JSON files")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()
    rows = load_rows(args.results)
    if not rows:
        print("No result files loaded; nothing to aggregate.")
        return
    paths = make_table(rows, out_dir=args.out_dir)
    print("Wrote {0} and {1} ({2} rows).".format(paths["md"], paths["csv"], len(rows)))


if __name__ == "__main__":
    _cli()
