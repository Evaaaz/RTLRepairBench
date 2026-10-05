"""stats -- confidence intervals + paired significance for the benchmark tables.

Adds the statistical rigor the workshop paper needs on top of the existing
point estimates produced by ``benchmarks/make_benchmark_table.py`` and
``benchmarks/score_clean.py``:

  * pass@k with a **bootstrap 95% CI** (resample tasks with replacement), for a
    single run's ``result.json``.
  * a **paired** base-vs-tuned comparison: per-task differences in pass@k /
    compile_pass / test_pass with a paired-bootstrap CI + two-sided p-value, and
    an exact **McNemar** test on the binary "task solved at least once" outcome.

It deliberately depends on nothing but the standard library (no numpy/scipy), so
it runs in any environment and re-reads the same ``result.json`` schema the
harness already writes:

    {"model": ..., "n": ..., "temp": ...,
     "problems": [{"id", "n", "c", "compile_pass", "test_pass"}, ...]}

``c`` is the number of the ``n`` samples that passed the testbench, so pass@k is
computed over functional pass (pass@1 == mean test_pass, by construction).

Usage:
    # single run: pass@k + CIs
    python benchmarks/stats.py generated/benchmark_results/tuned/result.json
    # paired comparison (B vs A == tuned vs base)
    python benchmarks/stats.py \
        generated/benchmark_results/base/result.json \
        generated/benchmark_results/tuned/result.json
    # restrict to the 137-task clean (0-leakage) subset
    python benchmarks/stats.py .../base/result.json .../tuned/result.json --clean

Options: --ks 1,5,10  --boot 10000  --seed 0  --clean
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from typing import Dict, List, Optional, Sequence, Set, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_DATA = os.path.abspath(os.path.join(_HERE, "..", "..", "data"))
_DATA = os.environ.get("RTLREPAIR_DATA") or (
    _REPO_DATA if os.path.isdir(_REPO_DATA) else os.path.join(_HERE, "data"))
DEFAULT_CLEAN = os.path.join(_DATA, "eval_tasks_clean.jsonl")


# ---------------------------------------------------------------------------
# pass@k -- mirrors benchmarks/make_benchmark_table.py::pass_at_k exactly.
# (Kept local so this module has zero import side effects.)
# ---------------------------------------------------------------------------
def pass_at_k(n: int, c: int, k: int) -> float:
    """Unbiased pass@k estimator: 1 - C(n-c, k) / C(n, k)."""
    if n <= 0 or k <= 0 or c < 0:
        return 0.0
    if c == 0:  # zero passes => 0 (must precede the n-c<k shortcut)
        return 0.0
    if c >= n:
        return 1.0
    if n - c < k:  # cannot draw k failures => guaranteed at least one pass
        return 1.0
    return 1.0 - (math.comb(n - c, k) / math.comb(n, k))


# ---------------------------------------------------------------------------
# IO
# ---------------------------------------------------------------------------
def load_result(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, dict) or "problems" not in data:
        raise SystemExit("[stats] {0}: expected an object with a 'problems' list".format(path))
    if data.get("is_placeholder"):
        print("[stats] WARNING: {0} has is_placeholder=true -- numbers are not "
              "publishable; re-run with a healthy endpoint (repair_stubs.py).".format(path))
    return data


def load_clean_ids(path: str = DEFAULT_CLEAN) -> Set[str]:
    ids: Set[str] = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if "id" in rec:
                ids.add(str(rec["id"]))
    if not ids:
        raise SystemExit("[stats] clean allowlist empty: {0}".format(path))
    return ids


def problems_by_id(data: Dict, keep: Optional[Set[str]] = None) -> Dict[str, Dict]:
    out: Dict[str, Dict] = {}
    for p in data["problems"]:
        pid = str(p.get("id"))
        if keep is not None and pid not in keep:
            continue
        out[pid] = p
    return out


# ---------------------------------------------------------------------------
# per-task metric vectors
# ---------------------------------------------------------------------------
def task_values(probs: Sequence[Dict], metric: str, k: int = 1) -> List[float]:
    """metric in {'passk','compile','test'}. For 'passk', use cutoff k."""
    vals: List[float] = []
    for p in probs:
        n = int(p.get("n", 0))
        c = int(p.get("c", 0))
        if metric == "passk":
            vals.append(pass_at_k(n, c, k))
        elif metric == "compile":
            vals.append(float(p.get("compile_pass", 0.0)))
        elif metric == "test":
            vals.append(float(p.get("test_pass", 0.0)))
        else:
            raise ValueError(metric)
    return vals


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------
def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def bootstrap_ci(values: Sequence[float], boot: int, seed: int,
                 alpha: float = 0.05) -> Tuple[float, float, float]:
    """Return (point_mean, lo, hi) from a task-level resampling bootstrap."""
    n = len(values)
    if n == 0:
        return 0.0, 0.0, 0.0
    rng = random.Random(seed)
    means: List[float] = []
    vals = list(values)
    for _ in range(boot):
        s = 0.0
        for _ in range(n):
            s += vals[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2.0) * boot)]
    hi = means[min(boot - 1, int((1.0 - alpha / 2.0) * boot))]
    return _mean(values), lo, hi


def paired_bootstrap(diffs: Sequence[float], boot: int, seed: int,
                     alpha: float = 0.05) -> Tuple[float, float, float, float]:
    """Paired bootstrap on per-task differences (b - a).

    Returns (mean_diff, lo, hi, two_sided_p). The p-value is the bootstrap
    proportion of resampled mean-diffs on the opposite side of 0 (doubled,
    clamped to 1) -- a percentile-bootstrap analogue of a paired test.
    """
    n = len(diffs)
    if n == 0:
        return 0.0, 0.0, 0.0, 1.0
    rng = random.Random(seed)
    d = list(diffs)
    means: List[float] = []
    for _ in range(boot):
        s = 0.0
        for _ in range(n):
            s += d[rng.randrange(n)]
        means.append(s / n)
    means.sort()
    lo = means[int((alpha / 2.0) * boot)]
    hi = means[min(boot - 1, int((1.0 - alpha / 2.0) * boot))]
    point = _mean(diffs)
    frac_le0 = sum(1 for m in means if m <= 0.0) / boot
    frac_ge0 = sum(1 for m in means if m >= 0.0) / boot
    p = min(1.0, 2.0 * min(frac_le0, frac_ge0))
    return point, lo, hi, p


def mcnemar_exact(b01: int, b10: int) -> float:
    """Exact two-sided McNemar p-value via the binomial distribution.

    b01 = #tasks A-fail & B-pass, b10 = #tasks A-pass & B-fail.
    """
    m = b01 + b10
    if m == 0:
        return 1.0
    k = min(b01, b10)
    tail = sum(math.comb(m, i) for i in range(0, k + 1)) * (0.5 ** m)
    return min(1.0, 2.0 * tail)


# ---------------------------------------------------------------------------
# reports
# ---------------------------------------------------------------------------
def report_single(data: Dict, ks: Sequence[int], boot: int, seed: int,
                  keep: Optional[Set[str]]) -> None:
    probs = list(problems_by_id(data, keep).values())
    label = data.get("model", "?")
    subset = "clean({0})".format(len(probs)) if keep is not None else "full({0})".format(len(probs))
    print("\n=== {0}  [{1}]  n={2} temp={3} ===".format(
        label, subset, data.get("n"), data.get("temp")))
    for metric, name in (("compile", "compile_pass"), ("test", "testbench_pass")):
        pt, lo, hi = bootstrap_ci(task_values(probs, metric), boot, seed)
        print("  {0:<16} {1:.3f}  [95% CI {2:.3f}, {3:.3f}]".format(name, pt, lo, hi))
    for k in ks:
        pt, lo, hi = bootstrap_ci(task_values(probs, "passk", k), boot, seed)
        print("  pass@{0:<13} {1:.3f}  [95% CI {2:.3f}, {3:.3f}]".format(k, pt, lo, hi))


def report_paired(a: Dict, b: Dict, ks: Sequence[int], boot: int, seed: int,
                  keep: Optional[Set[str]]) -> None:
    pa = problems_by_id(a, keep)
    pb = problems_by_id(b, keep)
    common = sorted(set(pa) & set(pb))
    la, lb = a.get("model", "A"), b.get("model", "B")
    print("\n=== PAIRED  {0}  vs  {1}   (delta = {1} - {0}) ===".format(la, lb))
    print("  matched tasks: {0}  (A={1}, B={2})".format(len(common), len(pa), len(pb)))
    if not common:
        return

    def diffs_for(metric: str, k: int = 1) -> List[float]:
        out = []
        for pid in common:
            va = task_values([pa[pid]], metric, k)[0]
            vb = task_values([pb[pid]], metric, k)[0]
            out.append(vb - va)
        return out

    rows = [("compile_pass", "compile", 1), ("testbench_pass", "test", 1)]
    rows += [("pass@{0}".format(k), "passk", k) for k in ks]
    for name, metric, k in rows:
        d, lo, hi, p = paired_bootstrap(diffs_for(metric, k), boot, seed)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print("  {0:<14} delta={1:+.3f}  [95% CI {2:+.3f}, {3:+.3f}]  p={4:.4f} {5}".format(
            name, d, lo, hi, p, sig))

    # McNemar on "solved at least once" (c > 0)
    b01 = b10 = 0
    for pid in common:
        a_solved = int(pa[pid].get("c", 0)) > 0
        b_solved = int(pb[pid].get("c", 0)) > 0
        if (not a_solved) and b_solved:
            b01 += 1
        elif a_solved and (not b_solved):
            b10 += 1
    p_mc = mcnemar_exact(b01, b10)
    print("  McNemar (solved>=1):  {0}-only={1}  {2}-only={3}  exact p={4:.4f}".format(
        lb, b01, la, b10, p_mc))


def report_md(results: Sequence[Dict], ks: Sequence[int], boot: int, seed: int,
              keep: Optional[Set[str]]) -> None:
    """Emit a paper-ready Markdown table: one row per model, metrics with 95% CI."""
    cols = ["compile_pass", "testbench_pass"] + ["pass@{0}".format(k) for k in ks]
    specs = [("compile", 1), ("test", 1)] + [("passk", k) for k in ks]
    subset = "clean" if keep is not None else "full"
    print("\n| model ({0}) | {1} |".format(subset, " | ".join(cols)))
    print("|" + "---|" * (len(cols) + 1))
    for data in results:
        probs = list(problems_by_id(data, keep).values())
        cells = []
        for metric, k in specs:
            pt, lo, hi = bootstrap_ci(task_values(probs, metric, k), boot, seed)
            cells.append("{0:.3f} [{1:.3f},{2:.3f}]".format(pt, lo, hi))
        print("| {0} (n={1}) | {2} |".format(data.get("model", "?"), len(probs), " | ".join(cells)))
    print("\n_95% CI = task-level bootstrap (B={0}, seed={1}). pass@1 == mean test_pass._".format(boot, seed))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("result_a", help="result.json (single-run mode), or the baseline A in paired mode")
    ap.add_argument("result_b", nargs="?", help="result.json for B -> paired A-vs-B comparison")
    ap.add_argument("--ks", default="1,5,10", help="pass@k cutoffs (comma-sep)")
    ap.add_argument("--boot", type=int, default=10000, help="bootstrap resamples")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible)")
    ap.add_argument("--clean", action="store_true",
                    help="restrict to the clean (0-leakage) ids in data/eval_tasks_clean.jsonl")
    ap.add_argument("--clean-file", default=DEFAULT_CLEAN)
    ap.add_argument("--md", action="store_true",
                    help="also emit a paper-ready Markdown table (Table 1) with CIs")
    args = ap.parse_args()

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    keep = load_clean_ids(args.clean_file) if args.clean else None

    a = load_result(args.result_a)
    report_single(a, ks, args.boot, args.seed, keep)
    loaded = [a]
    if args.result_b:
        b = load_result(args.result_b)
        report_single(b, ks, args.boot, args.seed, keep)
        report_paired(a, b, ks, args.boot, args.seed, keep)
        loaded.append(b)
    if args.md:
        report_md(loaded, ks, args.boot, args.seed, keep)


if __name__ == "__main__":
    main()
