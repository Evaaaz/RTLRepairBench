"""Assemble the expanded RepairBench results: per-bucket recovery with Wilson 95%
CIs, per-mutation semantic breakdown, and exact-McNemar paired tests between arms.

Reads the result JSONs written by repairbench_eval.py / realbug_repair_eval.py
(each has {summary, records}). Pairs are matched by record id.

Usage:
  .venv/bin/python benchmarks/analyze_repairbench.py
  (auto-discovers generated/rb_*.json + generated/realbug_*.json)
"""
from __future__ import annotations

import collections
import glob
import json
import math
import os
import sys
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
if CODE not in sys.path:
    sys.path.insert(0, CODE)

from project_paths import output_root  # noqa: E402

# Read outputs from $RTLREPAIR_OUT; keep run data outside the source tree by default.
GEN = os.environ.get("RTLREPAIR_OUT") or str(output_root() / "repairbench")


def wilson(k: int, n: int, z: float = 1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(recs1, recs2):
    m1 = {r["id"]: int(bool(r["recovered"])) for r in recs1}
    m2 = {r["id"]: int(bool(r["recovered"])) for r in recs2}
    ids = set(m1) & set(m2)
    b = sum(1 for i in ids if not m1[i] and m2[i])   # arm1 fail -> arm2 pass
    c = sum(1 for i in ids if m1[i] and not m2[i])   # arm1 pass -> arm2 fail
    n = b + c
    p = min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) * 0.5 ** n) if n else 1.0
    return {"paired": len(ids), "b_gain": b, "c_loss": c, "p_exact": round(p, 5)}


def load(path):
    if not os.path.exists(path):
        return None
    return json.load(open(path))


def fmt_bucket(d):
    lo, hi = wilson(d["recovered"], d["n"])
    return f"{d['recovered']}/{d['n']} = {d['recovery_rate']*100:.1f}% [{lo*100:.1f}, {hi*100:.1f}]"


def main() -> None:
    files = sorted(glob.glob(os.path.join(GEN, "rb_*.json")) +
                   glob.glob(os.path.join(GEN, "realbug_*.json")))
    runs = {}
    for f in files:
        d = load(f)
        if d and "summary" in d:
            runs[os.path.basename(f)[:-5]] = d

    print("=" * 78)
    print("REPAIRBENCH -- per-arm recovery (Wilson 95% CI)")
    print("=" * 78)
    for name, d in runs.items():
        s = d["summary"]
        flags = []
        if s.get("anon"):
            flags.append("anon")
        if s.get("spec"):
            flags.append("+spec")
        tag = f"{s.get('model')} [{','.join(flags) or 'leaked,diag'}]"
        print(f"\n## {name}  ->  {tag}")
        if "by_bucket" in s:
            for b, bd in s["by_bucket"].items():
                print(f"   {b:<14} {fmt_bucket(bd)}")
        else:  # realbug
            print(f"   overall        {s['recovered']}/{s['n']} = {s['recovery_rate']*100:.1f}% "
                  f"{[round(x*100,1) for x in wilson(s['recovered'], s['n'])]}")
            for k, kd in s.get("by_kind", {}).items():
                print(f"   {k:<14} {fmt_bucket(kd)}")

    # per-mutation semantic breakdown for each RepairBench arm
    print("\n" + "=" * 78)
    print("SEMANTIC per-mutation recovery (RepairBench arms)")
    print("=" * 78)
    for name, d in runs.items():
        recs = [r for r in d.get("records", []) if r.get("bucket") == "semantic"]
        if not recs:
            continue
        tot = collections.Counter()
        rec = collections.Counter()
        for r in recs:
            tot[r["mutation"]] += 1
            rec[r["mutation"]] += int(bool(r["recovered"]))
        print(f"\n## {name}")
        for mut in sorted(tot):
            print(f"   {mut:<22} {rec[mut]}/{tot[mut]}")

    # key paired McNemar comparisons
    print("\n" + "=" * 78)
    print("PAIRED McNEMAR (exact)")
    print("=" * 78)
    pairs = [
        ("rb_base_leaked", "rb_base_anon", "leakage effect (leaked->anon)"),
        ("rb_base_anon", "rb_base_anon_spec", "spec effect (anon->anon+spec)"),
        ("realbug_base", "realbug_base_spec", "spec effect on REAL bugs"),
        ("rb_verireason_diag", "rb_verireason_diag_spec", "spec effect on VeriReason"),
        ("rb_base_diag", "rb_base_diag_spec", "spec effect on base"),
        ("rb_v5_diag", "rb_v5_diag_spec", "spec effect on v5"),
        ("rb_base_diag", "rb_v5_diag", "v5 fine-tune vs base (diag)"),
        ("rb_base_diag_spec", "rb_v5_diag_spec", "v5 fine-tune vs base (diag+spec)"),
        ("rb_base_diag", "rb_verireason_diag", "VeriReason vs base (diag)"),
        ("rb_vrqwen_diag", "rb_vrqwen_diag_spec", "spec effect on VR-Qwen"),
        ("rb_base_diag", "rb_vrqwen_diag", "VR-Qwen vs base (diag)"),
        ("rb_verireason_diag", "rb_vrqwen_diag", "VR-Qwen vs VeriReason-CodeLlama (diag)"),
    ]
    for a, b, label in pairs:
        if a in runs and b in runs:
            for bucket in ("lint", "semantic", None):
                ra = [r for r in runs[a]["records"] if bucket is None or r.get("bucket") == bucket]
                rb = [r for r in runs[b]["records"] if bucket is None or r.get("bucket") == bucket]
                if not ra or not rb:
                    continue
                res = mcnemar(ra, rb)
                bl = bucket or "all"
                print(f"  {label} [{bl}]: paired={res['paired']} "
                      f"gain={res['b_gain']} loss={res['c_loss']} p={res['p_exact']}")


if __name__ == "__main__":
    main()
