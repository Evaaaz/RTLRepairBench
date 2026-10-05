"""Experiment 3: taxonomy of the semantic bugs that survive BOTH repair levers.

The intent lever (base + spec) and the capability lever (Claude Opus 4.8) each move
functional repair but neither closes it. This analysis characterizes the intersection
-- the semantic cases that BOTH miss -- by mutation class, turning the negative headline
into a diagnosis of which functional bugs are hard. Pure analysis over banked runs ($0).
"""
from __future__ import annotations

import collections
import json
import math
import os

import pathlib as _pl  # noqa: E402
ROOT = str(_pl.Path(os.environ.get("RTLREPAIR_ROOT", "")) if os.environ.get("RTLREPAIR_ROOT") else _pl.Path(__file__).resolve().parents[2])
GEN = os.path.join(ROOT, "generated")

# family grouping (the qualitative categories)
FAMILY = {
    "swap_ternary": "polarity/select ambiguity",
    "flip_reset_polarity": "polarity/select ambiguity",
    "op_swap": "operator substitution",
    "swap_logical": "operator substitution",
    "flip_equality": "operator substitution",
    "off_by_one_literal": "value/constant",
}


def wilson(k, n, z=1.96):
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def recs(name):
    d = json.load(open(os.path.join(GEN, name + ".json")))
    return {r["id"]: r for r in d["records"] if r.get("bucket") == "semantic"}


def main():
    diag = recs("rb_base_anon")            # base, diagnostic only
    spec = recs("rb_base_anon_spec")       # base + spec (intent lever)
    claude = recs("rb_claude_opus_anon")   # Claude Opus (capability lever)

    ids = sorted(set(spec) & set(claude))
    by_mut = collections.defaultdict(lambda: {"n": 0, "both_survive": 0, "spec_fail": 0, "claude_fail": 0})
    survivors = []
    for i in ids:
        m = spec[i]["mutation"]
        sfail = not spec[i]["recovered"]
        cfail = not claude[i]["recovered"]
        d = by_mut[m]
        d["n"] += 1
        d["spec_fail"] += int(sfail)
        d["claude_fail"] += int(cfail)
        if sfail and cfail:
            d["both_survive"] += 1
            survivors.append({"id": i, "mutation": m, "family": FAMILY.get(m, "?"),
                              "seed_id": spec[i]["seed_id"]})

    # overall
    both = sum(d["both_survive"] for d in by_mut.values())
    ntot = sum(d["n"] for d in by_mut.values())
    print(f"BOTH-LEVER SURVIVORS: {both}/{ntot} = {both/ntot*100:.1f}% of semantic cases "
          f"survive base+spec AND Claude Opus 4.8")
    lo, hi = wilson(both, ntot)
    print(f"  95% CI [{lo*100:.1f}, {hi*100:.1f}]\n")

    print(f"{'mutation':<22}{'n':>4}{'survive both':>14}{'rate':>9}{'  95% CI':>16}   family")
    for m in sorted(by_mut, key=lambda k: -by_mut[k]["both_survive"]):
        d = by_mut[m]
        r = d["both_survive"] / d["n"] if d["n"] else 0
        lo, hi = wilson(d["both_survive"], d["n"])
        print(f"{m:<22}{d['n']:>4}{d['both_survive']:>14}{r*100:>8.0f}%   [{lo*100:>4.0f},{hi*100:>4.0f}]   {FAMILY.get(m,'?')}")

    # family rollup
    print("\nby family:")
    fam = collections.defaultdict(lambda: {"n": 0, "both": 0})
    for m, d in by_mut.items():
        f = FAMILY.get(m, "?")
        fam[f]["n"] += d["n"]
        fam[f]["both"] += d["both_survive"]
    for f, d in sorted(fam.items(), key=lambda kv: -kv[1]["both"]):
        r = d["both"] / d["n"] if d["n"] else 0
        lo, hi = wilson(d["both"], d["n"])
        print(f"  {f:<26} {d['both']}/{d['n']} = {r*100:.0f}% survive both  [{lo*100:.0f},{hi*100:.0f}]")

    # ladder: how each lever alone and together reduces the pool
    print("\nreduction ladder (semantic, n=%d):" % ntot)
    diag_fail = sum(1 for i in ids if i in diag and not diag[i]["recovered"])
    print(f"  fail base (diagnostic only):     {diag_fail}/{ntot} = {diag_fail/ntot*100:.0f}%")
    sf = sum(1 for i in ids if not spec[i]["recovered"])
    cf = sum(1 for i in ids if not claude[i]["recovered"])
    print(f"  fail base+spec (intent lever):   {sf}/{ntot} = {sf/ntot*100:.0f}%")
    print(f"  fail Claude Opus (capability):   {cf}/{ntot} = {cf/ntot*100:.0f}%")
    print(f"  fail BOTH levers (the residue):  {both}/{ntot} = {both/ntot*100:.0f}%")

    json.dump({"both_survivors": survivors,
               "by_mutation": {m: dict(v) for m, v in by_mut.items()},
               "n_semantic": ntot, "both_count": both},
              open(os.path.join(GEN, "survivor_taxonomy.json"), "w"), indent=2)
    print(f"\nwrote {GEN}/survivor_taxonomy.json  ({len(survivors)} survivor cases for the qualitative pass)")


if __name__ == "__main__":
    main()
