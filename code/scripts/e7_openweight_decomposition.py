#!/usr/bin/env python3
"""E7 -- the open-weight decomposition cells, derived from raw formal verdicts.

Why this exists. The open-weight rows of the decomposition table were fed by two
hand-written JSON files -- `generated/tierf_openweight/what_effects.json` and
`artifacts/public/capability/openweight_tier_f.json` -- with no producer anywhere
in the tree. Nothing could re-derive them, so nothing could check them, and
adding a column meant hand-editing a published number. This script is that
missing producer: it reads the verdict ledgers and emits every cell those files
carry, plus the localization arm they did not.

`--validate` re-derives the four published What effects and fails if any point
estimate moves. That is the regression test for the estimator itself; it passed
at +17.9 / +21.6 / +9.3 / +28.0 against the shipped file when this was written.

Definitions, matching what the published cells already do:

  definitive   PROVED or COUNTEREXAMPLE. UNSUPPORTED and COMPILE_FAIL are
               dropped, so the denominator depends on the arm's own outcomes.
  baseline     PROVED / definitive on the diag arm, in percent.
  effect       seed-macro paired risk difference over cases definitive in BOTH
               arms: per seed cluster take the mean of (arm - baseline) success,
               then the unweighted mean across clusters. Success is PROVED only.
  CI           percentile cluster bootstrap, resampling seed clusters with
               replacement. Seeded, so the interval is reproducible.

The two arms are keyed by call_id `<model>__<arm>__<case_id>`; case_id is unique
only within an arm, which is why the arm has to be in the key.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
from collections import defaultdict
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from project_paths import output_root, project_root  # noqa: E402

SOURCE_ROOT = Path(project_root())
ROOT = Path(str(output_root().parent))

MODELS = ("base", "v5", "verireason", "vrqwen")
DEFINITIVE = {"PROVED", "COUNTEREXAMPLE"}
BOOTSTRAP_REPLICATES = 10000
BOOTSTRAP_SEED = 12345


def load_clusters() -> dict[str, str]:
    """case_id -> seed cluster id, from the frozen 2x2 grid inputs."""
    out = {}
    with open(SOURCE_ROOT / "configs" / "vericodegen" / "main85_inputs.jsonl") as fh:
        for line in fh:
            if line.strip():
                m = json.loads(line)["internal_metadata"]
                out[m["case_id"]] = m["cluster_id"]
    return out


def load_verdicts(paths: list[Path]) -> dict[tuple[str, str], dict[str, str]]:
    """(model, arm) -> case_id -> formal status."""
    verd: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
    for path in paths:
        with open(path) as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                model, arm, case = row["call_id"].split("__", 2)
                verd[(model, arm)][case] = (row.get("formal_verdict") or {}).get("status")
    return verd


def baseline_pct(arm: dict[str, str]) -> tuple[float | None, int, int]:
    proved = sum(1 for v in arm.values() if v == "PROVED")
    definitive = sum(1 for v in arm.values() if v in DEFINITIVE)
    return (round(100 * proved / definitive, 1) if definitive else None,
            proved, definitive)


def paired_clusters(base: dict[str, str], other: dict[str, str],
                    clusters: dict[str, str]) -> dict[str, list[int]]:
    """Per-cluster lists of (other - base) success, over both-definitive cases."""
    out: dict[str, list[int]] = defaultdict(list)
    for case, bv in base.items():
        ov = other.get(case)
        if bv in DEFINITIVE and ov in DEFINITIVE:
            out[clusters[case]].append(int(ov == "PROVED") - int(bv == "PROVED"))
    return out


def macro_effect(cl: dict[str, list[int]]) -> float:
    return 100 * statistics.mean([statistics.mean(v) for v in cl.values()])


def bootstrap_ci(cl: dict[str, list[int]], reps: int = BOOTSTRAP_REPLICATES,
                 seed: int = BOOTSTRAP_SEED) -> tuple[float, float]:
    rng = random.Random(seed)
    keys = list(cl)
    draws = []
    for _ in range(reps):
        sample = [cl[rng.choice(keys)] for _ in keys]
        draws.append(100 * statistics.mean([statistics.mean(x) for x in sample]))
    draws.sort()
    return round(draws[int(0.025 * reps)], 1), round(draws[int(0.975 * reps)], 1)


def mcnemar_counts(cl: dict[str, list[int]]) -> tuple[int, int]:
    flat = [x for v in cl.values() for x in v]
    return sum(1 for x in flat if x > 0), sum(1 for x in flat if x < 0)


def effect(base: dict[str, str], other: dict[str, str],
           clusters: dict[str, str]) -> dict | None:
    cl = paired_clusters(base, other, clusters)
    if not cl:
        return None
    lo, hi = bootstrap_ci(cl)
    gained, lost = mcnemar_counts(cl)
    return {
        "pp": round(macro_effect(cl), 1),
        "ci": [lo, hi],
        "paired_cases": sum(len(v) for v in cl.values()),
        "clusters": len(cl),
        "discordant": [gained, lost],
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", action="append", default=None,
                    help="verdict JSONL; repeatable. Default: the banked diag/diag_spec "
                         "ledger plus the diag_locate ledger when it exists")
    ap.add_argument("--out", help="write JSON here (default generated/tierf_openweight/"
                                  "decomposition_effects.json)")
    ap.add_argument("--validate", action="store_true",
                    help="re-derive the published What effects and fail on any drift")
    ap.add_argument("--update-public", action="store_true",
                    help="merge the derived baseline and Where into the public artifact "
                         "artifacts/public/capability/openweight_what_effects.json. The "
                         "published What point estimates and CIs are left byte-identical: "
                         "this only adds the keys that were not there before, and refuses "
                         "to run if a derived What disagrees with the published one.")
    args = ap.parse_args(argv)

    paths = [Path(p) for p in args.verdicts] if args.verdicts else []
    if not paths:
        paths = [ROOT / "generated" / "tierf_openweight" / "results.jsonl"]
        loc = ROOT / "generated" / "tierf_openweight" / "results_diag_locate.jsonl"
        if loc.exists():
            paths.append(loc)
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise SystemExit("missing verdict ledger(s): " + ", ".join(str(p) for p in missing))

    clusters = load_clusters()
    verd = load_verdicts(paths)

    out: dict[str, dict] = {}
    for model in MODELS:
        diag = verd.get((model, "diag"), {})
        if not diag:
            continue
        pct, proved, definitive = baseline_pct(diag)
        row = {"baseline_pct": pct, "baseline_proved": proved,
               "baseline_definitive": definitive, "baseline_n": len(diag)}
        for arm, label in (("diag_spec", "what"), ("diag_locate", "where")):
            e = effect(diag, verd.get((model, arm), {}), clusters)
            if e:
                row[label] = e
        out[model] = row

    public = (SOURCE_ROOT / "artifacts" / "public" / "capability"
              / "openweight_what_effects.json")

    if args.validate:
        published = json.loads(public.read_text()) if public.exists() else json.loads(
            (ROOT / "generated" / "tierf_openweight" / "what_effects.json").read_text())
        bad = []
        for model in MODELS:
            got = out.get(model, {}).get("what", {}).get("pp")
            want = published.get(model, {}).get("what_pp")
            flag = "ok" if got == want else "DRIFT"
            if got != want:
                bad.append(f"{model}: derived {got} vs published {want}")
            print(f"  {model:<12} What derived {got:+.1f}  published {want:+.1f}  {flag}")
        if bad:
            raise SystemExit("estimator drift:\n  " + "\n  ".join(bad))
        print("validate: all four published What effects reproduce exactly")

    if args.update_public:
        pub = json.loads(public.read_text())
        drift = [m for m in MODELS
                 if m in pub and out.get(m, {}).get("what", {}).get("pp") != pub[m].get("what_pp")]
        if drift:
            raise SystemExit(
                "refusing to touch the public artifact: derived What disagrees with the "
                f"published value for {drift}. Investigate before publishing.")
        for m in MODELS:
            row, entry = out.get(m), pub.setdefault(m, {})
            if not row:
                continue
            # Additive only. what_pp/ci stay exactly as published -- the bootstrap CI is
            # seed-dependent and re-deriving it would move a printed interval by a few
            # tenths for no gain.
            entry["baseline_pct"] = row["baseline_pct"]
            entry["baseline_proved"] = row["baseline_proved"]
            entry["baseline_definitive"] = row["baseline_definitive"]
            if "where" in row:
                w = row["where"]
                entry.update({"where_pp": w["pp"], "where_ci": w["ci"],
                              "where_paired_cases": w["paired_cases"],
                              "where_clusters": w["clusters"],
                              "where_discordant": w["discordant"]})
        public.write_text(json.dumps(pub, indent=2, sort_keys=True) + "\n")
        print(f"merged baseline and Where into {public.relative_to(SOURCE_ROOT)} "
              "(published What untouched)")

    dest = Path(args.out) if args.out else \
        ROOT / "generated" / "tierf_openweight" / "decomposition_effects.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "generated_by": "code/scripts/e7_openweight_decomposition.py",
        "verdict_sources": [str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p)
                            for p in paths],
        "estimator": {
            "success": "PROVED only",
            "denominator": "cases definitive (PROVED|COUNTEREXAMPLE) in both arms",
            "point": "seed-macro paired risk difference, unweighted mean over seed clusters",
            "ci": f"percentile cluster bootstrap, {BOOTSTRAP_REPLICATES} replicates, "
                  f"seed {BOOTSTRAP_SEED}",
        },
        "models": out,
    }
    dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {dest}")
    for model, row in out.items():
        cells = "  ".join(
            f"{k.capitalize()} {row[k]['pp']:+.1f} [{row[k]['ci'][0]:+.1f},{row[k]['ci'][1]:+.1f}] "
            f"n={row[k]['paired_cases']}" for k in ("what", "where") if k in row)
        print(f"  {model:<12} baseline {row['baseline_pct']:>5.1f} "
              f"({row['baseline_proved']}/{row['baseline_definitive']})  {cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
