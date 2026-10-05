#!/usr/bin/env python3
"""Re-score the mined-failure panel with the comparison defect fixed.

The oracle appendix shows the differential testbench compares outputs in cycle 0, before
either instance is initialized, where a sequential reference still holds x. A candidate that
resolves a don't-care to a concrete value mismatches there and is recorded as a failed repair,
and specification-conditioned candidates do that far more often because they are regenerations
rather than minimal edits.

That defect was found on the synthetic benchmark and fixed there. This applies the same
one-line fix to the mined-failure panel, which was scored by the uncorrected testbench and
whose headline is a null. Both oracles are reported for every cell so the reader can see what
the fix moves rather than having to trust that it moves nothing.

The retained candidates are the ones the panel was originally scored on, so this is a
re-scoring rather than a new experiment: no model is called.

Writes artifacts/public/realbugs/cycle1_rescore.json.
"""
import json
import os
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

_CODE = Path(__file__).resolve().parent.parent
for _p in (str(_CODE), str(_CODE / "benchmarks"), str(_CODE / "datagen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)
from project_paths import output_root, project_root  # noqa: E402
import re  # noqa: E402
from datagen.diff_testbench import parse_module, run_diff_test  # noqa: E402

SOURCE_ROOT = project_root()
RUNS = output_root().parent / "generated/realbug_frontier"
OUT = SOURCE_ROOT / "artifacts/public/realbugs/cycle1_rescore.json"
ARMS = ("nospec", "spec", "speconly")
# the six models the panel reports, in the paper's order
MODELS = ("gpt55", "claude_opus_4_8", "gpt54mini", "gpt41", "gpt4omini", "claude_opus_4_6")


def mcnemar(b: int, c: int) -> float:
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, 2 * sum(comb(n, k) for k in range(0, min(b, c) + 1)) / 2 ** n)


def golden_for(case_id: str, goldens: dict) -> str | None:
    return goldens.get(case_id)


def main() -> int:
    # the golden for a mined failure is the reference its task was generated against;
    # candidate files are named realbug_<task>_<source>_<n>.sv and the dataset id is
    # realbug:<task>:<source>:<n>, so index the dataset both ways
    goldens = {}
    ds = SOURCE_ROOT / "data/repairbench_realbugs.jsonl"
    for line in ds.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        goldens[row["id"].replace(":", "_")] = row["golden_rtl"]
        goldens[row["id"]] = row["golden_rtl"]

    report = {
        "generated_by": "code/scripts/x4_realbug_cycle1_rescore.py",
        "note": (
            "Same retained candidates as the panel; only the comparison start cycle changes. "
            "cycle0 is the scorer the panel shipped with, cycle1 guards the pre-initialization "
            "compare that the oracle appendix identifies."
        ),
        "models": {},
    }

    for model in MODELS:
        mdir = RUNS / model
        if not mdir.is_dir():
            print(f"  skip {model}: no run directory", flush=True)
            continue
        scored = {}
        for arm in ARMS:
            adir = mdir / arm
            if not adir.is_dir():
                continue
            for sv in sorted(adir.glob("*.sv")):
                case = sv.stem
                golden = golden_for(case, goldens)
                if golden is None:
                    continue
                info = parse_module(golden)
                if info is None:
                    continue
                cand = sv.read_text()
                # candidates were generated against an anonymized TopModule, so give the
                # module back the golden's name before the pair is compiled together
                ci = parse_module(cand)
                if ci and ci.name != info.name:
                    cand = re.sub(rf"\bmodule\s+{re.escape(ci.name)}\b",
                                  f"module {info.name}", cand, count=1)
                for tag, start in (("cycle0", 0), ("cycle1", 1)):
                    r = run_diff_test(golden, cand, info, compare_from_cycle=start)
                    scored.setdefault(tag, {}).setdefault(arm, {})[case] = bool(
                        r.compiled and not r.diverged)
        if not scored:
            print(f"  skip {model}: nothing scored", flush=True)
            continue

        entry = {}
        for tag in ("cycle0", "cycle1"):
            arms = scored.get(tag, {})
            cases = set.intersection(*(set(arms[a]) for a in arms)) if arms else set()
            rates = {a: round(sum(arms[a][c] for c in cases) / len(cases) * 100, 1)
                     for a in arms} if cases else {}
            row = {"n": len(cases), "pct": rates}
            if "spec" in arms and "speconly" in arms and cases:
                b = sum(1 for c in cases if arms["spec"][c] and not arms["speconly"][c])
                d = sum(1 for c in cases if not arms["spec"][c] and arms["speconly"][c])
                row["speconly_minus_spec_pp"] = round((d - b) / len(cases) * 100, 1)
                row["discordant"] = [b, d]
                row["mcnemar_p"] = round(mcnemar(b, d), 4)
            entry[tag] = row
        report["models"][model] = entry
        c0, c1 = entry.get("cycle0", {}), entry.get("cycle1", {})
        print(f"  {model:18s} n={c0.get('n',0):3d}  "
              f"cycle0 {c0.get('speconly_minus_spec_pp','--'):>6}pp p={c0.get('mcnemar_p','--'):<8} "
              f"cycle1 {c1.get('speconly_minus_spec_pp','--'):>6}pp p={c1.get('mcnemar_p','--')}",
              flush=True)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(f"wrote {OUT.relative_to(SOURCE_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
