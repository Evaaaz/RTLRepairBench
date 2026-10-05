#!/usr/bin/env python3
"""Regenerate the golden .sv references needed for SEMANTIC re-validation.

Semantic recovery is scored by differential simulation of the model's fix against
the golden reference for each seed. RepairBench's semantic cases span N distinct
VerilogEval seeds; this materializes one golden .sv per seed into $RTLREPAIR_SEEDS
and — crucially — VALIDATES each against the benchmark before writing it:

    * golden-vs-golden must PASS   (harness sanity)
    * golden-vs-broken must DIVERGE for every case of that seed
      (the golden really is the reference the bug was injected against)

then writes a SHA-256 manifest. A golden that fails validation is NOT written, so
a bad reference can never silently corrupt semantic scores.

Source of goldens: the repo's verified canonical goldens (data/formal/
canonical_goldens by default). Override with --source. Uses the FIXED gate from
../datagen/diff_testbench.py.

Usage:
  python make_seeds.py                      # populate generated/repairbench/seeds
  python make_seeds.py --source /path/to/goldens --dry-run
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.dirname(HERE)
for _p in (HERE, CODE, os.path.join(CODE, "datagen")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import repairbench_eval as R  # noqa: E402  (DATA/SEEDS resolution + parse_item)
from diff_testbench import parse_module, run_diff_test  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bench", default=os.path.join(R.DATA, "repairbench_sem85.jsonl"),
                    help="semantic benchmark jsonl (source of seed ids + broken RTL)")
    ap.add_argument("--source", default=os.path.join(R.DATA, "formal", "canonical_goldens"),
                    help="dir of verified golden <seed_id>.sv files")
    ap.add_argument(
        "--out", default=R.SEEDS,
        help="target seeds dir (default: $RTLREPAIR_SEEDS or generated/repairbench/seeds)",
    )
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cases = [json.loads(l) for l in open(args.bench)]
    by_seed = collections.defaultdict(list)
    for c in cases:
        broken, _ = R.parse_item(c)
        by_seed[c["seed_id"]].append((c.get("id"), broken))
    print(f"{len(cases)} semantic cases across {len(by_seed)} seeds")
    print(f"source: {args.source}\ntarget: {args.out}\n")

    manifest, missing, invalid = {}, [], []
    for seed_id, items in sorted(by_seed.items()):
        gpath = os.path.join(args.source, seed_id + ".sv")
        if not os.path.exists(gpath):
            missing.append(seed_id); continue
        golden = open(gpath).read()
        info = parse_module(golden)
        gg = run_diff_test(golden, golden, info) if info else None
        if not (info and gg and gg.compiled and not gg.diverged):
            invalid.append((seed_id, "golden-vs-golden not clean")); continue
        # every case of this seed must be detected as divergent against the golden
        bad_case = None
        for cid, broken in items:
            gb = run_diff_test(golden, broken, info)
            if not (gb.compiled and gb.diverged):
                bad_case = cid; break
        if bad_case is not None:
            invalid.append((seed_id, f"golden did not detect bug in {bad_case}")); continue
        manifest[seed_id] = hashlib.sha256(golden.encode()).hexdigest()
        if not args.dry_run:
            os.makedirs(args.out, exist_ok=True)
            with open(os.path.join(args.out, seed_id + ".sv"), "w") as f:
                f.write(golden)

    print(f"validated & {'would write' if args.dry_run else 'wrote'}: {len(manifest)}/{len(by_seed)} goldens")
    if missing:
        print(f"MISSING from source ({len(missing)}): {missing[:10]}")
    if invalid:
        print(f"INVALID ({len(invalid)}):")
        for s in invalid[:10]:
            print("  ", s)
    if not args.dry_run and not missing and not invalid:
        mpath = os.path.join(args.out, "_manifest.sha256.json")
        with open(mpath, "w") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)
        print(f"manifest -> {mpath}")
    if missing or invalid:
        sys.exit(f"{len(missing)} missing + {len(invalid)} invalid — semantic eval would be incomplete")


if __name__ == "__main__":
    main()
