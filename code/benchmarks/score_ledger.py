#!/usr/bin/env python3
"""Score a banked candidate ledger. Reads what the model said; calls no model.

The inverse of generate_ledger.py. `candidates.jsonl` is what the model
produced; this writes `result.json`, one oracle's reading of it. Because the
ledger holds the candidate RTL verbatim, an arm can be rescored under different
settings -- or by a different oracle entirely -- without spending a token.

    python score_ledger.py --model base --version 20260801-66fe9cc --bank
    python score_ledger.py --ledger candidates_diag.jsonl --arm diag --out result.json

Two oracles, dispatched per record on `bucket`, exactly as repairbench_eval.py
does it:

  lint      `verilator --lint-only -Wall -sv` on the candidate alone. No golden.
  semantic  Tier S -- golden-vs-candidate differential simulation, 200 cycles
            under iverilog. Needs data/processed/verilogeval_seeds/<seed>.sv.

Records from the main85 grid carry no `bucket` (every case there is semantic by
construction), so a missing bucket is read as semantic.

Tier F (formal equivalence) is deliberately not implemented here: it needs the
pinned container (docker + yosys + sby), and `FormalVerifier.verify` already
exposes the same string-in/verdict-out shape when that runtime exists.

Both oracles fail *closed* but in different ways, so both are checked up front:
verilator raises if absent, while iverilog's absence would silently score every
semantic case as unrecovered -- a 0% arm that looks like a real result.
"""
from __future__ import annotations

import argparse
import collections
import concurrent.futures
import json
import os
import shutil
import sys
import threading
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = os.environ.get("RTLREPAIR_ROOT") or str(HERE.parents[3])
sys.path.insert(0, str(HERE))

import repairbench_eval as rbe  # noqa: E402  (also puts ../ and ../datagen on the path)
from storage import (RunKey, default_version_id, get_run, put_run,  # noqa: E402
                     read_ledger, write_index)
from storage.layout import slot_for  # noqa: E402

_print_lock = threading.Lock()

# Keys put_run derives itself. `**meta` is spliced last, so leaving any of these
# in the meta carried over from a previous run.json makes the stale value win.
MANAGED_KEYS = frozenset(
    {"model", "version", "signal", "summary", "n_records", "scored", "ledger", "raw"})


def check_tools(need_semantic: bool, need_lint: bool) -> None:
    """Fail before scoring, not silently during it."""
    missing = []
    if need_lint and not shutil.which("verilator"):
        missing.append("verilator (lint oracle)")
    if need_semantic:
        for tool in ("iverilog", "vvp"):
            if not shutil.which(tool):
                # diff_testbench catches the OSError and reports compiled=False,
                # so a missing iverilog is a silent 0% arm rather than a crash.
                missing.append(f"{tool} (Tier S oracle -- absence scores 0%, silently)")
    if missing:
        raise SystemExit("required tool(s) not on PATH: " + ", ".join(missing))


def score_one(rec: dict, anon: bool) -> dict:
    """One ledger record -> one result record. Never raises."""
    bucket = rec.get("bucket") or "semantic"
    rtl = rec.get("extracted_rtl")
    # A call that errored is not a model answer. Keeping the two apart is what
    # makes used_llm_rate a real validity gate rather than a recovery number.
    used_llm = rec.get("error") is None
    out = {
        "id": rec.get("case_id"),
        "bucket": bucket,
        "mutation": rec.get("mutation") or rec.get("mutation_family"),
        "seed_id": rec.get("seed_id"),
        "recovered": False,
        "used_llm": used_llm,
        "had_spec": bool(rec.get("had_spec")),
        "parsed_ok": bool(rec.get("parsed_ok")),
        "oracle": "verilator_lint" if bucket == "lint" else "tier_s_diffsim",
    }
    if not rtl:
        return out
    a = rec.get("anon", anon)
    try:
        out["recovered"] = bool(
            rbe.recovers_lint(rec["seed_id"], rtl, a) if bucket == "lint"
            else rbe.recovers_semantic(rec["seed_id"], rtl, a))
    except Exception as exc:  # noqa: BLE001 - an oracle crash is data, not a stop
        out["oracle_error"] = f"{type(exc).__name__}: {exc}"[:300]
    return out


def score_records(records: list[dict], anon: bool, workers: int) -> list[dict]:
    buckets = {r.get("bucket") or "semantic" for r in records}
    check_tools("semantic" in buckets, "lint" in buckets)

    done = 0

    def work(rec):
        nonlocal done
        out = score_one(rec, anon)
        with _print_lock:
            done += 1
            if done % 50 == 0 or done == len(records):
                print(f"  scored {done}/{len(records)}", flush=True)
        return out

    if workers > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as ex:
            return list(ex.map(work, records))
    return [work(r) for r in records]


def summarize(results: list[dict], meta: dict, arm: str) -> dict:
    """The summary shape repairbench_eval.py writes, plus which oracle ran."""
    by: dict = collections.defaultdict(
        lambda: {"n": 0, "recovered": 0, "used_llm": 0, "spec": 0, "parsed": 0})
    for r in results:
        b = by[r["bucket"]]
        b["n"] += 1
        b["recovered"] += int(r["recovered"])
        b["used_llm"] += int(r["used_llm"])
        b["spec"] += int(r["had_spec"])
        b["parsed"] += int(r["parsed_ok"])

    gen = (meta or {}).get("generation", {})
    summary = {
        "model": gen.get("requested_model"),
        "arm": arm,
        "n": len(results),
        "anon": gen.get("anon"),
        "temp": gen.get("temp"),
        "scored_from": "ledger",
        "oracles": {"lint": "verilator --lint-only -Wall -sv",
                    "semantic": "tier_s: golden-vs-candidate diff sim, 200 cycles, iverilog"},
        "by_bucket": {},
    }
    for b, d in sorted(by.items()):
        summary["by_bucket"][b] = {
            "n": d["n"], "recovered": d["recovered"],
            "recovery_rate": round(d["recovered"] / d["n"], 4) if d["n"] else 0.0,
            "used_llm_rate": round(d["used_llm"] / d["n"], 4) if d["n"] else 0.0,
            "spec_rate": round(d["spec"] / d["n"], 4) if d["n"] else 0.0,
            "parsed_rate": round(d["parsed"] / d["n"], 4) if d["n"] else 0.0,
        }
    return summary


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_argument_group("ledger source (a banked run, or a loose file)")
    src.add_argument("--model", help="model id or slot of a banked run")
    src.add_argument("--slot", help="override the slot derived from --model")
    src.add_argument("--version", help="version_id (default YYYYMMDD-<git sha>)")
    src.add_argument("--arms", help="comma-separated signals to score "
                                    "(default: every banked arm at that model+version)")
    src.add_argument("--ledger", help="score this candidates.jsonl instead")
    src.add_argument("--arm", help="arm name, with --ledger")

    ap.add_argument("--anon", action="store_true", default=True,
                    help="fallback when a record has no `anon` field (default true)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", help="write result.json here instead of banking")
    ap.add_argument("--bank", action="store_true",
                    help="write result.json into the banked run")
    ap.add_argument("--root", help="tree root (default $RTLREPAIR_STORE)")
    args = ap.parse_args(argv)

    if not (args.bank or args.out):
        raise SystemExit("pass --bank (into the tree) or --out <path>")

    jobs: list[tuple[str, list[dict], dict, RunKey | None]] = []
    if args.ledger:
        if not args.arm:
            raise SystemExit("--ledger needs --arm")
        recs = [json.loads(l) for l in open(args.ledger) if l.strip()]
        jobs.append((args.arm, recs, {}, None))
    else:
        if not args.model:
            raise SystemExit("pass --model (a banked run) or --ledger <path>")
        slot = args.slot or slot_for(args.model)
        version = args.version or default_version_id()
        if args.arms:
            arms = [a.strip() for a in args.arms.split(",")]
        else:
            from storage import list_runs
            arms = [k.signal for k in list_runs(args.root)
                    if k.model == slot and k.version == version]
            if not arms:
                raise SystemExit(f"no banked runs at {slot}/{version}")
        for arm in arms:
            key = RunKey(model=slot, version=version, signal=arm)
            run = get_run(key, root=args.root)
            # get_run's `ledger` is the *path* to candidates.jsonl; read_ledger
            # is what parses it into records.
            if not run.get("ledger"):
                print(f"{arm}: no ledger banked, skipping", flush=True)
                continue
            jobs.append((arm, read_ledger(key, root=args.root),
                         run.get("run") or {}, key))

    if not jobs:
        raise SystemExit("nothing to score")

    for arm, recs, meta, key in jobs:
        print(f"{arm}: scoring {len(recs)} records", flush=True)
        results = score_records(recs, args.anon, args.workers)
        summary = summarize(results, meta, arm)
        print(json.dumps(summary["by_bucket"], indent=2))
        for b, d in summary["by_bucket"].items():
            if d["used_llm_rate"] < 1.0:
                print(f"  WARNING: {arm}/{b} used_llm_rate={d['used_llm_rate']} < 1.0 "
                      "-- failed calls are being scored as no-fix. Regenerate.",
                      flush=True)
        payload = {"summary": summary, "records": results}
        if args.out:
            path = (os.path.join(args.out, f"result_{arm}.json")
                    if len(jobs) > 1 else args.out)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            with open(path, "w") as fh:
                json.dump(payload, fh, indent=2)
            print(f"  wrote {path}")
        if args.bank and key is not None:
            # put_run is write-once and its guard covers the ledger too, so the
            # ledger has to be re-supplied alongside the result under overwrite.
            # meta is the previous run.json, and put_run splices `**meta` LAST --
            # so any key it computes itself must be stripped or the stale value
            # wins. Carrying `summary` through is what silently re-emptied it:
            # the generation-only bank wrote {}, and that {} then overwrote the
            # summary computed from the result blob.
            newmeta = {k: v for k, v in meta.items() if k not in MANAGED_KEYS}
            newmeta["scoring"] = {
                "scorer": "score_ledger.py",
                "oracles": summary["oracles"],
                "workers": args.workers,
            }
            dest = put_run(key, ledger=recs, result=payload, meta=newmeta,
                           root=args.root, overwrite=True)
            write_index(args.root)
            print(f"  banked -> {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
