#!/usr/bin/env python3
"""Fail-closed differential scoring for a canonical 274x3 real-bug run."""
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "datagen"))
from diff_testbench import _rename_module, parse_module, require_simulator, run_diff_test

ARMS = ("nospec", "spec", "speconly")


def candidate_id(record, index):
    value = record.get("id") or f"{record.get('design', 'x')}_{record.get('mutation', 'm')}_{index}"
    return value.replace(":", "_").replace("/", "_")


def score(golden, candidate, info):
    if not candidate:
        return False
    candidate_info = parse_module(candidate)
    if candidate_info is None:
        return False
    if candidate_info.name != info.name:
        candidate = _rename_module(candidate, candidate_info.name, info.name)
    result = run_diff_test(golden, candidate, info)
    if not result.compiled:
        if result.failure_kind == "compile_failure":
            return False
        raise RuntimeError(
            f"differential scorer could not classify candidate: {result.failure_kind}"
        )
    return not result.diverged


def mcnemar(fixed, lost):
    discordant = fixed + lost
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(fixed, lost) + 1))
    return min(1.0, 2.0 * tail / (2 ** discordant))


def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path, payload):
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def write_jsonl_atomic(path, rows):
    write_text_atomic(
        path,
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
    )


def load_records(path):
    records = [json.loads(line) for line in path.read_text().splitlines() if line]
    ids = [candidate_id(record, index) for index, record in enumerate(records)]
    if len(records) != 274 or len(ids) != len(set(ids)):
        raise RuntimeError(
            f"scoring requires exactly 274 uniquely identified real-bug cases; found {len(records)}"
        )
    return records, ids


def load_generation(run_dir, input_path, ids):
    manifest = json.loads((run_dir / "generation_manifest.json").read_text())
    if (
        manifest.get("schema_version") != 1
        or manifest.get("status") != "complete"
        or manifest.get("n_cases") != 274
        or manifest.get("scheduled") != 822
        or manifest.get("arms") != list(ARMS)
    ):
        raise RuntimeError("generation manifest is not a complete canonical 274x3 run")
    if manifest.get("input_sha256") != hashlib.sha256(input_path.read_bytes()).hexdigest():
        raise RuntimeError("generation manifest input hash does not match scoring input")
    rows_relative = manifest.get("rows")
    if rows_relative != "generation_rows.json":
        raise RuntimeError("generation manifest names an unexpected rows file")
    rows_path = run_dir / rows_relative
    rows_sha256 = manifest.get("rows_sha256")
    if not isinstance(rows_sha256, str) or not re.fullmatch(r"[0-9a-f]{64}", rows_sha256):
        raise RuntimeError("generation manifest rows_sha256 is missing or malformed")
    if hashlib.sha256(rows_path.read_bytes()).hexdigest() != rows_sha256:
        raise RuntimeError("generation rows hash does not match the manifest")
    rows = json.loads(rows_path.read_text())
    expected = {(cid, arm) for cid in ids for arm in ARMS}
    observed = [(row.get("candidate_id"), row.get("arm")) for row in rows]
    if len(rows) != 822 or len(observed) != len(set(observed)) or set(observed) != expected:
        raise RuntimeError("generation rows do not contain the exact 274x3 inventory")
    if any("endpoint_error" in row for row in rows):
        raise RuntimeError("generation rows contain endpoint failures")
    by_cell = {(row["candidate_id"], row["arm"]): row for row in rows}
    for key, row in by_cell.items():
        if type(row.get("parsed_ok")) is not bool:
            raise RuntimeError(f"generation row parsed_ok is not Boolean for {key}")
        if not isinstance(row.get("finish_reason"), str) or not row["finish_reason"]:
            raise RuntimeError(f"generation row finish_reason is missing for {key}")
        response_sha256 = row.get("response_sha256")
        if not isinstance(response_sha256, str) or not re.fullmatch(
            r"[0-9a-f]{64}", response_sha256
        ):
            raise RuntimeError(f"generation row response hash is malformed for {key}")
        if row["parsed_ok"]:
            relative = Path(row["candidate_path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise RuntimeError(f"unsafe candidate path for {key}: {relative}")
            path = run_dir / relative
            if not path.is_file():
                raise RuntimeError(f"missing parsed candidate for {key}: {path}")
            if hashlib.sha256(path.read_bytes()).hexdigest() != row.get("candidate_sha256"):
                raise RuntimeError(f"candidate hash mismatch for {key}")
        elif "candidate_path" in row or "candidate_sha256" in row:
            raise RuntimeError(f"unparsed generation row carries candidate metadata for {key}")
    return by_cell


def preflight_goldens(records, ids):
    prepared = {}
    for record, cid in zip(records, ids):
        golden = record.get("golden_rtl")
        if not isinstance(golden, str) or not golden.strip():
            raise RuntimeError(f"missing golden RTL for {cid}")
        info = parse_module(golden)
        if info is None:
            raise RuntimeError(f"golden preflight parse failure: {cid}")
        result = run_diff_test(golden, golden, info)
        if not result.compiled or result.diverged:
            raise RuntimeError(
                f"golden preflight differential failure: {cid} "
                f"({result.failure_kind or 'unexpected divergence'})"
            )
        prepared[cid] = (golden, info)
    return prepared


def paired_counts(results, ids, first, second):
    fixed = sum(not results[first][cid] and results[second][cid] for cid in ids)
    lost = sum(results[first][cid] and not results[second][cid] for cid in ids)
    return fixed, lost, mcnemar(fixed, lost)


def main(tag, input_name):
    require_simulator()
    input_path = Path(input_name).resolve()
    records, ids = load_records(input_path)
    run_dir = Path(ROOT) / "generated" / "realbug_frontier" / tag
    summary_path = run_dir / "score.json"
    if summary_path.exists() or (run_dir / "FAILED_SCORE.json").exists():
        raise RuntimeError(f"refusing to overwrite a terminal scoring state in {run_dir}")
    generation = load_generation(run_dir, input_path, ids)
    goldens = preflight_goldens(records, ids)

    results = {arm: {} for arm in ARMS}
    score_rows = []
    try:
        for record, cid in zip(records, ids):
            golden, info = goldens[cid]
            for arm in ARMS:
                generation_row = generation[(cid, arm)]
                candidate = None
                if generation_row.get("parsed_ok"):
                    candidate = (run_dir / generation_row["candidate_path"]).read_text()
                recovered = score(golden, candidate, info)
                results[arm][cid] = recovered
                score_rows.append(
                    {"candidate_id": cid, "arm": arm, "recovered": recovered}
                )
    except Exception as exc:
        write_jsonl_atomic(run_dir / "score_rows.partial.jsonl", score_rows)
        write_json_atomic(
            run_dir / "FAILED_SCORE.json",
            {
                "status": "incomplete",
                "completed_cells": len(score_rows),
                "error": f"{type(exc).__name__}: {exc}"[:500],
            },
        )
        raise

    if len(score_rows) != 822:
        raise RuntimeError(f"scoring completeness gate expected 822 cells; found {len(score_rows)}")
    write_jsonl_atomic(run_dir / "score_rows.jsonl", score_rows)

    summary = {"status": "complete", "tag": tag, "n": 274}
    for arm in ARMS:
        recovered = sum(results[arm].values())
        summary[arm] = {
            "pct": round(100 * recovered / 274, 1),
            "recovered": recovered,
            "n": 274,
        }
        print(f"  {arm:9s}: {recovered}/274 = {summary[arm]['pct']}%")
    fixed, lost, p_value = paired_counts(results, ids, "nospec", "spec")
    summary["intent_effect"] = {
        "fixed": fixed,
        "lost": lost,
        "mcnemar_p": p_value,
    }
    fixed, lost, p_value = paired_counts(results, ids, "spec", "speconly")
    summary["regen_vs_repair"] = {
        "speconly_fixes_spec_misses": fixed,
        "spec_fixes_speconly_misses": lost,
        "mcnemar_p": p_value,
    }
    summary.update(
        {
            "generation_manifest": "generation_manifest.json",
            "score_rows": "score_rows.jsonl",
            "candidates_retained": True,
        }
    )
    write_json_atomic(summary_path, summary)
    print(f"[saved {summary_path}]")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit("usage: general_score.py <TAG> <INPUT_JSONL>")
    main(sys.argv[1], sys.argv[2])
