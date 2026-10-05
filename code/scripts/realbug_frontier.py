#!/usr/bin/env python3
"""Fail-closed three-arm generation for the 274-case real-bug benchmark.

The canonical run is exactly 274 cases by ``nospec/spec/speconly``. Every cell
gets a terminal generation record; extracted candidates are checksum-bound and
retained. Endpoint errors make the whole run incomplete, and an existing run
directory is never overwritten.
"""
import concurrent.futures
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
_SD = str(Path(__file__).resolve().parent)
sys.path.insert(0, _SD)
from eval_endpoint import auth_headers, chat_url

MODEL = os.environ.get("FRONTIER_MODEL", "azure/deepseek-ai/deepseek-v4-pro")
TAG = os.environ.get("FRONTIER_TAG", "deepseek_v4_pro")
OUT = Path(ROOT) / "generated" / "realbug_frontier" / TAG
ARMS = ("nospec", "spec", "speconly")
ARM_CONFIG = {
    "nospec": (False, True),
    "spec": (True, True),
    "speconly": (True, False),
}


def call(system, user, max_tokens=4096):
    body = json.dumps(
        {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        }
    ).encode()
    for attempt in range(6):
        req = urllib.request.Request(chat_url(), data=body, headers=auth_headers())
        try:
            with urllib.request.urlopen(req, timeout=240) as response:
                payload = json.load(response)
            choice = payload["choices"][0]
            message = choice["message"]
            content = message.get("content") or message.get("reasoning_content") or ""
            return content, choice.get("finish_reason")
        except urllib.error.HTTPError as exc:
            if exc.code == 429 and attempt < 5:
                time.sleep(2 ** attempt + (hash(user) % 3))
                continue
            raise


def extract(text):
    match = re.search(r"(module\b.*?endmodule)", text or "", re.S)
    return match.group(1).strip() if match else None


def record_id(record, index):
    value = record.get("id") or f"{record.get('design', 'x')}_{record.get('mutation', 'm')}_{index}"
    return value, value.replace(":", "_").replace("/", "_")


def build_user(record, use_spec, use_broken=True):
    parts = []
    if use_broken:
        parts.append(
            "Broken RTL (`TopModule.sv`):\n```systemverilog\n"
            + record["broken_rtl"].strip()
            + "\n```"
        )
    if use_spec and record.get("spec"):
        parts.append("\nSpecification:\n" + record["spec"].strip())
    if use_broken:
        parts.append(
            "\nThe module compiles but is functionally incorrect. Return the complete "
            "corrected module (module ... endmodule), interface and module name unchanged."
        )
    else:
        parts.append(
            "\nImplement the module described above. Return the complete module "
            "(module ... endmodule)."
        )
    return "\n".join(parts)


def write_text_atomic(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def write_json_atomic(path, payload):
    write_text_atomic(path, json.dumps(payload, indent=2, sort_keys=True) + "\n")


def generate_one(job):
    record, arm, use_spec, use_broken, index = job
    rid, candidate_id = record_id(record, index)
    row = {"id": rid, "candidate_id": candidate_id, "arm": arm}
    try:
        response, finish_reason = call(
            "You repair broken SystemVerilog.",
            build_user(record, use_spec, use_broken),
        )
    except Exception as exc:
        row["endpoint_error"] = f"{type(exc).__name__}: {exc}"[:500]
        return row

    row.update(
        {
            "finish_reason": finish_reason,
            "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
        }
    )
    if not isinstance(finish_reason, str) or not finish_reason:
        row["endpoint_error"] = "ProtocolError: response omitted finish_reason"
        return row
    module = extract(response)
    row["parsed_ok"] = bool(module)
    if module:
        candidate_text = module + "\n"
        candidate_path = OUT / "candidates" / arm / f"{candidate_id}.sv"
        write_text_atomic(candidate_path, candidate_text)
        row.update(
            {
                "candidate_path": str(candidate_path.relative_to(OUT)),
                "candidate_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
            }
        )
    return row


def load_records(path):
    records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
    ids = [record_id(record, index)[1] for index, record in enumerate(records)]
    if len(ids) != len(set(ids)):
        raise RuntimeError("real-bug benchmark contains duplicate normalized IDs")
    return records


def main(limit=None):
    input_path = Path(
        os.environ.get("FRONTIER_INPUT", f"{ROOT}/data/repairbench_realbugs.jsonl")
    )
    records = load_records(input_path)
    requested_arms = tuple(
        arm for arm in os.environ.get("FRONTIER_ARMS", ",".join(ARMS)).split(",") if arm
    )
    if any(arm not in ARM_CONFIG for arm in requested_arms):
        raise RuntimeError(f"unknown FRONTIER_ARMS value: {requested_arms}")
    diagnostic = limit is not None or requested_arms != ARMS
    if diagnostic and os.environ.get("FRONTIER_ALLOW_PARTIAL") != "1":
        raise RuntimeError(
            "partial runs require FRONTIER_ALLOW_PARTIAL=1 and never publish canonical results"
        )
    if not diagnostic and len(records) != 274:
        raise RuntimeError(f"canonical real-bug run requires exactly 274 cases; found {len(records)}")
    if limit is not None:
        if limit < 1 or limit > len(records):
            raise RuntimeError(f"invalid diagnostic limit: {limit}")
        records = records[:limit]
    if OUT.exists():
        raise RuntimeError(f"refusing to overwrite an existing immutable run: {OUT}")
    OUT.mkdir(parents=True)

    jobs = [
        (record, arm, *ARM_CONFIG[arm], index)
        for index, record in enumerate(records)
        for arm in requested_arms
    ]
    results = []
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=int(os.environ.get("FRONTIER_WORKERS", "6"))
    ) as executor:
        for done, row in enumerate(executor.map(generate_one, jobs), 1):
            results.append(row)
            if done % 50 == 0:
                print(
                    f"  {done}/{len(jobs)} parsed "
                    f"{sum(bool(item.get('parsed_ok')) for item in results)}",
                    flush=True,
                )

    observed = {(row["candidate_id"], row["arm"]) for row in results}
    expected = {
        (record_id(record, index)[1], arm)
        for index, record in enumerate(records)
        for arm in requested_arms
    }
    errors = [row for row in results if row.get("endpoint_error")]
    rows_path = OUT / "generation_rows.json"
    write_json_atomic(rows_path, results)
    if len(results) != len(expected) or observed != expected or errors or diagnostic:
        write_json_atomic(
            OUT / "FAILED.json",
            {
                "status": "diagnostic" if diagnostic and not errors else "incomplete",
                "model": MODEL,
                "expected_cells": len(expected),
                "observed_cells": len(observed),
                "endpoint_errors": errors,
            },
        )
        raise RuntimeError("generation did not complete the canonical 274x3 inventory")

    manifest = {
        "schema_version": 1,
        "status": "complete",
        "model": MODEL,
        "tag": TAG,
        "input_path": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "n_cases": 274,
        "arms": list(ARMS),
        "scheduled": 822,
        "parsed": sum(bool(row.get("parsed_ok")) for row in results),
        "rows": "generation_rows.json",
        "rows_sha256": hashlib.sha256(rows_path.read_bytes()).hexdigest(),
        "candidates_retained": True,
    }
    write_json_atomic(OUT / "generation_manifest.json", manifest)
    print(
        f"complete: 822 cells, {manifest['parsed']} parsed candidates -> {OUT}",
        flush=True,
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else None)
