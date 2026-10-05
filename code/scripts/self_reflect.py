#!/usr/bin/env python3
"""Fail-closed iterative-feedback experiment on the post hoc retained-harness 68-case cohort.

Each case receives up to ``SR_ROUNDS`` attempts. Later prompts include only the
differential-simulation verdict from the preceding attempt. A run retains every
extracted candidate and a checksum/status ledger, refuses to overwrite a prior
run, and publishes ``result.json`` only after the exact cohort completes.
"""
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from pathlib import Path

ROOT = os.environ.get("RTLREPAIR_ROOT") or str(Path(__file__).resolve().parents[2])
_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE))
sys.path.insert(0, str(_CODE / "datagen"))
_SD = str(Path(__file__).resolve().parent)
sys.path.insert(0, _SD)

from eval_endpoint import auth_headers, chat_url
from diff_testbench import (
    SimulatorProtocolError,
    SimulatorUnavailableError,
    _rename_module,
    parse_module,
    require_simulator,
    run_diff_test,
)

INPUTS = f"{ROOT}/configs/vericodegen/main85_inputs.jsonl"
COHORT_MANIFEST = f"{ROOT}/configs/vericodegen/curve68_manifest.json"
ROWS = f"{ROOT}/generated/reports/vericodegen_full85_rows.jsonl"
CR = f"{ROOT}/generated/formal/candidates"
MODEL = os.environ.get("SR_MODEL", "nvcf/meta/llama-3.3-70b-instruct")
K = int(os.environ.get("SR_ROUNDS", "3"))
_DEFAULT_TAG = re.sub(r"[^A-Za-z0-9_.-]+", "_", MODEL).strip("_") + f"_k{K}"
TAG = os.environ.get("SR_TAG", _DEFAULT_TAG)
OUT = Path(ROOT) / "generated" / "self_reflect" / TAG


def call(user, max_tokens=1536):
    body = json.dumps(
        {
            "model": MODEL,
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": "You repair broken SystemVerilog. Return only the corrected module.",
                },
                {"role": "user", "content": user},
            ],
        }
    ).encode()
    req = urllib.request.Request(chat_url(), data=body, headers=auth_headers())
    with urllib.request.urlopen(req, timeout=180) as response:
        choice = json.load(response)["choices"][0]["message"]
    return choice.get("content") or choice.get("reasoning_content") or ""


def extract(text):
    match = re.search(r"(module\b.*?endmodule)", text or "", re.S)
    return match.group(1).strip() if match else None


def load_inputs():
    records = [json.loads(line) for line in Path(INPUTS).read_text().splitlines() if line]
    by_id = {}
    for record in records:
        aid = record["internal_metadata"]["anonymous_id"]
        if aid in by_id:
            raise RuntimeError(f"duplicate main85 anonymous_id: {aid}")
        by_id[aid] = record
    if len(by_id) != 85:
        raise RuntimeError(f"iterative experiment requires exactly 85 inputs; found {len(by_id)}")
    return by_id


def golden_map():
    by_id = {}
    sha_by_id = {}
    for line in Path(ROWS).read_text().splitlines():
        row = json.loads(line)
        if row["mode"] != "main":
            continue
        path = Path(CR) / row["call_id"] / "formal" / "pdr" / "src" / "golden.sv"
        if path.is_file():
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            previous = sha_by_id.get(row["case_id"])
            if previous is not None and previous != digest:
                raise RuntimeError(
                    f"inconsistent retained goldens for case {row['case_id']}"
                )
            sha_by_id[row["case_id"]] = digest
            by_id.setdefault(row["case_id"], path)
    return by_id


def load_cohort():
    manifest = json.loads(Path(COHORT_MANIFEST).read_text(encoding="utf-8"))
    case_ids = manifest.get("case_ids")
    if (
        manifest.get("schema_version") != 1
        or manifest.get("count") != 68
        or not isinstance(case_ids, list)
        or len(case_ids) != 68
        or len(set(case_ids)) != 68
        or case_ids != sorted(case_ids)
    ):
        raise RuntimeError("curve68 cohort manifest is malformed")
    digest = hashlib.sha256(
        json.dumps(case_ids, separators=(",", ":")).encode()
    ).hexdigest()
    if digest != manifest.get("case_ids_sha256"):
        raise RuntimeError("curve68 cohort manifest hash mismatch")
    return case_ids, digest


def preflight_goldens(cases, golden_paths):
    sources = {}
    for aid in cases:
        source = golden_paths[aid].read_text(encoding="utf-8")
        info = parse_module(source)
        if info is None:
            raise RuntimeError(f"golden preflight parse failure: {aid}")
        result = run_diff_test(source, source, info)
        if not result.compiled or result.diverged:
            raise RuntimeError(
                f"golden preflight differential failure: {aid} "
                f"({result.failure_kind or 'unexpected divergence'})"
            )
        sources[aid] = source
    return sources


def score(golden, candidate):
    info = parse_module(golden)
    if info is None:
        raise RuntimeError("retained golden is not parseable by the scoring harness")
    if not candidate:
        return False, None
    candidate_info = parse_module(candidate)
    if candidate_info is None:
        return False, None
    if candidate_info.name != info.name:
        candidate = _rename_module(candidate, candidate_info.name, info.name)
    result = run_diff_test(golden, candidate, info)
    if not result.compiled and result.failure_kind != "compile_failure":
        raise RuntimeError(
            f"differential scorer could not classify candidate: {result.failure_kind}"
        )
    return bool(result.compiled and not result.diverged), result


def feedback(result):
    if result is None:
        return "Your previous fix did not parse. Return a complete corrected module."
    if not result.compiled:
        return "Your previous fix does not compile. Return a complete, compilable corrected module."
    if result.first_cycle is not None and result.first_cycle >= 0:
        outputs = ", ".join(result.out_diffs) if result.out_diffs else "an output"
        return (
            f"Your previous fix still diverges from the reference: at cycle "
            f"{result.first_cycle}, {outputs} differs. It is still functionally "
            "incorrect; try a different fix."
        )
    return "Your previous fix still diverges from the reference. Try a different fix."


def prompt(broken, prior_feedback):
    text = (
        "The following SystemVerilog module compiles and lints clean but is functionally "
        "incorrect (diverges from intended behavior in simulation). Fix the functional bug.\n\n"
        f"```systemverilog\n{broken.strip()}\n```"
    )
    if prior_feedback:
        text += "\n\n" + prior_feedback
    return text + "\n\nReturn only the corrected module."


def exact_mcnemar_p(additions, regressions):
    discordant = additions + regressions
    if discordant == 0:
        return 1.0
    tail = sum(math.comb(discordant, i) for i in range(min(additions, regressions) + 1))
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


def main():
    if K < 1:
        raise RuntimeError("SR_ROUNDS must be at least one")
    require_simulator()
    if OUT.exists() and (not OUT.is_dir() or any(OUT.iterdir())):
        raise RuntimeError(f"refusing to overwrite an existing immutable run: {OUT}")

    inputs = load_inputs()
    golden_paths = golden_map()
    cases, cohort_sha256 = load_cohort()
    if not set(cases).issubset(inputs) or set(golden_paths) != set(cases):
        raise RuntimeError(
            "iterative experiment inputs/goldens do not match the retained-harness 68-case cohort"
        )
    golden_sources = preflight_goldens(cases, golden_paths)
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    failures = []
    for index, aid in enumerate(cases, 1):
        broken = inputs[aid]["model_input"]["broken_rtl"]
        prior_feedback = None
        recovered_round = None
        attempts = []
        for round_number in range(1, K + 1):
            try:
                response = call(prompt(broken, prior_feedback))
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"[:500]
                attempts.append({"round": round_number, "endpoint_error": message})
                failures.append({"case": aid, "round": round_number, "endpoint_error": message})
                break

            attempt = {
                "round": round_number,
                "response_sha256": hashlib.sha256(response.encode()).hexdigest(),
            }
            candidate = extract(response)
            if candidate:
                candidate_text = candidate + "\n"
                candidate_path = OUT / "candidates" / aid / f"round_{round_number}.sv"
                write_text_atomic(candidate_path, candidate_text)
                attempt.update(
                    {
                        "candidate_path": str(candidate_path.relative_to(OUT)),
                        "candidate_sha256": hashlib.sha256(candidate_text.encode()).hexdigest(),
                    }
                )
            else:
                attempt["parse_failure"] = True

            try:
                recovered, result = score(golden_sources[aid], candidate)
            except (SimulatorProtocolError, SimulatorUnavailableError, RuntimeError) as exc:
                message = f"{type(exc).__name__}: {exc}"[:500]
                attempt["verifier_error"] = message
                attempts.append(attempt)
                failures.append({"case": aid, "round": round_number, "verifier_error": message})
                break

            attempt["recovered"] = recovered
            if result is not None:
                attempt["simulation"] = {
                    "compiled": result.compiled,
                    "failure_kind": result.failure_kind,
                    "diverged": result.diverged,
                    "first_cycle": result.first_cycle,
                    "out_diffs": result.out_diffs,
                }
            attempts.append(attempt)
            if recovered:
                recovered_round = round_number
                break
            prior_feedback = feedback(result)

        rows.append(
            {
                "case": aid,
                "recovered_round": recovered_round,
                "attempts": attempts,
            }
        )
        write_jsonl_atomic(OUT / "rows.jsonl", rows)
        if failures:
            break
        if index % 15 == 0:
            pass_at_1 = sum(row["recovered_round"] == 1 for row in rows)
            pass_at_k = sum(row["recovered_round"] is not None for row in rows)
            print(
                f"  {index}/{len(cases)}  pass@1={pass_at_1} pass@{K}={pass_at_k}",
                flush=True,
            )

    if failures or len(rows) != len(cases):
        write_json_atomic(
            OUT / "FAILED.json",
            {
                "status": "incomplete",
                "model": MODEL,
                "rounds": K,
                "expected_cases": len(cases),
                "completed_cases": len(rows),
                "failures": failures,
            },
        )
        raise RuntimeError("iterative experiment incomplete; no scientific result was written")

    pass_at_1 = sum(row["recovered_round"] == 1 for row in rows)
    pass_at_k = sum(row["recovered_round"] is not None for row in rows)
    additions = pass_at_k - pass_at_1
    regressions = 0
    summary = {
        "status": "complete",
        "model": MODEL,
        "rounds": K,
        "n": len(rows),
        "pass_at_1": pass_at_1,
        "pass_at_1_pct": round(100 * pass_at_1 / len(rows), 1),
        "pass_at_K": pass_at_k,
        "pass_at_K_pct": round(100 * pass_at_k / len(rows), 1),
        "gain_pp": round(100 * additions / len(rows), 1),
        "additions": additions,
        "regressions": regressions,
        "mcnemar_exact_two_sided_p": exact_mcnemar_p(additions, regressions),
        "row_ledger": "rows.jsonl",
        "candidates_retained": True,
        "cohort_manifest": "configs/vericodegen/curve68_manifest.json",
        "case_ids_sha256": cohort_sha256,
    }
    write_json_atomic(OUT / "result.json", summary)
    print(
        f"\nSELF-REFLECTION ({MODEL}, no-spec, K={K}): pass@1 "
        f"{summary['pass_at_1_pct']}% -> pass@{K} {summary['pass_at_K_pct']}% "
        f"(gain {summary['gain_pp']}pp over {len(rows)} cases; "
        f"exact McNemar p={summary['mcnemar_exact_two_sided_p']})"
    )


if __name__ == "__main__":
    main()
