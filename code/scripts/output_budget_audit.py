#!/usr/bin/env python3
"""Reconstruct the output-budget confound from the sanitized public ledger."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable


ROOT = Path(os.environ.get("RTLREPAIR_ROOT", "")) if os.environ.get(
    "RTLREPAIR_ROOT"
) else Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = ROOT / "artifacts/public/primary/ledger/public_events.jsonl"
DEFAULT_OUTPUT = ROOT / "build/analysis/output_budget_confound.json"
BUDGET = 4096
ARMS = ("spec0_loc0", "spec0_loc1", "spec1_loc0", "spec1_loc1")
BASELINE = "spec0_loc0"
EXPECTED_CASES = 85


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{number}: expected a JSON object")
            rows.append(value)
    return rows


def _exact_sign_test(negative: int, positive: int) -> float:
    non_ties = negative + positive
    if non_ties == 0:
        return 1.0
    tail = min(negative, positive)
    probability = 2.0 * sum(
        math.comb(non_ties, k) for k in range(tail + 1)
    ) / (2**non_ties)
    return min(1.0, probability)


def analyze(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    requests: dict[tuple[str, str], list[dict[str, Any]]] = {}
    responses: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("mode") != "main":
            continue
        event = row.get("event")
        if event not in {"request_attempt", "response"}:
            continue
        arm = row.get("arm")
        case_id = row.get("case_id")
        if arm not in ARMS or not isinstance(case_id, str) or not case_id:
            raise ValueError("research ledger row has an invalid case/arm identity")
        key = (case_id, arm)
        if event == "response" and row.get("enters_research_denominator") is not True:
            raise ValueError(f"{case_id}/{arm}: response is outside research denominator")
        if event == "request_attempt":
            requests.setdefault(key, []).append(row)
        else:
            if key in responses:
                raise ValueError(f"duplicate response row for {case_id}/{arm}")
            responses[key] = row

    expected_cells = EXPECTED_CASES * len(ARMS)
    if len(requests) != expected_cells or len(responses) != expected_cells:
        raise ValueError(
            f"expected {expected_cells} request/response cells, found "
            f"{len(requests)}/{len(responses)}"
        )
    if set(requests) != set(responses):
        raise ValueError("request/response cell inventories differ")

    by_arm_cases = {
        arm: {case_id for case_id, found_arm in responses if found_arm == arm}
        for arm in ARMS
    }
    baseline_cases = by_arm_cases[BASELINE]
    if len(baseline_cases) != EXPECTED_CASES or any(
        cases != baseline_cases for cases in by_arm_cases.values()
    ):
        raise ValueError("the four arms do not share the exact 85-case roster")

    reasoning: dict[tuple[str, str], int] = {}
    arm_summary: dict[str, dict[str, int]] = {}
    truncated_reasoning: list[int] = []
    for arm in ARMS:
        truncations = 0
        for case_id in sorted(baseline_cases):
            key = (case_id, arm)
            request_budgets = {
                request.get("request_metadata", {}).get("max_output_tokens")
                for request in requests[key]
            }
            if request_budgets != {BUDGET}:
                raise ValueError(
                    f"{case_id}/{arm}: request budgets are not exactly {{{BUDGET}}}"
                )
            response = responses[key]
            usage = response.get("usage")
            if not isinstance(usage, dict):
                raise ValueError(f"{case_id}/{arm}: missing usage")
            output_tokens = usage.get("output_tokens")
            token_details = usage.get("output_tokens_details")
            reasoning_tokens = (
                token_details.get("reasoning_tokens")
                if isinstance(token_details, dict)
                else None
            )
            if (
                isinstance(output_tokens, bool)
                or not isinstance(output_tokens, int)
                or isinstance(reasoning_tokens, bool)
                or not isinstance(reasoning_tokens, int)
                or not 0 <= reasoning_tokens <= output_tokens <= BUDGET
            ):
                raise ValueError(f"{case_id}/{arm}: invalid output-token accounting")
            reasoning[key] = reasoning_tokens
            truncated = response.get("incomplete_reason") == "max_output_tokens"
            if truncated:
                if response.get("response_status") != "incomplete" or output_tokens != BUDGET:
                    raise ValueError(f"{case_id}/{arm}: inconsistent truncation record")
                truncations += 1
                truncated_reasoning.append(reasoning_tokens)
        arm_summary[arm] = {"n": EXPECTED_CASES, "truncations": truncations}

    sign_tests: dict[str, dict[str, int | float]] = {}
    for arm in ARMS[1:]:
        differences = [
            reasoning[(case_id, arm)] - reasoning[(case_id, BASELINE)]
            for case_id in sorted(baseline_cases)
        ]
        negative = sum(value < 0 for value in differences)
        positive = sum(value > 0 for value in differences)
        ties = sum(value == 0 for value in differences)
        sign_tests[arm] = {
            "decreases": negative,
            "increases": positive,
            "ties": ties,
            "two_sided_p": _exact_sign_test(negative, positive),
        }

    baseline_truncations = arm_summary[BASELINE]["truncations"]
    joint_truncations = arm_summary["spec1_loc1"]["truncations"]
    return {
        "schema_version": 1,
        "source": "artifacts/public/primary/ledger/public_events.jsonl",
        "budget_tokens": BUDGET,
        "n_cases": EXPECTED_CASES,
        "arms": arm_summary,
        "paired_reasoning_sign_tests_vs_spec0_loc0": sign_tests,
        "baseline_to_joint_truncation_reduction": (
            baseline_truncations - joint_truncations
        ),
        "upper_bound_flipped_truncations": baseline_truncations - joint_truncations,
        "truncated_cells": len(truncated_reasoning),
        "minimum_reasoning_tokens_in_truncated_cell": min(truncated_reasoning),
        "all_truncated_cells_used_full_output_budget": True,
    }


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(value, indent=2, sort_keys=True) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=path.parent, prefix=f".{path.name}.",
            suffix=".tmp", delete=False
        ) as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
            temporary = Path(handle.name)
        os.replace(temporary, path)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args(argv)
    result = analyze(_load_jsonl(args.ledger))
    write_json_atomic(args.output, result)
    print(f"verified output-budget audit -> {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
