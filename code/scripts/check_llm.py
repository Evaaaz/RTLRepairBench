#!/usr/bin/env python3
"""End-to-end smoke test for the configured RTLRepair vLLM endpoint.

Usage::

    source .env  # exports RTLREPAIR_LLM_URL and RTLREPAIR_LLM_API_KEY
    python code/scripts/check_llm.py
    python code/scripts/check_llm.py --model tunedv6
    python code/scripts/check_llm.py --skip-repair    # generation only

What it checks:
  1. ``GET /models`` reachability + auth (via ``llm_client.health_check``).
  2. A short generation against the selected model (``generate_strict``).
  3. A repair round-trip (``repair_agent.repair`` with a synthetic Verilator
     error). Confirms the model returns a fenced ``systemverilog`` code block.

Exit code is 0 on full success, 1 on any failure -- safe for CI / pre-demo.
"""
from __future__ import annotations

import argparse
import os
import sys
import textwrap
import time
from pathlib import Path

# Make `from backend import ...` work when run from anywhere: prefer the
# consolidated code/ tree (code/backend/), with code/benchmarks/ as a fallback
# in case the backend package lives at code/benchmarks/backend/.
_CODE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_CODE / "benchmarks"))
sys.path.insert(0, str(_CODE))


def _hr(title: str) -> None:
    print()
    print("=" * 72)
    print("  " + title)
    print("=" * 72)


def _ok(msg: str) -> None:
    print("\033[32mOK\033[0m  " + msg)


def _bad(msg: str) -> None:
    print("\033[31mFAIL\033[0m  " + msg)


def _info(msg: str) -> None:
    print("    " + msg)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model",
        default="tuned",
        help="model id to exercise (default: tuned). Try 'base' or 'tunedv6'.",
    )
    parser.add_argument(
        "--skip-repair",
        action="store_true",
        help="skip the repair round-trip (only test reachability + generation).",
    )
    args = parser.parse_args()

    from backend import llm_client  # lazy
    import repair_agent

    failures: list[str] = []

    _hr("1/3  endpoint reachability")
    health = llm_client.health_check(timeout_s=6.0)
    if health["ok"]:
        _ok(f"GET /models  ·  {health['base_url']}  ·  auth={health['auth']}")
        _info("served: " + ", ".join(health["models"]))
        if args.model not in health["models"]:
            _bad(f"requested model {args.model!r} is NOT in the served list")
            failures.append("requested-model-missing")
    else:
        _bad(f"endpoint unreachable  ·  {health['base_url']}")
        _info(health["error"] or "(no error)")
        _info("Did you `source .env`? Default is " + llm_client.DEFAULT_LLM_URL)
        return 1

    _hr(f"2/3  generation  ·  model={args.model}")
    prompt = (
        "Write a synthesizable SystemVerilog module named adder that adds two "
        "8-bit inputs a and b into a 9-bit output sum."
    )
    t0 = time.monotonic()
    try:
        outs = llm_client.generate_strict(
            prompt, model=args.model, n=1, temp=0.2, max_tokens=256
        )
        dt_ms = (time.monotonic() - t0) * 1000.0
        _ok(f"generate_strict returned {len(outs)} completion(s) in {dt_ms:.0f} ms")
        snippet = outs[0].strip().splitlines()[:8]
        _info("first lines of the response:")
        for line in snippet:
            _info("  " + line)
        if "endmodule" not in outs[0]:
            _bad("response did not contain `endmodule` -- model may be misbehaving")
            failures.append("generation-missing-endmodule")
    except Exception as exc:  # noqa: BLE001
        _bad(f"generate_strict raised: {type(exc).__name__}: {exc}")
        failures.append("generation-exception")
        return 1

    if args.skip_repair:
        _hr("3/3  repair  ·  SKIPPED (--skip-repair)")
    else:
        _hr(f"3/3  repair round-trip  ·  model={args.model}")
        broken_rtl = textwrap.dedent(
            """
            module counter (
                input  logic       clk,
                input  logic       rst_n,
                output logic [7:0] count
            );
              logic [3:0] step;
              assign step = 4_d1;
              always_ff @(posedge clk or negedge rst_n) begin
                if (!rst_n) count <= 8_d0;
                else        count <= count + step;
              end
            endmodule
            """
        ).strip()
        broken_err = (
            "%Error: counter.sv:8: syntax error, unexpected '_', expecting '''\n"
            "%Error: Exiting due to errors"
        )
        try:
            res = repair_agent.repair(broken_rtl, broken_err, model=args.model)
        except Exception as exc:  # noqa: BLE001
            _bad(f"repair() raised: {type(exc).__name__}: {exc}")
            failures.append("repair-exception")
            return 1

        if res.used_llm:
            latency = f" in {res.elapsed_ms:.0f} ms" if res.elapsed_ms else ""
            _ok(f"live LLM repair{latency}")
        else:
            _bad("repair fell back to the rule-based map (LLM did not respond)")
            _info(res.error or "(no upstream error captured)")
            failures.append("repair-fallback")

        _info("explanation:")
        for line in (res.explanation or "(empty)").splitlines()[:6]:
            _info("  " + line)

        if res.fixed_rtl:
            _ok("model returned a fenced ```systemverilog code block")
            head = res.fixed_rtl.splitlines()[:4]
            for line in head:
                _info("  " + line)
        else:
            _bad("no SystemVerilog code block in the response")
            failures.append("repair-no-code-block")

    print()
    if failures:
        print("\033[31mSMOKE TEST FAILED\033[0m  ·  " + "  ".join(failures))
        return 1
    print("\033[32mSMOKE TEST PASSED\033[0m  ·  endpoint, generation, repair all good.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
