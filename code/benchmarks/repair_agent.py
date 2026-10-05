"""repair_agent -- explain (and propose a fix for) a tool error on RTL.

This is the harness's default repair arm: it receives the failing RTL plus the
Verilator/tool error text and returns a short plain-English explanation plus a
suggested SystemVerilog fix. The fix is parsed out of a fenced
``systemverilog`` code block so the caller can re-lint it.

Lives in ``benchmarks/`` (not ``backend/``) because it is harness logic --
prompt construction, response parsing, repair policy. Its model-specific
counterparts are in ``prompts.py`` (OriGen, VeriReason native templates);
``backend/`` holds only serving/inference infrastructure. Every arm scored in
the paper routes through either this module or ``prompts.py``.

Behaviour:
  * The primary path calls ``llm_client.generate_strict(...)`` (no silent stub
    fallback), so a 401 / unreachable endpoint is observable to the caller
    instead of being papered over.
  * If the LLM call fails (no server, no key, returned no code block), we
    fall back to a rule-based regex map over the error text. The fallback
    can still produce a useful explanation but won't propose a fix.

Public API:
  * ``repair(rtl, error, model="tuned") -> RepairResult`` -- structured.
  * ``explain_error(rtl, error, model="tuned") -> str`` -- markdown string;
    used by ``run_repair_eval.py`` and ``run_ablation_tooluse.py``.
  * ``extract_code_block(text) -> str | None`` -- pulls the fix out of a
    fenced block (used internally by ``_split_explanation_and_code``).
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Tuple

from backend import llm_client

# ---------------------------------------------------------------------------
# Rule-based fallback (only used when the LLM is unavailable).
# (regex over the error text, explanation) -- checked in order.
# ---------------------------------------------------------------------------
_KNOWN_PATTERNS: List[Tuple["re.Pattern[str]", str]] = [
    (
        re.compile(r"PINMISSING|pin\s+missing", re.IGNORECASE),
        "A module instantiation is missing a port connection. Verilator flags "
        "every unconnected pin. Connect the named port (or explicitly tie it off "
        "with `.port()`), then re-lint.",
    ),
    (
        re.compile(r"WIDTH|width mismatch|bits", re.IGNORECASE),
        "There is a bit-width mismatch between an assignment's left- and "
        "right-hand sides. Size the literal or operand to match (e.g. use "
        "`ACC_WIDTH'(product)` when widening), or fix the declared width.",
    ),
    (
        re.compile(r"UNDRIVEN|not driven", re.IGNORECASE),
        "A net is declared but never driven. Add the missing `assign`/`always` "
        "driver, or remove the unused net so the design elaborates cleanly.",
    ),
    (
        re.compile(r"IMPLICIT|implicit (?:wire|definition)|signal.*undeclared", re.IGNORECASE),
        "A signal is used before it is declared. With `\\`default_nettype none` "
        "Verilator will not auto-create wires; declare the signal explicitly.",
    ),
    (
        re.compile(r"syntax error|unexpected", re.IGNORECASE),
        "Verilator hit a syntax error. Check the reported line for a missing "
        "semicolon, unbalanced `begin`/`end`, or a SystemVerilog keyword used in "
        "a Verilog-only context (lint with `-sv`).",
    ),
]


# ---------------------------------------------------------------------------
# Prompting
# ---------------------------------------------------------------------------
_REPAIR_SYSTEM = (
    "You are an expert hardware design assistant. The user reports a tool "
    "error on a SystemVerilog module. First write ONE short paragraph "
    "explaining the cause of the error in plain English. Then output the "
    "complete corrected SystemVerilog inside a single fenced code block "
    "tagged ```systemverilog. Do not add commentary after the code block. "
    "Preserve the original module name and port list unless the error "
    "specifically requires changing them."
)


def _build_prompt(
    rtl: str, error: str, spec: Optional[str] = None, locate: Optional[str] = None
) -> str:
    rtl_text = (rtl or "").strip() or "(no RTL provided)"
    err_text = (error or "").strip() or "(no error text provided)"
    spec_block = ""
    if spec and spec.strip():
        spec_block = (
            "The module is supposed to implement this specification:\n\n"
            "{spec}\n\n"
        ).format(spec=spec.strip())
    loc_block = ""
    if locate and locate.strip():
        loc_block = (
            "A localizer has pinpointed the bug to this exact line:\n\n"
            "```\n{loc}\n```\n\n"
        ).format(loc=locate.strip())
    return (
        "{spec_block}{loc_block}"
        "A verification tool reported the following problem with this "
        "SystemVerilog module:\n\n"
        "```\n{err}\n```\n\n"
        "The failing module is:\n\n"
        "```systemverilog\n{rtl}\n```\n\n"
        "Explain the cause and return the fixed module."
    ).format(spec_block=spec_block, loc_block=loc_block, err=err_text, rtl=rtl_text)


# ---------------------------------------------------------------------------
# Code-block extraction
# ---------------------------------------------------------------------------
_TAGGED_BLOCK_RE = re.compile(
    r"```(?:systemverilog|verilog|sv)\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_PLAIN_BLOCK_RE = re.compile(r"```\s*\n(.*?)```", re.DOTALL)


def extract_code_block(text: str) -> Optional[str]:
    """Pull out the first SystemVerilog/Verilog code block from ``text``.

    Prefers blocks tagged ``systemverilog`` / ``verilog`` / ``sv``. Falls back
    to any plain ``` block if it looks like Verilog (mentions ``module``).
    """
    if not text:
        return None
    m = _TAGGED_BLOCK_RE.search(text)
    if m:
        return m.group(1).rstrip()
    m2 = _PLAIN_BLOCK_RE.search(text)
    if m2:
        cand = m2.group(1).rstrip()
        if "module" in cand or "endmodule" in cand:
            return cand
    return None


def _split_explanation_and_code(text: str) -> Tuple[str, Optional[str]]:
    """Return ``(explanation, fixed_rtl_or_None)`` parsed from a model reply."""
    code = extract_code_block(text)
    if code is None:
        return (text or "").strip(), None
    pre = re.split(r"```", text, maxsplit=1)[0].strip()
    if not pre:
        pre = "(model returned the corrected module without an explanation)"
    return pre, code


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
@dataclass
class RepairResult:
    """Structured repair output for the IDE / chat orchestrator."""

    explanation: str
    fixed_rtl: Optional[str]
    model: str
    used_llm: bool
    error: Optional[str] = None
    elapsed_ms: Optional[float] = None


def _regex_fallback(err: str, model: str, llm_error: Optional[str]) -> RepairResult:
    """Build a RepairResult from the rule-based explanation map."""
    explanation: Optional[str] = None
    for pat, msg in _KNOWN_PATTERNS:
        if pat.search(err):
            explanation = msg
            break
    if explanation is None:
        explanation = (
            "The tool reported an error that the rule-based fallback does not "
            "recognise. Read the first `%Error` line (file:line) and address "
            "that location first; cascading errors usually clear once the "
            "first is fixed."
        )
    first_line = err.splitlines()[0] if err.splitlines() else err
    body = "**Error:** `{0}`\n\n**Likely cause:** {1}".format(first_line, explanation)
    return RepairResult(
        explanation=body,
        fixed_rtl=None,
        model=model,
        used_llm=False,
        error=llm_error,
    )


def repair(
    rtl: str,
    error: str,
    model: str = "tuned",
    temperature: float = 0.2,
    spec: Optional[str] = None,
    locate: Optional[str] = None,
) -> RepairResult:
    """LLM-driven repair. Returns a RepairResult with explanation + fixed RTL.

    ``spec`` optionally injects the design's natural-language intent (used by
    the spec-conditioned repair ablation). ``temperature`` is exposed so callers
    can run deterministic (greedy) paired comparisons.

    Falls back to the rule-based regex map only when the live LLM call fails
    (network, auth, empty response). The fallback never invents code -- it
    just explains the error class.
    """
    err = (error or "").strip()
    if not err:
        return RepairResult(
            explanation=(
                "No error text was provided. If Verilator reported a failure, "
                "paste the `%Error`/`%Warning` lines so the repair agent can "
                "localise the fix."
            ),
            fixed_rtl=None,
            model=model,
            used_llm=False,
        )

    try:
        outputs = llm_client.generate_strict(
            _build_prompt(rtl, err, spec=spec, locate=locate),
            model=model,
            n=1,
            temp=temperature,
            max_tokens=int(os.environ.get("RTLREPAIR_REPAIR_MAX_TOKENS", "1024")),
            system=_REPAIR_SYSTEM,
        )
    except Exception as exc:  # noqa: BLE001
        return _regex_fallback(err, model, "{0}: {1}".format(type(exc).__name__, exc))

    text = outputs[0] if outputs else ""
    explanation, fixed_rtl = _split_explanation_and_code(text)
    if not text.strip():
        return _regex_fallback(err, model, "empty response")

    status = llm_client.last_call_status()
    return RepairResult(
        explanation=explanation,
        fixed_rtl=fixed_rtl,
        model=model,
        used_llm=True,
        error=None,
        elapsed_ms=status.get("elapsed_ms"),
    )


def explain_error(rtl: str, error: str, model: str = "tuned") -> str:
    """Backwards-compatible wrapper: returns a markdown explanation string.

    Used by ``app.py`` (Streamlit verification tab) and
    ``benchmarks/run_repair_eval.py``. Internally calls :func:`repair`, then
    formats the structured result as one self-contained markdown blob.
    """
    res = repair(rtl, error, model=model)
    parts: List[str] = []
    if res.used_llm:
        latency = (
            " ({0:.0f} ms)".format(res.elapsed_ms) if res.elapsed_ms else ""
        )
        parts.append(
            "**Repair agent (`{0}` -- live LLM){1}**\n".format(res.model, latency)
        )
    else:
        parts.append("**Repair agent (`{0}` -- rule-based fallback)**\n".format(res.model))
        if res.error:
            parts.append("_LLM unavailable: {0}_\n".format(res.error))
    parts.append(res.explanation)
    if res.fixed_rtl:
        parts.append("\n**Suggested fix:**\n")
        parts.append("```systemverilog\n{0}\n```".format(res.fixed_rtl))
    return "\n".join(parts).strip()
