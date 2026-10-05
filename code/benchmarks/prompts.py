"""Prompt registry for the repair eval.

Different repair models were trained against different prompt formats. Feeding a
model a template it never saw runs it off-distribution and posts an artifactually
low number, so each model gets scored with *its own* template.

- ``default`` — the harness's generic repair prompt (``repair_agent``).
- ``origen`` — OriGen_Fix's native ``### Instruction ... ### Response`` template
  (https://huggingface.co/henryen/OriGen_Fix). Its format REQUIRES a task
  description, so OriGen is naturally a diagnostic+spec arm.
- ``verireason`` — VeriReason's native ``<think>…</think><answer>…</answer>``
  format (https://huggingface.co/Nellyw888/VeriReason-codeLlama-7b-RTLCoder-
  Verilog-GRPO-reasoning-tb). Its native task is spec→RTL generation, not
  repair, so repair is off-distribution under any prompt; this template stays
  in the model card's dialect to remove the *format* mismatch only.

``PROMPTERS`` is the set of valid ``--prompt-style`` choices.
"""
from __future__ import annotations

import re
from typing import Optional, Tuple

PROMPTERS = {"default", "origen", "verireason"}

# --- OriGen_Fix native template ------------------------------------------------
# Verbatim structure from the model card. Served as chat, we place the whole
# instruction block as the user turn and drop the trailing ``### Response:{header}``
# priming (the assistant turn is the completion). {description}=spec,
# {original_code}=broken RTL, {error}=tool diagnostic.
_ORIGEN_INSTRUCTION = (
    "Please act as a professional Verilog designer. Your task is to debug a "
    "Verilog module.\n"
    "You will receive a task, an original verilog code with syntax and function "
    "errors, and the corresponding error messages.\n"
    "You should generate a corrected code based on the original code and the "
    "error messages.\n"
    "Your task:\n{description}\n"
    "Original code:\n{original_code}\n"
    "Error message:\n{error}\n"
    "You should now generate a correct code."
)

_ORIGEN_NO_SPEC = (
    "Repair the following Verilog module so it is functionally correct. "
    "(No natural-language specification was provided.)"
)


def origen_prompt(broken: str, err: str, spec: Optional[str] = None,
                  locate: Optional[str] = None) -> Tuple[str, str]:
    """Return (system, user) for OriGen_Fix.

    OriGen's instruction is self-contained, so the system message is minimal.
    Oracle localization (if any) is appended to the error text, since the native
    template has no dedicated slot for it.
    """
    description = (spec or "").strip() or _ORIGEN_NO_SPEC
    err_text = (err or "").strip() or "(no error message provided)"
    if locate and locate.strip():
        err_text += f"\nThe faulty line has been localized to: {locate.strip()}"
    user = _ORIGEN_INSTRUCTION.format(
        description=description,
        original_code=(broken or "").strip(),
        error=err_text,
    )
    system = "You are a professional Verilog designer."
    return system, user


# --- VeriReason native template ------------------------------------------------
# Dialect from the model card: prompts open with "Please act as a professional
# verilog designer." and outputs are <think>reasoning</think> then <answer> with
# a ```verilog fence. Served raw-completion (custom jinja concatenates
# system + "\n" + user), so the system line reproduces the card's opener.
_VERIREASON_INSTRUCTION = (
    "Your task is to debug a Verilog module.\n"
    "You will receive an original verilog code with errors and the "
    "corresponding error message.\n"
    "Fix the code so the error no longer occurs, keeping the module interface "
    "unchanged.\n"
    "{spec_block}"
    "Original code:\n{original_code}\n"
    "Error message:\n{error}\n"
    "First reason about the bug inside <think> </think> tags, then output the "
    "complete corrected module inside <answer> </answer> tags in a "
    "```verilog code fence."
)


def verireason_prompt(broken: str, err: str, spec: Optional[str] = None,
                      locate: Optional[str] = None) -> Tuple[str, str]:
    """Return (system, user) for VeriReason's think/answer format.

    The spec slot is optional (unlike OriGen's): VeriReason's training prompts
    always carried a task description, so the +spec arm is its natural
    comparison, but a diagnostic-only arm is still well-formed.
    """
    err_text = (err or "").strip() or "(no error message provided)"
    if locate and locate.strip():
        err_text += f"\nThe faulty line has been localized to: {locate.strip()}"
    spec_block = f"Your task:\n{spec.strip()}\n" if spec and spec.strip() else ""
    user = _VERIREASON_INSTRUCTION.format(
        spec_block=spec_block,
        original_code=(broken or "").strip(),
        error=err_text,
    )
    system = "Please act as a professional verilog designer."
    return system, user


_THINK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_ANSWER = re.compile(r"<answer>(.*?)(?:</answer>|$)", re.IGNORECASE | re.DOTALL)


def extract_verireason(text: Optional[str]) -> Optional[str]:
    """Pull the module out of <answer>…</answer>, tolerating truncated output.

    A completion cut off at max_tokens may lack the closing </answer> (the
    ``$`` alternative) or the <answer> block entirely; in the latter case the
    reasoning is stripped so fenced code inside <think> is never mistaken for
    the fix.
    """
    if not text:
        return None
    m = _ANSWER.search(text)
    return extract_module(m.group(1) if m else _THINK.sub("", text))


# --- lenient module extraction -------------------------------------------------
_FENCED = re.compile(r"```(?:systemverilog|verilog|sv)?\s*\n(.*?)```", re.IGNORECASE | re.DOTALL)
_MODULE = re.compile(r"\bmodule\b.*?\bendmodule\b", re.IGNORECASE | re.DOTALL)


def extract_module(text: str) -> Optional[str]:
    """Pull a Verilog module out of a raw completion.

    OriGen emits raw Verilog after ``### Response:`` (no code fence), so the
    fenced-block extractors used elsewhere miss it. Prefer a fenced block if
    present; otherwise take the first ``module ... endmodule`` span.
    """
    if not text:
        return None
    m = _FENCED.search(text)
    if m and "module" in m.group(1):
        return m.group(1).rstrip() + "\n"
    m2 = _MODULE.search(text)
    if m2:
        return m2.group(0).rstrip() + "\n"
    return None


# --- semantic-repair 2x2 grammar (spec x locate) -------------------------------
# One generic prompt with two independent levers: whether the behavioral spec is
# supplied, and whether the buggy line is pointed at. Unlike the templates above,
# these cases lint and compile clean — there is no tool diagnostic to give, which
# is why the `diag*` signal grammar does not apply to them.
#
# Frozen — the banked llama-3.3-70b / gpt-4.1 / gpt-4o-mini / OriGen_Fix runs
# were generated from this exact text, so editing it invalidates comparison
# against them. Lives here rather than in a driver so the generation script and
# the ledger writer cannot drift; scripts/second_model_explore.py imports it.

SPEC_LOCATE_SYSTEM = ("You are an expert RTL debugging engineer. Repair the supplied "
  "compile-clean SystemVerilog module. Return the complete replacement module, "
  "from `module` through `endmodule`; never return a patch, explanation, testbench, "
  "or placeholder. Preserve the given module interface and emit synthesizable RTL.")

# (arm_name, use_spec, use_locate)
SPEC_LOCATE_ARMS = [("spec0_loc0", False, False), ("spec1_loc0", True, False),
                    ("spec0_loc1", False, True), ("spec1_loc1", True, True)]


def spec_locate_prompt(broken, spec, loc, use_spec, use_loc):
    """The 2x2 semantic-repair user turn. `loc` is {line_number, broken_line}."""
    parts = ["Repair the following compile-clean SystemVerilog module, which fails functional verification.",
             "\nBuggy module:\n```systemverilog\n" + broken.strip() + "\n```"]
    if use_spec:
        parts.append("\nBehavioral specification:\n" + spec.strip())
    if use_loc:
        parts.append("\nOracle localization (this identifies the buggy source line but does not give the fix):\n"
                     f"Line {loc['line_number']}: `{loc['broken_line']}`")
    parts.append("\nReturn only the complete corrected `TopModule`, including its unchanged interface.")
    return "\n".join(parts)
