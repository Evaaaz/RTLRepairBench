#!/usr/bin/env python3
"""Generate and verify the exact-prompts appendix from the live builders.

Imports the prompt builders from the consolidated harness and renders them with a small
worked example, so the prompts printed in the paper are the prompts the code sends. Nothing
is transcribed by hand; re-run after any prompt change and the appendix follows.

The ordinary CLI writes under ``build/paper-inputs``.  ``--check`` regenerates
in memory and byte-compares against the frozen ``paper/app_prompts.tex``.
"""
import argparse
import os
import sys
import tempfile
from pathlib import Path

_CODE_DIR = Path(__file__).resolve().parent.parent
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))
from project_paths import output_root, project_root  # noqa: E402

# The split the rest of the bundle uses: configs and inputs come from the bundle, the
# heavy run tree is reached through RTLREPAIR_OUT. Neither encodes a nesting depth.
SOURCE_ROOT = str(project_root())
ROOT = str(output_root().parent)
CODE = Path(ROOT) / "code"
for p in (CODE, CODE / "benchmarks", CODE / "datagen", CODE / "scripts"):
    sys.path.insert(0, str(p))

import vericodegen_eval as VE          # frozen primary: 2x2 arms + counterexample pair
from benchmarks import repair_agent as RA  # tool-in-the-loop repair (Table 3)
import realbug_frontier as RB           # real-bug three arms (Table 6)
import second_model_redact as RD        # redaction control (Section 5)
import second_model_explore as SE       # post hoc capability-curve 2x2 arms

DEFAULT_OUTPUT = Path(ROOT) / "build" / "paper-inputs" / "app_prompts.tex"
FROZEN_OUTPUT = Path(ROOT) / "paper" / "app_prompts.tex"

BROKEN = "module TopModule(input clk, input rst_n, input a, output reg z);\n" \
         "  always @(posedge clk)\n    if (rst_n) z <= 1'b0;\n    else z <= a;\nendmodule"
SPEC = "z is registered a, cleared while the active-low reset rst_n is asserted."
LINE_NO, LINE = 3, "if (rst_n) z <= 1'b0;"
ERROR = "%Warning-WIDTH: TopModule.sv:4: Operator ASSIGNDLY expects 1 bit on the Assign RHS."
PREV = BROKEN.replace("if (rst_n)", "if (!rst_n)").replace("z <= a;", "z <= 1'b1;")
WITNESS = "cycle 0: rst_n=0 a=1 | expected z=0 got z=1\ncycle 1: rst_n=1 a=1 | expected z=1 got z=1"


class Case:
    """Minimal stand-in for the harness's ExperimentCase, same attribute names."""
    case_id = "main_0001"
    broken_rtl = BROKEN
    previous_candidate = PREV
    counterexample = WITNESS
    location_line_number = LINE_NO
    location_line = LINE


def verbatim(title, system, user, note=""):
    body = ""
    if system:
        body += "\\textit{System.}\n\\begin{Verbatim}[fontsize=\\scriptsize,breaklines,breakanywhere]\n" \
                + system.strip() + "\n\\end{Verbatim}\n"
    if user:
        body += "\\textit{User.}\n\\begin{Verbatim}[fontsize=\\scriptsize,breaklines,breakanywhere]\n" \
                + user.strip() + "\n\\end{Verbatim}\n"
    return f"\\paragraph{{{title}}}\n{note}\n{body}\n"


def render() -> str:
    case = Case()
    blocks = []

    blocks.append(verbatim(
        "RepairBench no-spec arm (Table~\\ref{tab:recovery}).",
        RA._REPAIR_SYSTEM, RA._build_prompt(BROKEN, ERROR),
        "The diagnostic is the verbatim tool output that defined the bug. This is a one-shot "
        "repair prompt; the returned module is post-scored by the class-appropriate gate."))

    _rb_base = RA._build_prompt(BROKEN, ERROR).splitlines()
    blocks.append(verbatim(
        "RepairBench specification arm (Table~\\ref{tab:recovery}).", None,
        "\n".join(l for l in RA._build_prompt(BROKEN, ERROR, spec=SPEC).splitlines()
                  if l.strip() and l not in _rb_base),
        "Same system message and same builder as the arm above; only these lines are added. "
        "RepairBench's builder words things differently from the decomposition harness, which "
        "is why the two are shown separately."))

    blocks.append(verbatim(
        "Decomposition arms, frozen primary (Table~\\ref{tab:primary}).",
        VE.REPAIR_SYSTEM,
        VE._repair_prompt(case, specification=None, include_location=False),
        "Arm $\\mathrm{spec}0\\,\\mathrm{loc}0$, the no-signal baseline. The three remaining "
        "arms add the blocks below, unchanged otherwise."))

    curve_location = {"line_number": LINE_NO, "broken_line": LINE}
    curve_prompts = {
        (use_spec, use_location): SE.repair_prompt(
            BROKEN, SPEC, curve_location, use_spec, use_location
        )
        for use_spec in (False, True)
        for use_location in (False, True)
    }
    primary_prompts = {
        (use_spec, use_location): VE._repair_prompt(
            case,
            specification=(SPEC if use_spec else None),
            include_location=use_location,
        )
        for use_spec in (False, True)
        for use_location in (False, True)
    }
    if curve_prompts != primary_prompts:
        raise RuntimeError("primary and capability-curve user prompts have drifted")
    blocks.append(verbatim(
        "Decomposition arms, post hoc capability curve (Figure~\\ref{fig:curve}).",
        SE.REPAIR_SYSTEM, None,
        "The four user messages are byte-identical to the frozen-primary messages above, "
        "which this script asserts on every run. Only the system message differs, because the "
        "historical curve runner renders it as a single line."))

    what = VE._repair_prompt(case, specification=SPEC, include_location=False)
    where = VE._repair_prompt(case, specification=None, include_location=True)
    blocks.append(verbatim(
        "Shared What (specification) block, added for $\\mathrm{spec}1$.", None,
        "\n".join(l for l in what.splitlines() if l.strip() and l not in
                  VE._repair_prompt(case, specification=None, include_location=False).splitlines())))
    blocks.append(verbatim(
        "Shared Where (oracle localization) block, added for $\\mathrm{loc}1$.", None,
        "\n".join(l for l in where.splitlines() if l.strip() and l not in
                  VE._repair_prompt(case, specification=None, include_location=False).splitlines()),
        "The line number and the original broken line only; never the fix."))

    blocks.append(verbatim(
        "Counterexample revision pair (Table~\\ref{tab:primary}, third contrast).",
        VE.REPAIR_SYSTEM, VE._feedback_prompt(case, concrete=True),
        "The concrete arm. The generic arm is byte-identical except that the trace block is "
        "replaced by the single sentence ``The previous candidate still did not pass "
        "verification.'', which is what isolates the counterexample from re-exposure."))

    rec = {"broken_rtl": BROKEN, "spec": SPEC}
    blocks.append(verbatim(
        "Mined-failure arms (Table~\\ref{tab:multimodel}).",
        "You repair broken SystemVerilog.", RB.build_user(rec, False, True),
        "This is \\code{nospec}. The \\code{spec} arm is the same message with the specification "
        "block appended, and \\code{speconly} is the same message with the broken RTL omitted "
        "and the specification kept, so it carries no module to edit and no divergence trace."))

    blocks.append(verbatim(
        "Redaction control (\\S\\ref{sec:c2realbug}).", RD.RUBRIC,
        "(the specification to be redacted)",
        "Applied to the specification only. The redacted specifications are not released; "
        "only aggregate summaries are included in this artifact."))

    tex = ("% GENERATED by scripts/ws_i_prompts_appendix.py -- do not edit by hand.\n"
           "% Prompts are imported from the harness, so they cannot drift from the code.\n"
           "\\section{Exact prompts}\\label{app:prompts}\n"
           "Every prompt below is rendered by importing the harness's own builder and filling it "
           "with one small worked example, so what appears here is what the code sends. "
           "Module text is abbreviated for space; nothing else is edited.\n\n"
           + "\n".join(blocks))
    return tex


def write_atomic(path: Path, payload: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--output",
        type=Path,
        help="write the rendered appendix here (default: build/paper-inputs/app_prompts.tex)",
    )
    action.add_argument(
        "--check",
        action="store_true",
        help="do not write; byte-compare the rendered appendix with paper/app_prompts.tex",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    tex = render()
    block_count = tex.count("\\paragraph{")
    if args.check:
        if not FROZEN_OUTPUT.is_file():
            raise FileNotFoundError(f"frozen prompt appendix is missing: {FROZEN_OUTPUT}")
        if tex.encode("utf-8") != FROZEN_OUTPUT.read_bytes():
            raise RuntimeError("regenerated prompt appendix differs from paper/app_prompts.tex")
        print(
            f"verified {os.path.relpath(FROZEN_OUTPUT, ROOT)} "
            f"({block_count} prompt blocks, {len(tex.splitlines())} lines)"
        )
        return 0
    output = (args.output or DEFAULT_OUTPUT).resolve()
    write_atomic(output, tex)
    print(
        f"wrote {os.path.relpath(output, ROOT)} "
        f"({block_count} prompt blocks, {len(tex.splitlines())} lines)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
