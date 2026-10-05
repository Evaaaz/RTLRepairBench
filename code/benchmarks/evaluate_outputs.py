"""evaluate_outputs -- compile + simulate a generated RTL module against a testbench.

The scoring leg of the benchmark harness. Given a generated module (.sv) and a
matching testbench (.sv), plus the testbench's ``top_module`` and a
``pass_regex`` that must match the simulator's stdout, this module reports:

    {"compile_pass": bool, "test_pass": Optional[bool], "tool": str, "log": str}

Scoring strategy (in order of preference):
  1. If ``iverilog`` + ``vvp`` are on PATH: compile both files and run the
     simulation. ``compile_pass`` is true iff iverilog returns 0; ``test_pass``
     is true iff vvp returns 0 AND ``pass_regex`` matches stdout (and no FAIL
     line appears). This is the MEASURED path.
  2. If only ``verilator`` is present: ``--lint-only`` gives ``compile_pass``;
     simulation is not run so ``test_pass`` is None.
  3. If NO tools are present: ``compile_pass`` is estimated with a cheap bracket
     / module-keyword sanity check and ``test_pass`` is None. Honest, offline.

The module is import-safe (pure stdlib) and never raises on missing inputs:
a missing file degrades to a failed/None result with an explanatory log.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from typing import Dict, Optional

# Default regex the systolic testbench (and most of our suite) prints on success.
DEFAULT_PASS_REGEX = r"\bPASS\b"
# Lines that unambiguously mean the test failed even if rc happens to be 0.
_FAIL_REGEX = re.compile(r"\bFAIL\b", re.IGNORECASE)
DEFAULT_TIMEOUT_S = 60


def _read_text(path: Optional[str]) -> Optional[str]:
    """Read a file as text, returning None on any problem (missing/binary)."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            return fh.read()
    except OSError:
        return None


def _bracket_compile_check(module_src: Optional[str]) -> bool:
    """Cheap offline proxy for ``compile_pass`` when no tools are installed.

    Heuristic: the source must declare a module, terminate it with ``endmodule``,
    have balanced ()/{}/[] brackets, and balanced module/endmodule and
    begin/end counts. This is intentionally conservative -- it catches obvious
    truncation / brace-mismatch failures without a real compiler.
    """
    if not module_src or not module_src.strip():
        return False
    src = module_src
    # Strip line + block comments so brackets inside them don't skew the count.
    src = re.sub(r"//[^\n]*", "", src)
    src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
    if "module" not in src or "endmodule" not in src:
        return False
    for open_c, close_c in (("(", ")"), ("{", "}"), ("[", "]")):
        if src.count(open_c) != src.count(close_c):
            return False
    # Keyword balance (word-boundary so `endmodule` doesn't match `module`).
    n_module = len(re.findall(r"\bmodule\b", src))
    n_endmodule = len(re.findall(r"\bendmodule\b", src))
    if n_module == 0 or n_module != n_endmodule:
        return False
    n_begin = len(re.findall(r"\bbegin\b", src))
    n_end = len(re.findall(r"\bend\b", src))
    if n_begin != n_end:
        return False
    return True


def _run(cmd, cwd, timeout_s):
    """Run a subprocess, returning (rc, stdout, stderr); never raises."""
    try:
        proc = subprocess.run(
            cmd,
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return proc.returncode, proc.stdout or "", proc.stderr or ""
    except subprocess.TimeoutExpired as exc:
        out = exc.stdout or ""
        if isinstance(out, bytes):
            out = out.decode("utf-8", "replace")
        return 124, out, "TIMEOUT after {0}s".format(timeout_s)
    except Exception as exc:  # OSError etc.
        return 127, "", "invocation failed: {0}".format(exc)


def _result(compile_pass, test_pass, tool, log, measured=True):
    return {
        "compile_pass": bool(compile_pass),
        "test_pass": test_pass,  # may be None
        "tool": tool,
        "log": log,
        "is_measured": bool(measured),
    }


def evaluate_outputs(
    module_path: str,
    testbench_path: Optional[str] = None,
    top_module: Optional[str] = None,
    pass_regex: Optional[str] = None,
    timeout_s: int = DEFAULT_TIMEOUT_S,
) -> Dict:
    """Compile + simulate ``module_path`` against ``testbench_path``.

    Args:
        module_path: path to the generated module .sv/.v.
        testbench_path: path to the self-checking testbench .sv/.v. If absent,
            only a compile/lint check of the module is performed (test_pass=None).
        top_module: name of the simulation top (the testbench module). Passed to
            iverilog with ``-s`` when provided so the right top is elaborated.
        pass_regex: regex that must match simulator stdout for ``test_pass``.
            Defaults to ``\\bPASS\\b``. Ignored when no simulation is run.
        timeout_s: per-tool wall-clock timeout.

    Returns:
        dict {compile_pass, test_pass, tool, log, is_measured}. Robust to
        missing files and missing tools -- never raises.
    """
    module_src = _read_text(module_path)
    if module_src is None:
        return _result(False, None, "none",
                       "module file not found: {0}".format(module_path),
                       measured=False)

    regex = pass_regex or DEFAULT_PASS_REGEX
    try:
        pass_pat = re.compile(regex)
    except re.error:
        pass_pat = re.compile(re.escape(regex))

    have_iverilog = bool(shutil.which("iverilog")) and bool(shutil.which("vvp"))
    have_verilator = bool(shutil.which("verilator"))
    tb_src = _read_text(testbench_path)

    # ---- Path 1: iverilog + vvp (full compile + simulate) ----
    if have_iverilog:
        workdir = tempfile.mkdtemp(prefix="rtlrepair_sim_")
        try:
            mod_file = os.path.join(workdir, "dut.sv")
            with open(mod_file, "w", encoding="utf-8") as fh:
                fh.write(module_src)

            sources = [mod_file]
            if tb_src is not None:
                tb_file = os.path.join(workdir, "tb.sv")
                with open(tb_file, "w", encoding="utf-8") as fh:
                    fh.write(tb_src)
                # Testbench first so an unguarded top is picked up naturally.
                sources = [tb_file, mod_file]

            out_bin = os.path.join(workdir, "a.out")
            compile_cmd = ["iverilog", "-g2012", "-o", out_bin]
            if top_module:
                compile_cmd += ["-s", top_module]
            compile_cmd += sources

            rc, cout, cerr = _run(compile_cmd, workdir, timeout_s)
            compile_log = "$ {0}\n{1}\n{2}".format(" ".join(compile_cmd), cout, cerr)
            compile_pass = rc == 0

            if not compile_pass:
                return _result(False, (None if tb_src is None else False),
                               "iverilog", compile_log, measured=True)

            # No testbench -> compile-only score.
            if tb_src is None:
                return _result(True, None, "iverilog",
                               compile_log + "\n(no testbench -- compile only)",
                               measured=True)

            # Simulate.
            rc_sim, sout, serr = _run(["vvp", out_bin], workdir, timeout_s)
            sim_log = "$ vvp a.out\n{0}\n{1}".format(sout, serr)
            matched = bool(pass_pat.search(sout))
            failed = bool(_FAIL_REGEX.search(sout))
            test_pass = (rc_sim == 0) and matched and not failed
            return _result(True, test_pass, "iverilog+vvp",
                           compile_log + "\n" + sim_log, measured=True)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # ---- Path 2: verilator lint (compile_pass only) ----
    if have_verilator:
        workdir = tempfile.mkdtemp(prefix="rtlrepair_lint_")
        try:
            mod_file = os.path.join(workdir, "dut.sv")
            with open(mod_file, "w", encoding="utf-8") as fh:
                fh.write(module_src)
            cmd = ["verilator", "--lint-only", "-Wno-fatal"]
            if top_module:
                cmd += ["--top-module", top_module]
            cmd += [mod_file]
            rc, cout, cerr = _run(cmd, workdir, timeout_s)
            log = "$ {0}\n{1}\n{2}".format(" ".join(cmd), cout, cerr)
            return _result(rc == 0, None, "verilator(lint)",
                           log + "\n(verilator present, no simulator -- test_pass unknown)",
                           measured=True)
        finally:
            shutil.rmtree(workdir, ignore_errors=True)

    # ---- Path 3: no tools -- bracket heuristic, honest stub ----
    compile_pass = _bracket_compile_check(module_src)
    return _result(
        compile_pass,
        None,
        "bracket(stub)",
        "no iverilog/verilator on PATH -- compile_pass via bracket heuristic, "
        "test_pass unknown (None).",
        measured=False,
    )


def _cli():
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Compile+simulate an RTL module vs a testbench.")
    ap.add_argument("module", help="generated module .sv/.v")
    ap.add_argument("--testbench", "--tb", dest="testbench", default=None)
    ap.add_argument("--top-module", dest="top_module", default=None)
    ap.add_argument("--pass-regex", dest="pass_regex", default=None)
    ap.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT_S)
    args = ap.parse_args()
    res = evaluate_outputs(
        args.module,
        testbench_path=args.testbench,
        top_module=args.top_module,
        pass_regex=args.pass_regex,
        timeout_s=args.timeout,
    )
    print(json.dumps(res, indent=2))


if __name__ == "__main__":
    _cli()
