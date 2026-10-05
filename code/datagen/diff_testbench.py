"""Differential testbench gate for the semantic repair bucket.

A semantic bug compiles and lints clean — only its *behavior* is wrong — so the
Verilator gate cannot see it. Instead we verify it the way you'd test real
hardware: run the original (golden) module and the mutated one on identical
random stimulus and watch whether their outputs ever diverge. No hand-written,
per-module functional testbench is needed; the golden module *is* the reference.

Flow for one candidate pair:

    parse_module(source)            -> name, params, typed ports (or None if we
                                       can't safely parse/drive it)
    generate_tb(info)               -> a Verilog TB instantiating <name>_good and
                                       <name>_bad, driving random inputs, counting
                                       per-output mismatches
    run_diff_test(good, bad, info)  -> iverilog compile + vvp run; DIFF_FAIL means
                                       the bug is real and simulation-detectable

We deliberately keep only pairs the harness can drive: modules with parseable
ANSI ports, no unpacked-array ports, and at least one input and one output. Seeds
that don't fit are skipped — with thousands of seeds that's fine.
"""

from __future__ import annotations

import re
import secrets
import shutil
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from candidate_security import candidate_security_reason, input_port_drive_reason

CYCLES = 200  # random stimulus cycles per differential run


@dataclass
class Port:
    name: str
    direction: str  # input | output | inout
    width: int
    is_clock: bool
    is_reset: bool


@dataclass
class ModuleInfo:
    name: str
    params: dict[str, int]
    ports: list[Port]

    @property
    def inputs(self) -> list[Port]:
        return [p for p in self.ports if p.direction == "input"]

    @property
    def outputs(self) -> list[Port]:
        return [p for p in self.ports if p.direction in ("output", "inout")]


# --- parsing ------------------------------------------------------------------

_MODULE_RE = re.compile(r"\bmodule\s+(\w+)")
_PARAM_RE = re.compile(r"parameter\s+(?:\w+\s+)?(\w+)\s*=\s*([^,]+)")
_CLK_RE = re.compile(r"cl(oc)?k", re.IGNORECASE)
_RST_RE = re.compile(r"rst|reset", re.IGNORECASE)


def _match_paren(s: str, open_idx: int) -> int:
    depth = 0
    for j in range(open_idx, len(s)):
        if s[j] == "(":
            depth += 1
        elif s[j] == ")":
            depth -= 1
            if depth == 0:
                return j
    return -1


def _safe_eval(expr: str, params: dict[str, int]) -> int | None:
    """Evaluate an int width expression with the module's params bound."""
    expr = expr.strip().rstrip(";").strip()
    expr = re.sub(r"\d*'[sS]?[dDhHbBoO]", "", expr)  # strip SV literal size prefixes
    try:
        return int(eval(expr, {"__builtins__": {}}, dict(params)))
    except Exception:
        return None


def _parse_params(text: str) -> dict[str, int]:
    params: dict[str, int] = {}
    for m in _PARAM_RE.finditer(text):
        val = _safe_eval(m.group(2), params)
        if val is not None:
            params[m.group(1)] = val
    return params


def _split_commas(text: str) -> list[str]:
    out, depth, cur = [], 0, ""
    for ch in text:
        if ch in "[({":
            depth += 1
        elif ch in "])}":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(cur)
            cur = ""
        else:
            cur += ch
    out.append(cur)
    return out


def _parse_ports(text: str, params: dict[str, int]) -> list[Port] | None:
    ports: list[Port] = []
    last_dir: str | None = None
    for raw in _split_commas(text):
        seg = raw.strip()
        if not seg:
            continue
        dirm = re.match(r"(input|output|inout)\b", seg)
        if dirm:
            last_dir = dirm.group(1)
            seg = seg[dirm.end() :]
        if last_dir is None:
            return None  # non-ANSI header (directions declared in body) — skip
        body = re.sub(r"\b(wire|reg|logic|var|signed|unsigned)\b", " ", seg)
        if re.search(r"\b\w+\s*\[[^\]]*\]\s*$", body):
            return None  # unpacked-array port — too hard to drive; skip seed
        nm = re.search(r"(\w+)\s*$", body)
        if not nm:
            return None
        rng = re.search(r"\[([^:]+):([^\]]+)\]", body)
        if rng:
            msb, lsb = _safe_eval(rng.group(1), params), _safe_eval(rng.group(2), params)
            if msb is None or lsb is None:
                return None
            width = abs(msb - lsb) + 1
        else:
            width = 1
        ports.append(
            Port(nm.group(1), last_dir, width, bool(_CLK_RE.search(nm.group(1))),
                 bool(_RST_RE.search(nm.group(1))))
        )
    return ports or None


def _strip_comments(s: str) -> str:
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)
    return re.sub(r"//[^\n]*", "", s)


def parse_module(source: str) -> ModuleInfo | None:
    """Parse an ANSI-header module into name/params/ports, or None if unsupported."""
    source = _strip_comments(source)
    m = _MODULE_RE.search(source)
    if not m:
        return None
    pos = m.end()
    params: dict[str, int] = {}
    hash_pos, paren_pos = source.find("#", pos), source.find("(", pos)
    if hash_pos != -1 and (paren_pos == -1 or hash_pos < paren_pos):
        popen = source.find("(", hash_pos)
        pclose = _match_paren(source, popen)
        if pclose == -1:
            return None
        params = _parse_params(source[popen + 1 : pclose])
        port_open = source.find("(", pclose + 1)
    else:
        port_open = paren_pos
    if port_open == -1:
        return None
    port_close = _match_paren(source, port_open)
    if port_close == -1:
        return None
    ports = _parse_ports(source[port_open + 1 : port_close], params)
    if ports is None:
        return None
    return ModuleInfo(m.group(1), params, ports)


# --- testbench generation -----------------------------------------------------

def _decl_width(w: int) -> str:
    return "" if w <= 1 else f"[{w - 1}:0] "


def generate_tb(
    info: ModuleInfo,
    cycles: int = CYCLES,
    protocol_nonce: str | None = None,
    compare_from_cycle: int = 0,
) -> str | None:
    """Emit a differential TB, or None if the module isn't drivable."""
    nonce = protocol_nonce or secrets.token_hex(16)
    if not re.fullmatch(r"[0-9a-f]{32}", nonce):
        raise ValueError("protocol_nonce must be 32 lowercase hexadecimal characters")
    ins, outs = info.inputs, info.outputs
    if not ins or not outs:
        return None
    driven = [p for p in ins if not p.is_clock]  # everything but the clock
    if not driven:
        return None
    clocks = [p for p in ins if p.is_clock]

    # TB-internal identifiers must not collide with the DUT's port names — e.g. a
    # module with an input port literally named `b` would clash with an instance
    # named `b`, or a port `i` with the loop counter, breaking TB compilation and
    # silently dropping the seed. Reserve the port names and pick free identifiers.
    reserved = {p.name for p in ins} | {p.name for p in outs}

    def _uniq(base: str) -> str:
        cand = base
        while cand in reserved:
            cand += "_"
        reserved.add(cand)
        return cand

    # Every harness-visible identifier is nonce-derived.  Candidate RTL is
    # compiled in the same simulation, so predictable top/instance/signal names
    # would permit an upward hierarchical reference to the golden or scoreboard.
    suffix = f"_{nonce}"
    good_inst, bad_inst = _uniq(f"dut_good{suffix}"), _uniq(f"dut_bad{suffix}")
    iv, fc = _uniq(f"i{suffix}"), _uniq(f"first_cycle{suffix}")
    top_name = f"rtlrepair_diff_tb_{nonce}"

    input_names = {p.name: _uniq(f"fc_in_{p.name}{suffix}") for p in ins}
    good_input_names = {
        p.name: _uniq(f"fc_good_in_{p.name}{suffix}") for p in ins
    }
    bad_input_names = {
        p.name: _uniq(f"fc_bad_in_{p.name}{suffix}") for p in ins
    }
    good_names = {p.name: _uniq(f"fc_good_{p.name}{suffix}") for p in outs}
    bad_names = {p.name: _uniq(f"fc_bad_{p.name}{suffix}") for p in outs}
    error_names = {p.name: _uniq(f"fc_err_{p.name}{suffix}") for p in outs}

    L: list[str] = ["`timescale 1ns/1ps", f"module {top_name};"]
    for p in ins:
        L.append(f"  reg {_decl_width(p.width)}{input_names[p.name]};")
        L.append(f"  wire {_decl_width(p.width)}{good_input_names[p.name]};")
        L.append(f"  wire {_decl_width(p.width)}{bad_input_names[p.name]};")
        L.append(
            f"  assign {good_input_names[p.name]} = {input_names[p.name]};"
        )
        L.append(f"  assign {bad_input_names[p.name]} = {input_names[p.name]};")
    for p in outs:
        L.append(f"  wire {_decl_width(p.width)}{good_names[p.name]};")
        L.append(f"  wire {_decl_width(p.width)}{bad_names[p.name]};")

    def conns(suffix: str) -> str:
        parts = []
        for p in ins:
            names = good_input_names if suffix == "good" else bad_input_names
            # Expression connection prevents input-net collapse back into the
            # verifier-owned stimulus even if candidate RTL drives its input.
            parts.append(f".{p.name}({{{names[p.name]}}})")
        for p in outs:
            names = good_names if suffix == "good" else bad_names
            parts.append(f".{p.name}({names[p.name]})")
        return ", ".join(parts)

    L.append(f"  {info.name}_good {good_inst}({conns('good')});")
    L.append(f"  {info.name}_bad  {bad_inst}({conns('bad')});")
    if clocks:
        clk_name = input_names[clocks[0].name]
        L.append(f"  always #5 {clk_name} = ~{clk_name};")

    L.append(f"  integer {iv};")
    for p in outs:
        L.append(f"  integer {error_names[p.name]};")
    L.append(f"  integer {fc};")
    L.append("  initial begin")
    L.append(f"    {fc} = -1;")
    for p in outs:
        L.append(f"    {error_names[p.name]} = 0;")
    for p in ins:
        L.append(f"    {input_names[p.name]} = 0;")
    L.append(f"    for ({iv} = 0; {iv} < {cycles}; {iv} = {iv} + 1) begin")
    for p in driven:
        L.append(f"      {input_names[p.name]} = $random;")
    L.append(
        f"      {'@(negedge ' + input_names[clocks[0].name] + ');' if clocks else '#10;'}"
    )
    # A sequential reference still holds x on its outputs in the first compared cycle,
    # before either instance is initialized, so a candidate that resolves a don't-care to a
    # concrete value mismatches there and is recorded as a failed repair. Skipping that one
    # cycle is the whole fix; see the oracle appendix for the measurement.
    _guard = f"if ({iv} >= {compare_from_cycle}) " if compare_from_cycle else ""
    for p in outs:
        L.append(f"      {_guard}if ({good_names[p.name]} !== {bad_names[p.name]}) begin")
        L.append(
            f"        {error_names[p.name]} = {error_names[p.name]} + 1;"
        )
        L.append(f"        if ({fc} < 0) {fc} = {iv};")
        L.append("      end")
    L.append("    end")
    # Report
    total = " + ".join(error_names[p.name] for p in outs)
    L.append(f"    if (({total}) > 0) begin")
    for p in outs:
        L.append(
            f'      if ({error_names[p.name]} > 0) '
            f'$display("RTLREPAIR_DIFF_DETAIL_{nonce} OUTDIFF {p.name} %0d", '
            f'{error_names[p.name]});'
        )
    L.append(
        f'      $display("RTLREPAIR_DIFF_{nonce} FAIL first_cycle=%0d", {fc});'
    )
    L.append("    end else begin")
    L.append(f'      $display("RTLREPAIR_DIFF_{nonce} PASS");')
    L.append("    end")
    # $finish with no argument makes iverilog print "$finish called at ..." AFTER the
    # verdict, which breaks the rule that the authenticated verdict is the final
    # protocol line. $finish(0) is the silent form.
    L.append("    $finish(0);")
    L.append("  end")
    L.append("endmodule")
    return "\n".join(L) + "\n"


# --- running ------------------------------------------------------------------


class SimulatorUnavailableError(RuntimeError):
    """The differential-simulation gate cannot run on this machine."""


class SimulatorProtocolError(RuntimeError):
    """The simulator ran, but did not complete the scoring protocol reliably."""


def require_simulator() -> None:
    """Fail before scoring if either Icarus executable is unavailable."""

    missing = [name for name in ("iverilog", "vvp") if shutil.which(name) is None]
    if missing:
        raise SimulatorUnavailableError(
            "differential simulation requires these executables on PATH: "
            + ", ".join(missing)
        )


@dataclass
class DiffResult:
    diverged: bool = False          # outputs differ -> bug is real & sim-detectable
    compiled: bool = True
    failure_kind: str | None = None
    first_cycle: int = -1
    out_diffs: dict[str, int] = field(default_factory=dict)  # output -> mismatch count
    log: str = ""

    def error_text(self) -> str:
        """A grounded, human-readable failure message for the training input."""
        outs = ", ".join(f"`{n}` ({c}/{CYCLES} cycles)" for n, c in self.out_diffs.items())
        return (
            "Differential simulation against the reference design failed: output(s) "
            f"{outs} diverged, first at cycle {self.first_cycle}. The module compiles "
            "and lints clean, so this is a behavioral bug, not a syntax/lint error."
        )


def _rename_module(source: str, old: str, new: str) -> str:
    return re.sub(rf"\bmodule\s+{re.escape(old)}\b", f"module {new}", source, count=1)


def _interface_reason(expected: ModuleInfo, actual: ModuleInfo | None) -> str:
    if actual is None:
        return "candidate interface is outside the audited ANSI-port subset"
    if actual.name != expected.name:
        return (
            "candidate changed top-module name "
            f"from {expected.name!r} to {actual.name!r}"
        )
    expected_ports = {
        port.name: (port.direction, port.width) for port in expected.ports
    }
    actual_ports = {port.name: (port.direction, port.width) for port in actual.ports}
    if actual_ports != expected_ports:
        return "candidate changed the audited port names, directions, or widths"
    return ""


def run_diff_test(good_src: str, bad_src: str, info: ModuleInfo, cycles: int = CYCLES,
                  compare_from_cycle: int = 0) -> DiffResult:
    """Compile golden+mutant+TB with Icarus and run; detect output divergence."""
    require_simulator()
    for role, source in (("golden", good_src), ("candidate", bad_src)):
        security_reason = candidate_security_reason(source)
        if not security_reason:
            security_reason = input_port_drive_reason(
                source, (port.name for port in info.inputs)
            )
        if security_reason:
            return DiffResult(
                compiled=False,
                failure_kind="security_rejection",
                log=f"{role} source rejected: {security_reason}",
            )
    interface_reason = _interface_reason(info, parse_module(bad_src))
    if interface_reason:
        return DiffResult(
            compiled=False,
            failure_kind="security_rejection",
            log=interface_reason,
        )
    protocol_nonce = secrets.token_hex(16)
    tb = generate_tb(info, cycles, protocol_nonce, compare_from_cycle=compare_from_cycle)
    if tb is None:
        return DiffResult(
            compiled=False,
            failure_kind="unsupported",
            log="module not drivable (no I/O or no data inputs)",
        )

    with tempfile.TemporaryDirectory(prefix="diff_tb_") as tmp:
        d = Path(tmp)
        (d / f"{info.name}_good.v").write_text(_rename_module(good_src, info.name, f"{info.name}_good"))
        (d / f"{info.name}_bad.v").write_text(_rename_module(bad_src, info.name, f"{info.name}_bad"))
        (d / "tb.v").write_text(tb)
        sim = d / "sim.out"
        try:
            comp = subprocess.run(
                ["iverilog", "-g2012", "-o", str(sim), str(d / "tb.v"),
                 str(d / f"{info.name}_good.v"), str(d / f"{info.name}_bad.v")],
                capture_output=True, text=True, timeout=20,
            )
            if comp.returncode != 0:
                return DiffResult(
                    compiled=False,
                    failure_kind="compile_failure",
                    log=comp.stderr or comp.stdout,
                )
            # vvp can spin forever on a zero-delay combinational loop; bound it.
            run = subprocess.run(
                ["vvp", str(sim)], capture_output=True, text=True, timeout=10
            )
            out = run.stdout
            if run.returncode != 0:
                log = (
                    run.stdout
                    + ("\n" if run.stdout and run.stderr else "")
                    + run.stderr
                )
                raise SimulatorProtocolError(
                    f"vvp exited with status {run.returncode}: {log.strip()}"
                )
        except subprocess.TimeoutExpired as exc:
            raise SimulatorProtocolError(f"differential simulation timed out: {exc}") from exc
        except OSError as exc:
            raise SimulatorUnavailableError(
                f"failed to execute the differential-simulation toolchain: {exc}"
            ) from exc

    verdict_pattern = re.compile(
        rf"^RTLREPAIR_DIFF_{protocol_nonce} "
        r"(?:(PASS)|(FAIL) first_cycle=(-?\d+))$"
    )
    output_lines = [line.strip() for line in out.splitlines() if line.strip()]
    verdicts = [match for line in output_lines if (match := verdict_pattern.fullmatch(line))]
    if len(verdicts) != 1:
        raise SimulatorProtocolError(
            f"simulation produced {len(verdicts)} authenticated terminal verdicts; "
            "expected exactly one\n" + out
        )
    terminal = verdicts[0]
    if not output_lines or verdict_pattern.fullmatch(output_lines[-1]) is None:
        raise SimulatorProtocolError(
            "authenticated verdict was not the final non-empty protocol line\n" + out
        )

    result = DiffResult(compiled=True, log=out)
    if terminal.group(2) == "FAIL":
        result.diverged = True
        result.first_cycle = int(terminal.group(3))
        detail_pattern = re.compile(
            rf"^RTLREPAIR_DIFF_DETAIL_{protocol_nonce} OUTDIFF (\w+) (\d+)$",
            re.MULTILINE,
        )
        for m in detail_pattern.finditer(out):
            result.out_diffs[m.group(1)] = int(m.group(2))
    return result
