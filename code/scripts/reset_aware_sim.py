#!/usr/bin/env python3
"""A differential simulator that honours the initialization contract.

`datagen/diff_testbench.py` drives every non-clock input with $random from cycle 0, reset
included, and compares outputs from the first cycle. On a design with a reset that produces
false divergence: both instances start from X, reset is toggled at random, and the outputs
disagree before the design has ever been initialised. Measured against the frozen gpt-5.5
baseline candidates, that scorer marks 55.7% equivalent where the formal proof marks 91.8%.

The formal side does not have this problem because a sequential proof carries an
initialization contract -- ours holds reset for two cycles and then leaves it free. This
module reproduces that contract in simulation: find the reset port, assert it (active level
inferred from the name) for RESET_CYCLES, release it, and only then drive stimulus and
compare.

Deliberately kept separate from diff_testbench so the banked Tier S numbers, and E4's parity
check against them, keep reproducing exactly. Use this where an absolute recovery rate has to
be right; use diff_testbench where the job is to reproduce what was banked.
"""
import os, re, subprocess, sys, tempfile
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "datagen"))
from diff_testbench import parse_module, _decl_width, _rename_module  # noqa: E402

RESET_CYCLES = 2      # matches the frozen proofs' initialization contract
CYCLES = 200


@dataclass
class Result:
    compiled: bool = True
    diverged: bool = False
    first_cycle: int = -1
    out_diffs: dict = field(default_factory=dict)
    reset_port: str = ""
    active_low: bool = False
    log: str = ""


def find_reset(info):
    """Return (port, active_low) for the reset input, or (None, False).

    Active level comes from the name: a trailing _n / _b, or a leading n, is active-low,
    which is the SystemVerilog convention this corpus follows (rst_n, resetn, n_rst).
    """
    for p in info.inputs:
        if getattr(p, "is_clock", False):
            continue
        n = p.name.lower()
        if not re.search(r"(^|_)(rst|reset|nreset|nrst)(_|$)|^n?rst_?n?$|reset", n):
            continue
        low = bool(re.search(r"(_n|_b)$", n) or re.match(r"^n[_]?(rst|reset)", n) or n.endswith("n"))
        return p, low
    return None, False


def generate_tb(info, reset, active_low, cycles=CYCLES):
    ins, outs = info.inputs, info.outputs
    clocks = [p for p in ins if getattr(p, "is_clock", False)]
    driven = [p for p in ins if not getattr(p, "is_clock", False) and (reset is None or p.name != reset.name)]
    # A design whose only inputs are a clock and a reset is still drivable: after the
    # contract window it free-runs on the clock. Only bail when there is nothing to drive
    # it with at all.
    if not ins or not outs or (not driven and not clocks):
        return None
    reserved = {p.name for p in ins} | {p.name for p in outs}

    def uniq(base):
        c = base
        while c in reserved:
            c += "_"
        reserved.add(c)
        return c

    gi, bi, iv, fc = uniq("dut_good"), uniq("dut_bad"), uniq("i"), uniq("first_cycle")
    L = ["`timescale 1ns/1ps", "module tb;"]
    for p in ins:
        L.append(f"  reg {_decl_width(p.width)}{p.name};")
    for p in outs:
        L.append(f"  wire {_decl_width(p.width)}{p.name}_good;")
        L.append(f"  wire {_decl_width(p.width)}{p.name}_bad;")
    conns = lambda sfx: ", ".join([f".{p.name}({p.name})" for p in ins] +
                                  [f".{p.name}({p.name}_{sfx})" for p in outs])
    L.append(f"  {info.name}_good {gi}({conns('good')});")
    L.append(f"  {info.name}_bad  {bi}({conns('bad')});")
    if clocks:
        L.append(f"  always #5 {clocks[0].name} = ~{clocks[0].name};")
    L.append(f"  integer {iv};")
    for p in outs:
        L.append(f"  integer err_{p.name};")
    L.append(f"  integer {fc};")
    L.append("  initial begin")
    L.append(f"    {fc} = -1;")
    for p in outs:
        L.append(f"    err_{p.name} = 0;")
    for p in ins:
        L.append(f"    {p.name} = 0;")
    tick = f"@(negedge {clocks[0].name});" if clocks else "#10;"
    if reset is not None:
        # hold reset asserted, inputs quiet, for the contract window
        L.append(f"    {reset.name} = {'0' if active_low else '1'};")
        L.append(f"    for ({iv} = 0; {iv} < {RESET_CYCLES}; {iv} = {iv} + 1) begin {tick} end")
        L.append(f"    {reset.name} = {'1' if active_low else '0'};")
    L.append(f"    for ({iv} = 0; {iv} < {cycles}; {iv} = {iv} + 1) begin")
    for p in driven:
        L.append(f"      {p.name} = $random;")
    L.append(f"      {tick}")
    for p in outs:
        L.append(f"      if ({p.name}_good !== {p.name}_bad) begin")
        L.append(f"        err_{p.name} = err_{p.name} + 1;")
        L.append(f"        if ({fc} < 0) {fc} = {iv};")
        L.append("      end")
    L.append("    end")
    total = " + ".join(f"err_{p.name}" for p in outs)
    L.append(f"    if (({total}) > 0) begin")
    L.append(f'      $display("DIFF_FAIL first_cycle=%0d", {fc});')
    for p in outs:
        L.append(f'      if (err_{p.name} > 0) $display("OUTDIFF {p.name} %0d", err_{p.name});')
    L.append("    end else $display("'"DIFF_PASS"'");")
    L.append("    $finish;")
    L.append("  end")
    L.append("endmodule")
    return "\n".join(L) + "\n"


def run(golden_src, cand_src, cycles=CYCLES):
    info = parse_module(golden_src)
    if info is None:
        return Result(compiled=False, log="golden unparsed")
    ci = parse_module(cand_src)
    if ci and ci.name != info.name:
        cand_src = _rename_module(cand_src, ci.name, info.name)
    reset, low = find_reset(info)
    tb = generate_tb(info, reset, low, cycles)
    if tb is None:
        return Result(compiled=False, log="not drivable")
    with tempfile.TemporaryDirectory() as d:
        g = Path(d) / "good.sv"; b = Path(d) / "bad.sv"; t = Path(d) / "tb.sv"
        g.write_text(_rename_module(golden_src, info.name, info.name + "_good"))
        b.write_text(_rename_module(cand_src, info.name, info.name + "_bad"))
        t.write_text(tb)
        exe = Path(d) / "sim"
        cp = subprocess.run(["iverilog", "-g2012", "-o", str(exe), str(t), str(g), str(b)],
                            capture_output=True, text=True, timeout=120)
        if cp.returncode != 0:
            return Result(compiled=False, log=cp.stderr[:400], reset_port=reset.name if reset else "",
                          active_low=low)
        rp = subprocess.run(["vvp", str(exe)], capture_output=True, text=True, timeout=120)
        out = rp.stdout
    res = Result(reset_port=reset.name if reset else "", active_low=low, log=out[:400])
    if "DIFF_PASS" in out:
        return res
    m = re.search(r"DIFF_FAIL first_cycle=(\d+)", out)
    res.diverged = True
    res.first_cycle = int(m.group(1)) if m else -1
    res.out_diffs = {mm.group(1): int(mm.group(2)) for mm in re.finditer(r"OUTDIFF (\S+) (\d+)", out)}
    return res


def equivalent(golden_src, cand_src):
    r = run(golden_src, cand_src)
    return bool(r.compiled and not r.diverged)
