"""Inject single lint-class bugs into known-good SystemVerilog seeds.

Phase 1 of the repair-dataset pipeline (see EXECUTION_PLAN.md §7.1). Each
mutation takes clean, Verilator-passing source and returns a *broken* variant
that should fail ``verilator --lint-only``. We never hand-write the resulting
tool error: ``run_verilator_for_repairs.py`` lints the broken file and captures
whatever Verilator actually prints, and the *fix* we train on is just the
original source (so the unified-diff label, produced in build_repair_dataset.py,
is exact). Mutations here only have to break the code in a realistic way.

Design choices:

* **One mutation per pair.** A repair example should isolate a single bug so the
  fix is unambiguous; we apply exactly one mutation to a clean seed.
* **Deterministic.** Each mutation is seeded from ``(seed, mutation_name)`` so a
  given corpus regenerates byte-for-byte.
* **Self-validating downstream.** A mutation returns ``None`` when it does not
  apply to a module (e.g. ``corrupt_pin`` on a module with no instantiation).
  Even when it applies, the verifier gate downstream drops the pair if Verilator
  does not actually flag it — so a mutation that silently produces still-legal
  code costs nothing.

The lint-class taxonomy (phase 1):

    drop_endmodule    delete the closing ``endmodule``      -> syntax error
    delete_semicolon  drop a statement terminator           -> syntax error
    drop_one_end      unbalance a ``begin``/``end`` block    -> syntax error
    corrupt_pin       misname a port/param in an instance   -> PINNOTFOUND/PARAM
    width_mismatch    shrink a vector declaration's width   -> WIDTH

Semantic-class mutations (missing reset, blocking/nonblocking misuse, ...) are
phase 2 and live behind iverilog simulation, not lint; they are not here yet.
"""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path
from typing import Callable, Optional

# A mutation: clean source + a seeded RNG -> broken source, or None if N/A.
Mutation = Callable[[str, random.Random], Optional[str]]


def _is_comment(line: str) -> bool:
    """True for a ``//`` line comment (the only comment style our seeds use)."""
    return line.lstrip().startswith("//")


def _code_lines(lines: list[str]) -> list[int]:
    """Indices of non-blank, non-comment lines — the editable surface."""
    return [i for i, ln in enumerate(lines) if ln.strip() and not _is_comment(ln)]


def drop_endmodule(src: str, rng: random.Random) -> Optional[str]:
    """Delete the final standalone ``endmodule`` — leaves the module unclosed."""
    lines = src.splitlines(keepends=True)
    for i in range(len(lines) - 1, -1, -1):
        if lines[i].strip() == "endmodule":
            del lines[i]
            return "".join(lines)
    return None


def delete_semicolon(src: str, rng: random.Random) -> Optional[str]:
    """Drop the trailing ``;`` from one statement — a classic syntax error."""
    lines = src.splitlines(keepends=True)
    cands = [
        i
        for i in _code_lines(lines)
        if lines[i].rstrip().endswith(";")
    ]
    if not cands:
        return None
    i = rng.choice(cands)
    nl = "\n" if lines[i].endswith("\n") else ""
    body = lines[i].rstrip("\n")
    cut = body.rfind(";")
    lines[i] = body[:cut] + body[cut + 1 :] + nl
    return "".join(lines)


def drop_one_end(src: str, rng: random.Random) -> Optional[str]:
    """Remove one ``end`` keyword, unbalancing a ``begin``/``end`` block."""
    lines = src.splitlines(keepends=True)
    cands = [i for i, ln in enumerate(lines) if ln.strip() == "end"]
    if not cands:
        return None
    del lines[rng.choice(cands)]
    return "".join(lines)


# `.name(` in an instantiation port/param map: `.clk(clk)`, `.DATA_W(DATA_W)`.
_PIN_RE = re.compile(r"\.(\w+)\s*\(")


def corrupt_pin(src: str, rng: random.Random) -> Optional[str]:
    """Misname one connected port/param in an instantiation -> PINNOTFOUND."""
    lines = src.splitlines(keepends=True)
    cands: list[tuple[int, re.Match]] = []
    for i, ln in enumerate(lines):
        if _is_comment(ln):
            continue
        cands.extend((i, m) for m in _PIN_RE.finditer(ln))
    if not cands:
        return None
    i, m = rng.choice(cands)
    ln = lines[i]
    lines[i] = ln[: m.start(1)] + m.group(1) + "_x" + ln[m.end(1) :]
    return "".join(lines)


# A parameterized vector range like `[DATA_W-1:0]` or `[ACC_W - 1 : 0]`.
_WIDTH_RE = re.compile(r"\[\s*[A-Za-z_]\w*\s*-\s*1\s*:\s*0\s*\]")


def width_mismatch(src: str, rng: random.Random) -> Optional[str]:
    """Shrink a parameterized vector to a fixed ``[3:0]`` -> WIDTH mismatch."""
    lines = src.splitlines(keepends=True)
    cands: list[tuple[int, re.Match]] = []
    for i, ln in enumerate(lines):
        if _is_comment(ln):
            continue
        cands.extend((i, m) for m in _WIDTH_RE.finditer(ln))
    if not cands:
        return None
    i, m = rng.choice(cands)
    ln = lines[i]
    lines[i] = ln[: m.start()] + "[3:0]" + ln[m.end() :]
    return "".join(lines)


# A local net/var declaration: `reg [7:0] foo;`, `logic signed bar;`, `wire baz;`.
_DECL_RE = re.compile(r"\b(?:reg|wire|logic)\b(?:\s+signed)?(?:\s*\[[^\]]*\])?\s+(\w+)\s*;")


def rename_declaration(src: str, rng: random.Random) -> Optional[str]:
    """Rename a local declaration so every use of it becomes undeclared.

    Touches only the declaration line; the original name still appears at every
    use site, so Verilator sees an implicit/undriven net (IMPLICIT/UNDRIVEN/WIDTH).
    Unlike corrupt_pin this needs no instantiation, so it fires on leaf modules.
    """
    lines = src.splitlines(keepends=True)
    cands = [
        (i, m)
        for i, ln in enumerate(lines)
        if not _is_comment(ln)
        for m in [_DECL_RE.search(ln)]
        if m
    ]
    if not cands:
        return None
    i, m = rng.choice(cands)
    ln = lines[i]
    lines[i] = ln[: m.start(1)] + m.group(1) + "_undecl" + ln[m.end(1) :]
    return "".join(lines)


# A nonblocking-assignment statement: `q <= d;`, `acc[3:0] <= x;` (not `a <= b`
# as a comparison — we require it to start the line and end in a semicolon).
_NBA_RE = re.compile(r"^\s*[\w.\[\]:'$-]+\s*<=")


def blocking_in_seq(src: str, rng: random.Random) -> Optional[str]:
    """Turn a nonblocking ``<=`` assignment into a blocking ``=`` one.

    In a clocked (``always_ff``/``posedge``) block this trips Verilator's BLKSEQ
    warning. In combinational context it is harmless, so the gate simply drops
    those — making this a safe, instantiation-free way to add a non-syntax bug.
    """
    lines = src.splitlines(keepends=True)
    cands = [
        i
        for i, ln in enumerate(lines)
        if not _is_comment(ln) and _NBA_RE.match(ln) and ln.rstrip().endswith(";")
    ]
    if not cands:
        return None
    i = rng.choice(cands)
    ln = lines[i]
    pos = ln.index("<=")
    lines[i] = ln[:pos] + "=" + ln[pos + 2 :]
    return "".join(lines)


# --- Semantic mutations -------------------------------------------------------
# Unlike the lint mutations above, these keep the code syntactically valid AND
# lint-clean — they only change *behavior*. They cannot be verified by lint; the
# differential testbench (diff_testbench.py) catches them by simulating the
# original against the mutant on random stimulus and watching for output divergence.

# A binary operator with whitespace on both sides, so we don't touch `-1`,
# `[X-1:0]`, `<=`, `&&`, `||`, etc. Maps each to its behavioral opposite.
_OP_SWAP: dict[str, str] = {" + ": " - ", " - ": " + ", " & ": " | ", " | ": " & "}


def op_swap(src: str, rng: random.Random) -> Optional[str]:
    """Swap one arithmetic/bitwise operator for its opposite (+/-, &/|)."""
    lines = src.splitlines(keepends=True)
    cands = [
        (i, op)
        for i, ln in enumerate(lines)
        if not _is_comment(ln)
        for op in _OP_SWAP
        if op in ln
    ]
    if not cands:
        return None
    i, op = rng.choice(cands)
    ln = lines[i]
    pos = ln.index(op)
    lines[i] = ln[:pos] + _OP_SWAP[op] + ln[pos + len(op) :]
    return "".join(lines)


# `!rst_n` / `!reset` inside a condition — the active-low reset test.
_RST_NEG_RE = re.compile(r"(!\s*)(\w*(?:rst|reset)\w*)", re.IGNORECASE)


def flip_reset_polarity(src: str, rng: random.Random) -> Optional[str]:
    """Drop the ``!`` from an active-low reset test, inverting reset behavior."""
    lines = src.splitlines(keepends=True)
    cands = [
        (i, m)
        for i, ln in enumerate(lines)
        if not _is_comment(ln)
        for m in _RST_NEG_RE.finditer(ln)
    ]
    if not cands:
        return None
    i, m = rng.choice(cands)
    ln = lines[i]
    lines[i] = ln[: m.start(1)] + ln[m.start(2) :]  # remove the leading '!'
    return "".join(lines)


def _swap_one_token(src: str, rng: random.Random, mapping: dict[str, str]) -> Optional[str]:
    """Generic single-token swap (op_swap's logic, reused for any operator map)."""
    lines = src.splitlines(keepends=True)
    cands = [
        (i, tok)
        for i, ln in enumerate(lines)
        if not _is_comment(ln)
        for tok in mapping
        if tok in ln
    ]
    if not cands:
        return None
    i, tok = rng.choice(cands)
    ln = lines[i]
    pos = ln.index(tok)
    lines[i] = ln[:pos] + mapping[tok] + ln[pos + len(tok) :]
    return "".join(lines)


_EQ_SWAP = {" == ": " != ", " != ": " == "}
_LOGIC_SWAP = {" && ": " || ", " || ": " && "}
_SHIFT_SWAP = {" << ": " >> ", " >> ": " << "}


def flip_equality(src: str, rng: random.Random) -> Optional[str]:
    """Swap an equality test for its opposite (== <-> !=)."""
    return _swap_one_token(src, rng, _EQ_SWAP)


def swap_logical(src: str, rng: random.Random) -> Optional[str]:
    """Swap a logical connective (&& <-> ||)."""
    return _swap_one_token(src, rng, _LOGIC_SWAP)


def shift_swap(src: str, rng: random.Random) -> Optional[str]:
    """Swap a shift direction (<< <-> >>)."""
    return _swap_one_token(src, rng, _SHIFT_SWAP)


# A simple ternary `cond ? A : B` with atomic operands -> swap the two arms.
_TERNARY_RE = re.compile(r"\?\s*([A-Za-z0-9_'\.\[\]\{\}]+)\s*:\s*([A-Za-z0-9_'\.\[\]\{\}]+)")


def swap_ternary(src: str, rng: random.Random) -> Optional[str]:
    """Swap the two arms of a ternary/mux select (cond ? a : b -> cond ? b : a)."""
    lines = src.splitlines(keepends=True)
    cands = [
        (i, m)
        for i, ln in enumerate(lines)
        if not _is_comment(ln)
        for m in _TERNARY_RE.finditer(ln)
    ]
    if not cands:
        return None
    i, m = rng.choice(cands)
    ln = lines[i]
    lines[i] = ln[: m.start()] + "? " + m.group(2) + " : " + m.group(1) + ln[m.end() :]
    return "".join(lines)


# Sized literals like 4'd9 / 8'hA5 — nudge the value by one (off-by-one bug).
_LIT_RE = re.compile(r"(\d+)'([dh])([0-9a-fA-F]+)")


def off_by_one_literal(src: str, rng: random.Random) -> Optional[str]:
    """Change a sized numeric literal by +-1 (off-by-one in a state/counter/limit)."""
    lines = src.splitlines(keepends=True)
    cands = [
        (i, m)
        for i, ln in enumerate(lines)
        if not _is_comment(ln)
        for m in _LIT_RE.finditer(ln)
    ]
    if not cands:
        return None
    i, m = rng.choice(cands)
    width, base, digits = m.group(1), m.group(2), m.group(3)
    val = int(digits, 16 if base == "h" else 10)
    new_val = val - 1 if val > 0 else val + 1
    new_digits = format(new_val, "x" if base == "h" else "d")
    ln = lines[i]
    lines[i] = ln[: m.start()] + f"{width}'{base}{new_digits}" + ln[m.end() :]
    return "".join(lines)


# Registry — order is stable so ids/diffs are reproducible across runs.
MUTATIONS: dict[str, Mutation] = {
    "drop_endmodule": drop_endmodule,
    "delete_semicolon": delete_semicolon,
    "drop_one_end": drop_one_end,
    "corrupt_pin": corrupt_pin,
    "width_mismatch": width_mismatch,
    "rename_declaration": rename_declaration,
    "blocking_in_seq": blocking_in_seq,
}

# Behavior-changing mutations, verified by simulation rather than lint.
SEMANTIC_MUTATIONS: dict[str, Mutation] = {
    "op_swap": op_swap,
    "flip_reset_polarity": flip_reset_polarity,
    "flip_equality": flip_equality,
    "swap_logical": swap_logical,
    "shift_swap": shift_swap,
    "swap_ternary": swap_ternary,
    "off_by_one_literal": off_by_one_literal,
}


def _apply(registry: dict[str, Mutation], source: str, seed: int) -> list[tuple[str, str]]:
    out: list[tuple[str, str]] = []
    for name, fn in registry.items():
        rng = random.Random(f"{seed}:{name}")
        broken = fn(source, rng)
        if broken is not None and broken != source:
            out.append((name, broken))
    return out


def inject_all(source: str, *, seed: int = 0) -> list[tuple[str, str]]:
    """Apply every lint-class mutation that fits ``source``.

    Returns ``(mutation_name, broken_source)`` for each mutation that produced a
    real change. Verifying that the change actually fails lint is the job of the
    verifier gate, not this function.
    """
    return _apply(MUTATIONS, source, seed)


def inject_semantic(source: str, *, seed: int = 0) -> list[tuple[str, str]]:
    """Apply every semantic mutation that fits ``source`` (verified by sim)."""
    return _apply(SEMANTIC_MUTATIONS, source, seed)


def main() -> None:
    ap = argparse.ArgumentParser(description="Inject lint-class bugs into a .sv seed.")
    ap.add_argument("seed_file", help="path to a clean .sv module")
    ap.add_argument("--seed", type=int, default=0, help="RNG seed (reproducible)")
    ap.add_argument(
        "--out-dir",
        help="write each broken variant here as <stem>__<mutation>.sv; "
        "if omitted, just list which mutations applied",
    )
    args = ap.parse_args()

    src = Path(args.seed_file).read_text()
    variants = inject_all(src, seed=args.seed)
    if not variants:
        print(f"no mutations applied to {args.seed_file}")
        return

    if args.out_dir:
        out_dir = Path(args.out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(args.seed_file).stem
        for name, broken in variants:
            dest = out_dir / f"{stem}__{name}.sv"
            dest.write_text(broken)
            print(f"{name:18} -> {dest}")
    else:
        for name, _ in variants:
            print(f"applied: {name}")


if __name__ == "__main__":
    main()
