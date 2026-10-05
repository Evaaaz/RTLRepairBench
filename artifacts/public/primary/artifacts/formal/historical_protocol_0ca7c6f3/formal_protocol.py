"""Formal contracts and harness generation for verifier-guided RTL repair.

This module is intentionally tool-independent.  It classifies the canonical
designs, records the audited initialization assumptions, and emits equivalence
harnesses consumed by :mod:`benchmarks.formal_verify`.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from tuning.data_gen.diff_testbench import ModuleInfo, Port, parse_module


ROOT = Path(__file__).resolve().parents[1]
INDEPENDENT_SIMULATION_SEEDS = tuple(range(1001, 1011))
INDEPENDENT_SIMULATION_CYCLES = 200
BMC_COUNTEREXAMPLE_DEPTH = 256


class FormalStatus(str, Enum):
    PROVED = "PROVED"
    COUNTEREXAMPLE = "COUNTEREXAMPLE"
    TIMEOUT = "TIMEOUT"
    UNSUPPORTED = "UNSUPPORTED"
    COMPILE_FAIL = "COMPILE_FAIL"


@dataclass(frozen=True)
class InitializationContract:
    kind: str
    description: str
    preamble_cycles: int = 0
    clock_port: str | None = None
    reset_port: str | None = None
    reset_active: int | None = None
    reset_synchronous: bool | None = None
    preamble_assignments: dict[str, int] = field(default_factory=dict)
    clock_edge: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class FormalResult:
    status: FormalStatus
    engine: str
    initialization_contract: InitializationContract
    elapsed_seconds: float
    candidate_sha256: str
    golden_sha256: str
    vcd_path: str | None = None
    witness_path: str | None = None
    witness: dict[str, Any] | None = None
    counterexample_replayed: bool | None = None
    detail: str = ""
    command: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["status"] = self.status.value
        return row

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "FormalResult":
        data = dict(row)
        data["status"] = FormalStatus(data["status"])
        data["initialization_contract"] = InitializationContract(
            **data["initialization_contract"]
        )
        return cls(**data)


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _sequential(source: str) -> bool:
    return bool(
        re.search(r"\balways_(?:ff|latch)\b", source)
        or re.search(r"\balways\s*@\s*\([^)]*\b(?:pos|neg)edge\b", source, re.DOTALL)
    )


def _clock_port(info: ModuleInfo) -> Port | None:
    clocks = [port for port in info.inputs if port.is_clock]
    return clocks[0] if len(clocks) == 1 else None


def _reset_port(info: ModuleInfo) -> Port | None:
    resets = [port for port in info.inputs if port.is_reset]
    return resets[0] if len(resets) == 1 else None


def _clock_edge(source: str, clock: str) -> str | None:
    edges = {
        match.group(1).lower()
        for match in re.finditer(
            rf"\b(posedge|negedge)\s+{re.escape(clock)}\b", source, re.IGNORECASE
        )
    }
    return next(iter(edges)) if len(edges) == 1 else None


def _active_reset_value(source: str, reset: str) -> int:
    escaped = re.escape(reset)
    low_patterns = (
        rf"\bif\s*\(\s*[!~]\s*{escaped}\b",
        rf"\bif\s*\(\s*{escaped}\s*={2,3}\s*(?:1\s*'\s*b\s*)?0\b",
        rf"\bif\s*\(\s*(?:1\s*'\s*b\s*)?0\s*={2,3}\s*{escaped}\b",
    )
    if any(re.search(pattern, source, re.IGNORECASE) for pattern in low_patterns):
        return 0
    return 0 if reset.lower().endswith(("_n", "n")) else 1


def _asynchronous_reset(source: str, reset: str) -> bool:
    escaped = re.escape(reset)
    for sensitivity in re.findall(r"\balways\s*@\s*\(([^)]*)\)", source, re.DOTALL):
        if re.search(rf"\b(?:pos|neg)edge\s+{escaped}\b", sensitivity):
            return True
    for sensitivity in re.findall(r"\balways_ff\s*@\s*\(([^)]*)\)", source, re.DOTALL):
        if re.search(rf"\b(?:pos|neg)edge\s+{escaped}\b", sensitivity):
            return True
    return False


# Audited public-input synchronization sequences.  No hidden reference state or
# expected output is exposed to the model.  Empty assignments mean arbitrary
# but shared inputs for each preamble cycle.
_NO_RESET_PREAMBLES: dict[str, tuple[int, dict[str, int], str]] = {
    "Prob054_edgedetect_ref": (
        2,
        {},
        "two clock edges with arbitrary inputs shared by both instances",
    ),
    "Prob056_ece241_2013_q7_ref": (
        1,
        {"j": 0, "k": 1},
        "one shared edge with J=0 and K=1",
    ),
    "Prob063_review2015_shiftcount_ref": (
        4,
        {"shift_ena": 1},
        "four shared edges with shift_ena=1",
    ),
    "Prob105_rotate100_ref": (
        1,
        {"load": 1},
        "one shared edge with load=1 and arbitrary shared data",
    ),
    "Prob117_circuit9_ref": (
        1,
        {"a": 1},
        "one shared edge with a=1",
    ),
    "Prob104_mt2015_muxdff_ref": (
        1,
        {},
        "one common clock edge after the audited initial declaration",
    ),
}


def initialization_contract(seed_id: str, source: str) -> InitializationContract:
    info = parse_module(source)
    if info is None:
        return InitializationContract(
            kind="unsupported",
            description="module header is outside the audited ANSI-port subset",
        )
    if not _sequential(source):
        return InitializationContract(
            kind="combinational",
            description="stateless exhaustive equivalence; no initialization",
        )

    clock = _clock_port(info)
    if clock is None:
        return InitializationContract(
            kind="unsupported",
            description="sequential module does not have exactly one recognized clock port",
        )
    edge = _clock_edge(source, clock.name)
    if edge is None:
        return InitializationContract(
            kind="unsupported",
            description="sequential module does not have one consistent recognized clock edge",
            clock_port=clock.name,
        )
    reset = _reset_port(info)
    if reset is not None:
        active = _active_reset_value(source, reset.name)
        synchronous = not _asynchronous_reset(source, reset.name)
        style = "synchronous" if synchronous else "asynchronous"
        return InitializationContract(
            kind="reset",
            description=(
                f"assert the real {style} {reset.name} reset for two shared clock edges, "
                "then hold it inactive"
            ),
            preamble_cycles=2,
            clock_port=clock.name,
            reset_port=reset.name,
            reset_active=active,
            reset_synchronous=synchronous,
            clock_edge=edge,
        )

    if seed_id in _NO_RESET_PREAMBLES:
        cycles, assignments, description = _NO_RESET_PREAMBLES[seed_id]
        return InitializationContract(
            kind="preamble",
            description=description,
            preamble_cycles=cycles,
            clock_port=clock.name,
            preamble_assignments=assignments,
            clock_edge=edge,
        )
    return InitializationContract(
        kind="unsupported",
        description="no reset and no pre-registered public-input synchronization contract",
        clock_port=clock.name,
        clock_edge=edge,
    )


_PROB151_ENUM = re.compile(
    r"typedef\s+enum\s+logic\s*\[\s*3\s*:\s*0\s*\]\s*\{\s*"
    r"S\s*,\s*S1\s*,\s*S11\s*,\s*S110\s*,\s*B0\s*,\s*B1\s*,\s*B2\s*,\s*B3\s*,\s*Count\s*,\s*Wait\s*"
    r"\}\s*States\s*;\s*States\s+state\s*,\s*next\s*;",
    re.DOTALL,
)
_PROB151_X_OUTPUT_GUARD = re.compile(
    r"\n\s*if\s*\(\s*\|state\s*===\s*1'bx\s*\)\s*begin\s*"
    r"\{\s*shift_ena\s*,\s*counting\s*,\s*done\s*\}\s*=\s*'x\s*;\s*"
    r"end\s*",
    re.DOTALL,
)
_PROB151_MODULE_NAME = re.compile(r"(\bmodule\s+)[A-Za-z_$][\w$]*")
_PROB151_AUDITED_SOURCE_SHA256 = frozenset(
    {
        # Canonical plus repair_sem_0008/0009/0010 known mutants.  Hashing
        # ignores only the top-module identifier and outer line endings so the
        # anonymous TopModule materialization has the same audited identity.
        "f6d97d8e67188c200d657dcc73d3016750afb0d4221d270f1387fd9e9600585d",
        "43b7481353580b9923a030d03a0558510168a01fbbe996e625107b66b2014dfb",
        "26fd79160c2c64b4d12987f52ad83c4ba9dc9542b4ae12ef8226ac70816ae4e7",
        "e17d92dfe2b0a0e0f3b83828030d90cdf17ad88bb5d8ab23f616392552218688",
    }
)


def _prob151_audited_source_sha256(source: str) -> str:
    normalized = source.replace("\r\n", "\n").replace("\r", "\n").strip() + "\n"
    normalized, replacements = _PROB151_MODULE_NAME.subn(
        r"\1__PROB151_TOP__", normalized, count=1
    )
    if replacements != 1:
        raise ValueError("Prob151 source has no auditable top-module declaration")
    return sha256_text(normalized)


def normalize_prob151(seed_id: str, source: str) -> tuple[str, bool]:
    """Apply the sole pre-audited frontend normalization.

    Stock Yosys cannot parse the legal ``States'(expr)`` casts in Prob151.  The
    enum has the default consecutive 4-bit encoding, so we replace it with
    explicit 4-bit localparams and a 4-bit state vector, then remove only that
    named cast.  No other source or enum shape is normalized.
    """

    if seed_id != "Prob151_review2015_fsm_ref":
        return source, False
    has_named_cast = "States'(" in source
    has_x_output_guard = _PROB151_X_OUTPUT_GUARD.search(source) is not None
    if not has_named_cast and not has_x_output_guard:
        # A complete model rewrite that avoids both unsupported constructs is
        # ordinary RTL and needs no source normalization.
        return source, False
    audited_sha = _prob151_audited_source_sha256(source)
    if audited_sha not in _PROB151_AUDITED_SOURCE_SHA256:
        raise ValueError(
            "Prob151 cast/X-guard source is outside the audited normalization allowlist"
        )
    if not has_named_cast:
        raise ValueError("Prob151 audited source unexpectedly lacks its named casts")
    replacement = (
        "localparam logic [3:0] S = 4'd0, S1 = 4'd1, S11 = 4'd2, "
        "S110 = 4'd3, B0 = 4'd4, B1 = 4'd5, B2 = 4'd6, B3 = 4'd7, "
        "Count = 4'd8, Wait = 4'd9;\n\n  logic [3:0] state, next;"
    )
    normalized, replacements = _PROB151_ENUM.subn(replacement, source, count=1)
    if replacements != 1:
        return source, False
    normalized = normalized.replace("States'(", "(")
    # This exact guard is meaningful only in four-state simulation.  After the
    # audited two-cycle reset, Prob151's state is one of the ten explicit
    # encodings and every transition stays in that set, so the guard is
    # unreachable.  Yosys otherwise lowers its X literal to an independent
    # anyseq in each miter instance and creates a false self-counterexample.
    # Require the one audited shape so source drift fails closed.
    normalized, guard_replacements = _PROB151_X_OUTPUT_GUARD.subn(
        "\n", normalized, count=1
    )
    if guard_replacements != 1:
        raise ValueError("Prob151 audited X-output guard shape drifted")
    return normalized, True


def _decl(port: Port, attributes: str = "") -> str:
    width = "" if port.width == 1 else f"[{port.width - 1}:0] "
    prefix = f"{attributes} " if attributes else ""
    return f"  {prefix}reg {width}{port.name};"


def rename_top_module(source: str, new_name: str) -> tuple[str, str]:
    match = re.search(r"\bmodule\s+([A-Za-z_$][\w$]*)", source)
    if not match:
        raise ValueError("source has no module declaration")
    old_name = match.group(1)
    return source[: match.start(1)] + new_name + source[match.end(1) :], old_name


def compatible_interfaces(golden: ModuleInfo, candidate: ModuleInfo) -> tuple[bool, str]:
    expected = sorted((p.name, p.direction, p.width) for p in golden.ports)
    actual = sorted((p.name, p.direction, p.width) for p in candidate.ports)
    if expected != actual:
        return False, f"candidate interface {actual!r} does not match golden {expected!r}"
    return True, ""


def _connections(info: ModuleInfo, suffix: str) -> str:
    connections = []
    for port in info.inputs:
        connections.append(f".{port.name}({port.name})")
    for port in info.outputs:
        connections.append(f".{port.name}({port.name}_{suffix})")
    return ", ".join(connections)


def generate_equivalence_wrapper(
    info: ModuleInfo,
    contract: InitializationContract,
    golden_module: str = "design_golden",
    candidate_module: str = "design_candidate",
    seed_id: str | None = None,
) -> str:
    """Emit a shared-input, double-instance equivalence top."""

    lines = ["module equiv_top;"]
    sequential = contract.kind in {"reset", "preamble"}
    for port in info.inputs:
        if sequential and port.name == contract.clock_port:
            # SymbiYosys recognizes this attribute as the engine's global
            # formal clock.  ``$global_clock`` is not a Verilog frontend symbol
            # in the pinned Yosys release and would become an undriven wire.
            lines.append(f"  (* gclk, keep *) reg {port.name};")
        else:
            attribute = "(* anyseq, keep *)" if sequential else "(* anyconst, keep *)"
            lines.append(_decl(port, attribute))
    for port in info.outputs:
        width = "" if port.width == 1 else f"[{port.width - 1}:0] "
        lines.append(f"  wire {width}{port.name}_golden;")
        lines.append(f"  wire {width}{port.name}_candidate;")
    lines.append(f"  {golden_module} golden ({_connections(info, 'golden')});")
    lines.append(f"  {candidate_module} candidate ({_connections(info, 'candidate')});")

    if seed_id == "Prob154_fsm_ps2data_ref":
        output_names = {port.name for port in info.outputs}
        if output_names != {"out_bytes", "done"}:
            raise ValueError("Prob154 audited output-validity interface drifted")
        # The public contract explicitly makes out_bytes invalid until done.
        # Compare the validity bit always and compare the payload only while it
        # is valid.  This avoids treating the specified 'x payload as data.
        equality = (
            "(done_golden == done_candidate) && "
            "((!done_golden) || (out_bytes_golden == out_bytes_candidate))"
        )
    else:
        equality = " && ".join(
            f"({port.name}_golden == {port.name}_candidate)" for port in info.outputs
        )
    if not sequential:
        lines.append(f"  always @* assert ({equality});")
        lines.append("endmodule")
        return "\n".join(lines) + "\n"

    width = max(1, (contract.preamble_cycles + 1).bit_length())
    lines.append(f"  reg [{width - 1}:0] formal_step = 0;")
    lines.append(f"  always @({contract.clock_edge} {contract.clock_port}) begin")
    lines.append(
        f"    if (formal_step < {contract.preamble_cycles + 1}) "
        "formal_step <= formal_step + 1'b1;"
    )
    lines.append("  end")
    # Check stable outputs continuously after initialization.  Sampling only in
    # the posedge process would observe pre-NBA values and could miss a
    # one-cycle divergence that reconverges before the next edge.
    lines.append(
        f"  always @* if (formal_step >= {contract.preamble_cycles}) assert ({equality});"
    )
    if contract.kind == "reset":
        assert_value = int(contract.reset_active or 0)
        inactive_value = 1 - assert_value
        lines.append("  always @* begin")
        lines.append(
            f"    if (formal_step < {contract.preamble_cycles}) "
            f"assume ({contract.reset_port} == 1'b{assert_value});"
        )
        lines.append(
            f"    else assume ({contract.reset_port} == 1'b{inactive_value});"
        )
        lines.append("  end")
    else:
        lines.append("  always @* begin")
        for name, value in sorted(contract.preamble_assignments.items()):
            lines.append(
                f"    if (formal_step < {contract.preamble_cycles}) assume ({name} == {value});"
            )
        if not contract.preamble_assignments:
            lines.append("    // Preamble inputs are arbitrary but shared by both instances.")
        lines.append("  end")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def protocol_manifest(designs: list[tuple[str, str]]) -> dict[str, Any]:
    rows = []
    for seed_id, source in sorted(designs):
        contract = initialization_contract(seed_id, source)
        rows.append(
            {
                "seed_id": seed_id,
                "canonical_sha256": sha256_text(source),
                "contract": contract.to_dict(),
                "prob151_normalization_allowed": seed_id == "Prob151_review2015_fsm_ref",
                "output_validity_mask": (
                    "compare done always; compare out_bytes iff golden done=1"
                    if seed_id == "Prob154_fsm_ps2data_ref"
                    else None
                ),
            }
        )
    design_counts: dict[str, int] = {}
    for row in rows:
        kind = row["contract"]["kind"]
        design_counts[kind] = design_counts.get(kind, 0) + 1
    return {
        "schema_version": 1,
        "design_count": len(rows),
        "design_contract_counts": design_counts,
        "bmc": {
            "depth": BMC_COUNTEREXAMPLE_DEPTH,
            "pass_is_proof": False,
            "counterexample_requires_icarus_replay": True,
        },
        "independent_simulation": {
            "seeds": list(INDEPENDENT_SIMULATION_SEEDS),
            "cycles_per_seed": INDEPENDENT_SIMULATION_CYCLES,
            "may_be_used_for_mutation_selection_or_feedback": False,
        },
        "designs": rows,
    }


def write_protocol_manifest(designs: list[tuple[str, str]], path: Path) -> dict[str, Any]:
    manifest = protocol_manifest(designs)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest
