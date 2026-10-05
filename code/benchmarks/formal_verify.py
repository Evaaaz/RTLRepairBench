"""Executable formal and independent-simulation protocol for sem85.

The verifier is fail-closed:

* combinational designs are checked by an exhaustive Yosys SAT query;
* sequential designs use ABC PDR first, bounded model checking only to obtain a
  witness, and SMT induction as the second proof engine;
* a bounded PASS is never returned as :class:`FormalStatus.PROVED`;
* a formal counterexample is returned only after the trace is replayed by
  Icarus.  A legal design rejected by the formal frontend is ``UNSUPPORTED``,
  while a candidate rejected by both Yosys and Icarus is ``COMPILE_FAIL``.

Run ``python -m benchmarks.formal_verify --help`` for data construction,
single-pair verification, independent simulation, and the 85+85 calibration
gate.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import math
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import time
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

_CODE_ROOT = Path(__file__).resolve().parent.parent
if str(_CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(_CODE_ROOT))

from project_paths import project_root  # noqa: E402
from candidate_security import (  # noqa: E402
    candidate_security_reason,
    input_port_drive_reason,
)

from benchmarks.formal_data import (  # noqa: E402
    DEFAULT_BENCHMARK,
    DEFAULT_MANIFEST,
    DEFAULT_OUTPUT_DIR,
    SemanticCase,
    assert_seed_consistency,
    load_semantic_cases,
    write_canonical_dataset,
)
from benchmarks.formal_protocol import (  # noqa: E402
    BMC_COUNTEREXAMPLE_DEPTH,
    INDEPENDENT_SIMULATION_CYCLES,
    INDEPENDENT_SIMULATION_SEEDS,
    FormalResult,
    FormalStatus,
    InitializationContract,
    compatible_interfaces,
    generate_equivalence_wrapper,
    initialization_contract,
    normalize_prob151,
    parse_module,
    rename_top_module,
    sha256_text,
    write_protocol_manifest,
)


ROOT = project_root()
DEFAULT_PROTOCOL_MANIFEST = ROOT / "data" / "formal" / "protocol_manifest.json"
DEFAULT_ARTIFACTS = ROOT / "generated" / "formal"
TOOLCHAIN_LOCK = ROOT / "docker" / "formal" / "toolchain.lock.json"
DEFAULT_PROTOCOL_AMENDMENT = (
    ROOT / "configs" / "vericodegen" / "protocol_amendment_2026-07-15.json"
)
MAIN_2X2_ARMS = ("spec0_loc0", "spec1_loc0", "spec0_loc1", "spec1_loc1")
FORMAL_PROTOCOL_VERSION = "vericodegen-formal-v5"
SMT_INDUCTION_DEPTH = 64

# The Day-3 harness calibration has one audited simulator fallback.  This is
# deliberately a source-identity allowlist rather than a case-name exception:
# changing the benchmark, recovered golden, mutation, or seed makes the entry
# ineligible.  FormalResult remains UNSUPPORTED; the independent simulation is
# separate evidence that the known mutant is distinguishable by the harness.
COMPOSITE_CALIBRATION_POLICY_NAME = (
    "exact_allowlisted_unsupported_with_frozen_independent_simulation"
)
COMPOSITE_CALIBRATION_POLICY_VERSION = 1
COMPOSITE_CALIBRATION_ALLOWLIST = {
    "repair_sem_0059": {
        "case_id": "repair_sem_0059",
        "seed_id": "Prob129_ece241_2013_q8_ref",
        "mutation": "flip_reset_polarity",
        "golden_sha256": (
            "1ee2398da6c8170f1d7937eac091a160ec8bb037bc389ae0d66a8d19a5a49716"
        ),
        "candidate_sha256": (
            "35c79ef6db0aec962710791bc4a3465979d011f1ae883379ea5fdc1d6f2b57c4"
        ),
        "required_formal_status": FormalStatus.UNSUPPORTED.value,
    }
}

# Yosys treats its ``always_comb`` process attribute as a request to reject any
# inferred latch.  SystemVerilog simulation semantics still define such a
# process (and several canonical HDLBits designs intentionally omit a default
# arm for unreachable states).  Clearing the *tool attribute* after parsing,
# before ``prep`` lowers the original process to the corresponding $dlatch;
# it does not rewrite the RTL or add a default assignment.  Keep the selection
# reset beside it so later passes operate on the complete design.
YOSYS_LOWER_ALWAYS_COMB = "setattr -unset always_comb p:*; select -clear"


def formal_protocol_fingerprint() -> str:
    """Hash code, toolchain lock, and image identity that affect verdicts."""

    _load_protocol_amendment()
    digest = hashlib.sha256()
    digest.update(FORMAL_PROTOCOL_VERSION.encode("utf-8"))
    for path in (
        Path(__file__),
        Path(__file__).resolve().parent / "formal_protocol.py",
        _CODE_ROOT / "candidate_security.py",
        TOOLCHAIN_LOCK,
        DEFAULT_PROTOCOL_AMENDMENT,
    ):
        digest.update(path.name.encode("utf-8"))
        digest.update(path.read_bytes())
    digest.update(os.environ.get("RTLREPAIR_FORMAL_IMAGE_ID", "no-container-id").encode("utf-8"))
    return digest.hexdigest()


@dataclass
class CommandResult:
    command: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    elapsed_seconds: float
    timed_out: bool = False

    @property
    def log(self) -> str:
        return self.stdout + (("\n" if self.stdout else "") + self.stderr if self.stderr else "")


@dataclass
class PreparedPair:
    seed_id: str
    golden_original: str
    candidate_original: str
    golden_source: str
    candidate_source: str
    wrapper_source: str
    contract: InitializationContract
    info: Any
    normalized_prob151: bool


def _run(command: list[str], cwd: Path, timeout_seconds: int) -> CommandResult:
    started = time.monotonic()
    try:
        process = subprocess.Popen(
            command,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            start_new_session=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            return CommandResult(
                command, process.returncode, stdout, stderr, time.monotonic() - started
            )
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate()
            return CommandResult(
                command,
                None,
                stdout,
                stderr,
                time.monotonic() - started,
                timed_out=True,
            )
    except OSError as exc:
        return CommandResult(
            command, None, "", str(exc), time.monotonic() - started, timed_out=False
        )


def _normalize_sby_status(raw_status: str) -> str:
    """Return the semantic token from SBY's ``STATUS rc signal`` file."""

    fields = raw_status.strip().upper().split()
    return fields[0] if fields else "UNKNOWN"


def collect_toolchain_manifest() -> dict[str, Any]:
    """Capture the actual executable paths/versions used for a result set."""

    probes = {
        "yosys": ["yosys", "-V"],
        "sby": ["sby", "--version"],
        "abc": ["yosys-abc", "-h"],
        "z3": ["z3", "--version"],
        "boolector": ["boolector", "--version"],
        "iverilog": ["iverilog", "-V"],
        "vvp": ["vvp", "-V"],
        "verilator": ["verilator", "--version"],
    }
    tools: dict[str, Any] = {}
    for name, command in probes.items():
        executable = shutil.which(command[0])
        if executable is None:
            tools[name] = {"available": False, "path": None, "version": None}
            continue
        result = _run(command, ROOT, 15)
        version_lines = [line.strip() for line in result.log.splitlines() if line.strip()]
        tools[name] = {
            "available": True,
            "path": executable,
            "version": version_lines[0] if version_lines else "",
            "probe_returncode": result.returncode,
        }
    lock_sha = None
    if TOOLCHAIN_LOCK.exists():
        lock_sha = sha256_text(TOOLCHAIN_LOCK.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "container_image": os.environ.get("RTLREPAIR_FORMAL_IMAGE"),
        "container_image_id": os.environ.get("RTLREPAIR_FORMAL_IMAGE_ID"),
        "toolchain_lock_path": str(TOOLCHAIN_LOCK.relative_to(ROOT)),
        "toolchain_lock_sha256": lock_sha,
        "platform": platform.platform(),
        "python": sys.version,
        "tools": tools,
    }


def write_toolchain_manifest(path: Path) -> dict[str, Any]:
    manifest = collect_toolchain_manifest()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest


def _result(
    status: FormalStatus,
    engine: str,
    contract: InitializationContract,
    golden: str,
    candidate: str,
    started: float,
    **kwargs: Any,
) -> FormalResult:
    metadata = dict(kwargs.pop("metadata", {}))
    metadata.setdefault("formal_protocol_sha256", formal_protocol_fingerprint())
    return FormalResult(
        status=status,
        engine=engine,
        initialization_contract=contract,
        elapsed_seconds=time.monotonic() - started,
        candidate_sha256=sha256_text(candidate),
        golden_sha256=sha256_text(golden),
        metadata=metadata,
        **kwargs,
    )


def _candidate_security_reason(candidate: str) -> str:
    """Return a fail-closed reason for candidate-controlled verifier logic."""

    return candidate_security_reason(
        candidate,
        reserved_hierarchy=(
            "equiv_top",
            "independent_tb",
            "replay_tb",
            "design_golden",
        ),
    )


def _candidate_port_security_reason(candidate: str, candidate_info: Any) -> str:
    """Reject candidate ports that can drive verifier-owned shared nets."""

    inout_names = sorted(
        port.name for port in candidate_info.ports if port.direction == "inout"
    )
    if inout_names:
        return (
            "candidate security preflight rejected inout port(s): "
            + ", ".join(inout_names)
        )
    return input_port_drive_reason(
        candidate,
        (port.name for port in candidate_info.ports if port.direction == "input"),
    )


def _prepare_pair(seed_id: str, golden: str, candidate: str) -> tuple[PreparedPair | None, str]:
    security_reason = _candidate_security_reason(candidate)
    if security_reason:
        return None, security_reason
    golden_info = parse_module(golden)
    candidate_info = parse_module(candidate)
    if golden_info is None:
        return None, "golden module is outside the audited ANSI-port subset"
    if candidate_info is None:
        return None, "candidate has no parseable ANSI module/interface"
    port_security_reason = _candidate_port_security_reason(candidate, candidate_info)
    if port_security_reason:
        return None, port_security_reason
    compatible, reason = compatible_interfaces(golden_info, candidate_info)
    if not compatible:
        return None, reason
    contract = initialization_contract(seed_id, golden)
    if contract.kind == "unsupported":
        return None, contract.description

    try:
        golden_normalized, golden_changed = normalize_prob151(seed_id, golden)
        candidate_normalized, candidate_changed = normalize_prob151(seed_id, candidate)
    except ValueError as exc:
        return None, f"audited frontend normalization rejected source drift: {exc}"
    golden_renamed, _ = rename_top_module(golden_normalized, "design_golden")
    candidate_renamed, _ = rename_top_module(candidate_normalized, "design_candidate")
    wrapper = generate_equivalence_wrapper(golden_info, contract, seed_id=seed_id)
    return (
        PreparedPair(
            seed_id=seed_id,
            golden_original=golden,
            candidate_original=candidate,
            golden_source=golden_renamed,
            candidate_source=candidate_renamed,
            wrapper_source=wrapper,
            contract=contract,
            info=golden_info,
            normalized_prob151=golden_changed or candidate_changed,
        ),
        "",
    )


def _prepare_simulation_pair(
    seed_id: str, golden: str, candidate: str
) -> tuple[PreparedPair | None, str]:
    """Prepare source for independent four-state simulation without lowering.

    Formal compatibility normalization is intentionally absent here.  In
    particular, Prob151's enum casts and X-output guard must reach Icarus
    unchanged so the fallback remains independent of the two-state formal
    abstraction.  Only interface validation, initialization-contract lookup,
    and deterministic top-module renaming are shared with formal preparation.
    """

    security_reason = _candidate_security_reason(candidate)
    if security_reason:
        return None, security_reason
    golden_info = parse_module(golden)
    candidate_info = parse_module(candidate)
    if golden_info is None:
        return None, "golden module is outside the audited ANSI-port subset"
    if candidate_info is None:
        return None, "candidate has no parseable ANSI module/interface"
    port_security_reason = _candidate_port_security_reason(candidate, candidate_info)
    if port_security_reason:
        return None, port_security_reason
    compatible, reason = compatible_interfaces(golden_info, candidate_info)
    if not compatible:
        return None, reason
    contract = initialization_contract(seed_id, golden)
    if contract.kind == "unsupported":
        return None, contract.description
    golden_renamed, _ = rename_top_module(golden, "design_golden")
    candidate_renamed, _ = rename_top_module(candidate, "design_candidate")
    return (
        PreparedPair(
            seed_id=seed_id,
            golden_original=golden,
            candidate_original=candidate,
            golden_source=golden_renamed,
            candidate_source=candidate_renamed,
            wrapper_source=generate_equivalence_wrapper(
                golden_info, contract, seed_id=seed_id
            ),
            contract=contract,
            info=golden_info,
            normalized_prob151=False,
        ),
        "",
    )


def _write_pair(pair: PreparedPair, directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "golden.sv").write_text(pair.golden_source, encoding="utf-8")
    (directory / "candidate.sv").write_text(pair.candidate_source, encoding="utf-8")
    (directory / "equiv_top.sv").write_text(pair.wrapper_source, encoding="utf-8")


_LEGAL_YOSYS_UNSUPPORTED = (
    "unsupported",
    "syntax error, unexpected TOK_USER_TYPE",
    "syntax error, unexpected TOK_CAST",
    "cannot handle",
    "not supported in formal",
)


def _candidate_compiles_with_icarus(pair: PreparedPair, directory: Path, timeout: int = 20) -> bool | None:
    if shutil.which("iverilog") is None:
        return None
    command = [
        "iverilog",
        "-g2012",
        "-s",
        "design_candidate",
        "-o",
        str(directory / "candidate_compile.out"),
        str(directory / "candidate.sv"),
    ]
    return _run(command, directory, timeout).returncode == 0


def _source_compiles_with_icarus(source: str, directory: Path, timeout: int = 20) -> bool | None:
    """Classify an otherwise-unparsed candidate without assuming an ANSI top."""

    if shutil.which("iverilog") is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "unparsed_candidate.sv"
    source_path.write_text(source, encoding="utf-8")
    run = _run(
        ["iverilog", "-g2012", "-t", "null", str(source_path)], directory, timeout
    )
    (directory / "unparsed_candidate_compile.log").write_text(
        run.log, encoding="utf-8"
    )
    return run.returncode == 0


def _source_compiles_with_verilator(source: str, directory: Path, timeout: int = 20) -> bool | None:
    """Use the independent SV frontend to distinguish legal unsupported RTL."""

    if shutil.which("verilator") is None:
        return None
    directory.mkdir(parents=True, exist_ok=True)
    source_path = directory / "verilator_candidate.sv"
    source_path.write_text(source, encoding="utf-8")
    run = _run(
        ["verilator", "--lint-only", "--timing", "-Wno-fatal", str(source_path)],
        directory,
        timeout,
    )
    (directory / "verilator_candidate_compile.log").write_text(
        run.log, encoding="utf-8"
    )
    return run.returncode == 0


def _frontend_failure_result(
    pair: PreparedPair,
    run: CommandResult,
    directory: Path,
    started: float,
    engine: str,
) -> FormalResult:
    log_lower = run.log.lower()
    icarus_ok = _candidate_compiles_with_icarus(pair, directory)
    verilator_ok = _source_compiles_with_verilator(
        pair.candidate_original, directory / "verilator_frontend"
    )
    legal_unsupported = icarus_ok is True or verilator_ok is True or any(
        marker.lower() in log_lower for marker in _LEGAL_YOSYS_UNSUPPORTED
    )
    status = FormalStatus.UNSUPPORTED if legal_unsupported else FormalStatus.COMPILE_FAIL
    reason = (
        "formal frontend rejected RTL that Icarus accepts"
        if legal_unsupported
        else "candidate failed frontend compilation"
    )
    return _result(
        status,
        engine,
        pair.contract,
        pair.golden_original,
        pair.candidate_original,
        started,
        detail=f"{reason}: {run.log[-4000:]}",
        command=run.command,
        metadata={"prob151_normalized": pair.normalized_prob151},
    )


def _vcd_variable_map(lines: list[str]) -> dict[str, tuple[str, int, int]]:
    """Map a VCD identifier to (hierarchical name, width, scope depth)."""

    scope: list[str] = []
    variables: dict[str, tuple[str, int, int]] = {}
    for raw in lines:
        line = raw.strip()
        if line.startswith("$scope"):
            parts = line.split()
            if len(parts) >= 3:
                scope.append(parts[2])
        elif line.startswith("$upscope"):
            if scope:
                scope.pop()
        elif line.startswith("$var"):
            parts = line.split()
            if len(parts) >= 5:
                width = int(parts[2])
                code = parts[3]
                leaf = parts[4].lstrip("\\")
                variables[code] = (".".join(scope + [leaf]), width, len(scope))
        elif line.startswith("$enddefinitions"):
            break
    return variables


def _parse_vcd_states(
    path: Path,
    input_names: Iterable[str],
    clock_name: str | None,
    clock_edge: str | None = None,
) -> list[dict[str, str]]:
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    variables = _vcd_variable_map(lines)
    wanted = set(input_names)
    chosen: dict[str, str] = {}
    chosen_depth: dict[str, int] = {}
    for code, (hierarchy, _width, depth) in variables.items():
        leaf = hierarchy.rsplit(".", 1)[-1]
        if leaf in wanted and (leaf not in chosen or depth < chosen_depth[leaf]):
            chosen[leaf] = code
            chosen_depth[leaf] = depth
    if set(chosen) != wanted:
        missing = sorted(wanted - set(chosen))
        raise ValueError(f"VCD omits wrapper input(s): {missing}")
    code_to_name = {code: name for name, code in chosen.items()}

    current: dict[str, str] = {}
    snapshots: list[dict[str, str]] = []
    in_values = False
    for raw in lines:
        line = raw.strip()
        if line.startswith("#"):
            if in_values and len(current) == len(wanted):
                snapshots.append(dict(current))
            in_values = True
            continue
        if not in_values or not line or line.startswith("$"):
            continue
        if line[0] in "01xXzZ":
            value, code = line[0], line[1:]
        elif line[0] in "bB":
            parts = line[1:].split(None, 1)
            if len(parts) != 2:
                continue
            value, code = parts
        else:
            continue
        if code in code_to_name:
            current[code_to_name[code]] = value.lower()
    if in_values and len(current) == len(wanted):
        snapshots.append(dict(current))
    if not snapshots:
        raise ValueError("VCD contains no complete input sample")

    if clock_name is None:
        return [snapshots[-1]]
    edge = clock_edge or "posedge"
    edge_samples = []
    previous = "0" if edge == "posedge" else "1"
    for snapshot in snapshots:
        clock = snapshot.get(clock_name, "0")[-1]
        triggered = (
            previous == "0" and clock == "1"
            if edge == "posedge"
            else previous == "1" and clock == "0"
        )
        if triggered:
            edge_samples.append(snapshot)
        previous = clock
    # smtbmc traces sometimes omit explicit half-cycle clock toggles.  In that
    # representation each timestamp is one logical edge.
    return edge_samples or snapshots


def _sv_literal(bits: str, width: int) -> str:
    normalized = bits.lower().replace("x", "0").replace("z", "0")
    normalized = normalized[-width:].rjust(width, "0")
    return f"{width}'b{normalized}"


_SMTBMC_INTERNAL_INIT = re.compile(
    r"^\s*UUT\.(golden|candidate)\.([^=]+?)\s*=\s*"
    r"([1-9][0-9]*'[bB][01_]+)\s*;\s*$"
)
_SAFE_INTERNAL_PATH = re.compile(r"^[A-Za-z_$][A-Za-z0-9_$]*(?:\[[0-9]+\])?"
                                 r"(?:\.[A-Za-z_$][A-Za-z0-9_$]*(?:\[[0-9]+\])?)*$")


def _parse_smtbmc_internal_init(trace_tb: Path | None) -> list[dict[str, str]]:
    """Extract concrete register initializers emitted by ``yosys-smtbmc``.

    A formal trace starts from an arbitrary *two-state* register valuation.
    Icarus instead starts uninitialized registers at X.  Replaying only the
    public inputs can therefore lose a valid witness when the bug prevents
    reset from initializing a register.  The generated ``trace_tb.v`` records
    the concrete valuation selected by the solver.  We accept only simple,
    known binary assignments below the two DUT instances; X/Z values, escaped
    identifiers, expressions, and wrapper state are deliberately rejected.
    """

    if trace_tb is None or not trace_tb.exists():
        return []
    assignments: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()
    for line in trace_tb.read_text(encoding="utf-8", errors="replace").splitlines():
        match = _SMTBMC_INTERNAL_INIT.match(line)
        if match is None:
            continue
        instance, path, value = match.groups()
        path = path.strip()
        if _SAFE_INTERNAL_PATH.fullmatch(path) is None:
            continue
        # SBY's witness testbench also names synthesis-only anyinit cells
        # below a synthetic ``_witness_`` scope.  They do not exist in the
        # source RTL compiled by Icarus and must never be injected there.
        if path == "_witness_" or path.startswith("_witness_."):
            continue
        key = (instance, path)
        if key in seen:
            continue
        seen.add(key)
        assignments.append(
            {"instance": instance, "path": path, "value": value.lower()}
        )
    return assignments


def _generate_replay_tb(
    pair: PreparedPair,
    states: list[dict[str, str]],
    internal_init: list[dict[str, str]] | None = None,
) -> str:
    info, contract = pair.info, pair.contract
    lines = ["`timescale 1ns/1ps", "module replay_tb;"]
    for port in info.inputs:
        width = "" if port.width == 1 else f"[{port.width - 1}:0] "
        lines.append(f"  reg {width}{port.name};")
        lines.append(f"  wire {width}rtlrepair_candidate_input_{port.name};")
        lines.append(
            f"  assign rtlrepair_candidate_input_{port.name} = {port.name};"
        )
    for port in info.outputs:
        width = "" if port.width == 1 else f"[{port.width - 1}:0] "
        lines.append(f"  wire {width}{port.name}_golden, {port.name}_candidate;")
    lines.append(f"  design_golden golden ({_connections(info, 'golden')});")
    lines.append(
        "  design_candidate candidate "
        f"({_connections(info, 'candidate', isolated_candidate_inputs=True)});"
    )
    lines.extend(["  integer mismatches;", "  initial begin", "    mismatches = 0;"])
    lines.append('    $dumpfile("replay.vcd");')
    lines.append("    $dumpvars(0, replay_tb);")
    for port in info.inputs:
        lines.append(f"    {port.name} = 0;")
    for assignment in internal_init or []:
        lines.append(
            f"    {assignment['instance']}.{assignment['path']} = "
            f"{assignment['value']};"
        )

    sequential = contract.kind in {"reset", "preamble"}
    valid_from_cycle = max(0, contract.preamble_cycles - 1) if sequential else 0
    for cycle, state in enumerate(states):
        for port in info.inputs:
            if sequential and port.name == contract.clock_port:
                continue
            if port.name not in state:
                continue
            lines.append(f"    {port.name} = {_sv_literal(state[port.name], port.width)};")
        if sequential:
            if contract.clock_edge == "negedge":
                lines.append(f"    {contract.clock_port} = 1; #1;")
                lines.append(f"    {contract.clock_port} = 0; #1;")
            else:
                lines.append(f"    {contract.clock_port} = 0; #1;")
                lines.append(f"    {contract.clock_port} = 1; #1;")
        else:
            lines.append("    #1;")
        for input_port in info.inputs:
            lines.append(
                f'    $display("TRACE_INPUT cycle={cycle} name={input_port.name} '
                f'value=%h", {input_port.name});'
            )
        for output in info.outputs:
            lines.append(
                f'    $display("TRACE_OUTPUT cycle={cycle} name={output.name} '
                f'expected=%h got=%h", {output.name}_golden, {output.name}_candidate);'
            )
            if cycle >= valid_from_cycle:
                lines.append(
                    f"    if ((^{output.name}_golden !== 1'bx) && "
                    f"(^{output.name}_candidate !== 1'bx) && "
                    f"({output.name}_golden !== {output.name}_candidate)) begin"
                )
                lines.append(
                    f'      $display("REPLAY_MISMATCH cycle={cycle} output={output.name} '
                    f'expected=%h got=%h", {output.name}_golden, {output.name}_candidate);'
                )
                lines.extend(["      mismatches = mismatches + 1;", "    end"])
        if sequential:
            idle = 1 if contract.clock_edge == "negedge" else 0
            lines.append(f"    {contract.clock_port} = {idle}; #1;")
    lines.append('    if (mismatches > 0) $display("REPLAY_PASS");')
    lines.append('    else $display("REPLAY_FAIL no divergence reproduced");')
    lines.extend(["    $finish;", "  end", "endmodule"])
    return "\n".join(lines) + "\n"


def _connections(
    info: Any,
    suffix: str,
    *,
    isolated_candidate_inputs: bool = False,
) -> str:
    parts = []
    for port in info.inputs:
        signal = (
            f"rtlrepair_candidate_input_{port.name}"
            if isolated_candidate_inputs
            else port.name
        )
        connection = "{" + signal + "}" if isolated_candidate_inputs else signal
        parts.append(f".{port.name}({connection})")
    parts.extend(f".{port.name}({port.name}_{suffix})" for port in info.outputs)
    return ", ".join(parts)


def replay_counterexample(
    pair: PreparedPair,
    formal_vcd: Path,
    directory: Path,
    timeout_seconds: int = 30,
) -> tuple[bool, Path | None, Path | None, str]:
    """Replay a formal trace in Icarus and persist a machine-readable witness."""

    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        return False, None, None, "Icarus tools are unavailable"
    try:
        states = _parse_vcd_states(
            formal_vcd,
            [port.name for port in pair.info.inputs],
            pair.contract.clock_port,
            pair.contract.clock_edge,
        )
    except (OSError, ValueError) as exc:
        return False, None, None, f"cannot parse formal VCD: {exc}"

    internal_init = _parse_smtbmc_internal_init(
        formal_vcd.with_name("trace_tb.v")
    )

    replay_dir = directory / "replay"
    replay_dir.mkdir(parents=True, exist_ok=True)
    tb_path = replay_dir / "replay_tb.sv"
    tb_path.write_text(
        _generate_replay_tb(pair, states, internal_init), encoding="utf-8"
    )
    executable = replay_dir / "replay.out"
    compile_result = _run(
        [
            "iverilog",
            "-g2012",
            "-o",
            str(executable),
            str(directory / "golden.sv"),
            str(directory / "candidate.sv"),
            str(tb_path),
        ],
        replay_dir,
        timeout_seconds,
    )
    if compile_result.returncode != 0:
        return False, None, None, f"Icarus replay compile failed: {compile_result.log}"
    run_result = _run(["vvp", str(executable)], replay_dir, timeout_seconds)
    log_path = replay_dir / "replay.log"
    log_path.write_text(run_result.log, encoding="utf-8")
    replay_vcd = replay_dir / "replay.vcd"
    witness_path = replay_dir / "witness.json"
    trace_by_cycle: dict[int, dict[str, Any]] = {}
    first_divergence: dict[str, Any] | None = None
    valid_from_cycle = (
        max(0, pair.contract.preamble_cycles - 1)
        if pair.contract.kind in {"reset", "preamble"}
        else 0
    )
    for match in re.finditer(
        r"TRACE_INPUT cycle=(\d+) name=([^ ]+) value=([^\s]+)", run_result.log
    ):
        cycle = int(match.group(1))
        trace_by_cycle.setdefault(cycle, {"cycle": cycle, "inputs": {}, "outputs": {}})
        trace_by_cycle[cycle]["inputs"][match.group(2)] = match.group(3)
    for match in re.finditer(
        r"TRACE_OUTPUT cycle=(\d+) name=([^ ]+) expected=([^\s]+) got=([^\s]+)",
        run_result.log,
    ):
        cycle = int(match.group(1))
        output = {
            "expected": match.group(3),
            "got": match.group(4),
            "eligible": cycle >= valid_from_cycle,
            "known": not bool(re.search(r"[xz]", match.group(3) + match.group(4), re.I)),
        }
        output["diverged"] = (
            output["eligible"]
            and output["known"]
            and match.group(3).lower() != match.group(4).lower()
        )
        trace_by_cycle.setdefault(cycle, {"cycle": cycle, "inputs": {}, "outputs": {}})
        trace_by_cycle[cycle]["outputs"][match.group(2)] = output
        if output["diverged"] and first_divergence is None:
            first_divergence = {
                "cycle": cycle,
                "output": match.group(2),
                "expected": output["expected"],
                "got": output["got"],
            }
    structured_trace = [trace_by_cycle[index] for index in sorted(trace_by_cycle)]
    if first_divergence is not None:
        structured_trace = [
            row for row in structured_trace if row["cycle"] <= first_divergence["cycle"]
        ]
    witness = {
        "source_vcd": str(formal_vcd),
        "internal_initialization": internal_init,
        "input_trace": states,
        "trace": structured_trace,
        "initialization_cycles": pair.contract.preamble_cycles,
        "valid_from_cycle": valid_from_cycle,
        "first_divergence": first_divergence,
        "icarus_returncode": run_result.returncode,
        "replayed": (
            run_result.returncode == 0
            and "REPLAY_PASS" in run_result.log
            and first_divergence is not None
        ),
        "replay_log": run_result.log,
    }
    witness_path.write_text(json.dumps(witness, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return witness["replayed"], (replay_vcd if replay_vcd.exists() else None), witness_path, run_result.log


def _counterexample_result(
    pair: PreparedPair,
    formal_vcd: Path,
    directory: Path,
    started: float,
    engine: str,
    command: list[str],
    detail: str,
) -> FormalResult:
    replayed, replay_vcd, witness, replay_log = replay_counterexample(
        pair, formal_vcd, directory
    )
    witness_record = None
    used_internal_initialization = False
    if witness is not None and witness.exists():
        full_witness = json.loads(witness.read_text(encoding="utf-8"))
        used_internal_initialization = bool(
            full_witness.get("internal_initialization")
        )
        # The ledger/prompt receives only the grounded semantic trace.  Raw VCD
        # samples and simulator logs stay in the artifact referenced by
        # witness_path, avoiding accidental prompt bloat and host-path leakage.
        witness_record = {
            "initialization_contract": pair.contract.to_dict(),
            "initialization_cycles": full_witness.get("initialization_cycles"),
            "valid_from_cycle": full_witness.get("valid_from_cycle"),
            "trace": full_witness.get("trace"),
            "first_divergence": full_witness.get("first_divergence"),
        }
    if not replayed:
        return _result(
            FormalStatus.UNSUPPORTED,
            engine,
            pair.contract,
            pair.golden_original,
            pair.candidate_original,
            started,
            vcd_path=str(formal_vcd),
            witness_path=str(witness) if witness else None,
            witness=witness_record,
            counterexample_replayed=False,
            detail=f"formal witness was not Icarus-replayable: {replay_log[-3000:]}",
            command=command,
            metadata={
                "prob151_normalized": pair.normalized_prob151,
                "replay_internal_initialization": used_internal_initialization,
            },
        )
    return _result(
        FormalStatus.COUNTEREXAMPLE,
        engine,
        pair.contract,
        pair.golden_original,
        pair.candidate_original,
        started,
        vcd_path=str(replay_vcd or formal_vcd),
        witness_path=str(witness),
        witness=witness_record,
        counterexample_replayed=True,
        detail=detail,
        command=command,
        metadata={
            "prob151_normalized": pair.normalized_prob151,
            "replay_internal_initialization": used_internal_initialization,
        },
    )


class FormalVerifier:
    def __init__(self, fast_timeout: int = 60, fallback_timeout: int = 300):
        self.fast_timeout = fast_timeout
        self.fallback_timeout = fallback_timeout

    def verify(
        self,
        seed_id: str,
        golden: str,
        candidate: str,
        artifact_dir: Path,
    ) -> FormalResult:
        started = time.monotonic()
        artifact_dir = artifact_dir.resolve()
        pair, reason = _prepare_pair(seed_id, golden, candidate)
        contract = initialization_contract(seed_id, golden)
        if pair is None:
            if reason == "candidate has no parseable ANSI module/interface":
                compiles = _source_compiles_with_icarus(candidate, artifact_dir)
                verilator_compiles = _source_compiles_with_verilator(
                    candidate, artifact_dir / "verilator_frontend"
                )
                status = (
                    FormalStatus.COMPILE_FAIL
                    if compiles is False and verilator_compiles is False
                    else FormalStatus.UNSUPPORTED
                )
                reason += (
                    "; Icarus compilation failed"
                    if compiles is False and verilator_compiles is False
                    else "; legal/unclassified frontend form, use simulation fallback"
                )
            else:
                status = (
                    FormalStatus.COMPILE_FAIL
                    if reason.startswith("candidate interface")
                    else FormalStatus.UNSUPPORTED
                )
            return _result(status, "preflight", contract, golden, candidate, started, detail=reason)
        _write_pair(pair, artifact_dir)
        if contract.kind == "combinational":
            return self._verify_combinational(pair, artifact_dir, started)
        return self._verify_sequential(pair, artifact_dir, started)

    def _frontend_preflight(
        self, pair: PreparedPair, directory: Path, started: float
    ) -> FormalResult | None:
        if shutil.which("yosys") is None:
            return _result(
                FormalStatus.UNSUPPORTED,
                "yosys-frontend",
                pair.contract,
                pair.golden_original,
                pair.candidate_original,
                started,
                detail="yosys executable is unavailable",
            )
        script = (
            "read_verilog -formal -sv golden.sv candidate.sv equiv_top.sv; "
            f"{YOSYS_LOWER_ALWAYS_COMB}; prep -top equiv_top"
        )
        run = _run(["yosys", "-q", "-p", script], directory, min(30, self.fast_timeout))
        if run.returncode == 0:
            return None
        return _frontend_failure_result(
            pair, run, directory, started, "yosys-frontend"
        )

    def _verify_combinational(
        self, pair: PreparedPair, directory: Path, started: float
    ) -> FormalResult:
        preflight = self._frontend_preflight(pair, directory, started)
        if preflight is not None:
            return preflight
        vcd = directory / "formal_counterexample.vcd"
        log = directory / "yosys_sat.log"
        script = (
            "read_verilog -formal -sv golden.sv candidate.sv equiv_top.sv; "
            f"{YOSYS_LOWER_ALWAYS_COMB}; prep -top equiv_top -flatten; "
            "memory_map; opt_clean; chformal -lower; "
            f"sat -verify -prove-asserts -set-def-formal -show-all -dump_vcd {vcd}"
        )
        run = _run(
            ["yosys", "-ql", str(log), "-p", script],
            directory,
            self.fast_timeout,
        )
        if run.timed_out:
            return _result(
                FormalStatus.TIMEOUT,
                "yosys-sat",
                pair.contract,
                pair.golden_original,
                pair.candidate_original,
                started,
                detail=f"exhaustive SAT exceeded {self.fast_timeout}s",
                command=run.command,
            )
        full_log = run.log + (log.read_text(errors="replace") if log.exists() else "")
        if run.returncode == 0:
            return _result(
                FormalStatus.PROVED,
                "yosys-sat-exhaustive",
                pair.contract,
                pair.golden_original,
                pair.candidate_original,
                started,
                detail="exhaustive combinational SAT proof",
                command=run.command,
                metadata={"prob151_normalized": pair.normalized_prob151},
            )
        proof_failed = bool(
            re.search(r"proof did fail|model found:\s*fail|assert.*failed", full_log, re.I)
        )
        if proof_failed and vcd.exists():
            return _counterexample_result(
                pair,
                vcd,
                directory,
                started,
                "yosys-sat-exhaustive",
                run.command,
                "exhaustive SAT counterexample replayed in Icarus",
            )
        return _frontend_failure_result(pair, run, directory, started, "yosys-sat")

    def _sby_config(self, mode: str, engine: str, depth: int | None = None) -> str:
        # The audited corpus is single-clock.  In Yosys 0.64 the multiclock
        # transform can disconnect flattened child flops from the gclk and
        # produce a false gold-vs-gold counterexample.
        options = [f"mode {mode}"]
        if depth is not None:
            options.append(f"depth {depth}")
        return "\n".join(
            [
                "[options]",
                *options,
                "",
                "[engines]",
                engine,
                "",
                "[script]",
                # ``read`` is deferred by SBY, so no process exists yet when
                # the following attribute-lowering pass runs.  Invoke the
                # frontend explicitly to preserve the intended pass order.
                "read_verilog -formal -sv golden.sv candidate.sv equiv_top.sv",
                YOSYS_LOWER_ALWAYS_COMB,
                "prep -top equiv_top",
                "chformal -lower",
                "",
                "[files]",
                "golden.sv",
                "candidate.sv",
                "equiv_top.sv",
                "",
            ]
        )

    def _run_sby(
        self,
        directory: Path,
        stage: str,
        mode: str,
        engine: str,
        timeout: int,
        depth: int | None = None,
    ) -> tuple[str, CommandResult, Path | None]:
        config = directory / f"{stage}.sby"
        config.write_text(self._sby_config(mode, engine, depth), encoding="utf-8")
        work = directory / stage
        if work.exists():
            shutil.rmtree(work)
        run = _run(["sby", "-f", "-d", str(work), str(config)], directory, timeout)
        status = "TIMEOUT" if run.timed_out else "UNKNOWN"
        status_path = work / "status"
        if status_path.exists():
            raw_status = status_path.read_text(
                encoding="utf-8", errors="replace"
            ).strip().upper()
            # SBY 0.64 persists e.g. ``PASS 0 0`` or ``ERROR 16 0``;
            # only the first field is the semantic status.
            status = _normalize_sby_status(raw_status)
        else:
            match = re.findall(r"(?:DONE \([^)]*\)|summary):.*\b(PASS|FAIL|UNKNOWN|TIMEOUT)\b", run.log, re.I)
            if match:
                status = match[-1].upper()
        traces = sorted(work.rglob("*.vcd")) if work.exists() else []
        trace = next((path for path in traces if path.name == "trace.vcd"), traces[0] if traces else None)
        return status, run, trace

    def _verify_sequential(
        self, pair: PreparedPair, directory: Path, started: float
    ) -> FormalResult:
        if shutil.which("sby") is None:
            return _result(
                FormalStatus.UNSUPPORTED,
                "sby",
                pair.contract,
                pair.golden_original,
                pair.candidate_original,
                started,
                detail="sby executable is unavailable",
            )
        preflight = self._frontend_preflight(pair, directory, started)
        if preflight is not None:
            return preflight

        # Primary unbounded engine.
        status, pdr, _pdr_trace = self._run_sby(
            directory, "pdr", "prove", "abc pdr", self.fast_timeout
        )
        if status == "PASS":
            return _result(
                FormalStatus.PROVED,
                "abc-pdr",
                pair.contract,
                pair.golden_original,
                pair.candidate_original,
                started,
                detail="unbounded ABC PDR proof",
                command=pdr.command,
                metadata={"prob151_normalized": pair.normalized_prob151},
            )
        # ABC can prove/falsify the property yet fail while converting its
        # witness back to RTL names.  Frontend parsing was already checked
        # above.  PDR is proof-only in this protocol: every FAIL/ERROR falls
        # through to smtbmc BMC, the sole source of reportable witnesses.

        # BMC is witness-only.  PASS continues to induction and is never proof.
        bmc_status, bmc, bmc_trace = self._run_sby(
            directory,
            "bmc256",
            "bmc",
            "smtbmc z3",
            self.fast_timeout,
            BMC_COUNTEREXAMPLE_DEPTH,
        )
        if bmc_status == "FAIL" and bmc_trace is not None:
            return _counterexample_result(
                pair,
                bmc_trace,
                directory,
                started,
                "smtbmc-bmc256",
                bmc.command,
                "depth-256 BMC counterexample replayed in Icarus; BMC was not used as proof",
            )

        induction_status, induction, _induction_trace = self._run_sby(
            directory,
            "induction",
            "prove",
            "smtbmc z3",
            self.fallback_timeout,
            SMT_INDUCTION_DEPTH,
        )
        if induction_status == "PASS":
            return _result(
                FormalStatus.PROVED,
                "smtbmc-z3-induction",
                pair.contract,
                pair.golden_original,
                pair.candidate_original,
                started,
                detail="SMT induction proof after PDR did not conclude",
                command=induction.command,
                metadata={
                    "prob151_normalized": pair.normalized_prob151,
                    "bmc_status": bmc_status,
                    "bmc_pass_used_as_proof": False,
                    "induction_depth": SMT_INDUCTION_DEPTH,
                },
            )
        statuses = (status, bmc_status, induction_status)
        actually_timed_out = any(
            stage_status == "TIMEOUT" or stage_run.timed_out
            for stage_status, stage_run in (
                (status, pdr),
                (bmc_status, bmc),
                (induction_status, induction),
            )
        )
        if "ERROR" in statuses and not actually_timed_out:
            # Parsing succeeded in the explicit preflight, but a later formal
            # lowering/backend (for example AIG conversion of a combinational
            # loop) rejected the legal source.  Calling this TIMEOUT obscures
            # the support boundary and can inflate formal coverage.
            logs = "\n".join((pdr.log, bmc.log, induction.log))
            return _result(
                FormalStatus.UNSUPPORTED,
                "formal-backend",
                pair.contract,
                pair.golden_original,
                pair.candidate_original,
                started,
                detail=(
                    f"formal backend error after successful frontend; "
                    f"PDR={status}, BMC={bmc_status}, induction={induction_status}: "
                    f"{logs[-4000:]}"
                ),
                command=induction.command,
                metadata={
                    "prob151_normalized": pair.normalized_prob151,
                    "bmc_pass_used_as_proof": False,
                    "induction_depth": SMT_INDUCTION_DEPTH,
                },
            )
        # An induction-step FAIL can be an unreachable state and is not a
        # concrete execution.  Only the preceding depth-256 base-case BMC may
        # supply a counterexample; induction is PASS=>proof, otherwise unknown.
        detail = (
            f"PDR={status}, BMC={bmc_status} (witness-only), induction={induction_status}; "
            f"fast/fallback limits={self.fast_timeout}/{self.fallback_timeout}s"
        )
        return _result(
            FormalStatus.TIMEOUT,
            "abc-pdr+smtbmc-z3-induction",
            pair.contract,
            pair.golden_original,
            pair.candidate_original,
            started,
            detail=detail,
            command=induction.command,
            metadata={
                "bmc_pass_used_as_proof": False,
                "induction_depth": SMT_INDUCTION_DEPTH,
            },
        )


def _random_expression(width: int) -> str:
    words = max(1, math.ceil(width / 32))
    if words == 1:
        return "$random(seed_state)"
    return "{" + ", ".join("$random(seed_state)" for _ in range(words)) + "}"


def _independent_output_checks(
    pair: PreparedPair,
    *,
    indent: str,
    phase: str,
    sample_point: str,
    cycle_expression: str,
) -> list[str]:
    """Emit validity-aware differential checks at one simulation sample."""

    lines: list[str] = []
    for output in pair.info.outputs:
        if pair.seed_id == "Prob154_fsm_ps2data_ref" and output.name == "out_bytes":
            # Match the audited formal wrapper: out_bytes is public/meaningful
            # only while the golden design asserts done.  The done bit itself
            # is still compared unconditionally below.
            condition = (
                "(done_golden === 1'b1) && "
                "(out_bytes_golden !== out_bytes_candidate)"
            )
        else:
            condition = f"{output.name}_golden !== {output.name}_candidate"
        lines.append(f"{indent}if ({condition}) begin")
        lines.append(
            f'{indent}  $display("SIM_COUNTEREXAMPLE phase={phase} '
            f'sample_point={sample_point} cycle=%0d output={output.name} '
            f'expected=%h got=%h", {cycle_expression}, '
            f'{output.name}_golden, {output.name}_candidate);'
        )
        lines.extend([f"{indent}  mismatches = mismatches + 1;", f"{indent}end"])
    return lines


def _generate_independent_tb(pair: PreparedPair, cycles: int) -> str:
    info, contract = pair.info, pair.contract
    lines = ["`timescale 1ns/1ps", "module independent_tb;"]
    for port in info.inputs:
        width = "" if port.width == 1 else f"[{port.width - 1}:0] "
        lines.append(f"  reg {width}{port.name};")
        lines.append(f"  wire {width}rtlrepair_candidate_input_{port.name};")
        lines.append(
            f"  assign rtlrepair_candidate_input_{port.name} = {port.name};"
        )
    for port in info.outputs:
        width = "" if port.width == 1 else f"[{port.width - 1}:0] "
        lines.append(f"  wire {width}{port.name}_golden, {port.name}_candidate;")
    lines.append(f"  design_golden golden ({_connections(info, 'golden')});")
    lines.append(
        "  design_candidate candidate "
        f"({_connections(info, 'candidate', isolated_candidate_inputs=True)});"
    )
    lines.extend(
        [
            "  integer seed_state, cycle, preamble, mismatches;",
            "  reg [255:0] verdict_nonce;",
            "  initial begin",
        ]
    )
    lines.append('    if (!$value$plusargs("SEED=%d", seed_state)) seed_state = 1001;')
    lines.append(
        '    if (!$value$plusargs("RTLREPAIR_VERDICT_NONCE=%s", verdict_nonce)) '
        '$fatal(1, "missing verifier nonce");'
    )
    lines.append('    $dumpfile("independent.vcd");')
    lines.append("    $dumpvars(0, independent_tb);")
    lines.append("    mismatches = 0;")
    for port in info.inputs:
        lines.append(f"    {port.name} = 0;")
    sequential = contract.kind in {"reset", "preamble"}

    if sequential:
        for name, value in contract.preamble_assignments.items():
            lines.append(f"    {name} = {value};")
        if contract.kind == "reset":
            lines.append(f"    {contract.reset_port} = {int(contract.reset_active or 0)};")
        lines.append(
            f"    for (preamble = 0; preamble < {contract.preamble_cycles}; "
            "preamble = preamble + 1) begin"
        )
        for port in info.inputs:
            if port.name in {contract.clock_port, contract.reset_port}:
                continue
            if port.name not in contract.preamble_assignments:
                lines.append(f"      {port.name} = {_random_expression(port.width)};")
        if contract.clock_edge == "negedge":
            lines.append(
                f"      {contract.clock_port} = 1; #1; "
                f"{contract.clock_port} = 0; #1;"
            )
        else:
            lines.append(
                f"      {contract.clock_port} = 0; #1; "
                f"{contract.clock_port} = 1; #1;"
            )
        if contract.kind == "reset":
            lines.extend(
                _independent_output_checks(
                    pair,
                    indent="      ",
                    phase="initialization",
                    sample_point="post_edge",
                    cycle_expression="preamble",
                )
            )
        lines.append("    end")
        idle = 1 if contract.clock_edge == "negedge" else 0
        lines.append(f"    {contract.clock_port} = {idle};")
        if contract.kind == "reset":
            lines.append(f"    {contract.reset_port} = {1 - int(contract.reset_active or 0)};")

    lines.append(f"    for (cycle = 0; cycle < {cycles}; cycle = cycle + 1) begin")
    for port in info.inputs:
        if sequential and port.name in {contract.clock_port, contract.reset_port}:
            continue
        lines.append(f"      {port.name} = {_random_expression(port.width)};")
    # Let blocking input assignments propagate through Mealy/combinational
    # outputs before the state transition.  Sampling only after the edge misses
    # bugs whose observable value is consumed by that edge (Prob129 exposed
    # this hole).
    lines.append("      #1;")
    if sequential:
        lines.extend(
            _independent_output_checks(
                pair,
                indent="      ",
                phase="post_initialization",
                sample_point="pre_edge",
                cycle_expression="cycle",
            )
        )
        if contract.clock_edge == "negedge":
            lines.append(
                f"      {contract.clock_port} = 0; #1;"
            )
        else:
            lines.append(
                f"      {contract.clock_port} = 1; #1;"
            )
        lines.extend(
            _independent_output_checks(
                pair,
                indent="      ",
                phase="post_initialization",
                sample_point="post_edge",
                cycle_expression="cycle",
            )
        )
    else:
        lines.extend(
            _independent_output_checks(
                pair,
                indent="      ",
                phase="post_initialization",
                sample_point="settled_combinational",
                cycle_expression="cycle",
            )
        )
    if sequential:
        idle = 1 if contract.clock_edge == "negedge" else 0
        lines.append(f"      {contract.clock_port} = {idle}; #1;")
    lines.append("    end")
    lines.append(
        '    if (mismatches == 0) $display("RTLREPAIR_SIM_VERDICT '
        'nonce=%0s status=PASS", verdict_nonce);'
    )
    lines.append(
        '    else $display("RTLREPAIR_SIM_VERDICT nonce=%0s '
        'status=COUNTEREXAMPLE", verdict_nonce);'
    )
    lines.extend(["    $finish;", "  end", "endmodule"])
    return "\n".join(lines) + "\n"


_SIM_COUNTEREXAMPLE_RE = re.compile(
    r"SIM_COUNTEREXAMPLE phase=([^\s]+) sample_point=([^\s]+) "
    r"cycle=(\d+) output=([^\s]+) expected=([^\s]+) got=([^\s]+)"
)


def _parse_independent_verdict(log: str, nonce: str) -> tuple[str | None, str]:
    """Accept exactly one nonce-bound verdict as the final non-empty log line."""

    expected = re.compile(
        rf"^RTLREPAIR_SIM_VERDICT nonce={re.escape(nonce)} "
        r"status=(PASS|COUNTEREXAMPLE)$"
    )
    nonempty_lines = [line for line in log.splitlines() if line.strip()]
    protocol_lines = [
        line for line in nonempty_lines if line.startswith("RTLREPAIR_SIM_VERDICT")
    ]
    if len(protocol_lines) != 1:
        return None, f"expected exactly one anchored verdict, found {len(protocol_lines)}"
    match = expected.fullmatch(protocol_lines[0])
    if match is None:
        return None, "anchored verdict carried an invalid or stale nonce/status"
    if not nonempty_lines or protocol_lines[0] != nonempty_lines[-1]:
        return None, "nonce-bound verdict was not the terminal log line"
    return match.group(1), ""


def _summarize_sim_counterexamples(log: str) -> dict[str, Any]:
    """Summarize simulator evidence without conflating reset X with 0/1 CEs."""

    samples: list[dict[str, Any]] = []
    by_sample: collections.Counter[str] = collections.Counter()
    first_post_initialization_concrete: dict[str, Any] | None = None
    post_initialization_concrete_count = 0
    for match in _SIM_COUNTEREXAMPLE_RE.finditer(log):
        phase, sample_point, cycle, output, expected, got = match.groups()
        concrete = not bool(re.search(r"[xz]", expected + got, re.IGNORECASE))
        sample = {
            "phase": phase,
            "sample_point": sample_point,
            "cycle": int(cycle),
            "output": output,
            "expected": expected,
            "got": got,
            "concrete_two_state": concrete,
        }
        samples.append(sample)
        by_sample[f"{phase}:{sample_point}"] += 1
        if phase == "post_initialization" and concrete:
            post_initialization_concrete_count += 1
            if first_post_initialization_concrete is None:
                first_post_initialization_concrete = sample
    return {
        "total": len(samples),
        "by_phase_and_sample_point": dict(sorted(by_sample.items())),
        "post_initialization_concrete_count": post_initialization_concrete_count,
        "first_post_initialization_concrete": first_post_initialization_concrete,
    }


def run_independent_simulation(
    seed_id: str,
    golden: str,
    candidate: str,
    artifact_dir: Path,
) -> dict[str, Any]:
    """Run the frozen 10 x 200 independent differential simulation protocol."""

    artifact_dir = artifact_dir.resolve()
    pair, reason = _prepare_simulation_pair(seed_id, golden, candidate)
    if pair is None:
        return {"status": "UNSUPPORTED", "reason": reason}
    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        return {"status": "UNSUPPORTED", "reason": "Icarus tools are unavailable"}
    _write_pair(pair, artifact_dir)
    tb_path = artifact_dir / "independent_tb.sv"
    tb_path.write_text(
        _generate_independent_tb(pair, INDEPENDENT_SIMULATION_CYCLES), encoding="utf-8"
    )
    executable = artifact_dir / "independent.out"
    compile_result = _run(
        [
            "iverilog",
            "-g2012",
            "-o",
            str(executable),
            str(artifact_dir / "golden.sv"),
            str(artifact_dir / "candidate.sv"),
            str(tb_path),
        ],
        artifact_dir,
        30,
    )
    compile_log_path = artifact_dir / "independent_compile.log"
    compile_log_path.write_text(compile_result.log, encoding="utf-8")
    if compile_result.returncode != 0:
        return {
            "status": "COMPILE_FAIL",
            "engine": "iverilog-g2012-original-rtl",
            "reason": "independent simulator rejected original unnormalized RTL",
            "log": compile_result.log,
            "log_path": str(compile_log_path),
            "formal_protocol_sha256": formal_protocol_fingerprint(),
        }
    runs = []
    status = "PASS"
    for simulation_seed in INDEPENDENT_SIMULATION_SEEDS:
        run_dir = artifact_dir / f"seed_{simulation_seed}"
        run_dir.mkdir(parents=True, exist_ok=True)
        verdict_nonce = secrets.token_hex(16)
        result = _run(
            [
                "vvp",
                str(executable),
                f"+SEED={simulation_seed}",
                f"+RTLREPAIR_VERDICT_NONCE={verdict_nonce}",
            ],
            run_dir,
            30,
        )
        log_path = run_dir / "simulation.log"
        log_path.write_text(result.log, encoding="utf-8")
        counterexample_summary = _summarize_sim_counterexamples(result.log)
        anchored_verdict, verdict_error = _parse_independent_verdict(
            result.log, verdict_nonce
        )
        passed = result.returncode == 0 and anchored_verdict == "PASS"
        if result.timed_out:
            run_status = "TIMEOUT"
        elif result.returncode == 0 and anchored_verdict is not None:
            run_status = anchored_verdict
        else:
            run_status = "ERROR"
        runs.append(
            {
                "seed": simulation_seed,
                "cycles": INDEPENDENT_SIMULATION_CYCLES,
                "passed": passed,
                "status": run_status,
                "verdict_protocol_error": verdict_error or None,
                "counterexample_summary": counterexample_summary,
                "log_path": str(log_path),
                "vcd_path": str(run_dir / "independent.vcd"),
            }
        )
        if run_status in {"TIMEOUT", "ERROR"}:
            status = run_status
        elif run_status == "COUNTEREXAMPLE" and status == "PASS":
            status = "COUNTEREXAMPLE"
    aggregate_by_sample: collections.Counter[str] = collections.Counter()
    aggregate_post_initialization_concrete = 0
    first_post_initialization_concrete = None
    for run in runs:
        summary = run["counterexample_summary"]
        aggregate_by_sample.update(summary["by_phase_and_sample_point"])
        aggregate_post_initialization_concrete += summary[
            "post_initialization_concrete_count"
        ]
        if (
            first_post_initialization_concrete is None
            and summary["first_post_initialization_concrete"] is not None
        ):
            first_post_initialization_concrete = {
                "seed": run["seed"],
                **summary["first_post_initialization_concrete"],
            }
    report = {
        "status": status,
        "seeds": list(INDEPENDENT_SIMULATION_SEEDS),
        "cycles_per_seed": INDEPENDENT_SIMULATION_CYCLES,
        "golden_sha256": sha256_text(golden),
        "candidate_sha256": sha256_text(candidate),
        "formal_protocol_sha256": formal_protocol_fingerprint(),
        "initialization_contract": pair.contract.to_dict(),
        "counterexample_summary": {
            "by_phase_and_sample_point": dict(sorted(aggregate_by_sample.items())),
            "post_initialization_concrete_count": aggregate_post_initialization_concrete,
            "first_post_initialization_concrete": first_post_initialization_concrete,
        },
        "runs": runs,
    }
    (artifact_dir / "independent_simulation.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def _append_jsonl(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _sha256_file(path: Path) -> str | None:
    if not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_json_sha256(record: dict[str, Any]) -> str:
    payload = json.dumps(
        record, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _portable_repo_path(path: Path) -> str:
    """Prefer a repository-relative provenance path over /work or host paths."""

    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def _source_from_validation_row(row: dict[str, Any], stem: str) -> str:
    direct = row.get(f"{stem}_rtl")
    if isinstance(direct, str) and direct.strip():
        return direct.rstrip() + "\n"
    path_value = row.get(f"{stem}_path")
    if path_value:
        return Path(str(path_value)).read_text(encoding="utf-8")
    raise ValueError(f"validation row needs {stem}_rtl or {stem}_path")


def run_validation_batch(
    jobs_path: Path,
    results_path: Path,
    artifact_root: Path,
    fast_timeout: int = 60,
    fallback_timeout: int = 300,
) -> dict[str, Any]:
    """Validate call-id keyed candidates and emit ledger-ingestible JSONL.

    Input rows contain ``call_id``, ``seed_id``, and either inline
    ``golden_rtl``/``candidate_rtl`` or corresponding ``*_path`` fields.  The
    output contract is accepted directly by ``ExperimentLedger.ingest_validation_jsonl``.
    """

    if shutil.which("iverilog") is None or shutil.which("vvp") is None:
        raise RuntimeError("validate-batch requires Icarus to establish compile/simulation verdicts")
    verifier = FormalVerifier(fast_timeout, fallback_timeout)
    protocol_sha = formal_protocol_fingerprint()
    completed: dict[str, dict[str, Any]] = {}
    if results_path.exists():
        for row in _read_jsonl_file(results_path):
            completed[str(row["call_id"])] = row
    written = 0
    for row in _read_jsonl_file(jobs_path):
        call_id = str(row.get("call_id") or "")
        seed_id = str(row.get("seed_id") or "")
        if not call_id or not seed_id:
            raise ValueError("validation row requires call_id and seed_id")
        golden = _source_from_validation_row(row, "golden")
        candidate = _source_from_validation_row(row, "candidate")
        golden_sha = sha256_text(golden)
        candidate_sha = sha256_text(candidate)
        if call_id in completed:
            if (
                completed[call_id].get("candidate_sha256") != candidate_sha
                or completed[call_id].get("golden_sha256") != golden_sha
                or completed[call_id].get("formal_protocol_sha256") != protocol_sha
            ):
                raise ValueError(
                    f"candidate/golden/formal-protocol drift for completed call_id {call_id}"
                )
            continue
        safe_call = re.sub(r"[^A-Za-z0-9_.-]+", "_", call_id)[:80]
        call_artifacts = artifact_root.resolve() / safe_call
        candidate_compiles = _source_compiles_with_icarus(
            candidate, call_artifacts / "compile"
        )
        candidate_verilator_compiles = _source_compiles_with_verilator(
            candidate, call_artifacts / "compile" / "verilator"
        )
        formal = verifier.verify(
            seed_id, golden, candidate, call_artifacts / "formal"
        )
        simulation = run_independent_simulation(
            seed_id, golden, candidate, call_artifacts / "simulation"
        )
        compile_passed = (
            candidate_compiles is True or candidate_verilator_compiles is True
        )
        output = {
            "call_id": call_id,
            "seed_id": seed_id,
            "golden_sha256": golden_sha,
            "candidate_sha256": candidate_sha,
            "formal_protocol_sha256": protocol_sha,
            "compile_verdict": {
                "passed": compile_passed,
                "engine": "iverilog-g2012+verilator-lint",
            },
            "formal_verdict": formal.to_dict(),
            "simulation_verdict": simulation,
        }
        _append_jsonl(results_path, output)
        completed[call_id] = output
        written += 1
        print(
            f"{call_id}: formal={formal.status.value} simulation={simulation.get('status')}",
            flush=True,
        )
    return {"jobs": len(completed), "newly_written": written, "results": str(results_path)}


def _read_jsonl_file(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row is not an object")
            rows.append(row)
    return rows


def _load_completed(path: Path) -> dict[str, dict[str, Any]]:
    completed: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return completed
    lines = path.read_text(encoding="utf-8").splitlines()
    for index, line in enumerate(lines, 1):
        if not line.strip():
            raise ValueError(f"{path}:{index}: blank calibration JSONL row")
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path}:{index}: incomplete or invalid calibration JSONL; "
                "use a fresh results path"
            ) from exc
        if not isinstance(row, dict) or not isinstance(row.get("run_id"), str):
            raise ValueError(f"{path}:{index}: malformed calibration result row")
        run_id = row["run_id"]
        if run_id in completed:
            raise ValueError(f"duplicate calibration run_id in {path}: {run_id}")
        completed[run_id] = row
    return completed


def _load_protocol_amendment() -> dict[str, Any]:
    """Load and validate the frozen pre-output amendment used by this gate."""

    try:
        manifest = json.loads(DEFAULT_PROTOCOL_AMENDMENT.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("frozen protocol amendment is missing or invalid") from exc
    if not isinstance(manifest, dict):
        raise RuntimeError("frozen protocol amendment is not an object")
    declared_sha = manifest.get("manifest_sha256")
    unsigned = dict(manifest)
    unsigned.pop("manifest_sha256", None)
    if declared_sha != _canonical_json_sha256(unsigned):
        raise RuntimeError("frozen protocol amendment canonical SHA mismatch")
    required_header = {
        "schema_version": 1,
        "kind": "vericodegen_pre_output_protocol_amendment",
        "status": "FROZEN",
        "decision_timing": "BEFORE_ANY_MODEL_OUTPUT",
        "model_outputs_seen_before_freeze": False,
    }
    for field, expected in required_header.items():
        if manifest.get(field) != expected:
            raise RuntimeError(f"frozen protocol amendment {field} mismatch")
    amendments = manifest.get("amendments")
    if not isinstance(amendments, list):
        raise RuntimeError("frozen protocol amendment entries are missing")
    entry = next(
        (
            amendment
            for amendment in amendments
            if isinstance(amendment, dict)
            and amendment.get("id") == "DAY3_EXACT_COMPOSITE_CALIBRATION_EXCEPTION"
        ),
        None,
    )
    if entry is None:
        raise RuntimeError("Day-3 composite amendment entry is missing")
    expected_success = {
        "golden_self_proofs": 85,
        "formal_replayable_mutant_counterexamples": 84,
        "independent_simulation_fallback_mutant_counterexamples": 1,
        "distinguishable_known_mutants": 85,
        "false_mutant_proofs": 0,
    }
    if entry.get("calibration_success_definition") != expected_success:
        raise RuntimeError("Day-3 composite success definition drifted")
    allowed = COMPOSITE_CALIBRATION_ALLOWLIST["repair_sem_0059"]
    expected_allowlist = [
        {
            "case_id": allowed["case_id"],
            "seed_id": allowed["seed_id"],
            "mutation": allowed["mutation"],
            "golden_sha256": allowed["golden_sha256"],
            "candidate_sha256": allowed["candidate_sha256"],
            "required_formal_status": allowed["required_formal_status"],
            "simulation_seeds": list(INDEPENDENT_SIMULATION_SEEDS),
            "cycles_per_seed": INDEPENDENT_SIMULATION_CYCLES,
            "required_status_per_seed": "COUNTEREXAMPLE",
            "required_counterexample_runs": len(INDEPENDENT_SIMULATION_SEEDS),
            "require_concrete_post_initialization_counterexample": True,
        }
    ]
    if entry.get("allowlist") != expected_allowlist:
        raise RuntimeError("Day-3 composite allowlist drifted")
    if (
        entry.get("forbid_any_other_fallback_case") is not True
        or entry.get("forbid_timeout_or_compile_fail_fallback") is not True
    ):
        raise RuntimeError("Day-3 composite fail-closed restrictions drifted")
    return manifest


def _composite_calibration_policy() -> dict[str, Any]:
    """Return the public, JSON-serializable Day-3 fallback contract."""

    amendment = _load_protocol_amendment()
    allowlist = []
    for case_id in sorted(COMPOSITE_CALIBRATION_ALLOWLIST):
        entry = dict(COMPOSITE_CALIBRATION_ALLOWLIST[case_id])
        entry.update(
            {
                "seeds": list(INDEPENDENT_SIMULATION_SEEDS),
                "cycles_per_seed": INDEPENDENT_SIMULATION_CYCLES,
                "required_seed_status": "COUNTEREXAMPLE",
                "requires_post_initialization_concrete_two_state": True,
            }
        )
        allowlist.append(entry)
    return {
        "name": COMPOSITE_CALIBRATION_POLICY_NAME,
        "version": COMPOSITE_CALIBRATION_POLICY_VERSION,
        "amendment_path": _portable_repo_path(DEFAULT_PROTOCOL_AMENDMENT),
        "protocol_amendment_sha256": _sha256_file(DEFAULT_PROTOCOL_AMENDMENT),
        "amendment_manifest_sha256": amendment["manifest_sha256"],
        "allowlist": allowlist,
    }


def _allowed_composite_identity(row: dict[str, Any]) -> dict[str, Any] | None:
    """Return the exact allowlist entry matched by a calibration result row."""

    case_id = row.get("case_id")
    allowed = COMPOSITE_CALIBRATION_ALLOWLIST.get(str(case_id))
    if allowed is None:
        return None
    result = row.get("result")
    if not isinstance(result, dict):
        return None
    actual = {
        "case_id": case_id,
        "seed_id": row.get("seed_id"),
        "mutation": row.get("mutation"),
        "golden_sha256": result.get("golden_sha256"),
        "candidate_sha256": result.get("candidate_sha256"),
        "required_formal_status": result.get("status"),
    }
    return allowed if actual == allowed else None


def _resolve_calibration_artifact_path(value: Any) -> Path | None:
    if not isinstance(value, str) or not value:
        return None
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def _concrete_post_initialization_witness(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    expected = value.get("expected")
    got = value.get("got")
    return (
        value.get("phase") == "post_initialization"
        and value.get("concrete_two_state") is True
        and isinstance(expected, str)
        and isinstance(got, str)
        and re.fullmatch(r"[01]+", expected) is not None
        and re.fullmatch(r"[01]+", got) is not None
        and expected != got
    )


def _validate_composite_fallback_row(
    row: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    """Re-open and validate the byte-bound independent-simulation sidecar.

    Nothing in ``composite_fallback`` is accepted on assertion alone.  The
    report is hashed as raw bytes, parsed again, and checked against the exact
    benchmark identity, formal-protocol fingerprint, and frozen 10x200 run
    contract.  The returned summary is safe to copy into the gate manifest.
    """

    errors: list[str] = []
    allowed = _allowed_composite_identity(row)
    if allowed is None:
        errors.append("row does not match the exact composite allowlist identity")
    pointer = row.get("composite_fallback")
    if not isinstance(pointer, dict):
        return ({"case_id": row.get("case_id"), "qualified": False}, errors + [
            "missing composite_fallback evidence pointer"
        ])
    if pointer.get("schema_version") != 1:
        errors.append("fallback evidence schema_version is not 1")
    if pointer.get("policy_name") != COMPOSITE_CALIBRATION_POLICY_NAME:
        errors.append("fallback evidence policy_name mismatch")
    if pointer.get("policy_version") != COMPOSITE_CALIBRATION_POLICY_VERSION:
        errors.append("fallback evidence policy_version mismatch")

    result = row.get("result") if isinstance(row.get("result"), dict) else {}
    protocol_sha = row.get("formal_protocol_sha256")
    amendment_sha = _sha256_file(DEFAULT_PROTOCOL_AMENDMENT)
    if row.get("protocol_amendment_sha256") != amendment_sha:
        errors.append("calibration row protocol amendment SHA mismatch")
    expected_pointer_fields = {
        "case_id": row.get("case_id"),
        "seed_id": row.get("seed_id"),
        "golden_sha256": result.get("golden_sha256"),
        "candidate_sha256": result.get("candidate_sha256"),
        "formal_protocol_sha256": protocol_sha,
        "protocol_amendment_sha256": amendment_sha,
    }
    for field, expected in expected_pointer_fields.items():
        if pointer.get(field) != expected:
            errors.append(f"fallback evidence {field} mismatch")

    report_path = _resolve_calibration_artifact_path(pointer.get("report_path"))
    report_sha = pointer.get("report_sha256")
    report: dict[str, Any] = {}
    if report_path is None or not report_path.is_file():
        errors.append("fallback simulation report is missing")
    else:
        actual_sha = _sha256_file(report_path)
        if not isinstance(report_sha, str) or actual_sha != report_sha:
            errors.append("fallback simulation report byte SHA mismatch")
        try:
            parsed = json.loads(report_path.read_text(encoding="utf-8"))
            if isinstance(parsed, dict):
                report = parsed
            else:
                errors.append("fallback simulation report is not an object")
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            errors.append("fallback simulation report is unreadable JSON")

    expected_report_fields = {
        "status": "COUNTEREXAMPLE",
        "seeds": list(INDEPENDENT_SIMULATION_SEEDS),
        "cycles_per_seed": INDEPENDENT_SIMULATION_CYCLES,
        "golden_sha256": result.get("golden_sha256"),
        "candidate_sha256": result.get("candidate_sha256"),
        "formal_protocol_sha256": protocol_sha,
        "protocol_amendment_sha256": amendment_sha,
    }
    for field, expected in expected_report_fields.items():
        if report.get(field) != expected:
            errors.append(f"fallback simulation report {field} mismatch")

    runs = report.get("runs")
    if not isinstance(runs, list):
        runs = []
        errors.append("fallback simulation report runs is not a list")
    expected_seeds = list(INDEPENDENT_SIMULATION_SEEDS)
    if [run.get("seed") for run in runs if isinstance(run, dict)] != expected_seeds:
        errors.append("fallback simulation report does not contain exactly the frozen seeds")
    invalid_runs = [
        run.get("seed") if isinstance(run, dict) else None
        for run in runs
        if not isinstance(run, dict)
        or run.get("cycles") != INDEPENDENT_SIMULATION_CYCLES
        or run.get("status") != "COUNTEREXAMPLE"
        or run.get("passed") is not False
    ]
    if invalid_runs:
        errors.append(
            "fallback simulation requires 10/10 COUNTEREXAMPLE runs at 200 cycles"
        )

    summary = report.get("counterexample_summary")
    if not isinstance(summary, dict):
        summary = {}
    concrete_count = summary.get("post_initialization_concrete_count")
    first_concrete = summary.get("first_post_initialization_concrete")
    if (
        not isinstance(concrete_count, int)
        or isinstance(concrete_count, bool)
        or concrete_count < 1
        or not _concrete_post_initialization_witness(first_concrete)
    ):
        errors.append(
            "fallback simulation lacks a concrete post-initialization 0/1 counterexample"
        )

    evidence = {
        "case_id": row.get("case_id"),
        "seed_id": row.get("seed_id"),
        "mutation": row.get("mutation"),
        "formal_status": result.get("status"),
        "golden_sha256": result.get("golden_sha256"),
        "candidate_sha256": result.get("candidate_sha256"),
        "formal_protocol_sha256": protocol_sha,
        "protocol_amendment_sha256": amendment_sha,
        "report_path": pointer.get("report_path"),
        "report_sha256": report_sha,
        "simulation_status": report.get("status"),
        "seeds": report.get("seeds"),
        "cycles_per_seed": report.get("cycles_per_seed"),
        "counterexample_seed_runs": sum(
            isinstance(run, dict) and run.get("status") == "COUNTEREXAMPLE"
            for run in runs
        ),
        "post_initialization_concrete_count": concrete_count,
        "first_post_initialization_concrete": first_concrete,
        "qualified": not errors,
        "validation_errors": errors,
    }
    return evidence, errors


def _fallback_report_path(artifact_root: Path, case_id: str) -> Path:
    return (
        artifact_root
        / case_id
        / "golden_vs_mutant"
        / "independent_simulation_fallback"
        / "independent_simulation.json"
    )


def _run_composite_fallback(
    row: dict[str, Any],
    golden: str,
    candidate: str,
    artifact_root: Path,
) -> dict[str, Any]:
    """Run the frozen simulator and return a raw-byte-bound evidence pointer."""

    case_id = str(row["case_id"])
    report_path = _fallback_report_path(artifact_root, case_id)
    report = run_independent_simulation(
        str(row["seed_id"]), golden, candidate, report_path.parent
    )
    # Add the amendment binding and serialize the exact object that the gate
    # will later re-open.  Persist early failures too so a failed gate remains
    # byte-auditable rather than silently losing why the fallback did not
    # qualify.
    report["protocol_amendment_sha256"] = _sha256_file(DEFAULT_PROTOCOL_AMENDMENT)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    result = row["result"]
    return {
        "schema_version": 1,
        "policy_name": COMPOSITE_CALIBRATION_POLICY_NAME,
        "policy_version": COMPOSITE_CALIBRATION_POLICY_VERSION,
        "case_id": row["case_id"],
        "seed_id": row["seed_id"],
        "golden_sha256": result["golden_sha256"],
        "candidate_sha256": result["candidate_sha256"],
        "formal_protocol_sha256": row["formal_protocol_sha256"],
        "protocol_amendment_sha256": row["protocol_amendment_sha256"],
        "report_path": _portable_repo_path(report_path),
        "report_sha256": _sha256_file(report_path),
    }


def _validate_or_rebuild_resumed_fallback(
    row: dict[str, Any],
    golden: str,
    candidate: str,
    artifact_root: Path,
) -> None:
    """Fail closed on a partial row; deterministically rebuild a stale sidecar."""

    pointer = row.get("composite_fallback")
    if not isinstance(pointer, dict):
        raise RuntimeError(
            "resumed allowlisted calibration row lacks composite fallback evidence; "
            "use a fresh results path"
        )
    expected_path = _fallback_report_path(artifact_root, str(row["case_id"]))
    if pointer.get("report_path") != _portable_repo_path(expected_path):
        raise RuntimeError(
            "resumed composite fallback report path does not match its calibration artifact"
        )
    _evidence, errors = _validate_composite_fallback_row(row)
    if not errors:
        return
    locked_sha = pointer.get("report_sha256")
    rebuilt_pointer = _run_composite_fallback(
        row, golden, candidate, artifact_root
    )
    if rebuilt_pointer.get("report_sha256") != locked_sha:
        raise RuntimeError(
            "resumed composite fallback sidecar cannot be reconstructed at its locked byte SHA"
        )
    _evidence, rebuilt_errors = _validate_composite_fallback_row(row)
    if rebuilt_errors:
        raise RuntimeError(
            "resumed composite fallback evidence remains invalid: "
            + "; ".join(rebuilt_errors)
        )


def calibration_gate(records: list[dict[str, Any]], expected_cases: int = 85) -> dict[str, Any]:
    gold = [row for row in records if row["comparison"] == "golden_vs_golden"]
    mutant = [row for row in records if row["comparison"] == "golden_vs_mutant"]
    gold_counts = collections.Counter(row["result"]["status"] for row in gold)
    mutant_counts = collections.Counter(row["result"]["status"] for row in mutant)
    false_proofs = [row["case_id"] for row in mutant if row["result"]["status"] == "PROVED"]
    replayed = sum(
        row["result"]["status"] == "COUNTEREXAMPLE"
        and row["result"].get("counterexample_replayed") is True
        for row in mutant
    )
    fallback_rows = [
        row for row in mutant if row["result"].get("status") == "UNSUPPORTED"
    ]
    fallback_evidence = []
    fallback_errors: list[dict[str, Any]] = []
    for row in fallback_rows:
        evidence, errors = _validate_composite_fallback_row(row)
        fallback_evidence.append(evidence)
        if errors:
            fallback_errors.append(
                {"case_id": row.get("case_id"), "errors": errors}
            )
    fallback_counterexamples = sum(
        evidence.get("qualified") is True for evidence in fallback_evidence
    )
    distinguishable = replayed + fallback_counterexamples
    gold_ids = [row.get("case_id") for row in gold]
    mutant_ids = [row.get("case_id") for row in mutant]
    run_ids = [row.get("run_id") for row in records]
    duplicate_run_ids = sorted(
        run_id
        for run_id, count in collections.Counter(run_ids).items()
        if isinstance(run_id, str) and count > 1
    )
    case_id_sets_match = (
        len(set(gold_ids)) == expected_cases
        and len(set(mutant_ids)) == expected_cases
        and set(gold_ids) == set(mutant_ids)
    )
    run_ids_unique = (
        len(run_ids) == expected_cases * 2
        and all(isinstance(run_id, str) and run_id for run_id in run_ids)
        and len(set(run_ids)) == len(run_ids)
    )
    amendment_sha = _sha256_file(DEFAULT_PROTOCOL_AMENDMENT)
    amendment_bound_rows = sum(
        row.get("protocol_amendment_sha256") == amendment_sha for row in records
    )
    protocol_values = {
        row.get("formal_protocol_sha256")
        for row in records
        if isinstance(row.get("formal_protocol_sha256"), str)
    }
    current_protocol_sha = formal_protocol_fingerprint()
    protocol_rows_consistent = (
        protocol_values == {current_protocol_sha}
        and all(
            row.get("formal_protocol_sha256") == current_protocol_sha for row in records
        )
    )
    expected_formal_counterexamples = expected_cases - 1
    passed = (
        expected_cases == 85
        and len(gold) == expected_cases
        and len(mutant) == expected_cases
        and len(records) == expected_cases * 2
        and case_id_sets_match
        and run_ids_unique
        and amendment_bound_rows == expected_cases * 2
        and protocol_rows_consistent
        and gold_counts["PROVED"] == expected_cases
        and set(gold_counts) == {"PROVED"}
        and replayed == expected_formal_counterexamples
        and mutant_counts == collections.Counter(
            {"COUNTEREXAMPLE": expected_formal_counterexamples, "UNSUPPORTED": 1}
        )
        and len(fallback_rows) == 1
        and fallback_counterexamples == 1
        and fallback_rows[0].get("case_id") == "repair_sem_0059"
        and distinguishable == expected_cases
        and not fallback_errors
        and not false_proofs
    )
    return {
        "schema_version": 3,
        "gate": "day3_harness_calibration",
        "expected_cases": expected_cases,
        "golden_vs_golden_completed": len(gold),
        "golden_vs_golden_statuses": dict(gold_counts),
        "golden_vs_mutant_completed": len(mutant),
        "golden_vs_mutant_statuses": dict(mutant_counts),
        "replayable_mutant_counterexamples": replayed,
        "fallback_mutant_counterexamples": fallback_counterexamples,
        "distinguishable_mutants": distinguishable,
        "composite_policy": _composite_calibration_policy(),
        "fallback_mutant_evidence": fallback_evidence,
        "fallback_validation_errors": fallback_errors,
        "duplicate_run_ids": duplicate_run_ids,
        "unique_run_ids": run_ids_unique,
        "comparison_case_id_sets_match": case_id_sets_match,
        "amendment_bound_rows": amendment_bound_rows,
        "formal_protocol_rows_consistent": protocol_rows_consistent,
        "mutants_incorrectly_proved": false_proofs,
        "passed": passed,
    }


def formal_primary_gate(
    records: list[dict[str, Any]],
    expected_cases_per_arm: int = 85,
    minimum_definitive: int = 77,
    required_arms: Iterable[str] = MAIN_2X2_ARMS,
) -> dict[str, Any]:
    """Evaluate the Week-1 coverage and formal/simulation consistency gate.

    Each record must contain ``arm`` and a formal result under ``formal_result``
    (or ``formal``), plus an independent simulation report under
    ``simulation_result`` (or ``simulation``).  The accepted simulation statuses
    are ``PASS`` (equivalent in the frozen tests) and ``COUNTEREXAMPLE``.
    """

    required = tuple(required_arms)
    required_set = set(required)
    by_arm: dict[str, list[dict[str, Any]]] = collections.defaultdict(list)
    malformed: list[int] = []
    for index, row in enumerate(records):
        mode = row.get("mode")
        if (mode is not None and mode != "main") or row.get("nonresearch") is True:
            continue
        arm = row.get("arm")
        # A combined results file may also contain preflight, redacted,
        # feedback, and distribution-shift arms; they are outside this gate.
        if arm not in required_set:
            continue
        formal = row.get(
            "formal_result", row.get("formal", row.get("formal_verdict"))
        )
        simulation = row.get(
            "simulation_result", row.get("simulation", row.get("simulation_verdict"))
        )
        if not isinstance(formal, dict) or not isinstance(simulation, dict):
            malformed.append(index)
            continue
        by_arm[str(arm)].append(
            {"row": row, "formal": formal, "simulation": simulation, "index": index}
        )

    arm_reports: dict[str, Any] = {}
    all_conflicts: list[dict[str, Any]] = []
    all_inconclusive_simulation: list[dict[str, Any]] = []
    reference_roster: set[str] | None = None
    roster_mismatches: dict[str, dict[str, list[str]]] = {}
    for arm in required:
        valid_ids = {
            item["row"].get("case_id")
            for item in by_arm[arm]
            if isinstance(item["row"].get("case_id"), str)
            and item["row"].get("case_id").strip()
        }
        if reference_roster is None:
            reference_roster = valid_ids
        elif valid_ids != reference_roster:
            roster_mismatches[arm] = {
                "missing_from_reference_arm": sorted(reference_roster - valid_ids),
                "extra_vs_reference_arm": sorted(valid_ids - reference_roster),
            }

    for arm in required:
        rows = by_arm[arm]
        case_ids = [item["row"].get("case_id") for item in rows]
        invalid_case_id_indexes = [
            item["index"]
            for item in rows
            if not isinstance(item["row"].get("case_id"), str)
            or not item["row"].get("case_id").strip()
        ]
        valid_case_ids = [
            case_id
            for case_id in case_ids
            if isinstance(case_id, str) and case_id.strip()
        ]
        case_id_counts = collections.Counter(valid_case_ids)
        duplicate_case_ids = sorted(
            case_id for case_id, count in case_id_counts.items() if count > 1
        )
        definitive = 0
        conflicts = []
        inconclusive_simulation = []
        for item in rows:
            formal_status = item["formal"].get("status")
            simulation_status = item["simulation"].get("status")
            replay_ok = item["formal"].get("counterexample_replayed") is True
            is_definitive = formal_status == "PROVED" or (
                formal_status == "COUNTEREXAMPLE" and replay_ok
            )
            definitive += int(is_definitive)
            # A fixed-random PASS does not contradict a concrete formal
            # witness; it may simply not sample that input trace.  The soundness
            # contradiction is a proof of equivalence plus a simulation trace
            # that demonstrates divergence.
            conflict = formal_status == "PROVED" and simulation_status == "COUNTEREXAMPLE"
            if conflict:
                conflict_row = {
                    "arm": arm,
                    "case_id": item["row"].get("case_id"),
                    "formal_status": formal_status,
                    "simulation_status": simulation_status,
                }
                conflicts.append(conflict_row)
                all_conflicts.append(conflict_row)
            if simulation_status not in {"PASS", "COUNTEREXAMPLE"}:
                inconclusive_row = {
                    "arm": arm,
                    "case_id": item["row"].get("case_id"),
                    "simulation_status": simulation_status,
                }
                inconclusive_simulation.append(inconclusive_row)
                all_inconclusive_simulation.append(inconclusive_row)
        arm_passed = (
            len(rows) == expected_cases_per_arm
            and len(valid_case_ids) == expected_cases_per_arm
            and len(case_id_counts) == expected_cases_per_arm
            and not invalid_case_id_indexes
            and not duplicate_case_ids
            and arm not in roster_mismatches
            and definitive >= minimum_definitive
            and not conflicts
            and not inconclusive_simulation
        )
        arm_reports[arm] = {
            "records": len(rows),
            "unique_nonempty_case_ids": len(case_id_counts),
            "invalid_case_id_record_indexes": invalid_case_id_indexes,
            "duplicate_case_ids": duplicate_case_ids,
            "roster_matches_reference_arm": arm not in roster_mismatches,
            "definitive_formal_results": definitive,
            "minimum_definitive": minimum_definitive,
            "conflicts": conflicts,
            "inconclusive_simulation": inconclusive_simulation,
            "passed": arm_passed,
        }

    case_id_sets_match = not roster_mismatches
    passed = bool(arm_reports) and not malformed and case_id_sets_match and all(
        report["passed"] for report in arm_reports.values()
    )
    return {
        "schema_version": 1,
        "gate": "formal_primary_coverage",
        "expected_cases_per_arm": expected_cases_per_arm,
        "minimum_definitive_per_arm": minimum_definitive,
        "required_arms": list(required),
        "arms": arm_reports,
        "malformed_record_indexes": malformed,
        "arm_case_id_sets_match": case_id_sets_match,
        "case_id_roster_mismatches": roster_mismatches,
        "conflicts": all_conflicts,
        "inconclusive_simulation": all_inconclusive_simulation,
        "passed": passed,
        "reporting_mode": (
            "formal_primary"
            if passed
            else "formal_where_supported_plus_independent_simulation_fallback"
        ),
    }


def _merge_event_ledger(
    raw_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Join response/verdict events without silently overwriting duplicates."""

    responses: dict[str, dict[str, Any]] = {}
    verdicts: dict[str, dict[str, Any]] = {}
    errors: list[str] = []
    for event_name, destination in (("response", responses), ("verdict", verdicts)):
        counts: collections.Counter[str] = collections.Counter(
            str(row.get("call_id"))
            for row in raw_records
            if row.get("event") == event_name and row.get("call_id")
        )
        duplicates = sorted(call_id for call_id, count in counts.items() if count > 1)
        if duplicates:
            errors.append(f"duplicate {event_name} event(s) for call_id: {duplicates}")
        for row in raw_records:
            if row.get("event") == event_name and row.get("call_id"):
                destination.setdefault(str(row["call_id"]), row)
    records = [
        {**response, **verdicts.get(call_id, {})}
        for call_id, response in responses.items()
    ]
    return records, errors


def run_calibration(
    benchmark: Path,
    artifact_root: Path,
    results_path: Path,
    gate_path: Path,
    fast_timeout: int,
    fallback_timeout: int,
    case_ids: set[str] | None = None,
) -> dict[str, Any]:
    cases = load_semantic_cases(benchmark)
    if case_ids:
        available = {case.case_id for case in cases}
        missing = sorted(case_ids - available)
        if missing:
            raise ValueError(f"unknown calibration case id(s): {missing}")
        cases = [case for case in cases if case.case_id in case_ids]
    toolchain_manifest_path = artifact_root / "toolchain_manifest.json"
    write_toolchain_manifest(toolchain_manifest_path)
    verifier = FormalVerifier(fast_timeout, fallback_timeout)
    protocol_sha = formal_protocol_fingerprint()
    amendment_sha = _sha256_file(DEFAULT_PROTOCOL_AMENDMENT)
    amendment_manifest_sha = _load_protocol_amendment()["manifest_sha256"]
    completed = _load_completed(results_path)
    expected_run_ids = {
        (
            f"{case.case_id}:{comparison}:{sha256_text(case.canonical_rtl)}:"
            f"{sha256_text(candidate)}:{protocol_sha}"
        )
        for case in cases
        for comparison, candidate in (
            ("golden_vs_golden", case.canonical_rtl),
            ("golden_vs_mutant", case.broken_rtl),
        )
    }
    stale_run_ids = sorted(set(completed) - expected_run_ids)
    if stale_run_ids:
        raise ValueError(
            "calibration results contain another scope or protocol; use a fresh "
            f"results path (first stale run_id: {stale_run_ids[0]})"
        )
    current_records: list[dict[str, Any]] = []
    halted_reason = ""
    for case in cases:
        comparisons = (
            ("golden_vs_golden", case.canonical_rtl),
            ("golden_vs_mutant", case.broken_rtl),
        )
        for comparison, candidate in comparisons:
            run_id = (
                f"{case.case_id}:{comparison}:{sha256_text(case.canonical_rtl)}:"
                f"{sha256_text(candidate)}:{protocol_sha}"
            )
            if run_id in completed:
                row = completed[run_id]
                result = FormalResult.from_dict(row["result"])
                if _allowed_composite_identity(row) is not None:
                    _validate_or_rebuild_resumed_fallback(
                        row,
                        case.canonical_rtl,
                        candidate,
                        artifact_root,
                    )
            else:
                result = verifier.verify(
                    case.seed_id,
                    case.canonical_rtl,
                    candidate,
                    artifact_root / case.case_id / comparison,
                )
                row = {
                    "run_id": run_id,
                    "case_id": case.case_id,
                    "seed_id": case.seed_id,
                    "mutation": case.mutation,
                    "comparison": comparison,
                    "formal_protocol_sha256": protocol_sha,
                    "protocol_amendment_sha256": amendment_sha,
                    "result": result.to_dict(),
                }
                if _allowed_composite_identity(row) is not None:
                    # Keep FormalResult=UNSUPPORTED and attach independent
                    # evidence only after the frozen simulation is complete.
                    # The comparison row is appended exactly once.
                    row["composite_fallback"] = _run_composite_fallback(
                        row,
                        case.canonical_rtl,
                        candidate,
                        artifact_root,
                    )
                _append_jsonl(results_path, row)
                completed[run_id] = row
            current_records.append(completed[run_id])
            print(f"{case.case_id} {comparison}: {result.status.value}", flush=True)
            if comparison == "golden_vs_mutant" and result.status == FormalStatus.PROVED:
                halted_reason = (
                    f"HALT: known mutant {case.case_id} was proved equivalent; inspect harness"
                )
                print(halted_reason, file=sys.stderr, flush=True)
                break
        if halted_reason:
            break
    gate = calibration_gate(current_records, expected_cases=len(cases))
    gate["schema_version"] = 3
    gate["halted_reason"] = halted_reason
    gate["benchmark_source"] = _portable_repo_path(benchmark)
    gate["benchmark_source_sha256"] = _sha256_file(benchmark)
    gate["canonical_manifest"] = _portable_repo_path(DEFAULT_MANIFEST)
    gate["canonical_manifest_sha256"] = _sha256_file(DEFAULT_MANIFEST)
    gate["protocol_manifest"] = _portable_repo_path(DEFAULT_PROTOCOL_MANIFEST)
    gate["protocol_manifest_sha256"] = _sha256_file(DEFAULT_PROTOCOL_MANIFEST)
    gate["results_path"] = _portable_repo_path(results_path)
    gate["calibration_results_sha256"] = _sha256_file(results_path)
    gate["toolchain_manifest"] = _portable_repo_path(toolchain_manifest_path)
    gate["toolchain_manifest_sha256"] = _sha256_file(toolchain_manifest_path)
    gate["formal_protocol_sha256"] = protocol_sha
    gate["amendment_path"] = _portable_repo_path(DEFAULT_PROTOCOL_AMENDMENT)
    gate["protocol_amendment_sha256"] = amendment_sha
    gate["amendment_manifest_sha256"] = amendment_manifest_sha
    gate["scope"] = "subset_smoke" if case_ids else "full_sem85"
    gate["scope_case_ids"] = sorted(case_ids) if case_ids else None
    gate["full_day3_gate_passed"] = gate["passed"] and not case_ids and len(cases) == 85
    # Self-authenticate the semantic gate object.  This is intentionally a
    # canonical compact JSON digest excluding the digest field itself, not the
    # hash of the pretty-printed file bytes.
    gate["gate_manifest_sha256"] = _canonical_json_sha256(gate)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return gate


def _build_data(args: argparse.Namespace) -> int:
    manifest = write_canonical_dataset(args.benchmark, args.output_dir, args.manifest)
    cases = load_semantic_cases(args.benchmark)
    by_seed = assert_seed_consistency(cases)
    protocol = write_protocol_manifest(
        [(seed, case.canonical_rtl) for seed, case in by_seed.items()], args.protocol_manifest
    )
    case_contract_counts = collections.Counter(
        initialization_contract(case.seed_id, case.canonical_rtl).kind for case in cases
    )
    reset_style_counts = collections.Counter(
        "synchronous"
        if initialization_contract(case.seed_id, case.canonical_rtl).reset_synchronous
        else "asynchronous"
        for case in cases
        if initialization_contract(case.seed_id, case.canonical_rtl).kind == "reset"
    )
    protocol["case_audit"] = {
        "case_count": len(cases),
        "contract_counts": dict(case_contract_counts),
        "reset_style_counts": dict(reset_style_counts),
    }
    args.protocol_manifest.write_text(
        json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    summary = {
        "canonical_manifest": str(args.manifest),
        "protocol_manifest": str(args.protocol_manifest),
        "cases": manifest["case_count"],
        "designs": manifest["design_count"],
        "case_contracts": dict(case_contract_counts),
        "protocol_designs": len(protocol["designs"]),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def _verify_one(args: argparse.Namespace) -> int:
    golden = args.golden.read_text(encoding="utf-8")
    candidate = args.candidate.read_text(encoding="utf-8")
    result = FormalVerifier(args.fast_timeout, args.fallback_timeout).verify(
        args.seed_id, golden, candidate, args.artifacts
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result.to_dict(), indent=2, sort_keys=True) + "\n")
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.status in {FormalStatus.PROVED, FormalStatus.COUNTEREXAMPLE} else 2


def _simulate_one(args: argparse.Namespace) -> int:
    report = run_independent_simulation(
        args.seed_id,
        args.golden.read_text(encoding="utf-8"),
        args.candidate.read_text(encoding="utf-8"),
        args.artifacts,
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] in {"PASS", "COUNTEREXAMPLE"} else 2


def _formal_gate(args: argparse.Namespace) -> int:
    raw_records = []
    with args.records.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                raw_records.append(json.loads(line))
    if any(row.get("event") for row in raw_records):
        records, event_ledger_errors = _merge_event_ledger(raw_records)
    else:
        records = raw_records
        event_ledger_errors = []
    gate = formal_primary_gate(
        records,
        args.expected_cases,
        args.minimum_definitive,
        args.required_arm or MAIN_2X2_ARMS,
    )
    gate["event_ledger_errors"] = event_ledger_errors
    if event_ledger_errors:
        gate["passed"] = False
        gate["reporting_mode"] = (
            "formal_where_supported_plus_independent_simulation_fallback"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(gate, indent=2, sort_keys=True))
    return 0 if gate["passed"] else 3


def _toolchain_info(args: argparse.Namespace) -> int:
    manifest = write_toolchain_manifest(args.output)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0 if all(tool["available"] for tool in manifest["tools"].values()) else 2


def _validate_batch(args: argparse.Namespace) -> int:
    summary = run_validation_batch(
        args.jobs,
        args.results,
        args.artifacts,
        args.fast_timeout,
        args.fallback_timeout,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    build = subparsers.add_parser("build-data", help="reconstruct canonical goldens and manifests")
    build.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    build.add_argument(
        "--output-dir", type=Path,
        default=DEFAULT_ARTIFACTS / "canonical_goldens",
    )
    build.add_argument(
        "--manifest", type=Path,
        default=DEFAULT_ARTIFACTS / "sem85_manifest.json",
    )
    build.add_argument(
        "--protocol-manifest", type=Path,
        default=DEFAULT_ARTIFACTS / "protocol_manifest.json",
    )
    build.set_defaults(handler=_build_data)

    verify = subparsers.add_parser("verify", help="verify one golden/candidate pair")
    verify.add_argument("--seed-id", required=True)
    verify.add_argument("--golden", type=Path, required=True)
    verify.add_argument("--candidate", type=Path, required=True)
    verify.add_argument("--artifacts", type=Path, required=True)
    verify.add_argument("--result", type=Path, required=True)
    verify.add_argument("--fast-timeout", type=int, default=60)
    verify.add_argument("--fallback-timeout", type=int, default=300)
    verify.set_defaults(handler=_verify_one)

    simulate = subparsers.add_parser(
        "simulate", help="run frozen seeds 1001..1010, 200 cycles each"
    )
    simulate.add_argument("--seed-id", required=True)
    simulate.add_argument("--golden", type=Path, required=True)
    simulate.add_argument("--candidate", type=Path, required=True)
    simulate.add_argument("--artifacts", type=Path, required=True)
    simulate.set_defaults(handler=_simulate_one)

    calibration = subparsers.add_parser(
        "calibrate", help="run 85 golden/golden and 85 golden/known-mutant checks"
    )
    calibration.add_argument("--benchmark", type=Path, default=DEFAULT_BENCHMARK)
    calibration.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACTS / "calibration")
    calibration.add_argument(
        "--results", type=Path, default=DEFAULT_ARTIFACTS / "calibration_results.jsonl"
    )
    calibration.add_argument(
        "--gate", type=Path, default=DEFAULT_ARTIFACTS / "calibration_gate.json"
    )
    calibration.add_argument("--fast-timeout", type=int, default=60)
    calibration.add_argument("--fallback-timeout", type=int, default=300)
    calibration.add_argument(
        "--case-id", action="append", help="run only this case (repeatable smoke-test filter)"
    )
    calibration.set_defaults(handler=None)

    coverage = subparsers.add_parser(
        "formal-gate", help="check >=77/85 definitive coverage and zero simulation conflicts"
    )
    coverage.add_argument("--records", type=Path, required=True)
    coverage.add_argument("--output", type=Path, required=True)
    coverage.add_argument("--expected-cases", type=int, default=85)
    coverage.add_argument("--minimum-definitive", type=int, default=77)
    coverage.add_argument(
        "--required-arm",
        action="append",
        help="arm to gate (repeatable; defaults to the four main spec/location arms)",
    )
    coverage.set_defaults(handler=_formal_gate)

    toolchain = subparsers.add_parser(
        "toolchain-info", help="record container image ID and actual tool versions"
    )
    toolchain.add_argument(
        "--output", type=Path, default=DEFAULT_ARTIFACTS / "toolchain_manifest.json"
    )
    toolchain.set_defaults(handler=_toolchain_info)

    batch = subparsers.add_parser(
        "validate-batch",
        help="formal+10x200 simulation validation for call-id keyed candidate JSONL",
    )
    batch.add_argument("--jobs", type=Path, required=True)
    batch.add_argument("--results", type=Path, required=True)
    batch.add_argument("--artifacts", type=Path, required=True)
    batch.add_argument("--fast-timeout", type=int, default=60)
    batch.add_argument("--fallback-timeout", type=int, default=300)
    batch.set_defaults(handler=_validate_batch)

    args = parser.parse_args(argv)
    if args.command == "calibrate":
        gate = run_calibration(
            args.benchmark,
            args.artifacts,
            args.results,
            args.gate,
            args.fast_timeout,
            args.fallback_timeout,
            set(args.case_id) if args.case_id else None,
        )
        print(json.dumps(gate, indent=2, sort_keys=True))
        return 0 if gate["passed"] else 3
    return args.handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
