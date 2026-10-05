"""Mine a REAL-bug semantic repair set from the model's own VerilogEval failures.

A "real-bug semantic case" is a model generation that COMPILES (iverilog) and
LINTS CLEAN (verilator) but DIVERGES from the reference under the task's
testbench -- i.e. a genuine compiles-but-wrong functional bug produced by a real
model, not a synthetic one-line mutation. Each case ships with the exact
divergence trace parsed from the simulator (which output, how many mismatches,
first failing time), the golden reference, the NL spec, and a light taxonomy.

This is leak-free by construction: the generations are named `module TopModule`
(the VerilogEval DUT name), so they carry no Prob-id identity.

The ordinary CLI writes a new file under ``generated/realbugs`` and refuses
partial or roster-drifted historical reconstructions.  The tracked benchmark
is never overwritten by this command.
"""
from __future__ import annotations

import argparse
import collections
import glob
import hashlib
import json
import os
import re
import sys
import tempfile
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
CODE = os.path.abspath(os.path.join(HERE, ".."))  # standalone code/
sys.path.insert(0, CODE)  # for benchmarks.bench_common
sys.path.insert(0, HERE)  # for sibling datagen modules

from benchmarks.bench_common import extract_rtl  # noqa: E402
if __package__:
    from project_paths import project_root
    from .run_verilator_for_repairs import lint_source
else:  # compatibility: direct script execution
    from project_paths import project_root
    from run_verilator_for_repairs import lint_source

REPO_ROOT = project_root()

# Inputs/outputs follow the workshop harness convention: $RTLREPAIR_DATA for
# the data dir, $RTLREPAIR_SEEDS / $RTLREPAIR_EVAL_TASKS for the two specific
# inputs, defaulting to the parent repo's data/ and generated/ trees.
DATA = os.environ.get("RTLREPAIR_DATA") or str(REPO_ROOT / "data")
RESULTS = str(REPO_ROOT / "generated" / "benchmark_results")
SEEDS = os.environ.get("RTLREPAIR_SEEDS", os.path.join(DATA, "processed", "verilogeval_seeds"))
EVAL_TASKS = os.environ.get("RTLREPAIR_EVAL_TASKS", os.path.join(DATA, "eval_tasks.jsonl"))
FROZEN_OUTPUT = REPO_ROOT / "data" / "repairbench_realbugs.jsonl"
DEFAULT_OUTPUT = REPO_ROOT / "generated" / "realbugs" / "repairbench_realbugs.rebuilt.jsonl"
CAP_PER_TASK = 3  # keep at most N distinct real bugs per task for diversity

_TRACE_OUT = re.compile(r"Output '([^']+)' has (\d+) mismatches. First mismatch occurred at time (\d+)")
_TRACE_TOT = re.compile(r"Mismatches:\s*(\d+)\s*in\s*(\d+)\s*samples")


def parse_trace(log: str) -> dict:
    t: dict = {}
    m = _TRACE_TOT.search(log or "")
    if m:
        t["mismatches"], t["samples"] = int(m.group(1)), int(m.group(2))
    h = _TRACE_OUT.search(log or "")
    if h:
        t["output"], t["first_mismatch_time"] = h.group(1), int(h.group(3))
    return t


def taxonomy(rtl: str) -> dict:
    body = [l for l in rtl.splitlines() if l.strip() and not l.strip().startswith("//")]
    return {
        "code_lines": len(body),
        "sequential": bool(re.search(r"always\s*@\s*\(\s*posedge", rtl)),
        "has_case": "case" in rtl,
        "has_state": "state" in rtl.lower(),
    }


def norm(rtl: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"//[^\n]*", "", rtl)).strip()


def load_specs(eval_tasks: Path) -> dict:
    specs: dict[str, str] = {}
    for line in eval_tasks.open(encoding="utf-8"):
        d = json.loads(line)
        instr = (d.get("instruction") or "").strip()
        if instr:
            specs[str(d.get("id", "")).split(":")[-1]] = instr
    return specs


def load_goldens(seeds: Path) -> dict:
    g: dict[str, str] = {}
    for p in glob.glob(os.path.join(seeds, "*_ref.sv")):
        stem = os.path.basename(p)[:-3]          # Prob120_fsm3s_ref
        g[stem[:-4] if stem.endswith("_ref") else stem] = p  # key Prob120_fsm3s
    return g


def load_expected_ids(path: Path) -> set[str]:
    if not path.is_file():
        raise FileNotFoundError(f"expected frozen roster is missing: {path}")
    ids: list[str] = []
    with path.open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            case_id = row.get("id") if isinstance(row, dict) else None
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"{path}:{number}: missing case id")
            ids.append(case_id)
    if not ids or len(ids) != len(set(ids)):
        raise ValueError(f"{path}: expected roster is empty or contains duplicate ids")
    return set(ids)


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", newline="", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
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
    parser.add_argument("--results", type=Path, default=Path(RESULTS))
    parser.add_argument("--seeds", type=Path, default=Path(SEEDS))
    parser.add_argument("--eval-tasks", type=Path, default=Path(EVAL_TASKS))
    parser.add_argument("--expected-roster", type=Path, default=FROZEN_OUTPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--overwrite", action="store_true",
        help="replace an existing generated output; never permits overwriting the frozen dataset",
    )
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    results = args.results.resolve()
    seeds = args.seeds.resolve()
    eval_tasks = args.eval_tasks.resolve()
    expected_roster = args.expected_roster.resolve()
    output = args.output.resolve()
    if output == FROZEN_OUTPUT.resolve():
        raise ValueError("refusing to overwrite the frozen real-bug benchmark")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"output exists; pass --overwrite for a generated file: {output}")
    if not results.is_dir():
        raise FileNotFoundError(f"banked benchmark results are missing: {results}")
    if not seeds.is_dir():
        raise FileNotFoundError(f"golden seed directory is missing: {seeds}")
    if not eval_tasks.is_file():
        raise FileNotFoundError(f"evaluation task file is missing: {eval_tasks}")
    expected_ids = load_expected_ids(expected_roster)
    specs, goldens = load_specs(eval_tasks), load_goldens(seeds)
    if not specs or not goldens:
        raise RuntimeError("evaluation specifications or golden seeds are empty")
    seen: set = set()
    per_task: collections.Counter = collections.Counter()
    cases = []
    skipped_lint = skipped_extract = skipped_nogolden = 0

    result_files = [
        path
        for model in ("base", "tuned")
        for path in sorted(
            glob.glob(os.path.join(results, model, "verilogeval:*", "sample_*", "eval.json"))
        )
    ]
    if not result_files:
        raise RuntimeError(f"no banked eval.json files found under {results}")
    for ev in result_files:
        model = Path(ev).relative_to(results).parts[0]
        try:
            with Path(ev).open(encoding="utf-8") as handle:
                d = json.load(handle)
        except (OSError, json.JSONDecodeError):
            continue
        if not (d.get("compile_pass") and not d.get("test_pass")):
            continue
        task = ev.split("verilogeval:")[1].split(os.sep)[0]
        raw_path = Path(ev).with_name("raw.txt")
        if not raw_path.is_file():
            continue
        rtl = extract_rtl(raw_path.read_text(encoding="utf-8"))
        if not rtl or "module" not in rtl:
            skipped_extract += 1
            continue
        rtl = rtl.rstrip() + "\n"
        # semantic-class requires lint-clean (else it is a lint bug, not behavioral)
        if lint_source("TopModule.sv", rtl).failed:
            skipped_lint += 1
            continue
        key = (task, hashlib.md5(norm(rtl).encode()).hexdigest())
        if key in seen:
            continue
        if per_task[task] >= CAP_PER_TASK:
            continue
        gpath = goldens.get(task)
        if not gpath:
            skipped_nogolden += 1
            continue
        seen.add(key)
        per_task[task] += 1
        cases.append({
            "id": f"realbug:{task}:{model}:{per_task[task]}",
            "task": task,
            "source_model": model,
            "bucket": "semantic-real",
            "broken_rtl": rtl,
            "divergence_trace": parse_trace(d.get("log", "")),
            "golden_rtl": Path(gpath).read_text(encoding="utf-8"),
            "spec": specs.get(task),
            "taxonomy": taxonomy(rtl),
        })

    actual_ids = [case["id"] for case in cases]
    if len(actual_ids) != len(set(actual_ids)):
        raise RuntimeError("reconstructed real-bug ids are duplicated")
    actual_set = set(actual_ids)
    if actual_set != expected_ids:
        missing = sorted(expected_ids - actual_set)
        extra = sorted(actual_set - expected_ids)
        raise RuntimeError(
            "reconstructed roster differs from the frozen 274-case roster: "
            f"missing={len(missing)}, extra={len(extra)}"
        )
    if any(not case.get("spec") for case in cases):
        raise RuntimeError("reconstructed roster contains a case without a specification")
    if any(not case.get("divergence_trace", {}).get("mismatches") for case in cases):
        raise RuntimeError("reconstructed roster contains a case without a divergence trace")
    write_jsonl_atomic(output, cases)

    # ---- summary ----
    by_model = collections.Counter(c["source_model"] for c in cases)
    seq = sum(c["taxonomy"]["sequential"] for c in cases)
    with_trace = sum(1 for c in cases if c["divergence_trace"].get("mismatches"))
    with_spec = sum(1 for c in cases if c["spec"])
    yld = collections.Counter(per_task.values())
    print(f"REAL-BUG SEMANTIC REPAIR SET -> {output}")
    print(f"  cases: {len(cases)}  | distinct tasks: {len(per_task)}")
    print(f"  by source model: {dict(by_model)}")
    print(f"  sequential: {seq}  combinational: {len(cases)-seq}")
    print(f"  with divergence trace: {with_trace}  with spec: {with_spec}")
    print(f"  cases-per-task histogram (k cases -> n tasks): {dict(sorted(yld.items()))}")
    print(f"  skipped: lint-dirty={skipped_lint} no-extract={skipped_extract} no-golden={skipped_nogolden}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
