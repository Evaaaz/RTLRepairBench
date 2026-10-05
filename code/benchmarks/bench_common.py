"""bench_common -- shared helpers for the RTLRepair benchmark drivers.

Loads benchmark tasks, extracts RTL code from raw LLM completions, calls the
model via ``backend.llm_client.generate`` (which falls back to ``serve_stub``
offline), scores with ``benchmarks.evaluate_outputs.evaluate_outputs``, and
writes raw outputs + logs under ``generated/benchmark_results/<run>/``.

Everything degrades gracefully: missing task file -> built-in fixture task;
missing model server -> serve_stub fallback (results flagged PLACEHOLDER);
missing iverilog/verilator -> bracket-heuristic compile check.
"""
from __future__ import annotations

import json
import os
import re
import sys
from typing import Dict, List, Optional

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_THIS_DIR)
for _p in (_THIS_DIR, _REPO_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from evaluate_outputs import evaluate_outputs  # noqa: E402

# Standalone-root resolution: this module lives at code/benchmarks/, so the
# release root (which owns data/ and generated/) is exactly two levels up.
_FC_ROOT = os.environ.get("RTLREPAIR_ROOT") or os.path.abspath(
    os.path.join(_THIS_DIR, "..", "..")
)
RESULTS_DIR = os.path.join(_FC_ROOT, "generated", "benchmark_results")
# Runnable, self-validated eval set released at top-level data/.
# NB: data/benchmark_tasks.jsonl is the *decontamination* set (id/source/code only,
# no testbenches) and must NOT be used here -- it scores compile-only.
DEFAULT_TASKS = os.path.join(_FC_ROOT, "data", "eval_tasks.jsonl")
FIXTURE_TASKS = os.path.join(_FC_ROOT, "generated", "fixtures", "benchmark_tasks.jsonl")

# Markdown / SystemVerilog code fences the model may wrap RTL in.
_FENCE_RE = re.compile(
    r"```(?:systemverilog|verilog|sv|v)?\s*\n(.*?)```",
    re.IGNORECASE | re.DOTALL,
)
_MODULE_RE = re.compile(r"(\bmodule\b.*?\bendmodule\b)", re.DOTALL)


def extract_rtl(text: str, top_module: Optional[str] = None) -> str:
    """Pull synthesizable RTL out of a raw model completion.

    Prefers a fenced code block; else the first module..endmodule span; else the
    raw text. If ``top_module`` is given, prefers the module block that declares
    that name. Never raises -- returns the original text as a last resort.
    """
    if not text:
        return ""
    candidates: List[str] = []
    for m in _FENCE_RE.finditer(text):
        candidates.append(m.group(1).strip())
    body = "\n".join(candidates) if candidates else text
    blocks = _MODULE_RE.findall(body)
    if not blocks:
        # Maybe the fence stripped the module markers; try the whole text.
        blocks = _MODULE_RE.findall(text)
    if not blocks:
        return body.strip()
    if top_module:
        for b in blocks:
            if re.search(r"\bmodule\s+" + re.escape(top_module) + r"\b", b):
                return b.strip()
    # Join all module blocks (a design may have helper modules) preserving order.
    return "\n\n".join(b.strip() for b in blocks)


def load_tasks(path: Optional[str] = None) -> List[Dict]:
    """Load benchmark tasks (JSONL). Falls back to fixtures, then a built-in.

    Each task is the shared JSONL record schema. Lines that don't parse are
    skipped. Never raises.
    """
    for candidate in [path, DEFAULT_TASKS, FIXTURE_TASKS]:
        if candidate and os.path.exists(candidate):
            tasks = _read_jsonl(candidate)
            if tasks:
                return tasks
    return [_builtin_task()]


def _read_jsonl(path: str) -> List[Dict]:
    out: List[Dict] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    except OSError:
        return []
    return out


def _builtin_task() -> Dict:
    """A self-contained spec2rtl task (counter) with an inline testbench.

    Used only when no task file exists, so the drivers always produce output.
    """
    tb = (
        "`timescale 1ns/1ps\n"
        "module tb;\n"
        "  logic clk=0, rst=1; logic [3:0] q;\n"
        "  counter4 dut(.clk(clk), .rst(rst), .q(q));\n"
        "  always #5 clk = ~clk;\n"
        "  integer i;\n"
        "  initial begin\n"
        "    @(negedge clk); rst=0;\n"
        "    for (i=0;i<5;i=i+1) @(negedge clk);\n"
        "    if (q==4'd5) $display(\"PASS q=%0d\", q);\n"
        "    else $display(\"FAIL q=%0d expected 5\", q);\n"
        "    $finish;\n"
        "  end\n"
        "  initial begin #1000 $display(\"FAIL timeout\"); $finish; end\n"
        "endmodule\n"
    )
    return {
        "id": "builtin_counter4",
        "task": "spec2rtl",
        "instruction": "Write a 4-bit synchronous up counter.",
        "input": "module counter4(input clk, input rst, output reg [3:0] q); "
                 "resets q to 0 when rst is high, increments on each posedge clk.",
        "output": "",
        "meta": {
            "source": "builtin",
            "module_family": "counter",
            "held_out": True,
            "top_module": "tb",
            "testbench": tb,
            "pass_regex": r"\bPASS\b",
        },
    }


def build_prompt(task: Dict) -> str:
    """Compose an RTL-generation prompt from a task record."""
    instr = task.get("instruction") or "Write the requested SystemVerilog module."
    spec = task.get("input") or ""
    parts = [instr.strip()]
    if spec.strip():
        parts.append("\nSpecification:\n" + spec.strip())
    parts.append(
        "\nRespond with a single synthesizable SystemVerilog module only, "
        "inside a ```systemverilog code block."
    )
    return "\n".join(parts)


def _materialize_tb(task: Dict, run_dir: str, task_id: str) -> Optional[str]:
    """Write the task's testbench to disk; return its path or None."""
    meta = task.get("meta") or {}
    tb_text = meta.get("testbench")
    tb_path = meta.get("testbench_path")
    if isinstance(tb_text, str) and tb_text.strip():
        path = os.path.join(run_dir, "{0}_tb.sv".format(task_id))
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(tb_text)
        return path
    if isinstance(tb_path, str) and os.path.exists(tb_path):
        return tb_path
    return None


def evaluate_completion(task: Dict, completion: str, run_dir: str, tag: str) -> Dict:
    """Extract RTL from a completion, write it, and score it against the tb.

    Returns {compile_pass, test_pass, tool, module_path, ...}.
    """
    meta = task.get("meta") or {}
    top_module = meta.get("top_module")
    # For module extraction we want the DUT module name if known, else the
    # task input's module name; the tb top is usually a separate `tb`.
    dut_hint = meta.get("dut_module") or _guess_dut_name(task)
    rtl = extract_rtl(completion, top_module=dut_hint)

    os.makedirs(run_dir, exist_ok=True)
    mod_path = os.path.join(run_dir, "{0}.sv".format(tag))
    with open(mod_path, "w", encoding="utf-8") as fh:
        fh.write(rtl or "// (empty completion)\n")

    tb_path = _materialize_tb(task, run_dir, tag)
    res = evaluate_outputs(
        mod_path,
        testbench_path=tb_path,
        top_module=top_module,
        pass_regex=meta.get("pass_regex"),
    )
    res["module_path"] = mod_path
    res["testbench_path"] = tb_path
    return res


def _guess_dut_name(task: Dict) -> Optional[str]:
    src = (task.get("input") or "") + "\n" + (task.get("instruction") or "")
    m = re.search(r"\bmodule\s+([A-Za-z_][A-Za-z0-9_]*)", src)
    return m.group(1) if m else None


def run_pass_at_k(
    model: str,
    n: int = 20,
    temp: float = 0.8,
    tasks_path: Optional[str] = None,
    run_name: Optional[str] = None,
    wandb_run=None,
) -> Dict:
    """Sample ``n`` completions per task, score each, compute c per task.

    Returns a model-result row consumable by make_benchmark_table:
    {"model", "problems":[{"n","c","compile_pass","test_pass"}...],
     "compile_pass", "testbench_pass", "is_placeholder", ...}.
    """
    from backend import llm_client

    tasks = load_tasks(tasks_path)
    run_name = run_name or "passk_{0}".format(_safe(model))
    run_dir = os.path.join(RESULTS_DIR, run_name)
    os.makedirs(run_dir, exist_ok=True)

    problems: List[Dict] = []
    is_placeholder = False
    raw_log = []

    for task in tasks:
        task_id = str(task.get("id", "task"))
        try:
            completions = llm_client.generate(
                build_prompt(task), model=model, n=n, temp=temp
            )
        except Exception as exc:
            completions = []
            raw_log.append("generate failed for {0}: {1}".format(task_id, exc))
        if not completions:
            completions = [""]
        # Heuristic: if every completion is byte-identical, it's almost certainly
        # the deterministic serve_stub fallback -> placeholder.
        if len(set(completions)) <= 1 and n > 1:
            is_placeholder = True

        c = 0
        comp_hits = 0
        for i, comp in enumerate(completions):
            sample_dir = os.path.join(run_dir, task_id, "sample_{0:03d}".format(i))
            res = evaluate_completion(task, comp, sample_dir, "cand")
            # Save raw completion + eval log.
            with open(os.path.join(sample_dir, "raw.txt"), "w", encoding="utf-8") as fh:
                fh.write(comp)
            with open(os.path.join(sample_dir, "eval.json"), "w", encoding="utf-8") as fh:
                json.dump(res, fh, indent=2)
            if res.get("compile_pass"):
                comp_hits += 1
            # A "pass" = test_pass when measured; else fall back to compile_pass.
            tp = res.get("test_pass")
            passed = tp if tp is not None else res.get("compile_pass")
            if passed:
                c += 1
        problems.append({
            "id": task_id,
            "n": len(completions),
            "c": c,
            "compile_pass": comp_hits / max(1, len(completions)),
            "test_pass": c / max(1, len(completions)),
        })
        # Live progress to W&B so the run shows charts during the run, not just
        # at the end. No-op if no run was passed.
        if wandb_run is not None:
            try:
                wandb_run.log({
                    "live/{0}/tasks_done".format(model): len(problems),
                    "live/{0}/running_compile_pass".format(model): _avg([p["compile_pass"] for p in problems]),
                    "live/{0}/running_test_pass".format(model): _avg([p["test_pass"] for p in problems]),
                })
            except Exception:
                pass

    row = {
        "model": model,
        "problems": problems,
        "compile_pass": _avg([p["compile_pass"] for p in problems]),
        "testbench_pass": _avg([p["test_pass"] for p in problems]),
        "is_placeholder": is_placeholder,
        "run_dir": run_dir,
        "n": n,
        "temp": temp,
    }
    with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as fh:
        json.dump(row, fh, indent=2)
    if raw_log:
        with open(os.path.join(run_dir, "run.log"), "w", encoding="utf-8") as fh:
            fh.write("\n".join(raw_log))
    return row


def _avg(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return (sum(xs) / len(xs)) if xs else None


def _safe(s: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s))
