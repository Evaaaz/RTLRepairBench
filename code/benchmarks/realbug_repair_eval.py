"""Repair evaluation on the mined REAL-bug semantic set.

Unlike the synthetic RepairBench (validated by random-stimulus differential
simulation), each real-bug case was DEFINED by a specific VerilogEval testbench,
so we validate a proposed fix against that SAME testbench via evaluate_outputs
(iverilog + vvp + the task's pass_regex). Recovery = the fix now passes the
testbench the original generation failed.

Backends: rtlrepair (the served base/v5 via repair_agent.repair) or claude.
Flags --spec injects the NL spec; --temp sets the (rtlrepair) sampling temp.

Usage:
  RTLREPAIR_LLM_URL=... RTLREPAIR_LLM_API_KEY=... \
    .venv/bin/python benchmarks/realbug_repair_eval.py --model base --temp 0.0
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))   # .../code/benchmarks
CODE = os.path.dirname(HERE)                          # .../code
for _p in (HERE, CODE):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import repair_agent  # noqa: E402
from bench_common import extract_rtl  # noqa: E402
from evaluate_outputs import evaluate_outputs  # noqa: E402
from repairbench_eval import claude_repair, parse_item, DATA, OUT_DIR  # noqa: E402

BENCH = os.path.join(DATA, "repairbench_realbugs_fmt.jsonl")
EVAL_TASKS = os.path.join(DATA, "eval_tasks.jsonl")


def load_tasks() -> dict:
    """task id -> (testbench_src, top_module, pass_regex, spec)."""
    t: dict = {}
    for line in open(EVAL_TASKS):
        d = json.loads(line)
        m = d.get("meta", {})
        key = str(d.get("id", "")).split(":")[-1]
        t[key] = (m.get("testbench", ""), m.get("top_module", "tb"),
                  m.get("pass_regex"), (d.get("instruction") or "").strip())
    return t


def passes_testbench(fixed: str, tb_src: str, top: str, regex) -> bool:
    if not (fixed and tb_src):
        return False
    with tempfile.TemporaryDirectory(prefix="realbug_") as td:
        dut = os.path.join(td, "dut.sv")
        tb = os.path.join(td, "tb.sv")
        open(dut, "w").write(fixed)
        open(tb, "w").write(tb_src)
        res = evaluate_outputs(dut, tb, top_module=top, pass_regex=regex)
    return bool(res.get("test_pass"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="base")
    ap.add_argument("--backend", default="rtlrepair", choices=["rtlrepair", "claude"])
    ap.add_argument("--claude-model", default="claude-opus-4-8")
    ap.add_argument("--spec", action="store_true")
    ap.add_argument("--temp", type=float, default=0.2)
    ap.add_argument("--bench", default=BENCH)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks = load_tasks()
    items = [json.loads(line) for line in open(args.bench)]
    if args.limit:
        items = items[: args.limit]

    by_src = collections.defaultdict(lambda: {"n": 0, "recovered": 0})
    by_kind = collections.defaultdict(lambda: {"n": 0, "recovered": 0})
    records = []
    overall = {"n": 0, "recovered": 0, "used_llm": 0}
    for i, item in enumerate(items):
        broken, err = parse_item(item)
        task = item["task"]
        tb_src, top, regex, spec_text = tasks.get(task, ("", "tb", None, ""))
        spec = spec_text if args.spec else None
        fixed, used_llm = "", False
        try:
            if args.backend == "claude":
                raw = claude_repair(broken, err, spec, args.claude_model)
                used_llm = bool(raw)
                fixed = extract_rtl(raw) if raw else ""
            else:
                rr = repair_agent.repair(broken, err, model=args.model, temperature=args.temp, spec=spec)
                used_llm = bool(getattr(rr, "used_llm", False))
                fixed = extract_rtl(rr.fixed_rtl) if rr.fixed_rtl else ""
            if fixed:
                fixed = fixed.rstrip() + "\n"
        except Exception as exc:  # noqa: BLE001
            print(f"  [{i}] repair error: {exc}", flush=True)
        ok = passes_testbench(fixed, tb_src, top, regex) if fixed else False
        src = item["id"].split(":")[2] if item["id"].count(":") >= 2 else "?"
        kind = "sequential" if "always @ (posedge" in broken.replace("@(", "@ (") or "posedge" in broken else "combinational"
        by_src[src]["n"] += 1
        by_src[src]["recovered"] += int(ok)
        by_kind[kind]["n"] += 1
        by_kind[kind]["recovered"] += int(ok)
        overall["n"] += 1
        overall["recovered"] += int(ok)
        overall["used_llm"] += int(used_llm)
        records.append({"id": item["id"], "task": task, "recovered": ok, "used_llm": used_llm})
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{len(items)} done  (recovered so far {overall['recovered']})", flush=True)

    def rate(d):
        return {"n": d["n"], "recovered": d["recovered"],
                "recovery_rate": round(d["recovered"] / d["n"], 4) if d["n"] else 0.0}

    model_label = args.claude_model if args.backend == "claude" else args.model
    summary = {
        "model": model_label, "backend": args.backend, "spec": args.spec,
        "n": overall["n"], "recovered": overall["recovered"],
        "recovery_rate": round(overall["recovered"] / overall["n"], 4) if overall["n"] else 0.0,
        "by_source_model": {k: rate(v) for k, v in by_src.items()},
        "by_kind": {k: rate(v) for k, v in by_kind.items()},
    }
    print(json.dumps(summary, indent=2))
    out = args.out or os.path.join(OUT_DIR, f"realbug_{model_label.replace('/', '-')}{'_spec' if args.spec else ''}.json")
    json.dump({"summary": summary, "records": records}, open(out, "w"), indent=2)
    print("wrote", out)


if __name__ == "__main__":
    main()
