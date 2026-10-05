# RTLRepair Real-Bug Semantic Repair Set — Datasheet

A held-out benchmark of **real, model-produced** functional (compiles-but-wrong)
Verilog bugs, paired with exact simulator divergence traces, golden references,
and natural-language specifications. It complements the synthetic-mutation
RepairBench by testing repair on bugs that real models actually make, not bugs we
injected.

## Motivation

Prior LLM-for-RTL repair resources are either compilation-error sets (RTLFixer,
OriGen's VerilogFixEval) or synthetic single-line mutations. Neither captures the
functional bugs an LLM produces when it writes RTL that **compiles and lints clean
but is behaviorally wrong**. This set fills that gap: every case is a generation
that a real model emitted for a VerilogEval task, which passed iverilog
compilation and `verilator --lint-only` but failed the task's self-checking
testbench under simulation.

## Construction (`code/datagen/build_realbug_repairset.py`)

Source: greedy/sampled generations from the served `base`
(Qwen2.5-Coder-7B-Instruct) and `tuned` (v5 LoRA) models on VerilogEval v2,
already scored by the harness. A candidate sample is kept iff:

1. `iverilog` **compiles** it (compile_pass = true), and
2. the task's self-checking **testbench fails** it (test_pass = false), and
3. `verilator --lint-only` reports it **lint-clean** (so it is a behavioral bug,
   not a lint bug in disguise — 409 lint-dirty candidates were excluded), and
4. it is **distinct** per (task, normalized-RTL) (deduplicated), with at most
   **3** cases per task to prevent any one task from dominating.

The divergence trace is parsed from the recorded simulator output: the diverging
output name, the first mismatch time, and the mismatch count out of total samples.

**Leak-free by construction.** The generations are named `module TopModule` (the
VerilogEval DUT name), so they carry no Prob-id identity — unlike a filename-keyed
synthetic set, there is no problem-identity token for a pretrained model to exploit.

## Composition

| field | value |
|---|---|
| cases | 274 |
| distinct VerilogEval tasks | 112 |
| source model: base / tuned | 252 / 22 |
| sequential / combinational | 169 / 105 |
| cases with divergence trace | 274 / 274 |
| cases with paired NL spec | 274 / 274 |
| cases-per-task (1 / 2 / 3) | 26 / 10 / 76 tasks |
| candidates excluded as lint-dirty | 409 |

## Schema (`data/repairbench_realbugs.jsonl`, one JSON object per line)

```
id              "realbug:<task>:<model>:<idx>"
task            VerilogEval task id (e.g. Prob006_vectorr)
source_model    "base" | "tuned"
bucket          "semantic-real"
broken_rtl      the model generation (module TopModule ...), compiles + lints clean
divergence_trace {output, mismatches, samples, first_mismatch_time}
golden_rtl      the VerilogEval reference solution
spec            the VerilogEval natural-language instruction
taxonomy        {code_lines, sequential, has_case, has_state}
```

A RepairBench-format mirror (`data/repairbench_realbugs_fmt.jsonl`) wraps each case
as a chat prompt (broken RTL + divergence trace) for direct use with
`benchmarks/realbug_repair_eval.py`.

## Recommended use

- **Repair evaluation.** `benchmarks/realbug_repair_eval.py` proposes a fix from
  the broken RTL + the divergence trace (optionally + the spec) and validates the
  fix against the task's **own** testbench via `evaluate_outputs` (iverilog + vvp +
  pass_regex). Recovery = the fix now passes the testbench the generation failed.
- **Spec-conditioning ablation.** Run with/without `--spec` to test whether the
  natural-language intent closes the functional-repair gap on real bugs.
- **Cross-model.** `--backend claude` evaluates a frontier model on the same set.

## Limitations and honest caveats

- The set is **selected for the source models' weaknesses**: a base/v5 failure set
  over-represents the bug modes those models produce, so cross-model recovery
  numbers describe *this distribution of real bugs*, not a model-agnostic gap.
- Random-stimulus equivalence is not used here; correctness is judged by each
  task's fixed testbench, which has finite coverage (a fix that passes the
  testbench may still differ from the golden on unexercised inputs).
- VerilogEval tasks may appear in model pretraining; the set is a *repair* (not
  generation) benchmark, but pretraining familiarity is a named threat.
- Single benchmark family (VerilogEval); RTLLM and real silicon bugs are future work.

## Provenance / reproducibility

Regenerate from a complete banked-generation tree into a fresh path with:

```bash
python3 code/datagen/build_realbug_repairset.py \
  --results /shared/rtlrepair/benchmark_results \
  --seeds /shared/rtlrepair/verilogeval_seeds \
  --eval-tasks data/eval_tasks.jsonl \
  --expected-roster data/repairbench_realbugs.jsonl \
  --output generated/realbugs/repairbench_realbugs.rebuilt.jsonl
```

The command rejects missing inputs, an empty or partial bank, roster drift,
missing specifications/traces, an existing output, and any attempt to overwrite
the tracked 274-case snapshot. The released validators are
`code/datagen/run_verilator_for_repairs.py` and
`code/benchmarks/evaluate_outputs.py`; pin Verilator and Icarus versions when
reproducing. Release maintainers compare a complete regenerated file before an
explicit snapshot promotion.
