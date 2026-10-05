# Benchmark and verification harness

This package owns candidate generation/evaluation, formal validation, and
statistical analysis. It consumes inputs from top-level `data/` and writes new
runs outside the source package.

## Modules

| Area | Files |
|---|---|
| RepairBench | `repairbench_eval.py`, `realbug_repair_eval.py`, `iterative_repair_eval.py` |
| run matrix | `run_experiments.py`, `configs/experiments.json`, `analyze_repairbench.py` |
| validation | `evaluate_outputs.py`, `formal_{data,protocol,verify}.py`, sibling `datagen/` gates |
| primary study | `vericodegen_{data,eval,stats}.py`, `export_vericodegen_supplement.py` |
| providers | `backend/` and `backend/nvidia_responses.py` |
| legacy baseline drivers | `run_all.py`, `run_*_eval.py`, `score_clean.py`, `build_final_tables.py` |

Frozen result evidence is under `artifacts/public/`. New runs default to
`generated/repairbench/`; set `RTLREPAIR_OUT` to an absolute run directory to
override that location.

## RepairBench safety contract

Each item is a broken module plus its tool diagnostic. Lint repairs must pass
`verilator --lint-only`. Semantic repairs must compile and match a checksum-
bound golden under differential `iverilog`/`vvp` simulation.

A paper-valid full arm requires all of the following before its result is
committed:

- the benchmark has the exact expected unique IDs;
- every scheduled spec join and model response is present;
- every semantic golden and its SHA-256 manifest entry is present;
- every simulator process exits successfully and emits a verdict;
- the run is complete, non-diagnostic, and written atomically.

`--limit` and explicit missing-seed overrides are diagnostic. Their summaries
are marked `paper_valid=false` and cannot silently replace a canonical arm.

## Inspect and run a matrix

The experiment matrix is the source of truth for model-by-signal cells. A dry
run does not call any endpoint:

```bash
python code/benchmarks/run_experiments.py --dry-run
```

For a real run, use a fresh immutable output directory and inject credentials
through the environment or an approved secret manager:

```bash
export RTLREPAIR_OUT=/absolute/path/to/new-run
python code/benchmarks/run_experiments.py --only base_diag base_diag_spec
RTLREPAIR_OUT="$RTLREPAIR_OUT" \
  python code/benchmarks/analyze_repairbench.py
```

The generic served-model endpoint is `RTLREPAIR_LLM_URL`. Model-specific arms
use the endpoint variables named in `configs/experiments.json`. Do not put keys
in configs, commands committed to Git, result JSON, or the artifact manifest.

## Model-specific interpretation

OriGen_Fix is a repair model, but its native error field was trained on compiler
messages; semantic differential-simulation diagnostics are out of distribution.
Its natural comparison is the diagnostic-plus-spec base arm.

VeriReason and VR-Qwen are spec-to-RTL reasoning models rather than repair
models. Their repair results are lower-bound transfer measurements. Native
prompt arms and generic-prompt parity arms must remain separately labeled.

## Reproduction boundary

`make reproduce` does not call a model. It rebuilds released analyses from
sanitized, checksum-bound rows under `artifacts/public/`. Exact model reruns
need external endpoints/weights and retained candidate trees; they are new
replications unless an immutable provider revision was recorded. See
`docs/RESULTS_AND_PROVENANCE.md` and `claims.json` for claim-level status.
