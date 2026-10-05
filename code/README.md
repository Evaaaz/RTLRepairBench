# Code map

The source tree has three boundaries:

| Directory | Responsibility | Typical entry point |
|---|---|---|
| `datagen/` | build, mutate, gate, and decontaminate datasets | `datagen/build_dataset.sh` |
| `benchmarks/` | run RepairBench, validate candidates, and compute statistics | `benchmarks/run_experiments.py` |
| `scripts/` | named follow-up analyses and paper exhibit producers | see `scripts/README.md` |

`project_paths.py` is the canonical repository-root resolver for shared
modules. Named one-file workstreams resolve the fixed standalone
`code/scripts/` or `code/benchmarks/` layout to the same bundle root; no runtime
path walks out to an enclosing monorepo. The supported overrides are:

- `RTLREPAIR_ROOT` for the repository root
- `RTLREPAIR_DATA` for read-only/materialized benchmark data
- `RTLREPAIR_OUT` for a new run directory

Without overrides, inputs resolve under top-level `data/` and outputs under
top-level `generated/`. Generated candidates, raw model responses, waveforms,
and proof work directories do not belong in source directories.

The frozen public evidence used by the paper is under `artifacts/public/`, not
under the executable packages. Repository-level protocol files live only under
top-level `configs/`; the former duplicate `code/configs/` tree was removed.

Start with `make reproduce` for the offline paper result. Use the lower-level
entry points only when creating a new dataset, verifier run, or model run.
