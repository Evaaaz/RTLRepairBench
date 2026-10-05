# Experiment and exhibit scripts

These are named research workstreams, not a second benchmark library. Shared
evaluation, formal, and statistics logic belongs in `code/benchmarks/`; shared
dataset gates belong in `code/datagen/`.

| Group | Scripts | Purpose |
|---|---|---|
| coverage and exhibits | `e0_coverage_audit.py`, `e4_tier_s_rescore.py`, `e5_formal_coverage.py`, `e6_*.py`, `make_capability_figure.py` | audit paper cells and render derived tables/figures |
| second-model studies | `second_model_*.py`, `decomp_second_anchor.py` | capability, leakage, and decomposition analyses |
| RepairBench extensions | `repairbench_*.py`, `origen_run.py`, `self_reflect.py` | additional model or prompting arms |
| larger designs | `build_largescale.py`, `realbug_frontier.py`, `general_score.py`, `largescale_formal.py` | transfer study and verification |
| protocol checks | `contract_robustness_experiment.py`, `ws_i_prompts_appendix.py` | robustness and appendix generation |
| operations | `run_formal_container.sh`, `check_llm.py`, `eval_endpoint.py` | toolchain and endpoint diagnostics |

Scripts read versioned inputs from top-level `data/` and `configs/`; ordinary
derived outputs default to excluded `generated/` or `build/` paths. Set
`RTLREPAIR_ROOT`, `RTLREPAIR_DATA`, `RTLREPAIR_OUT`, or
`RTLREPAIR_ARTIFACTS` explicitly for external materializations. Never point
ordinary output variables at `paper/` or `artifacts/public/`; frozen snapshots
are promoted there only after review and checksum binding. The offline
reproducer regenerates all four TeX inputs in staging and byte-compares them
before publishing a verified report.

`second_model_capability_curve.py` is the canonical fail-closed curve analyzer.
It requires the exact retained-harness 68-by-4 roster, a complete typed verdict
ledger, one strict `PROVED`-only rule for all five models, and writes a
hash-bound cohort/source audit alongside the curve. The equally sized primary
formal intersection is a different manifest and must never be substituted.

The parent research tree also contains the exploratory drafts
`x1_counterexample_power.py` and `x2_anchor_resample.py`. They mix historical
cohorts or scorers and are intentionally excluded from the standalone
allowlist; they are not current curve or primary-analysis producers.

Not every historical workstream has complete retained inputs. `claims.json` is
the authority for whether a result is reproducible, needs external artifacts,
is incomplete, or is stale.
