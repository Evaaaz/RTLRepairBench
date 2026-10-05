# Reproduction runbook

Run all commands from the repository root. The commands below are ordered from
read-only and offline to expensive and external.

## 1. Inspect the environment

```bash
make doctor
```

The offline analysis requires only Python. Verilator, Icarus, Docker, and TeX
are reported as optional because they are needed only by the corresponding
validation or paper-build lane.

## 2. Run tests

```bash
make test
```

This runs the public-compatible scientific and release-policy suite. The four
tests that require pre-amendment private calibration bindings are selected by
exact test ID and reported separately; `make test-full` runs them when those
private inputs are materialized.

## 3. Reconstruct released analyses

```bash
make reproduce
```

Outputs are written under `build/reproduced/`:

- `primary_stats.json`
- `clean75_stats.json`
- `specialization_summary.json`
- `output_budget_confound.json`
- `paper/fig_curve_body.tex`
- `paper/tab_curve.tex`
- `paper/tab_multimodel.tex`
- `paper/app_prompts.tex`
- `reproduction_report.json`

The command validates frozen denominators, cluster counts, point estimands,
coverage decisions, per-case specialization summaries, and the output-budget
sign tests/truncation counts from the sanitized ledger. It never calls a model
endpoint.
The four regenerated TeX fragments, including the appendix imported from the
five live prompt builders, must match the frozen paper inputs byte-for-byte
before any successful report is published.

For byte-identical bootstrap JSON, use CPython 3.9.6 and run:

```bash
make reproduce-strict
```

Expected analysis hashes:

- full85: `311f180b8200d35fee20c961afc027f44646982458ac399eb4008bf79d6e4ba2`
- clean75: `77591d9cad02c419fb7c7d011e8fbe47eccfd8b4b5727e3150d33132a5b9a171`

Other Python versions can yield slightly different finite Monte Carlo CIs and
p-values. The command does not hide this: it records the runtime and
`byte_exact=false` while still requiring the scientific estimands and counts to
match.

## 4. Validate the release and fresh export

```bash
make check
make smoke
make verify
```

`make check` validates the allowlist, paper-input hashes, paths, credentials,
raw-output policy, and Python syntax. `make smoke` creates a fresh temporary
export, compiles the code, runs the public test suite, and executes offline
reproduction there. `make verify` combines the supported checks.
Runtime outputs under `build/reproduced*` are excluded from the immutable
manifest, so this sequence remains valid after Section 3 has been run.

To create a persistent export, the target path must not exist and must be
outside this source tree:

```bash
make export OUT=/absolute/path/to/release-directory
```

## 5. Build the paper

If `latexmk` or `tectonic` is installed:

```bash
make paper
```

This rebuilds both poster figures and their previews under
`build/paper/assets/`, then builds the current `paper/paper.tex`. The result is
`build/paper/paper.pdf`. The target fails unless the main text is 3--4 pages,
the NeurIPS checklist is present, and the extracted PDF has no unresolved
references, visible review comments, or TODO markers. The hash-bound snapshot
under `paper/` is not overwritten by an ordinary build.

## 6. Tool-gate smoke tests

The full dataset and RepairBench lanes require:

```text
verilator
iverilog
vvp
```

The scripts check these before scientific work begins. A missing tool,
non-zero simulator exit, missing differential verdict, empty semantic output,
or empty held-out reference is an error by default.

The formal lane uses the pinned container:

```bash
bash code/scripts/run_formal_container.sh --build toolchain-info \
  --output generated/formal/toolchain_manifest.json
```

Building the image can require network access to the checksum-pinned upstream
archive. The tracked `data/formal/toolchain_manifest.json` is a frozen
historical snapshot. Subsequent runs must record the image ID under
`generated/` (or another fresh external path) and compare it with the snapshot,
not replace it.

## 7. External artifact materialization

Level-B re-scoring requires candidate/proof trees that are intentionally not in
Git:

```bash
export RTLREPAIR_ARTIFACTS=/absolute/path/outside/the/repository
```

Do not trust a matching filename. Each object must be present in a real
manifest with its immutable URI, byte size, SHA-256, producer, source commit,
and access class. See `docs/SHARED_STORAGE.md`.

## 8. Model reruns

Model calls are Level C and produce new experimental runs. Before starting one:

1. choose a new immutable run ID and output directory;
2. pass all dataset, golden, spec-join, and scheduled-cell preflights;
3. confirm the expected cost through the operator workflow;
4. keep credentials only in environment/approved credential storage;
5. record provider/model response metadata without recording authorization;
6. run the pinned verifier and offline analysis on the new candidates.

Historical hosted-model revisions were not all immutable. Therefore a rerun
must not overwrite or masquerade as the released result, even when the model
alias is unchanged.

## Known non-closures

`claims.json` is normative. In particular:

- the released training file has 5,775 rows and lacks 6 `rename_declaration`
  and 3 `width_mismatch` pairs relative to the historical 5,784-row output,
  which is not released (see `data/README.md`); historical source snapshots are
  not fully pinned in the release;
- the paraphrase audit lacks its original producer/input;
- several historical Boolean-only benchmark results lack candidate RTL for
  independent re-scoring;
- the larger-design appendix now labels the 20-case result as a legacy
  sensitivity and discloses the newer untracked 18-case supported-set audit;
  independent re-scoring still requires external artifacts.
- the Claude Opus 4.8 second-anchor aggregate lacks its historical row-level
  outcomes and candidates, so its intervals cannot be recomputed.

A successful command does not erase these disclosures.
