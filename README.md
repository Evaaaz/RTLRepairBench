# RTLRepairBench

Code, data, and evidence for *What Do RTL Repair Benchmarks Actually Measure?*
by Yeyin (Eva) Zhu and Rishabh Ranawat (NeurIPS 2026 Workshop on AI for Chip
Design; [OpenReview](https://openreview.net/forum?id=Qemvbzkv4n)). It contains
the camera-ready paper, RTLRepairBench, the RTLRepairDataset training file, the
canonical data-generation and verification code, sanitized aggregate evidence,
and an allowlisted exporter.

The central idea is simple: a Verilog module that compiles is not necessarily
functionally correct. RTLRepair therefore separates two bug classes:

| Class | Admission gate | Repair success gate |
|---|---|---|
| lint | the mutant fails `verilator --lint-only` while the golden passes | the repaired module lints clean |
| semantic | the mutant lints clean but diverges from the golden in differential simulation | the repaired module matches the golden under the declared simulation/formal protocol |

## Start here

The default workflow makes no network requests and no model calls:

```bash
make doctor
make verify
make paper
```

`make verify` runs the release check, public tests, Level-A reproduction, and a
fresh-tree smoke test. It is safe to run after `make reproduce`; generated files
under `build/` are explicitly outside the immutable release manifest.

`make reproduce` reconstructs the full85 and clean75 statistical analyses, the
eight-arm specialization summary, the output-budget audit, and three generated
paper exhibits plus the live-builder prompt appendix under `build/reproduced/`.
The frozen
10,000-draw bootstrap is byte-identical under CPython 3.9.6; on another Python
version the tool still checks denominators, clusters, point estimands, and
coverage decisions, but records that the finite Monte Carlo output is not byte
exact. Use `make reproduce-strict` when the exact runtime is available.

No CUDA installation, API key, or third-party Python package is required for
the offline analysis path. Building the paper additionally requires TeX and
Poppler's `pdftotext`/`pdftoppm` tools.

## Repository map

```text
paper/                    camera-ready paper.tex/PDF, style, and figures
code/
  datagen/                canonical mutation, tool-gating, and decontamination pipeline
  benchmarks/             RepairBench, formal verifier, statistics, and providers
  scripts/                named paper experiments and exhibit producers
configs/                  frozen protocols and materialized primary/shift inputs
data/                     compact benchmark inputs and canonical formal goldens
artifacts/public/         sanitized rows, verdicts, summaries, and expected statistics
docker/formal/            checksum-pinned formal toolchain
tools/                    offline reproduction and standalone export
tests/                    scientific, fail-closed, and release-policy tests
claims.json               paper claim to producer/input/output/status ledger
environment.lock.json     analysis and formal runtime contract
```

There is one current manuscript source: `paper/paper.tex`; `make paper` builds
it and enforces the workshop's 3--4-page body limit plus checklist presence.
The other small TeX fragments in `paper/` are frozen Level-A reconstruction
targets, not alternate manuscripts.

There is one canonical implementation of dataset construction:
`code/datagen/`. The divergent files formerly duplicated at `code/` were
removed. Migration records and the obsolete June working PDF are retained only
under `docs/history/` and are excluded from release exports.

## What “reproduce” means

The package distinguishes three levels:

1. Level A reconstructs supported tables and statistics from sanitized frozen
   rows. This is the reviewer path implemented by `make reproduce`.
2. Level B re-scores saved candidate RTL in the pinned simulation/formal
   toolchain. It needs candidate and proof trees from shared artifact storage.
3. Level C calls model endpoints or local weights to generate new candidates,
   then runs Levels B and A. Hosted-model revisions are not fully immutable, so
   this is a new replication, not a promise of byte-identical generations.

Every manuscript claim is classified in [`claims.json`](claims.json) as
`reproducible_from_snapshot`, `rerunnable_with_external_artifacts`,
`incomplete`, or `stale`. The detailed evidence map and known gaps are in
[`docs/RESULTS_AND_PROVENANCE.md`](docs/RESULTS_AND_PROVENANCE.md). A successful
build never silently upgrades an incomplete or stale claim.

## Dataset regeneration

The current pipeline fails closed if Verilator, Icarus, held-out references, or
certified semantic outputs are missing:

```bash
cd code/datagen
SEEDS=/path/to/my/seeds OUT=out bash build_dataset.sh
# -> out/repair_lint.jsonl, out/repair_semantic.jsonl, out/repair_train.jsonl
```

The released training file is `data/rtlrepairdataset_train.jsonl`: 5,775
repair pairs (5,394 lint and 381 semantic) in chat format. Relative to the
historical 5,784-row output (SHA-256
`73fd32ae2e3a1d419691fc16a67965aa929bc912155edc10c155895eac36518a`), which is
not released, it lacks 6 `rename_declaration` and 3 `width_mismatch` pairs; see
`data/README.md`. The original source snapshots are not vendored. A fresh run therefore creates a new certified dataset rather than
proving byte identity with the historical corpus; `claims.json` records this
boundary.

## Model and verifier runs

RepairBench configuration and preflight rules are documented in
[`code/benchmarks/README.md`](code/benchmarks/README.md). Full runs now reject
missing semantic goldens, incomplete spec joins, missing model responses,
duplicate/missing case IDs, and partial arms. Diagnostic overrides are marked
`paper_valid=false` and cannot overwrite a canonical result silently.

Formal validation uses:

```bash
bash code/scripts/run_formal_container.sh --build toolchain-info \
  --output generated/formal/toolchain_manifest.json
```

The container pins the base image and architecture-specific OSS CAD Suite
archive by digest. The tracked `data/formal/toolchain_manifest.json` is frozen
historical evidence; new toolchain records and candidate re-scores must write
under `generated/` (or another fresh external path), then be compared with the
frozen files rather than overwriting them.

## Shared large outputs

Keep raw responses, generated RTL, proof directories, waveforms, full ledgers,
and model weights outside Git. Materialize them through one local root:

```bash
export RTLREPAIR_ARTIFACTS=/absolute/path/outside/this/repository
```

The storage backend can be a team object store or shared mounted filesystem.
What matters scientifically is an immutable logical path, exact byte size,
SHA-256, producer, source commit, and access class. The complete contract is
[`docs/SHARED_STORAGE.md`](docs/SHARED_STORAGE.md). No bucket or URI is invented
in this repository; add one only after upload and read-back verification.

## Standalone export

Never publish a normal fork with the full monorepo history. Build a clean
allowlisted tree instead:

```bash
make check
make export OUT=/path/to/new-directory
make smoke
```

The exporter rejects symlinks, caches, raw reasoning traces, credential-shaped
strings, workstation paths, oversized files, migration history, and private
artifact directories. It emits a content manifest and tests the exported tree
from a fresh temporary path.

This snapshot does not imply a license for upstream datasets or model outputs.
Required upstream notices are reproduced in [`NOTICE`](NOTICE); known upstream
versions, declared licenses, and unresolved boundaries are listed in
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md), and open release items in
[`docs/RELEASE_CHECKLIST.md`](docs/RELEASE_CHECKLIST.md).

## Citation

```bibtex
@inproceedings{zhu2026rtlrepair,
  title     = {What Do {RTL} Repair Benchmarks Actually Measure?},
  author    = {Zhu, Yeyin and Ranawat, Rishabh},
  booktitle = {NeurIPS 2026 Workshop on AI for Chip Design},
  year      = {2026},
  url       = {https://openreview.net/forum?id=Qemvbzkv4n}
}
```
