# Pinned formal environment

The image uses the official YosysHQ OSS CAD Suite `2026-05-08` archive and
verifies its architecture-specific SHA-256 during the build.  The Ubuntu 24.04
base is pinned by multi-platform OCI digest; no packages are installed from a
mutable apt index.  Exact pins are in `toolchain.lock.json`.

Build once and record the actual image ID/tool versions:

```bash
code/scripts/run_formal_container.sh --build toolchain-info \
  --output generated/formal/toolchain_manifest.json
```

Subsequent invocations reuse the image.  Reconstruct canonical sources and the
initialization protocol manifests with:

```bash
code/scripts/run_formal_container.sh build-data \
  --output-dir generated/formal/canonical_goldens \
  --manifest generated/formal/sem85_manifest.json \
  --protocol-manifest generated/formal/protocol_manifest.json
```

Compare these fresh outputs with the tracked historical evidence under
`data/formal/`; do not rebuild in place over the frozen files.

Run the fail-fast 85 golden/golden + 85 golden/known-mutant gate with:

```bash
code/scripts/run_formal_container.sh calibrate
```

The calibration is resumable through
`generated/formal/calibration_results.jsonl`.  It immediately stops if a known
mutant is reported `PROVED`; the gate passes only when all golden self-checks
are proved and all known mutants have Icarus-replayed witnesses.

For generated candidates, `validate-batch` consumes JSONL rows containing
`call_id`, `seed_id`, and inline `golden_rtl`/`candidate_rtl` (or `*_path`).  Its
output can be ingested by `ExperimentLedger.ingest_validation_jsonl`:

```bash
code/scripts/run_formal_container.sh validate-batch \
  --jobs generated/vericodegen/validation_jobs.jsonl \
  --results generated/vericodegen/validation_results.jsonl \
  --artifacts generated/vericodegen/validation
```

`formal-gate` evaluates only the four main `spec{0,1}_loc{0,1}` arms by
default.  It requires at least 77 definitive formal results out of exactly 85
per arm and rejects any `PROVED` result contradicted by independent simulation.
