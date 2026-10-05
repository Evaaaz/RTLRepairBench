# Configuration contract

This is the single repository-level configuration directory. Files here are
inputs to a recorded protocol, not mutable personal defaults.

- `submission.json` is the only current submission/venue record; it points to
  the AI for Chip Design poster paper and its 3--4-page contract.
- `vericodegen2026.json` fixes a historical generation and analysis protocol.
  Its `historical_venue` field is provenance for those runs, not the venue of
  the current submission.
- `vericodegen/` contains the materialized full85/clean75/shift inputs, their
  manifests, the redaction contract, and the protocol amendment.
- `vericodegen/curve68_manifest.json` names the post hoc retained-harness
  cohort used to schedule the exploratory capability curve;
  `anchor_formal68_manifest.json` separately names the anchor's true four-arm
  formal-definitive intersection. Both contain 68 cases but overlap on 52.
- `eval.yaml` and `model.yaml` describe legacy evaluation/model settings kept
  for provenance. They are not the source of truth for the offline reviewer
  path.

The RepairBench experiment matrix remains next to its runner at
`code/benchmarks/configs/experiments.json`; it controls model-by-signal cells,
not the VeriCodeGen protocol.

Do not edit a frozen file in place for a new experiment. Copy it to a newly
named protocol or run directory, record the source commit and runtime, and bind
every materialized input by byte size and SHA-256. `make check` verifies the
files admitted to the standalone release.
