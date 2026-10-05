# VeriCodeGen frozen data artifacts

These files preregister data choices that must not depend on model outputs.
They contain identifiers and hashes, not API credentials.

Regenerate the deterministic artifacts from the repository root:

```bash
python3 code/benchmarks/vericodegen_data.py shift
python3 code/benchmarks/vericodegen_data.py clean75
python3 code/benchmarks/vericodegen_data.py redaction-template
python3 code/benchmarks/vericodegen_data.py main-inputs
python3 code/benchmarks/vericodegen_data.py shift-inputs
python3 code/benchmarks/vericodegen_data.py verify \
  configs/vericodegen/shift50_manifest.json
```

- `shift50_manifest.json` pins RNG 42, the requested 20/20/5/5 strata, the
  fallback path used for each selection, and one source-record SHA per task.
  `tuned` source rows are reported as the preregistered `adapter` group. These
  are model-generated compile-clean failures, not real-world bugs.
- `clean75_manifest.json` is derived only by intersecting semantic cases with
  the existing audited `data/eval_tasks_clean.jsonl` KEEP set. Its 10 excluded
  cases are not hand-guessed in this code path.
- `redaction_annotations.template.json` is intentionally blank. Duplicate it
  outside the frozen config, have two annotators complete it before inspecting
  model outputs, adjudicate, and freeze it with:

```bash
python3 code/benchmarks/vericodegen_data.py redaction-freeze \
  --annotations path/to/completed_annotations.json
```

The freeze command rejects incomplete labels, identical annotator IDs,
source-spec or mutation-target SHA drift, and any redaction that is not exactly
one verbatim clause deletion for `explicit`/`derivable` cases. An `absent` label
must retain the full spec unchanged. Each template row includes the committed diff's
single broken/canonical line pair so annotators can identify the changed
behavior. This `mutation_target` block is human-only audit context: it is not in
runner-facing `model_input` and must never be copied into a model prompt.
For `absent`, the full spec may correctly remain unchanged because it already
omits the target behavior. If more than 20% of cases are judged unredactable
without hinting, the frozen decision is `DROP_REDACTED_ARM`.

The labels mean:

- `explicit`: the original spec directly states the mutated behavior;
- `derivable`: the behavior follows from the spec but is not directly stated;
- `absent`: the spec does not determine the intended behavior.

The code can verify syntactic completeness and hashes; it cannot automate the
semantic judgment. Reviewers should inspect the two annotations and adjudication.

Runner-facing files are separately materialized:

- `main85_inputs.jsonl` has 85 rows. `model_input` contains only normalized,
  anonymized broken RTL, full spec, expected neutral module name, and the
  committed-diff-derived broken line/number. `internal_metadata` contains the
  cluster, mutation family, source ID, and clean75 flag and must never be copied
  into a prompt.
- `shift50_inputs.jsonl` resolves every frozen selection back to a source row,
  verifies its SHA, and omits `golden_rtl`. Since this study has no shift-location
  arm and the pool has no committed mutation diff, `oracle_location` is explicitly
  `null`.
- Each JSONL has a `.manifest.json` sidecar that pins its own SHA and every source
  file SHA. `verify` checks both manifest content and these recorded files.

Neither materialized JSONL contains golden RTL. The displayed main RTL has edge
blank lines normalized, and its invariant is
`broken_rtl.splitlines()[line_number - 1] == broken_line` for all 85 rows.
The statistics loader uses each sidecar's `case_index` to map internal IDs to
the anonymous IDs recorded by the runner; do not join these sets by row order.
