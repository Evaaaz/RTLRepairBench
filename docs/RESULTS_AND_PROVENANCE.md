# Results and provenance

This document maps the current anonymous poster, `paper/paper.tex`, to the
released evidence. The machine-readable source of truth is `claims.json`.
Older manuscript drafts and their claim maps are historical material and are
not part of the anonymous export.

## Reproduction levels

- **Level A — aggregate reconstruction.** Recompute tables, rates, paired
  contrasts, and audit reports from released sanitized rows or frozen
  aggregates. This is the no-network path implemented by `make reproduce`.
- **Level B — candidate re-scoring.** Re-run simulation/formal checks on saved
  candidate RTL. This requires external candidate and proof trees that are not
  included in this package.
- **Level C — model replication.** Generate new candidates through hosted
  endpoints or local weights, then run Levels B and A. Provider aliases are not
  guaranteed to identify immutable model revisions, so this produces a new
  replication rather than byte-identical historical output.

The four claim statuses are:

- `reproducible_from_snapshot`
- `rerunnable_with_external_artifacts`
- `incomplete`
- `stale`

A successful build or test never upgrades a claim status.

## Poster evidence map

### Training dataset

The historical `repair_train.jsonl` was located outside the release tree and
audited directly:

| property | value |
|---|---:|
| rows | 5,784 |
| unique IDs | 5,784 |
| distinct normalized seed IDs | 1,895 |
| lint rows | 5,403 |
| semantic rows | 381 |
| bytes | 10,814,575 |
| SHA-256 | `73fd32ae2e3a1d419691fc16a67965aa929bc912155edc10c155895eac36518a` |

The exact historical file is not released. The public package releases:

- the fail-closed construction pipeline under `code/datagen/`;
- `data/rtlrepairdataset_train.jsonl`, the 5,775-row repair bucket of the SFT
  mixture, which lacks 6 `rename_declaration` and 3 `width_mismatch` pairs
  relative to the composition above (see `data/README.md`); and
- the hash-bound composition above.

The released benchmark has 130 distinct normalized seed IDs. A direct audit of
the located historical file found an empty identifier intersection with the
1,895 training seeds. That audit ran on the historical file, so it remains
hash-bound external evidence; the released file's seed modules can be checked
against the benchmark directly.

### RTLRepairBench

The released benchmark counts can be recomputed directly:

| file | rows | SHA-256 |
|---|---:|---|
| `data/repairbench_heldout.jsonl` | 459 | `05c749bb980838a7e2a57a1c34ec16314acc4068b732912d29a18d91af9c8b8f` |
| `data/repairbench_lint.jsonl` | 374 | `51438f5065308c4aada978813d14136471ee85c6a056d3a59b6369c1aa67c9a4` |
| `data/repairbench_sem85.jsonl` | 85 | `e98c2bec7cb4325088030bafdab17096c9b0cae204257515ea2b4676cf6a97dc` |

The appendix mutation table is derived from the `bucket` and `mutation` fields
in these rows. The released canonical goldens and protocol manifests document
the evaluation contracts; complete historical candidate re-scoring remains a
Level-B operation.

### Syntactic/behavioral capability gap

`artifacts/public/capability/split_discrimination.json` records the six-model
subset cited in the poster. Syntactic recovery spans 87.7–100.0%, while
behavioral recovery spans 10.6–75.3%. Its SHA-256 is
`1f72773267e1b05e1392a7e5d39a92a7df5f97465b3a691f72942615c670ccc7`.

### Oracle sensitivity

`artifacts/public/capability/frontier_oracle_flip.json` is the frozen evidence
for Table 1 (SHA-256
`b63f693c7e27028932e8d8b73ae7a6bdc1a6e530b3cca8284875a75817694506`).
It records:

- 85 scheduled semantic cases and 79 cases with definitive formal verdicts in
  both arms;
- Tier S over cycles 0--199: 60/79 (75.9%) diagnostic versus 51/79
  (64.6%) diagnostic+specification, a -11.4 pp paired contrast computed from
  counts;
- Tier S over cycles 1--199: 72/79 (91.1%) versus 78/79 (98.7%), a +7.6 pp
  paired contrast;
- reset-aware Tier S: 71/79 (89.9%) versus 78/79 (98.7%), a +8.9 pp paired
  contrast;
- Tier F: 71/79 (89.9%) versus 78/79 (98.7%), also a +8.9 pp paired contrast;
- reset-aware Tier S and Tier F agree on all 158 candidate-arm verdicts in this
  79-case cohort; this cohort-specific agreement does not make bounded
  simulation a formal proof;
- on the 79-case intersection, 12 diagnostic-arm and 27 specification-arm
  Tier-S-fail/Tier-F-proved cases; and
- across all 85 cases, reset-aware rescoring clears 12/12 diagnostic-arm and
  27/30 specification-arm false divergences.

The last item is deliberately stated as **almost all**, not all. The artifact
does not justify attributing every simulation/formal mismatch to cycle 0.

### Repair versus regeneration

`data/repairbench_realbugs.jsonl` contains 274 mined failures from 112 distinct
task values (SHA-256
`a8cc99355a9ab934558e664cf8fcf33c07b0f8bbce5269406e569f15838a9ae6`).

`artifacts/public/realbugs/cycle1_rescore.json` contains the six-model
complete-case panel displayed in Table 2 (SHA-256
`e6102fc7d95198c6b948764fb166d7dfed3d14abc68f9a7bfca1fb75e5581824`).
The displayed sample size varies from 260 to 274 because an observation must
have a retained, scoreable candidate in all three arms. The cycle-1
specification-only minus specification-conditioned contrast ranges from -8.5
to +18.3 percentage points.

The paper reports point estimates only. The 274 rows cluster within 112 tasks,
the panel includes six model comparisons, and it was not designed as a
multiplicity-adjusted confirmatory analysis.

## Current non-closures

The following remain explicit release boundaries:

1. The exact 5,784-row training output is not released. The released training
   file has 5,775 rows and lacks 6 `rename_declaration` and 3 `width_mismatch`
   pairs relative to it.
2. Historical raw-source snapshots are not fully pinned, so rebuilding the
   dataset produces a newly certified dataset rather than a byte-identical
   historical copy.
3. Several historical results retain aggregate/Boolean evidence but not all
   candidate RTL and proof trees needed for independent Level-B re-scoring.
4. Hosted-model aliases do not consistently identify immutable revisions;
   reruns are Level-C replications.
5. Upstream and provider terms for all redistributed HDL and model-generated
   records still require final review; see `docs/RELEASE_CHECKLIST.md` and
   `THIRD_PARTY_NOTICES.md`.

These limitations are reflected in the paper, checklist, license scope, and
`claims.json` rather than hidden behind a passing build.

## Verification contract

From a fresh anonymous export:

```bash
make doctor
make verify
make paper
```

`make verify` checks the allowlist and pinned inputs, runs public-compatible
tests, reconstructs Level-A outputs, and tests a fresh exported tree. `make
paper` builds the two current figures and the single poster source, then checks
the 3–4 page main-text limit and presence of the official NeurIPS checklist.
