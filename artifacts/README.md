# External artifacts

This directory contains only the small metadata needed to locate and verify large experiment artifacts. Large datasets, candidate RTL trees, proof work directories, raw ledgers, and model outputs belong outside Git under `RTLREPAIR_ARTIFACTS`.

The complete storage and security contract is [`../docs/SHARED_STORAGE.md`](../docs/SHARED_STORAGE.md). Result-to-artifact mappings are in [`../claims.json`](../claims.json), with the three reproduction levels documented in [`../docs/RESULTS_AND_PROVENANCE.md`](../docs/RESULTS_AND_PROVENANCE.md).

## Current state

No shared-storage object URI has been verified and recorded here yet. Therefore `manifest.json` and `MANIFEST.sha256` are intentionally absent. Do not add placeholder bucket names, fabricated URIs, expiring presigned links, or guessed model revisions.

After real objects have been uploaded and read back successfully, this directory should contain:

```text
manifest.json
MANIFEST.sha256
README.md
```

Every manifest object must include a unique logical path, immutable/versioned URI, exact byte size, SHA-256 of the stored bytes, access class, producer, and source commit. Consumers must verify size and SHA-256 before using or extracting an object.

## Local setup

Point the environment variable at an absolute directory outside the Git worktree:

```bash
export RTLREPAIR_ARTIFACTS="/absolute/path/to/materialized/rtlrepair-artifacts"
```

This is a materialization/cache root, not a credential. Authentication for the chosen storage backend stays in the operator's approved credential mechanism and must never be copied into this directory, a manifest, or a run ledger.

## First objects to publish

Priority order:

1. The private primary ledger, normalized rows, frozen statistics, retained candidate/proof tree, and the separately audited anonymous supplement archive.
2. Capability-curve candidate/formal work trees needed to re-score the tracked verdicts,
   including row-level outcomes and candidates behind the Claude Opus 4.8 second-anchor aggregate.
3. Candidate RTL and complete response metadata for RepairBench and the six-model real-bug runs.
4. The larger-design common-verifier producer, 100 row-level results,
   supported-set audit, and protocol hash needed to complete the disclosed
   20-case versus 18-case sensitivity provenance.
5. Contract-robustness and Tier-S re-score work trees.

An upload is not complete until another collaborator can fetch it through the normal access path and reproduce its recorded byte size and SHA-256.
