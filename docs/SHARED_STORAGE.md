# Shared storage contract

Large experiment outputs must not be passed through chat, committed to ordinary Git, or left only on one collaborator's laptop. This document defines a storage-backend-neutral contract for sharing them.

No storage location has been provisioned or recorded in this repository yet. Accordingly, this document contains no made-up bucket name, object URI, or model revision. Add those values only after the corresponding object exists and has been read back successfully.

## Local materialization root

Every script and collaborator uses one environment variable:

```bash
export RTLREPAIR_ARTIFACTS="/absolute/path/to/materialized/rtlrepair-artifacts"
```

`RTLREPAIR_ARTIFACTS` is always a local filesystem directory. It may be a shared mounted directory, or it may be a local cache populated from remote object storage. Experiment code should consume files below this directory and should not embed provider-specific remote URIs.

Requirements:

- The value must be absolute.
- The directory must not be inside the Git worktree.
- Scripts must fail if the variable is unset when an external artifact is required.
- Scripts must not silently fall back to a developer-specific path.
- The directory must never contain API keys or credential files.

## Logical layout

Use stable logical paths independent of the storage provider:

```text
$RTLREPAIR_ARTIFACTS/
  primary/
    internal/
    anonymous/
  repairbench/
    historical/
    candidates/
  capability/
    candidates/
    formal-work/
  realbug/
    candidates/
  largescale/
    legacy/
    common-verifier/
  audits/
    contract-robustness/
    tier-s-rescore/
    formal-coverage/
    budget/
  runs/
    <study>/<run-id>/
  cache/
```

`internal/` may contain author-only identifiers and raw ledgers. `anonymous/` contains only the separately audited double-blind release. Never derive access policy from the directory name alone; the manifest's access field is authoritative.

## What belongs in shared storage

Store these outside Git:

- Raw and sanitized primary ledgers, normalized rows, strict statistics, and the deterministic anonymous supplement archive.
- Candidate RTL, golden RTL, equivalence miters, SBY work directories, proof logs, replay traces, and batch-validation outputs.
- Capability-curve and redaction work directories omitted from the tracked `generated/second_model` summaries.
- Historical or regenerated RepairBench candidates needed for independent re-scoring.
- Six-model and larger-design candidate directories and complete provider response/status/usage ledgers.
- The newer larger-design common-verifier rows, summary, protocol description, protocol hash, and its source-controlled driver snapshot.
- Raw dataset snapshots only when their licenses permit redistribution.
- Model checkpoints or adapter weights only when their licenses permit redistribution and the project actually controls the object.
- Deterministic archives of any directory with many small files.

Small, reviewable metadata should remain in Git:

- `claims.json`.
- Analysis and verification source code.
- Frozen protocol, dataset, and toolchain manifests.
- Compact sanitized per-case verdicts and summary JSON where policy permits.
- Generated LaTeX table/figure fragments.
- The external-artifact manifest and its checksum sidecar once real objects exist.

Do not upload or commit:

- API keys, bearer tokens, cookies, credential files, authorization headers, or environment dumps.
- Presigned download URLs or any URI containing a credential or query-string secret.
- Git credentials, private remotes, home-directory paths, email addresses, or author-identifying metadata in the anonymous release.
- Third-party datasets or model weights whose licenses prohibit redistribution.
- Unredacted provider responses in a public or double-blind artifact unless they have passed the release audit.

## Manifest location

After the first real upload, create:

```text
artifacts/manifest.json
artifacts/MANIFEST.sha256
```

Until then, do not commit a template containing fictional object locations. The manifest is small and may be tracked once populated. A private deployment may keep a separate restricted manifest with internal-object URIs; the public manifest must contain only public-release objects.

## Normative manifest shape

The manifest is UTF-8 JSON with this shape. The fragment below is a type contract, not an artifact entry and contains no URI value:

```json
{
  "schema_version": "integer, currently 1",
  "artifact_set": "non-empty string",
  "created_at_utc": "RFC 3339 UTC timestamp",
  "source_commit": "full Git commit hash",
  "objects": [
    {
      "logical_path": "safe relative POSIX path",
      "uri": "non-empty immutable object URI assigned after upload",
      "size_bytes": "non-negative integer",
      "sha256": "64 lowercase hexadecimal characters",
      "media_type": "non-empty media type",
      "role": "non-empty human-readable role",
      "producer": "repository path plus optional command or version",
      "runtime": "exact implementation and version when output depends on it",
      "source_commit": "full Git commit hash for this object",
      "access": "public, restricted, or private",
      "contains_secrets": false,
      "content_encoding": "identity or a declared archive/compression encoding",
      "members_manifest": "optional logical path of a manifest for archive members"
    }
  ]
}
```

Validation rules:

1. `logical_path` is unique, relative, uses `/`, and contains no empty, `.` or `..` segment.
2. `uri` identifies an immutable object or an explicitly versioned generation. A mutable “latest” object is invalid.
3. `size_bytes` is the exact byte length returned after upload and read-back, not an estimate.
4. `sha256` is computed over the exact stored bytes. For a compressed archive, hash the compressed archive; use `members_manifest` to hash unpacked members.
5. `source_commit` is the commit used to produce the artifact. If the worktree was dirty, the run metadata must also include a patch hash; do not pretend the commit alone identifies the code.
6. `contains_secrets` must be exactly `false`. An object that cannot satisfy this rule is not admissible.
7. `producer` names code that exists in source control. If the producer is missing, the object may be archived for evidence but its claim remains `incomplete` or `stale`.
8. An object must not appear in the manifest until upload, server-side persistence, download/read-back, byte-size verification, and SHA-256 verification all succeed.
9. `runtime` is mandatory for stochastic or version-sensitive analysis. The frozen primary bootstrap must record CPython 3.9.6; a generic `python3` label is insufficient.

`MANIFEST.sha256` contains the SHA-256 of the canonical manifest bytes. It is not a self-referential field inside `manifest.json`.

## Publishing workflow

Use this sequence for every object:

1. Freeze the source commit, protocol, inputs, and run ID.
2. Write into a new staging directory, never into an existing frozen run.
3. Scan for secrets, absolute host paths, author identifiers, malformed records, missing scheduled cells, and duplicate IDs.
4. For a directory, create a deterministic archive with sorted members and normalized metadata, or create a member-level checksum manifest.
5. Compute local byte size and SHA-256.
6. Upload under an immutable/versioned remote object name.
7. Read the stored object back through the normal collaborator access path.
8. Verify byte size and SHA-256 again.
9. Add the real URI and verified metadata to the appropriate manifest.
10. Review and commit the manifest change separately from experiment code.

Never overwrite an object referenced by a manifest. A corrected artifact gets a new object, new hash, new run ID, and a documented supersession relation.

## Fetch and verification workflow

A consumer must:

1. Select a manifest entry by exact `logical_path`.
2. Reject the entry if its access class is inappropriate or its URI is not immutable/versioned.
3. Download to a temporary file under `$RTLREPAIR_ARTIFACTS/cache/`.
4. Check exact byte size.
5. Compute and check SHA-256 before extraction or use.
6. If archived, extract into a new temporary directory, reject path traversal and links that escape the directory, and verify the member manifest.
7. Atomically rename the verified object or directory to its logical materialization path.
8. Record the manifest hash in the local run metadata.

A file that merely has the expected name is not trusted. Size and SHA-256 are mandatory.

## Concurrent runs

Shared storage is not a scratch directory. To prevent collaborators and agents from overwriting one another:

- Use `runs/<study>/<run-id>/`, where `run-id` is unique and fixed before the first request.
- Never use `latest`, a model name alone, or a paper table number as a writable run directory.
- Write attempt-local files first and publish a terminal run manifest only after all scheduled records have exactly one terminal state.
- Treat a published terminal run manifest as immutable.
- Keep retries in the same run ledger with explicit attempt numbers; do not replace the failed attempt.
- Store model-generated candidates separately from verifier outputs and bind them by candidate SHA-256.

## Artifact sets needed by this paper

The first useful manifests should cover these sets:

| Artifact set | Minimum contents | Access |
|---|---|---|
| `primary-internal` | Raw/internal ledger, normalized full85 and clean75 rows, frozen stats, CPython 3.9.6 runtime binding, candidate/proof tree, calibration and validation rows | private or restricted |
| `primary-anonymous` | Audited deterministic supplement archive, archive inventory, archive checksum | public or review-only |
| `capability-rescore` | Per-model candidate/golden/miter work trees and formal logs corresponding to tracked verdicts | restricted unless audited |
| `repairbench-candidates` | Candidate RTL and response metadata for rerunnable historical/new Tier-S rows | restricted unless audited |
| `realbug-candidates` | Six-model candidate RTL, generation completeness ledger, and simulation logs | restricted unless audited |
| `largescale-common` | 20 inputs, 100 scheduled rows, 18-case supported-set audit, protocol hash, common-verifier outputs, producer snapshot | restricted until reconciled |
| `formal-audits` | Contract-robustness sources/logs, E4 work trees, E5 inputs, toolchain metadata | restricted or public after audit |

No URI is currently asserted for any set above. Populate only after an actual backend and access policy are chosen.

## Release gates

An artifact set is publishable only when:

- Its manifest passes the contract above.
- Every claim that references it names its exact logical paths.
- Scheduled counts and paired cohorts are validated before aggregation.
- The code producer is tracked and the source commit is recorded.
- The object passes credential and identity scans appropriate to its access tier.
- The double-blind artifact contains no author-identifying repository link or metadata.
- A second collaborator can materialize and verify it using only the manifest and their authorized storage credentials.

This contract makes the storage backend replaceable. The scientific identity of an artifact is its logical path, exact bytes, size, SHA-256, producer, and source commit, not the vendor that stores it.
