# Release checklist

This checklist is intentionally stricter than “the code runs on the authors'
machine.” A public repository and an anonymous review artifact are different
deliverables and must be checked separately.

Status for this review snapshot: the checked source, paper, Level-A, and
anonymous-export gates have been executed. Unchecked entries are genuine
licensing, external-artifact, toolchain, or camera-ready closures and must not
be inferred from a passing build.

## Scientific closure

- [x] Every current poster claim has a `claims.json` entry.
- [x] Every entry marked reproducible names an immutable released input or
      output hash.
- [x] `make reproduce` regenerates all supported tables and figures without a
      network connection or API key.
- [x] `make verify` compares regenerated outputs against the artifact lock.
- [x] Claims marked `incomplete` or `stale` are resolved in the paper or clearly
      disclosed. A successful build must never upgrade those statuses.
- [ ] Model reruns pin provider, model identifier, model revision when available,
      prompt, decoding parameters, seeds, and endpoint contract.
- [ ] Paid model calls require an explicit cost acknowledgement and write only to
      a new run directory.

## Data and licensing

- [x] Select a repository license. Apache-2.0 for the code, pipeline and result
      artifacts authored here; see `LICENSE`, whose third-party section scopes the
      grant away from VerilogEval/MG-Verilog-derived seeds and from redistributed
      third-party model generations.
- [ ] Review redistribution terms for MG-Verilog, VerilogEval, RTLLM, model
      outputs, and mined model-failure records.
- [ ] Record complete upstream versions, licenses, and required notices.
      `THIRD_PARTY_NOTICES.md` records the known license declarations, the MG
      snapshot, and abbreviated VerilogEval/RTLLM revisions; full immutable
      upstream manifests and per-sample review remain open.
- [ ] Add camera-ready citation metadata only after anonymity is lifted.

## Anonymous artifact

- [x] Export from the allowlist tool, never by archiving the monorepo or a fork
      with history.
- [x] Exclude `.git`, migration history, review notes, raw responses, reasoning
      traces, model credentials, local paths, caches, proof trees, waveforms,
      and filesystem metadata.
- [x] Scan source, PDF text, PDF metadata, filenames, and manifests for author or
      organization identifiers.
- [x] Keep the cleartext identity denylist and any keyed identity scanner outside
      the anonymous payload; do not encode low-entropy names or handles in public
      tests or source.
- [x] Scan for secrets, bearer tokens, private endpoints, and absolute paths.
- [x] Verify that the PDF still renders `Anonymous Author(s)` and has blank
      title/author/subject/keyword metadata fields.
- [ ] Generate `SHA256SUMS` after the final archive is created.

## Fresh-environment acceptance

- [ ] Install from the frozen environment description.
- [ ] Run `make doctor`, `make test`, `make smoke`, `make reproduce`, and
      `make verify` from a fresh exported directory under a different path.
- [ ] Run the toy RTL gate with real Verilator and Icarus tools.
- [ ] Run the formal toy gate with the checksum-pinned container.
- [x] Build the paper and check page count, references, warnings, and claim/table
      consistency.
- [ ] Confirm that the source checkout and exported artifact remain clean after
      all read-only verification commands.
