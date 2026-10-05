# Data included in the reproducibility snapshot

This directory contains the small, versioned inputs needed to inspect and
score RepairBench without depending on the parent monorepo. It deliberately
does not contain model weights, raw model responses, generated proof trees,
waveforms, or the source corpora used to rebuild the training set.

## Benchmark inputs

| File | Rows | Purpose |
|---|---:|---|
| `repairbench_heldout.jsonl` | 459 | Full synthetic benchmark, 374 lint and 85 semantic cases |
| `repairbench_lint.jsonl` | 374 | Lint-class cases |
| `repairbench_sem85.jsonl` | 85 | Canonical semantic cases |
| `repairbench_realbugs_fmt.jsonl` | 274 | Prompt-ready mined real-bug cases |
| `repairbench_realbugs.jsonl` | 274 | Structured mined real-bug records |
| `eval_tasks.jsonl` | 155 | Specifications used by spec-conditioned arms |
| `eval_tasks_clean.jsonl` | 137 | Decontaminated evaluation subset |
| `rtlrepairdataset_train.jsonl` | 5,775 | RTLRepairDataset training pairs (chat format; 5,394 lint, 381 semantic) |
| `train.sample.jsonl` | 50 | Schema sample of spec2rtl rows from the SFT pipeline |

`formal/canonical_goldens/` contains 52 reference modules used by the semantic
and formal validation lanes. The adjacent JSON manifests define the supported
protocol and toolchain snapshot.

The remaining JSONL files are documented fixtures or subsets used by existing
analysis scripts. They are retained because they are small and because removing
them would silently change old command behavior.

## Training file

`rtlrepairdataset_train.jsonl` holds 5,775 rows, 10,135,382 bytes, SHA-256
`7c0c2004bfcb23e2e2fb6c7eb414e8d4824813db69d42378b526d487d6cc8150`. It is the `repair` bucket of
`sft_train_v3.dec.jsonl` (SHA-256 `356dde1a0f5e91a683cd59c0339e329708800adbb063e1ca2b8fe19624d53b6f`) from the Fairchild RTL Coder SFT
mixture on Hugging Face, copied line for line. Each row is `{messages, source}`:
the system message names the bug class, the user message gives the broken module
(with Verilator's output for lint bugs), and the assistant message is the
unified diff that repairs it. Per-operator counts, from classifying each row's
diff, against Table 3 of the paper:

| operator | released | Table 3 |
|---|---:|---:|
| `drop_endmodule` | 1,000 | 1,000 |
| `delete_semicolon` | 1,000 | 1,000 |
| `drop_one_end` | 1,000 | 1,000 |
| `rename_declaration` | 994 | 1,000 |
| `blocking_in_seq` | 1,000 | 1,000 |
| `width_mismatch` | 400 | 403 |
| `op_swap` | 300 | 300 |
| `flip_reset_polarity` | 81 | 81 |
| total | 5,775 | 5,784 |

Table 3 describes the historical combined output: 5,784 rows (5,403 lint and
381 semantic), 10,814,575 bytes, SHA-256 `73fd32ae2e3a1d419691fc16a67965aa929bc912155edc10c155895eac36518a`. That exact file is not
released. The released file has 1,892 distinct seed modules; the historical file
has 1,895 distinct normalized seed identifiers. Seeds come from MG-Verilog; see
`../NOTICE`.

## What is external

Rebuilding the training corpus also needs local copies of MG-Verilog,
VerilogEval, and RTLLM. Those upstream corpora are not vendored here. Point the
dataset pipeline at them explicitly; see `../REPRODUCE.md`.

Large run outputs follow the artifact contract in
[`../docs/SHARED_STORAGE.md`](../docs/SHARED_STORAGE.md). Their manifest entries
must record a content hash and provenance; a filename alone is not enough.

## Release review

The records are derived from multiple upstream research datasets. Before a
public release, verify each upstream dataset's current redistribution terms and
complete the license fields in the release checklist. Inclusion in this
research snapshot is not a declaration that every upstream license permits
unrestricted redistribution. See `../THIRD_PARTY_NOTICES.md` for the recorded
upstream revisions, declared licenses, and remaining provenance boundary.
