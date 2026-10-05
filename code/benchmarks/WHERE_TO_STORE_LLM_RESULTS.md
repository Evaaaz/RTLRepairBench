# Where to store LLM results

One run = one directory. Same paths on disk and in the HF dataset repo.

```
results/batched/{model_name}/{version_id}/{signal_type}/
├── candidates.jsonl  # the ledger — one record per model call (generation)
├── result.json       # {"summary": ..., "records": ...} — recovery (scoring)
└── run.json          # provenance: arm, full config, sha256s, record counts
```

(No `raw/` — see [`raw/` is legacy and now empty](#raw-is-legacy-and-now-empty).)

**Generation and scoring are separate.** `candidates.jsonl` is what the model
produced; `result.json` is one oracle's reading of it. A run needs *either* to
be bankable, and a run with a ledger and no result is complete — it has been
generated, not yet scored, and any scorer can score it later without touching
the model. See [Generate a ledger](#generate-a-ledger).

| component | what it is | values today |
|---|---|---|
| `model_name` | model **slot**, not the served checkpoint name | `base`, `v5`, `claude-opus-4-8`, `origen-fix`, `verireason`, `verireason-default`, `vrqwen` |
| `version_id` | `YYYYMMDD-<git sha>` of the harness that produced it | `20260801-e6dddd8` |
| `signal_type` | levers on top of the tool diagnostic | `diag`, `diag_spec`, `diag_locate`, `diag_spec_locate` |

**Runs are write-once.** `put_run` refuses to overwrite a banked arm — re-runs go
under a new `version_id`. That is the whole point of the component.

**Prompt style has no path component**, so two arms can differ only by it
(`verireason_default_diag` vs `verireason_diag`). Those are separated in the
model slot via `storage.layout.ARM_MODEL_OVERRIDES`, and `python -m storage check`
fails if any two arms still collide. Run it after adding an arm.

## Generate a ledger

`generate_ledger.py` calls the model and banks the ledger. It does not score.

```bash
export RTLREPAIR_LLM_URL=https://<modal-app>.modal.run/v1
export RTLREPAIR_LLM_API_KEY=<the served endpoint's key>
python generate_ledger.py --model OriGen_Fix --slot origen-fix --bank
```

One run is banked per arm, keyed `(model_slot, version_id, arm)` — the arm name
(`spec0_loc0` … `spec1_loc1`) is the signal component, matching the identifiers
already in `generated/second_model/verdicts.jsonl`. The `diag*` grammar does not
apply to these: the cases lint and compile clean, so there is no diagnostic.

All 85 cases are generated, not the 68 formally-tractable subset — banking the
full set makes narrowing it a rescore rather than a regeneration.

### Ledger record

| field | why it must be recorded at generation time |
|---|---|
| `case_id` | the join key. Enforced unique and non-empty — see below |
| `raw` | source of truth; extraction can be redone, a lost completion cannot |
| `extracted_rtl`, `extractor` | what *this* run parsed, and with which extractor |
| `parsed_ok` | a function of the extractor, so it moves when the extractor does |
| `finish_reason` | `length` = truncated. Unrecoverable afterwards: a completion cut at max_tokens that still contains a whole module looks clean |
| `error`, `error_type` | distinguishes a 503 from an empty completion. Both leave `raw` unusable, but only one is the model's answer |
| `usage` | the only cost signal |
| `prompt_sha256`, `system_sha256` | which prompt variant produced this ledger |

`put_run` **rejects a ledger whose records lack a unique `case_id`.** That is the
check the banked `vr_raw_vrqwen.jsonl` would have failed: an append-mode dump
with no case key that fused two arms into 918 unsplittable lines.

Use `generate_strict`, never `generate` — the non-strict path silently returns
canned RTL from `serve_stub` on any failure, which would land in a ledger
indistinguishable from a real completion. The driver already does.

## Bank a run

Works for outputs from any source — Modal, a local vLLM, a hosted API, a
collaborator's machine. `result`, `ledger` and `raw` take values *or* paths.

```python
from storage import RunKey, put_run, push

key = RunKey(model="verireason", version="20260801-e6dddd8", signal="diag")
put_run(key,
        ledger="dumps/candidates.jsonl",            # or [dicts] — generation
        result="results/rb_verireason_diag.json",   # or the dict itself — scoring
        meta={"arm": "verireason_diag", "endpoint": "modal"})
push(key)   # -> HF dataset repo, identical paths
```

At least one of `result` / `ledger` is required. Never put a credential in
`meta`: `run.json` is pushed to the dataset repo.

`repairbench_eval.py` still writes a flat `rb_<arm>.json` into `$RTLREPAIR_OUT`;
that file is the *scratch* output of a run. Bank it, then it lives in the tree.

From the shell:

```bash
python -m storage put --arm verireason_diag --ledger candidates.jsonl --result rb_verireason_diag.json
```

## `raw/` is legacy and now empty

The ledger's `raw` field holds the completion verbatim, keyed by `case_id`, so
`raw/*.jsonl` has nothing left to store. **The tree contains no raw dumps.**

The last one, `vrqwen/…/{diag,diag_spec}/raw/vr_raw_vrqwen.jsonl`, was deleted:
an append-mode dump with no arm field and no case key, holding both vrqwen arms
fused (918 = 2×459) and unsplittable. Both arms even referenced the *same*
sha256 — one file, copied. It is recoverable from git `642cb1d` if ever needed;
those arms will be regenerated with a ledger instead.

`put_run(raw=...)` survives only because `storage migrate` imports old flat
`rb_<arm>.json` runs together with their dumps. It bypasses `validate_ledger`,
which is precisely how the vrqwen file became unusable. Do not write new dumps
through it.

`put_run` no longer creates an empty `raw/` for runs that have none — git does
not track empty directories, so the "uniform tree shape" that justified them
never survived a checkout.

## Reading a ledger back

```python
from storage import RunKey, read_ledger
records = read_ledger(RunKey("origen-fix", "20260801-e6dddd8", "spec0_loc0"))
```

Raises for the eight arms banked before the split: their candidates were never
written, so they can only be regenerated, not rescored.

## HuggingFace dataset repo

Git-backed, so each push is a commit and the history is the audit trail. Private
now, flip to public at submission without moving a byte. `--revision <sha>` pulls
the exact bytes a published table was computed from.

```bash
export RTLREPAIR_HF_REPO=<org>/repairbench-results
huggingface-cli login          # or export HF_TOKEN=...
pip install 'huggingface_hub>=0.23'

python -m storage push         # whole tree (creates the repo, private, on first push)
python -m storage push --refresh-card   # also overwrite the repo README
python -m storage pull --model v5 --version 20260801-e6dddd8 --signal diag
python -m storage diff         # what is banked locally but not pushed
```

The dataset card is written only when absent, so an existing repo keeps its old
card until `--refresh-card`. Pushing uploads raw completions — it is the one
step here that leaves your machine.

## Analysis still globs the flat layout

`analyze_repairbench.py` globs `$RTLREPAIR_OUT/rb_*.json` **non-recursively**, so
the tree is invisible to it. Project a version back down first:

```bash
python -m storage flatten --dest results/flat --version 20260801-e6dddd8
RTLREPAIR_OUT=results/flat python analyze_repairbench.py
```

## Migrating flat results into the tree

The banked 8-arm matrix was migrated under `20260801-e6dddd8`, and the flat
`rb_*.json` + shared `raw/` it came from have been deleted — the tree is the
only copy now (plus git history and, once pushed, the HF repo).

`migrate` remains for bulk-importing any other directory of flat `rb_<arm>.json`.
It **copies**, and writes nothing under `--dry-run`:

```bash
python -m storage migrate --src <dir> --dry-run
python -m storage migrate --src <dir> --version 20260801-e6dddd8
```

It attaches a raw dump only when exactly one arm can claim it. A dump named for
a *model* is ambiguous across that model's arms, and gets reported unattached
rather than guessed at — attach it explicitly with `put --raw`.

## Env vars

| var | default | what |
|---|---|---|
| `RTLREPAIR_STORE` | the `benchmarks/` dir | root the tree hangs off |
| `RTLREPAIR_HF_REPO` | — | `<org>/repairbench-results` |
| `RTLREPAIR_RUN_VERSION` | `YYYYMMDD-<git sha>` | override `version_id` for outputs produced elsewhere |
| `HF_TOKEN` | — | HF auth (or `huggingface-cli login`) |
