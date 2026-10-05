"""Sync the banked tree to a HuggingFace **dataset** repo.

The repo mirrors the local layout exactly — `results/batched/{model}/{version}/
{signal}/` at the repo root — so a path is the same string in both places and
nothing has to be translated when citing a run.

The repo is git-backed, so every push is a commit: the version history is the
audit trail, and a private repo flips to public at submission without moving a
byte.

Auth: `huggingface-cli login`, or $HF_TOKEN. Repo id from $RTLREPAIR_HF_REPO.
"""

from __future__ import annotations

import os

from .layout import BATCHED_PREFIX, RunKey
from .local import batched_root, list_runs, out_root, write_index

CARD = """---
license: mit
tags:
  - verilog
  - rtl-repair
  - llm-evaluation
---

# RepairBench results

Banked candidate RTL, eval results and **raw model completions** for the
RepairBench lint-vs-semantic repair matrix.

## Layout

```
results/batched/{model_name}/{version_id}/{signal_type}/
├── candidates.jsonl  # one record per model call — the generation artifact
├── result.json       # {"summary": ..., "records": ...} — per-item recovery
└── run.json          # provenance: arm, config, git sha, sha256s
```

Completions live in `candidates.jsonl`, keyed by `case_id`. Earlier unkeyed
`raw/*.jsonl` dumps have been removed — they could not be joined back to cases.

- **model_name** — model slot (`base`, `v5`, `claude-opus-4-8`, `origen-fix`,
  `verireason`, `verireason-default`, `vrqwen`).
- **version_id** — `YYYYMMDD-<git sha>` of the harness that produced the run.
  Runs are write-once; a re-run banks under a new `version_id`.
- **signal_type** — repair signal on top of the tool diagnostic: `diag`,
  `diag_spec`, `diag_locate`, `diag_spec_locate`.

## candidates.jsonl

Generation and scoring are separate. `candidates.jsonl` is what the model
produced; `result.json` is one oracle's reading of it. A run may carry either
or both — a run with a ledger and no result has been generated but not scored,
and can be scored by any oracle later without re-running the model.

| field | meaning |
|---|---|
| `case_id` | join key — unique and non-empty in every ledger |
| `raw` | the completion, verbatim |
| `extracted_rtl` | module pulled from `raw` by `extractor` |
| `extractor` | which extractor produced it (extraction is a scoring choice) |
| `parsed_ok` | whether a module was recovered |
| `finish_reason` | OpenAI finish reason — `length` means truncated |
| `error` / `error_type` | set when the call failed; distinguishes failure from an empty completion |
| `usage` | prompt/completion token counts, when the endpoint reports them |
| `prompt_sha256` | identity of the exact prompt sent |

`results/batched/index.json` lists every run — including whether it has a
ledger and whether it has been scored — without walking the tree.
"""


def _api():
    try:
        from huggingface_hub import HfApi
    except ImportError as exc:  # pragma: no cover - depends on env
        raise ImportError(
            "huggingface_hub is required to sync results. Install it with:\n"
            "    pip install 'huggingface_hub>=0.23'"
        ) from exc
    return HfApi()


def repo_id(explicit: str | None = None) -> str:
    rid = explicit or os.environ.get("RTLREPAIR_HF_REPO")
    if not rid:
        raise ValueError(
            "no dataset repo set. Pass repo=... or export "
            "RTLREPAIR_HF_REPO=<org>/repairbench-results"
        )
    return rid


def ensure_repo(repo: str | None = None, private: bool = True,
                refresh_card: bool = False) -> str:
    """Create the dataset repo if absent and seed its card. Idempotent.

    The card is written only when absent, so an existing repo keeps whatever
    card it has — pass `refresh_card=True` to overwrite it with CARD (needed
    after the layout it documents changes).
    """
    rid, api = repo_id(repo), _api()
    api.create_repo(rid, repo_type="dataset", private=private, exist_ok=True)
    if refresh_card or "README.md" not in set(
            api.list_repo_files(rid, repo_type="dataset")):
        api.upload_file(path_or_fileobj=CARD.encode(), path_in_repo="README.md",
                        repo_id=rid, repo_type="dataset",
                        commit_message="Update dataset card")
    return rid


def push(key: RunKey | None = None, repo: str | None = None,
         root: str | None = None, private: bool = True,
         message: str | None = None, refresh_card: bool = False) -> str:
    """Upload one run (`key`) or the whole tree (`key=None`). Returns the repo id.

    Refreshes index.json first so the manifest never lags the tree.
    """
    rid = ensure_repo(repo, private=private, refresh_card=refresh_card)
    api = _api()
    write_index(root)

    if key is None:
        api.upload_folder(
            folder_path=batched_root(root),
            path_in_repo="/".join(BATCHED_PREFIX),
            repo_id=rid, repo_type="dataset",
            commit_message=message or "Sync banked results",
        )
        return rid

    local = key.path(out_root(root))
    if not os.path.isdir(local):
        raise FileNotFoundError(f"{key} is not banked at {local}")
    api.upload_folder(folder_path=local, path_in_repo=key.as_posix(),
                      repo_id=rid, repo_type="dataset",
                      commit_message=message or f"Bank {key}")
    api.upload_file(
        path_or_fileobj=os.path.join(batched_root(root), "index.json"),
        path_in_repo="/".join(BATCHED_PREFIX) + "/index.json",
        repo_id=rid, repo_type="dataset", commit_message="Refresh index")
    return rid


def pull(key: RunKey | None = None, repo: str | None = None,
         root: str | None = None, revision: str | None = None) -> str:
    """Download one run or the whole tree into the local layout. Returns the dir.

    `revision` takes a commit sha / tag — how you retrieve the exact bytes a
    published table was computed from.
    """
    from huggingface_hub import snapshot_download

    rid = repo_id(repo)
    patterns = ([f"{key.as_posix()}/**"] if key
                else ["/".join(BATCHED_PREFIX) + "/**"])
    snapshot_download(repo_id=rid, repo_type="dataset", revision=revision,
                      allow_patterns=patterns, local_dir=out_root(root))
    return key.path(out_root(root)) if key else batched_root(root)


def list_remote(repo: str | None = None) -> list[RunKey]:
    """Runs present in the dataset repo, derived from result.json paths."""
    rid = repo_id(repo)
    prefix = "/".join(BATCHED_PREFIX) + "/"
    keys = []
    for f in _api().list_repo_files(rid, repo_type="dataset"):
        if not (f.startswith(prefix) and f.endswith("/result.json")):
            continue
        parts = f[len(prefix):].split("/")
        if len(parts) == 4:
            keys.append(RunKey(parts[0], parts[1], parts[2]))
    return sorted(keys, key=str)


def diff(repo: str | None = None, root: str | None = None) -> dict:
    """What is banked locally but not pushed, and vice versa."""
    local = {str(k) for k in list_runs(root)}
    remote = {str(k) for k in list_remote(repo)}
    return {"local_only": sorted(local - remote),
            "remote_only": sorted(remote - local),
            "both": sorted(local & remote)}
