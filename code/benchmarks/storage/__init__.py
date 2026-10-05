"""Banked storage for eval results + raw model completions.

    from storage import RunKey, put_run, push

    key = RunKey(model="verireason", version="20260801-e6dddd8", signal="diag")
    put_run(key, result="results/batched/rb_verireason_diag.json",
            raw="results/batched/raw/vr_raw_verireason.jsonl",
            meta={"arm": "verireason_diag", "endpoint": "modal"})
    push(key)   # -> HF dataset repo, same paths

See WHERE_TO_STORE_LLM_RESULTS.md for the layout and the env vars.
"""

from .layout import (
    RunKey,
    check_no_collisions,
    default_version_id,
    model_name,
    signal_type,
)
from .local import (
    batched_root,
    get_run,
    list_runs,
    materialize_flat,
    out_root,
    put_run,
    read_ledger,
    validate_ledger,
    write_index,
)

__all__ = [
    "RunKey", "check_no_collisions", "default_version_id", "model_name",
    "signal_type", "put_run", "get_run", "list_runs", "write_index",
    "materialize_flat", "batched_root", "out_root", "read_ledger",
    "validate_ledger", "push", "pull", "list_remote", "diff", "ensure_repo",
]


def __getattr__(name):
    # HF sync is imported lazily so the local half works without huggingface_hub.
    if name in {"push", "pull", "list_remote", "diff", "ensure_repo"}:
        from . import hf
        return getattr(hf, name)
    raise AttributeError(name)
