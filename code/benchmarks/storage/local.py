"""Write, read and index banked runs on the local disk.

Source-agnostic by construction: `put_run` takes the result blob, the candidate
ledger and the raw completions as *values or paths*, so an arm inferred on
Modal, on a laptop, or against a hosted API banks through the same call.

A run may bank a `result` (scored), a `ledger` (generated, not yet scored), or
both. Generation and scoring are separable: a ledger-only run is a complete,
publishable artifact that any scorer can consume later.

Writes are once-only. A banked arm is an input to a published table, so
`put_run` refuses to overwrite unless asked explicitly — the guarantee the
`version_id` component exists to provide.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from typing import Any, Iterable

from .layout import BATCHED_PREFIX, RunKey, default_version_id

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(HERE)

RESULT_FILE = "result.json"
RUN_FILE = "run.json"
LEDGER_FILE = "candidates.jsonl"
RAW_DIR = "raw"
INDEX_FILE = "index.json"


def out_root(root: str | None = None) -> str:
    """Root the tree hangs off: explicit arg, else $RTLREPAIR_STORE, else the bench dir."""
    return root or os.environ.get("RTLREPAIR_STORE") or BENCH_DIR


def batched_root(root: str | None = None) -> str:
    return os.path.join(out_root(root), *BATCHED_PREFIX)


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _write_raw(raw: Any, dest_dir: str) -> list[str]:
    """Materialize `raw` into dest_dir/. Returns the basenames written.

    Accepts a path to a file or directory, a list of paths, an iterable of
    dicts (written as completions.jsonl), or a pre-serialized str/bytes.
    """
    if raw is None:
        return []
    os.makedirs(dest_dir, exist_ok=True)

    if isinstance(raw, (str, os.PathLike)) and os.path.isdir(raw):
        raw = [os.path.join(raw, n) for n in sorted(os.listdir(raw))
               if os.path.isfile(os.path.join(raw, n))]
    elif isinstance(raw, (str, os.PathLike)) and os.path.isfile(raw):
        raw = [raw]
    elif isinstance(raw, (str, bytes)):
        # already-serialized JSONL text, not a path
        name = "completions.jsonl"
        mode, data = ("wb", raw) if isinstance(raw, bytes) else ("w", raw)
        with open(os.path.join(dest_dir, name), mode) as fh:
            fh.write(data)
        return [name]

    written: list[str] = []
    if isinstance(raw, (list, tuple)) and raw and all(
            isinstance(p, (str, os.PathLike)) and os.path.isfile(p) for p in raw):
        for src in raw:
            name = os.path.basename(src)
            shutil.copy2(src, os.path.join(dest_dir, name))
            written.append(name)
        return written

    # iterable of records -> one JSONL
    records = list(raw) if isinstance(raw, Iterable) else [raw]
    if not records:
        return []
    name = "completions.jsonl"
    with open(os.path.join(dest_dir, name), "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return [name]


def _ledger_records(ledger: Any) -> list[dict]:
    """Normalize `ledger` (path to JSONL, JSONL text, or iterable of dicts) to records."""
    if isinstance(ledger, (str, os.PathLike)) and os.path.isfile(ledger):
        with open(ledger) as fh:
            text = fh.read()
    elif isinstance(ledger, bytes):
        text = ledger.decode()
    elif isinstance(ledger, str):
        text = ledger
    else:
        return [dict(r) for r in ledger]
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError as exc:
            raise ValueError(f"ledger line {n} is not valid JSON: {exc}") from exc
    return out


def validate_ledger(records: list[dict]) -> None:
    """Reject a ledger that cannot be joined back to cases.

    A ledger exists so a scorer can rebuild the proof obligation for each case
    long after generation. That requires exactly one thing of the format: every
    record carries a unique, non-empty `case_id`. The banked
    `vr_raw_vrqwen.jsonl` is what this guards against — an append-mode dump with
    no case key that silently fused two arms into 918 unsplittable lines.
    """
    if not records:
        raise ValueError("ledger is empty; a run with no candidates is not bankable")
    seen: dict[str, int] = {}
    for n, rec in enumerate(records, 1):
        if not isinstance(rec, dict):
            raise ValueError(f"ledger line {n} is not a JSON object")
        cid = rec.get("case_id")
        if not cid or not str(cid).strip():
            raise ValueError(
                f"ledger line {n} has no case_id. Without it the candidate cannot be "
                f"joined to a golden and the run is unscoreable.")
        if cid in seen:
            raise ValueError(
                f"ledger has duplicate case_id {cid!r} (lines {seen[cid]} and {n}). "
                f"Two arms appended to one file, or a case was generated twice — "
                f"bank them as separate runs.")
        seen[cid] = n


def _write_ledger(ledger: Any, dest_dir: str) -> dict | None:
    """Validate and materialize the candidate ledger. Returns its run.json block."""
    if ledger is None:
        return None
    records = _ledger_records(ledger)
    validate_ledger(records)
    os.makedirs(dest_dir, exist_ok=True)
    path = os.path.join(dest_dir, LEDGER_FILE)
    with open(path, "w") as fh:
        for rec in records:
            fh.write(json.dumps(rec) + "\n")
    return {
        "file": LEDGER_FILE,
        "sha256": _sha256(path),
        "bytes": os.path.getsize(path),
        "n_candidates": len(records),
        "n_parsed": sum(1 for r in records if r.get("parsed_ok")),
        "n_errors": sum(1 for r in records if r.get("error")),
    }


def _load_result(result: Any) -> dict:
    if isinstance(result, (str, os.PathLike)):
        with open(result) as fh:
            return json.load(fh)
    if not isinstance(result, dict):
        raise TypeError("result must be a dict or a path to a JSON file")
    return result


def put_run(key: RunKey, result: Any = None, raw: Any = None,
            ledger: Any = None, meta: dict | None = None,
            root: str | None = None, overwrite: bool = False) -> str:
    """Bank one run. Returns the directory written.

    At least one of `result` or `ledger` is required:

    * `ledger` — the candidate ledger, one record per model call (path to a
      JSONL, JSONL text, or an iterable of dicts). This is the *generation*
      artifact: it is what a scorer reads, and it is validated on the way in
      (see `validate_ledger`). A run banked with a ledger alone is complete.
    * `result` — the {"summary", "records"} blob (dict or path to rb_*.json).
      This is a *scoring* artifact, derived from the ledger by an oracle.

    `raw` is **legacy only**: unkeyed completion dumps from before the ledger
    existed, kept so `storage migrate` can still import them. The tree holds
    none — the last, `vr_raw_vrqwen.jsonl`, was deleted as unjoinable. This path
    bypasses `validate_ledger`, which is exactly how that file (918 lines, two
    arms fused, no case key) became unsplittable. New runs put completions in
    the ledger, where every record carries a `case_id`.
    `meta` is free-form provenance merged into run.json — arm name, endpoint,
    sampling params, prompt identity, anything needed to reproduce the cell.
    Never put a credential in `meta`: run.json is pushed to the dataset repo.
    """
    if result is None and ledger is None:
        raise ValueError(
            "nothing to bank: pass ledger= (generation) or result= (scoring), or both")
    dest = key.path(out_root(root))
    result_path = os.path.join(dest, RESULT_FILE)
    # Write-once covers either artifact: a ledger-only run is a real run, and a
    # later scoring pass must not be able to clobber the candidates it scored.
    existing = [p for p in (result_path, os.path.join(dest, LEDGER_FILE))
                if os.path.exists(p)]
    if existing and not overwrite:
        raise FileExistsError(
            f"{key} is already banked at {existing[0]}. Bank re-runs under a new "
            f"version_id (RTLREPAIR_RUN_VERSION=...), or pass overwrite=True if "
            f"you really mean to replace a banked arm."
        )
    # Validate before destroying anything: a bad ledger must not cost the old run.
    ledger_records = _ledger_records(ledger) if ledger is not None else None
    if ledger_records is not None:
        validate_ledger(ledger_records)

    if os.path.isdir(dest) and overwrite:
        shutil.rmtree(dest)
    os.makedirs(dest, exist_ok=True)

    blob = _load_result(result) if result is not None else None
    if blob is not None:
        with open(result_path, "w") as fh:
            json.dump(blob, fh, indent=2)

    ledger_block = _write_ledger(ledger_records, dest)

    # No empty raw/ when an arm dumped nothing: git does not track empty
    # directories, so the "uniform tree shape" it used to create never survived
    # a checkout anyway. A run with no raw/ is the normal case now.
    raw_names = _write_raw(raw, os.path.join(dest, RAW_DIR))

    run = {
        "model": key.model,
        "version": key.version,
        "signal": key.signal,
        "summary": (blob or {}).get("summary", {}),
        # Record count is the scored records when scored, else the ledger size,
        # so `ls`/`index` report something honest for a generation-only run.
        "n_records": (len(blob.get("records", []) or []) if blob is not None
                      else (ledger_block or {}).get("n_candidates", 0)),
        "scored": blob is not None,
        "ledger": ledger_block,
        "raw": [{"file": n, "sha256": _sha256(os.path.join(dest, RAW_DIR, n)),
                 "bytes": os.path.getsize(os.path.join(dest, RAW_DIR, n))}
                for n in raw_names],
        **(meta or {}),
    }
    with open(os.path.join(dest, RUN_FILE), "w") as fh:
        json.dump(run, fh, indent=2)
    return dest


def get_run(key: RunKey, root: str | None = None) -> dict:
    """Read back one banked run: {'dir', 'result', 'run', 'raw', 'ledger'}.

    `result` is None for a generation-only run; `ledger` is the path to
    candidates.jsonl, or None for a run banked before the split.
    """
    d = key.path(out_root(root))
    result_path = os.path.join(d, RESULT_FILE)
    ledger_path = os.path.join(d, LEDGER_FILE)
    if not (os.path.exists(result_path) or os.path.exists(ledger_path)):
        raise FileNotFoundError(f"{key} is not banked at {d}")
    result = None
    if os.path.exists(result_path):
        with open(result_path) as fh:
            result = json.load(fh)
    run = {}
    if os.path.exists(os.path.join(d, RUN_FILE)):
        with open(os.path.join(d, RUN_FILE)) as fh:
            run = json.load(fh)
    raw_dir = os.path.join(d, RAW_DIR)
    raw = ([os.path.join(raw_dir, n) for n in sorted(os.listdir(raw_dir))]
           if os.path.isdir(raw_dir) else [])
    return {"dir": d, "result": result, "run": run, "raw": raw,
            "ledger": ledger_path if os.path.exists(ledger_path) else None}


def read_ledger(key: RunKey, root: str | None = None) -> list[dict]:
    """The candidate records for one banked run — the scorer's input.

    Raises if the run has no ledger, which is the honest answer for the eight
    arms banked before generation and scoring were split: their candidates were
    never written, so they can only be regenerated, not rescored.
    """
    path = get_run(key, root)["ledger"]
    if path is None:
        raise FileNotFoundError(
            f"{key} has no {LEDGER_FILE}: it was banked without a candidate ledger, "
            f"so there is no RTL to rescore. Regeneration is the only path.")
    return _ledger_records(path)


def list_runs(root: str | None = None) -> list[RunKey]:
    """Every banked run under the tree, sorted."""
    base = batched_root(root)
    found: list[RunKey] = []
    if not os.path.isdir(base):
        return found
    for model in sorted(os.listdir(base)):
        for version in sorted(_subdirs(os.path.join(base, model))):
            for signal in sorted(_subdirs(os.path.join(base, model, version))):
                d = os.path.join(base, model, version, signal)
                # Either artifact makes it a run: generation-only runs are listed
                # too, or a banked ledger would be invisible until someone scored it.
                if any(os.path.exists(os.path.join(d, f))
                       for f in (RESULT_FILE, LEDGER_FILE)):
                    found.append(RunKey(model, version, signal))
    return found


def _subdirs(path: str) -> list[str]:
    if not os.path.isdir(path):
        return []
    return [n for n in os.listdir(path) if os.path.isdir(os.path.join(path, n))]


def write_index(root: str | None = None) -> str:
    """Regenerate results/batched/index.json — one row per banked run.

    Makes the dataset browsable without walking the tree (and readable straight
    off the HF repo).
    """
    base = batched_root(root)
    os.makedirs(base, exist_ok=True)
    rows = []
    for key in list_runs(root):
        run = get_run(key, root)
        ledger = run["run"].get("ledger") or {}
        rows.append({
            "path": key.as_posix(),
            "model": key.model,
            "version": key.version,
            "signal": key.signal,
            "n_records": run["run"].get("n_records"),
            "arm": run["run"].get("arm"),
            # Whether this run can be rescored, and whether it has been scored —
            # the two questions the index exists to answer without a tree walk.
            "has_ledger": run["ledger"] is not None,
            "n_candidates": ledger.get("n_candidates"),
            "n_parsed": ledger.get("n_parsed"),
            "scored": run["result"] is not None,
            "by_bucket": (run["run"].get("summary") or {}).get("by_bucket", {}),
            "raw_files": len(run["raw"]),
        })
    path = os.path.join(base, INDEX_FILE)
    with open(path, "w") as fh:
        json.dump({"runs": rows}, fh, indent=2)
    return path


def materialize_flat(dest: str, root: str | None = None,
                     version: str | None = None) -> list[str]:
    """Write the flat `rb_<arm>.json` view that analyze_repairbench.py globs.

    analyze_repairbench globs `$RTLREPAIR_OUT/rb_*.json` **non-recursively**, so
    the tree is invisible to it. This projects a chosen version back down to
    that shape; point RTLREPAIR_OUT at `dest` and analysis runs unchanged.

    With no `version`, the newest version_id present per (model, signal) wins.
    Generation-only runs are skipped: analyze_repairbench reads scored records,
    and an unscored run has none to project.
    """
    os.makedirs(dest, exist_ok=True)
    latest: dict[tuple[str, str], RunKey] = {}
    for key in list_runs(root):
        if version and key.version != version:
            continue
        if get_run(key, root)["result"] is None:
            continue
        slot = (key.model, key.signal)
        if slot not in latest or key.version > latest[slot].version:
            latest[slot] = key

    written = []
    for key in latest.values():
        run = get_run(key, root)
        arm = run["run"].get("arm") or f"{key.model}_{key.signal}"
        path = os.path.join(dest, f"rb_{arm}.json")
        with open(path, "w") as fh:
            json.dump(run["result"], fh, indent=2)
        written.append(path)
    return sorted(written)
