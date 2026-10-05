"""CLI for the banked results store — `python -m storage <cmd>`.

    python -m storage migrate --dry-run     # flat rb_*.json -> the tree
    python -m storage put --arm verireason_diag --result r.json --raw raw.jsonl
    python -m storage ls
    python -m storage push                  # -> HF dataset repo
    python -m storage pull --model v5
    python -m storage flatten --dest results/flat   # view analyze_* can glob
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

from .layout import RunKey, check_no_collisions, default_version_id
from .local import (
    RESULT_FILE,
    batched_root,
    get_run,
    list_runs,
    materialize_flat,
    out_root,
    put_run,
    write_index,
)

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH_DIR = os.path.dirname(HERE)
CONFIG = os.path.join(BENCH_DIR, "configs", "experiments.json")


def _experiments(path: str = CONFIG) -> tuple[list[dict], dict]:
    with open(path) as fh:
        cfg = json.load(fh)
    defaults = cfg.get("defaults", {})
    exps = [e for e in cfg.get("experiments", []) if e.get("name")]
    return exps, defaults


def _arm(name: str, path: str = CONFIG) -> dict:
    exps, _ = _experiments(path)
    for e in exps:
        if e["name"] == name:
            return e
    raise SystemExit(f"no arm named {name!r} in {path}. "
                     f"Known: {', '.join(sorted(e['name'] for e in exps))}")


def _meta(exp: dict, defaults: dict) -> dict:
    cfg = {**defaults, **{k: v for k, v in exp.items() if not k.startswith("_")}}
    return {"arm": exp["name"], "config": cfg}


def cmd_put(args) -> int:
    exps, defaults = _experiments()
    if args.arm:
        exp = _arm(args.arm)
        key = RunKey.from_experiment(exp, version=args.version)
        meta = _meta(exp, defaults)
    else:
        if not (args.model and args.signal):
            raise SystemExit("pass --arm, or both --model and --signal")
        key = RunKey(args.model, args.version or default_version_id(), args.signal)
        meta = {"arm": args.name} if args.name else {}
    if args.note:
        meta["note"] = args.note
    if not (args.result or args.ledger):
        raise SystemExit("pass --ledger (generation), --result (scoring), or both")
    dest = put_run(key, result=args.result, raw=args.raw, ledger=args.ledger,
                   meta=meta, root=args.root, overwrite=args.overwrite)
    write_index(args.root)
    print(f"banked {key} -> {dest}")
    return 0


def cmd_migrate(args) -> int:
    """Copy the existing flat rb_<arm>.json (+ raw/) into the tree.

    Copies rather than moves: the banked flat files stay put until you delete
    them yourself, so analyze_repairbench keeps working mid-migration.
    """
    exps, defaults = _experiments()
    check_no_collisions(exps)
    src_dir = args.src or batched_root(args.root)
    version = args.version or default_version_id()
    by_name = {e["name"]: e for e in exps}

    flat = sorted(glob.glob(os.path.join(src_dir, "rb_*.json")))
    if not flat:
        print(f"no rb_*.json in {src_dir}", file=sys.stderr)
        return 1

    raw_files = sorted(glob.glob(os.path.join(src_dir, "raw", "*")))
    planned, skipped, unattached = [], [], list(raw_files)

    for path in flat:
        arm = os.path.basename(path)[len("rb_"):-len(".json")]
        exp = by_name.get(arm)
        if not exp:
            skipped.append((arm, "no matching arm in configs/experiments.json"))
            continue
        key = RunKey.from_experiment(exp, version=version)
        # Attach a raw dump only when exactly one arm can claim it — a dump named
        # for a model (vr_raw_vrqwen.jsonl) is ambiguous across that model's arms.
        claimants = [e for e in exps if _model_token(e) == _model_token(exp)]
        mine = [r for r in raw_files if _model_token(exp) in os.path.basename(r)]
        raw = mine if (mine and len(claimants) == 1) else None
        if raw:
            for r in raw:
                if r in unattached:
                    unattached.remove(r)
        planned.append((arm, key, path, raw or []))

    for arm, key, path, raw in planned:
        rel = os.path.relpath(key.path(out_root(args.root)), os.getcwd())
        print(f"{'would bank' if args.dry_run else 'banked'} {arm:24s} -> {rel}"
              + (f"  (+{len(raw)} raw)" if raw else ""))
        if not args.dry_run:
            put_run(key, result=path, raw=raw, meta=_meta(by_name[arm], defaults),
                    root=args.root, overwrite=args.overwrite)

    for arm, why in skipped:
        print(f"skipped {arm}: {why}", file=sys.stderr)
    for r in unattached:
        print(f"unattached raw dump: {os.path.relpath(r, os.getcwd())} — "
              f"attach it with `put --arm <arm> --result <r> --raw {r}`",
              file=sys.stderr)

    if not args.dry_run:
        print("index:", write_index(args.root))
    else:
        print("\n(dry run — nothing written; drop --dry-run to bank)")
    return 0


def _model_token(exp: dict) -> str:
    raw = (exp.get("claude_model") if exp.get("backend") == "claude"
           else exp.get("model")) or "base"
    return str(raw).lower()


def cmd_ls(args) -> int:
    keys = list_runs(args.root)
    if not keys:
        print(f"nothing banked under {batched_root(args.root)}")
        return 0
    for key in keys:
        got = get_run(key, args.root)
        run = got["run"]
        buckets = (run.get("summary") or {}).get("by_bucket", {})
        rates = " ".join(f"{b}={d.get('recovery_rate')}" for b, d in buckets.items())
        led = run.get("ledger") or {}
        state = (f"ledger={led.get('n_parsed')}/{led.get('n_candidates')}"
                 if got["ledger"] else "ledger=-")
        print(f"{str(key):52s} n={run.get('n_records', '?'):<5} {state:<16} "
              f"{'scored' if got['result'] is not None else 'unscored':8s} "
              f"raw={len(run.get('raw', []))} {rates}")
    return 0


def cmd_index(args) -> int:
    print(write_index(args.root))
    return 0


def cmd_flatten(args) -> int:
    written = materialize_flat(args.dest, root=args.root, version=args.version)
    for p in written:
        print(p)
    print(f"\n{len(written)} file(s). Analyze with:\n"
          f"  RTLREPAIR_OUT={args.dest} python analyze_repairbench.py")
    return 0


def cmd_push(args) -> int:
    from . import hf
    key = RunKey(args.model, args.version, args.signal) if args.model else None
    rid = hf.push(key, repo=args.repo, root=args.root,
                  private=not args.public, message=args.message,
                  refresh_card=args.refresh_card)
    print(f"pushed {'everything' if key is None else key} -> "
          f"https://huggingface.co/datasets/{rid}")
    return 0


def cmd_pull(args) -> int:
    from . import hf
    key = (RunKey(args.model, args.version, args.signal)
           if args.model and args.version and args.signal else None)
    dest = hf.pull(key, repo=args.repo, root=args.root, revision=args.revision)
    print(f"pulled -> {dest}")
    return 0


def cmd_diff(args) -> int:
    from . import hf
    d = hf.diff(repo=args.repo, root=args.root)
    for label in ("local_only", "remote_only", "both"):
        for item in d[label]:
            print(f"{label:12s} {item}")
    return 0


def cmd_check(args) -> int:
    exps, _ = _experiments()
    check_no_collisions(exps)
    print(f"ok — {len(exps)} arms map to {len(exps)} distinct run keys")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="python -m storage", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", help="tree root (default $RTLREPAIR_STORE or the bench dir)")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("put", help="bank one run")
    p.add_argument("--arm", help="arm name from configs/experiments.json")
    p.add_argument("--model"); p.add_argument("--signal"); p.add_argument("--name")
    p.add_argument("--version"); p.add_argument("--note")
    p.add_argument("--result", help="scored rb_*.json path")
    p.add_argument("--ledger", help="candidates.jsonl — one record per model call")
    p.add_argument("--raw", nargs="*", help="raw completion file(s) or a directory")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_put)

    p = sub.add_parser("migrate", help="flat rb_*.json -> the tree (copies)")
    p.add_argument("--src", help="dir holding rb_*.json (default results/batched)")
    p.add_argument("--version"); p.add_argument("--dry-run", action="store_true")
    p.add_argument("--overwrite", action="store_true")
    p.set_defaults(func=cmd_migrate)

    sub.add_parser("ls", help="list banked runs").set_defaults(func=cmd_ls)
    sub.add_parser("index", help="regenerate index.json").set_defaults(func=cmd_index)
    sub.add_parser("check", help="assert no two arms collide").set_defaults(func=cmd_check)

    p = sub.add_parser("flatten", help="project the tree back to rb_*.json")
    p.add_argument("--dest", required=True); p.add_argument("--version")
    p.set_defaults(func=cmd_flatten)

    p = sub.add_parser("push", help="upload to the HF dataset repo")
    p.add_argument("--repo"); p.add_argument("--model"); p.add_argument("--version")
    p.add_argument("--signal"); p.add_argument("--message")
    p.add_argument("--public", action="store_true", help="create the repo public")
    p.add_argument("--refresh-card", action="store_true",
                   help="overwrite the repo README with the current CARD")
    p.set_defaults(func=cmd_push)

    p = sub.add_parser("pull", help="download from the HF dataset repo")
    p.add_argument("--repo"); p.add_argument("--model"); p.add_argument("--version")
    p.add_argument("--signal"); p.add_argument("--revision", help="commit sha or tag")
    p.set_defaults(func=cmd_pull)

    p = sub.add_parser("diff", help="local vs remote")
    p.add_argument("--repo")
    p.set_defaults(func=cmd_diff)
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
