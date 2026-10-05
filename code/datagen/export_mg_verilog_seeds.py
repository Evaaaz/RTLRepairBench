"""Export MG-Verilog rows into self-contained, lint-clean .sv seed files.

The repair pipeline needs *clean* seeds: complete modules that lint cleanly, so
the only bug in each pair is the one we inject. MG-Verilog does not ship those
directly — each row is

    row["code"]        = the module BODY only (no header), ending in endmodule
    row["description"] = Llama-2-wrapped NL spec whose tail is the module header

so we reconstruct ``header + body`` and keep a module only if it passes Verilator
lint. Most rows are dropped — many instantiate submodules we don't have, or use
undeclared signals — but ~11k rows still yield far more than the >=500 clean leaf
modules the repair set needs. The kept files feed build_repair_dataset.py via
``--seeds-dir``.
"""

from __future__ import annotations

import argparse
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

if __package__:
    from project_paths import project_root
    from .mg_verilog_to_prime_sft import GRANULARITIES, clean_description, iter_rows
    from .run_verilator_for_repairs import lint_source
else:  # compatibility: ``python datagen/export_mg_verilog_seeds.py``
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from project_paths import project_root
    from mg_verilog_to_prime_sft import GRANULARITIES, clean_description, iter_rows
    from run_verilator_for_repairs import lint_source

REPO_ROOT = project_root()

_MODULE_START = re.compile(r"\bmodule\b")
_MODULE_NAME = re.compile(r"\bmodule\s+(\w+)")


def extract_header(text: str) -> str | None:
    """The ``module name (...);`` header at the tail of a cleaned description."""
    starts = [m.start() for m in _MODULE_START.finditer(text)]
    if not starts:
        return None
    end = text.find(");", starts[-1])
    if end == -1:
        return None
    return text[starts[-1] : end + 2].strip()


def reconstruct(row: dict) -> tuple[str, str] | None:
    """Rebuild a full module from a row -> (module_name, full_source) or None."""
    code = (row.get("code") or "").strip()
    if not code or "endmodule" not in code:
        return None
    desc = row.get("description") or {}
    for g in GRANULARITIES:
        header = extract_header(clean_description(desc.get(g, "")))
        if not header:
            continue
        name_match = _MODULE_NAME.search(header)
        if not name_match:
            continue
        return name_match.group(1), header + "\n" + code + "\n"
    return None


def _try_seed(row: dict) -> tuple[str, str] | None:
    """Reconstruct + lint one row; return (name, source) only if it lints clean."""
    rebuilt = reconstruct(row)
    if rebuilt is None:
        return None
    name, source = rebuilt
    if lint_source(f"{name}.sv", source).failed:
        return None
    return name, source


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(REPO_ROOT / "data" / "raw" / "mg-verilog" / "merged_dataset"))
    ap.add_argument(
        "--out-dir", default=str(REPO_ROOT / "generated" / "datagen" / "mg_seeds")
    )
    ap.add_argument("--limit", type=int, default=0, help="cap rows scanned (0=all)")
    ap.add_argument("--jobs", type=int, default=8, help="parallel Verilator workers")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i, row in enumerate(iter_rows(args.src)):
        if args.limit and i >= args.limit:
            break
        rows.append(row)

    kept = 0
    seen_names: dict[str, int] = {}
    with ThreadPoolExecutor(max_workers=args.jobs) as ex:
        for result in (f.result() for f in as_completed(ex.submit(_try_seed, r) for r in rows)):
            if result is None:
                continue
            name, source = result
            # Disambiguate duplicate module names so seeds never collide.
            n = seen_names.get(name, 0)
            seen_names[name] = n + 1
            fname = f"{name}.sv" if n == 0 else f"{name}__{n}.sv"
            (out_dir / fname).write_text(source)
            kept += 1

    print(
        f"scanned={len(rows)} clean_seeds={kept} "
        f"survival={kept / max(len(rows), 1):.1%} -> {out_dir}"
    )


if __name__ == "__main__":
    main()
