"""Convert the MG-Verilog dataset into SFT format (FULL-MODULE target).

MG-Verilog (GaTech-EIC/MG-Verilog) is a single `save_to_disk` Arrow dataset.
Each row is:

    {
      "code": "<verilog logic ... >\nendmodule",   # body only, NOT the header
      "description": {
        "block_summary":             "<s>[INST] <<SYS>> ... [/INST]",
        "detailed_global_summary":   "<s>[INST] <<SYS>> ... [/INST]",
        "high_level_global_summary": "<s>[INST] <<SYS>> ... [/INST]",
      }
    }

Each `description` value is Llama-2-wrapped and ends with the module header.

*** FIX (2026-06-25) ***
The previous version trained the model to emit the *body only* (no `module`
header, no code fence). The benchmark's extract_rtl wants a fenced, COMPLETE
`module ... endmodule`, so body-only targets are un-extractable / un-compilable
and the tuned model scored BELOW base (a pure format artifact). We now:
  * reconstruct the FULL module (header + body), and
  * wrap it in a ```systemverilog fence, and
  * frame the user turn exactly like benchmarks/bench_common.build_prompt,
so the training output == what the eval extracts and compiles.

Output: JSONL of `{"messages": [...]}`:
    system    -> "produce a complete module in a code block"
    user      -> instruction + "Specification:\\n" + spec + "...code block."
    assistant -> ```systemverilog\\n<module header+body>\\n```
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

if __package__:
    from project_paths import project_root
else:  # compatibility: ``python datagen/mg_verilog_to_prime_sft.py``
    sys.path.append(str(Path(__file__).resolve().parent.parent))
    from project_paths import project_root

REPO_ROOT = project_root()

GRANULARITIES = (
    "high_level_global_summary",
    "detailed_global_summary",
    "block_summary",
)

SYSTEM_PROMPT = (
    "You are an expert hardware design assistant. Given a natural-language "
    "specification, write a complete, synthesizable SystemVerilog module "
    "(module header and body, ending with endmodule) inside a ```systemverilog "
    "code block."
)

# Mirror benchmarks/bench_common.build_prompt so train and eval prompts match.
USER_INSTRUCTION = "Write the requested SystemVerilog module."
USER_SUFFIX = (
    "\nRespond with a single synthesizable SystemVerilog module only, "
    "inside a ```systemverilog code block."
)

_SYS_BLOCK = re.compile(r"<<SYS>>.*?<</SYS>>", re.DOTALL)
_INST_TAGS = re.compile(r"</?s>|\[/?INST\]")
_MODULE_START = re.compile(r"\bmodule\b")
_MODULE_NAME = re.compile(r"\bmodule\s+(\w+)")


def clean_description(raw: str) -> str:
    """Strip the Llama-2 [INST]/<<SYS>> scaffold; keep the spec + module header."""
    if raw is None:
        return ""
    text = _SYS_BLOCK.sub("", raw)
    text = _INST_TAGS.sub("", text)
    return text.strip()


def extract_header(text: str) -> str | None:
    """The ``module name (...);`` header at the tail of a cleaned description."""
    starts = [m.start() for m in _MODULE_START.finditer(text)]
    if not starts:
        return None
    end = text.find(");", starts[-1])
    if end == -1:
        return None
    return text[starts[-1] : end + 2].strip()


def full_module(row: dict) -> str | None:
    """Rebuild the FULL module source (header + body) or None."""
    code = (row.get("code") or "").strip()
    if not code or "endmodule" not in code:
        return None
    desc = row.get("description") or {}
    for g in GRANULARITIES:
        header = extract_header(clean_description(desc.get(g, "")))
        if header and _MODULE_NAME.search(header):
            return (header + "\n" + code).strip()
    return None


def to_records(row: dict, granularities: tuple[str, ...]) -> list[dict]:
    """Emit one full-module SFT record per requested granularity."""
    module_src = full_module(row)
    if not module_src:
        return []
    fenced = "```systemverilog\n" + module_src + "\n```"
    desc = row.get("description") or {}
    records = []
    for g in granularities:
        spec = clean_description(desc.get(g, ""))
        if not spec:
            continue
        user = USER_INSTRUCTION + "\nSpecification:\n" + spec + USER_SUFFIX
        records.append(
            {
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": fenced},
                ]
            }
        )
    return records


def iter_rows(src: str):
    """Yield rows from a load_from_disk dataset dir or a JSONL sample file."""
    if src.endswith(".jsonl"):
        with open(src) as f:
            for line in f:
                line = line.strip()
                if line:
                    yield json.loads(line)
        return
    from datasets import load_from_disk

    ds = load_from_disk(src)
    if hasattr(ds, "keys"):
        ds = ds[next(iter(ds.keys()))]
    yield from ds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--src", default=str(REPO_ROOT / "data" / "raw" / "mg-verilog" / "merged_dataset"))
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "generated" / "datagen" / "mg_verilog_sft.jsonl")
    )
    ap.add_argument(
        "--granularities",
        nargs="+",
        default=["high_level_global_summary"],
        choices=GRANULARITIES + ("all",),
    )
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    grans = GRANULARITIES if "all" in args.granularities else tuple(args.granularities)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = n_recs = n_skipped = 0
    with open(out_path, "w") as out:
        for row in iter_rows(args.src):
            if args.limit and n_rows >= args.limit:
                break
            n_rows += 1
            recs = to_records(row, grans)
            if not recs:
                n_skipped += 1
            for r in recs:
                out.write(json.dumps(r) + "\n")
                n_recs += 1

    print(
        f"rows={n_rows} records={n_recs} skipped_no_header={n_skipped} "
        f"granularities={grans} -> {out_path}"
    )


if __name__ == "__main__":
    main()
