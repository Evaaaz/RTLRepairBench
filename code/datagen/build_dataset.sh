#!/usr/bin/env bash
# Build the lint-vs-semantic repair dataset from clean SystemVerilog seeds.
#
# This is the standalone lint-versus-semantic dataset pipeline; it
# keeps ONLY the steps that produce and validate the two repair buckets
# (lint-class and semantic-class) plus decontamination. The SFT-mixture and
# spec2rtl/explain steps from the full pipeline are intentionally dropped.
#
# The scripts are location-independent: all inputs/outputs come from flags or
# env vars, so this runs from any checkout without a fixed repo layout.
#
# Prereqs (see README.md §Requirements):
#   * verilator and iverilog+vvp on PATH (the two gates — no stub fallback)
#   * Python 3 with `datasets` installed (ONLY for step 1, the MG-Verilog export)
#   * MG-Verilog corpus on disk          (input to step 1; a load_from_disk dir)
#   * VerilogEval + RTLLM clones under $RAW (input to step 4, for decontam)
#
# If you already have a directory of clean .sv seeds, set SEEDS=/path and steps
# 2-3 run standalone with no `datasets` install and no MG-Verilog download.
#
# Usage:
#   MG=/path/to/mg-verilog RAW=/path/to/data/raw bash build_dataset.sh
#   SEEDS=/path/to/my/seeds OUT=/fresh/output bash build_dataset.sh  # skip export
#
set -euo pipefail
cd "$(dirname "$0")"                       # code/datagen

PY="${PY:-python3}"
CODE_DIR="$(cd .. && pwd)"
# Shared resolver supports both the historical nested checkout and a standalone
# extraction. RTLREPAIR_ROOT remains the explicit compatibility override.
REPO_ROOT="${RTLREPAIR_ROOT:-$($PY "$CODE_DIR/project_paths.py")}"
OUT="${OUT:-$REPO_ROOT/generated/datagen/dataset}"  # never default inside release sources
RAW="${RAW:-$REPO_ROOT/data/raw}"          # held-out benchmark clones (decontam)
MG="${MG:-$REPO_ROOT/data/raw/mg-verilog/merged_dataset}"   # MG-Verilog corpus (load_from_disk)
SEEDS="${SEEDS:-$OUT/mg_seeds}"            # seed dir consumed by steps 2-3
CAP="${CAP:-1000}"                         # per-mutation cap on the lint bucket

mkdir -p "$OUT"

if [ "$SEEDS" = "$OUT/mg_seeds" ]; then
  mkdir -p "$SEEDS"
  echo "==> [1/5] export clean, lint-passing MG-Verilog seeds -> $SEEDS"
  # export_mg_verilog_seeds.py loads MG-Verilog via load_from_disk; point --src
  # at your local copy. Needs the `datasets` package (see ../requirements.txt).
  $PY export_mg_verilog_seeds.py --src "$MG" --out-dir "$SEEDS"
else
  echo "==> [1/5] SKIP MG-Verilog export — using provided SEEDS=$SEEDS"
fi

echo "==> [2/5] lint-class repair pairs  (gate: verilator --lint-only)"
$PY build_repair_dataset.py --seeds-dir "$SEEDS" \
    --out "$OUT/repair_lint.jsonl" --cap-per-mutation "$CAP"

echo "==> [3/5] semantic-class repair pairs  (gate: differential iverilog sim)"
$PY build_semantic_repair.py --seeds-dir "$SEEDS" \
    --out "$OUT/repair_semantic.jsonl"

UNFILTERED="$OUT/repair_train.pre_decontamination.jsonl"
cat "$OUT/repair_lint.jsonl" "$OUT/repair_semantic.jsonl" > "$UNFILTERED"

echo "==> [4/5] held-out benchmark fingerprints  (from VerilogEval + RTLLM)"
$PY build_benchmark_tasks.py --raw "$RAW" --out "$OUT/benchmark_tasks.jsonl"

echo "==> [5/5] decontaminate the combined training bucket"
$PY decontaminate.py --in "$UNFILTERED" \
    --out "$OUT/repair_train.jsonl" \
    --benchmark "$OUT/benchmark_tasks.jsonl"

echo "==> DONE"
echo "    lint-class     -> $OUT/repair_lint.jsonl"
echo "    semantic-class -> $OUT/repair_semantic.jsonl"
echo "    combined       -> $OUT/repair_train.jsonl (decontaminated)"
echo "    audit input    -> $UNFILTERED (not for training)"
