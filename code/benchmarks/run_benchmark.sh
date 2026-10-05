#!/usr/bin/env bash
#
# run_benchmark.sh -- robust wrapper around `python benchmarks/run_all.py`.
#
# Runs the headline base/tuned/repair benchmark against the live model endpoint
# and writes generated/benchmark_results/table.{md,csv} (+ logs to W&B).
#
# Usage:
#   bash benchmarks/run_benchmark.sh [smoke|full]
#
#   smoke   (default)  -> --limit 20 --n 5   (fast sanity pass)
#   full               -> --n 20             (the real headline run)
#
# Required environment (NEVER hardcode the token in this file):
#   RTLREPAIR_LLM_URL       e.g. http://host:8000/v1
#   RTLREPAIR_LLM_API_KEY   bearer token for the endpoint
#
# Optional environment:
#   WANDB_PROJECT          defaults to "rtlrepair-bench"
#   WANDB_ENTITY           passed through only if set
#
set -euo pipefail

# --- locate repo root (this script lives 4 levels below it, in ---------------
# code/benchmarks; override with RTLREPAIR_ROOT) -----------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${RTLREPAIR_ROOT:-$(cd "${SCRIPT_DIR}/../.." && pwd)}"

# --- pick the project venv python if present, else system python3 ------------
PY="${REPO_ROOT}/.venv/bin/python"
if [[ ! -x "${PY}" ]]; then
  PY="$(command -v python3 || true)"
fi
if [[ -z "${PY}" ]]; then
  echo "ERROR: no python interpreter found (looked for ${REPO_ROOT}/.venv/bin/python and python3)." >&2
  exit 1
fi

# --- require endpoint credentials --------------------------------------------
if [[ -z "${RTLREPAIR_LLM_URL:-}" ]]; then
  echo "ERROR: RTLREPAIR_LLM_URL is not set." >&2
  echo "       export RTLREPAIR_LLM_URL=http://<host>:8000/v1" >&2
  exit 1
fi
if [[ -z "${RTLREPAIR_LLM_API_KEY:-}" ]]; then
  echo "ERROR: RTLREPAIR_LLM_API_KEY is not set (the endpoint bearer token)." >&2
  echo "       export RTLREPAIR_LLM_API_KEY=<your-token>" >&2
  exit 1
fi

# --- W&B defaults ------------------------------------------------------------
WANDB_PROJECT="${WANDB_PROJECT:-rtlrepair-bench}"

# --- mode -> sampling args ---------------------------------------------------
MODE="${1:-smoke}"
case "${MODE}" in
  smoke) MODE_ARGS=(--limit 20 --n 5) ;;
  full)  MODE_ARGS=(--n 20) ;;
  *)
    echo "ERROR: unknown mode '${MODE}'. Use 'smoke' (default) or 'full'." >&2
    exit 1
    ;;
esac

# --- assemble the command ----------------------------------------------------
CMD=("${PY}" "${SCRIPT_DIR}/run_all.py" "${MODE_ARGS[@]}" --wandb-project "${WANDB_PROJECT}")
if [[ -n "${WANDB_ENTITY:-}" ]]; then
  CMD+=(--wandb-entity "${WANDB_ENTITY}")
fi

# --- echo resolved command + run ---------------------------------------------
echo "mode:           ${MODE}"
echo "endpoint:       ${RTLREPAIR_LLM_URL}"
echo "wandb project:  ${WANDB_PROJECT}"
echo "wandb entity:   ${WANDB_ENTITY:-<unset>}"
echo "running:        ${CMD[*]}"
echo

# run from the repo root so relative paths (data/, generated/) resolve
cd "${REPO_ROOT}"
"${CMD[@]}"

echo
echo "results table:  ${REPO_ROOT}/generated/benchmark_results/table.md"
echo "results csv:    ${REPO_ROOT}/generated/benchmark_results/table.csv"
