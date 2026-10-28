#!/usr/bin/env bash
set -euo pipefail
: "${VERO_EVAL_DIR:?}" "${CHECKPOINT_PATH:?}" "${OUTPUT_ROOT:?}" "${TOOLS:?}"
cd "$VERO_EVAL_DIR"
export TMPDIR="$(mktemp -d /tmp/q35.${SLURM_JOB_ID:?}.XXXXXX)"
export VLLM_RPC_BASE_PATH="$TMPDIR"
export PORT=$((20000 + SLURM_JOB_ID % 20000))
export MASTER_PORT=$((40000 + SLURM_JOB_ID % 20000))
export LOG_DIR="$OUTPUT_ROOT/runtime_logs"
export LMMS_EVAL_REQUEST_TRACE_DIR="$OUTPUT_ROOT/request_traces"
mkdir -p "$LOG_DIR"
# Keep the normal server shutdown/cleanup path from the canonical evaluator.
bash run_eval.sbatch
