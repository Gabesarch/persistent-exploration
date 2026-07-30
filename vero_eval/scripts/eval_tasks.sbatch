#!/bin/bash
#SBATCH --job-name=eval_tasks
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --output=vero_eval/logs/eval_tasks_%j.out
#SBATCH --error=vero_eval/logs/eval_tasks_%j.err
#
# Generic task evaluator. Submit from repo root:
#   TASKS=countqa_qwen3_drawing sbatch vero_eval/scripts/eval_tasks.sbatch
#   TASKS=blink_drawing,cv_bench_drawing sbatch vero_eval/scripts/eval_tasks.sbatch
#
# Environment variables:
#   TASKS          (required) Comma-separated task names
#   LIMIT          (optional) Limit number of samples (default: full dataset)
#   OUTPUT_PATH    (optional) Output directory (default: results/<task> or results/multi_task_<timestamp>)
#   WITH_TOOLS     (optional) 1=enable MCP tools, 0=disable (default: 1)
#   SYSTEM_PROMPT  (optional) Path to system prompt file (default: none)
#   MAX_TOOL_ITERS (optional) Max tool iterations (default: 10)

set -euo pipefail

# Locate vero_eval directory
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$SUBMIT_DIR/set_paths.sh.example" && "$(basename "$SUBMIT_DIR")" == "vero_eval" ]]; then
  VERO_EVAL_DIR="$SUBMIT_DIR"
else
  VERO_EVAL_DIR="$SUBMIT_DIR/vero_eval"
fi
if [[ ! -f "$VERO_EVAL_DIR/set_paths.sh.example" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  VERO_EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [[ ! -f "$VERO_EVAL_DIR/set_paths.sh.example" ]]; then
  echo "ERROR: vero_eval dir not found. Submit from repo root:" >&2
  echo "  TASKS=<task> sbatch vero_eval/scripts/eval_tasks.sbatch" >&2
  exit 1
fi

set +u
if [[ -f "$VERO_EVAL_DIR/set_paths.sh" ]]; then
  source "$VERO_EVAL_DIR/set_paths.sh"
else
  echo "WARN: set_paths.sh not found; using set_paths.sh.example" >&2
  source "$VERO_EVAL_DIR/set_paths.sh.example"
fi
set -u

# Required: TASKS
if [[ -z "${TASKS:-}" ]]; then
  echo "ERROR: TASKS is required. Example:" >&2
  echo "  TASKS=countqa_qwen3_drawing sbatch vero_eval/scripts/eval_tasks.sbatch" >&2
  exit 1
fi

# Defaults
LIMIT="${LIMIT:-}"
# Use first task name for single task, "multi_task_<timestamp>" for multiple
if [[ "$TASKS" == *","* ]]; then
  DEFAULT_OUTPUT="$VERO_EVAL_DIR/results/multi_task_$(date +%Y%m%d_%H%M%S)"
else
  DEFAULT_OUTPUT="$VERO_EVAL_DIR/results/$TASKS"
fi
OUTPUT_PATH="${OUTPUT_PATH:-$DEFAULT_OUTPUT}"
WITH_TOOLS="${WITH_TOOLS:-1}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERS:-10}"
# Use a job-specific policy port to avoid conflicts when multiple jobs share a node.
PORT="${PORT:-$((9000 + SLURM_JOB_ID % 1000))}"
API_KEY="${API_KEY:-qwen}"
export JUDGE_BACKEND="${JUDGE_BACKEND:-engine}"
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$VERO_EVAL_DIR/logs" "$OUTPUT_PATH"
cd "$VERO_EVAL_DIR"

# Start or reuse vLLM server
if curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models"; then
  echo "=== Reusing existing vLLM server on :$PORT ==="
else
  echo "=== Starting vLLM server ==="; bash scripts/serve_qwen_vllm.sh
fi
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
  || { echo "ERROR: vLLM not responding on :$PORT"; exit 1; }

echo ""
echo "########## Evaluating: $TASKS ##########"
echo "Output:     $OUTPUT_PATH"
echo "With tools: $WITH_TOOLS"
echo ""

# Run evaluation
if [[ "$WITH_TOOLS" == "1" ]]; then
  SYSTEM_PROMPT="$SYSTEM_PROMPT" MAX_TOOL_ITERATIONS="$MAX_TOOL_ITERATIONS" \
    OUTPUT_PATH="$OUTPUT_PATH" \
    bash examples/eval_domain.sh --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"}
else
  SYSTEM_PROMPT="$SYSTEM_PROMPT" MCP_SERVER_PATH="" \
    OUTPUT_PATH="$OUTPUT_PATH" \
    bash examples/eval_domain.sh --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"}
fi

echo ""
echo "=== Evaluation complete. Results: $OUTPUT_PATH ==="
