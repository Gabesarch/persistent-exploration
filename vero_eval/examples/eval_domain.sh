#!/bin/bash
# eval_domain.sh — canonical runner for the vero_eval agentic (MCP tool-use) eval.
#
# Run from the package root (vero_eval/) so `python -m lmms_eval` picks up the
# local lmms_eval package:
#   cd vero_eval && bash examples/eval_domain.sh --tasks <TASKS> [--output-path <PATH>]
#
# Examples:
#   bash examples/eval_domain.sh --tasks countqa_reasoning
#   SYSTEM_PROMPT=prompts/countqa_system_prompt.txt bash examples/eval_domain.sh --tasks countqa_reasoning
#
# Pick the task yaml by base model (see README): vero -> *_reasoning,
# vero-thinking -> *_reasoning_samplingq3, qwen3vl-instruct -> *_qwen3_zs,
# qwen3vl-thinking -> *_qwen3_thinking_zs.

set -euo pipefail

# ── account config (caches, conda env, offline flags, PORT, ...) ──────
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERO_EVAL_DIR="$(cd "$SELF/.." && pwd)"
set +u
if [[ -f "$VERO_EVAL_DIR/set_paths.sh" ]]; then
  source "$VERO_EVAL_DIR/set_paths.sh"
else
  echo "WARN: $VERO_EVAL_DIR/set_paths.sh not found; using set_paths.sh.example (copy & edit it)." >&2
  source "$VERO_EVAL_DIR/set_paths.sh.example"
fi
set -u

# ── defaults (all env-overridable) ────────────────────────────────────
TASKS=""
OUTPUT_PATH="${OUTPUT_PATH:-./results}"
BATCH_SIZE="${BATCH_SIZE:-1}"
PORT="${PORT:-9001}"
API_KEY="${API_KEY:-qwen}"
MODEL_VERSION="${MODEL_VERSION:-qwen_vllm}"
MCP_SERVER_PATH="${MCP_SERVER_PATH:-lmms_eval/mcp/vero2_mcp.py}"   # set empty to disable tools
MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERATIONS:-10}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
LIMIT="${LIMIT:-}"

# ── parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)        TASKS="$2"; shift 2 ;;
    --output-path)  OUTPUT_PATH="$2"; shift 2 ;;
    --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
    --limit)        LIMIT="$2"; shift 2 ;;
    *)              shift ;;
  esac
done

if [[ -z "$TASKS" ]]; then
  echo "Usage: bash examples/eval_domain.sh --tasks <TASKS> [--output-path <PATH>] [--limit N]"
  exit 1
fi

# ── judge backend (only used by judge-based tasks; inert for exact_match like
#    countqa). Mirrors reason-vlm-eval/examples/eval_model.sh. get_judge_engine()
#    reads these env vars to pick the backend. ────────────────────────────────
JUDGE_BACKEND="${JUDGE_BACKEND:-engine}"
if [[ "$JUDGE_BACKEND" == "engine" ]]; then
  # In-process vLLM judge (default), loaded by the task's process_results.
  export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-$HF_HOME/Qwen/Qwen3-32B}"
  export CHARXIV_JUDGE_GPU_MEMORY_UTILIZATION="${CHARXIV_JUDGE_GPU_MEMORY_UTILIZATION:-0.9}"
  export CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE="${CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE:-${NUM_GPUS:-1}}"
  export CHARXIV_JUDGE_MAX_MODEL_LEN="${CHARXIV_JUDGE_MAX_MODEL_LEN:-18384}"
  export VLLM_JUDGE_MAX_NUM_SEQS="${VLLM_JUDGE_MAX_NUM_SEQS:-32}"
  unset VLLM_SERVER_JUDGE VLLM_JUDGE_BASE_URL VLLM_JUDGE_PORT VLLM_JUDGE_MODEL_PATH 2>/dev/null || true
elif [[ "$JUDGE_BACKEND" == "server" ]]; then
  # Judge served over HTTP via wake/sleep (VLLMServerJudge).
  export VLLM_SERVER_JUDGE=1
  if [[ "${START_VLLM_JUDGE_SERVER:-0}" == "1" ]]; then
    # Launch a DEDICATED judge server (e.g. Qwen3-32B). Needs a free GPU — on a
    # single H100 give it its own: request gpu:2 and set VLLM_JUDGE_CUDA_VISIBLE_DEVICES=1.
    # serve_judge_vllm.sh exports VLLM_JUDGE_BASE_URL.
    source "$VERO_EVAL_DIR/scripts/serve_judge_vllm.sh"
  else
    # Default: reuse the policy server on $PORT as the judge (judge == policy model,
    # zero extra GPU). wake/sleep are harmless no-ops against the non-sleep server.
    export VLLM_JUDGE_BASE_URL="${VLLM_JUDGE_BASE_URL:-http://localhost:${PORT}/v1}"
  fi
  export VLLM_JUDGE_MODEL_PATH="${VLLM_JUDGE_MODEL_PATH:-qwen_vllm}"
  export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-$VLLM_JUDGE_MODEL_PATH}"
else
  echo "ERROR: unknown JUDGE_BACKEND '$JUDGE_BACKEND' (use engine|server)" >&2
  exit 1
fi
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ── build model_args ──────────────────────────────────────────────────
MODEL_ARGS="model_version=${MODEL_VERSION},base_url=http://localhost:${PORT}/v1,api_key=${API_KEY},max_tool_iterations=${MAX_TOOL_ITERATIONS}"
[[ -n "$MCP_SERVER_PATH" ]] && MODEL_ARGS="${MODEL_ARGS},mcp_server_path=${MCP_SERVER_PATH}"
[[ -n "$SYSTEM_PROMPT"   ]] && MODEL_ARGS="${MODEL_ARGS},system_prompt=${SYSTEM_PROMPT}"

echo "Tasks:      $TASKS"
echo "Output:     $OUTPUT_PATH"
echo "Model args: $MODEL_ARGS"
echo ""

python -m lmms_eval \
    --model async_openai \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --log_samples \
    --output_path "$OUTPUT_PATH" \
    ${LIMIT:+--limit "$LIMIT"}
