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
export PYTHONPATH="$(cd "$VERO_EVAL_DIR/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
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
MCP_SERVER_PATH="${MCP_SERVER_PATH-lmms_eval/mcp/vero2_mcp.py}"    # set empty (MCP_SERVER_PATH='') to disable tools
MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERATIONS:-10}"
MAX_TOOL_IMAGES="${MAX_TOOL_IMAGES:-10}"
FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS="${FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS:-false}"
FORCE_FINAL_ANSWER_PREFIX="${FORCE_FINAL_ANSWER_PREFIX:-}"
DEFAULT_FORCE_FINAL_ANSWER_THINK_PREFILL="This is the final turn, so I need to answer from the evidence gathered so far."
export FORCE_FINAL_ANSWER_THINK_PREFILL="${FORCE_FINAL_ANSWER_THINK_PREFILL:-$DEFAULT_FORCE_FINAL_ANSWER_THINK_PREFILL}"
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

# ── judge backend (set up ONLY when the run contains a judge-based task; inert for
#    exact_match tasks like countqa). DEFAULT = an in-process local vLLM judge on the
#    policy GPU. The async_openai driver coordinates policy sleep/wake across policy
#    generation and judge postprocessing; the judge remains cached for the process.
#    Hosted API_TYPE=openai|azure judges do not manage the local policy lifecycle. The old
#    dedicated judge server remains available explicitly with JUDGE_BACKEND=server
#    and needs two GPUs unless policy self-judging is deliberately requested. ─────
_JUDGE_STEMS="visual_probe simplevqa mathvista mathvision charxiv mmvetv2 mia_bench mm_mt_bench mmifeval fvqa chartmuseum"
RUN_HAS_JUDGE=0
MANAGE_VLLM_SERVER_LIFECYCLE=false
for _t in ${TASKS//,/ }; do
  for _s in $_JUDGE_STEMS; do [[ "$_t" == ${_s}* ]] && RUN_HAS_JUDGE=1; done
done

if [[ "$RUN_HAS_JUDGE" == 1 ]]; then
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  JUDGE_BACKEND="${JUDGE_BACKEND:-engine}"
  if [[ "$JUDGE_BACKEND" == "engine" ]]; then
    export API_TYPE="${API_TYPE:-vllm}"
    if [[ "$API_TYPE" == "vllm" ]]; then
      # Sequential one-GPU path: policy generation -> deep sleep -> in-process judging.
      # The judge stays cached until process exit; a later generation phase wakes policy.
      export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${VLLM_JUDGE_MODEL_PATH:-$HF_HOME/Qwen/Qwen3-32B}}"
      export CHARXIV_JUDGE_GPU_MEMORY_UTILIZATION="${CHARXIV_JUDGE_GPU_MEMORY_UTILIZATION:-0.9}"
      export CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE="${CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE:-1}"
      export CHARXIV_JUDGE_MAX_MODEL_LEN="${CHARXIV_JUDGE_MAX_MODEL_LEN:-18432}"
      export VLLM_JUDGE_MAX_NUM_SEQS="${VLLM_JUDGE_MAX_NUM_SEQS:-32}"
      unset VLLM_SERVER_JUDGE VLLM_JUDGE_BASE_URL VLLM_JUDGE_PORT 2>/dev/null || true
      MANAGE_VLLM_SERVER_LIFECYCLE=true
    elif [[ "$API_TYPE" == "openai" || "$API_TYPE" == "azure" ]]; then
      # Hosted judges do not own, sleep, or wake the local policy server.
      MANAGE_VLLM_SERVER_LIFECYCLE=false
    else
      echo "ERROR: unknown API_TYPE '$API_TYPE' (use vllm|openai|azure)" >&2
      exit 1
    fi
  elif [[ "$JUDGE_BACKEND" == "server" ]]; then
    export VLLM_SERVER_JUDGE=1
    _NGPU="$(nvidia-smi -L 2>/dev/null | wc -l || echo 1)"
    if [[ "${START_VLLM_JUDGE_SERVER:-1}" != "0" ]]; then
      if [[ "${_NGPU:-1}" -lt 2 ]]; then
        echo "ERROR: JUDGE_BACKEND=server requires 2 GPUs for separate policy and judge servers." >&2
        echo "       Use the one-GPU default JUDGE_BACKEND=engine, or explicitly set" >&2
        echo "       START_VLLM_JUDGE_SERVER=0 for a non-reportable policy self-judge." >&2
        exit 1
      fi
      # Explicit legacy path: dedicated Qwen3-32B judge on the last GPU.
      export VLLM_JUDGE_MODEL_PATH="${VLLM_JUDGE_MODEL_PATH:-$HF_HOME/Qwen/Qwen3-32B}"
      export VLLM_JUDGE_CUDA_VISIBLE_DEVICES="${VLLM_JUDGE_CUDA_VISIBLE_DEVICES:-$((_NGPU - 1))}"
      source "$VERO_EVAL_DIR/scripts/serve_judge_vllm.sh"   # exports VLLM_JUDGE_BASE_URL
    else
      # Deliberate fallback: reuse the policy server as judge.
      export VLLM_JUDGE_BASE_URL="${VLLM_JUDGE_BASE_URL:-http://localhost:${PORT}/v1}"
      export VLLM_JUDGE_MODEL_PATH="qwen_vllm"
      echo "WARNING: NON-REPORTABLE SELF-JUDGE: the policy model is judging its own outputs." >&2
      echo "         Use JUDGE_BACKEND=engine for reportable one-GPU local judging." >&2
    fi
    export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${VLLM_JUDGE_MODEL_PATH:-qwen_vllm}}"
  else
    echo "ERROR: unknown JUDGE_BACKEND '$JUDGE_BACKEND' (use engine|server)" >&2
    exit 1
  fi
fi

# ── build model_args ──────────────────────────────────────────────────
MODEL_ARGS="model_version=${MODEL_VERSION},base_url=http://localhost:${PORT}/v1,api_key=${API_KEY},max_tool_iterations=${MAX_TOOL_ITERATIONS},max_tool_images=${MAX_TOOL_IMAGES}"
[[ "$MANAGE_VLLM_SERVER_LIFECYCLE" == true ]] && MODEL_ARGS="${MODEL_ARGS},manage_vllm_server_lifecycle=true"
[[ -n "$MCP_SERVER_PATH" ]] && MODEL_ARGS="${MODEL_ARGS},mcp_server_path=${MCP_SERVER_PATH}"
[[ -n "$SYSTEM_PROMPT"   ]] && MODEL_ARGS="${MODEL_ARGS},system_prompt=${SYSTEM_PROMPT}"
case "${FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS,,}" in
  1|true|yes|on)
    MODEL_ARGS="${MODEL_ARGS},force_answer_after_max_tool_iterations=true"
    [[ -n "$FORCE_FINAL_ANSWER_PREFIX" ]] && MODEL_ARGS="${MODEL_ARGS},force_final_answer_prefix=${FORCE_FINAL_ANSWER_PREFIX}"
    ;;
esac

echo "Tasks:      $TASKS"
echo "Output:     $OUTPUT_PATH"
echo "Model args: $MODEL_ARGS"
echo "Final think prefill: ${FORCE_FINAL_ANSWER_THINK_PREFILL:-<empty>}"
echo ""

python -m lmms_eval \
    --model async_openai \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --log_samples \
    --output_path "$OUTPUT_PATH" \
    ${LIMIT:+--limit "$LIMIT"}
