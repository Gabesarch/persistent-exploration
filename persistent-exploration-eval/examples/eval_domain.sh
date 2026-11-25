#!/bin/bash
# Internal lmms_eval runner for the selected paper benchmarks.

set -euo pipefail

# ── account config (caches, conda env, offline flags, PORT, ...) ──────
SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERO_EVAL_DIR="$(cd "$SELF/.." && pwd)"
export PYTHONPATH="$(cd "$VERO_EVAL_DIR/.." && pwd)${PYTHONPATH:+:$PYTHONPATH}"
set +u
if [[ -f "$VERO_EVAL_DIR/set_paths.sh" ]]; then
  source "$VERO_EVAL_DIR/set_paths.sh"
else
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
FORCE_TOOL_CALLS_UNTIL_FINAL_TURN="${FORCE_TOOL_CALLS_UNTIL_FINAL_TURN:-false}"
FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS="${FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS:-false}"
FORCE_FINAL_ANSWER_PREFIX="${FORCE_FINAL_ANSWER_PREFIX:-}"
CRITIC_GUIDED_ANSWER_RETRY="${CRITIC_GUIDED_ANSWER_RETRY:-false}"
CRITIC_GUIDED_ANSWER_RETRY_SOURCE="${CRITIC_GUIDED_ANSWER_RETRY_SOURCE:-policy}"
CRITIC_GUIDED_ANSWER_RETRY_MODE="${CRITIC_GUIDED_ANSWER_RETRY_MODE:-token}"
CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD="${CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD:-0.5}"
CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE="${CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE:-persist}"
DEFAULT_FORCE_FINAL_ANSWER_THINK_PREFILL="This is the final turn, so I need to answer from the evidence gathered so far."
export FORCE_FINAL_ANSWER_THINK_PREFILL="${FORCE_FINAL_ANSWER_THINK_PREFILL-$DEFAULT_FORCE_FINAL_ANSWER_THINK_PREFILL}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
LIMIT="${LIMIT:-}"
OFFSET="${OFFSET:-0}"

# ── parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)        TASKS="$2"; shift 2 ;;
    --output-path)  OUTPUT_PATH="$2"; shift 2 ;;
    --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
    --limit)        LIMIT="$2"; shift 2 ;;
    --offset)       OFFSET="$2"; shift 2 ;;
    *)              shift ;;
  esac
done

if [[ -z "$TASKS" ]]; then
  echo "Usage: bash examples/eval_domain.sh --tasks <TASKS> [--output-path <PATH>] [--limit N]"
  exit 1
fi

# Partition copies share a canonical result directory.
mkdir -p "$OUTPUT_PATH"
exec {evaluation_lock_fd}>"$OUTPUT_PATH/.evaluation.lock"
flock "$evaluation_lock_fd"

# The local judge runs after policy generation. The async_openai driver manages
# policy sleep and wake while the judge uses the GPU.
_JUDGE_STEMS="visual_probe pinpoint_bench o3_bench"
RUN_HAS_JUDGE=0
MANAGE_VLLM_SERVER_LIFECYCLE=false
for _t in ${TASKS//,/ }; do
  for _s in $_JUDGE_STEMS; do [[ "$_t" == ${_s}* ]] && RUN_HAS_JUDGE=1; done
done

if [[ "$RUN_HAS_JUDGE" == 1 ]]; then
  export VLLM_WORKER_MULTIPROC_METHOD=spawn
  [[ "${JUDGE_BACKEND:-engine}" == engine && "${API_TYPE:-vllm}" == vllm ]] || {
    echo "ERROR: this evaluation uses the local vLLM judge engine" >&2
    exit 1
  }
  export JUDGE_BACKEND=engine API_TYPE=vllm
  export JUDGE_MODEL_PATH="${JUDGE_MODEL_PATH:-${VLLM_JUDGE_MODEL_PATH:-$HF_HOME/Qwen/Qwen3.5-27B}}"
  export CHARXIV_JUDGE_GPU_MEMORY_UTILIZATION="${CHARXIV_JUDGE_GPU_MEMORY_UTILIZATION:-0.9}"
  export CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE="${CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE:-1}"
  export CHARXIV_JUDGE_MAX_MODEL_LEN="${CHARXIV_JUDGE_MAX_MODEL_LEN:-18432}"
  export VLLM_JUDGE_MAX_NUM_SEQS="${VLLM_JUDGE_MAX_NUM_SEQS:-32}"
  unset VLLM_SERVER_JUDGE VLLM_JUDGE_BASE_URL VLLM_JUDGE_PORT 2>/dev/null || true
  MANAGE_VLLM_SERVER_LIFECYCLE=true
fi

# ── build model_args ──────────────────────────────────────────────────
MODEL_ARGS="model_version=${MODEL_VERSION},base_url=http://localhost:${PORT}/v1,api_key=${API_KEY},max_tool_iterations=${MAX_TOOL_ITERATIONS},max_tool_images=${MAX_TOOL_IMAGES}"
[[ "$MANAGE_VLLM_SERVER_LIFECYCLE" == true ]] && MODEL_ARGS="${MODEL_ARGS},manage_vllm_server_lifecycle=true"
[[ -n "$MCP_SERVER_PATH" ]] && MODEL_ARGS="${MODEL_ARGS},mcp_server_path=${MCP_SERVER_PATH}"
[[ -n "$SYSTEM_PROMPT"   ]] && MODEL_ARGS="${MODEL_ARGS},system_prompt=${SYSTEM_PROMPT}"
case "${TOOL_IMAGE_AS_USER_TURN:-false}" in
  1|true|yes|on) MODEL_ARGS="${MODEL_ARGS},tool_image_as_user_turn=true" ;;
esac
FORCED_ACTION_ENABLED=false
case "${FORCE_TOOL_CALLS_UNTIL_FINAL_TURN,,}" in
  1|true|yes|on)
    MODEL_ARGS="${MODEL_ARGS},force_tool_calls_until_final_turn=true"
    FORCED_ACTION_ENABLED=true
    ;;
esac
case "${FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS,,}" in
  1|true|yes|on)
    MODEL_ARGS="${MODEL_ARGS},force_answer_after_max_tool_iterations=true"
    FORCED_ACTION_ENABLED=true
    ;;
esac
if [[ "$FORCED_ACTION_ENABLED" == true && -n "$FORCE_FINAL_ANSWER_PREFIX" ]]; then
  MODEL_ARGS="${MODEL_ARGS},force_final_answer_prefix=${FORCE_FINAL_ANSWER_PREFIX}"
fi
case "${CRITIC_GUIDED_ANSWER_RETRY,,}" in
  1|true|yes|on)
    MODEL_ARGS="${MODEL_ARGS},critic_guided_answer_retry=true"
    MODEL_ARGS="${MODEL_ARGS},critic_guided_answer_retry_source=${CRITIC_GUIDED_ANSWER_RETRY_SOURCE}"
    MODEL_ARGS="${MODEL_ARGS},critic_guided_answer_retry_mode=${CRITIC_GUIDED_ANSWER_RETRY_MODE}"
    MODEL_ARGS="${MODEL_ARGS},critic_guided_answer_retry_threshold=${CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD}"
    MODEL_ARGS="${MODEL_ARGS},critic_guided_answer_retry_prefill_profile=${CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE}"
    if [[ "$CRITIC_GUIDED_ANSWER_RETRY_MODE" == reasoning ]]; then
      MODEL_ARGS="${MODEL_ARGS},critic_guided_answer_retry_temperature=${REPLAY_CRITIC_TEMPERATURE:-0}"
      MODEL_ARGS="${MODEL_ARGS},critic_guided_answer_retry_top_k=${REPLAY_CRITIC_TOP_K:--1}"
      MODEL_ARGS="${MODEL_ARGS},critic_guided_answer_retry_top_p=${REPLAY_CRITIC_TOP_P:-1}"
    fi
    ;;
esac

if [[ "${FIRST_CANDIDATE_ANSWER_TURN:-0}" != 0 ]]; then
  MODEL_ARGS="${MODEL_ARGS},first_candidate_answer_turn=${FIRST_CANDIDATE_ANSWER_TURN}"
fi
case "${ALWAYS_RETRY_ANSWERS:-false}" in
  1|true|yes|on) MODEL_ARGS="${MODEL_ARGS},always_retry_answers=true" ;;
esac
if [[ "${FOLLOWUP_CANDIDATE_ANSWER_TURN:-0}" != 0 ]]; then
  MODEL_ARGS="${MODEL_ARGS},followup_candidate_answer_turn=${FOLLOWUP_CANDIDATE_ANSWER_TURN}"
fi
case "${RECORD_ANSWER_CANDIDATES:-false}" in
  1|true|yes|on) MODEL_ARGS="${MODEL_ARGS},record_answer_candidates=true" ;;
esac

echo "Tasks:      $TASKS"
echo "Output:     $OUTPUT_PATH"
echo "Model args: $MODEL_ARGS"
echo "Final think prefill: ${FORCE_FINAL_ANSWER_THINK_PREFILL:-<empty>}"
echo "Critic retry: enabled=$CRITIC_GUIDED_ANSWER_RETRY threshold=$CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD prefill_profile=$CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE"
echo "Critic domain thresholds: ${VERO_EVAL_CRITIC_DOMAIN_THRESHOLDS:-<none; using fallback>}"
echo "First candidate deadline: ${FIRST_CANDIDATE_ANSWER_TURN:-0} (0=disabled); record candidates: ${RECORD_ANSWER_CANDIDATES:-false}"
echo ""

python -m lmms_eval \
    --model async_openai \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --log_samples \
    --output_path "$OUTPUT_PATH" \
    --offset "$OFFSET" \
    ${LIMIT:+--limit "$LIMIT"}
