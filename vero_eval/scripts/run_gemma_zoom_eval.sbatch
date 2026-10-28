#!/usr/bin/env bash
# Resource requests are provided by examples/submit_zoom_eval_gemma_visual_probe.sh.
set -euo pipefail
: "${VERO_EVAL_DIR:?}" "${GEMMA_FAMILY:?}" "${CHECKPOINT_PATH:?}" "${OUTPUT_ROOT:?}"
cd "$VERO_EVAL_DIR"
export ROOT_PATH="${ROOT_PATH:-/scratch/gpfs/ZHUANGL/gs1693}"
export GEMMA_TRANSFORMERS_OVERLAY="${GEMMA_TRANSFORMERS_OVERLAY:-$ROOT_PATH/vero2/runtime/gemma_zoom_transformers}"
export PYTHONPATH="$GEMMA_TRANSFORMERS_OVERLAY:$VERO_EVAL_DIR/..${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export RUN_SYSTEM_PROMPT_MODE=task_metadata SYSTEM_PROMPT=''
export VERO_TRAINING_CONFIG='' TRAINING_SYSTEM_PROMPT=''
export FORCE_TOOL_CALLS_UNTIL_FINAL_TURN=false FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS=false
export FORCE_FINAL_ANSWER_THINK_PREFILL='' FORCE_FINAL_ANSWER_PREFIX=''
export CRITIC_GUIDED_ANSWER_RETRY=false CRITIC_GUIDED_ANSWER_RETRY_SOURCE=policy
export FIRST_CANDIDATE_ANSWER_TURN=0 FOLLOWUP_CANDIDATE_ANSWER_TURN=0 ALWAYS_RETRY_ANSWERS=false
export MODEL="$CHECKPOINT_PATH" TOOLS=on
case "${GEMMA_EVAL_MODE:-zoom}" in
  zoom) ;;
  thinking_only)
    [[ "$GEMMA_FAMILY" == gemma4_e4b ]] || { echo 'thinking_only requires Gemma 4' >&2; exit 1; }
    export GEMMA4_ENABLE_THINKING=true TOOLS=off MCP_SERVER_PATH=''
    export MAX_TOOL_ITERATIONS=0 MAX_TOOL_IMAGES=0
    ;;
  *) echo 'GEMMA_EVAL_MODE must be zoom or thinking_only' >&2; exit 1 ;;
esac
export MODEL_NAME="$GEMMA_FAMILY" MODEL_VERSION="$GEMMA_FAMILY"
export NUM_GPUS=1 MAX_MODEL_LEN="${MAX_MODEL_LEN:-49152}"
export MAX_TOOL_IMAGES="${MAX_TOOL_IMAGES:-10}" MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERATIONS:-10}"
export TOOL_IMAGE_AS_USER_TURN=true
export LIMIT_MM_PER_PROMPT='{"image":11,"audio":0,"video":0}' VLLM_MAX_NUM_SEQS=4
export JUDGE_BACKEND=engine API_TYPE=vllm
export JUDGE_MODEL_PATH="${VLLM_JUDGE_MODEL_PATH:-$ROOT_PATH/hf_cache/Qwen/Qwen3.5-27B}"
export VLLM_JUDGE_MODEL_PATH="$JUDGE_MODEL_PATH"
export VLLM_JUDGE_CHAT_TEMPLATE="$JUDGE_MODEL_PATH/chat_template_no_think.jinja"
export VLLM_JUDGE_ENFORCE_EAGER=true VLLM_JUDGE_GDN_PREFILL_BACKEND=triton
export VLLM_USE_FLASHINFER_SAMPLER=0 LMMS_EVAL_SAVE_PREJUDGE_SAMPLES=true
export WORKERS="${WORKERS:-2}" BATCH_SIZE=1
export LMMS_EVAL_REQUEST_TRACE_DIR="$OUTPUT_ROOT/request_traces"
export LOG_DIR="$OUTPUT_ROOT/runtime_logs"
# vLLM appends a UUID to this directory for Unix-domain IPC sockets. Putting
# it under OUTPUT_ROOT exceeds Linux's 107-byte socket-path limit.
export TMPDIR="$(mktemp -d "/tmp/gemma.${SLURM_JOB_ID:?}.XXXXXX")"
export VLLM_RPC_BASE_PATH="$TMPDIR"
mkdir -p "$LOG_DIR"
echo "TMPDIR=$TMPDIR VLLM_RPC_BASE_PATH=$VLLM_RPC_BASE_PATH"
unset CUSTOM_CHAT_TEMPLATE REASONING_PARSER
export LMMS_EVAL_PRESERVE_REASONING=false
case "$GEMMA_FAMILY" in
  gemma3_4b)
    export TOOL_CALL_PARSER=hermes CHAT_TEMPLATE="$VERO_EVAL_DIR/prompts/gemma3_zoom_tools.jinja"
    export MM_PROCESSOR_KWARGS='{"do_pan_and_scan":false}' DEFAULT_CHAT_TEMPLATE_KWARGS='{}'
    ;;
  gemma4_e4b)
    export TOOL_CALL_PARSER=gemma4 CHAT_TEMPLATE="$CHECKPOINT_PATH/chat_template.jinja"
    export MM_PROCESSOR_KWARGS='{"max_soft_tokens":1120}' DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'
    case "${GEMMA4_ENABLE_THINKING:-false}" in
      true|1|yes|on)
        export GEMMA4_ENABLE_THINKING=true REASONING_PARSER=gemma4
        export DEFAULT_CHAT_TEMPLATE_KWARGS='{"enable_thinking":true,"preserve_thinking":true}'
        export LMMS_EVAL_PRESERVE_REASONING=true
        export MODEL_NAME=gemma4_e4b_thinking MODEL_VERSION=gemma4_e4b_thinking
        ;;
      false|0|no|off) export GEMMA4_ENABLE_THINKING=false ;;
      *) echo 'GEMMA4_ENABLE_THINKING must be true or false' >&2; exit 1 ;;
    esac
    ;;
  *) echo "Unknown GEMMA_FAMILY=$GEMMA_FAMILY" >&2; exit 1 ;;
esac
if [[ "${GEMMA_EVAL_MODE:-zoom}" == thinking_only ]]; then
  export MODEL_NAME=gemma4_e4b_thinking_only MODEL_VERSION=gemma4_e4b_thinking_only
  export LIMIT_MM_PER_PROMPT='{"image":1,"audio":0,"video":0}'
fi
# Job-specific ports also prevent collision when two tests share a node.
export PORT=$((20000 + SLURM_JOB_ID % 20000))
set +u
source "$VERO_EVAL_DIR/set_paths.sh"
set -u
python scripts/preflight_gemma_zoom.py --model "$GEMMA_FAMILY" --checkpoint "$CHECKPOINT_PATH" --tasks "$TASKS" --output "$OUTPUT_ROOT/protocol"
if [[ -n "${GEMMA_TRAINING_CONFIG:-}" ]]; then
  python scripts/prepare_gemma4_training_eval.py --environment > "$OUTPUT_ROOT/protocol/environment.sh"
  source "$OUTPUT_ROOT/protocol/environment.sh"
fi
bash run_eval.sbatch
python scripts/summarize_gemma_zoom_test.py "$OUTPUT_ROOT"
