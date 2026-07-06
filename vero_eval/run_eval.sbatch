#!/bin/bash
#SBATCH --job-name=vero_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --output=logs/vero_eval_%j.out
#SBATCH --error=logs/vero_eval_%j.err
#
# ============================================================================
# ONE entrypoint — pick MODEL, TASKS, TOOLS, submit from vero_eval/:
#
#   sbatch --export=ALL,MODEL=qwen3vl,TASKS=countqa_reasoning,visual_probe_easy_qwen3_zs,TOOLS=on \
#          run_eval.sbatch
#
#   sbatch --export=ALL,MODEL=vero,CHECKPOINT_PATH=/scratch/.../actor/huggingface,TASKS=all,TOOLS=on \
#          run_eval.sbatch
#
#   # exact-match-only run (no judge) -> 1 GPU is enough:
#   sbatch --gres=gpu:1 --export=ALL,MODEL=qwen3vl,TASKS=countqa_reasoning,TOOLS=on \
#          run_eval.sbatch
# ----------------------------------------------------------------------------
# MODEL  : vero | vero-thinking | qwen3vl | qwen3vl-thinking | qwen25vl | <abs ckpt path>
#          Sets CHECKPOINT_PATH + the default task SUFFIX. For Vero the reasoning system
#          prompt is baked into its chat template, so none is passed.
# TASKS  : a comma-list of task NAMES (countqa_reasoning,chartqa_reasoning,...)  OR a DOMAIN
#          (chart_ocr|stem|spatial_action|knowledge_rec|grounding_counting_search|
#           instruction_following|ball_drop|all). For a domain, SUFFIX is appended per task.
# TOOLS  : on (default) | off              enable/disable the MCP visual tools
# SUFFIX : override the per-task suffix for a DOMAIN run (default per MODEL)
# Other  : LIMIT, MAX_TOOL_ITERATIONS (turns, default 10), JUDGE_BACKEND, SYSTEM_PROMPT, OUTPUT_ROOT
#
# Judge: a judge task auto-launches a dedicated Qwen3-32B judge on GPU1 (the reportable
# judge). Submit with --gres=gpu:1 for exact-match-only runs (e.g. countqa).
# ============================================================================
set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$SUBMIT_DIR/set_paths.sh.example" && "$(basename "$SUBMIT_DIR")" == "vero_eval" ]]; then
  VERO_EVAL_DIR="$SUBMIT_DIR"
else
  VERO_EVAL_DIR="$SUBMIT_DIR/vero_eval"
fi
[[ -f "$VERO_EVAL_DIR/set_paths.sh.example" ]] || { echo "ERROR: $VERO_EVAL_DIR not found (submit from /home/gs1693/repo/vero2/vero_eval)" >&2; exit 1; }
set +u
[[ -f "$VERO_EVAL_DIR/set_paths.sh" ]] && source "$VERO_EVAL_DIR/set_paths.sh" || source "$VERO_EVAL_DIR/set_paths.sh.example"
set -u
source "$VERO_EVAL_DIR/examples/_domain_map.sh"

: "${TASKS:?set TASKS=<task1,task2,...> or a domain (e.g. TASKS=grounding_counting_search)}"
MODEL="${MODEL:-qwen3vl}"; TOOLS="${TOOLS:-on}"; SUFFIX="${SUFFIX:-}"; LIMIT="${LIMIT:-}"
PORT="${PORT:-9001}"; API_KEY="${API_KEY:-qwen}"
export NUM_GPUS="${NUM_GPUS:-1}"   # policy uses 1 GPU; a judge task gets the 2nd
MERGE_BACKEND="${MERGE_BACKEND:-fsdp}"
export FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS="${FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS:-false}"
export FORCE_FINAL_ANSWER_PREFIX="${FORCE_FINAL_ANSWER_PREFIX:-}"
DEFAULT_FORCE_FINAL_ANSWER_THINK_PREFILL="This is the final turn, so I need to answer from the evidence gathered so far."
export FORCE_FINAL_ANSWER_THINK_PREFILL="${FORCE_FINAL_ANSWER_THINK_PREFILL:-$DEFAULT_FORCE_FINAL_ANSWER_THINK_PREFILL}"

has_hf_weights() {
  local hf_dir="$1"
  shopt -s nullglob
  local -a weights=("$hf_dir"/*.safetensors "$hf_dir"/*.bin)
  shopt -u nullglob
  (( ${#weights[@]} > 0 ))
}

ensure_hf_checkpoint_path() {
  local input_path="$1"
  local actor_dir=""
  local hf_target=""

  if [[ -d "$input_path/actor" ]]; then
    actor_dir="$input_path/actor"
    hf_target="$actor_dir/huggingface"
  elif [[ "$(basename "$input_path")" == "huggingface" && "$(basename "$(dirname "$input_path")")" == "actor" ]]; then
    hf_target="$input_path"
    actor_dir="$(dirname "$hf_target")"
  elif [[ "$(basename "$input_path")" == "actor" ]]; then
    actor_dir="$input_path"
    hf_target="$actor_dir/huggingface"
  else
    printf '%s\n' "$input_path"
    return 0
  fi

  mkdir -p "$hf_target"
  if has_hf_weights "$hf_target"; then
    echo "[INFO] Found Hugging Face weights in $hf_target; skipping conversion." >&2
    printf '%s\n' "$hf_target"
    return 0
  fi

  if ! compgen -G "$actor_dir/model_world_size_*_rank_*.pt" >/dev/null 2>&1; then
    echo "[WARN] No HF weights in $hf_target and no FSDP shards in $actor_dir; using original path." >&2
    printf '%s\n' "$input_path"
    return 0
  fi

  merge_hf_weights() {
    if has_hf_weights "$hf_target"; then
      echo "[INFO] Found Hugging Face weights in $hf_target after waiting for lock; skipping conversion." >&2
      return 0
    fi
    echo "[INFO] Hugging Face weights not detected at $hf_target. Converting from FSDP checkpoint." >&2
    python -m verl.model_merger merge \
      --backend "$MERGE_BACKEND" \
      --local_dir "$actor_dir" \
      --target_dir "$hf_target" \
      --trust-remote-code >&2
    if ! has_hf_weights "$hf_target"; then
      echo "[ERROR] Failed to produce Hugging Face weights at $hf_target" >&2
      return 1
    fi
  }

  if command -v flock >/dev/null 2>&1; then
    (
      exec 200>"$actor_dir/.merge.lock"
      flock 200
      merge_hf_weights
    )
  else
    merge_hf_weights
  fi

  printf '%s\n' "$hf_target"
}

# ── 1) MODEL -> CHECKPOINT_PATH + default SUFFIX + system-prompt policy ──
case "$MODEL" in
  vero)             : "${CHECKPOINT_PATH:?MODEL=vero needs CHECKPOINT_PATH=<vero ckpt>}"; SUFFIX="${SUFFIX:-reasoning}"; export SYSTEM_PROMPT="${SYSTEM_PROMPT:-}";;
  vero-thinking)    : "${CHECKPOINT_PATH:?MODEL=vero-thinking needs CHECKPOINT_PATH=<ckpt>}"; SUFFIX="${SUFFIX:-reasoning_samplingq3}"; export SYSTEM_PROMPT="${SYSTEM_PROMPT:-}";;
  qwen3vl)          export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/scratch/gpfs/ZHUANGL/hw0171/Qwen3-VL-8B-Instruct}"; SUFFIX="${SUFFIX:-qwen3_zs}";;
  qwen3vl-thinking) : "${CHECKPOINT_PATH:?MODEL=qwen3vl-thinking needs CHECKPOINT_PATH=<Qwen3-VL-Thinking>}"; SUFFIX="${SUFFIX:-qwen3_thinking_zs}";;
  qwen25vl)         : "${CHECKPOINT_PATH:?MODEL=qwen25vl needs CHECKPOINT_PATH=<Qwen2.5-VL>}"; SUFFIX="${SUFFIX:-qwen25_zs}";;
  /*)               export CHECKPOINT_PATH="$MODEL"; SUFFIX="${SUFFIX:-reasoning}";;
  *)                echo "ERROR: unknown MODEL '$MODEL' (vero|vero-thinking|qwen3vl|qwen3vl-thinking|qwen25vl|<abs path>)" >&2; exit 1;;
esac
CHECKPOINT_PATH="$(ensure_hf_checkpoint_path "$CHECKPOINT_PATH")"
export LMMS_EVAL_HF_TOKENIZER="${LMMS_EVAL_HF_TOKENIZER:-$VERO_EVAL_DIR/local_hf/Qwen3-8B}"
export LMMS_EVAL_HF_TOKENIZER_LOCAL_ONLY="${LMMS_EVAL_HF_TOKENIZER_LOCAL_ONLY:-1}"
export CHECKPOINT_PATH PORT API_KEY
export MODEL_NAME="${MODEL_NAME:-qwen_vllm}" MODEL_VERSION="${MODEL_VERSION:-qwen_vllm}"

# Vero RL checkpoints must be SERVED with their forced-think TRAINING chat template
# (it ends each assistant turn with '<think>\n' + renders the tool-call format). vLLM
# otherwise uses tokenizer_config.json's base template and Vero skips <think>/tools and
# answers directly. Default CHAT_TEMPLATE to the checkpoint's chat_template.jinja.
if [[ "$MODEL" == vero || "$MODEL" == vero-thinking ]] && [[ -z "${CHAT_TEMPLATE:-}" ]] && [[ -f "$CHECKPOINT_PATH/chat_template.jinja" ]]; then
  export CHAT_TEMPLATE="$CHECKPOINT_PATH/chat_template.jinja"
  echo "MODEL=$MODEL -> serving with forced-think chat template: $CHAT_TEMPLATE"
fi

if [[ "$MODEL" == vero || "$MODEL" == vero-thinking ]]; then
  export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
  export MAX_TOOL_IMAGES="${MAX_TOOL_IMAGES:-10}"
  export MM_MAX_PIXELS="${MM_MAX_PIXELS:-10000000}"
  export MM_MIN_PIXELS="${MM_MIN_PIXELS-}"
fi

# ── 2) TOOLS on/off ──
if [[ "$TOOLS" == "off" ]]; then export MCP_SERVER_PATH=""; else export MCP_SERVER_PATH="${MCP_SERVER_PATH:-lmms_eval/mcp/vero2_mcp.py}"; fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$VERO_EVAL_DIR/results/${MODEL//\//_}_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$VERO_EVAL_DIR/logs"; cd "$VERO_EVAL_DIR"
echo "MODEL=$MODEL  CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "TASKS=$TASKS  SUFFIX=$SUFFIX  TOOLS=$TOOLS  OUTPUT_ROOT=$OUTPUT_ROOT"
echo "FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS=$FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS"

# ── 3) serve the policy (GPU0) ──
bash scripts/serve_qwen_vllm.sh
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
  || { echo "ERROR: vLLM policy not up on :$PORT"; exit 1; }

# ── 4) TASKS = a domain (or 'all') -> loop domains; else a literal task list ──
if [[ "$TASKS" == "all" || -n "${DOMAIN_TASK_BASES[$TASKS]:-}" ]]; then
  bash examples/eval_across_domains.sh --no-serve --domains "$TASKS" --model-suffix "$SUFFIX" \
    --output-root "$OUTPUT_ROOT" ${LIMIT:+--limit "$LIMIT"}
else
  bash examples/eval_tasks.sh --no-serve --tasks "$TASKS" \
    --output-path "$OUTPUT_ROOT" ${LIMIT:+--limit "$LIMIT"}
fi
echo "=== Done. Results under: $OUTPUT_ROOT ==="
