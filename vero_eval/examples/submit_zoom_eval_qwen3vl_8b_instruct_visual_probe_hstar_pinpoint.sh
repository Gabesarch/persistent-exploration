#!/usr/bin/env bash
# Submit two SLURM eval jobs for the base Qwen3-VL-8B-Instruct model.
# The default task set covers VisualProbe Easy/Medium/Hard plus the primary
# Vero zoom tasks for HStarBench, Pinpoint Bench, and ScreenSpot-Pro. The cached
# Qwen3.5-27B model is used as an in-process sequential LLM judge where a task
# requests it; ScreenSpot-Pro retains geometric point-in-box judging.
# The two jobs use the same base model and zoom tasks: one runs normal inference,
# and one forces tool calls through MAX_TOOL_ITERATIONS before forcing an answer.
#
# Usage, from /home/gs1693/repo/vero2/vero_eval:
#   DRY_RUN=1 bash examples/submit_zoom_eval_qwen3vl_8b_instruct_visual_probe_hstar_pinpoint.sh
#   bash examples/submit_zoom_eval_qwen3vl_8b_instruct_visual_probe_hstar_pinpoint.sh
#
# Common overrides:
#   EVAL_MODE=normal bash ...          # submit only normal inference
#   EVAL_MODE=budget_forcing bash ...  # submit only forced tool calls + final answer
#   MAX_TOOL_ITERATIONS=10 bash ...    # budget-forcing tool turns before the answer
#   TASK_GROUP=visual_probe bash ...
#   TASKS=task1,task2 bash ...
#   BASE_MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct bash ...
#   EVAL_PROFILE=debug bash ...   # default GPU test queue/gpu80, 1h, limit=100, 1 GPU
#   EVAL_PROFILE=full bash ...    # default pli-c/h100 config
#   EVAL_PROFILE=vlmrl bash ...   # pli partition with vlmrl account
#   EVAL_PROFILE=ailab bash ...   # ailab partition with no account
#   MERGE_BACKEND=fsdp bash ...   # convert FSDP shards to HF if actor/huggingface lacks weights
#   PORT_OFFSET=200 bash ...      # optional stable offset added to all per-job ports
#   SKIP_IF_EXISTS=false SKIP_IF_QUEUED=false DRY_RUN=1 bash ...
set -euo pipefail

VERO_EVAL_DIR="$(pwd)"
if [[ ! -f "$VERO_EVAL_DIR/run_eval.sbatch" || ! -d "$VERO_EVAL_DIR/lmms_eval" ]]; then
  echo "ERROR: run this script from /home/gs1693/repo/vero2/vero_eval:" >&2
  echo "  bash examples/submit_zoom_eval_qwen3vl_8b_instruct_visual_probe_hstar_pinpoint.sh" >&2
  exit 1
fi

# -------------------------
# Runtime/config knobs
# -------------------------
EVAL_NAME_EXPLICIT="${EVAL_NAME+x}"
EVAL_NAME="${EVAL_NAME:-qwen3vl_8b_instruct}"
EVAL_MODE="${EVAL_MODE:-all}"  # all|normal|budget_forcing
TASK_GROUP_EXPLICIT="${TASK_GROUP+x}"
TASK_GROUP="${TASK_GROUP:-visual_probe_hstar_pinpoint}"
TRAIN_DIR_EXPLICIT="${TRAIN_DIR+x}"
TASKS_EXPLICIT="${TASKS+x}"
OUTPUT_ROOT_EXPLICIT="${OUTPUT_ROOT+x}"
LIMIT="${LIMIT:-}"
STRICT_TASKS="${STRICT_TASKS:-1}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
RUN_SYSTEM_PROMPT_MODE="${RUN_SYSTEM_PROMPT_MODE:-task_metadata}"  # task_metadata|run_env|none

MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-10000000}"
MAX_TOOL_IMAGES="${MAX_TOOL_IMAGES:-10}"
MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERATIONS:-10}"
export WORKERS="${WORKERS:-10}"
ACCELERATE_PORT_BASE="${ACCELERATE_PORT_BASE:-12346}"
POLICY_PORT_BASE="${POLICY_PORT_BASE:-10001}"
PORT_OFFSET="${PORT_OFFSET:-}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-}"
MAX_STEPS_EVAL="${MAX_STEPS_EVAL:-}"
USE_MOST_RECENT_MAX_CHECKPOINTS="${USE_MOST_RECENT_MAX_CHECKPOINTS:-false}"
ONLY_EVAL_EVERY_X="${ONLY_EVAL_EVERY_X:-}"
ONLY_EVAL_GLOBAL_STEPS="${ONLY_EVAL_GLOBAL_STEPS:-}"
OVERRIDE_EXISTING="${OVERRIDE_EXISTING:-true}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-true}"
SKIP_IF_QUEUED="${SKIP_IF_QUEUED:-true}"
DRY_RUN="${DRY_RUN:-0}"
MEM_OVERRIDE="${MEM_OVERRIDE:-}"
TIME_OVERRIDE="${TIME_OVERRIDE:-}"
DEFAULT_MEM="${DEFAULT_MEM:-180G}"
MERGE_BACKEND="${MERGE_BACKEND:-fsdp}"
FORCE_FINAL_ANSWER_PREFIX="${FORCE_FINAL_ANSWER_PREFIX:-}"
DEFAULT_FORCE_FINAL_ANSWER_THINK_PREFILL="This is the final turn, so I need to answer from the evidence gathered so far."
FORCE_FINAL_ANSWER_THINK_PREFILL="${FORCE_FINAL_ANSWER_THINK_PREFILL:-$DEFAULT_FORCE_FINAL_ANSWER_THINK_PREFILL}"
CRITIC_GUIDED_ANSWER_RETRY_EXPLICIT="${CRITIC_GUIDED_ANSWER_RETRY+x}"
CRITIC_GUIDED_ANSWER_RETRY="${CRITIC_GUIDED_ANSWER_RETRY:-false}"
CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD="${CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD:-0.5}"
VLLM_JUDGE_MODEL_PATH="${VLLM_JUDGE_MODEL_PATH:-/scratch/gpfs/ZHUANGL/gs1693/hf_cache/Qwen/Qwen3.5-27B}"
VLLM_JUDGE_CHAT_TEMPLATE="${VLLM_JUDGE_CHAT_TEMPLATE:-$VLLM_JUDGE_MODEL_PATH/chat_template_no_think.jinja}"
# Export this through sbatch so the parent evaluator and its in-process judge
# both avoid FlashInfer's node-local CUDA JIT dependency.
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/scratch/gpfs/ZHUANGL/hw0171/Qwen3-VL-8B-Instruct}"

if [[ "${EVAL_MODE,,}" == "all" ]]; then
  failed_modes=()
  for eval_mode in normal budget_forcing; do
    child_env=(EVAL_MODE="$eval_mode")
    if [[ -n "$OUTPUT_ROOT_EXPLICIT" && -n "$OUTPUT_ROOT" ]]; then
      child_env+=(OUTPUT_ROOT="${OUTPUT_ROOT%/}/$eval_mode")
    fi
    if ! env "${child_env[@]}" bash "$0"; then
      failed_modes+=("$eval_mode")
    fi
  done
  if [[ ${#failed_modes[@]} -gt 0 ]]; then
    echo "ERROR: eval submission failed for modes: ${failed_modes[*]}" >&2
    exit 1
  fi
  echo "Completed base-model submissions for normal and budget-forcing inference."
  exit 0
fi

case "${EVAL_MODE,,}" in
  normal)
    EVAL_MODE_TAG="normal"
    FORCE_TOOL_CALLS_UNTIL_FINAL_TURN=false
    FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS=false
    ;;
  budget_forcing)
    EVAL_MODE_TAG="budget_forcing_t${MAX_TOOL_ITERATIONS}"
    FORCE_TOOL_CALLS_UNTIL_FINAL_TURN=true
    # force_tool_calls_until_final_turn itself schedules the separate forced
    # <answer> request once MAX_TOOL_ITERATIONS tool calls have completed.
    FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS=false
    ;;
  *)
    echo "ERROR: EVAL_MODE must be 'all', 'normal', or 'budget_forcing' (got '$EVAL_MODE')." >&2
    exit 1
    ;;
esac

TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING="/home/gs1693/repo/vero2/vero2_scripts/prompts/system_prompt_chatting_tool_calling.txt"

declare -A TRAIN_DIR_BY_EVAL_NAME=(
  [qwen3vl_8b_instruct]="$BASE_MODEL_PATH"
  [branchstrata20_fresh_lr1e5_critic]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_lr1e5_critic_20260810"
  # Training SLURM continuation array 12794189 (tasks 1-10 share this directory).
  [branchstrata20_fresh_lr1e5_critic_zoom_joint3]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_lr1e5_critic_zoom_joint3_20260821"
  [hardonly_llmjudge_lr5e6]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/GDPO_HDPOtool_GSPO_fp16_math_verify_chart_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_llmjudge_lr5e6_20260524_160610"
)

declare -A SYSTEM_PROMPT_BY_EVAL_NAME=(
  [qwen3vl_8b_instruct]=""
  [branchstrata20_fresh_lr1e5_critic]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh_lr1e5_critic_zoom_joint3]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [hardonly_llmjudge_lr5e6]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
)

declare -A EVAL_NAMES_BY_GROUP=(
  [branch_comparison]="branchstrata20_fresh_lr1e5_critic branchstrata20_fresh_lr1e5_critic_zoom_joint3 hardonly_llmjudge_lr5e6"
)

declare -A CRITIC_RETRY_PROFILE_BY_EVAL_NAME=(
  [branchstrata20_fresh_lr1e5_critic]="persist"
  [branchstrata20_fresh_lr1e5_critic_zoom_joint3]="persist"
)

CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE="${CRITIC_RETRY_PROFILE_BY_EVAL_NAME[$EVAL_NAME]:-}"
# Critic-trained eval configurations always reproduce their answer-retry policy.
# Fail fast instead of silently allowing an accidental false override.
if [[ -n "$CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE" ]]; then
  if [[ -n "$CRITIC_GUIDED_ANSWER_RETRY_EXPLICIT" ]]; then
    case "${CRITIC_GUIDED_ANSWER_RETRY,,}" in
      1|true|yes|on) ;;
      *)
        echo "ERROR: EVAL_NAME=$EVAL_NAME is critic-trained and requires CRITIC_GUIDED_ANSWER_RETRY=true." >&2
        exit 1
        ;;
    esac
  fi
  CRITIC_GUIDED_ANSWER_RETRY=true
fi

CRITIC_RETRY_ENABLED=false
case "${CRITIC_GUIDED_ANSWER_RETRY,,}" in
  1|true|yes|on) CRITIC_RETRY_ENABLED=true ;;
  0|false|no|off) ;;
  *)
    echo "ERROR: CRITIC_GUIDED_ANSWER_RETRY must be true or false (got '$CRITIC_GUIDED_ANSWER_RETRY')." >&2
    exit 1
    ;;
esac
if [[ "$CRITIC_RETRY_ENABLED" == true ]] && ! python3 - "$CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD" <<'PY'
import math
import sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 0.0 <= value <= 1.0 else 1)
PY
then
  echo "ERROR: CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD must be a finite number in [0, 1]." >&2
  exit 1
fi
if [[ "$CRITIC_RETRY_ENABLED" == true && -z "$CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE" ]]; then
  echo "ERROR: critic-guided answer retry is only supported for mapped critic EVAL_NAME values: ${!CRITIC_RETRY_PROFILE_BY_EVAL_NAME[*]} (got '$EVAL_NAME')." >&2
  exit 1
fi

eval_group_key="${EVAL_NAME,,}"
if [[ -n "${EVAL_NAMES_BY_GROUP[$eval_group_key]:-}" ]]; then
  eval_group_members="${EVAL_NAMES_BY_GROUP[$eval_group_key]:-}"
  if [[ -n "$TRAIN_DIR_EXPLICIT" ]]; then
    echo "ERROR: TRAIN_DIR cannot be combined with grouped EVAL_NAME=$EVAL_NAME." >&2
    exit 1
  fi

  read -r -a GROUP_EVAL_NAMES <<< "$eval_group_members"
  failed_eval_names=()
  skipped_eval_names=()
  submitted_eval_names=()
  for eval_name in "${GROUP_EVAL_NAMES[@]}"; do
    train_dir="${TRAIN_DIR_BY_EVAL_NAME[$eval_name]:-}"
    if [[ ! -d "$train_dir" ]]; then
      echo "[INFO] Skipping EVAL_NAME=$eval_name (training directory does not exist yet: $train_dir)."
      skipped_eval_names+=("$eval_name")
      continue
    fi
    mapfile -t group_checkpoints < <(find "$train_dir" -maxdepth 1 -type d \( \
      -regex '.*/checkpoint-[0-9]+$' -o \
      -regex '.*/checkpoint_[0-9]+$' -o \
      -regex '.*/global_step_[0-9]+$' \
    \) -print -quit)
    if [[ ${#group_checkpoints[@]} -eq 0 ]]; then
      echo "[INFO] Skipping EVAL_NAME=$eval_name (no checkpoints available yet under $train_dir)."
      skipped_eval_names+=("$eval_name")
      continue
    fi

    echo
    echo "========== EVAL_NAME=$eval_group_key MEMBER=$eval_name =========="
    child_env=(EVAL_NAME="$eval_name" EVAL_PROFILE="$EVAL_PROFILE")
    if [[ -n "$OUTPUT_ROOT_EXPLICIT" && -n "$OUTPUT_ROOT" ]]; then
      child_env+=(OUTPUT_ROOT="${OUTPUT_ROOT%/}/$eval_name")
    fi

    if env "${child_env[@]}" bash "$0"; then
      submitted_eval_names+=("$eval_name")
    else
      failed_eval_names+=("$eval_name")
    fi
  done

  if [[ ${#failed_eval_names[@]} -gt 0 ]]; then
    echo "ERROR: eval submission failed for group members: ${failed_eval_names[*]}" >&2
    exit 1
  fi
  echo "Completed grouped EVAL_NAME=$eval_group_key: processed ${#submitted_eval_names[@]}, skipped unavailable ${#skipped_eval_names[@]}."
  exit 0
fi

if [[ "$EVAL_NAME" == "all" ]]; then
  if [[ -n "$TRAIN_DIR_EXPLICIT" ]]; then
    echo "ERROR: TRAIN_DIR cannot be combined with EVAL_NAME=all; 'all' uses every entry in TRAIN_DIR_BY_EVAL_NAME." >&2
    exit 1
  fi

  mapfile -t ALL_EVAL_NAMES < <(printf '%s\n' "${!TRAIN_DIR_BY_EVAL_NAME[@]}" | sort)
  failed_eval_names=()
  for eval_name in "${ALL_EVAL_NAMES[@]}"; do
    echo
    echo "========== EVAL_NAME=$eval_name =========="

    child_env=(EVAL_NAME="$eval_name")
    # Keep an explicit output root collision-free when dispatching multiple runs.
    if [[ -n "$OUTPUT_ROOT_EXPLICIT" && -n "$OUTPUT_ROOT" ]]; then
      child_env+=(OUTPUT_ROOT="${OUTPUT_ROOT%/}/$eval_name")
    fi

    if ! env "${child_env[@]}" bash "$0"; then
      failed_eval_names+=("$eval_name")
    fi
  done

  if [[ ${#failed_eval_names[@]} -gt 0 ]]; then
    echo "ERROR: eval submission failed for: ${failed_eval_names[*]}" >&2
    exit 1
  fi
  echo "Completed checkpoint checks/submissions for ${#ALL_EVAL_NAMES[@]} eval configurations."
  exit 0
fi

declare -A TASKS_BY_TASK_GROUP=(
  [visual_probe]="visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom"
  [visual_probe_hstar_pinpoint]="visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom,hstarbench_mixed_reasoning_zoom,pinpoint_bench_reasoning_zoom,screenspotpro_point_in_box_reasoning_zoom"
  [visual_probe_vstar]="visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom,vstar_bench_reasoning_zoom"
  [visual_probe_vstar_hrbench]="visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom,vstar_bench_reasoning_zoom,hrbench_4k_reasoning_zoom,hrbench_8k_reasoning_zoom"
  [visual_probe_vstar_hstar_hrbench]="visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom,vstar_bench_reasoning_zoom,hstarbench_mixed_reasoning_zoom,hrbench_4k_reasoning_zoom,hrbench_8k_reasoning_zoom"
  [grounding_counting_search]="countbenchqa_reasoning_zoom,countqa_reasoning_zoom,mme_realworld_lite_reasoning_zoom,vstar_bench_reasoning_zoom,aerialvg_bbox_reasoning_zoom,visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom,screenspot_point_in_box_reasoning_zoom,screenspotpro_point_in_box_reasoning_zoom"
  [chart_ocr]="chartqa_reasoning_zoom,chartqa_pro_reasoning_zoom,infovqa_val_reasoning_zoom,chartmuseum_reasoning_zoom,evochart_reasoning_zoom"
  [stem]="mathvision_test_reasoning_zoom,mathvista_testmini_reasoning_zoom,mmmu_pro_standard_reasoning_zoom,mmmu_pro_vision_reasoning_zoom"
  [knowledge_rec]="realworldqa_reasoning_zoom,simplevqa_en_reasoning_zoom,fvqa_reasoning_zoom,mmvetv2_group_img_reasoning_zoom"
  [instruction_following]="mia_bench_reasoning_zoom,mm_mt_bench_reasoning_zoom,mmifeval_reasoning_zoom"
)

if [[ -n "$TRAIN_DIR_EXPLICIT" ]]; then
  TRAIN_DIR="$TRAIN_DIR"
else
  TRAIN_DIR="${TRAIN_DIR_BY_EVAL_NAME[$EVAL_NAME]:-}"
  if [[ -z "$TRAIN_DIR" ]]; then
    echo "ERROR: unknown EVAL_NAME '$EVAL_NAME'." >&2
    echo "Valid EVAL_NAME values: all ${!TRAIN_DIR_BY_EVAL_NAME[*]}" >&2
    exit 1
  fi
fi

SYSTEM_PROMPT="${SYSTEM_PROMPT:-${SYSTEM_PROMPT_BY_EVAL_NAME[$EVAL_NAME]:-}}"
if [[ -z "$TASKS_EXPLICIT" ]]; then
  TASKS="${TASKS_BY_TASK_GROUP[$TASK_GROUP]:-}"
  if [[ -z "$TASKS" ]]; then
    echo "ERROR: unknown TASK_GROUP '$TASK_GROUP'." >&2
    echo "Valid TASK_GROUP values: ${!TASKS_BY_TASK_GROUP[*]}" >&2
    exit 1
  fi
elif [[ -z "$TASK_GROUP_EXPLICIT" ]]; then
  TASK_GROUP="custom"
fi

OUTPUT_ROOT_TOKEN="$EVAL_NAME"
if [[ -n "$TRAIN_DIR_EXPLICIT" && -z "$EVAL_NAME_EXPLICIT" ]]; then
  OUTPUT_ROOT_TOKEN="$(basename "$TRAIN_DIR")"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_qwen35_27b_judge/zoom_eval_${OUTPUT_ROOT_TOKEN}_${TASK_GROUP}_${EVAL_MODE_TAG}}"
CRITIC_RETRY_JOB_TAG=""
if [[ "$CRITIC_RETRY_ENABLED" == true ]]; then
  CRITIC_THRESHOLD_TAG="${CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD//./p}"
  CRITIC_THRESHOLD_TAG="${CRITIC_THRESHOLD_TAG//[^A-Za-z0-9]/_}"
  CRITIC_RETRY_JOB_TAG="critic_retry_thr${CRITIC_THRESHOLD_TAG}"
  OUTPUT_ROOT="${OUTPUT_ROOT}_${CRITIC_RETRY_JOB_TAG}"
fi

case "${EVAL_PROFILE,,}" in
  full)
    PARTITION="${PARTITION:-pli-c}"
    ACCOUNT="${ACCOUNT:-}"
    CONSTRAINT="${CONSTRAINT:-}"
    GPU_TYPE="${GPU_TYPE:-}"
    NUM_GPUS="${NUM_GPUS:-1}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
    DEFAULT_TIME="${DEFAULT_TIME:-6:00:00}"
    ;;
  vlmrl)
    PARTITION="${PARTITION:-pli}"
    ACCOUNT="${ACCOUNT:-vlmrl}"
    CONSTRAINT="${CONSTRAINT:-}"
    GPU_TYPE="${GPU_TYPE:-}"
    NUM_GPUS="${NUM_GPUS:-1}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
    DEFAULT_TIME="${DEFAULT_TIME:-6:00:00}"
    ;;
  ailab)
    PARTITION="${PARTITION:-ailab}"
    ACCOUNT="${ACCOUNT:-}"
    CONSTRAINT="${CONSTRAINT:-}"
    GPU_TYPE="${GPU_TYPE:-}"
    NUM_GPUS="${NUM_GPUS:-1}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
    DEFAULT_TIME="${DEFAULT_TIME:-6:00:00}"
    ;;
  debug)
    PARTITION="gpu-test"
    ACCOUNT="${ACCOUNT:-}"
    CONSTRAINT="${CONSTRAINT:-gpu80}"
    GPU_TYPE="${GPU_TYPE:-}"
    NUM_GPUS="${NUM_GPUS:-1}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
    DEFAULT_TIME="${DEFAULT_TIME:-1:00:00}"
    LIMIT="${LIMIT:-100}"
    # Retain the dedicated multi-benchmark default. Only the legacy visual_probe
    # group narrows debug runs to one task because --limit is applied per task.
    # Respect an explicit TASKS= override in either case.
    [[ -z "$TASKS_EXPLICIT" && "$TASK_GROUP" == "visual_probe" ]] && TASKS="visual_probe_easy_reasoning_zoom"
    # Tag debug outputs with _DEBUG so they don't collide with real runs.
    # Respect an explicit OUTPUT_ROOT= override.
    [[ -z "$OUTPUT_ROOT_EXPLICIT" ]] && OUTPUT_ROOT="${OUTPUT_ROOT}_DEBUG"
    ;;
  *)
    echo "ERROR: EVAL_PROFILE must be 'full', 'vlmrl', 'ailab', or 'debug' (got '$EVAL_PROFILE')" >&2
    exit 1
    ;;
esac
SBATCH_OPTS_RAW="${SBATCH_OPTS:-}"
SBATCH_OPTS=()
if [[ -n "$SBATCH_OPTS_RAW" ]]; then
  read -r -a SBATCH_OPTS <<< "$SBATCH_OPTS_RAW"
fi

extract_checkpoint_step() {
  local checkpoint_name="$1"
  local step_raw=""
  if [[ "$checkpoint_name" =~ checkpoint-([0-9]+)$ ]]; then
    step_raw="${BASH_REMATCH[1]}"
  elif [[ "$checkpoint_name" =~ checkpoint_([0-9]+)$ ]]; then
    step_raw="${BASH_REMATCH[1]}"
  elif [[ "$checkpoint_name" =~ global_step_([0-9]+)$ ]]; then
    step_raw="${BASH_REMATCH[1]}"
  else
    return 1
  fi
  printf '%d' "$((10#$step_raw))"
}

sanitize_job_token() {
  local value="$1"
  value="${value//[^A-Za-z0-9_.-]/_}"
  printf '%s' "${value:0:80}"
}

checkpoint_eval_complete() {
  local out_dir="$1"
  [[ -d "$out_dir" ]] || return 1

  TASKS_ENV="$TASKS" OUT_DIR_ENV="$out_dir" python3 - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR_ENV"])
requested = [t.strip() for t in os.environ.get("TASKS_ENV", "").split(",") if t.strip()]

for path in sorted(out_dir.rglob("*_results.json")):
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    results = payload.get("results")
    if not isinstance(results, dict):
        continue
    if all(task in results for task in requested):
        raise SystemExit(0)

raise SystemExit(1)
PY
}

has_sbatch_option() {
  local kind="$1"
  local opt
  for opt in "${SBATCH_OPTS[@]}"; do
    case "$kind:$opt" in
      mem:--mem=*|mem:--mem-per-cpu=*|mem:--mem-per-gpu=*) return 0 ;;
      time:--time|time:--time=*|time:-t|time:-t=*|time:-t[0-9]*) return 0 ;;
      gres:--gres=*|gres:--gpus=*|gres:--gpus-per-node=*|gres:--gpus-per-task=*) return 0 ;;
      partition:--partition=*|partition:-p|partition:-p*) return 0 ;;
      constraint:--constraint=*|constraint:-C|constraint:-C*) return 0 ;;
      account:--account=*|account:-A|account:-A*) return 0 ;;
      cpus:--cpus-per-task=*|cpus:-c|cpus:-c*) return 0 ;;
      job-name:--job-name=*|job-name:-J|job-name:-J*) return 0 ;;
    esac
  done
  return 1
}

validate_positive_int_or_empty() {
  local name="$1"
  local value="$2"
  [[ -z "$value" ]] && return 0
  if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1 ]]; then
    echo "$name must be a positive integer when set" >&2
    exit 1
  fi
}

validate_port() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1024 || "$value" -gt 65535 ]]; then
    echo "$name must be an integer port in [1024, 65535] (got '$value')" >&2
    exit 1
  fi
}

[[ -d "$TRAIN_DIR" ]] || { echo "ERROR: TRAIN_DIR not found: $TRAIN_DIR" >&2; exit 1; }
validate_positive_int_or_empty MAX_CHECKPOINTS "$MAX_CHECKPOINTS"
validate_positive_int_or_empty MAX_TOOL_ITERATIONS "$MAX_TOOL_ITERATIONS"
validate_positive_int_or_empty MAX_STEPS_EVAL "$MAX_STEPS_EVAL"
validate_positive_int_or_empty ONLY_EVAL_EVERY_X "$ONLY_EVAL_EVERY_X"
validate_port POLICY_PORT_BASE "$POLICY_PORT_BASE"
validate_port ACCELERATE_PORT_BASE "$ACCELERATE_PORT_BASE"
case "${RUN_SYSTEM_PROMPT_MODE,,}" in
  task_metadata|run_env|none) ;;
  *)
    echo "ERROR: RUN_SYSTEM_PROMPT_MODE must be 'task_metadata', 'run_env', or 'none' (got '$RUN_SYSTEM_PROMPT_MODE')" >&2
    exit 1
    ;;
esac
if [[ -n "$SYSTEM_PROMPT" && ! -f "$SYSTEM_PROMPT" ]]; then
  echo "ERROR: configured SYSTEM_PROMPT not found: $SYSTEM_PROMPT" >&2
  exit 1
fi
if [[ ! -d "$VLLM_JUDGE_MODEL_PATH" ]]; then
  echo "ERROR: judge model directory not found: $VLLM_JUDGE_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$VLLM_JUDGE_CHAT_TEMPLATE" ]]; then
  echo "ERROR: judge chat template not found: $VLLM_JUDGE_CHAT_TEMPLATE" >&2
  exit 1
fi
declare -A ONLY_EVAL_GLOBAL_STEP_SET=()
ONLY_EVAL_GLOBAL_STEPS_NORMALIZED=""
if [[ -n "$ONLY_EVAL_GLOBAL_STEPS" ]]; then
  IFS=',' read -r -a ONLY_EVAL_GLOBAL_STEPS_LIST <<< "$ONLY_EVAL_GLOBAL_STEPS"
  for raw_step in "${ONLY_EVAL_GLOBAL_STEPS_LIST[@]}"; do
    step="${raw_step//[[:space:]]/}"
    if [[ -z "$step" || ! "$step" =~ ^[0-9]+$ || "$step" -lt 1 ]]; then
      echo "ONLY_EVAL_GLOBAL_STEPS must be a comma-separated list of positive integers" >&2
      exit 1
    fi
    ONLY_EVAL_GLOBAL_STEP_SET["$((10#$step))"]=1
  done
  mapfile -t ONLY_EVAL_GLOBAL_STEPS_KEYS < <(printf '%s\n' "${!ONLY_EVAL_GLOBAL_STEP_SET[@]}" | sort -n)
  ONLY_EVAL_GLOBAL_STEPS_NORMALIZED=$(IFS=,; echo "${ONLY_EVAL_GLOBAL_STEPS_KEYS[*]}")
fi

_TASKNAMES="$(command grep -rhoE '^(task|group):[[:space:]]*["'\'']?[A-Za-z0-9_]+' \
  "$VERO_EVAL_DIR/lmms_eval/tasks"/*/*.yaml 2>/dev/null \
  | sed -E 's/^(task|group):[[:space:]]*//; s/["'\'']//g' | sort -u)"

missing=()
for task in ${TASKS//,/ }; do
  command grep -qxF "$task" <<< "$_TASKNAMES" || missing+=("$task")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "WARN: missing task yaml(s): ${missing[*]}" >&2
  if [[ "$STRICT_TASKS" == 1 ]]; then
    echo "ERROR: refusing to submit partial run. Set STRICT_TASKS=0 or pass TASKS=... explicitly." >&2
    exit 1
  fi
fi

CHECKPOINTS=("$TRAIN_DIR")
FOUND_CHECKPOINTS=1

declare -i SKIPPED_EXISTING=0
declare -i SKIPPED_QUEUED=0
declare -i SKIPPED_MAX_STEP=0
declare -i SKIPPED_INTERVAL=0
declare -i SKIPPED_EXPLICIT=0

if [[ -n "$MAX_STEPS_EVAL" ]]; then
  FILTERED_CHECKPOINTS=()
  for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name="$(basename "$ckpt")"
    step="$(extract_checkpoint_step "$ckpt_name" || true)"
    if [[ -n "$step" && "$step" -le "$MAX_STEPS_EVAL" ]]; then
      FILTERED_CHECKPOINTS+=("$ckpt")
    else
      SKIPPED_MAX_STEP+=1
    fi
  done
  CHECKPOINTS=("${FILTERED_CHECKPOINTS[@]}")
fi

if [[ "${SKIP_IF_EXISTS,,}" == "true" ]]; then
  FILTERED_CHECKPOINTS=()
  for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name="$(basename "$ckpt")"
    out="$OUTPUT_ROOT/$ckpt_name"
    if checkpoint_eval_complete "$out"; then
      echo "[INFO] Skipping $ckpt_name (complete results detected under $out)."
      SKIPPED_EXISTING+=1
      continue
    fi
    FILTERED_CHECKPOINTS+=("$ckpt")
  done
  CHECKPOINTS=("${FILTERED_CHECKPOINTS[@]}")
fi

declare -A QUEUED_JOB_SET=()
if [[ "${SKIP_IF_QUEUED,,}" == "true" ]]; then
  if command -v squeue >/dev/null 2>&1; then
    QUEUE_USER="${USER:-$(whoami)}"
    while IFS= read -r job_name; do
      [[ -n "$job_name" ]] && QUEUED_JOB_SET["$job_name"]=1
    done < <(squeue -u "$QUEUE_USER" -h -r -o "%j" -t PD,R)
  else
    echo "[WARN] SKIP_IF_QUEUED enabled but squeue not found; skipping queued check."
  fi
fi

if [[ "${SKIP_IF_QUEUED,,}" == "true" && ${#QUEUED_JOB_SET[@]} -gt 0 ]]; then
  FILTERED_CHECKPOINTS=()
  for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name="$(basename "$ckpt")"
    job_name="zoom-eval-$(sanitize_job_token "${EVAL_MODE_TAG}-${EVAL_NAME}-${TASK_GROUP}-${ckpt_name}")"
    queued_match=""
    if [[ -n "${QUEUED_JOB_SET[$job_name]:-}" ]]; then
      queued_match="$job_name"
    fi
    if [[ -n "$queued_match" ]]; then
      echo "[INFO] Skipping $ckpt_name (job $queued_match already queued/running)."
      SKIPPED_QUEUED+=1
      continue
    fi
    FILTERED_CHECKPOINTS+=("$ckpt")
  done
  CHECKPOINTS=("${FILTERED_CHECKPOINTS[@]}")
fi

if [[ -n "$ONLY_EVAL_EVERY_X" ]]; then
  FILTERED_CHECKPOINTS=()
  for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name="$(basename "$ckpt")"
    step="$(extract_checkpoint_step "$ckpt_name" || true)"
    if [[ -n "$step" && $((step % ONLY_EVAL_EVERY_X)) -eq 0 ]]; then
      FILTERED_CHECKPOINTS+=("$ckpt")
    else
      SKIPPED_INTERVAL+=1
    fi
  done
  CHECKPOINTS=("${FILTERED_CHECKPOINTS[@]}")
fi

if [[ ${#ONLY_EVAL_GLOBAL_STEP_SET[@]} -gt 0 ]]; then
  FILTERED_CHECKPOINTS=()
  for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name="$(basename "$ckpt")"
    step="$(extract_checkpoint_step "$ckpt_name" || true)"
    if [[ -n "$step" && -n "${ONLY_EVAL_GLOBAL_STEP_SET[$step]:-}" ]]; then
      FILTERED_CHECKPOINTS+=("$ckpt")
    else
      SKIPPED_EXPLICIT+=1
    fi
  done
  CHECKPOINTS=("${FILTERED_CHECKPOINTS[@]}")
fi

SELECTABLE_CHECKPOINTS=${#CHECKPOINTS[@]}
if [[ -n "$MAX_CHECKPOINTS" && $SELECTABLE_CHECKPOINTS -gt "$MAX_CHECKPOINTS" ]]; then
  if [[ "${USE_MOST_RECENT_MAX_CHECKPOINTS,,}" == "true" ]]; then
    CHECKPOINTS=("${CHECKPOINTS[@]:SELECTABLE_CHECKPOINTS-MAX_CHECKPOINTS:MAX_CHECKPOINTS}")
  else
    CHECKPOINTS=("${CHECKPOINTS[@]:0:MAX_CHECKPOINTS}")
  fi
fi

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "[INFO] No checkpoints to submit after applying filters."
  exit 0
fi

# Avoid every invocation's first selected checkpoint using the same policy port.
# This matters on shared nodes: two independent submitter runs can both
# start with PORT=10001 and collide if Slurm places them together.
if [[ -z "$PORT_OFFSET" ]]; then
  printf -v SUBMIT_TIME_TOKEN '%(%H%M%S)T' -1
  PORT_OFFSET=$(( (10#$SUBMIT_TIME_TOKEN + $$) % 20000 ))
else
  validate_positive_int_or_empty PORT_OFFSET "$PORT_OFFSET"
fi

MEM_REQUEST="${MEM_OVERRIDE:-$DEFAULT_MEM}"
TIME_REQUEST="${TIME_OVERRIDE:-$DEFAULT_TIME}"

BASE_SBATCH_OPTS=("${SBATCH_OPTS[@]}")
has_sbatch_option partition || [[ -z "$PARTITION" ]] || BASE_SBATCH_OPTS+=("--partition=$PARTITION")
has_sbatch_option account || [[ -z "$ACCOUNT" ]] || BASE_SBATCH_OPTS+=("--account=$ACCOUNT")
has_sbatch_option constraint || [[ -z "$CONSTRAINT" ]] || BASE_SBATCH_OPTS+=("--constraint=$CONSTRAINT")
has_sbatch_option cpus || BASE_SBATCH_OPTS+=("--cpus-per-task=$CPUS_PER_TASK")
has_sbatch_option mem || BASE_SBATCH_OPTS+=("--mem=$MEM_REQUEST")
has_sbatch_option time || BASE_SBATCH_OPTS+=("--time=$TIME_REQUEST")
if ! has_sbatch_option gres; then
  if [[ -n "$GPU_TYPE" ]]; then
    BASE_SBATCH_OPTS+=("--gres=gpu:${GPU_TYPE}:${NUM_GPUS}")
  else
    BASE_SBATCH_OPTS+=("--gres=gpu:${NUM_GPUS}")
  fi
fi

CHECKPOINT_INFO="${#CHECKPOINTS[@]} selected from $FOUND_CHECKPOINTS found"
(( SKIPPED_EXISTING > 0 )) && CHECKPOINT_INFO+="; skipped $SKIPPED_EXISTING existing"
(( SKIPPED_QUEUED > 0 )) && CHECKPOINT_INFO+="; skipped $SKIPPED_QUEUED queued/running"
[[ -n "$MAX_STEPS_EVAL" ]] && CHECKPOINT_INFO+="; max_steps_eval=$MAX_STEPS_EVAL skipped $SKIPPED_MAX_STEP"
[[ -n "$ONLY_EVAL_EVERY_X" ]] && CHECKPOINT_INFO+="; only_eval_every_x=$ONLY_EVAL_EVERY_X skipped $SKIPPED_INTERVAL"
[[ -n "$ONLY_EVAL_GLOBAL_STEPS_NORMALIZED" ]] && CHECKPOINT_INFO+="; only_eval_global_steps=$ONLY_EVAL_GLOBAL_STEPS_NORMALIZED skipped $SKIPPED_EXPLICIT"
[[ -n "$MAX_CHECKPOINTS" ]] && CHECKPOINT_INFO+="; max_checkpoints=$MAX_CHECKPOINTS most_recent=${USE_MOST_RECENT_MAX_CHECKPOINTS}"

cat <<INFO
Submitting Qwen3-VL-8B-Instruct zoom eval
  Eval name:       $EVAL_NAME
  Eval mode:       ${EVAL_MODE,,}
  Model path:      $TRAIN_DIR
  Task group:      $TASK_GROUP
  Profile:         ${EVAL_PROFILE,,}
  Tasks:           $TASKS
  System prompt:   ${SYSTEM_PROMPT:-<none>} (mode=${RUN_SYSTEM_PROMPT_MODE,,})
  Output root:     $OUTPUT_ROOT
  Models:          $CHECKPOINT_INFO
  Image pixels:    max=$MAX_IMAGE_PIXELS
  Max tool images: $MAX_TOOL_IMAGES
  Max tool turns:  $MAX_TOOL_ITERATIONS
  Budget forcing:  tool_calls_until_final=$FORCE_TOOL_CALLS_UNTIL_FINAL_TURN
  Judge model:     $VLLM_JUDGE_MODEL_PATH
  Judge template:  $VLLM_JUDGE_CHAT_TEMPLATE
  FlashInfer sampler: $VLLM_USE_FLASHINFER_SAMPLER
  Forced final:    after_max_tool_iterations=$FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS prefix=${FORCE_FINAL_ANSWER_PREFIX:-<empty>} think_prefill=${FORCE_FINAL_ANSWER_THINK_PREFILL:-<empty>}
  Critic retry:    enabled=$CRITIC_RETRY_ENABLED threshold=$CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD prefill_profile=${CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE:-<none>} tag=${CRITIC_RETRY_JOB_TAG:-<none>}
  Workers:         $WORKERS
  Port bases:      policy=$POLICY_PORT_BASE accelerate=$ACCELERATE_PORT_BASE offset=$PORT_OFFSET
  Sbatch options:  ${BASE_SBATCH_OPTS[*]}
INFO

submitted=0
for i in "${!CHECKPOINTS[@]}"; do
  CKPT_DIR="${CHECKPOINTS[$i]}"
  CKPT_NAME="$(basename "$CKPT_DIR")"
  HF_PATH="$CKPT_DIR/actor/huggingface"
  [[ -d "$HF_PATH" ]] || HF_PATH="$CKPT_DIR"

  PORT=$((POLICY_PORT_BASE + PORT_OFFSET + i))
  MASTER_PORT=$((ACCELERATE_PORT_BASE + PORT_OFFSET + i))
  validate_port PORT "$PORT"
  validate_port MASTER_PORT "$MASTER_PORT"
  OUT="$OUTPUT_ROOT/$CKPT_NAME"
  JOB_NAME="zoom-eval-$(sanitize_job_token "${EVAL_MODE_TAG}-${EVAL_NAME}-${TASK_GROUP}-${CKPT_NAME}")"

  env_list=(
    MODEL=qwen3vl
    SUFFIX=reasoning_zoom
    TOOLS=on
    MM_MAX_PIXELS="$MAX_IMAGE_PIXELS"
    MM_MIN_PIXELS=
    MAX_TOOL_IMAGES="$MAX_TOOL_IMAGES"
    MAX_TOOL_ITERATIONS="$MAX_TOOL_ITERATIONS"
    CHECKPOINT_PATH="$HF_PATH"
    TASKS="$TASKS"
    PORT="$PORT"
    JUDGE_BACKEND=engine
    API_TYPE=vllm
    JUDGE_MODEL_PATH="$VLLM_JUDGE_MODEL_PATH"
    VLLM_JUDGE_MODEL_PATH="$VLLM_JUDGE_MODEL_PATH"
    VLLM_JUDGE_CHAT_TEMPLATE="$VLLM_JUDGE_CHAT_TEMPLATE"
    VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER"
    CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE=1
    OUTPUT_ROOT="$OUT"
    NUM_GPUS=1
    WORKERS="$WORKERS"
    MASTER_PORT="$MASTER_PORT"
    OVERRIDE_EXISTING="$OVERRIDE_EXISTING"
    MERGE_BACKEND="$MERGE_BACKEND"
    TRAINING_SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
    RUN_SYSTEM_PROMPT_MODE="${RUN_SYSTEM_PROMPT_MODE,,}"
    FORCE_TOOL_CALLS_UNTIL_FINAL_TURN="$FORCE_TOOL_CALLS_UNTIL_FINAL_TURN"
    FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS="$FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS"
    FORCE_FINAL_ANSWER_PREFIX="$FORCE_FINAL_ANSWER_PREFIX"
    FORCE_FINAL_ANSWER_THINK_PREFILL="$FORCE_FINAL_ANSWER_THINK_PREFILL"
    CRITIC_GUIDED_ANSWER_RETRY="$CRITIC_RETRY_ENABLED"
    CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD="$CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD"
    CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE="${CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE:-persist}"
  )
  [[ -n "$LIMIT" ]] && env_list+=(LIMIT="$LIMIT")
  if [[ "${RUN_SYSTEM_PROMPT_MODE,,}" == "run_env" && -n "$SYSTEM_PROMPT" ]]; then
    env_list+=(SYSTEM_PROMPT="$SYSTEM_PROMPT")
  fi

  sbatch_opts=("${BASE_SBATCH_OPTS[@]}")
  if ! has_sbatch_option job-name; then
    sbatch_opts+=("--job-name=$JOB_NAME")
  fi
  cmd=(sbatch "${sbatch_opts[@]}" --export=ALL run_eval.sbatch)

  echo "[$CKPT_NAME] checkpoint=$HF_PATH"
  echo "  tasks=$TASKS"
  echo "  port=$PORT judge=engine-sequential master_port=$MASTER_PORT -> $OUT"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '  '; printf '%q ' env "${env_list[@]}" "${cmd[@]}"; printf '\n'
  else
    env "${env_list[@]}" "${cmd[@]}"
  fi
  submitted=$((submitted + 1))
done

echo "Submitted ${submitted} job(s). Watch with: squeue -u \$USER"
