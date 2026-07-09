#!/bin/bash
#SBATCH --job-name=crop_debug_are
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log

##SBATCH --job-name=crop_debug
##SBATCH --partition=gpu-test
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --gres=gpu:2
##SBATCH --cpus-per-task=24
##SBATCH --mem=800G
##SBATCH --constraint=gpu80
##SBATCH --time=24:00:00
##SBATCH --output=slurm/output/%x_%j.log
##SBATCH --error=slurm/error/%x_%j.log

set -x
set -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SET_PATHS_SH="./set_paths.sh"

if [ ! -f "$SET_PATHS_SH" ]; then
  echo "error: set_paths.sh not found at $SET_PATHS_SH" >&2
  exit 2
fi

source "$SET_PATHS_SH"

ENGINE="vllm"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
  ENGINE="$1"
  shift
fi
HYDRA_ARGS=("$@")

CHECKPOINT_DATE_TIME="${CHECKPOINT_DATE_TIME:-20260524_160610}"
RUN_UNIQUE_ID="${RUN_UNIQUE_ID:-${SLURM_JOB_ID:-pid_$$}}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
DEBUG_TRAIN_DATA="/scratch/gpfs/ZHUANGL/gs1693/vero2_datasets/train_rl_datasets/chart_search_counting_visual_probe_train_debug256_rollout_score_leq05.verl.jsonl"
DEBUG_VAL_DATA="/scratch/gpfs/ZHUANGL/gs1693/vero2_datasets/train_rl_datasets/chart_search_counting_visual_probe_train_debug256.verl.jsonl"
TOOL_EFFICIENCY_REWARD_MODE="${TOOL_EFFICIENCY_REWARD_MODE:-concave_length_penalty}"
TOOL_EFFICIENCY_FREE_CALLS="${TOOL_EFFICIENCY_FREE_CALLS:-1}"
TOOL_EFFICIENCY_CONCAVE_K="${TOOL_EFFICIENCY_CONCAVE_K:-10.0}"
TOOL_EFFICIENCY_CONCAVE_Q="${TOOL_EFFICIENCY_CONCAVE_Q:-0.95}"

ALGORITHM_TAG="${ALGORITHM_TAG:-GDPO_HDPOtool_GSPO_fp16_math_verify}"
TRAINER_EXPERIMENT_SUFFIX="${TRAINER_EXPERIMENT_SUFFIX:-chart_qwen3vl_visual_probe_cropzoom_debug256_gdpo_hdpotool_hardonly_answer_retry_forcedprefill}"
CHECKPOINT_EXPERIMENT_NAME="${CHECKPOINT_EXPERIMENT_NAME:-${ALGORITHM_TAG}_${TRAINER_EXPERIMENT_SUFFIX}_${CHECKPOINT_DATE_TIME}}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-${CHECKPOINT_EXPERIMENT_NAME}_${RUN_UNIQUE_ID}}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-vero2_verl}"
TRAINER_DEFAULT_LOCAL_DIR="${TRAINER_DEFAULT_LOCAL_DIR:-${SAVE_PATH}/checkpoints/${TRAINER_PROJECT_NAME}/${CHECKPOINT_EXPERIMENT_NAME}}"
LEGACY_CHECKPOINT_DIR_LINKED=""
if [ ! -e "$TRAINER_DEFAULT_LOCAL_DIR" ]; then
  LEGACY_CHECKPOINT_CANDIDATES=()
  for candidate in "${TRAINER_DEFAULT_LOCAL_DIR}"_*; do
    if [ -d "$candidate" ]; then
      LEGACY_CHECKPOINT_CANDIDATES+=("$candidate")
    fi
  done
  if [ "${#LEGACY_CHECKPOINT_CANDIDATES[@]}" -eq 1 ]; then
    ln -s "${LEGACY_CHECKPOINT_CANDIDATES[0]}" "$TRAINER_DEFAULT_LOCAL_DIR" 2>/dev/null || true
    if [ -L "$TRAINER_DEFAULT_LOCAL_DIR" ]; then
      LEGACY_CHECKPOINT_DIR_LINKED="${LEGACY_CHECKPOINT_CANDIDATES[0]}"
    fi
  fi
fi

DEBUG_LOGGING_ENABLED=0
case "${VERO2_COMPREHENSIVE_DEBUG_LOGGING:-1}" in
  1|true|TRUE|True|yes|YES|Yes) DEBUG_LOGGING_ENABLED=1 ;;
esac
RUN_ANSWER_RETRY_SMOKE_CHECK="${RUN_ANSWER_RETRY_SMOKE_CHECK:-1}"
DEFAULT_ANSWER_RETRY_ASSISTANT_PREFILL="Let's see, I do not have enough information to answer and need to continue exploring with additional tool calls."
ANSWER_RETRY_ASSISTANT_PREFILL="${ANSWER_RETRY_ASSISTANT_PREFILL:-$DEFAULT_ANSWER_RETRY_ASSISTANT_PREFILL}"
export ANSWER_RETRY_ASSISTANT_PREFILL

DEBUG_LOG_DIR="${DEBUG_LOG_DIR:-${TRAINER_DEFAULT_LOCAL_DIR}/debug_logs/${RUN_UNIQUE_ID}}"
if [ "$DEBUG_LOGGING_ENABLED" = "1" ]; then
  mkdir -p "$DEBUG_LOG_DIR"
  LOG_FILE="${LOG_FILE:-$DEBUG_LOG_DIR/log.txt}"
else
  LOG_FILE="${LOG_FILE:-$SCRIPT_DIR/log.txt}"
fi
exec > >(tee -a "$LOG_FILE") 2>&1

mkdir -p examples/gaze_rl/slurm/output examples/gaze_rl/slurm/error logs
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
if [ -z "${VERO2_CROP_CACHE_ROOT:-}" ]; then
  CROP_CACHE_RUN_ID="${SLURM_JOB_ID:-pid_$$}"
  export VERO2_CROP_CACHE_ROOT="/tmp/${USER:-unknown}/vero2_crop_cache/${CROP_CACHE_RUN_ID}"
fi

LOCAL_NO_PROXY="localhost,127.0.0.1,::1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"

export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

if [ "${SKIP_SQUEUE:-0}" != "1" ] && command -v squeue >/dev/null 2>&1; then
  timeout 10s squeue -u "${SLURM_JOB_USER:-$USER}" || true
fi

if [ "$ENGINE" != "vllm" ]; then
  HYDRA_ARGS=("actor_rollout_ref.rollout.name=$ENGINE" "${HYDRA_ARGS[@]}")
fi

HYDRA_ARGS=(
  "data.train_files=$DEBUG_TRAIN_DATA"
  "data.val_files=$DEBUG_VAL_DATA"
  "reward.custom_reward_function.reward_kwargs.tool_efficiency_reward_mode=$TOOL_EFFICIENCY_REWARD_MODE"
  "reward.custom_reward_function.reward_kwargs.tool_efficiency_free_calls=$TOOL_EFFICIENCY_FREE_CALLS"
  "reward.custom_reward_function.reward_kwargs.tool_efficiency_concave_k=$TOOL_EFFICIENCY_CONCAVE_K"
  "reward.custom_reward_function.reward_kwargs.tool_efficiency_concave_q=$TOOL_EFFICIENCY_CONCAVE_Q"
  "${HYDRA_ARGS[@]}"
)

if [ "$DEBUG_LOGGING_ENABLED" = "1" ]; then
  HAS_DEBUG_OVERRIDE=0
  for arg in "${HYDRA_ARGS[@]}"; do
    if [[ "$arg" == trainer.comprehensive_debug_logging.enabled=* ]]; then
      HAS_DEBUG_OVERRIDE=1
      break
    fi
  done
  if [ "$HAS_DEBUG_OVERRIDE" = "0" ]; then
    HYDRA_ARGS=("trainer.comprehensive_debug_logging.enabled=true" "${HYDRA_ARGS[@]}")
  fi
  HAS_GPU_DEBUG_OVERRIDE=0
  for arg in "${HYDRA_ARGS[@]}"; do
    if [[ "$arg" == trainer.comprehensive_debug_logging.gpu_memory_logging.enabled=* ]]; then
      HAS_GPU_DEBUG_OVERRIDE=1
      break
    fi
  done
  if [ "$HAS_GPU_DEBUG_OVERRIDE" = "0" ]; then
    HYDRA_ARGS=("trainer.comprehensive_debug_logging.gpu_memory_logging.enabled=true" "${HYDRA_ARGS[@]}")
  fi
  HYDRA_ARGS=("trainer.comprehensive_debug_logging.dir=$DEBUG_LOG_DIR" "${HYDRA_ARGS[@]}")
fi

has_hydra_override() {
  local key="$1"
  local arg
  for arg in "${HYDRA_ARGS[@]}"; do
    if [[ "$arg" == "$key="* ]]; then
      return 0
    fi
  done
  return 1
}

# if [ "$NUM_GPUS" -le 2 ]; then
#   if ! has_hydra_override "actor_rollout_ref.rollout.gpu_memory_utilization"; then
#     HYDRA_ARGS+=("actor_rollout_ref.rollout.gpu_memory_utilization=${CROP_DEBUG_GPU_MEMORY_UTILIZATION:-0.5}")
#   fi
#   if ! has_hydra_override "actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes"; then
#     HYDRA_ARGS+=("actor_rollout_ref.rollout.checkpoint_engine.update_weights_bucket_megabytes=${CROP_DEBUG_UPDATE_WEIGHTS_BUCKET_MB:-2048}")
#   fi
# fi

if ! has_hydra_override "actor_rollout_ref.rollout.multi_turn.conditional_answer_retry_assistant_prefill"; then
  HYDRA_ARGS=(
    'actor_rollout_ref.rollout.multi_turn.conditional_answer_retry_assistant_prefill=${oc.env:ANSWER_RETRY_ASSISTANT_PREFILL}'
    "${HYDRA_ARGS[@]}"
  )
fi

echo "LOG_FILE=$LOG_FILE"
echo "TRAINER_PROJECT_NAME=$TRAINER_PROJECT_NAME"
echo "TRAINER_EXPERIMENT_NAME=$TRAINER_EXPERIMENT_NAME"
echo "CHECKPOINT_EXPERIMENT_NAME=$CHECKPOINT_EXPERIMENT_NAME"
echo "TRAINER_DEFAULT_LOCAL_DIR=$TRAINER_DEFAULT_LOCAL_DIR"
echo "LEGACY_CHECKPOINT_DIR_LINKED=$LEGACY_CHECKPOINT_DIR_LINKED"
echo "VERO2_COMPREHENSIVE_DEBUG_LOGGING=$DEBUG_LOGGING_ENABLED"
echo "DEBUG_LOG_DIR=$DEBUG_LOG_DIR"
echo "RUN_ANSWER_RETRY_SMOKE_CHECK=$RUN_ANSWER_RETRY_SMOKE_CHECK"
echo "TOOL_EFFICIENCY_REWARD_MODE=$TOOL_EFFICIENCY_REWARD_MODE"
echo "TOOL_EFFICIENCY_FREE_CALLS=$TOOL_EFFICIENCY_FREE_CALLS"
echo "TOOL_EFFICIENCY_CONCAVE_K=$TOOL_EFFICIENCY_CONCAVE_K"
echo "TOOL_EFFICIENCY_CONCAVE_Q=$TOOL_EFFICIENCY_CONCAVE_Q"
echo "ANSWER_RETRY_ASSISTANT_PREFILL=$ANSWER_RETRY_ASSISTANT_PREFILL"
printf 'HYDRA_ARGS:'; printf ' %q' "${HYDRA_ARGS[@]}"; printf '\n'

run_answer_retry_smoke_check() {
  local sample_traces_path="$DEBUG_LOG_DIR/sample_traces.jsonl"
  if [ "$DEBUG_LOGGING_ENABLED" != "1" ]; then
    echo "error: answer-retry smoke check requires VERO2_COMPREHENSIVE_DEBUG_LOGGING=1" >&2
    return 2
  fi
  if [ ! -s "$sample_traces_path" ]; then
    echo "error: debug sample traces not found at $sample_traces_path" >&2
    return 2
  fi

  SAMPLE_TRACES_PATH="$sample_traces_path" python3 - <<'PY'
import json
import os
import sys
from collections import Counter

path = os.environ["SAMPLE_TRACES_PATH"]
checked = 0
failures = []
observed = Counter()

with open(path, "r", encoding="utf-8") as f:
    for line_no, line in enumerate(f, 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            failures.append(f"line {line_no}: invalid json: {exc}")
            continue
        if record.get("event") != "sample_final":
            continue
        trajectory = record.get("trajectory") or {}
        if trajectory.get("validate"):
            continue

        checked += 1
        request_id = record.get("request_id") or f"line_{line_no}"
        events = record.get("debug_events") or []
        assistant_events = [event for event in events if event.get("event") == "assistant_turn"]
        retry_events = [event for event in events if event.get("event") == "incorrect_answer_retry"]
        skipped_events = [event for event in events if event.get("event") == "incorrect_answer_retry_skipped"]
        forced_errors = [event for event in events if event.get("event") == "forced_structure_error"]
        messages = record.get("messages") or []

        if forced_errors:
            failures.append(
                f"{request_id}: forced_structure_error="
                + ",".join(str(event.get("reason")) for event in forced_errors[:3])
            )
        observed["rollouts"] += 1
        observed["retries"] += len(retry_events)
        observed["skipped_retries"] += len(skipped_events)

        for retry_event in retry_events:
            retry_pos = events.index(retry_event)
            following_assistant = next(
                (
                    event
                    for event in events[retry_pos + 1 :]
                    if event.get("event") == "assistant_turn"
                ),
                None,
            )
            retry_count = retry_event.get("retry_count")
            if following_assistant is None:
                failures.append(f"{request_id}: retry_{retry_count}_missing_following_assistant_turn")
                continue
            text = following_assistant.get("text") or ""
            if "<tool_call>" not in text:
                failures.append(f"{request_id}: retry_{retry_count}_following_turn_missing_tool_call")
            if "<answer>" in text:
                failures.append(f"{request_id}: retry_{retry_count}_following_turn_contains_answer")

if checked == 0:
    print(f"error: no training sample_final records found in {path}", file=sys.stderr)
    sys.exit(2)

if failures:
    print(
        f"answer-retry smoke check failed: {len(failures)} failures across {checked} training rollouts",
        file=sys.stderr,
    )
    for failure in failures[:20]:
        print(f"  - {failure}", file=sys.stderr)
    if len(failures) > 20:
        print(f"  ... {len(failures) - 20} more failures", file=sys.stderr)
    sys.exit(1)

print(
    "answer-retry smoke check passed: "
    f"rollouts={observed['rollouts']} retries={observed['retries']} "
    f"skipped_retries={observed['skipped_retries']}"
)
PY
}

TRAIN_EXIT=0
python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/vero2_scripts/config" \
  --config-name=gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_debug256_cpueff_answer_retry_forcedprefill \
  trainer.project_name="$TRAINER_PROJECT_NAME" \
  trainer.experiment_name="$TRAINER_EXPERIMENT_NAME" \
  trainer.default_local_dir="$TRAINER_DEFAULT_LOCAL_DIR" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$NUM_GPUS" \
  "${HYDRA_ARGS[@]}" || TRAIN_EXIT=$?

if [ "$TRAIN_EXIT" -ne 0 ]; then
  exit "$TRAIN_EXIT"
fi

case "$RUN_ANSWER_RETRY_SMOKE_CHECK" in
  1|true|TRUE|True|yes|YES|Yes) run_answer_retry_smoke_check ;;
esac
