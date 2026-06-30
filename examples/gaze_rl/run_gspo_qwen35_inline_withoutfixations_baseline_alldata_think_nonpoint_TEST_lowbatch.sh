#!/bin/bash
#SBATCH --job-name=verl_gaze_gspo_qwen35_test
#SBATCH --partition=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=00:30:00
#SBATCH --constraint=gpu80
#SBATCH --output=examples/gaze_rl/slurm/output/%x_%A_%a.log
#SBATCH --error=examples/gaze_rl/slurm/error/%x_%A_%a.log

set -x
set -o pipefail

# SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# DEFAULT_TRAIN_JSONL="/scratch/gpfs/ZHUANGL/gs1693/vigorl_gaze_datasets/chart_search_counting_rl/chart_search_counting_train.verl.jsonl"
# DEFAULT_VAL_JSONL="/scratch/gpfs/ZHUANGL/gs1693/vigorl_gaze_datasets/chart_search_counting_rl/chart_search_counting_val.verl.jsonl"

DEFAULT_TRAIN_JSONL="/scratch/gpfs/ZHUANGL/gs1693/vigorl_gaze_datasets/chart_search_counting_rl/preprocessed_max_pixels_10240000/chart_search_counting_train.verl.preprocessed.jsonl"
DEFAULT_VAL_JSONL="/scratch/gpfs/ZHUANGL/gs1693/vigorl_gaze_datasets/chart_search_counting_rl/preprocessed_max_pixels_10240000/chart_search_counting_val.verl.preprocessed.jsonl"
export TRAIN_DATA="${TRAIN_DATA:-$DEFAULT_TRAIN_JSONL}"
export VAL_DATA="${VAL_DATA:-$DEFAULT_VAL_JSONL}"
source "./set_paths.sh"
export MODEL_PATH="${MODEL_PATH:-$HF_HOME/Qwen/Qwen3.5-9B-nothinkchattemplate}"

TRAIN_JSONL="$DEFAULT_TRAIN_JSONL"
VAL_JSONL="$DEFAULT_VAL_JSONL"

if [ ! -f "$TRAIN_DATA" ] && [ -f "$TRAIN_JSONL" ]; then
  echo "warning: TRAIN_DATA does not exist: $TRAIN_DATA; using $TRAIN_JSONL" >&2
  export TRAIN_DATA="$TRAIN_JSONL"
fi

if [ ! -f "$VAL_DATA" ] && [ -f "$VAL_JSONL" ]; then
  echo "warning: VAL_DATA does not exist: $VAL_DATA; using $VAL_JSONL" >&2
  export VAL_DATA="$VAL_JSONL"
fi

if [ ! -f "$TRAIN_DATA" ]; then
  echo "error: TRAIN_DATA does not exist: $TRAIN_DATA" >&2
  exit 2
fi

if [ ! -f "$VAL_DATA" ]; then
  echo "error: VAL_DATA does not exist: $VAL_DATA" >&2
  exit 2
fi

ENGINE="vllm"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]]; then
  ENGINE="$1"
  shift
fi
HYDRA_ARGS=("$@")

DATE_TIME="${DATE_TIME:-$(date +%Y%m%d_%H%M%S)}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-4}}"

DEBUG_TRAIN_BATCH_SIZE="${DEBUG_TRAIN_BATCH_SIZE:-32}"
DEBUG_PPO_MINI_BATCH_SIZE="${DEBUG_PPO_MINI_BATCH_SIZE:-16}"
DEBUG_MAX_RESPONSE_LENGTH="${DEBUG_MAX_RESPONSE_LENGTH:-2048}"

ALGORITHM_TAG="${ALGORITHM_TAG:-GSPO_fp16_math_verify}"
TRAINER_EXPERIMENT_SUFFIX="${TRAINER_EXPERIMENT_SUFFIX:-qwen3vl_4b_inline_withoutfixations_baseline_alldata_think_nonpoint}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-${ALGORITHM_TAG}_${TRAINER_EXPERIMENT_SUFFIX}_${DATE_TIME}}"

cd "$VERL_ROOT"
mkdir -p examples/gaze_rl/slurm/output examples/gaze_rl/slurm/error logs
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export CUDA_LAUNCH_BLOCKING="${CUDA_LAUNCH_BLOCKING:-1}"
export RAY_DEDUP_LOGS="${RAY_DEDUP_LOGS:-0}"

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

python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/examples/gaze_rl/config" \
  --config-name=gspo_qwen35_inline_withoutfixations_baseline_alldata_think_nonpoint \
  trainer.experiment_name="$TRAINER_EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$NUM_GPUS" \
  data.train_files="$TRAIN_DATA" \
  data.val_files="$VAL_DATA" \
  data.train_batch_size="$DEBUG_TRAIN_BATCH_SIZE" \
  actor_rollout_ref.actor.ppo_mini_batch_size="$DEBUG_PPO_MINI_BATCH_SIZE" \
  data.max_response_length="$DEBUG_MAX_RESPONSE_LENGTH" \
  actor_rollout_ref.rollout.response_length="$DEBUG_MAX_RESPONSE_LENGTH" \
  reward.reward_kwargs.max_resp_len="$DEBUG_MAX_RESPONSE_LENGTH" \
  "${HYDRA_ARGS[@]}"
