#!/bin/bash
#SBATCH --job-name=crop_debug
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=24
#SBATCH --mem=800G
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
##SBATCH --time=1:00:00
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

DATE_TIME="20260420_155518"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
DEBUG_DATA="/scratch/gpfs/ZHUANGL/gs1693/vero2_datasets/train_rl_datasets/chart_search_counting_visual_probe_train_debug256.verl.jsonl"

ALGORITHM_TAG="${ALGORITHM_TAG:-GSPO_fp16_math_verify}"
TRAINER_EXPERIMENT_SUFFIX="${TRAINER_EXPERIMENT_SUFFIX:-chart_qwen3vl_visual_probe_cropzoom_debug256}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-${ALGORITHM_TAG}_${TRAINER_EXPERIMENT_SUFFIX}_${DATE_TIME}}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-verl}"

DEBUG_LOGGING_ENABLED=0
case "${VERO2_COMPREHENSIVE_DEBUG_LOGGING:-0}" in
  1|true|TRUE|True|yes|YES|Yes) DEBUG_LOGGING_ENABLED=1 ;;
esac

DEBUG_LOG_DIR="${SAVE_PATH}/checkpoints/${TRAINER_PROJECT_NAME}/${TRAINER_EXPERIMENT_NAME}/debug_logs"
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
  "data.train_files=$DEBUG_DATA"
  "data.val_files=$DEBUG_DATA"
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
  HYDRA_ARGS=("trainer.comprehensive_debug_logging.dir=$DEBUG_LOG_DIR" "${HYDRA_ARGS[@]}")
fi

echo "LOG_FILE=$LOG_FILE"
echo "TRAINER_PROJECT_NAME=$TRAINER_PROJECT_NAME"
echo "TRAINER_EXPERIMENT_NAME=$TRAINER_EXPERIMENT_NAME"
echo "VERO2_COMPREHENSIVE_DEBUG_LOGGING=$DEBUG_LOGGING_ENABLED"
echo "DEBUG_LOG_DIR=$DEBUG_LOG_DIR"
printf 'HYDRA_ARGS:'; printf ' %q' "${HYDRA_ARGS[@]}"; printf '\n'

python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/vero2_scripts/config" \
  --config-name=gspo_chart_qwen3vl_visual_probe_cropzoom_debug256 \
  trainer.project_name="$TRAINER_PROJECT_NAME" \
  trainer.experiment_name="$TRAINER_EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$NUM_GPUS" \
  "${HYDRA_ARGS[@]}"
