#!/bin/bash
#SBATCH --job-name=crop_hard
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=968G
#SBATCH --time=24:00:00
#SBATCH --array=1-3%1
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log

##SBATCH --job-name=crop_hard
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

ALGORITHM_TAG="${ALGORITHM_TAG:-GDPO_HDPOtool_GSPO_fp16_math_verify}"
TRAINER_EXPERIMENT_SUFFIX="${TRAINER_EXPERIMENT_SUFFIX:-chart_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly}"
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

mkdir -p slurm/output slurm/error logs
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

CONFIG_NAME="gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_cpueff_hardonly"

echo "CONFIG_NAME=$CONFIG_NAME"
echo "TRAINER_PROJECT_NAME=$TRAINER_PROJECT_NAME"
echo "TRAINER_EXPERIMENT_NAME=$TRAINER_EXPERIMENT_NAME"
echo "CHECKPOINT_EXPERIMENT_NAME=$CHECKPOINT_EXPERIMENT_NAME"
echo "TRAINER_DEFAULT_LOCAL_DIR=$TRAINER_DEFAULT_LOCAL_DIR"
echo "LEGACY_CHECKPOINT_DIR_LINKED=$LEGACY_CHECKPOINT_DIR_LINKED"
printf 'HYDRA_ARGS:'; printf ' %q' "${HYDRA_ARGS[@]}"; printf '\n'

python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/vero2_scripts/config" \
  --config-name="$CONFIG_NAME" \
  trainer.project_name="$TRAINER_PROJECT_NAME" \
  trainer.experiment_name="$TRAINER_EXPERIMENT_NAME" \
  trainer.default_local_dir="$TRAINER_DEFAULT_LOCAL_DIR" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$NUM_GPUS" \
  "${HYDRA_ARGS[@]}"
