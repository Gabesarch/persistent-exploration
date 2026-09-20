#!/bin/bash
#SBATCH --job-name=brstrata20_prgspo_critic
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=968G
#SBATCH --time=24:00:00
#SBATCH --array=1-10%1
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log

set -e
set -o pipefail
set -x

REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/gs1693/repo/vero2}"
cd "$REPO_ROOT"
source "$REPO_ROOT/set_paths.sh"

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

CONFIG_NAME="branchstrata20_fresh_prefixremoved_gspo_lr1e5_critic"

python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/vero2_scripts/config" \
  --config-name="$CONFIG_NAME"

