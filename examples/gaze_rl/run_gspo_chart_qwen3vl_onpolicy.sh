#!/bin/bash
#SBATCH --job-name=verl_gaze_qwen3vl_onpolicy
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --array=1-20%1
#SBATCH --cpus-per-task=64
#SBATCH --mem=968G
#SBATCH --time=24:00:00
#SBATCH --output=examples/gaze_rl/slurm/output/%x_%j.log
#SBATCH --error=examples/gaze_rl/slurm/error/%x_%j.log

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

DATE_TIME="20260420_155518"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"

ALGORITHM_TAG="${ALGORITHM_TAG:-GSPO_fp16_math_verify}"
TRAINER_EXPERIMENT_SUFFIX="${TRAINER_EXPERIMENT_SUFFIX:-chart_qwen3vl_onpolicy}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-${ALGORITHM_TAG}_${TRAINER_EXPERIMENT_SUFFIX}_${DATE_TIME}}"

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

python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/examples/gaze_rl/config" \
  --config-name=gspo_chart_qwen3vl_onpolicy \
  trainer.experiment_name="$TRAINER_EXPERIMENT_NAME" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$NUM_GPUS" \
  "${HYDRA_ARGS[@]}"
