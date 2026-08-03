#!/bin/bash
# VAL-ONLY difficulty-measurement shard runner for the rollout-based hard-example filter:
# 8 sampled base-model drawing rollouts per counting sample, dump one jsonl row per rollout.
#
# Parameterized by env (fail-loud, no defaults):
#   SHARD_FILE  absolute path to the shard jsonl (from make_shards.py)
#   SHARD_TAG   short tag, e.g. shard00 / smoke64 -> deterministic experiment name + dump path
#
# Submit via submit_all_shards.sh, or directly from the REPO ROOT:
#   sbatch --export=ALL,SHARD_FILE=/abs/shard_00.verl.jsonl,SHARD_TAG=shard00 \
#     vero2_scripts/drawing/experiments/rollout_hard_filter/run_valonly_basefilter_counting_drawonly_full.sh
#
# Dump: ${SAVE_PATH}/checkpoints/vero2_verl/basefilter_counting_drawonly_${SHARD_TAG}/validation_data/0.jsonl
# (written once, AFTER the whole val loop — a killed job leaves nothing; rerun the shard).
#SBATCH --job-name=basefilter_count
#SBATCH --partition=pli-lc
#SBATCH --account=pli_xl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=960G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log

set -x
set -o pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 2

SET_PATHS_SH="./set_paths.sh"
if [ ! -f "$SET_PATHS_SH" ]; then
  echo "error: set_paths.sh not found at $SET_PATHS_SH (run from the repo root)" >&2
  exit 2
fi
source "$SET_PATHS_SH"

if [ -z "${SHARD_FILE:-}" ] || [ -z "${SHARD_TAG:-}" ]; then
  echo "error: SHARD_FILE and SHARD_TAG must be set (see header)" >&2
  exit 2
fi
if [ ! -f "$SHARD_FILE" ]; then
  echo "error: SHARD_FILE does not exist: $SHARD_FILE" >&2
  exit 2
fi

module load proxy/default 2>/dev/null || true

NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-vero2_verl}"
TRAINER_EXPERIMENT_NAME="basefilter_counting_drawonly_${SHARD_TAG}"
TRAINER_DEFAULT_LOCAL_DIR="${SAVE_PATH}/checkpoints/${TRAINER_PROJECT_NAME}/${TRAINER_EXPERIMENT_NAME}"

DUMP_FILE="${TRAINER_DEFAULT_LOCAL_DIR}/validation_data/0.jsonl"
if [ -e "$DUMP_FILE" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "error: dump already exists at $DUMP_FILE — set FORCE=1 to redo this shard" >&2
  exit 3
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

CONFIG_NAME="${CONFIG_NAME:-valonly_basefilter_counting_qwen3vl_drawonly}"

echo "CONFIG_NAME=$CONFIG_NAME"
echo "SHARD_FILE=$SHARD_FILE"
echo "SHARD_TAG=$SHARD_TAG"
echo "TRAINER_EXPERIMENT_NAME=$TRAINER_EXPERIMENT_NAME"
echo "TRAINER_DEFAULT_LOCAL_DIR=$TRAINER_DEFAULT_LOCAL_DIR"

python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/vero2_scripts/config" \
  --config-name="$CONFIG_NAME" \
  trainer.project_name="$TRAINER_PROJECT_NAME" \
  trainer.experiment_name="$TRAINER_EXPERIMENT_NAME" \
  trainer.default_local_dir="$TRAINER_DEFAULT_LOCAL_DIR" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$NUM_GPUS" \
  actor_rollout_ref.rollout.agent.num_workers="$NUM_GPUS" \
  data.val_files="$SHARD_FILE" \
  data.train_files="$SHARD_FILE" \
  "$@"
