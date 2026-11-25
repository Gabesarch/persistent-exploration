#!/bin/bash
# Run the published Qwen training recipe directly on a multi-GPU machine.

set -e
set -o pipefail
set -x

REPO_ROOT="${VERO2_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
source "$REPO_ROOT/set_paths.sh"
for required in TRAIN_FILES VAL_FILES MODEL_PATH; do
  [[ -n "${!required:-}" ]] || { echo "Set $required before training" >&2; exit 2; }
done
[[ -f "$TRAIN_FILES" && -f "$VAL_FILES" && -d "$MODEL_PATH" ]] || {
  echo "Missing TRAIN_FILES, VAL_FILES, or MODEL_PATH input" >&2
  exit 2
}

export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export RAY_ADDRESS="${RAY_ADDRESS:-local}"

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [ -z "${VERO2_CROP_CACHE_ROOT:-}" ]; then
  CROP_CACHE_RUN_ID="${TRAIN_RUN_ID:-pid_$$}"
  export VERO2_CROP_CACHE_ROOT="$SAVE_PATH/crop_cache/massbounds/${CROP_CACHE_RUN_ID}"
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

CONFIG_NAME="branchstrata20_fresh_lr1e5_critic_massbounds"

python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/persistent-exploration-rl/config" \
  --config-name="$CONFIG_NAME" "$@"
