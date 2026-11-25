#!/usr/bin/env bash
# Run the Gemma 4 E4B persistent exploration recipe on a multi-GPU machine.
set -euo pipefail

REPO_ROOT="${VERO2_REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "$REPO_ROOT"
source "$REPO_ROOT/set_paths.sh"

export GEMMA4_MODEL_PATH="${GEMMA4_MODEL_PATH:-$HF_HOME/google/gemma-4-E4B-it}"
for required in TRAIN_FILES VAL_FILES; do
  [[ -n "${!required:-}" ]] || { echo "Set $required before training" >&2; exit 2; }
done
[[ -f "$TRAIN_FILES" && -f "$VAL_FILES" && -d "$GEMMA4_MODEL_PATH" ]] || {
  echo "Missing TRAIN_FILES, VAL_FILES, or GEMMA4_MODEL_PATH input" >&2
  exit 2
}

if [[ -n "${GEMMA4_TRAIN_OVERLAY:-}" ]]; then
  [[ -d "$GEMMA4_TRAIN_OVERLAY/transformers" ]] || {
    echo "GEMMA4_TRAIN_OVERLAY must contain a Transformers package" >&2
    exit 2
  }
  export PYTHONPATH="$GEMMA4_TRAIN_OVERLAY:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
else
  export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
fi

export PYTHONDONTWRITEBYTECODE=1
export RAY_ADDRESS="${RAY_ADDRESS:-local}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export VERO2_CROP_CACHE_ROOT="${VERO2_CROP_CACHE_ROOT:-$SAVE_PATH/crop_cache/gemma4_e4b/${TRAIN_RUN_ID:-pid_$$}}"
export VLLM_CACHE_ROOT="$VLLM_CACHE_ROOT/gemma4_e4b"
mkdir -p "$VERO2_CROP_CACHE_ROOT" "$VLLM_CACHE_ROOT"

LOCAL_NO_PROXY="localhost,127.0.0.1,::1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

python -m verl.trainer.main_ppo \
  --config-path="$REPO_ROOT/persistent-exploration-rl/config" \
  --config-name=gemma4_e4b "$@"
