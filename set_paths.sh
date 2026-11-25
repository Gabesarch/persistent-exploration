#!/usr/bin/env bash
# Shared training environment. Export paths in your shell or set_paths.local.sh.
VERO2_PATHS_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
if [[ -f "$VERO2_PATHS_DIR/set_paths.local.sh" ]]; then
  source "$VERO2_PATHS_DIR/set_paths.local.sh"
fi

if [[ -n "${VERO2_CONDA_ENV:-}" ]]; then
  if ! command -v conda >/dev/null 2>&1; then
    source "${CONDA_BASE:-$HOME/miniconda3}/etc/profile.d/conda.sh"
  fi
  conda activate "$VERO2_CONDA_ENV"
fi

export VERL_ROOT="${VERL_ROOT:-$VERO2_PATHS_DIR}"
export ROOT_PATH="${ROOT_PATH:-$VERL_ROOT/runtime}"
export SAVE_PATH="${SAVE_PATH:-$ROOT_PATH/outputs}"
export HF_HOME="${HF_HOME:-$ROOT_PATH/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$ROOT_PATH/vllm_cache}"
export WANDB_DIR="${WANDB_DIR:-$SAVE_PATH/wandb}"
export TRITON_HOME="${TRITON_HOME:-$ROOT_PATH/.cache/triton}"
export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"
export IMAGE_ROOT="${IMAGE_ROOT:-}"
export MODEL_PATH="${MODEL_PATH:-$HF_HOME/Qwen/Qwen3-VL-8B-Instruct}"

mkdir -p "$SAVE_PATH" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$VLLM_CACHE_ROOT" "$WANDB_DIR"
