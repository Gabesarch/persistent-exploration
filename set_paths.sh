#!/bin/bash
# Environment-specific paths for gaze RL runs in this verl clone.

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate /scratch/gpfs/ZHUANGL/gs1693/conda/gaze-rl-verl
module load cudatoolkit/12.8
module load gcc/11

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export VERL_ROOT="/home/gs1693/repo/gaze-vlm/verl"

export ROOT_PATH="${ROOT_PATH:-/scratch/gpfs/ZHUANGL/gs1693}"
export SAVE_PATH="${SAVE_PATH:-$ROOT_PATH/verl_gaze_rl}"
export IMAGE_ROOT="${IMAGE_ROOT:-}"

export TRAIN_DATA="${TRAIN_DATA:-$ROOT_PATH/gaze_rl_data/vero_data/vero_chart_ocr_cgs_train_balanced.jsonl}"
export VAL_DATA="${VAL_DATA:-$ROOT_PATH/gaze_rl_data/vero_data/vero_chart_ocr_cgs_val_balanced.jsonl}"

export HF_HOME="${HF_HOME:-$ROOT_PATH/hf_cache}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-$ROOT_PATH/vllm_cache}"
export WANDB_DIR="${WANDB_DIR:-$SAVE_PATH/wandb}"
export TRITON_HOME="$ROOT_PATH/.cache/triton"


export TQDM_MININTERVAL="${TQDM_MININTERVAL:-10}"

if [ -z "${MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP:-}" ] && \
   [ -n "${SLURM_JOB_ID:-}" ] && command -v scontrol >/dev/null 2>&1; then
  SLURM_JOB_INFO=$(scontrol show job "$SLURM_JOB_ID" 2>/dev/null || true)
  SLURM_END_TIME=$(printf '%s\n' "$SLURM_JOB_INFO" | sed -n 's/.*EndTime=\([^ ]*\).*/\1/p')
  if [ -n "$SLURM_END_TIME" ] && [ "$SLURM_END_TIME" != "Unknown" ]; then
    SLURM_END_TS=$(date -d "$SLURM_END_TIME" +%s 2>/dev/null || true)
    if [ -n "$SLURM_END_TS" ]; then
      export MLP_CURRENT_CAPACITY_BLOCK_EXPIRATION_TIMESTAMP="$SLURM_END_TS"
    fi
  fi
  unset SLURM_JOB_INFO SLURM_END_TIME SLURM_END_TS
fi

for path in "$SAVE_PATH" "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE" "$VLLM_CACHE_ROOT" "$WANDB_DIR"; do
  if ! mkdir -p "$path" 2>/dev/null; then
    echo "warning: could not create $path; override ROOT_PATH, SAVE_PATH, or cache env vars if this run needs to write there" >&2
  fi
done
