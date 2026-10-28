#!/usr/bin/env bash
# Sourced only by the new Gemma launchers.
export VERO2_CONDA_ENV="${VERO2_CONDA_ENV:-/scratch/gpfs/ZHUANGL/gs1693/conda/vero2_gemma4_train}"
export GEMMA4_TRAIN_OVERLAY="${GEMMA4_TRAIN_OVERLAY:-/scratch/gpfs/ZHUANGL/gs1693/vero2/runtime/gemma4_train_transformers}"
export SAVE_PATH="${GEMMA4_SAVE_PATH:-/scratch/gpfs/ZHUANGL/gs1693/vero2/gemma4_training}"
source "$REPO_ROOT/set_paths.sh"
export PYTHONPATH="$GEMMA4_TRAIN_OVERLAY:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 WANDB_MODE=offline
export VERO2_CROP_CACHE_ROOT="$SAVE_PATH/crop_cache/$CONFIG_NAME/${SLURM_JOB_ID:-local_$$}"
export VLLM_CACHE_ROOT="$SAVE_PATH/vllm_cache/$CONFIG_NAME"
export NO_PROXY="${NO_PROXY:+$NO_PROXY,}localhost,127.0.0.1,::1"
export no_proxy="$NO_PROXY"
export NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0
export GEMMA4_ARTIFACT_DIR="$SAVE_PATH/manifests/$CONFIG_NAME/${SLURM_JOB_ID:-local_$$}"
mkdir -p "$GEMMA4_ARTIFACT_DIR" "$VERO2_CROP_CACHE_ROOT" "$VLLM_CACHE_ROOT"
python "$REPO_ROOT/vero2_scripts/validate_gemma4_training.py" --preflight --output "$GEMMA4_ARTIFACT_DIR"
python -m verl.trainer.main_ppo --config-path="$REPO_ROOT/vero2_scripts/config" --config-name="$CONFIG_NAME" --cfg job --resolve > "$GEMMA4_ARTIFACT_DIR/resolved.yaml"
