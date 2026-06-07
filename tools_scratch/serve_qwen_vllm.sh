#!/bin/bash
set -euo pipefail

# Hardcoded settings for your local VLLM server

# Checkpoint 1: qwen3vl-instruct mix all
# Earlier checkpoint
#CHECKPOINT_PATH="/scratch/gpfs/ZHUANGL/lc1556/checkpoints/verl/GSPO_fp16_math_verify_qwen3vl_8b_instruct_train_mix_all_v2_csg_kr_spatial_chart_stem_filtered_20260110_091531/global_step_680/actor/huggingface"
# Later checkpoint
#CHECKPOINT_PATH="/scratch/gpfs/ZHUANGL/lc1556/checkpoints/verl/GSPO_fp16_math_verify_qwen3vl_8b_instruct_train_mix_all_v2_csg_kr_spatial_chart_stem_filtered_20260110_091531/global_step_880/actor/huggingface"

# # checkpoint 2: qwen3vl-thinking mix all
#CHECKPOINT_PATH="/scratch/gpfs/ZHUANGL/lc1556/checkpoints/verl/GSPO_fp16_math_verify_qwen3vl_8b_thinking_train_mix_all_v2_csg_kr_spatial_chart_stem_filtered_20260110_091531/global_step_640/actor/huggingface"

# # Checkpoint 2: qwen2.5vl mix all
# # Earlier checkpoint
# CHECKPOINT_PATH="/scratch/gpfs/ZHUANGL/lc1556/checkpoints/verl/GSPO_fp16_math_verify_qwen2_5_vl_7b_train_mix_all_v2_csg_kr_spatial_chart_stem_filtered_20260110_091531/global_step_680/actor/huggingface"
# # Later checkpoint
CHECKPOINT_PATH="/scratch/gpfs/ZHUANGL/hw0171/Qwen3-VL-8B-Instruct"

NUM_GPUS=1
PORT=9001
MODEL_NAME="qwen_vllm"
API_KEY="qwen"

# Force offline everywhere
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1

export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HOME/.cache/huggingface/transformers}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"
mkdir -p "$TRANSFORMERS_CACHE" "$HF_DATASETS_CACHE"

export WANDB_MODE=offline
ROOT_PATH="${ROOT_PATH:-$(pwd)}"
WANDB_DIR="${WANDB_DIR:-$ROOT_PATH}"
export WANDB_DIR
[ -n "$WANDB_DIR" ] && mkdir -p "$WANDB_DIR"

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "NUM_GPUS: $NUM_GPUS"
echo "PORT: $PORT"

# Kill any existing vLLM servers to free GPU memory
if pgrep -f "vllm serve" > /dev/null 2>&1; then
  echo "Killing existing vLLM server(s)..."
  pkill -f "vllm serve" || true
  sleep 3
fi

echo "Starting vLLM server..."
mkdir -p ./examples/chat/vllm_logs

LOGFILE="./examples/chat/vllm_logs/vllm_logfile_$(date '+%Y-%m-%d_%H-%M-%S').txt"

nohup vllm serve "$CHECKPOINT_PATH" \
  --port "$PORT" \
  --served-model-name "$MODEL_NAME" \
  --gpu-memory-utilization 0.9 \
  --tensor-parallel-size "$NUM_GPUS" \
  --uvicorn-log-level info \
  --max-model-len 32768 \
  --no-enable-prefix-caching \
  --mm-processor-kwargs '{"max_pixels":12960000,"min_pixels":4096}' \
  --api-key "$API_KEY" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager \
  > "$LOGFILE" 2>&1 &
disown

echo "Waiting for vLLM server to be ready (this may take a few minutes)..."
for i in $(seq 1 60); do
  if curl --max-time 2 -s -o /dev/null "http://localhost:${PORT}/v1/models"; then
    echo "vLLM is up at http://localhost:${PORT}"
    exit 0
  fi
  sleep 5
done

echo "vLLM did not respond after 5 minutes. Check logs: $LOGFILE"