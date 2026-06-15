#!/bin/bash
# Serve the policy/tool model via vLLM (OpenAI-compatible) with the Qwen3-VL
# tool-calling flags. Config comes from set_paths.sh: CHECKPOINT_PATH, PORT,
# MODEL_NAME, API_KEY, NUM_GPUS.
set -euo pipefail

# Disable FlashInfer sampling to avoid JIT compilation issues with missing curand.h
# on cluster nodes. Uses PyTorch-native sampler instead (minimal performance impact).
export VLLM_USE_FLASHINFER_SAMPLER=0

# The variable VLLM_USE_FLASH_ATTN_V3 is not recognized by vLLM.
# Use the official attention config flag below to force FlashAttention v2.
unset VLLM_USE_FLASH_ATTN_V3 || true

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERO_EVAL_DIR="$(cd "$SELF/.." && pwd)"
set +u
if [[ -f "$VERO_EVAL_DIR/set_paths.sh" ]]; then
  source "$VERO_EVAL_DIR/set_paths.sh"
else
  echo "WARN: $VERO_EVAL_DIR/set_paths.sh not found; using set_paths.sh.example (copy & edit it)." >&2
  source "$VERO_EVAL_DIR/set_paths.sh.example"
fi
set -u

: "${CHECKPOINT_PATH:?set CHECKPOINT_PATH in set_paths.sh}"
PORT="${PORT:-9001}"; MODEL_NAME="${MODEL_NAME:-qwen_vllm}"
API_KEY="${API_KEY:-qwen}"; NUM_GPUS="${NUM_GPUS:-1}"
# Set CUSTOM_CHAT_TEMPLATE to use a custom template (e.g., one that doesn't auto-inject tools)
CUSTOM_CHAT_TEMPLATE="${CUSTOM_CHAT_TEMPLATE:-}"

LOG_DIR="${LOG_DIR:-$VERO_EVAL_DIR/logs/vllm}"
mkdir -p "$LOG_DIR"
LOGFILE="$LOG_DIR/vllm_$(date '+%Y-%m-%d_%H-%M-%S')_port${PORT}.txt"

echo "CHECKPOINT_PATH: $CHECKPOINT_PATH"
echo "PORT: $PORT   NUM_GPUS: $NUM_GPUS   served-model-name: $MODEL_NAME"

# Only kill vLLM servers on this specific port (avoids cross-job interference)
if pgrep -f "vllm serve.*--port $PORT" >/dev/null 2>&1; then
  echo "Killing existing vLLM server on port $PORT..."; pkill -f "vllm serve.*--port $PORT" || true; sleep 3
fi

echo "Starting vLLM server (log: $LOGFILE)..."
CHAT_TEMPLATE_ARG=""
[[ -n "$CUSTOM_CHAT_TEMPLATE" ]] && CHAT_TEMPLATE_ARG="--chat-template $CUSTOM_CHAT_TEMPLATE"
nohup vllm serve "$CHECKPOINT_PATH" \
  --port "$PORT" \
  --served-model-name "$MODEL_NAME" \
  --gpu-memory-utilization 0.9 \
  --attention-config.flash_attn_version 2 \
  --tensor-parallel-size "$NUM_GPUS" \
  --uvicorn-log-level info \
  --max-model-len 32768 \
  --no-enable-prefix-caching \
  --mm-processor-kwargs '{"max_pixels":12960000,"min_pixels":4096}' \
  --api-key "$API_KEY" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  --enforce-eager \
  $CHAT_TEMPLATE_ARG \
  > "$LOGFILE" 2>&1 &
disown

echo "Waiting for vLLM to be ready (up to 5 min)..."
for _ in $(seq 1 60); do
  if curl --max-time 2 -s -o /dev/null "http://localhost:${PORT}/v1/models"; then
    echo "vLLM is up at http://localhost:${PORT}"; exit 0
  fi
  sleep 5
done
echo "vLLM did not respond after 5 minutes. Check logs: $LOGFILE" >&2
exit 1
