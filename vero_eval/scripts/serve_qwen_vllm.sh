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

# Only kill a server on OUR port: on shared nodes a blanket `pkill -f "vllm serve"`
# would murder a sibling job's policy server mid-eval (give concurrent jobs unique
# PORTs; the trailing space anchors the match so 9001 doesn't match 90011).
if pgrep -f "vllm serve.*--port ${PORT} " >/dev/null 2>&1; then
  echo "Killing existing vLLM server(s) on port ${PORT}..."; pkill -f "vllm serve.*--port ${PORT} " || true; sleep 3
fi

# Optional explicit chat template. Needed for the RL-trained Vero checkpoints:
# vLLM otherwise serves the base template embedded in tokenizer_config.json, which
# does NOT force the <think> block or the training tool-call format, so Vero answers
# directly without thinking/tool use. Point CHAT_TEMPLATE at the training template
# (e.g. the checkpoint's chat_template.jinja) to reproduce training behavior.
CHAT_TEMPLATE="${CHAT_TEMPLATE:-}"
CHAT_TEMPLATE_ARGS=()
[[ -n "$CHAT_TEMPLATE" ]] && CHAT_TEMPLATE_ARGS=(--chat-template "$CHAT_TEMPLATE") && echo "Using chat template: $CHAT_TEMPLATE"
# max_pixels is now a serve-time knob (was hardcoded 12960000). Default is 10000000 — the
# canonical full-setting resolution (on the ~8-13MP accuracy plateau, so equivalent to the
# old 12.96MP within noise and slightly cheaper). Override per run via MM_MAX_PIXELS.
MM_MAX_PIXELS="${MM_MAX_PIXELS:-10000000}"
MM_MIN_PIXELS="${MM_MIN_PIXELS:-4096}"

echo "Starting vLLM server (log: $LOGFILE)..."
CHAT_TEMPLATE_ARG=""
[[ -n "$CUSTOM_CHAT_TEMPLATE" ]] && CHAT_TEMPLATE_ARG="--chat-template $CUSTOM_CHAT_TEMPLATE"
# `--attention-config.flash_attn_version 2` is only understood by vLLM >= 0.22; older
# stacks (e.g. the vero2 training env's vLLM 0.18) reject the unknown arg and refuse to
# start. Gate it on the running vLLM version so serving works across env stacks; the
# default (>= 0.22) behavior is unchanged.
ATTN_ARGS=()
if python - <<'PY' 2>/dev/null
from importlib.metadata import version
from packaging.version import Version
raise SystemExit(0 if Version(version("vllm")) >= Version("0.22") else 1)
PY
then ATTN_ARGS=(--attention-config.flash_attn_version 2)
else echo "vLLM < 0.22 detected -> omitting --attention-config.flash_attn_version (unsupported)"; fi
nohup vllm serve "$CHECKPOINT_PATH" \
  --port "$PORT" \
  --served-model-name "$MODEL_NAME" \
  --gpu-memory-utilization 0.9 \
  "${ATTN_ARGS[@]}" \
  --tensor-parallel-size "$NUM_GPUS" \
  --uvicorn-log-level info \
  --max-model-len "${MAX_MODEL_LEN:-32768}" \
  --no-enable-prefix-caching \
  --mm-processor-kwargs "{\"max_pixels\":${MM_MAX_PIXELS},\"min_pixels\":${MM_MIN_PIXELS}}" \
  --api-key "$API_KEY" \
  --enable-auto-tool-choice \
  --tool-call-parser hermes \
  "${CHAT_TEMPLATE_ARGS[@]}" \
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
