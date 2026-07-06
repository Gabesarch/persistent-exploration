#!/bin/bash
# Launch a DEDICATED vLLM judge server (sleep-mode), then point the judge at it.
# SOURCE this (don't exec) so the exported VLLM_JUDGE_BASE_URL reaches the caller:
#   source scripts/serve_judge_vllm.sh
# eval_domain.sh sources it automatically when JUDGE_BACKEND=server and
# START_VLLM_JUDGE_SERVER=1.
#
# GPU: on a single H100 the policy already holds 0.9 util, so give the judge its own
# GPU — request gpu:2 and set VLLM_JUDGE_CUDA_VISIBLE_DEVICES=1. (Sleep-mode lets the
# judge free VRAM between batches, but the non-sleeping policy still owns its GPU.)
# Ported from reason-vlm-eval/examples/eval_model.sh; uses a captured PID for cleanup
# instead of `pkill -f 'vllm serve'` (which would also kill the policy server).
#
# Env: VLLM_JUDGE_MODEL_PATH (default Qwen3-32B), VLLM_JUDGE_MODEL_NAME (=qwen_vllm),
#      VLLM_JUDGE_PORT (=8003), VLLM_JUDGE_API_KEY (=qwen), VLLM_JUDGE_TENSOR_PARALLEL_SIZE,
#      VLLM_JUDGE_CUDA_VISIBLE_DEVICES (empty=share GPU), VLLM_JUDGE_CHAT_TEMPLATE,
#      VLLM_JUDGE_GPU_MEMORY_UTILIZATION, VLLM_JUDGE_MAX_MODEL_LEN.

# Disable FlashInfer sampling to avoid JIT compilation issues with missing curand.h
# on cluster nodes. Uses PyTorch-native sampler instead (minimal performance impact).
export VLLM_USE_FLASHINFER_SAMPLER=0

# The variable VLLM_USE_FLASH_ATTN_V3 is not recognized by vLLM.
# Use the official attention config flag below to force FlashAttention v2.
unset VLLM_USE_FLASH_ATTN_V3 || true

: "${VERO_EVAL_DIR:?source set_paths.sh first (VERO_EVAL_DIR unset)}"
_JUDGE_MODEL_PATH="${VLLM_JUDGE_MODEL_PATH:-$HF_HOME/Qwen/Qwen3-32B}"
_JUDGE_MODEL_NAME="${VLLM_JUDGE_MODEL_NAME:-qwen_vllm}"
_JUDGE_PORT="${VLLM_JUDGE_PORT:-8003}"
_JUDGE_API_KEY="${VLLM_JUDGE_API_KEY:-qwen}"
_JUDGE_TP="${VLLM_JUDGE_TENSOR_PARALLEL_SIZE:-1}"
_JUDGE_TEMPLATE="${VLLM_JUDGE_CHAT_TEMPLATE:-$VERO_EVAL_DIR/prompts/qwen3_nonthinking.jinja}"
_JUDGE_MEM="${VLLM_JUDGE_GPU_MEMORY_UTILIZATION:-0.9}"
_JUDGE_MAXLEN="${VLLM_JUDGE_MAX_MODEL_LEN:-18432}"
_JUDGE_LOG="$VERO_EVAL_DIR/logs/vllm/judge_$(date '+%Y-%m-%d_%H-%M-%S')_port${_JUDGE_PORT}.txt"
mkdir -p "$(dirname "$_JUDGE_LOG")"

# Reuse an already-running judge server on this port (idempotent across a domain loop).
if curl --max-time 3 -s -o /dev/null "http://localhost:${_JUDGE_PORT}/v1/models"; then
  echo "=== Reusing existing judge server on :$_JUDGE_PORT ==="
  export VLLM_SERVER_JUDGE=1
  export VLLM_JUDGE_BASE_URL="http://localhost:${_JUDGE_PORT}/v1"
  export VLLM_JUDGE_PORT="$_JUDGE_PORT" VLLM_JUDGE_MODEL_NAME="$_JUDGE_MODEL_NAME"
  return 0 2>/dev/null || exit 0
fi

echo "=== Launching dedicated vLLM judge: $_JUDGE_MODEL_PATH on :$_JUDGE_PORT (CUDA_VISIBLE_DEVICES=${VLLM_JUDGE_CUDA_VISIBLE_DEVICES:-inherit}) ==="
_cvd=()
[[ -n "${VLLM_JUDGE_CUDA_VISIBLE_DEVICES:-}" ]] && _cvd=(CUDA_VISIBLE_DEVICES="$VLLM_JUDGE_CUDA_VISIBLE_DEVICES")
_ATTN_ARGS=()
if python - <<'PY' 2>/dev/null
from importlib.metadata import version
from packaging.version import Version
raise SystemExit(0 if Version(version("vllm")) >= Version("0.22") else 1)
PY
then _ATTN_ARGS=(--attention-config.flash_attn_version 2)
else echo "vLLM < 0.22 -> judge omitting --attention-config.flash_attn_version (unsupported)"; fi
nohup env "${_cvd[@]}" VLLM_SERVER_DEV_MODE=1 vllm serve "$_JUDGE_MODEL_PATH" \
  --port "$_JUDGE_PORT" \
  --served-model-name "$_JUDGE_MODEL_NAME" \
  --gpu-memory-utilization "$_JUDGE_MEM" \
  "${_ATTN_ARGS[@]}" \
  --tensor-parallel-size "$_JUDGE_TP" \
  --max-model-len "$_JUDGE_MAXLEN" \
  --enable-sleep-mode \
  --chat-template "$_JUDGE_TEMPLATE" \
  --api-key "$_JUDGE_API_KEY" \
  > "$_JUDGE_LOG" 2>&1 &
VLLM_JUDGE_SERVER_PID=$!
echo "$VLLM_JUDGE_SERVER_PID" > "$VERO_EVAL_DIR/logs/judge_server.pid"
disown "$VLLM_JUDGE_SERVER_PID" 2>/dev/null || true

echo "Waiting for judge server (log: $_JUDGE_LOG)..."
_ok=0
for _ in $(seq 1 90); do
  if curl --max-time 2 -s -o /dev/null "http://localhost:${_JUDGE_PORT}/v1/models"; then _ok=1; break; fi
  sleep 10
done
if [[ "$_ok" != 1 ]]; then
  echo "ERROR: judge server not ready after ~15min; see $_JUDGE_LOG" >&2
  return 1 2>/dev/null || exit 1
fi
echo "Judge server up at :$_JUDGE_PORT (pid $VLLM_JUDGE_SERVER_PID)"
# Free VRAM until the judge is actually called; VLLMServerJudge wakes/sleeps per batch.
curl --max-time 10 -s -X POST "http://localhost:${_JUDGE_PORT}/sleep?level=1" -o /dev/null || true

export VLLM_SERVER_JUDGE=1
export VLLM_JUDGE_BASE_URL="http://localhost:${_JUDGE_PORT}/v1"
export VLLM_JUDGE_PORT="$_JUDGE_PORT"
export VLLM_JUDGE_MODEL_NAME="$_JUDGE_MODEL_NAME"
