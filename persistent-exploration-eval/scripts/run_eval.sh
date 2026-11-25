#!/usr/bin/env bash
# Internal Qwen policy-server and benchmark runner. Use examples/evaluate_checkpoint.sh.
set -euo pipefail

VERO_EVAL_DIR="${VERO_EVAL_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)}"
[[ -f "$VERO_EVAL_DIR/set_paths.sh.example" && -d "$VERO_EVAL_DIR/lmms_eval" ]] || {
  echo "ERROR: invalid evaluation directory: $VERO_EVAL_DIR" >&2
  exit 1
}
export VERO_EVAL_DIR="$(cd -- "$VERO_EVAL_DIR" && pwd)"
set +u
[[ -f "$VERO_EVAL_DIR/set_paths.sh" ]] && source "$VERO_EVAL_DIR/set_paths.sh" || source "$VERO_EVAL_DIR/set_paths.sh.example"
set -u
export PYTHONPATH="$VERO_EVAL_DIR/..${PYTHONPATH:+:$PYTHONPATH}"

: "${TASKS:?set benchmark tasks through evaluate_checkpoint.sh}"
: "${CHECKPOINT_PATH:?set checkpoint through evaluate_checkpoint.sh}"
[[ -d "$CHECKPOINT_PATH" ]] || { echo "ERROR: missing checkpoint: $CHECKPOINT_PATH" >&2; exit 1; }
export CHECKPOINT_PATH
export PORT="${PORT:-9001}" API_KEY="${API_KEY:-qwen}"
export MODEL_NAME="${MODEL_NAME:-qwen_vllm}" MODEL_VERSION="${MODEL_VERSION:-qwen_vllm}"
export NUM_GPUS="${NUM_GPUS:-1}"
export LMMS_EVAL_HF_TOKENIZER="${LMMS_EVAL_HF_TOKENIZER:-$CHECKPOINT_PATH}"
export LMMS_EVAL_HF_TOKENIZER_LOCAL_ONLY="${LMMS_EVAL_HF_TOKENIZER_LOCAL_ONLY:-1}"
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
export MAX_TOOL_IMAGES="${MAX_TOOL_IMAGES:-10}"
export MM_MAX_PIXELS="${MM_MAX_PIXELS:-10000000}"
export MM_MIN_PIXELS="${MM_MIN_PIXELS-}"
export MCP_SERVER_PATH="${MCP_SERVER_PATH:-lmms_eval/mcp/vero2_mcp.py}"

# The trained checkpoint's template supplies the forced-think and tool-call format.
if [[ -z "${CHAT_TEMPLATE:-}" && -f "$CHECKPOINT_PATH/chat_template.jinja" ]]; then
  export CHAT_TEMPLATE="$CHECKPOINT_PATH/chat_template.jinja"
fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$VERO_EVAL_DIR/results/qwen_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$VERO_EVAL_DIR/logs"
cd "$VERO_EVAL_DIR"
echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "TASKS=$TASKS OUTPUT_ROOT=$OUTPUT_ROOT"
echo "CRITIC_GUIDED_ANSWER_RETRY=${CRITIC_GUIDED_ANSWER_RETRY:-false} threshold=${CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD:-0.5}"

bash scripts/serve_qwen_vllm.sh
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
  || { echo "ERROR: vLLM policy not up on :$PORT" >&2; exit 1; }

bash examples/eval_tasks.sh --no-serve --tasks "$TASKS" \
  --output-path "$OUTPUT_ROOT" --offset "${OFFSET:-0}" ${LIMIT:+--limit "$LIMIT"}
echo "=== Done. Results under: $OUTPUT_ROOT ==="
