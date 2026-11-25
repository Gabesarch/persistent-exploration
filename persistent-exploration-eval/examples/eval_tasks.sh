#!/bin/bash
# Internal task-list runner used by evaluate_checkpoint.sh. The policy server is
# already running when called with --no-serve.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; VERO_EVAL_DIR="$(cd "$SELF/.." && pwd)"
set +u
[[ -f "$VERO_EVAL_DIR/set_paths.sh" ]] && source "$VERO_EVAL_DIR/set_paths.sh" || source "$VERO_EVAL_DIR/set_paths.sh.example"
set -u

TASKS=""; OUTPUT_PATH=""; LIMIT="${LIMIT:-}"; OFFSET="${OFFSET:-0}"; NO_SERVE=0
PORT="${PORT:-9001}"; API_KEY="${API_KEY:-qwen}"; KEEP_SERVER="${KEEP_SERVER:-1}"
while [[ $# -gt 0 ]]; do case "$1" in
  --tasks) TASKS="$2"; shift 2;; --output-path) OUTPUT_PATH="$2"; shift 2;;
  --limit) LIMIT="$2"; shift 2;; --offset) OFFSET="$2"; shift 2;; --no-serve) NO_SERVE=1; shift;;
  *) echo "Unknown arg: $1" >&2; shift;;
esac; done
[[ -z "$TASKS" ]] && { echo "Usage: bash examples/eval_tasks.sh --tasks t1,t2,... [--limit N] [--output-path DIR] [--no-serve]"; exit 1; }
OUTPUT_PATH="${OUTPUT_PATH:-./results/tasks_$(date -u +%Y%m%d_%H%M%S)}"

cd "$VERO_EVAL_DIR"
STARTED_SERVER=0
if [[ "$NO_SERVE" != 1 ]]; then
  if curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models"; then
    echo "=== Reusing existing vLLM server on :$PORT ==="
  else
    echo "=== Starting vLLM server ==="; bash scripts/serve_qwen_vllm.sh; STARTED_SERVER=1
  fi
  curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
    || { echo "ERROR: vLLM not responding on :$PORT"; exit 1; }
fi

OUTPUT_PATH="$OUTPUT_PATH" bash examples/eval_domain.sh --tasks "$TASKS" --offset "$OFFSET" ${LIMIT:+--limit "$LIMIT"}

if [[ "$STARTED_SERVER" == 1 && "$KEEP_SERVER" != 1 ]]; then
  echo "=== Stopping vLLM server we started ==="; pkill -f "vllm serve" || true
fi
echo "=== Done. Results under: $OUTPUT_PATH ==="
