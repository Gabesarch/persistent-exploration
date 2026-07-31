#!/bin/bash
# TYPE 3 — evaluate ONE checkpoint across MULTIPLE domains. Serves CHECKPOINT_PATH
# once, then loops the domain->tasks map (examples/_domain_map.sh), running
# examples/eval_domain.sh per domain against the single long-lived server.
#
#   cd vero_eval && bash examples/eval_across_domains.sh --domains all --model-suffix reasoning --limit 50
#   bash examples/eval_across_domains.sh --domains chart_ocr,stem --model-suffix qwen3_zs
#
# Env (set_paths.sh): CHECKPOINT_PATH, PORT, API_KEY, MODEL_VERSION, SYSTEM_PROMPT,
# MCP_SERVER_PATH (''=tools off), MAX_TOOL_ITERATIONS (10), JUDGE_BACKEND (default
# engine = sequential in-process judge while the one-GPU policy server sleeps).
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; VERO_EVAL_DIR="$(cd "$SELF/.." && pwd)"
set +u
[[ -f "$VERO_EVAL_DIR/set_paths.sh" ]] && source "$VERO_EVAL_DIR/set_paths.sh" || source "$VERO_EVAL_DIR/set_paths.sh.example"
set -u
source "$VERO_EVAL_DIR/examples/_domain_map.sh"

DOMAINS="all"; SUFFIX="reasoning"; OUTPUT_ROOT=""; LIMIT="${LIMIT:-}"; NO_SERVE=0
PORT="${PORT:-9001}"; API_KEY="${API_KEY:-qwen}"; KEEP_SERVER="${KEEP_SERVER:-1}"
export JUDGE_BACKEND="${JUDGE_BACKEND:-engine}"
while [[ $# -gt 0 ]]; do case "$1" in
  --domains) DOMAINS="$2"; shift 2;; --model-suffix) SUFFIX="$2"; shift 2;;
  --output-root) OUTPUT_ROOT="$2"; shift 2;; --limit) LIMIT="$2"; shift 2;;
  --no-serve) NO_SERVE=1; shift;; *) echo "Unknown arg: $1" >&2; shift;;
esac; done

if [[ "$DOMAINS" == "all" ]]; then DOMAIN_LIST=("${VERO_DOMAINS[@]}"); else IFS=',' read -r -a DOMAIN_LIST <<< "$DOMAINS"; fi
OUTPUT_ROOT="${OUTPUT_ROOT:-./results/${MODEL_VERSION:-qwen_vllm}_$(date -u +%Y%m%d_%H%M%S)}"

cd "$VERO_EVAL_DIR"
STARTED_SERVER=0
if [[ "$NO_SERVE" != 1 ]]; then
  if curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models"; then
    echo "=== Reusing existing vLLM server on :$PORT ==="
  else
    echo "=== Starting vLLM server (one load for all domains) ==="; bash scripts/serve_qwen_vllm.sh; STARTED_SERVER=1
  fi
  curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
    || { echo "ERROR: vLLM not responding on :$PORT"; exit 1; }
fi

for DOMAIN in "${DOMAIN_LIST[@]}"; do
  TASKS="$(expand_domain "$DOMAIN" "$SUFFIX")" || { echo "skip domain $DOMAIN"; continue; }
  echo ""; echo "======== domain: $DOMAIN (suffix _$SUFFIX) ========"; echo "tasks: $TASKS"
  OUTPUT_PATH="$OUTPUT_ROOT/$DOMAIN" bash examples/eval_domain.sh --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"}
done

if [[ "$STARTED_SERVER" == 1 && "$KEEP_SERVER" != 1 ]]; then
  echo "=== Stopping vLLM server we started ==="; pkill -f "vllm serve" || true
fi
echo ""; echo "=== Done. Results under: $OUTPUT_ROOT ==="
