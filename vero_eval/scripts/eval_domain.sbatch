#!/bin/bash
#SBATCH --job-name=eval_domain
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --output=vero_eval/logs/eval_domain_%j.out
#SBATCH --error=vero_eval/logs/eval_domain_%j.err
#
# Domain-specific evaluator with predefined task sets. Submit from repo root:
#   DOMAIN=counting sbatch vero_eval/scripts/eval_domain.sbatch
#   DOMAIN=spatial  sbatch vero_eval/scripts/eval_domain.sbatch
#
# Available domains:
#   counting   - CountQA (object counting)
#   spatial    - CV-Bench (2D/3D spatial reasoning)
#   perception - BLINK (multi-image perception)
#   search     - V*Star Bench (visual search)
#   all        - All drawing-enabled tasks
#
# Environment variables:
#   DOMAIN         (required) One of: counting, spatial, perception, search, all
#   LIMIT          (optional) Limit samples (default: full dataset)
#   WITH_TOOLS     (optional) 1=enable tools, 0=disable (default: 1)
#   COMPARE        (optional) 1=run both with/without tools (default: 0)

set -euo pipefail

# Locate vero_eval directory
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$SUBMIT_DIR/set_paths.sh.example" && "$(basename "$SUBMIT_DIR")" == "vero_eval" ]]; then
  VERO_EVAL_DIR="$SUBMIT_DIR"
else
  VERO_EVAL_DIR="$SUBMIT_DIR/vero_eval"
fi
if [[ ! -f "$VERO_EVAL_DIR/set_paths.sh.example" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  VERO_EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [[ ! -f "$VERO_EVAL_DIR/set_paths.sh.example" ]]; then
  echo "ERROR: vero_eval dir not found. Submit from repo root:" >&2
  echo "  DOMAIN=counting sbatch vero_eval/scripts/eval_domain.sbatch" >&2
  exit 1
fi

set +u
if [[ -f "$VERO_EVAL_DIR/set_paths.sh" ]]; then
  source "$VERO_EVAL_DIR/set_paths.sh"
else
  echo "WARN: set_paths.sh not found; using set_paths.sh.example" >&2
  source "$VERO_EVAL_DIR/set_paths.sh.example"
fi
set -u

# Map domain to tasks
DOMAIN="${DOMAIN:-}"
case "$DOMAIN" in
  counting)
    TASKS="countqa_qwen3_drawing"
    DOMAIN_NAME="counting"
    ;;
  spatial)
    TASKS="cv_bench_qwen3_drawing"
    DOMAIN_NAME="spatial"
    ;;
  perception)
    TASKS="blink_qwen3_drawing"
    DOMAIN_NAME="perception"
    ;;
  search)
    TASKS="vstar_bench_qwen3_drawing"
    DOMAIN_NAME="search"
    ;;
  all)
    TASKS="countqa_qwen3_drawing,cv_bench_qwen3_drawing,blink_qwen3_drawing,vstar_bench_qwen3_drawing"
    DOMAIN_NAME="all"
    ;;
  *)
    echo "ERROR: DOMAIN is required. Options: counting, spatial, perception, search, all" >&2
    echo "  DOMAIN=counting sbatch vero_eval/scripts/eval_domain.sbatch" >&2
    exit 1
    ;;
esac

# Defaults
LIMIT="${LIMIT:-}"
WITH_TOOLS="${WITH_TOOLS:-1}"
COMPARE="${COMPARE:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$VERO_EVAL_DIR/results/${DOMAIN_NAME}}"
MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERS:-10}"
PORT="${PORT:-9001}"
API_KEY="${API_KEY:-qwen}"
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$VERO_EVAL_DIR/logs" "$OUTPUT_ROOT"
cd "$VERO_EVAL_DIR"

# Start or reuse vLLM server
if curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models"; then
  echo "=== Reusing existing vLLM server on :$PORT ==="
else
  echo "=== Starting vLLM server ==="; bash scripts/serve_qwen_vllm.sh
fi
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
  || { echo "ERROR: vLLM not responding on :$PORT"; exit 1; }

echo ""
echo "########## Domain: $DOMAIN_NAME ##########"
echo "Tasks:  $TASKS"
echo "Output: $OUTPUT_ROOT"
echo ""

if [[ "$COMPARE" == "1" ]]; then
  # Run both with and without tools for comparison
  echo ">>> WITH tools (max $MAX_TOOL_ITERATIONS iterations) <<<"
  SYSTEM_PROMPT="" MAX_TOOL_ITERATIONS="$MAX_TOOL_ITERATIONS" \
    OUTPUT_PATH="$OUTPUT_ROOT/with_tools" \
    bash examples/eval_domain.sh --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"}

  echo ""
  echo ">>> WITHOUT tools (single-turn) <<<"
  SYSTEM_PROMPT="" MCP_SERVER_PATH="" \
    OUTPUT_PATH="$OUTPUT_ROOT/no_tools" \
    bash examples/eval_domain.sh --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"}

  echo ""
  echo "=== Comparison complete. Results: $OUTPUT_ROOT/{with_tools,no_tools} ==="
else
  # Single run
  if [[ "$WITH_TOOLS" == "1" ]]; then
    SYSTEM_PROMPT="" MAX_TOOL_ITERATIONS="$MAX_TOOL_ITERATIONS" \
      OUTPUT_PATH="$OUTPUT_ROOT/with_tools" \
      bash examples/eval_domain.sh --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"}
  else
    SYSTEM_PROMPT="" MCP_SERVER_PATH="" \
      OUTPUT_PATH="$OUTPUT_ROOT/no_tools" \
      bash examples/eval_domain.sh --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"}
  fi

  echo ""
  echo "=== Evaluation complete. Results: $OUTPUT_ROOT ==="
fi
