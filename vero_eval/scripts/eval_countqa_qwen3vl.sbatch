#!/bin/bash
#SBATCH --job-name=countqa_qwen3vl
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=6:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --output=vero_eval/logs/countqa_qwen3vl_%j.out
#SBATCH --error=vero_eval/logs/countqa_qwen3vl_%j.err
#
# Submit from the repo root:  sbatch vero_eval/scripts/eval_countqa_qwen3vl.sbatch
#
# Evaluates Qwen3-VL-8B-Instruct (CHECKPOINT_PATH from set_paths.sh) on CountQA in
# TWO configs, serving the policy/tool model via vLLM once and reusing it:
#   1) WITH tools + multi-turn  (MCP tool loop, up to MAX_TOOL_ITERATIONS turns)
#   2) WITHOUT tools            (single-turn, tools disabled)
# Only the tool variable differs (no system prompt in either) so the two numbers
# are directly comparable. Task yaml = countqa_qwen3_zs (per-model: qwen3vl-instruct).
#
# Override LIMIT to smoke-test on a subset:  LIMIT=20 sbatch ...   (default: full 1001)

set -euo pipefail

# SLURM copies this script elsewhere, so derive the package dir from the submit
# directory (submit from the repo root, as the --output paths above assume).
VERO_EVAL_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/vero_eval"
if [[ ! -f "$VERO_EVAL_DIR/set_paths.sh.example" ]]; then
  echo "ERROR: $VERO_EVAL_DIR not found. Submit from the repo root:" >&2
  echo "  sbatch vero_eval/scripts/eval_countqa_qwen3vl.sbatch" >&2
  exit 1
fi
set +u
if [[ -f "$VERO_EVAL_DIR/set_paths.sh" ]]; then
  source "$VERO_EVAL_DIR/set_paths.sh"
else
  echo "WARN: $VERO_EVAL_DIR/set_paths.sh not found; using set_paths.sh.example (copy & edit it)." >&2
  source "$VERO_EVAL_DIR/set_paths.sh.example"
fi
set -u

LIMIT="${LIMIT:-}"                       # empty = full test split (1001)
PORT="${PORT:-9001}"; API_KEY="${API_KEY:-qwen}"
TASK="${TASK:-countqa_qwen3_zs}"
MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERATIONS:-10}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$VERO_EVAL_DIR/results/countqa_qwen3vl}"
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$VERO_EVAL_DIR/logs" "$OUTPUT_ROOT"
cd "$VERO_EVAL_DIR"

# Serve the policy/tool model once (reuse if already up).
if curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models"; then
  echo "=== Reusing existing vLLM server on :$PORT ==="
else
  echo "=== Starting vLLM server ==="; bash scripts/serve_qwen_vllm.sh
fi
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
  || { echo "ERROR: vLLM not responding on :$PORT"; exit 1; }

# 1) WITH tools + multi-turn (MCP loop). No system prompt.
echo ""; echo "########## CountQA — WITH tools + multi-turn (max ${MAX_TOOL_ITERATIONS} turns) ##########"
SYSTEM_PROMPT="" MAX_TOOL_ITERATIONS="$MAX_TOOL_ITERATIONS" \
  OUTPUT_PATH="$OUTPUT_ROOT/with_tools" \
  bash examples/eval_domain.sh --tasks "$TASK" ${LIMIT:+--limit "$LIMIT"}

# 2) WITHOUT tools (single-turn). Empty MCP_SERVER_PATH disables the tool loop.
echo ""; echo "########## CountQA — WITHOUT tools (single-turn) ##########"
SYSTEM_PROMPT="" MCP_SERVER_PATH="" \
  OUTPUT_PATH="$OUTPUT_ROOT/no_tools" \
  bash examples/eval_domain.sh --tasks "$TASK" ${LIMIT:+--limit "$LIMIT"}

echo ""
echo "=== CountQA qwen3vl eval complete. Results under: $OUTPUT_ROOT ==="
echo "    with_tools/  vs  no_tools/   (metric: exact_match)"
