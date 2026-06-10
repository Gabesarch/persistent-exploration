#!/bin/bash
#SBATCH --job-name=eval_bias_tools
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --output=vero_eval/logs/eval_bias_tools_%j.out
#SBATCH --error=vero_eval/logs/eval_bias_tools_%j.err
#
# Submit from the repo root:
#   sbatch vero_eval/scripts/eval_vlms_are_biased_with_tools.sbatch

set -euo pipefail

VERO_EVAL_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/vero_eval"
if [[ ! -f "$VERO_EVAL_DIR/set_paths.sh.example" ]]; then
  echo "ERROR: $VERO_EVAL_DIR not found. Submit from the repo root:" >&2
  echo "  sbatch vero_eval/scripts/eval_vlms_are_biased_with_tools.sbatch" >&2
  exit 1
fi

LIMIT="${LIMIT:-}"
TASK="${TASK:-vlms_are_biased_reasoning}"
OUTPUT_PATH="${OUTPUT_PATH:-$VERO_EVAL_DIR/results/vlms_are_biased/with_tools_reasoning}"
export JUDGE_BACKEND="${JUDGE_BACKEND:-server}"
export MCP_SERVER_PATH="${MCP_SERVER_PATH:-lmms_eval/mcp/vero2_mcp.py}"

mkdir -p "$VERO_EVAL_DIR/logs" "$OUTPUT_PATH"
cd "$VERO_EVAL_DIR"

echo "=== Running VLMs-are-biased WITH tools ==="
echo "Task:   $TASK"
echo "Output: $OUTPUT_PATH"

bash examples/eval_tasks.sh \
  --tasks "$TASK" \
  --output-path "$OUTPUT_PATH" \
  ${LIMIT:+--limit "$LIMIT"}

echo "=== VLMs-are-biased with-tools eval complete ==="
