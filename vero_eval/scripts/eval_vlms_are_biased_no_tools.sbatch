#!/bin/bash
#SBATCH --job-name=eval_bias_no_tools
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --output=vero_eval/logs/eval_bias_no_tools_%j.out
#SBATCH --error=vero_eval/logs/eval_bias_no_tools_%j.err
#
# Submit from the repo root:
#   sbatch vero_eval/scripts/eval_vlms_are_biased_no_tools.sbatch

set -euo pipefail

VERO_EVAL_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/vero_eval"
if [[ ! -f "$VERO_EVAL_DIR/set_paths.sh.example" ]]; then
  echo "ERROR: $VERO_EVAL_DIR not found. Submit from the repo root:" >&2
  echo "  sbatch vero_eval/scripts/eval_vlms_are_biased_no_tools.sbatch" >&2
  exit 1
fi

LIMIT="${LIMIT:-}"
TASK="${TASK:-vlms_are_biased_reasoning}"
OUTPUT_PATH="${OUTPUT_PATH:-$VERO_EVAL_DIR/results/vlms_are_biased/no_tools_reasoning}"
export JUDGE_BACKEND="${JUDGE_BACKEND:-server}"
export MCP_SERVER_PATH=""

mkdir -p "$VERO_EVAL_DIR/logs" "$OUTPUT_PATH"
cd "$VERO_EVAL_DIR"

echo "=== Running VLMs-are-biased WITHOUT tools ==="
echo "Task:   $TASK"
echo "Output: $OUTPUT_PATH"

bash examples/eval_tasks.sh \
  --tasks "$TASK" \
  --output-path "$OUTPUT_PATH" \
  ${LIMIT:+--limit "$LIMIT"}

echo "=== VLMs-are-biased no-tools eval complete ==="
