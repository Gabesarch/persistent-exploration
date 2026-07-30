#!/bin/bash
#SBATCH --job-name=eval_vprobe
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=2:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --output=vero_eval/logs/eval_vprobe_%j.out
#SBATCH --error=vero_eval/logs/eval_vprobe_%j.err
#
# Demo / runner for a JUDGE-BASED task (visual_probe -> metric judge_accuracy).
# Submit from the repo root:  sbatch vero_eval/scripts/eval_visual_probe.sbatch
#
# Default judge = JUDGE_BACKEND=engine: a reportable Qwen3-32B judge loads
# in-process after the policy server sleeps and remains cached until process exit.
# The legacy two-server layout remains explicit:
#   sbatch --gres=gpu:2 --export=ALL,JUDGE_BACKEND=server \
#     vero_eval/scripts/eval_visual_probe.sbatch

set -euo pipefail

VERO_EVAL_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/vero_eval"
if [[ ! -f "$VERO_EVAL_DIR/set_paths.sh.example" ]]; then
  echo "ERROR: $VERO_EVAL_DIR not found. Submit from the repo root: sbatch vero_eval/scripts/eval_visual_probe.sbatch" >&2
  exit 1
fi
set +u
[[ -f "$VERO_EVAL_DIR/set_paths.sh" ]] && source "$VERO_EVAL_DIR/set_paths.sh" || source "$VERO_EVAL_DIR/set_paths.sh.example"
set -u

TASK="${TASK:-visual_probe_easy_qwen3_zs}"
LIMIT="${LIMIT:-20}"
PORT="${PORT:-9001}"; API_KEY="${API_KEY:-qwen}"
export JUDGE_BACKEND="${JUDGE_BACKEND:-engine}"
OUTPUT_PATH="${OUTPUT_PATH:-$VERO_EVAL_DIR/results/${TASK}}"
export VLLM_USE_FLASHINFER_SAMPLER=0

mkdir -p "$VERO_EVAL_DIR/logs"
cd "$VERO_EVAL_DIR"

# Policy/tool server (reuse if already up). The default local lifecycle wakes it
# before generation and deep-sleeps it before in-process judge postprocessing.
if curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models"; then
  echo "=== Reusing existing vLLM server on :$PORT ==="
else
  echo "=== Starting vLLM server ==="; bash scripts/serve_qwen_vllm.sh
fi
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
  || { echo "ERROR: vLLM not responding on :$PORT"; exit 1; }

echo "=== Running judge-based eval: $TASK (JUDGE_BACKEND=$JUDGE_BACKEND, limit=$LIMIT) ==="
OUTPUT_PATH="$OUTPUT_PATH" bash examples/eval_domain.sh --tasks "$TASK" --limit "$LIMIT"
echo "=== Done. Results under: $OUTPUT_PATH ==="
