#!/bin/bash
#SBATCH --job-name=probe_toolimg
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:30:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --output=vero_eval/logs/probe_toolimg_%j.out
#SBATCH --error=vero_eval/logs/probe_toolimg_%j.err
#
# Submit from the repo root:  sbatch vero_eval/scripts/probe_tool_image.sbatch
# Checks whether an image inside a role:"tool" message renders for the served model.

set -euo pipefail

# SLURM copies this script elsewhere, so derive the package dir from the submit
# directory (submit from the repo root, as the --output paths above assume).
VERO_EVAL_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/vero_eval"
if [[ ! -f "$VERO_EVAL_DIR/set_paths.sh.example" ]]; then
  echo "ERROR: $VERO_EVAL_DIR not found. Submit from the repo root:" >&2
  echo "  sbatch vero_eval/scripts/probe_tool_image.sbatch" >&2
  exit 1
fi
set +u
if [[ -f "$VERO_EVAL_DIR/set_paths.sh" ]]; then
  source "$VERO_EVAL_DIR/set_paths.sh"
else
  echo "WARN: $VERO_EVAL_DIR/set_paths.sh not found; using set_paths.sh.example." >&2
  source "$VERO_EVAL_DIR/set_paths.sh.example"
fi
set -u

PORT="${PORT:-9001}"; API_KEY="${API_KEY:-qwen}"
export VLLM_USE_FLASHINFER_SAMPLER=0
mkdir -p "$VERO_EVAL_DIR/logs"
cd "$VERO_EVAL_DIR"

if curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models"; then
  echo "=== Reusing vLLM server on :$PORT ==="
else
  echo "=== Starting vLLM server ==="; bash scripts/serve_qwen_vllm.sh
fi
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
  || { echo "ERROR: vLLM not responding on :$PORT"; exit 1; }

echo "=== Running tool-image probe ==="
BASE_URL="http://localhost:${PORT}/v1" API_KEY="$API_KEY" MODEL_VERSION="${MODEL_NAME:-qwen_vllm}" \
  python scripts/probe_tool_image.py
