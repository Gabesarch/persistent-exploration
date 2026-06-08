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
#SBATCH --output=/home/lc1556/repo/reason-vlm/vero2/tools_scratch/logs/probe_toolimg_%j.out
#SBATCH --error=/home/lc1556/repo/reason-vlm/vero2/tools_scratch/logs/probe_toolimg_%j.err

set -euo pipefail
ROOT_DIR=/home/lc1556/repo/reason-vlm/vero2
TOOLS_DIR="$ROOT_DIR/tools_scratch"
BASE_URL="http://localhost:9001/v1"; API_KEY="qwen"

mkdir -p "$TOOLS_DIR/logs"
set +u; source ~/.bashrc; conda activate qwen35; set -u
export VLLM_USE_FLASHINFER_SAMPLER=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_DATASETS_OFFLINE=1
export no_proxy=localhost,127.0.0.1 NO_PROXY=localhost,127.0.0.1

cd "$TOOLS_DIR"
if curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/models"; then
  echo "=== Reusing vLLM server at $BASE_URL ==="
else
  echo "=== Starting vLLM server ==="; bash serve_qwen_vllm.sh
fi
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/models" \
  || { echo "ERROR: vLLM not responding"; exit 1; }

echo "=== Running tool-image probe ==="
BASE_URL="$BASE_URL" API_KEY="$API_KEY" MODEL_VERSION="qwen_vllm" python probe_tool_image.py
