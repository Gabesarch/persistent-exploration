#!/bin/bash
#SBATCH --job-name=eval_countqa_10
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --output=/home/hw0171/vero2/tools_scratch/logs/eval_countqa_10_%j.out
#SBATCH --error=/home/hw0171/vero2/tools_scratch/logs/eval_countqa_10_%j.err

set -euo pipefail

ROOT_DIR=/home/hw0171/vero2
TOOLS_DIR="$ROOT_DIR/tools_scratch"
EVAL_DIR="$TOOLS_DIR/eval"

TASK="${TASK:-countqa_reasoning}"
LIMIT="${LIMIT:-10}"
BASE_URL="${BASE_URL:-http://localhost:9001/v1}"
API_KEY="${API_KEY:-qwen}"
MODEL_VERSION="${MODEL_VERSION:-qwen_vllm}"
OUTPUT_PATH="${OUTPUT_PATH:-$EVAL_DIR/eval_results/countqa_10}"
USE_TOOLS="${USE_TOOLS:-1}"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-$TOOLS_DIR/prompts/countqa_system_prompt.txt}"

mkdir -p "$TOOLS_DIR/logs" "$OUTPUT_PATH"

set +u
source ~/.bashrc
conda activate verovlm
set -u

export VLLM_USE_FLASHINFER_SAMPLER=0

# Keep local resolution for tokenizer/datasets to avoid network dependencies.
export LMMS_EVAL_HF_TOKENIZER="${LMMS_EVAL_HF_TOKENIZER:-/scratch/gpfs/ZHUANGL/hf_models/Qwen3-32B}"
export HF_HOME="${HF_HOME:-/scratch/gpfs/ZHUANGL/hw0171/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/scratch/gpfs/ZHUANGL/hw0171/hf_cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/scratch/gpfs/ZHUANGL/hw0171/hf_cache/hub}"
export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE:-/scratch/gpfs/ZHUANGL/hw0171/hf_cache/hub}"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

cd "$TOOLS_DIR"

if curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/models"; then
  echo "=== Reusing existing vLLM server at $BASE_URL ==="
else
  echo "=== Starting qwen vLLM server with serve_qwen_vllm.sh ==="
  bash serve_qwen_vllm.sh
fi

echo "=== Verifying vLLM server responds at $BASE_URL ==="
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "${BASE_URL}/models" \
  || { echo "ERROR: vLLM not responding at $BASE_URL"; exit 1; }

cd "$EVAL_DIR"

echo "=== Running CountQA eval ==="
echo "Task:   $TASK"
echo "Limit:  $LIMIT"
echo "Output: $OUTPUT_PATH"
echo "Prompt: $SYSTEM_PROMPT_FILE"

if [[ "$USE_TOOLS" -eq 1 ]]; then
  MODEL_ARGS="model_version=${MODEL_VERSION},base_url=${BASE_URL},api_key=${API_KEY},mcp_server_path=lmms_eval/mcp/vero2_mcp.py,system_prompt=${SYSTEM_PROMPT_FILE}"
else
  MODEL_ARGS="model_version=${MODEL_VERSION},base_url=${BASE_URL},api_key=${API_KEY},system_prompt=${SYSTEM_PROMPT_FILE}"
fi

python -m lmms_eval \
  --model async_openai \
  --model_args "$MODEL_ARGS" \
  --tasks "$TASK" \
  --batch_size 1 \
  --log_samples \
  --output_path "$OUTPUT_PATH" \
  --limit "$LIMIT"

echo "=== CountQA eval complete ==="
