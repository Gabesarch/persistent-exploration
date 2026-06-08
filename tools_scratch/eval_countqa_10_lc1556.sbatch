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
#SBATCH --output=/home/lc1556/repo/reason-vlm/vero2/tools_scratch/logs/eval_countqa_10_%j.out
#SBATCH --error=/home/lc1556/repo/reason-vlm/vero2/tools_scratch/logs/eval_countqa_10_%j.err

# Adapted from eval_countqa_10_verovlm.sbatch (hw0171) for lc1556.
# Serves Qwen3-VL-8B-Instruct via vLLM once, then runs three CountQA configs
# at limit=10 to reproduce the uploaded reference numbers:
#   A) countqa_reasoning + system prompt + tools   (ref 0.5)
#   B) countqa_reasoning + tools, no system prompt (ref 0.3)
#   C) countqa_qwen3_zs  + tools, no system prompt (ref 0.0)

set -euo pipefail

ROOT_DIR=/home/lc1556/repo/reason-vlm/vero2
TOOLS_DIR="$ROOT_DIR/tools_scratch"
EVAL_DIR="$TOOLS_DIR/eval"

LIMIT="${LIMIT:-10}"
BASE_URL="${BASE_URL:-http://localhost:9001/v1}"
API_KEY="${API_KEY:-qwen}"
MODEL_VERSION="${MODEL_VERSION:-qwen_vllm}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$EVAL_DIR/eval_results/countqa_10_lc1556}"
MCP_PATH="lmms_eval/mcp/vero2_mcp.py"
SYSTEM_PROMPT_FILE="${SYSTEM_PROMPT_FILE:-$TOOLS_DIR/prompts/countqa_system_prompt.txt}"

mkdir -p "$TOOLS_DIR/logs" "$OUTPUT_ROOT"

set +u
source ~/.bashrc
conda activate qwen35
set -u

export VLLM_USE_FLASHINFER_SAMPLER=0

# Keep local resolution for tokenizer/datasets to avoid network dependencies.
export LMMS_EVAL_HF_TOKENIZER="${LMMS_EVAL_HF_TOKENIZER:-/scratch/gpfs/ZHUANGL/hf_models/Qwen3-32B}"
export HF_HOME="${HF_HOME:-/scratch/gpfs/ZHUANGL/lc1556/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-/scratch/gpfs/ZHUANGL/lc1556/hf_cache/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/scratch/gpfs/ZHUANGL/lc1556/hf_cache/hub}"
export LMMS_EVAL_DATASETS_CACHE="${LMMS_EVAL_DATASETS_CACHE:-/scratch/gpfs/ZHUANGL/lc1556/hf_cache/hub}"
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

run_eval () {
  local LABEL="$1" TASK="$2" MODEL_ARGS="$3"
  local OUTPUT_PATH="$OUTPUT_ROOT/$LABEL"
  mkdir -p "$OUTPUT_PATH"
  echo ""
  echo "======================================================="
  echo "=== Running CountQA eval: $LABEL ==="
  echo "Task:       $TASK"
  echo "Limit:      $LIMIT"
  echo "Output:     $OUTPUT_PATH"
  echo "Model args: $MODEL_ARGS"
  echo "======================================================="
  python -m lmms_eval \
    --model async_openai \
    --model_args "$MODEL_ARGS" \
    --tasks "$TASK" \
    --batch_size 1 \
    --log_samples \
    --output_path "$OUTPUT_PATH" \
    --limit "$LIMIT"
}

COMMON="model_version=${MODEL_VERSION},base_url=${BASE_URL},api_key=${API_KEY},mcp_server_path=${MCP_PATH},max_tool_iterations=10"

# A) reasoning + system prompt + tools
run_eval "reasoning_with_sys_prompt" "countqa_reasoning" \
  "${COMMON},system_prompt=${SYSTEM_PROMPT_FILE}"

# B) reasoning + tools, no system prompt
run_eval "reasoning_without_sys_prompt" "countqa_reasoning" \
  "${COMMON}"

# C) qwen3 zero-shot template + tools, no system prompt
run_eval "zs_without_sys_prompt" "countqa_qwen3_zs" \
  "${COMMON}"

echo ""
echo "=== CountQA eval complete. Results under: $OUTPUT_ROOT ==="
