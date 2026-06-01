#!/bin/bash
# eval.sh — Run lmms_eval with MCP tools
#
# Usage:
#   bash examples/eval_domain.sh --tasks <TASKS> [--output-path <PATH>] [--batch-size <N>]
#
# Examples:
#   bash examples/eval_domain.sh --tasks countqa_reasoning
#   bash examples/eval_domain.sh --tasks "countqa_reasoning,countbenchqa_reasoning" --output-path ./results

set -euo pipefail

# ── defaults ──────────────────────────────────────────────────────────
TASKS=""
OUTPUT_PATH="./eval_results"
BATCH_SIZE=1

# ── parse args ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tasks)        TASKS="$2"; shift 2 ;;
    --output-path)  OUTPUT_PATH="$2"; shift 2 ;;
    --batch-size)   BATCH_SIZE="$2"; shift 2 ;;
    *)              shift ;;
  esac
done

if [[ -z "$TASKS" ]]; then
  echo "Usage: bash examples/eval_domain.sh --tasks <TASKS>"
  echo ""
  echo "Example: bash examples/eval_domain.sh --tasks countqa_reasoning"
  exit 1
fi

# ── HuggingFace cache and offline mode ────────────────────────────────
export HF_HOME=/scratch/gpfs/ZHUANGL/lc1556/hf_cache
export HF_HUB_CACHE=/scratch/gpfs/ZHUANGL/lc1556/hf_cache/hub
export HF_DATASETS_CACHE=/scratch/gpfs/ZHUANGL/lc1556/hf_cache/hub
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1

# ── judge model via vLLM server ───────────────────────────────────────
export VLLM_SERVER_JUDGE=1
export VLLM_JUDGE_BASE_URL="http://localhost:8003/v1"
export VLLM_JUDGE_PORT=8003
export VLLM_JUDGE_MODEL_PATH="qwen_vllm"
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# ── run ───────────────────────────────────────────────────────────────
echo "Tasks:   $TASKS"
echo "Output:  $OUTPUT_PATH"
echo ""

python -m lmms_eval \
    --model async_openai \
    --model_args "model_version=qwen_vllm,base_url=http://localhost:8002/v1,api_key=qwen,mcp_server_path=lmms_eval/mcp/vero2_mcp.py" \
    --tasks "$TASKS" \
    --batch_size "$BATCH_SIZE" \
    --log_samples \
    --output_path "$OUTPUT_PATH"
