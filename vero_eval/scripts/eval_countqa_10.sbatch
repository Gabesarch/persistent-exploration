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
#SBATCH --output=vero_eval/logs/eval_countqa_10_%j.out
#SBATCH --error=vero_eval/logs/eval_countqa_10_%j.err
#
# Submit from the repo root:  sbatch vero_eval/scripts/eval_countqa_10.sbatch
# Serves Qwen3-VL via vLLM once, then runs three CountQA configs at limit=10:
#   A) countqa_reasoning + system prompt + tools
#   B) countqa_reasoning + tools, no system prompt
#   C) countqa_qwen3_zs  + tools, no system prompt   (per-model yaml: qwen3vl-instruct -> qwen3_zs)

set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERO_EVAL_DIR="$(cd "$SELF/.." && pwd)"
set +u
if [[ -f "$VERO_EVAL_DIR/set_paths.sh" ]]; then
  source "$VERO_EVAL_DIR/set_paths.sh"
else
  echo "WARN: $VERO_EVAL_DIR/set_paths.sh not found; using set_paths.sh.example (copy & edit it)." >&2
  source "$VERO_EVAL_DIR/set_paths.sh.example"
fi
set -u

LIMIT="${LIMIT:-10}"
PORT="${PORT:-9001}"; API_KEY="${API_KEY:-qwen}"
OUTPUT_ROOT="${OUTPUT_ROOT:-$VERO_EVAL_DIR/results/countqa_10}"
SYSTEM_PROMPT="${SYSTEM_PROMPT:-$VERO_EVAL_DIR/prompts/countqa_system_prompt.txt}"
export VLLM_USE_FLASHINFER_SAMPLER=0
export LMMS_EVAL_HF_TOKENIZER="${LMMS_EVAL_HF_TOKENIZER:-/scratch/gpfs/ZHUANGL/hf_models/Qwen3-32B}"

mkdir -p "$VERO_EVAL_DIR/logs" "$OUTPUT_ROOT"
cd "$VERO_EVAL_DIR"

# Serve the policy/tool model (reuse if already up).
if curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models"; then
  echo "=== Reusing existing vLLM server on :$PORT ==="
else
  echo "=== Starting vLLM server ==="; bash scripts/serve_qwen_vllm.sh
fi
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
  || { echo "ERROR: vLLM not responding on :$PORT"; exit 1; }

# A) reasoning + system prompt + tools
SYSTEM_PROMPT="$SYSTEM_PROMPT" OUTPUT_PATH="$OUTPUT_ROOT/reasoning_with_sys_prompt" \
  bash examples/eval_domain.sh --tasks countqa_reasoning --limit "$LIMIT"

# B) reasoning + tools, no system prompt
SYSTEM_PROMPT="" OUTPUT_PATH="$OUTPUT_ROOT/reasoning_without_sys_prompt" \
  bash examples/eval_domain.sh --tasks countqa_reasoning --limit "$LIMIT"

# C) qwen3 zero-shot template + tools, no system prompt
SYSTEM_PROMPT="" OUTPUT_PATH="$OUTPUT_ROOT/zs_without_sys_prompt" \
  bash examples/eval_domain.sh --tasks countqa_qwen3_zs --limit "$LIMIT"

echo ""
echo "=== CountQA eval complete. Results under: $OUTPUT_ROOT ==="
