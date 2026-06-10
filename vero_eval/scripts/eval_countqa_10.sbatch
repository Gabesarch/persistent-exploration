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

# SLURM copies this script elsewhere, so prefer the submit directory. In an
# interactive allocation SLURM_SUBMIT_DIR may already be the package dir.
SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$SUBMIT_DIR/set_paths.sh.example" && "$(basename "$SUBMIT_DIR")" == "vero_eval" ]]; then
  VERO_EVAL_DIR="$SUBMIT_DIR"
else
  VERO_EVAL_DIR="$SUBMIT_DIR/vero_eval"
fi

# When run directly with `bash path/to/script`, fall back to the script location
# if SLURM_SUBMIT_DIR points somewhere unrelated.
if [[ ! -f "$VERO_EVAL_DIR/set_paths.sh.example" ]]; then
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  VERO_EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
fi
if [[ ! -f "$VERO_EVAL_DIR/set_paths.sh.example" ]]; then
  echo "ERROR: $VERO_EVAL_DIR not found. Submit from the repo root:" >&2
  echo "  sbatch vero_eval/scripts/eval_countqa_10.sbatch" >&2
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
