#!/bin/bash
#SBATCH --job-name=vero_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:2
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --output=vero_eval/logs/vero_eval_%j.out
#SBATCH --error=vero_eval/logs/vero_eval_%j.err
#
# ============================================================================
# ONE entrypoint — pick MODEL, TASKS, TOOLS, submit from the repo root:
#
#   sbatch --export=ALL,MODEL=qwen3vl,TASKS=countqa_reasoning,visual_probe_easy_qwen3_zs,TOOLS=on \
#          vero_eval/run_eval.sbatch
#
#   sbatch --export=ALL,MODEL=vero,CHECKPOINT_PATH=/scratch/.../actor/huggingface,TASKS=all,TOOLS=on \
#          vero_eval/run_eval.sbatch
#
#   # exact-match-only run (no judge) -> 1 GPU is enough:
#   sbatch --gres=gpu:1 --export=ALL,MODEL=qwen3vl,TASKS=countqa_reasoning,TOOLS=on \
#          vero_eval/run_eval.sbatch
# ----------------------------------------------------------------------------
# MODEL  : vero | vero-thinking | qwen3vl | qwen3vl-thinking | qwen25vl | <abs ckpt path>
#          Sets CHECKPOINT_PATH + the default task SUFFIX. For Vero the reasoning system
#          prompt is baked into its chat template, so none is passed.
# TASKS  : a comma-list of task NAMES (countqa_reasoning,chartqa_reasoning,...)  OR a DOMAIN
#          (chart_ocr|stem|spatial_action|knowledge_rec|grounding_counting_search|
#           instruction_following|ball_drop|all). For a domain, SUFFIX is appended per task.
# TOOLS  : on (default) | off              enable/disable the MCP visual tools
# SUFFIX : override the per-task suffix for a DOMAIN run (default per MODEL)
# Other  : LIMIT, MAX_TOOL_ITERATIONS (turns, default 10), JUDGE_BACKEND, SYSTEM_PROMPT, OUTPUT_ROOT
#
# Judge: a judge task auto-launches a dedicated Qwen3-32B judge on GPU1 (the reportable
# judge). Submit with --gres=gpu:1 for exact-match-only runs (e.g. countqa).
# ============================================================================
set -euo pipefail

VERO_EVAL_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/vero_eval"
[[ -f "$VERO_EVAL_DIR/set_paths.sh.example" ]] || { echo "ERROR: $VERO_EVAL_DIR not found (submit from repo root)" >&2; exit 1; }
set +u
[[ -f "$VERO_EVAL_DIR/set_paths.sh" ]] && source "$VERO_EVAL_DIR/set_paths.sh" || source "$VERO_EVAL_DIR/set_paths.sh.example"
set -u
source "$VERO_EVAL_DIR/examples/_domain_map.sh"

: "${TASKS:?set TASKS=<task1,task2,...> or a domain (e.g. TASKS=grounding_counting_search)}"
MODEL="${MODEL:-qwen3vl}"; TOOLS="${TOOLS:-on}"; SUFFIX="${SUFFIX:-}"; LIMIT="${LIMIT:-}"
PORT="${PORT:-9001}"; API_KEY="${API_KEY:-qwen}"
export NUM_GPUS="${NUM_GPUS:-1}"   # policy uses 1 GPU; a judge task gets the 2nd

# ── 1) MODEL -> CHECKPOINT_PATH + default SUFFIX + system-prompt policy ──
case "$MODEL" in
  vero)             : "${CHECKPOINT_PATH:?MODEL=vero needs CHECKPOINT_PATH=<vero ckpt>}"; SUFFIX="${SUFFIX:-reasoning}"; export SYSTEM_PROMPT="";;
  vero-thinking)    : "${CHECKPOINT_PATH:?MODEL=vero-thinking needs CHECKPOINT_PATH=<ckpt>}"; SUFFIX="${SUFFIX:-reasoning_samplingq3}"; export SYSTEM_PROMPT="";;
  qwen3vl)          export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/scratch/gpfs/ZHUANGL/hw0171/Qwen3-VL-8B-Instruct}"; SUFFIX="${SUFFIX:-qwen3_zs}";;
  qwen3vl-thinking) : "${CHECKPOINT_PATH:?MODEL=qwen3vl-thinking needs CHECKPOINT_PATH=<Qwen3-VL-Thinking>}"; SUFFIX="${SUFFIX:-qwen3_thinking_zs}";;
  qwen25vl)         : "${CHECKPOINT_PATH:?MODEL=qwen25vl needs CHECKPOINT_PATH=<Qwen2.5-VL>}"; SUFFIX="${SUFFIX:-qwen25_zs}";;
  /*)               export CHECKPOINT_PATH="$MODEL"; SUFFIX="${SUFFIX:-reasoning}";;
  *)                echo "ERROR: unknown MODEL '$MODEL' (vero|vero-thinking|qwen3vl|qwen3vl-thinking|qwen25vl|<abs path>)" >&2; exit 1;;
esac
export CHECKPOINT_PATH PORT API_KEY
export MODEL_NAME="${MODEL_NAME:-qwen_vllm}" MODEL_VERSION="${MODEL_VERSION:-qwen_vllm}"

# ── 2) TOOLS on/off ──
if [[ "$TOOLS" == "off" ]]; then export MCP_SERVER_PATH=""; else export MCP_SERVER_PATH="${MCP_SERVER_PATH:-lmms_eval/mcp/vero2_mcp.py}"; fi

OUTPUT_ROOT="${OUTPUT_ROOT:-$VERO_EVAL_DIR/results/${MODEL//\//_}_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$VERO_EVAL_DIR/logs"; cd "$VERO_EVAL_DIR"
echo "MODEL=$MODEL  CHECKPOINT_PATH=$CHECKPOINT_PATH"
echo "TASKS=$TASKS  SUFFIX=$SUFFIX  TOOLS=$TOOLS  OUTPUT_ROOT=$OUTPUT_ROOT"

# ── 3) serve the policy (GPU0) ──
bash scripts/serve_qwen_vllm.sh
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
  || { echo "ERROR: vLLM policy not up on :$PORT"; exit 1; }

# ── 4) TASKS = a domain (or 'all') -> loop domains; else a literal task list ──
if [[ "$TASKS" == "all" || -n "${DOMAIN_TASK_BASES[$TASKS]:-}" ]]; then
  bash examples/eval_across_domains.sh --no-serve --domains "$TASKS" --model-suffix "$SUFFIX" \
    --output-root "$OUTPUT_ROOT" ${LIMIT:+--limit "$LIMIT"}
else
  bash examples/eval_tasks.sh --no-serve --tasks "$TASKS" \
    --output-path "$OUTPUT_ROOT" ${LIMIT:+--limit "$LIMIT"}
fi
echo "=== Done. Results under: $OUTPUT_ROOT ==="
