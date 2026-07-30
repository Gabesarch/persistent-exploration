#!/bin/bash
#SBATCH --job-name=eval_vprobe_full
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=8:00:00
#SBATCH --mem=120G
#SBATCH --cpus-per-task=12
#SBATCH --gres=gpu:1
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --output=vero_eval/logs/eval_vprobe_full_%j.out
#SBATCH --error=vero_eval/logs/eval_vprobe_full_%j.err
#
# ============================================================================
# FULL-SETTING VisualProbe zoom eval — the canonical, reportable configuration.
#
#   * max_pixels = 10,000,000  (the sweep plateau; set at SERVE time via MM_MAX_PIXELS)
#   * per-model task + sampling (sampling lives in the task yaml; you pick it by MODEL):
#       MODEL=vero    -> visual_probe_<probe>_reasoning_zoom   (TRAINING-FAITHFUL:
#                        forced-<think> chat template, temp 0.7 / top_p 1.0 / top_k -1,
#                        image_indexing: training_faithful)
#       MODEL=qwen3vl -> visual_probe_<probe>_qwen3_zoom       (base Qwen3-VL-Instruct:
#                        temp 1.0 / top_p 1.0 / top_k 40 / presence_penalty 2.0)
#   * judge = in-process Qwen3-32B, sequentially sharing one GPU with the policy.
#
# Submit from the repo ROOT (so SLURM_SUBMIT_DIR resolves vero_eval/):
#   # Vero RL checkpoint (training-faithful):
#   sbatch --export=ALL,MODEL=vero,CHECKPOINT_PATH=/scratch/.../actor/huggingface \
#          vero_eval/scripts/eval_visual_probe_full.sbatch
#   # Base Qwen3-VL-Instruct:
#   sbatch --export=ALL,MODEL=qwen3vl vero_eval/scripts/eval_visual_probe_full.sbatch
#   # easy/medium, or all three difficulties:
#   sbatch --export=ALL,MODEL=vero,CHECKPOINT_PATH=...,PROBE=all \
#          vero_eval/scripts/eval_visual_probe_full.sbatch
#   # quick smoke with the reportable one-GPU judge:
#   sbatch --export=ALL,MODEL=qwen3vl,LIMIT=10 \
#          vero_eval/scripts/eval_visual_probe_full.sbatch
# ----------------------------------------------------------------------------
# Env: MODEL(qwen3vl) | vero | vero-thinking | <abs ckpt path>; CHECKPOINT_PATH(req for
#      vero/abs); PROBE(hard) easy|medium|hard|all; MM_MAX_PIXELS(10000000); MM_MIN_PIXELS;
#      LIMIT; PORT(9001); NUM_GPUS(1, policy TP); JUDGE_BACKEND(engine); TASK(override).
# ============================================================================
set -euo pipefail

VERO_EVAL_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/vero_eval"
[[ -f "$VERO_EVAL_DIR/set_paths.sh.example" ]] || { echo "ERROR: $VERO_EVAL_DIR not found. Submit from the repo root: sbatch vero_eval/scripts/eval_visual_probe_full.sbatch" >&2; exit 1; }
set +u
[[ -f "$VERO_EVAL_DIR/set_paths.sh" ]] && source "$VERO_EVAL_DIR/set_paths.sh" || source "$VERO_EVAL_DIR/set_paths.sh.example"
set -u
export VLLM_USE_FLASHINFER_SAMPLER=0

MODEL="${MODEL:-qwen3vl}"; PROBE="${PROBE:-hard}"
export MM_MAX_PIXELS="${MM_MAX_PIXELS:-10000000}"   # the full-setting resolution (serve-time)
PORT="${PORT:-9001}"; API_KEY="${API_KEY:-qwen}"
export NUM_GPUS="${NUM_GPUS:-1}"                     # policy TP; judge reuses GPU sequentially
export JUDGE_BACKEND="${JUDGE_BACKEND:-engine}"

# ── MODEL -> CHECKPOINT_PATH + per-model zoom suffix + chat-template / system-prompt policy ──
case "$MODEL" in
  vero|vero-thinking)
    : "${CHECKPOINT_PATH:?MODEL=$MODEL needs CHECKPOINT_PATH=<vero ckpt, e.g. .../actor/huggingface>}"
    SUFFIX="reasoning_zoom"; export SYSTEM_PROMPT=""
    # Vero must be served with its forced-<think> TRAINING chat template, else it skips
    # <think>/tool calls and answers directly (mirrors run_eval.sbatch).
    if [[ -z "${CHAT_TEMPLATE:-}" && -f "$CHECKPOINT_PATH/chat_template.jinja" ]]; then
      export CHAT_TEMPLATE="$CHECKPOINT_PATH/chat_template.jinja"
    fi ;;
  qwen3vl)
    export CHECKPOINT_PATH="${CHECKPOINT_PATH:-/scratch/gpfs/ZHUANGL/hw0171/Qwen3-VL-8B-Instruct}"
    SUFFIX="qwen3_zoom" ;;
  /*)
    export CHECKPOINT_PATH="$MODEL"; SUFFIX="${SUFFIX:-reasoning_zoom}"
    [[ -z "${CHAT_TEMPLATE:-}" && -f "$CHECKPOINT_PATH/chat_template.jinja" ]] && export CHAT_TEMPLATE="$CHECKPOINT_PATH/chat_template.jinja" ;;
  *) echo "ERROR: unknown MODEL '$MODEL' (vero|vero-thinking|qwen3vl|<abs ckpt path>)" >&2; exit 1 ;;
esac
export CHECKPOINT_PATH PORT API_KEY
export MODEL_NAME="${MODEL_NAME:-qwen_vllm}" MODEL_VERSION="${MODEL_VERSION:-qwen_vllm}"
export MCP_SERVER_PATH="${MCP_SERVER_PATH:-lmms_eval/mcp/vero2_mcp.py}"   # zoom tool ON

# ── resolve the task list ──
if [[ -n "${TASK:-}" ]]; then
  TASKS="$TASK"
elif [[ "$PROBE" == "all" ]]; then
  TASKS="visual_probe_easy_${SUFFIX},visual_probe_medium_${SUFFIX},visual_probe_hard_${SUFFIX}"
else
  TASKS="visual_probe_${PROBE}_${SUFFIX}"
fi

OUTPUT_PATH="${OUTPUT_PATH:-$VERO_EVAL_DIR/results/vprobe_full_${MODEL//\//_}_px${MM_MAX_PIXELS}_$(date -u +%Y%m%d_%H%M%S)}"
mkdir -p "$VERO_EVAL_DIR/logs"; cd "$VERO_EVAL_DIR"
echo "=== VisualProbe FULL setting ==="
echo "MODEL=$MODEL  TASKS=$TASKS  MM_MAX_PIXELS=$MM_MAX_PIXELS"
echo "CHECKPOINT_PATH=$CHECKPOINT_PATH"
[[ -n "${CHAT_TEMPLATE:-}" ]] && echo "CHAT_TEMPLATE=$CHAT_TEMPLATE"

# ── serve the policy at the requested resolution (serve_qwen_vllm.sh restarts any server
#    on this PORT, so we never silently reuse a stale server at a different max_pixels) ──
bash scripts/serve_qwen_vllm.sh
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
  || { echo "ERROR: vLLM policy not up on :$PORT"; exit 1; }

# ── eval (wake for generation, then deep-sleep before in-process judge postprocessing) ──
OUTPUT_PATH="$OUTPUT_PATH" bash examples/eval_domain.sh --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"}
echo "=== Done. Results under: $OUTPUT_PATH ==="
