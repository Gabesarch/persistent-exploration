#!/usr/bin/env bash
# Default: 15 samples/model = first five Easy + first five Medium + first five Hard.
# DRY_RUN=1 bash examples/submit_zoom_eval_gemma_visual_probe.sh
# GEMMA_MODELS=gemma3_4b bash examples/submit_zoom_eval_gemma_visual_probe.sh
# Full dataset: LIMIT_PER_TASK='' PARTITION=pli-c ACCOUNT=... TIME_LIMIT=06:00:00 bash ...
set -euo pipefail
VERO_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
GEMMA_MODELS="${GEMMA_MODELS:-gemma3_4b gemma4_e4b}"
export GEMMA_EVAL_MODE="${GEMMA_EVAL_MODE:-zoom}"
case "$GEMMA_EVAL_MODE" in
  zoom) ;;
  thinking_only) GEMMA_MODELS=gemma4_e4b; export GEMMA4_ENABLE_THINKING=true ;;
  *) echo 'GEMMA_EVAL_MODE must be zoom or thinking_only' >&2; exit 1 ;;
esac
PARTITION="${PARTITION:-gputest}"; CONSTRAINT="${CONSTRAINT:-gpu80}"
TIME_LIMIT="${TIME_LIMIT:-01:00:00}"; LIMIT_PER_TASK="${LIMIT_PER_TASK-5}"
OUTPUT_BASE="${OUTPUT_BASE:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_gemma_zoom/test_$(date -u +%Y%m%d_%H%M%S)}"
GEMMA_CACHE_ROOT="${GEMMA_CACHE_ROOT:-/scratch/gpfs/ZHUANGL/gs1693/hf_cache}"
case "${GEMMA4_ENABLE_THINKING:-false}" in
  true|1|yes|on) export GEMMA4_ENABLE_THINKING=true ;;
  false|0|no|off) export GEMMA4_ENABLE_THINKING=false ;;
  *) echo 'GEMMA4_ENABLE_THINKING must be true or false' >&2; exit 1 ;;
esac
for family in $GEMMA_MODELS; do
  case "$family" in
    gemma3_4b) checkpoint="${GEMMA3_MODEL_PATH:-$GEMMA_CACHE_ROOT/google/gemma-3-4b-it}"; task_family=gemma3 ;;
    gemma4_e4b) checkpoint="${GEMMA4_MODEL_PATH:-$GEMMA_CACHE_ROOT/google/gemma-4-E4B-it}"; task_family=gemma4 ;;
    *) echo "Unknown model: $family" >&2; exit 1 ;;
  esac
  variant="$family"
  [[ "$family" != gemma4_e4b || "$GEMMA4_ENABLE_THINKING" != true ]] || variant="${family}_thinking"
  out="$OUTPUT_BASE/$variant"
  tasks="visual_probe_easy_${task_family}_zoom,visual_probe_medium_${task_family}_zoom,visual_probe_hard_${task_family}_zoom"
  if [[ "$GEMMA_EVAL_MODE" == thinking_only ]]; then
    variant="${family}_thinking_only"
    out="$OUTPUT_BASE/$variant"
    tasks="visual_probe_easy_gemma4_thinking_only,visual_probe_medium_gemma4_thinking_only,visual_probe_hard_gemma4_thinking_only"
    export LMMS_EVAL_USE_CACHE=False
    export GEMMA_EXPECTED_SAMPLE_COUNTS='{"visual_probe_easy_gemma4_thinking_only":141,"visual_probe_medium_gemma4_thinking_only":268,"visual_probe_hard_gemma4_thinking_only":106}'
  fi
  opts=(--parsable --job-name="gemma_zoom_${variant}" --constraint="$CONSTRAINT"
    --time="$TIME_LIMIT" --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=8 --mem=180G
    --chdir="$VERO_EVAL_DIR" --output="$out/slurm_%j.out" --error="$out/slurm_%j.err" --export=ALL)
  [[ -z "${ACCOUNT:-}" ]] || opts+=(--account="$ACCOUNT")
  # Della rejects literal --partition=gputest. Its submission plugin routes
  # GPU jobs with time <= 1h to partition=gputest, QOS=gpu-test automatically.
  if [[ "$PARTITION" == gputest ]]; then
    [[ "$TIME_LIMIT" == 01:00:00 || "$TIME_LIMIT" == 1:00:00 ]] || {
      echo 'PARTITION=gputest requires TIME_LIMIT=01:00:00 in this launcher.' >&2; exit 1;
    }
  else
    opts+=(--partition="$PARTITION")
  fi
  echo "Model=$family checkpoint=$checkpoint tasks=$tasks limit/task=${LIMIT_PER_TASK:-all} output=$out"
  printf 'sbatch '; printf '%q ' "${opts[@]}" "$VERO_EVAL_DIR/scripts/run_gemma_zoom_eval.sbatch"; printf '\n'
  if [[ "${DRY_RUN:-0}" != 1 ]]; then
    [[ -f "$checkpoint/config.json" ]] || { echo "Cache model first: $checkpoint (see scripts/cache_gemma_zoom_models.py)" >&2; exit 1; }
    [[ -f "${GEMMA_TRANSFORMERS_OVERLAY:-/scratch/gpfs/ZHUANGL/gs1693/vero2/runtime/gemma_zoom_transformers}/runtime_source.json" ]] || {
      echo 'Run python scripts/prepare_gemma_zoom_runtime.py first.' >&2; exit 1;
    }
    mkdir -p "$out"
    job=$(env VERO_EVAL_DIR="$VERO_EVAL_DIR" GEMMA_FAMILY="$family" CHECKPOINT_PATH="$checkpoint" OUTPUT_ROOT="$out" \
      TASKS="$tasks" LIMIT="$LIMIT_PER_TASK" sbatch "${opts[@]}" "$VERO_EVAL_DIR/scripts/run_gemma_zoom_eval.sbatch")
    printf '%s\n' "$job" | tee "$out/job_id.txt"
  fi
done
