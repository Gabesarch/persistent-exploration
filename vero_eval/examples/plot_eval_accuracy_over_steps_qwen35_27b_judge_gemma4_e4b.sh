#!/usr/bin/env bash
# Separate Gemma4 E4B Visual Probe accuracy/tool-use plots, using saved results only.
# Run from any directory. Optional overrides mirror the selected9 wrapper:
#   OUTPUT_DIR=/tmp/gemma-plots bash /path/to/this/script.sh
#   CONDITION_GDPO_DIR=/path/to/run CONDITION_CRITIC_DIR=/path/to/run bash ...
#   Additional --condition LABEL=DIR / --baseline LABEL=DIR options are forwarded.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERO_EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_gemma_zoom}"
OUTPUT_DIR="${OUTPUT_DIR:-/scratch/gpfs/ZHUANGL/gs1693/artifacts/vero2/plots}"
CONDITION_GDPO_DIR="${CONDITION_GDPO_DIR:-$EVAL_OUTPUT_ROOT/training_v1/gdpo_fresh_lr1e5_noexplore_accformat_gemma4_e4b}"
CONDITION_CRITIC_DIR="${CONDITION_CRITIC_DIR:-$EVAL_OUTPUT_ROOT/training_v1/branchstrata20_fresh_lr1e5_critic_gemma4_e4b}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-visual_probe_accuracy_over_steps_gemma4_e4b_qwen35_27b_judge}"
TOOL_CALLS_OUTPUT_BASENAME="${TOOL_CALLS_OUTPUT_BASENAME:-visual_probe_avg_tool_calls_over_steps_gemma4_e4b_qwen35_27b_judge}"
args=(
  --root "$EVAL_OUTPUT_ROOT"
  --metric "${METRIC:-judge_accuracy}"
  --latest-by "${LATEST_BY:-name}"
  --output "$OUTPUT_DIR/$OUTPUT_BASENAME.png"
  --csv "$OUTPUT_DIR/$OUTPUT_BASENAME.csv"
  --tool-calls-output "$OUTPUT_DIR/$TOOL_CALLS_OUTPUT_BASENAME.png"
  --tool-calls-csv "$OUTPUT_DIR/$TOOL_CALLS_OUTPUT_BASENAME.csv"
  --manifest "$OUTPUT_DIR/${OUTPUT_BASENAME}_sources.json"
  --title "${TITLE:-Gemma4 E4B Visual Probe Accuracy (Qwen3.5-27B Judge)}"
  --tool-calls-title "${TOOL_CALLS_TITLE:-Gemma4 E4B Visual Probe Tool Calls}"
)
if [[ -n "${CONDITION_GDPO_DIR:-}" ]]; then
  args+=(--condition "${CONDITION_GDPO_LABEL:-Gemma4 E4B GDPO (thinking)}=$CONDITION_GDPO_DIR")
fi
if [[ -n "${CONDITION_CRITIC_DIR:-}" ]]; then
  args+=(--condition "${CONDITION_CRITIC_LABEL:-Gemma4 E4B retry + critic (thinking)}=$CONDITION_CRITIC_DIR")
fi
if [[ -n "${BASELINE_NONTHINKING_DIR:-}" ]]; then
  args+=(--baseline "Gemma4 E4B (non-thinking)=$BASELINE_NONTHINKING_DIR")
fi
if [[ -n "${BASELINE_THINKING_DIR:-}" ]]; then
  args+=(--baseline "Gemma4 E4B (thinking)=$BASELINE_THINKING_DIR")
fi
case "${PERCENT:-1}" in
  1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]) args+=(--percent) ;;
esac
export MPLBACKEND=Agg
"${PYTHON:-python}" "$SCRIPT_DIR/plot_eval_accuracy_over_steps_gemma4_e4b.py" "${args[@]}" "$@"
