#!/usr/bin/env bash
# Plot lmms-eval benchmark accuracy over checkpoint steps for common Vero eval runs.
#
# Usage, from /home/gs1693/repo/vero2/vero_eval:
#   bash examples/plot_eval_accuracy_over_steps.sh
#
# Common overrides:
#   OUTPUT_DIR=/tmp bash examples/plot_eval_accuracy_over_steps.sh
#   PERCENT=0 METRIC=acc bash examples/plot_eval_accuracy_over_steps.sh
#   CONDITION_A_LABEL=answer_retry CONDITION_A_DIR=/path/to/eval_a \
#   CONDITION_B_LABEL=hardonly CONDITION_B_DIR=/path/to/eval_b \
#     bash examples/plot_eval_accuracy_over_steps.sh
#   CONDITION_C_LABEL=llmjudge CONDITION_C_DIR=/path/to/eval_c \
#   CONDITION_D_LABEL=branch_donly CONDITION_D_DIR=/path/to/eval_d \
#     bash examples/plot_eval_accuracy_over_steps.sh
#   CONDITION_E_LABEL=hardonly_llmjudge CONDITION_E_DIR=/path/to/eval_e \
#     bash examples/plot_eval_accuracy_over_steps.sh
#   CONDITION_F_LABEL=branchstrata20 CONDITION_F_DIR=/path/to/eval_f \
#   CONDITION_G_LABEL=branchstrata20_unmaskedprefill CONDITION_G_DIR=/path/to/eval_g \
#   CONDITION_H_LABEL=hardonly_llmjudge_lr5e6 CONDITION_H_DIR=/path/to/eval_h \
#     bash examples/plot_eval_accuracy_over_steps.sh
#   EXTRA_CONDITIONS='third=/path/to/eval_c fourth=/path/to/eval_d' bash ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERO_EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$VERO_EVAL_DIR/lmms_eval/__init__.py" ]]; then
  echo "ERROR: expected to find lmms_eval under $VERO_EVAL_DIR" >&2
  exit 1
fi

CONDITION_A_LABEL="${CONDITION_A_LABEL:-answer_retry_forcedprefill}"
CONDITION_A_DIR="${CONDITION_A_DIR:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2/zoom_eval_answer_retry_forcedprefill_visual_probe}"
CONDITION_B_LABEL="${CONDITION_B_LABEL:-hardonly}"
CONDITION_B_DIR="${CONDITION_B_DIR:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2/zoom_eval_hardonly_visual_probe}"
CONDITION_C_LABEL="${CONDITION_C_LABEL:-answer_retry_llmjudge_forcedprefill}"
CONDITION_C_DIR="${CONDITION_C_DIR:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2/zoom_eval_answer_retry_llmjudge_forcedprefill_visual_probe}"
CONDITION_D_LABEL="${CONDITION_D_LABEL:-answer_retry_branch_donly_llmjudge_forcedprefill}"
CONDITION_D_DIR="${CONDITION_D_DIR:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2/zoom_eval_answer_retry_branch_donly_llmjudge_forcedprefill_visual_probe}"
CONDITION_E_LABEL="${CONDITION_E_LABEL:-hardonly_llmjudge}"
CONDITION_E_DIR="${CONDITION_E_DIR:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2/zoom_eval_hardonly_llmjudge_visual_probe}"
CONDITION_F_LABEL="${CONDITION_F_LABEL:-branchstrata20}"
CONDITION_F_DIR="${CONDITION_F_DIR:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2/zoom_eval_branchstrata20_visual_probe}"
CONDITION_G_LABEL="${CONDITION_G_LABEL:-branchstrata20_unmaskedprefill}"
CONDITION_G_DIR="${CONDITION_G_DIR:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2/zoom_eval_branchstrata20_unmaskedprefill_visual_probe}"
CONDITION_H_LABEL="${CONDITION_H_LABEL:-hardonly_llmjudge_lr5e6}"
CONDITION_H_DIR="${CONDITION_H_DIR:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2/zoom_eval_hardonly_llmjudge_lr5e6_visual_probe}"

METRIC="${METRIC:-judge_accuracy}"
OUTPUT_DIR="${OUTPUT_DIR:-$VERO_EVAL_DIR/examples/plots}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-visual_probe_accuracy_over_steps}"
TITLE="${TITLE:-Visual Probe Accuracy over Training Steps}"
PERCENT="${PERCENT:-1}"
LATEST_BY="${LATEST_BY:-name}"

mkdir -p "$OUTPUT_DIR"

conditions=(
  "$CONDITION_A_LABEL=$CONDITION_A_DIR"
  "$CONDITION_B_LABEL=$CONDITION_B_DIR"
  "$CONDITION_C_LABEL=$CONDITION_C_DIR"
  "$CONDITION_D_LABEL=$CONDITION_D_DIR"
  "$CONDITION_E_LABEL=$CONDITION_E_DIR"
)

# Grouped comparison runs may not all have produced a checkpoint/eval directory
# yet (notably a newly launched learning-rate control). Add each condition as
# soon as its output root exists without making plots of the available runs fail.
optional_conditions=(
  "$CONDITION_F_LABEL=$CONDITION_F_DIR"
  "$CONDITION_G_LABEL=$CONDITION_G_DIR"
  "$CONDITION_H_LABEL=$CONDITION_H_DIR"
)
for condition in "${optional_conditions[@]}"; do
  condition_dir="${condition#*=}"
  if [[ -d "$condition_dir" ]]; then
    conditions+=("$condition")
  else
    echo "[INFO] Skipping unavailable plot condition: $condition"
  fi
done

if [[ -n "${EXTRA_CONDITIONS:-}" ]]; then
  read -r -a extra_conditions <<< "$EXTRA_CONDITIONS"
  conditions+=("${extra_conditions[@]}")
fi

args=(
  "${conditions[@]}"
  --metric "$METRIC"
  --output "$OUTPUT_DIR/${OUTPUT_BASENAME}.png"
  --csv "$OUTPUT_DIR/${OUTPUT_BASENAME}.csv"
  --title "$TITLE"
  --latest-by "$LATEST_BY"
  --visual-probe-overall
)

if [[ "$PERCENT" == "1" || "${PERCENT,,}" == "true" || "${PERCENT,,}" == "yes" ]]; then
  args+=(--percent)
fi

echo "Writing plot to $OUTPUT_DIR/${OUTPUT_BASENAME}.png"
echo "Writing CSV to $OUTPUT_DIR/${OUTPUT_BASENAME}.csv"
python "$SCRIPT_DIR/plot_eval_accuracy_over_steps.py" "${args[@]}"
