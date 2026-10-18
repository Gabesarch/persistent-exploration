#!/usr/bin/env bash
# Compare the Vero checkpoint runs evaluated with the Qwen3.5-27B judge.
#
# Usage, from /home/gs1693/repo/vero2/vero_eval:
#   bash examples/plot_eval_accuracy_over_steps_qwen35_27b_judge.sh
#
# Common overrides:
#   OUTPUT_DIR=/tmp bash examples/plot_eval_accuracy_over_steps_qwen35_27b_judge.sh
#   PERCENT=0 METRIC=acc bash examples/plot_eval_accuracy_over_steps_qwen35_27b_judge.sh
#   CONDITION_A_DIR=/path/to/hardonly_llmjudge \
#   CONDITION_B_DIR=/path/to/hardonly_llmjudge_lr5e6 \
#     bash examples/plot_eval_accuracy_over_steps_qwen35_27b_judge.sh
#   EXTRA_CONDITIONS='third=/path/to/eval_c fourth=/path/to/eval_d' bash ...
#   QWEN_NORMAL_BASELINE_DIR=/path/to/eval QWEN_NOTOOLS_BASELINE_DIR=/path/to/eval bash ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERO_EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$VERO_EVAL_DIR/lmms_eval/__init__.py" ]]; then
  echo "ERROR: expected to find lmms_eval under $VERO_EVAL_DIR" >&2
  exit 1
fi

# keep commented ones commented out

EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_qwen35_27b_judge}"
# CONDITION_A_LABEL="${CONDITION_A_LABEL:-hardonly_llmjudge}"
# CONDITION_A_DIR="${CONDITION_A_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_hardonly_llmjudge_visual_probe}"
CONDITION_B_LABEL="${CONDITION_B_LABEL:-hardonly_llmjudge_lr5e6}"
CONDITION_B_DIR="${CONDITION_B_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_hardonly_llmjudge_lr5e6_visual_probe}"
# CONDITION_C_LABEL="${CONDITION_C_LABEL:-branchstrata20}"
# CONDITION_C_DIR="${CONDITION_C_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_visual_probe}"
# CONDITION_D_LABEL="${CONDITION_D_LABEL:-branchstrata20_unmaskedprefill}"
# CONDITION_D_DIR="${CONDITION_D_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_unmaskedprefill_visual_probe}"
# CONDITION_E_LABEL="${CONDITION_E_LABEL:-answer_retry_branch_donly_llmjudge_forcedprefill}"
# CONDITION_E_DIR="${CONDITION_E_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_answer_retry_branch_donly_llmjudge_forcedprefill_visual_probe}"
CONDITION_F_LABEL="${CONDITION_F_LABEL:-branchstrata20_restart20}"
CONDITION_F_DIR="${CONDITION_F_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_restart20_visual_probe}"
# CONDITION_G_LABEL="${CONDITION_G_LABEL:-branchstrata20_unmasked_restart20}"
# CONDITION_G_DIR="${CONDITION_G_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_unmasked_restart20_visual_probe}"
CONDITION_H_LABEL="${CONDITION_H_LABEL:-branchstrata20_base_noforceanswer}"
CONDITION_H_DIR="${CONDITION_H_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_base_noforceanswer_visual_probe}"
CONDITION_I_LABEL="${CONDITION_I_LABEL:-teacheronly_retry_sft05_forceanswer}"
CONDITION_I_DIR="${CONDITION_I_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_teacheronly_retry_sft05_forceanswer_visual_probe}"
CONDITION_J_LABEL="${CONDITION_J_LABEL:-teacheronly_retry_sft05}"
CONDITION_J_DIR="${CONDITION_J_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_teacheronly_retry_sft05_visual_probe}"
# Training SLURM array task 11974343_1.
CONDITION_K_LABEL="${CONDITION_K_LABEL:-branchstrata20_fresh}"
CONDITION_K_DIR="${CONDITION_K_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_visual_probe}"
# Training SLURM arrays 11974347 and 12095251.
CONDITION_L_LABEL="${CONDITION_L_LABEL:-branchstrata20_fresh_prefixremoved_gspo}"
CONDITION_L_DIR="${CONDITION_L_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_prefixremoved_gspo_visual_probe}"
# Training SLURM array task 11981184_1.
CONDITION_M_LABEL="${CONDITION_M_LABEL:-branchstrata20_fresh_randomturncap_p05_x5_q075}"
CONDITION_M_DIR="${CONDITION_M_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_randomturncap_p05_x5_q075_visual_probe}"
# Training SLURM array 12095040.
CONDITION_N_LABEL="${CONDITION_N_LABEL:-branchstrata20_fresh_lr1e5}"
CONDITION_N_DIR="${CONDITION_N_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_lr1e5_visual_probe}"
# Training SLURM array 12095042.
CONDITION_O_LABEL="${CONDITION_O_LABEL:-branchstrata30_fresh}"
CONDITION_O_DIR="${CONDITION_O_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata30_fresh_visual_probe}"
# Training SLURM array task 12095041_1.
CONDITION_P_LABEL="${CONDITION_P_LABEL:-branchstrata10_fresh}"
CONDITION_P_DIR="${CONDITION_P_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata10_fresh_visual_probe}"
# Training SLURM array task 12095043_1.
CONDITION_Q_LABEL="${CONDITION_Q_LABEL:-branchstrata20_fresh_clip0005_0007}"
CONDITION_Q_DIR="${CONDITION_Q_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_clip0005_0007_visual_probe}"
# Critic-guided evals use tagged output roots so they cannot be confused with
# evaluations of the same checkpoints that did not enable answer retry.
CONDITION_R_LABEL="${CONDITION_R_LABEL:-branchstrata20_fresh_lr1e5_critic}"
CONDITION_R_DIR="${CONDITION_R_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_lr1e5_critic_visual_probe_critic_retry_thr0p5}"
CONDITION_S_LABEL="${CONDITION_S_LABEL:-branchstrata20_fresh_prefixremoved_gspo_lr1e5_critic}"
CONDITION_S_DIR="${CONDITION_S_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_prefixremoved_gspo_lr1e5_critic_visual_probe_critic_retry_thr0p5}"
# Dedicated-turn budget forcing with up to 10 tool calls, evaluated on the
# hardonly_llmjudge_lr5e6 checkpoints.
CONDITION_T_LABEL="${CONDITION_T_LABEL:-hardonly_llmjudge_lr5e6_budget_forcing_t10}"
CONDITION_T_DIR="${CONDITION_T_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_hardonly_llmjudge_lr5e6_visual_probe_budget_forcing_t10}"

# Fixed-model VisualProbe evals are plotted as dotted horizontal reference lines.
QWEN_NORMAL_BASELINE_LABEL="${QWEN_NORMAL_BASELINE_LABEL:-qwen3vl_8b_instruct_normal}"
QWEN_NORMAL_BASELINE_DIR="${QWEN_NORMAL_BASELINE_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_qwen3vl_8b_instruct_visual_probe_normal}"
QWEN_BUDGET_BASELINE_LABEL="${QWEN_BUDGET_BASELINE_LABEL:-qwen3vl_8b_instruct_budget_forcing_t10}"
QWEN_BUDGET_BASELINE_DIR="${QWEN_BUDGET_BASELINE_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_qwen3vl_8b_instruct_visual_probe_budget_forcing_t10}"
QWEN_NOTOOLS_BASELINE_LABEL="${QWEN_NOTOOLS_BASELINE_LABEL:-qwen3vl_8b_instruct_no_tools}"
QWEN_NOTOOLS_BASELINE_DIR="${QWEN_NOTOOLS_BASELINE_DIR:-$EVAL_OUTPUT_ROOT/base_qwen3vl8b_no_tools_visual_probe}"
VERO_NOTOOLS_BASELINE_LABEL="${VERO_NOTOOLS_BASELINE_LABEL:-vero_qwen3i_8b_no_tools}"
VERO_NOTOOLS_BASELINE_DIR="${VERO_NOTOOLS_BASELINE_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_vero_qwen3i_8b_visual_probe_normal}"

METRIC="${METRIC:-judge_accuracy}"
OUTPUT_DIR="${OUTPUT_DIR:-$VERO_EVAL_DIR/examples/plots}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-visual_probe_accuracy_over_steps_qwen35_27b_judge}"
TOOL_CALLS_OUTPUT_BASENAME="${TOOL_CALLS_OUTPUT_BASENAME:-visual_probe_avg_tool_calls_over_steps_qwen35_27b_judge}"
TITLE="${TITLE:-Visual Probe Accuracy over Training Steps (Qwen3.5-27B Judge)}"
TOOL_CALLS_TITLE="${TOOL_CALLS_TITLE:-Average Visual Probe Tool Calls over Training Steps (Qwen3.5-27B Judge)}"
PERCENT="${PERCENT:-1}"
LATEST_BY="${LATEST_BY:-name}"

mkdir -p "$OUTPUT_DIR"

conditions=(
  # "$CONDITION_A_LABEL=$CONDITION_A_DIR"
  "$CONDITION_B_LABEL=$CONDITION_B_DIR"
  # "$CONDITION_C_LABEL=$CONDITION_C_DIR"
  # "$CONDITION_D_LABEL=$CONDITION_D_DIR"
  # "$CONDITION_E_LABEL=$CONDITION_E_DIR"
  "$CONDITION_F_LABEL=$CONDITION_F_DIR"
  # "$CONDITION_G_LABEL=$CONDITION_G_DIR"
  "$CONDITION_H_LABEL=$CONDITION_H_DIR"
  "$CONDITION_I_LABEL=$CONDITION_I_DIR"
  "$CONDITION_J_LABEL=$CONDITION_J_DIR"
  "$CONDITION_K_LABEL=$CONDITION_K_DIR"
  "$CONDITION_L_LABEL=$CONDITION_L_DIR"
  "$CONDITION_M_LABEL=$CONDITION_M_DIR"
  "$CONDITION_N_LABEL=$CONDITION_N_DIR"
  "$CONDITION_O_LABEL=$CONDITION_O_DIR"
  "$CONDITION_P_LABEL=$CONDITION_P_DIR"
  "$CONDITION_Q_LABEL=$CONDITION_Q_DIR"
  "$CONDITION_T_LABEL=$CONDITION_T_DIR"
)

# Training and evaluation may still be producing these newer runs. Compare each
# critic condition as soon as its first result directory becomes available.
for critic_condition in \
  "$CONDITION_R_LABEL=$CONDITION_R_DIR" \
  "$CONDITION_S_LABEL=$CONDITION_S_DIR"; do
  critic_dir="${critic_condition#*=}"
  if [[ -d "$critic_dir" ]]; then
    conditions+=("$critic_condition")
  else
    echo "[WARN] Skipping unavailable critic condition: $critic_condition" >&2
  fi
done

if [[ -n "${EXTRA_CONDITIONS:-}" ]]; then
  read -r -a extra_conditions <<< "$EXTRA_CONDITIONS"
  conditions+=("${extra_conditions[@]}")
fi

baselines=()
for baseline in \
  "$QWEN_NORMAL_BASELINE_LABEL=$QWEN_NORMAL_BASELINE_DIR" \
  "$QWEN_BUDGET_BASELINE_LABEL=$QWEN_BUDGET_BASELINE_DIR" \
  "$QWEN_NOTOOLS_BASELINE_LABEL=$QWEN_NOTOOLS_BASELINE_DIR" \
  "$VERO_NOTOOLS_BASELINE_LABEL=$VERO_NOTOOLS_BASELINE_DIR"; do
  baseline_dir="${baseline#*=}"
  if [[ -d "$baseline_dir" ]]; then
    baselines+=("$baseline")
  else
    echo "[WARN] Skipping unavailable dotted-line baseline: $baseline" >&2
  fi
done

args=(
  "${conditions[@]}"
  --metric "$METRIC"
  --output "$OUTPUT_DIR/${OUTPUT_BASENAME}.png"
  --csv "$OUTPUT_DIR/${OUTPUT_BASENAME}.csv"
  --title "$TITLE"
  --latest-by "$LATEST_BY"
  --visual-probe-overall
  --tool-calls-output "$OUTPUT_DIR/${TOOL_CALLS_OUTPUT_BASENAME}.png"
  --tool-calls-csv "$OUTPUT_DIR/${TOOL_CALLS_OUTPUT_BASENAME}.csv"
  --tool-calls-title "$TOOL_CALLS_TITLE"
)

for baseline in "${baselines[@]}"; do
  args+=(--baseline "$baseline")
done

if [[ "$PERCENT" == "1" || "${PERCENT,,}" == "true" || "${PERCENT,,}" == "yes" ]]; then
  args+=(--percent)
fi

echo "Writing plot to $OUTPUT_DIR/${OUTPUT_BASENAME}.png"
echo "Writing CSV to $OUTPUT_DIR/${OUTPUT_BASENAME}.csv"
echo "Writing tool-calls plot to $OUTPUT_DIR/${TOOL_CALLS_OUTPUT_BASENAME}.png"
echo "Writing tool-calls CSV to $OUTPUT_DIR/${TOOL_CALLS_OUTPUT_BASENAME}.csv"
python "$SCRIPT_DIR/plot_eval_accuracy_over_steps.py" "${args[@]}"
