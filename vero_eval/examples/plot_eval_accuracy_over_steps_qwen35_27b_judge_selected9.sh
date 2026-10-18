#!/usr/bin/env bash
# Compare selected checkpoint runs and three fixed-model baselines with the Qwen3.5-27B judge.
#
# Usage, from /home/gs1693/repo/vero2/vero_eval:
#   bash examples/plot_eval_accuracy_over_steps_qwen35_27b_judge_selected9.sh
#
# Common overrides:
#   OUTPUT_DIR=/tmp bash examples/plot_eval_accuracy_over_steps_qwen35_27b_judge_selected9.sh
#   PERCENT=0 METRIC=acc bash examples/plot_eval_accuracy_over_steps_qwen35_27b_judge_selected9.sh
#   CONDITION_B_DIR=/path/to/hardonly_llmjudge_lr5e6 \
#     bash examples/plot_eval_accuracy_over_steps_qwen35_27b_judge_selected9.sh
#   QWEN_NORMAL_BASELINE_DIR=/path/to/eval QWEN_NOTOOLS_BASELINE_DIR=/path/to/eval bash ...
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VERO_EVAL_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

if [[ ! -f "$VERO_EVAL_DIR/lmms_eval/__init__.py" ]]; then
  echo "ERROR: expected to find lmms_eval under $VERO_EVAL_DIR" >&2
  exit 1
fi


EVAL_OUTPUT_ROOT="${EVAL_OUTPUT_ROOT:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_qwen35_27b_judge}"
CONDITION_QWEN35_VERO2_LABEL="${CONDITION_QWEN35_VERO2_LABEL:-Vero2 zoom-in (Qwen3.5, XML)}"
CONDITION_QWEN35_VERO2_DIR="${CONDITION_QWEN35_VERO2_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_qwen35_vero2_step200_visual_probe}"
CONDITION_MINIO3_VERO2_LABEL="${CONDITION_MINIO3_VERO2_LABEL:-Vero2 with mini-o3 loss}"
CONDITION_MINIO3_VERO2_DIR="${CONDITION_MINIO3_VERO2_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_minio3_vero2_step100_visual_probe}"
CONDITION_FIXJSON_VERO2_LABEL="${CONDITION_FIXJSON_VERO2_LABEL:-Vero2 exp22.5 fixjson}"
CONDITION_FIXJSON_VERO2_DIR="${CONDITION_FIXJSON_VERO2_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_fixjson_vero2_step200_visual_probe}"
CONDITION_MINIO3_STEP200_LABEL="${CONDITION_MINIO3_STEP200_LABEL:-Vero2 mini-o3 fixKL (step 200)}"
CONDITION_MINIO3_STEP200_DIR="${CONDITION_MINIO3_STEP200_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_minio3_vero2_step200_visual_probe}"
CONDITION_R3L_LABEL="${CONDITION_R3L_LABEL:-opmd_r3l_visual_probe_14728996}"
CONDITION_R3L_DIR="${CONDITION_R3L_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_opmd_r3l_visual_probe_14728996_visual_probe}"
CONDITION_R3L_15071396_LABEL="${CONDITION_R3L_15071396_LABEL:-opmd_r3l_visual_probe_15071396}"
CONDITION_R3L_15071396_DIR="${CONDITION_R3L_15071396_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_opmd_r3l_visual_probe_15071396_visual_probe}"
CONDITION_R3L_15071396_PAPER_FIXES_LABEL="${CONDITION_R3L_15071396_PAPER_FIXES_LABEL:-opmd_r3l_visual_probe_15071396_paper_fixes}"
CONDITION_R3L_15071396_PAPER_FIXES_DIR="${CONDITION_R3L_15071396_PAPER_FIXES_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_opmd_r3l_visual_probe_15071396_paper_fixes_visual_probe}"
CONDITION_GDPO_LABEL="${CONDITION_GDPO_LABEL:-gdpo_fresh_lr1e5_noexplore}"
CONDITION_GDPO_DIR="${CONDITION_GDPO_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_gdpo_fresh_lr1e5_noexplore_visual_probe}"
CONDITION_GDPO_ACCFORMAT_LABEL="${CONDITION_GDPO_ACCFORMAT_LABEL:-gdpo_fresh_lr1e5_noexplore_accformat}"
CONDITION_GDPO_ACCFORMAT_DIR="${CONDITION_GDPO_ACCFORMAT_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_gdpo_fresh_lr1e5_noexplore_accformat_visual_probe}"
CONDITION_GDPO_ACCFORMAT_BUDGET_LABEL="${CONDITION_GDPO_ACCFORMAT_BUDGET_LABEL:-gdpo_fresh_lr1e5_noexplore_accformat_budget_forcing_t10}"
CONDITION_GDPO_ACCFORMAT_BUDGET_DIR="${CONDITION_GDPO_ACCFORMAT_BUDGET_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_gdpo_fresh_lr1e5_noexplore_accformat_visual_probe_budget_forcing_t10}"
# Full step-200 runs 15333548 / 15333549, scored by recovery job 15366331.
CONDITION_GDPO_ACCFORMAT_FORCED10_NO_PREFILL_LABEL="${CONDITION_GDPO_ACCFORMAT_FORCED10_NO_PREFILL_LABEL:-gdpo_fresh_lr1e5_noexplore_accformat_forced10_no_final_prefill}"
CONDITION_GDPO_ACCFORMAT_FORCED10_NO_PREFILL_DIR="${CONDITION_GDPO_ACCFORMAT_FORCED10_NO_PREFILL_DIR:-$VERO_EVAL_DIR/../analysis/full_forced_runs_judge_recovery_20261011/job15333548}"
CONDITION_GDPO_ACCFORMAT_ANSWERRETRY_LABEL="${CONDITION_GDPO_ACCFORMAT_ANSWERRETRY_LABEL:-gdpo_fresh_lr1e5_noexplore_accformat_always_retry_answers_no_critic_final_prefill}"
CONDITION_GDPO_ACCFORMAT_ANSWERRETRY_DIR="${CONDITION_GDPO_ACCFORMAT_ANSWERRETRY_DIR:-$VERO_EVAL_DIR/../analysis/full_forced_runs_judge_recovery_20261011/job15333549}"
CONDITION_B_LABEL="${CONDITION_B_LABEL:-hardonly_llmjudge_lr5e6}"
CONDITION_B_DIR="${CONDITION_B_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_hardonly_llmjudge_lr5e6_visual_probe}"
CONDITION_N_LABEL="${CONDITION_N_LABEL:-branchstrata20_fresh_lr1e5}"
CONDITION_N_DIR="${CONDITION_N_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_lr1e5_visual_probe}"
CONDITION_RESTART80_LABEL="${CONDITION_RESTART80_LABEL:-branchstrata20_fresh_lr1e5_restart_step80}"
CONDITION_RESTART80_DIR="${CONDITION_RESTART80_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_lr1e5_restart_step80_visual_probe}"
CONDITION_O_LABEL="${CONDITION_O_LABEL:-branchstrata30_fresh}"
CONDITION_O_DIR="${CONDITION_O_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata30_fresh_visual_probe}"
CONDITION_P_LABEL="${CONDITION_P_LABEL:-branchstrata10_fresh}"
CONDITION_P_DIR="${CONDITION_P_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata10_fresh_visual_probe}"
CONDITION_R_LABEL="${CONDITION_R_LABEL:-branchstrata20_fresh_lr1e5_critic}"
CONDITION_R_DIR="${CONDITION_R_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_lr1e5_critic_visual_probe_critic_retry_thr0p5}"
CONDITION_CRITIC_MAXCALLS5_LABEL="${CONDITION_CRITIC_MAXCALLS5_LABEL:-branchstrata20_fresh_lr1e5_critic_maxcalls5}"
CONDITION_CRITIC_MAXCALLS5_DIR="${CONDITION_CRITIC_MAXCALLS5_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_lr1e5_critic_maxcalls5_visual_probe_critic_retry_thr0p5}"
CONDITION_CRITIC_MAXCALLS20_LABEL="${CONDITION_CRITIC_MAXCALLS20_LABEL:-branchstrata20_fresh_lr1e5_critic_maxcalls20}"
CONDITION_CRITIC_MAXCALLS20_DIR="${CONDITION_CRITIC_MAXCALLS20_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_lr1e5_critic_maxcalls20_visual_probe_critic_retry_thr0p5}"
CONDITION_CANDIDATE_LABEL="${CONDITION_CANDIDATE_LABEL:-branchstrata20_fresh_lr1e5_critic_candidate_schedule_v1}"
CONDITION_CANDIDATE_DIR="${CONDITION_CANDIDATE_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_lr1e5_critic_candidate_schedule_v1_visual_probe_critic_retry_thr0p5}"
CONDITION_CRITIC_REPLAY_LABEL="${CONDITION_CRITIC_REPLAY_LABEL:-branchstrata20_fresh_lr1e5_critic_replay_schedule_v1}"
CONDITION_CRITIC_REPLAY_DIR="${CONDITION_CRITIC_REPLAY_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_branchstrata20_fresh_lr1e5_critic_replay_schedule_v1_visual_probe_reasoning_critic}"
CONDITION_T_LABEL="${CONDITION_T_LABEL:-hardonly_llmjudge_lr5e6_budget_forcing_t10}"
CONDITION_T_DIR="${CONDITION_T_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_hardonly_llmjudge_lr5e6_visual_probe_budget_forcing_t10}"
QWEN_NORMAL_BASELINE_LABEL="${QWEN_NORMAL_BASELINE_LABEL:-qwen3vl_8b_instruct_normal}"
QWEN_NORMAL_BASELINE_DIR="${QWEN_NORMAL_BASELINE_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_qwen3vl_8b_instruct_visual_probe_normal}"
QWEN_NOTOOLS_BASELINE_LABEL="${QWEN_NOTOOLS_BASELINE_LABEL:-qwen3vl_8b_instruct_no_tools}"
QWEN_NOTOOLS_BASELINE_DIR="${QWEN_NOTOOLS_BASELINE_DIR:-$EVAL_OUTPUT_ROOT/base_qwen3vl8b_no_tools_visual_probe}"
VERO_NOTOOLS_BASELINE_LABEL="${VERO_NOTOOLS_BASELINE_LABEL:-vero_qwen3i_8b_no_tools}"
VERO_NOTOOLS_BASELINE_DIR="${VERO_NOTOOLS_BASELINE_DIR:-$EVAL_OUTPUT_ROOT/zoom_eval_vero_qwen3i_8b_visual_probe_normal}"

METRIC="${METRIC:-judge_accuracy}"
OUTPUT_DIR="${OUTPUT_DIR:-$VERO_EVAL_DIR/examples/plots}"
OUTPUT_BASENAME="${OUTPUT_BASENAME:-visual_probe_accuracy_over_steps_qwen35_27b_judge_selected9}"
TOOL_CALLS_OUTPUT_BASENAME="${TOOL_CALLS_OUTPUT_BASENAME:-visual_probe_avg_tool_calls_over_steps_qwen35_27b_judge_selected9}"
TITLE="${TITLE:-Visual Probe Accuracy over Training Steps (Qwen3.5-27B Judge)}"
TOOL_CALLS_TITLE="${TOOL_CALLS_TITLE:-Average Visual Probe Tool Calls over Training Steps (Qwen3.5-27B Judge)}"
PERCENT="${PERCENT:-1}"
LATEST_BY="${LATEST_BY:-name}"

mkdir -p "$OUTPUT_DIR"

conditions=(
  "$CONDITION_QWEN35_VERO2_LABEL=$CONDITION_QWEN35_VERO2_DIR"
  "$CONDITION_MINIO3_VERO2_LABEL=$CONDITION_MINIO3_VERO2_DIR"
  "$CONDITION_FIXJSON_VERO2_LABEL=$CONDITION_FIXJSON_VERO2_DIR"
  "$CONDITION_MINIO3_STEP200_LABEL=$CONDITION_MINIO3_STEP200_DIR"
  # "$CONDITION_R3L_LABEL=$CONDITION_R3L_DIR"
  # "$CONDITION_R3L_15071396_LABEL=$CONDITION_R3L_15071396_DIR"
  # "$CONDITION_R3L_15071396_PAPER_FIXES_LABEL=$CONDITION_R3L_15071396_PAPER_FIXES_DIR"
  "$CONDITION_GDPO_LABEL=$CONDITION_GDPO_DIR"
  "$CONDITION_GDPO_ACCFORMAT_LABEL=$CONDITION_GDPO_ACCFORMAT_DIR"
  "$CONDITION_GDPO_ACCFORMAT_BUDGET_LABEL=$CONDITION_GDPO_ACCFORMAT_BUDGET_DIR"
  "$CONDITION_GDPO_ACCFORMAT_FORCED10_NO_PREFILL_LABEL=$CONDITION_GDPO_ACCFORMAT_FORCED10_NO_PREFILL_DIR"
  "$CONDITION_GDPO_ACCFORMAT_ANSWERRETRY_LABEL=$CONDITION_GDPO_ACCFORMAT_ANSWERRETRY_DIR"
  # "$CONDITION_B_LABEL=$CONDITION_B_DIR"
  "$CONDITION_N_LABEL=$CONDITION_N_DIR"
  "$CONDITION_RESTART80_LABEL=$CONDITION_RESTART80_DIR"
  "$CONDITION_O_LABEL=$CONDITION_O_DIR"
  "$CONDITION_P_LABEL=$CONDITION_P_DIR"
  "$CONDITION_R_LABEL=$CONDITION_R_DIR"
  "$CONDITION_CRITIC_MAXCALLS5_LABEL=$CONDITION_CRITIC_MAXCALLS5_DIR"
  "$CONDITION_CRITIC_MAXCALLS20_LABEL=$CONDITION_CRITIC_MAXCALLS20_DIR"
  "$CONDITION_CANDIDATE_LABEL=$CONDITION_CANDIDATE_DIR"
  "$CONDITION_CRITIC_REPLAY_LABEL=$CONDITION_CRITIC_REPLAY_DIR"
  # "$CONDITION_T_LABEL=$CONDITION_T_DIR"
)

baselines=()
for baseline in \
  "$QWEN_NORMAL_BASELINE_LABEL=$QWEN_NORMAL_BASELINE_DIR" \
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
  # Local restart updates follow the source checkpoint at cumulative step 80.
  --restart-from "$CONDITION_RESTART80_LABEL=$CONDITION_N_LABEL:80"
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
