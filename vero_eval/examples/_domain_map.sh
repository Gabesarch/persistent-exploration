#!/bin/bash
# Shared domain -> base-task-stem map + expander, sourced by eval_across_domains.sh
# and scripts/eval_checkpoints_array.sbatch. A "domain" is a named group of tasks;
# expand_domain appends the model suffix (reasoning / reasoning_samplingq3 / qwen3_zs
# / qwen3_thinking_zs / qwen25_zs / mimo_zs / gpt5nano_zs) to each base stem and drops
# any (stem_suffix) that doesn't exist on disk (e.g. charxiv has no _qwen3_zs variant).
#
# Requires $VERO_EVAL_DIR to be set (the package root) before sourcing.

declare -gA DOMAIN_TASK_BASES=(
  [chart_ocr]="chartqa chartqa_pro infovqa_val charxiv chartmuseum evochart"
  [stem]="mmmu_pro mathvision_test mathvista_testmini"
  [spatial_action]="cv_bench embspatial erqa game_qa_lite blink"
  [knowledge_rec]="realworldqa simplevqa_en fvqa mmvetv2_group_img"
  [grounding_counting_search]="countbenchqa countqa mme_realworld_lite vstar_bench aerialvg_bbox visual_probe_easy visual_probe_medium visual_probe_hard screenspot_point_in_box screenspotpro_point_in_box"
  # Humanoid 360° visual search (Hstar bench). Not in `all`; use --domain hstarbench.
  [hstarbench]="hstarbench_mixed hstarbench_hos hstarbench_hps"
  # RL train split — monitor in-distribution performance during/after RL training.
  [hstarbench_rl]="hstarbench_mixed_rl hstarbench_hos_rl hstarbench_hps_rl"
  # Training-faithful zoom eval: RL distribution + held-out benchmark test.
  [hstarbench_train_monitor]="hstarbench_mixed_rl hstarbench_mixed"
  [instruction_following]="mia_bench mm_mt_bench mmifeval"
  [ball_drop]="balldrop"
  [vlms_are_biased]="vlms_are_biased"
  # Visual-search-only group (NOT in `all`): for a checkpoint sweep focused on VisualProbe,
  # e.g. submit_eval_checkpoints.sh --domain visual_probe --model-suffix reasoning_zoom.
  [visual_probe]="visual_probe_easy visual_probe_medium visual_probe_hard"
)
# The canonical content domains (order matters for `all`).
VERO_DOMAINS=(chart_ocr stem spatial_action knowledge_rec grounding_counting_search instruction_following)

_VERO_TASKNAMES=""
_vero_load_tasknames() {
  [[ -n "$_VERO_TASKNAMES" ]] && return
  : "${VERO_EVAL_DIR:?_domain_map.sh needs VERO_EVAL_DIR set (source set_paths.sh first)}"
  # `command grep` bypasses any shell grep function/alias (e.g. code-search wrappers).
  _VERO_TASKNAMES="$(command grep -rhoE '^(task|group):[[:space:]]*["'\'']?[A-Za-z0-9_]+' \
      "${VERO_EVAL_DIR}/lmms_eval/tasks"/*/*.yaml 2>/dev/null \
      | sed -E 's/^(task|group):[[:space:]]*//; s/["'\'']//g' | sort -u)"
  [[ -z "$_VERO_TASKNAMES" ]] && { echo "ERROR: no task yamls found under $VERO_EVAL_DIR/lmms_eval/tasks" >&2; return 1; }
}
vero_task_exists() { _vero_load_tasknames; command grep -qxF "$1" <<< "$_VERO_TASKNAMES"; }

# expand_domain <domain> <suffix> -> echoes a comma-separated list of existing tasks.
expand_domain() {
  local domain="$1" suffix="$2" bases="${DOMAIN_TASK_BASES[$1]:-}" out=() b t
  if [[ -z "$bases" ]]; then
    echo "ERROR: unknown domain '$domain' (valid: ${!DOMAIN_TASK_BASES[*]})" >&2
    return 1
  fi
  for b in $bases; do
    t="${b}_${suffix}"
    if vero_task_exists "$t"; then out+=("$t"); else echo "WARN: skipping $t (no such task for suffix '$suffix')" >&2; fi
  done
  [[ ${#out[@]} -eq 0 ]] && { echo "ERROR: no tasks for domain '$domain' suffix '$suffix'" >&2; return 1; }
  ( IFS=,; echo "${out[*]}" )
}
