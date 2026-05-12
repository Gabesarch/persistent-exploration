#!/bin/bash

set -euo pipefail

scripts=(
  # "examples/gaze_rl/run_gspo_chart_qwen35_onpolicy.sh"
  # "examples/gaze_rl/run_gspo_chart_qwen3vl_base.sh"
  # "examples/gaze_rl/run_gspo_chart_qwen3vl_sft.sh"
  # "examples/gaze_rl/run_gspo_chart_qwen3vl_onpolicy.sh"
  "examples/gaze_rl/run_gspo_chart_qwen35_base.sh"
  "examples/gaze_rl/run_gspo_chart_qwen35_sft.sh"
)

script_count="${#scripts[@]}"
if [ "$script_count" -eq 0 ]; then
  echo "error: no scripts configured" >&2
  exit 2
fi

if [ -z "${SLURM_ARRAY_TASK_ID:-}" ]; then
  echo "error: SLURM_ARRAY_TASK_ID is not set" >&2
  exit 2
fi

original_task_id="$SLURM_ARRAY_TASK_ID"
script_idx=$((original_task_id % script_count))
round=$((original_task_id / script_count + 1))

job_script="${scripts[$script_idx]}"

if [ ! -f "$job_script" ]; then
  echo "error: missing job script: $job_script" >&2
  exit 2
fi

export GAZE_ROUND="$round"
export ORIGINAL_META_ARRAY_TASK_ID="$original_task_id"

# Some downstream code may use the Slurm array task id as a 1-based run index.
export SLURM_ARRAY_TASK_ID="$round"

echo "meta task $ORIGINAL_META_ARRAY_TASK_ID -> round $GAZE_ROUND -> $job_script"
bash "$job_script"
