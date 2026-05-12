#!/bin/bash

set -euo pipefail

EXPECTED_ROOT="/home/gs1693/repo/gaze-vlm/verl"
CURRENT_DIR="$(pwd)"

if [ "$CURRENT_DIR" != "$EXPECTED_ROOT" ]; then
  echo "error: run this script from $EXPECTED_ROOT" >&2
  exit 2
fi

job_scripts=(
  "examples/gaze_rl/run_gspo_chart_qwen3vl_base.sh"
  "examples/gaze_rl/run_gspo_chart_qwen3vl_sft.sh"
  "examples/gaze_rl/run_gspo_chart_qwen3vl_onpolicy.sh"
  "examples/gaze_rl/run_gspo_chart_qwen35_base.sh"
  "examples/gaze_rl/run_gspo_chart_qwen35_sft.sh"
  "examples/gaze_rl/run_gspo_chart_qwen35_onpolicy.sh"
)

for job_script in "${job_scripts[@]}"; do
  if [ ! -f "$job_script" ]; then
    echo "error: missing job script: $job_script" >&2
    exit 2
  fi
done

for job_script in "${job_scripts[@]}"; do
  echo "Submitting $job_script"
  sbatch "$job_script"
done
