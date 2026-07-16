#!/bin/bash
#SBATCH --job-name=crop_hard_llmj_lr1e5
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=968G
#SBATCH --time=24:00:00
#SBATCH --array=1-15%1
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log
#SBATCH --dependency=afterany:11140006_2

set -x
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_cropzoom_cpueff_hardonly_llmjudge.sh"
export CONFIG_NAME="gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_cpueff_hardonly_llmjudge_lr1e5"

# Keep lr=1e-5 runs separate from checkpoints produced by the base lr=1e-6 run.
export TRAINER_EXPERIMENT_SUFFIX="${TRAINER_EXPERIMENT_SUFFIX:-chart_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_llmjudge_lr1e5}"

ENGINE="vllm"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]] && [[ "$1" != *=* ]]; then
  ENGINE="$1"
  shift
fi

exec "$BASE_SCRIPT" "$ENGINE" "$@"
