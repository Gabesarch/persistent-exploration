#!/bin/bash
#SBATCH --job-name=crop_hard_llmj_lr5e6
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=968G
#SBATCH --time=24:00:00
#SBATCH --array=1-10%1
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log
#SBATCH --dependency=afterany:11140006_2

set -x
set -o pipefail

# Slurm executes a copied submission script from /var/spool/slurmd, so
# BASH_SOURCE does not point back to the repository there. Prefer the directory
# from which sbatch was invoked; retain sibling resolution for direct execution.
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  BASE_SCRIPT="$SLURM_SUBMIT_DIR/vero2_scripts/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_cropzoom_cpueff_hardonly_llmjudge.sh"
else
  SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
  BASE_SCRIPT="$SCRIPT_DIR/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_cropzoom_cpueff_hardonly_llmjudge.sh"
fi

if [ ! -x "$BASE_SCRIPT" ]; then
  echo "error: executable base launcher not found at $BASE_SCRIPT" >&2
  exit 2
fi

# Keep lr=5e-6 runs separate from checkpoints produced by other learning rates.
export TRAINER_EXPERIMENT_SUFFIX="${TRAINER_EXPERIMENT_SUFFIX:-chart_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_llmjudge_lr5e6}"

ENGINE="vllm"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]] && [[ "$1" != *=* ]]; then
  ENGINE="$1"
  shift
fi

# Append this override last so this launcher always uses the requested learning rate.
exec "$BASE_SCRIPT" "$ENGINE" "$@" actor_rollout_ref.actor.optim.lr=5e-6
