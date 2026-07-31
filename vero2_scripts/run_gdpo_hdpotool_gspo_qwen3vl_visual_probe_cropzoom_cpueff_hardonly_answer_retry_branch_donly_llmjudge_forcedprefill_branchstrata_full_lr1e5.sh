#!/bin/bash
#SBATCH --job-name=crop_brstrata_lr1e5
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=968G
#SBATCH --time=24:00:00
#SBATCH --array=1-20%1
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log

set -x
set -o pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BASE_SCRIPT="$SCRIPT_DIR/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_cropzoom_cpueff_hardonly_answer_retry_branch_donly_llmjudge_forcedprefill_branchstrata_full.sh"

# Keep lr=1e-5 runs separate from checkpoints produced by the base run.
export TRAINER_EXPERIMENT_SUFFIX="${TRAINER_EXPERIMENT_SUFFIX:-qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_answer_retry_branch_donly_llmjudge_forcedprefill_branchstrata20_lr1e5}"

ENGINE="vllm"
if [ "$#" -gt 0 ] && [[ "$1" != -* ]] && [[ "$1" != *=* ]]; then
  ENGINE="$1"
  shift
fi

# Append this override last so this launcher always uses the requested learning rate.
exec "$BASE_SCRIPT" "$ENGINE" "$@" actor_rollout_ref.actor.optim.lr=1e-5
