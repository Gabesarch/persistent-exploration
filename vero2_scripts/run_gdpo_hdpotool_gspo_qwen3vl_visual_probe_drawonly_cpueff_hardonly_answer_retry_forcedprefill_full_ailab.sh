#!/bin/bash
# Parallel full DRAW-ONLY answer-retry run on the ailab H200 partition (18 nodes).
# ailab uses the DEFAULT account -- OMIT --account (per useful_command.sh header C;
# pli_xl/vlmrl give "Invalid qos" here). Thin wrapper over the full launcher with a
# partition-tagged experiment name so its checkpoint dir can't collide with the
# pli / pli-lc parallel jobs of the same config.
#   sbatch vero2_scripts/run_..._drawonly_..._answer_retry_forcedprefill_full_ailab.sh
#SBATCH --job-name=drawonly_ailab
#SBATCH --partition=ailab
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h200:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=1450G
#SBATCH --time=24:00:00
#SBATCH --array=1-5%1
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log

set -x
set -o pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 2
mkdir -p slurm/output slurm/error
source ./set_paths.sh                                   # env + online W&B (set_paths.local.sh)

# Partition-tagged experiment name -> distinct checkpoint dir + distinct W&B run,
# so the pli / pli-lc / ailab parallel jobs never write the same directory.
export TRAINER_EXPERIMENT_SUFFIX="qwen3vl_visual_probe_drawonly_gdpo_hdpotool_hardonly_answer_retry_forcedprefill_ailab"

exec bash vero2_scripts/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_full.sh
