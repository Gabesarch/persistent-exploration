#!/bin/bash
# Parallel full DRAW-ONLY answer-retry run on the big 'pli' H100 partition
# (38 nodes, --account=vlmrl). Thin wrapper over the CONFIG_NAME-overridable full
# launcher: only the #SBATCH header (partition/account) and a partition-tagged
# experiment name differ, so this run's checkpoint dir CANNOT collide with the
# pli-lc (10684588) or ailab parallel jobs of the same config.
#   sbatch vero2_scripts/run_..._drawonly_..._answer_retry_forcedprefill_full_pli.sh
#SBATCH --job-name=drawonly_pli
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=960G
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
export TRAINER_EXPERIMENT_SUFFIX="qwen3vl_visual_probe_drawonly_gdpo_hdpotool_hardonly_answer_retry_forcedprefill_pli"

exec bash vero2_scripts/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_full.sh
