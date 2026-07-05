#!/bin/bash
# Parallel full DRAW-ONLY answer-retry run on the STABLE pli-lc H100 partition
# (9 nodes, --account=pli_xl, PreemptMode=OFF -- the primary training target per CLAUDE.md).
# Thin wrapper over the full launcher with a partition-tagged experiment name so its
# checkpoint dir can't collide with the pli parallel job of the same config.
#   sbatch vero2_scripts/run_..._drawonly_..._answer_retry_forcedprefill_full_plilc.sh
#SBATCH --job-name=drawonly_plilc
#SBATCH --partition=pli-lc
#SBATCH --account=pli_xl
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:h100:8
#SBATCH --cpus-per-task=96
#SBATCH --mem=960G
#SBATCH --time=72:00:00
#SBATCH --array=1-5%1
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log

set -x
set -o pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 2
mkdir -p slurm/output slurm/error
source ./set_paths.sh                                   # env + online W&B (set_paths.local.sh)

# Partition-tagged experiment name -> distinct checkpoint dir + distinct W&B run,
# so the pli / pli-lc parallel jobs never write the same directory.
export TRAINER_EXPERIMENT_SUFFIX="qwen3vl_visual_probe_drawonly_gdpo_hdpotool_hardonly_answer_retry_forcedprefill_plilc"

# save_freq=5 (not 20): at this step cost a 24h task can't reach step 20, so with save_freq=20
# every resume restarts from scratch. Checkpoint by ~step 5 instead; keep only 3 (disk is ~full).
exec bash vero2_scripts/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_full.sh \
  vllm trainer.save_freq=5 trainer.max_actor_ckpt_to_keep=3
