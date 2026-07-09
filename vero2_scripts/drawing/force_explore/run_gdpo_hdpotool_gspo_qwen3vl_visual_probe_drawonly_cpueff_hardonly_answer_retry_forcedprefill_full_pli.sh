#!/bin/bash
# Parallel full DRAW-ONLY answer-retry run on the big 'pli' H100 partition
# (38 nodes, --account=vlmrl). Thin wrapper over the CONFIG_NAME-overridable full
# launcher: only the #SBATCH header (partition/account) and a partition-tagged
# experiment name differ, so this run's checkpoint dir CANNOT collide with the
# pli-lc (10684588) or ailab parallel jobs of the same config.
#   sbatch vero2_scripts/drawing/force_explore/run_..._drawonly_..._answer_retry_forcedprefill_full_pli.sh
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

# save_freq=5 (not 20): at this step cost a 24h task can't reach step 20, so with save_freq=20
# every resume restarts from scratch. Checkpoint by ~step 5 instead; keep only 3 (disk is ~full).
# Rollout throughput (inference-only, does NOT change the objective): prefix + mm caching so the
# 11-turn drawing trajectories don't re-prefill the growing context / re-embed the annotated image
# every turn, plus a bigger prefill batch. Biggest lever for multi-turn rollout speed.
exec bash vero2_scripts/drawing/force_explore/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_full.sh \
  vllm trainer.save_freq=5 trainer.max_actor_ckpt_to_keep=3 \
  actor_rollout_ref.rollout.enable_prefix_caching=true \
  actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384
