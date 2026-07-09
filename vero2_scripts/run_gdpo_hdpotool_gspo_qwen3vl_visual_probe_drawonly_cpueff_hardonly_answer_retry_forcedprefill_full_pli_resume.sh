#!/bin/bash
# BACKUP to un-stall the pli-lc run when pli-lc has no free node. This resumes the SAME
# checkpoint dir as the pli-lc job (job 10708304, currently stalled at step 80) by reusing the
# pli-lc experiment SUFFIX -> same TRAINER_DEFAULT_LOCAL_DIR -> resume_mode:auto continues from
# the latest checkpoint (step 80), NOT a fresh run. Submitted on the big 'pli' partition (38 H100
# nodes, --account=vlmrl) for far better allocation odds than pli-lc (9 nodes).
#
# CLOBBER SAFETY: this shares the pli-lc checkpoint dir, so it MUST NOT run at the same time as a
# pli-lc array task. Race rule: the instant EITHER this or the pli-lc array enters RUNNING, cancel
# the other (monitor does this). First checkpoint write is >= ~15 min in (well after vLLM init),
# so a <=2 min race-cancel window cannot produce a concurrent checkpoint write.
#   sbatch vero2_scripts/run_..._drawonly_..._forcedprefill_full_pli_resume.sh
#SBATCH --job-name=drawonly_pli_rsm
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
source ./set_paths.sh

# SAME suffix as the pli-lc wrapper -> SAME checkpoint dir -> resume from step 80 (not fresh).
export TRAINER_EXPERIMENT_SUFFIX="qwen3vl_visual_probe_drawonly_gdpo_hdpotool_hardonly_answer_retry_forcedprefill_plilc"
# Validated fast config (same as the pli-lc run this continues).
export CONFIG_NAME="gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill"

exec bash vero2_scripts/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_full.sh \
  vllm trainer.save_freq=5 trainer.max_actor_ckpt_to_keep=3
