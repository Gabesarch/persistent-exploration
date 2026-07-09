#!/bin/bash
# Speed/START smoke for the FAST drawing config (friend's speed levers + our answer-retry).
# Validates the ported per-turn cap (max_assistant_response_length) + bounded retries START
# and times a step. gpu-test, 4x A100, small batch, 2 steps. Extrapolate to full (8x H100 / bs256).
#   sbatch vero2_scripts/run_drawonly_fast_bench_smoke.sh
#SBATCH --job-name=draw_fast
#SBATCH --qos=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=48
#SBATCH --mem=900G
#SBATCH --time=0:59:00
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log

set -x
set -o pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 2
source ./set_paths.sh

export NUM_GPUS=4
export SKIP_SQUEUE=1
export WANDB_MODE=offline
export VERO2_COMPREHENSIVE_DEBUG_LOGGING=1
SMOKE_ID="${SLURM_JOB_ID:-pid_$$}"
export CONFIG_NAME="gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_fast"
export TRAINER_EXPERIMENT_NAME="draw_fast_${SMOKE_ID}"
export TRAINER_DEFAULT_LOCAL_DIR="${SAVE_PATH}/checkpoints/vero2_verl/draw_fast_${SMOKE_ID}"
export ANSWER_RETRY_ASSISTANT_PREFILL="Let's see, I do not have enough information to answer yet. My current annotations may be inaccurate or incomplete — let me use the drawing tools to correct and complete my markings, then re-examine the image before answering."

# FAST config carries all the speed settings; just shrink batch + steps for the smoke.
exec bash vero2_scripts/counting/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_full.sh vllm \
  data.train_batch_size=16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.rollout.n=8 \
  trainer.total_training_steps=2 \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.val_before_train=false \
  trainer.comprehensive_debug_logging.log_every_n_steps=1
