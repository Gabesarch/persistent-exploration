#!/bin/bash
# Speed/START smoke for the drawing run: uses the REAL full config (representative rollout --
# 24k response, 11 turns) + the new caching speedups, at a small batch, for 2 steps, on gpu-test.
# Goal: (1) confirm training STARTS cleanly with prefix+mm caching (de-risk the full run),
# (2) time one step. NOTE: 4x A100 + small batch, so extrapolate to full (128 batch / 8x H100).
#   sbatch vero2_scripts/drawing/experiments/run_drawonly_speed_bench_smoke.sh
#SBATCH --job-name=draw_bench
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
SMOKE_ID="${SLURM_JOB_ID:-pid_$$}"
export TRAINER_EXPERIMENT_NAME="draw_bench_${SMOKE_ID}"
export TRAINER_DEFAULT_LOCAL_DIR="${SAVE_PATH}/checkpoints/vero2_verl/draw_bench_${SMOKE_ID}"
export ANSWER_RETRY_ASSISTANT_PREFILL="Let's see, I do not have enough information to answer yet. My current annotations may be inaccurate or incomplete — let me use the drawing tools to correct and complete my markings, then re-examine the image before answering."

# Real full config (CONFIG_NAME unset -> the launcher default) + caching speedups, small batch, 2 steps.
exec bash vero2_scripts/drawing/force_explore/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_full.sh vllm \
  data.train_batch_size=16 \
  actor_rollout_ref.actor.ppo_mini_batch_size=16 \
  actor_rollout_ref.rollout.n=8 \
  trainer.total_training_steps=2 \
  trainer.save_freq=-1 trainer.test_freq=-1 trainer.val_before_train=false \
  actor_rollout_ref.rollout.enable_prefix_caching=true \
  actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=4 \
  actor_rollout_ref.rollout.max_num_batched_tokens=16384
