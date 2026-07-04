#!/bin/bash
# Phase-A verification smoke for the DRAW-ONLY answer-retry (forced-prefill) run.
# Same as the zoomdraw smoke but the toolset is drawing-only (no image_zoom_in_tool):
# tool_config = image_drawing_tools.yaml, generic system prompt, and a drawing-worded
# retry prefill (no "zoom in" mention). 4x A100-80GB gpu-test (<=1h).
#
# Delegates to the CONFIG_NAME-overridable full launcher (a generic main_ppo runner).
# Debug traces + crops land in ${TRAINER_DEFAULT_LOCAL_DIR}/debug_logs.
#   sbatch vero2_scripts/run_..._drawonly_..._answer_retry_forcedprefill_smoke.sh
#SBATCH --job-name=drawonly_arf_smoke
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
source ./set_paths.sh                              # need SAVE_PATH for the isolated smoke dir

SMOKE_ID="${SLURM_JOB_ID:-pid_$$}"
export NUM_GPUS=4
export SKIP_SQUEUE=1
export VERO2_COMPREHENSIVE_DEBUG_LOGGING=1
export WANDB_MODE=offline                          # smoke: offline, no proxy needed
export CONFIG_NAME="gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_drawonly_debug256_cpueff_answer_retry_forcedprefill"
# Draw-only retry prefill: no "zoom in"; assumes the model may have ALREADY drawn some
# (possibly wrong/incomplete) annotations, and nudges re-examining + correcting/completing
# them. Kept general (not counting-specific) so it transfers across task types.
export ANSWER_RETRY_ASSISTANT_PREFILL="Let's see, I do not have enough information to answer yet. My current annotations may be inaccurate or incomplete — let me use the drawing tools to correct and complete my markings, then re-examine the image before answering."
export TRAINER_EXPERIMENT_NAME="drawonly_arf_smoke_${SMOKE_ID}"
export TRAINER_DEFAULT_LOCAL_DIR="${SAVE_PATH}/checkpoints/vero2_verl/drawonly_arf_smoke_${SMOKE_ID}"

exec bash vero2_scripts/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_full.sh vllm \
  data.train_batch_size=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.rollout.n=8 \
  trainer.total_training_steps=2 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.comprehensive_debug_logging.log_every_n_steps=1
