#!/bin/bash
# Phase-A verification smoke for the drawing-tool answer-retry (forced-prefill) run.
# 4x A100-80GB gpu-test (<=1h). Validates the whole loop on the debug256 config:
# data -> vLLM async multi-turn rollout with the COMBINED zoom+draw toolset
# (Haoyang's image_drawing_tools.* draw classes) -> reward -> conditional
# answer-retry rollback+forced <tool_call> -> GDPO advantage -> FSDP2 step.
#
# It just sets the smoke env + overrides and delegates to the full launcher
# (which is CONFIG_NAME-overridable). Debug traces + crops land in
# ${TRAINER_DEFAULT_LOCAL_DIR}/debug_logs. Submit from the repo root:
#   sbatch vero2_scripts/run_..._zoomdraw_..._answer_retry_forcedprefill_smoke.sh
#SBATCH --job-name=zoomdraw_arf_smoke
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

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 2   # full launcher sources ./set_paths.sh (relative)
source ./set_paths.sh                          # need SAVE_PATH for the isolated smoke dir

SMOKE_ID="${SLURM_JOB_ID:-pid_$$}"
export NUM_GPUS=4
export SKIP_SQUEUE=1
export VERO2_COMPREHENSIVE_DEBUG_LOGGING=1
export CONFIG_NAME="gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_zoomdraw_debug256_cpueff_answer_retry_forcedprefill"
# Isolate the smoke's checkpoints/debug_logs from any real experiment dir, and
# use a _smoke experiment_name (per TASK.md). Fresh dir => no accidental resume.
export TRAINER_EXPERIMENT_NAME="zoomdraw_arf_smoke_${SMOKE_ID}"
export TRAINER_DEFAULT_LOCAL_DIR="${SAVE_PATH}/checkpoints/vero2_verl/zoomdraw_arf_smoke_${SMOKE_ID}"

exec bash vero2_scripts/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_zoomdraw_cpueff_hardonly_answer_retry_forcedprefill_full.sh vllm \
  data.train_batch_size=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.rollout.n=8 \
  trainer.total_training_steps=2 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=false \
  trainer.comprehensive_debug_logging.log_every_n_steps=1
