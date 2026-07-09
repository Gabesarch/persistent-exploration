#!/bin/bash
# SMOKE for the VANILLA drawing run (multi-turn draw tools, NO force-explore/answer-retry).
# Same config as the full run (Haoyang's tested countqa drawing config: batch 256, per-turn
# cap 4096) — this wrapper only shrinks batch/steps on the CLI, so the smoke exercises the
# exact settings the full run trains with. 4x A100-80GB gpu-test (<=1h).
#
# Data: the vanilla config trains on ${VERL_ROOT}/datasets/combined_counting_hq/ (Haoyang's
# hard-counting set, answers < 10). Materialize it once per checkout:
#   mkdir -p "$VERL_ROOT/datasets" && \
#   ln -sfn /scratch/gpfs/ZHUANGL/hw0171/vero2/datasets/combined_counting_hq "$VERL_ROOT/datasets/combined_counting_hq"
#
#   sbatch vero2_scripts/drawing/vanilla/run_gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_smoke.sh
#SBATCH --job-name=drawing_vanilla_smoke
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
source ./set_paths.sh                              # need SAVE_PATH + VERL_ROOT

# Fail fast with the fix if the dataset symlink is missing (see header).
if [ ! -e "${VERL_ROOT}/datasets/combined_counting_hq/train_hard_answer_lt10.verl.jsonl" ]; then
  echo "ERROR: dataset not found at \$VERL_ROOT/datasets/combined_counting_hq/." >&2
  echo "Fix:   mkdir -p \"$VERL_ROOT/datasets\" && ln -sfn /scratch/gpfs/ZHUANGL/hw0171/vero2/datasets/combined_counting_hq \"$VERL_ROOT/datasets/combined_counting_hq\"" >&2
  exit 3
fi

SMOKE_ID="${SLURM_JOB_ID:-pid_$$}"
export NUM_GPUS=4
export SKIP_SQUEUE=1
export WANDB_MODE=offline                          # smoke: offline, no proxy needed
export TRAINER_EXPERIMENT_NAME="drawing_vanilla_smoke_${SMOKE_ID}"
export TRAINER_DEFAULT_LOCAL_DIR="${SAVE_PATH}/checkpoints/vero2_verl/drawing_vanilla_smoke_${SMOKE_ID}"

# Vanilla base launcher (CONFIG_NAME default = gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_debug256_cpueff,
# which despite the historical "debug256" in its name is the FULL batch-256 config) + smoke overrides.
exec bash vero2_scripts/drawing/vanilla/run_gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_debug256_cpueff.sh vllm \
  data.train_batch_size=8 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.rollout.n=8 \
  trainer.total_training_steps=2 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.val_before_train=false
