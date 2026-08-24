#!/bin/bash
# gpu-test smoke for the val-only difficulty measurement: the stratified 64-row smoke shard
# x 8 rollouts = 512 drawing rollouts on 4x A100-80GB (<=1h). Validates the whole dump
# schema end-to-end (debug.uid / extra_info.id present, 8 rows per uid, no retry artifacts)
# before burning H100 time on the real shards.
#
# From the REPO ROOT (after make_shards.py has written the smoke shard):
#   sbatch vero2_scripts/drawing/experiments/rollout_hard_filter/run_valonly_basefilter_counting_drawonly_smoke.sh
#SBATCH --job-name=basefilter_smoke
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

SHARD_DIR="${SHARD_DIR:-/scratch/gpfs/ZHUANGL/lc1556/vero2/hard_filter_runs/shards}"
export SHARD_FILE="${SHARD_FILE:-${SHARD_DIR}/smoke_64.verl.jsonl}"
export SHARD_TAG="${SHARD_TAG:-smoke64}"
export NUM_GPUS=4
export WANDB_MODE=offline

# One 512-rollout val iteration; train dataloader must also fit the 64-row file.
exec bash vero2_scripts/drawing/experiments/rollout_hard_filter/run_valonly_basefilter_counting_drawonly_full.sh \
  data.val_batch_size=64 \
  data.train_batch_size=64
