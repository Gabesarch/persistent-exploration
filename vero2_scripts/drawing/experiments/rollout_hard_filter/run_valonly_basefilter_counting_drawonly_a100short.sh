#!/bin/bash
# gpu-short A100 lane for the val-only difficulty shards: same body as the full launcher,
# but with the A100 header (QOS-selected, NO partition/account — Della's router rejects
# `-p gpu`; the A100 pool is reached via --qos on the default zhuangl association).
# 4x A100-80GB ~= the smoke-validated geometry; a ~1,950-row shard takes ~4h, 8h wall.
#
# Submit from the REPO ROOT:
#   sbatch --export=ALL,SHARD_FILE=/abs/shard_00.verl.jsonl,SHARD_TAG=shard00_a100 \
#     vero2_scripts/drawing/experiments/rollout_hard_filter/run_valonly_basefilter_counting_drawonly_a100short.sh
#SBATCH --job-name=basefilter_a100
#SBATCH --qos=gpu-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:a100:4
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=48
#SBATCH --mem=480G
#SBATCH --time=8:00:00
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log

set -x
set -o pipefail

cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 2
export NUM_GPUS=4

exec bash vero2_scripts/drawing/experiments/rollout_hard_filter/run_valonly_basefilter_counting_drawonly_full.sh
