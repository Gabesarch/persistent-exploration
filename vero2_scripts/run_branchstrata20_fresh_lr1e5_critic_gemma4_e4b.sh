#!/usr/bin/env bash
#SBATCH --job-name=gemma4_critic
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --partition=pli-c
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=968G
#SBATCH --time=24:00:00
#SBATCH --array=1-10%1
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log
set -eo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/gs1693/repo/vero2}"
cd "$REPO_ROOT"
CONFIG_NAME=branchstrata20_fresh_lr1e5_critic_gemma4_e4b
source "$REPO_ROOT/vero2_scripts/gemma4_training_runtime.sh"
python -m verl.trainer.main_ppo --config-path="$REPO_ROOT/vero2_scripts/config" --config-name="$CONFIG_NAME" "$@" 2>&1 | tee "$GEMMA4_ARTIFACT_DIR/training.log"
