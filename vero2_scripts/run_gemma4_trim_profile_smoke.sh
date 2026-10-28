#!/usr/bin/env bash
# Della's submission plugin selects gputest/gpu-test; do not hardcode partition.
#SBATCH --job-name=gemma4_trim_profile_smoke
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log
set -eo pipefail
REPO_ROOT="${SLURM_SUBMIT_DIR:-/home/gs1693/repo/vero2}"
cd "$REPO_ROOT"
CONFIG_NAME=gemma4_trim_profile_smoke
source "$REPO_ROOT/vero2_scripts/gemma4_training_runtime.sh"
python -m pytest -q -p no:cacheprovider \
  tests/experimental/agent_loop/test_gemma4_trimming_on_cpu.py \
  tests/experimental/agent_loop/test_gemma4_attention_on_cpu.py \
  tests/experimental/agent_loop/test_gemma4_chunked_policy_on_cpu.py \
  --junitxml="$GEMMA4_ARTIFACT_DIR/trimming.xml" 2>&1 | tee "$GEMMA4_ARTIFACT_DIR/trimming.log"
python "$REPO_ROOT/vero2_scripts/validate_gemma4_training.py" --fixtures --output "$GEMMA4_ARTIFACT_DIR"
python -m verl.trainer.main_ppo --config-path="$REPO_ROOT/vero2_scripts/config" --config-name="$CONFIG_NAME" "$@" 2>&1 | tee "$GEMMA4_ARTIFACT_DIR/training.log"
python "$REPO_ROOT/vero2_scripts/summarize_gemma4_actor_profile.py" "$GEMMA4_ARTIFACT_DIR"
python "$REPO_ROOT/vero2_scripts/validate_gemma4_training.py" --audit --output "$GEMMA4_ARTIFACT_DIR"
