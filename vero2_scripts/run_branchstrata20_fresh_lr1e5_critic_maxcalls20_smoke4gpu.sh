#!/bin/bash
#SBATCH --job-name=brstr20_critic_tc20_smoke4
#SBATCH --partition=gpu-test
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=1:00:00
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log

set -e
set -o pipefail
set -x

REPO_ROOT="${VERO2_REPO_ROOT:-${SLURM_SUBMIT_DIR:-/home/gs1693/repo/vero2}}"
if [ ! -f "$REPO_ROOT/set_paths.sh" ]; then
  REPO_ROOT="/home/gs1693/repo/vero2"
fi
cd "$REPO_ROOT"
source "$REPO_ROOT/set_paths.sh"

mkdir -p slurm/output slurm/error logs
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export HF_HUB_DISABLE_TELEMETRY="${HF_HUB_DISABLE_TELEMETRY:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [ -z "${VERO2_CROP_CACHE_ROOT:-}" ]; then
  CROP_CACHE_RUN_ID="${SLURM_JOB_ID:-pid_$$}"
  export VERO2_CROP_CACHE_ROOT="/tmp/${USER:-unknown}/vero2_crop_cache/${CROP_CACHE_RUN_ID}"
fi

LOCAL_NO_PROXY="localhost,127.0.0.1,::1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"
export NCCL_DEBUG="${NCCL_DEBUG:-INFO}"
export NCCL_ASYNC_ERROR_HANDLING="${NCCL_ASYNC_ERROR_HANDLING:-1}"
export TORCH_NCCL_ASYNC_ERROR_HANDLING="${TORCH_NCCL_ASYNC_ERROR_HANDLING:-1}"
export NCCL_P2P_DISABLE="${NCCL_P2P_DISABLE:-0}"
export NCCL_P2P_LEVEL="${NCCL_P2P_LEVEL:-NVL}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

# One training step plus validation; retains the 20-call cap and full token limits.
# This samples natural trajectories and does not force all 20 tool calls.
CONFIG_NAME="branchstrata20_fresh_lr1e5_critic_maxcalls20"
DEBUG_TRAIN_DATA="/scratch/gpfs/ZHUANGL/gs1693/vero2_datasets/train_rl_datasets/chart_search_counting_visual_probe_train_debug256_rollout_score_leq05.verl.jsonl"
DEBUG_VAL_DATA="/scratch/gpfs/ZHUANGL/gs1693/vero2_datasets/train_rl_datasets/chart_search_counting_visual_probe_train_debug256.verl.jsonl"
RUN_ID="${SLURM_JOB_ID:-pid_$$}"
RUN_DIR="$REPO_ROOT/logs/branchstrata20_fresh_lr1e5_critic_maxcalls20_smoke4gpu_${RUN_ID}"
DEBUG_DIR="$RUN_DIR/debug"
TRAIN_LOG="$RUN_DIR/train.log"
mkdir -p "$DEBUG_DIR"

python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/vero2_scripts/config" \
  --config-name="$CONFIG_NAME" \
  "data.train_files=$DEBUG_TRAIN_DATA" \
  "data.val_files=$DEBUG_VAL_DATA" \
  "data.train_batch_size=8" \
  "data.val_batch_size=8" \
  "data.dataloader_num_workers=0" \
  "data.filter_overlong_prompts_workers=8" \
  "actor_rollout_ref.actor.ppo_mini_batch_size=8" \
  "actor_rollout_ref.actor.fsdp_config.fsdp_size=4" \
  "actor_rollout_ref.rollout.agent.num_workers=4" \
  "actor_rollout_ref.rollout.max_num_seqs=128" \
  "trainer.n_gpus_per_node=4" \
  "trainer.logger=[console]" \
  "trainer.experiment_name=branchstrata20_fresh_lr1e5_critic_maxcalls20_smoke4gpu_${RUN_ID}" \
  "trainer.default_local_dir=$RUN_DIR/checkpoint" \
  "trainer.total_training_steps=1" \
  "trainer.save_freq=-1" \
  "trainer.test_freq=1" \
  "trainer.resume_mode=disable" \
  "trainer.comprehensive_debug_logging.enabled=true" \
  "trainer.comprehensive_debug_logging.dir=$DEBUG_DIR" \
  "trainer.comprehensive_debug_logging.log_every_n_steps=1" \
  "trainer.comprehensive_debug_logging.save_cropped_images=false" \
  "trainer.comprehensive_debug_logging.save_source_images=false" \
  2>&1 | tee "$TRAIN_LOG"

python3 "$REPO_ROOT/vero2_scripts/validate_retry_critic_smoke.py" \
  --traces "$DEBUG_DIR/sample_traces.jsonl" \
  --train-log "$TRAIN_LOG"
