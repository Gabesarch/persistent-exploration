#!/bin/bash
#SBATCH --job-name=brstr20_replay_smoke4
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

# Keep imports/configs in the feature checkout even when submitted elsewhere.
FEATURE_REPO_ROOT="${VERO2_REPO_ROOT:-/home/gs1693/repo/vero2-critic-replay}"
if [ ! -f "$FEATURE_REPO_ROOT/set_paths.sh" ] || \
   [ ! -f "$FEATURE_REPO_ROOT/vero2_scripts/config/branchstrata20_fresh_lr1e5_critic_replay_schedule_v1.yaml" ]; then
  echo "invalid replay feature checkout: $FEATURE_REPO_ROOT" >&2
  exit 1
fi
FEATURE_REPO_ROOT="$(cd "$FEATURE_REPO_ROOT" && pwd)"
export REPO_ROOT="$FEATURE_REPO_ROOT" VERL_ROOT="$FEATURE_REPO_ROOT"
cd "$FEATURE_REPO_ROOT"
source "$FEATURE_REPO_ROOT/set_paths.sh"
export REPO_ROOT="$FEATURE_REPO_ROOT" VERL_ROOT="$FEATURE_REPO_ROOT"

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

CONFIG_NAME="branchstrata20_fresh_lr1e5_critic_replay_schedule_v1"

DEBUG_TRAIN_DATA="${REPLAY_SMOKE_TRAIN_FILE:-/scratch/gpfs/ZHUANGL/gs1693/vero2_datasets/train_rl_datasets/chart_search_counting_visual_probe_train_debug256_rollout_score_leq05.verl.jsonl}"
RUN_ID="${SLURM_JOB_ID:-pid_$$}"
SMOKE_RUN_ROOT="${REPLAY_SMOKE_RUN_ROOT:-$SAVE_PATH/replay_smoke}"
RUN_DIR="$SMOKE_RUN_ROOT/branchstrata20_fresh_lr1e5_critic_replay_schedule_v1_smoke4gpu_${RUN_ID}"
DEBUG_DIR="$RUN_DIR/debug"
CHECKPOINT_DIR="$RUN_DIR/checkpoint"
mkdir -p "$DEBUG_DIR"

COMMON_OVERRIDES=(
  "data.train_files=$DEBUG_TRAIN_DATA"
  "data.train_batch_size=20"
  "data.val_batch_size=20"
  "data.val_max_samples=20"
  "data.dataloader_num_workers=0"
  "data.filter_overlong_prompts_workers=8"
  "actor_rollout_ref.actor.ppo_mini_batch_size=20"
  "actor_rollout_ref.actor.fsdp_config.fsdp_size=4"
  "actor_rollout_ref.rollout.agent.num_workers=4"
  "actor_rollout_ref.rollout.max_num_seqs=192"
  "trainer.n_gpus_per_node=4"
  "trainer.logger=[console]"
  "trainer.experiment_name=branchstrata20_fresh_lr1e5_critic_replay_schedule_v1_smoke4gpu_${RUN_ID}"
  "trainer.default_local_dir=$CHECKPOINT_DIR"
  "trainer.save_freq=1"
  "trainer.test_freq=1"
  "trainer.esi_redundant_time=60"
  "trainer.comprehensive_debug_logging.enabled=true"
  "trainer.comprehensive_debug_logging.dir=$DEBUG_DIR"
  "trainer.comprehensive_debug_logging.log_every_n_steps=1"
  "trainer.comprehensive_debug_logging.save_cropped_images=false"
  "trainer.comprehensive_debug_logging.save_source_images=false"
)

# First phase warms the cache, samples replay, and saves a matching cache snapshot.
python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/vero2_scripts/config" \
  --config-name="$CONFIG_NAME" \
  "${COMMON_OVERRIDES[@]}" \
  "trainer.total_training_steps=3" \
  "trainer.resume_mode=disable" \
  2>&1 | tee "$RUN_DIR/train.log"

# A separate process must restore step 3 rather than consume a newer live cache.
python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/vero2_scripts/config" \
  --config-name="$CONFIG_NAME" \
  "${COMMON_OVERRIDES[@]}" \
  "trainer.total_training_steps=4" \
  "trainer.resume_mode=resume_path" \
  "trainer.resume_from_path=$CHECKPOINT_DIR/global_step_3" \
  2>&1 | tee "$RUN_DIR/resume.log"

python3 "$REPO_ROOT/vero2_scripts/validate_replay_critic_smoke.py" \
  --run-dir "$RUN_DIR"
