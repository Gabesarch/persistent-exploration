#!/usr/bin/env bash
#SBATCH --job-name=qwen35_9b_accformat_test
# Della's submission plugin assigns gputest from this one-hour GPU request.
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=24
#SBATCH --mem=400G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log

set -eo pipefail
REPO_ROOT="${VERO2_REPO_ROOT:-/home/gs1693/repo/vero2}"
RUN_ID="${SLURM_JOB_ID:-pid_$$}"
RUN_DIR="$REPO_ROOT/logs/qwen35_9b_accformat_gputest_$RUN_ID"
mkdir -p "$RUN_DIR"
cd "$REPO_ROOT"
# Same training data, reward, eight samples, tool limits, and judge as the full run.
# Bound batch/response sizes and run one optimizer step without validation/checkpoints.
bash "$REPO_ROOT/vero2_scripts/run_gdpo_fresh_lr1e5_noexplore_accformat_qwen35_9b.sh" \
  data.train_batch_size=6 \
  data.val_batch_size=6 \
  data.max_response_length=4096 \
  data.dataloader_num_workers=0 \
  data.filter_overlong_prompts_workers=4 \
  actor_rollout_ref.actor.ppo_mini_batch_size=6 \
  actor_rollout_ref.actor.fsdp_config.fsdp_size=4 \
  actor_rollout_ref.rollout.agent.num_workers=4 \
  actor_rollout_ref.rollout.max_model_len=20480 \
  actor_rollout_ref.rollout.max_num_seqs=16 \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=20480 \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu=20480 \
  trainer.n_gpus_per_node=4 \
  'trainer.logger=[console]' \
  "trainer.experiment_name=qwen35_9b_accformat_gputest_$RUN_ID" \
  "trainer.default_local_dir=$RUN_DIR/checkpoints" \
  "trainer.comprehensive_debug_logging.dir=$RUN_DIR/debug" \
  trainer.comprehensive_debug_logging.log_every_n_steps=1 \
  trainer.total_training_steps=1 \
  trainer.save_freq=-1 \
  trainer.test_freq=-1 \
  trainer.resume_mode=disable \
  trainer.esi_redundant_time=0 \
  "$@" 2>&1 | tee "$RUN_DIR/train.log"
