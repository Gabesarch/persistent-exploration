#!/bin/bash
# ---------------------------------------------------------------------------
# Counting-reward ablation launcher (PREPARED — DO NOT LAUNCH WITHOUT CHRIS).
#
# One launcher, one arm per invocation, selected by the ARM env var. Each arm =
#   (1) a counting-reward knob override (Hydra; see counting_reward_formula.md), and
#   (2) the counting training data mixed in (the standard jsonls have ZERO counting
#       rows — the configured chart_search_counting_* train file is 100% `search`).
#
# The reward knobs default to the shipped module constants, so the `graded` arm
# (no reward override) is bit-identical to current training.
#
# Usage (from THIS worktree root, after `source ./set_paths.sh` — verify $VERL_ROOT):
#   ARM=graded sbatch vero2_scripts/run_counting_reward_ablation.sh
#   ARM=exact  sbatch vero2_scripts/run_counting_reward_ablation.sh
#   ARM=wide_sigma sbatch vero2_scripts/run_counting_reward_ablation.sh
# Smoke first on gpu-test (<1h) — flip to the commented gpu-test header below.
# ---------------------------------------------------------------------------
#SBATCH --job-name=count_ablate
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=968G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log

## --- gpu-test smoke header (A100, <1h, 2 GPUs) — uncomment for a smoke ---
##SBATCH --job-name=count_ablate
##SBATCH --nodes=1
##SBATCH --ntasks=1
##SBATCH --gres=gpu:2
##SBATCH --qos=gpu-test
##SBATCH --constraint=gpu80
##SBATCH --cpus-per-task=24
##SBATCH --mem=800G
##SBATCH --time=00:59:00
##SBATCH --output=slurm/output/%x_%j.log
##SBATCH --error=slurm/error/%x_%j.log

set -x
set -o pipefail

SET_PATHS_SH="./set_paths.sh"
if [ ! -f "$SET_PATHS_SH" ]; then
  echo "error: set_paths.sh not found at $SET_PATHS_SH (launch from the worktree root)" >&2
  exit 2
fi
source "$SET_PATHS_SH"

# --- Arm selection --------------------------------------------------------
ARM="${ARM:-graded}"
COUNTING_OVERRIDES=()
case "$ARM" in
  exact)       # pre-PR#40 baseline: exact integer match, no band, no decay.
               # rel_band_cap=0.0 hard-guards against a per-row extra_info tolerance
               # opening a band even at rel_tol=0 (min(tol, rel_band_cap*gt)=0).
    COUNTING_OVERRIDES=(
      "+reward.custom_reward_function.reward_kwargs.counting_rel_tol=0.0"
      "+reward.custom_reward_function.reward_kwargs.counting_rel_band_cap=0.0"
      "+reward.custom_reward_function.reward_kwargs.counting_graded=false"
    ) ;;
  graded)      # shipped control: default knobs (bit-identical to current reward)
    COUNTING_OVERRIDES=() ;;
  band_only)   # keep the scale-aware band, drop the decay gradient
    COUNTING_OVERRIDES=(
      "+reward.custom_reward_function.reward_kwargs.counting_graded=false"
    ) ;;
  wide_sigma)  # wider Gaussian tail: densest gradient on near-misses (decisive treatment)
    COUNTING_OVERRIDES=(
      "+reward.custom_reward_function.reward_kwargs.counting_rel_sigma=0.30"
    ) ;;
  reltol0)     # hack-resistant treatment: delete the full-credit plateau (only exact -> 1.0),
               # keep the decay gradient. Alternate/swap-in if wide_sigma looks too lenient.
    COUNTING_OVERRIDES=(
      "+reward.custom_reward_function.reward_kwargs.counting_rel_tol=0.0"
    ) ;;
  laplacian)   # fatter tail (exp 1) instead of Gaussian (exp 2) — low priority
    COUNTING_OVERRIDES=(
      "+reward.custom_reward_function.reward_kwargs.counting_decay_exp=1.0"
    ) ;;
  *)
    echo "error: unknown ARM='$ARM' (expected: exact|graded|band_only|wide_sigma|reltol0|laplacian)" >&2
    exit 2 ;;
esac

# --- Counting data (mix into / replace the search-only default) -----------
# RLHFDataset accepts a list; these are the Gemini-filtered counting jsonls.
COUNTING_TRAIN="${COUNTING_TRAIN:-/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync/filtered/combined_counting_train.filtered.partial.verl.jsonl}"
COUNTING_VAL="${COUNTING_VAL:-/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync/filtered/combined_counting_val.filtered.partial.verl.jsonl}"
# Counting-only by default (cleanest for a reward ablation). To instead MIX with the
# existing blend, set DATA_MODE=mix and this appends the counting jsonls to the config's
# train_files/val_files. Chris to confirm which he wants.
DATA_MODE="${DATA_MODE:-counting_only}"
DATA_OVERRIDES=()
if [ "$DATA_MODE" = "counting_only" ]; then
  DATA_OVERRIDES=(
    "data.train_files=[${COUNTING_TRAIN}]"
    "data.val_files=[${COUNTING_VAL}]"
  )
else
  # mix: keep whatever the base config lists and add the counting jsonls
  BASE_TRAIN="/scratch/gpfs/ZHUANGL/gs1693/vero2_datasets/train_rl_datasets/chart_search_counting_visual_probe_train_rollout_score_leq05_all.verl.jsonl"
  BASE_VAL="/scratch/gpfs/ZHUANGL/gs1693/vero2_datasets/train_rl_datasets/chart_search_counting_visual_probe_val.verl.jsonl"
  DATA_OVERRIDES=(
    "data.train_files=[${BASE_TRAIN},${COUNTING_TRAIN}]"
    "data.val_files=[${BASE_VAL},${COUNTING_VAL}]"
  )
fi

# --- Naming / bookkeeping (distinct experiment_name per arm) ---------------
RUN_UNIQUE_ID="${RUN_UNIQUE_ID:-${SLURM_JOB_ID:-pid_$$}}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"
TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-vero2_verl}"
EXPERIMENT_BASE="${EXPERIMENT_BASE:-counting_reward_ablation}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-${EXPERIMENT_BASE}_${ARM}_${RUN_UNIQUE_ID}}"
TRAINER_DEFAULT_LOCAL_DIR="${TRAINER_DEFAULT_LOCAL_DIR:-${SAVE_PATH}/checkpoints/${TRAINER_PROJECT_NAME}/${TRAINER_EXPERIMENT_NAME}}"

mkdir -p slurm/output slurm/error logs
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export HF_DATASETS_OFFLINE="${HF_DATASETS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
if [ -z "${VERO2_CROP_CACHE_ROOT:-}" ]; then
  export VERO2_CROP_CACHE_ROOT="/tmp/${USER:-unknown}/vero2_crop_cache/${SLURM_JOB_ID:-pid_$$}"
fi
LOCAL_NO_PROXY="localhost,127.0.0.1,::1"
export NO_PROXY="${NO_PROXY:+${NO_PROXY},}${LOCAL_NO_PROXY}"
export no_proxy="${no_proxy:+${no_proxy},}${LOCAL_NO_PROXY}"

CONFIG_NAME="${CONFIG_NAME:-gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_cpueff_hardonly}"

echo "ARM=$ARM"
echo "CONFIG_NAME=$CONFIG_NAME"
echo "TRAINER_EXPERIMENT_NAME=$TRAINER_EXPERIMENT_NAME"
echo "DATA_MODE=$DATA_MODE"
printf 'COUNTING_OVERRIDES:'; printf ' %q' "${COUNTING_OVERRIDES[@]}"; printf '\n'
printf 'DATA_OVERRIDES:'; printf ' %q' "${DATA_OVERRIDES[@]}"; printf '\n'

python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/vero2_scripts/config" \
  --config-name="$CONFIG_NAME" \
  trainer.project_name="$TRAINER_PROJECT_NAME" \
  trainer.experiment_name="$TRAINER_EXPERIMENT_NAME" \
  trainer.default_local_dir="$TRAINER_DEFAULT_LOCAL_DIR" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$NUM_GPUS" \
  "${DATA_OVERRIDES[@]}" \
  "${COUNTING_OVERRIDES[@]}" \
  "$@"
