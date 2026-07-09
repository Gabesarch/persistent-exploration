#!/bin/bash
# ONE submit script for the force-explore drawing FULL run — the partition is a RUNTIME
# decision. Replaces the four per-partition wrapper files (_plilc / _pli / _ailab /
# _pli_resume): sbatch command-line flags override the base launcher's #SBATCH header,
# so per-cluster headers never needed to be separate files.
#
# Run from the REPO ROOT:
#   bash vero2_scripts/drawing/force_explore/submit.sh [plilc|pli|ailab] [hydra overrides...]
#
# Examples:
#   bash .../submit.sh                                  # pli-lc (primary target, default)
#   bash .../submit.sh pli                              # pli (vlmrl) — best allocation odds
#   bash .../submit.sh ailab                            # ailab H200 (guest jobs get reaped -> save_freq=1)
#   bash .../submit.sh plilc data.train_files=/abs/train.jsonl data.val_files=/abs/val.jsonl
#   CHECKPOINT_DATE_TIME=20260710_x bash .../submit.sh  # FRESH run (new checkpoint dir; required
#                                                       #   for a new experiment or it resumes the old one)
#   TRAINER_EXPERIMENT_SUFFIX=qwen3vl_visual_probe_drawonly_gdpo_hdpotool_hardonly_answer_retry_forcedprefill_plilc \
#     bash .../submit.sh pli                            # resume a plilc run's checkpoints ON pli
#                                                       #   (what the old _pli_resume wrapper did)
#   ROLLOUT_CACHING=1 bash .../submit.sh pli            # + inference-only rollout speedups
#                                                       #   (prefix cache, mm cache, bigger prefill batch;
#                                                       #   the old _pli wrapper had these baked in)
#
# What each target encodes (everything else comes from the base launcher's header:
# 1 node, 24 h wall, --array %1 self-resume chain, slurm/ log paths):
#   target | partition | account | gres        | cpus | mem   | save_freq
#   plilc  | pli-lc    | pli_xl  | gpu:h100:8  | 96   | 960G  | 5
#   pli    | pli       | vlmrl   | gpu:h100:8  | 96   | 960G  | 5
#   ailab  | ailab     | zhuangl | gpu:h200:8  | 64   | 1450G | 1   (reap-safe: ckpt every step)
# save_freq=5 not 20: at this step cost a 24h task can't reach step 20, so save_freq=20
# would make every resume restart from scratch.
set -euo pipefail

[ -f ./set_paths.sh ] || { echo "ERROR: run from the repo root (bash vero2_scripts/drawing/force_explore/submit.sh ...)" >&2; exit 2; }

TARGET="plilc"
if [ "$#" -gt 0 ] && [[ "$1" != *=* ]]; then TARGET="$1"; shift; fi

case "$TARGET" in
  plilc) SBFLAGS=(--partition=pli-lc --account=pli_xl --gres=gpu:h100:8 --cpus-per-task=96 --mem=960G --job-name=drawonly_plilc); SAVE_FREQ=5;;
  pli)   SBFLAGS=(--partition=pli    --account=vlmrl  --gres=gpu:h100:8 --cpus-per-task=96 --mem=960G --job-name=drawonly_pli);   SAVE_FREQ=5;;
  ailab) SBFLAGS=(--partition=ailab  --account=zhuangl --gres=gpu:h200:8 --cpus-per-task=64 --mem=1450G --job-name=drawonly_ailab); SAVE_FREQ=1;;
  *) echo "ERROR: unknown target '$TARGET' (plilc|pli|ailab)" >&2; exit 2;;
esac

# Target-tagged experiment suffix -> distinct checkpoint dir + W&B run per partition, so
# parallel submissions to different partitions never write the same directory. Override the
# env (see header) to resume one partition's checkpoints on another.
export TRAINER_EXPERIMENT_SUFFIX="${TRAINER_EXPERIMENT_SUFFIX:-qwen3vl_visual_probe_drawonly_gdpo_hdpotool_hardonly_answer_retry_forcedprefill_${TARGET}}"

# Optional inference-only rollout speedups (no effect on the training objective): prefix + mm
# caching so 11-turn drawing trajectories don't re-prefill the growing context / re-embed the
# annotated image every turn, plus a bigger prefill batch. Opt in with ROLLOUT_CACHING=1.
CACHING_ARGS=()
if [ "${ROLLOUT_CACHING:-0}" = "1" ]; then
  CACHING_ARGS=(
    actor_rollout_ref.rollout.enable_prefix_caching=true
    actor_rollout_ref.rollout.engine_kwargs.vllm.mm_processor_cache_gb=4
    actor_rollout_ref.rollout.max_num_batched_tokens=16384
  )
fi

BASE="vero2_scripts/drawing/force_explore/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_full.sh"
echo "submit.sh -> target=$TARGET  suffix=$TRAINER_EXPERIMENT_SUFFIX  save_freq=$SAVE_FREQ  caching=${ROLLOUT_CACHING:-0}"
exec sbatch "${SBFLAGS[@]}" "$BASE" \
  vllm trainer.save_freq="$SAVE_FREQ" trainer.max_actor_ckpt_to_keep=3 "${CACHING_ARGS[@]}" "$@"
