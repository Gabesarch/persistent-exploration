#!/bin/bash
#SBATCH --job-name=merge_hf
#SBATCH --qos=gpu-test
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=0:30:00
#SBATCH --output=vero_eval/logs/merge_hf_%j.out
#SBATCH --error=vero_eval/logs/merge_hf_%j.err
#
# Merge a verl FSDP checkpoint -> a HF dir servable by vLLM (~3 min on 1 A100).
# Submit from the REPO ROOT (uses the TRAINING env via ./set_paths.sh, not the eval env).
#
#   sbatch --export=ALL,CKPT_ROOT=/scratch/.../checkpoints/vero2_verl/<RUN>,STEP=140 \
#          vero_eval/scripts/merge_ckpt_to_hf.sbatch
#
# Knobs: CKPT_ROOT (run dir containing global_step_*), STEP,
#        OUT (default $ROOT_PATH/vero2_merged_hf/<run-basename>_step_<STEP>)
# Only steps whose actor/ still has model_world_size_*.pt shards are mergeable
# (max_actor_ckpt_to_keep prunes older ones to optimizer-only).
set -x
: "${CKPT_ROOT:?set CKPT_ROOT=<run checkpoint dir>}"
: "${STEP:?set STEP=<global step>}"
cd "${SLURM_SUBMIT_DIR:-$(pwd)}" || exit 2
module load proxy/default 2>/dev/null || true
source ./set_paths.sh

ACTOR="$CKPT_ROOT/global_step_${STEP}/actor"
[ -d "$ACTOR" ] || { echo "ERROR: $ACTOR not found" >&2; exit 3; }
ls "$ACTOR"/model_world_size_*.pt >/dev/null 2>&1 || { echo "ERROR: step $STEP has no model shards (pruned to optimizer-only)" >&2; exit 4; }
OUT="${OUT:-$ROOT_PATH/vero2_merged_hf/$(basename "$CKPT_ROOT")_step_${STEP}}"

python -m verl.model_merger merge --backend fsdp --local_dir "$ACTOR" --target_dir "$OUT"
# model_merger usually writes chat_template.jinja for VL models; guarantee it (vLLM 'vero' serving needs it)
[ -f "$OUT/chat_template.jinja" ] || cp "$ACTOR/huggingface/chat_template.jinja" "$OUT/"
ls -la "$OUT" | grep -E 'safetensors|jinja'
echo "=== MERGE DONE: $OUT ==="
