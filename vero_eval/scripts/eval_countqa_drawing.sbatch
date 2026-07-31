#!/bin/bash
#SBATCH --job-name=cqa_draw_eval
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=0:59:00
#SBATCH --mem=160G
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=gpu80
#SBATCH --qos=gpu-test
#SBATCH --output=vero_eval/logs/cqa_draw_eval_%j.out
#SBATCH --error=vero_eval/logs/cqa_draw_eval_%j.err
#
# ============================================================================
# CountQA DRAWING eval — the drawing counterpart of scripts/eval_visual_probe_full.sbatch,
# same MODEL->task convention:
#
#   MODEL=vero    -> countqa_reasoning_drawing   (TRAINING-FAITHFUL: draw_points-only
#                    toolset, forced-think template from the checkpoint, sampling 0.7/1.0/-1,
#                    per-turn generation cap 4096)   [requires CHECKPOINT_PATH = merged HF dir]
#   MODEL=qwen3vl -> countqa_qwen3_drawing        (base Qwen3-VL-Instruct: instruct draw
#                    toolset, native generation_config sampling 0.7/0.8/20, cap 4096)
#
# Submit from the REPO ROOT:
#   sbatch --export=ALL,MODEL=vero,CHECKPOINT_PATH=/scratch/.../vero2_merged_hf/<RUN>_step_140 \
#          vero_eval/scripts/eval_countqa_drawing.sbatch
#   sbatch --export=ALL,MODEL=qwen3vl vero_eval/scripts/eval_countqa_drawing.sbatch
#
# (FSDP checkpoints must be merged to HF first: scripts/merge_ckpt_to_hf.sbatch.)
#
# Knobs: LIMIT (default 400; the first 400 of 1528 flattened QA — unset for the full set),
#        PORT (default AUTO = 9100 + jobid%800 — prevents the same-node vLLM port collision
#              that invalidated a past run), TASKS (override the MODEL->task mapping),
#        OUTPUT_ROOT (default vero_eval/results/cqa_drawing/<label>_<jobid>).
# Metric: exact_match (integer count) — no judge, 1 GPU, ~35 min at LIMIT=400.
# ============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f vero_eval/set_paths.sh.example ]] || { echo "ERROR: submit from the repo root" >&2; exit 2; }

export MODEL="${MODEL:-qwen3vl}"
case "$MODEL" in
  vero)    export TASKS="${TASKS:-countqa_reasoning_drawing}"
           : "${CHECKPOINT_PATH:?MODEL=vero needs CHECKPOINT_PATH=<merged HF dir>}"
           LABEL="$(basename "$CHECKPOINT_PATH")";;
  qwen3vl) export TASKS="${TASKS:-countqa_qwen3_drawing}"
           LABEL="qwen3vl_base";;
  *) echo "ERROR: MODEL must be vero|qwen3vl (got '$MODEL')" >&2; exit 2;;
esac
export TOOLS="${TOOLS:-on}"
export LIMIT="${LIMIT:-400}"
export PORT="${PORT:-$(( 9100 + ${SLURM_JOB_ID:-0} % 800 ))}"
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/vero_eval/results/cqa_drawing/${LABEL}_${SLURM_JOB_ID:-manual}}"

echo "eval_countqa_drawing: MODEL=$MODEL TASKS=$TASKS LABEL=$LABEL PORT=$PORT LIMIT=$LIMIT"
exec bash vero_eval/run_eval.sbatch
