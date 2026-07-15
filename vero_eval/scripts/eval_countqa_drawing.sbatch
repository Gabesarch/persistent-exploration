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
# CountQA DRAWING eval — the training-faithful counting benchmark for the
# drawing (draw_points) experiments. Counterpart of scripts/eval_visual_probe_full.sbatch
# for the zoom side. Submit from the REPO ROOT.
#
#   # a trained checkpoint (merged HF dir — see scripts/merge_ckpt_to_hf.sbatch):
#   sbatch --export=ALL,CHECKPOINT_PATH=/scratch/.../vero2_merged_hf/<run>_step_140 \
#          vero_eval/scripts/eval_countqa_drawing.sbatch
#
#   # the BASE model, served under the same forced-think drawing template
#   # (fair base-vs-trained: same template, same task, same sampling — only weights differ):
#   sbatch --export=ALL,MODEL=base vero_eval/scripts/eval_countqa_drawing.sbatch
#
# Knobs (--export):
#   MODEL            base | vero (default vero; base = Qwen3-VL-8B-Instruct + committed template)
#   CHECKPOINT_PATH  merged HF dir (required unless MODEL=base)
#   TASKS            default countqa_drawing_vero_cap4k (per-turn gen cap 4096 — matches
#                    training's max_assistant_response_length AND avoids 32k-context overflow
#                    on CountQA's 10 MP images; do NOT use the uncapped variant for full runs)
#   LIMIT            default 400 (first 400 of 1528 flattened QA, deterministic; unset = full)
#   PORT             default AUTO from the job id (9100 + jobid%800) — prevents the
#                    same-node vLLM port collision that invalidated a past run
#   OUTPUT_ROOT      default vero_eval/results/cqa_drawing/<label>_<jobid>
#
# Metric: exact_match (integer count) — no judge needed, 1 GPU suffices.
# ~35 min at LIMIT=400 (vision prefill dominates). Results: <OUTPUT_ROOT>/**/*results*.json.
# ============================================================================
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"
[[ -f vero_eval/set_paths.sh.example ]] || { echo "ERROR: submit from the repo root" >&2; exit 2; }

export TASKS="${TASKS:-countqa_drawing_vero_cap4k}"
export TOOLS="${TOOLS:-on}"
export LIMIT="${LIMIT:-400}"
export PORT="${PORT:-$(( 9100 + ${SLURM_JOB_ID:-0} % 800 ))}"

if [[ "${MODEL:-vero}" == "base" ]]; then
  # Base model under the checkpoint-identical forced-think drawing template.
  # (The template is byte-stable across our drawing runs; committed copy below.)
  set +u; source vero_eval/set_paths.sh 2>/dev/null || source vero_eval/set_paths.sh.example; set -u
  export CHECKPOINT_PATH="${CHECKPOINT_PATH:-$HF_HOME/Qwen/Qwen3-VL-8B-Instruct}"
  export CHAT_TEMPLATE="${CHAT_TEMPLATE:-$PWD/vero_eval/chat_templates/vero_forcedthink_draw.jinja}"
  LABEL="base"
else
  : "${CHECKPOINT_PATH:?set CHECKPOINT_PATH=<merged HF dir> (or MODEL=base)}"
  LABEL="$(basename "$CHECKPOINT_PATH")"
fi
export MODEL=vero   # both sides use the vero serving path (forced-think template, no extra system prompt)
export OUTPUT_ROOT="${OUTPUT_ROOT:-$PWD/vero_eval/results/cqa_drawing/${LABEL}_${SLURM_JOB_ID:-manual}}"

echo "eval_countqa_drawing: LABEL=$LABEL PORT=$PORT LIMIT=$LIMIT TASKS=$TASKS"
echo "  CHECKPOINT_PATH=$CHECKPOINT_PATH"
exec bash vero_eval/run_eval.sbatch
