#!/usr/bin/env bash
# Submit one eval job per tool-calling task, each with its own PORT so several
# can share a node safely. Judge tasks (visual_probe, simplevqa, ...) get a 2nd
# GPU for the Qwen3-32B judge; exact-match tasks (screenspotpro, ...) get one.
#
# Usage (run from anywhere; the script cd's to the repo root):
#   bash vero_eval/examples/submit_tool_evals.sh                 # the 4 default tasks
#   TASKS="visual_probe_hard_qwen3_zoom screenspotpro_point_in_box_qwen3_draw" \
#     bash vero_eval/examples/submit_tool_evals.sh
#   DRY_RUN=1 bash vero_eval/examples/submit_tool_evals.sh       # print, don't submit
#
# Env knobs:
#   MODEL (qwen3vl) · TOOLS (on) · TASKS (4 zoom/draw tasks) · LIMIT (full split)
#   PARTITION (ailab) · ACCOUNT (zhuangl) · GPU_TYPE (h200) · BASE_PORT (9001)
#   TIME_JUDGE (04:00:00) · TIME_PLAIN (10:00:00) · DRY_RUN (0)
#   For the pli/H100 partition:  PARTITION=pli ACCOUNT=vlmrl GPU_TYPE=h100
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

MODEL="${MODEL:-qwen3vl}"
TOOLS="${TOOLS:-on}"
PARTITION="${PARTITION:-ailab}"
ACCOUNT="${ACCOUNT:-zhuangl}"
GPU_TYPE="${GPU_TYPE:-h200}"
BASE_PORT="${BASE_PORT:-9001}"
LIMIT="${LIMIT:-}"
DRY_RUN="${DRY_RUN:-0}"
TASKS="${TASKS:-visual_probe_hard_qwen3_zoom visual_probe_hard_qwen3_draw screenspotpro_point_in_box_qwen3_zoom screenspotpro_point_in_box_qwen3_draw}"

# Task name stems that trigger the LLM judge (=> need a 2nd GPU). Mirrors the
# _JUDGE_STEMS list in examples/eval_domain.sh.
JUDGE_STEMS="visual_probe simplevqa mathvista mathvision charxiv mmvetv2 mia_bench mm_mt_bench mmifeval fvqa chartmuseum"

i=0
for TASK in $TASKS; do
  PORT=$((BASE_PORT + i))
  JUDGE_PORT=$((8001 + i))
  GPUS=1
  TLIMIT="${TIME_PLAIN:-10:00:00}"
  for stem in $JUDGE_STEMS; do
    if [[ "$TASK" == ${stem}* ]]; then GPUS=2; TLIMIT="${TIME_JUDGE:-04:00:00}"; break; fi
  done
  OUT="$REPO_ROOT/vero_eval/results/eval_${TASK}"

  EXPORTS="ALL,MODEL=${MODEL},TOOLS=${TOOLS},TASKS=${TASK},PORT=${PORT},VLLM_JUDGE_PORT=${JUDGE_PORT},OUTPUT_ROOT=${OUT}"
  [[ -n "$LIMIT" ]] && EXPORTS="${EXPORTS},LIMIT=${LIMIT}"

  CMD=(sbatch -p "$PARTITION" --gres="gpu:${GPU_TYPE}:${GPUS}" --cpus-per-task="$((GPUS * 8))" --time="$TLIMIT")
  [[ -n "$ACCOUNT" ]] && CMD+=(--account="$ACCOUNT")
  CMD+=(--export="$EXPORTS" vero_eval/run_eval.sbatch)

  echo "[$TASK] gpus=${GPU_TYPE}:${GPUS} port=${PORT} judge_port=${JUDGE_PORT} -> ${OUT}"
  if [[ "$DRY_RUN" == "1" ]]; then
    printf '  '; printf '%q ' "${CMD[@]}"; printf '\n'
  else
    "${CMD[@]}"
  fi
  i=$((i + 1))
done

echo "Submitted ${i} job(s). Watch with: squeue -u \$USER"
