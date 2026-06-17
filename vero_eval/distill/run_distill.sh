#!/usr/bin/env bash
# Collect GPT-5.5 reasoning + tool-use distillation traces for tool-calling tasks.
# Runs the given TASKS sequentially (keeps total API concurrency under the
# AI-Sandbox ~200K tok/min org limit) through the Responses-API harness.
#
# The teacher is REMOTE (AI Sandbox / Portkey) — no GPU, no torch needed. This runs
# fine on a login node (network-bound) or a CPU node with internet. It uses an
# isolated `uv run --no-project` env so the repo's verl/vllm pyproject is ignored.
#
# Usage (from anywhere):
#   TASKS="vstar_bench_gpt55_zoomdraw" bash vero_eval/distill/run_distill.sh
#   TASKS="sketchvlm_ball_drop_gpt55_zoom sketchvlm_ball_drop_gpt55_draw" bash .../run_distill.sh
#   TASKS="vstar_bench_gpt55_zoomdraw" LIMIT=20 WORKERS=6 bash .../run_distill.sh
#
# Env knobs:
#   TASKS        (REQUIRED)               space-separated task-yaml stems (gpt55_* variants)
#   LIMIT        (full split = all docs)  cap #docs (debug)
#   WORKERS      (8)                      thread-pool size (mind the rate limit)
#   OUTPUT_ROOT  (auto: <base>/<task-stem>)  override the per-task output dir
#   RESUME       (1)                      skip doc_ids already in the trace jsonl
#   HF_HOME      (/scratch/gpfs/ZHUANGL/lc1556/hf_cache)
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"

TASKS="${TASKS:-vstar_bench_gpt55_zoomdraw}"
WORKERS="${WORKERS:-8}"
RESUME="${RESUME:-1}"
export HF_HOME="${HF_HOME:-/scratch/gpfs/ZHUANGL/lc1556/hf_cache}"

UV_DEPS=(--with portkey-ai --with pillow --with numpy --with pyyaml --with datasets
         --with mcp --with opencv-python-headless --with matplotlib)

resume_flag=""
[[ "$RESUME" == "1" ]] && resume_flag="--resume"
limit_flag=""
[[ -n "${LIMIT:-}" ]] && limit_flag="--limit ${LIMIT}"
out_flag=""
[[ -n "${OUTPUT_ROOT:-}" ]] && out_flag="--output-root ${OUTPUT_ROOT}"

for TASK in $TASKS; do
  echo "=================================================================="
  echo "[distill] task=$TASK workers=$WORKERS"
  echo "=================================================================="
  uv run --no-project "${UV_DEPS[@]}" python collect_traces.py \
    --task "$TASK" --workers "$WORKERS" \
    $resume_flag $limit_flag $out_flag
done

echo "[distill] all tasks done."
