#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

DEBUG_LOG_DIR="${DEBUG_LOG_DIR:-/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/verl_vero2/GSPO_fp16_math_verify_chart_qwen3vl_visual_probe_cropzoom_debug256_20260420_155518_cpueff/debug_logs}"
OUTPUT="${OUTPUT:-$DEBUG_LOG_DIR/trajectory_view.html}"
SELECT="${SELECT:-last}"
NUM_SAMPLES="${NUM_SAMPLES:-24}"

ARGS=(
  --debug-log-dir "$DEBUG_LOG_DIR"
  --output "$OUTPUT"
  --select "$SELECT"
  --num-samples "$NUM_SAMPLES"
)

if [ "${REQUEST_ID:-}" != "" ]; then
  ARGS+=(--request-id "$REQUEST_ID")
fi

if [ "${SAMPLE_INDEX:-}" != "" ]; then
  ARGS+=(--sample-index "$SAMPLE_INDEX")
fi

if [ "${ROLLOUT_N:-}" != "" ]; then
  ARGS+=(--rollout-n "$ROLLOUT_N")
fi

if [ "${MAX_TEXT_CHARS:-}" != "" ]; then
  ARGS+=(--max-text-chars "$MAX_TEXT_CHARS")
fi

if [ "${EMBED_IMAGES:-1}" = "1" ]; then
  ARGS+=(--embed-images)
fi

python3 "$SCRIPT_DIR/visualize_trajectory.py" "${ARGS[@]}"

echo
echo "Open this file in a browser:"
echo "$OUTPUT"
echo
echo "Examples:"
echo "  REQUEST_ID=1ef2c5da37f84e18a66663a31dad106c $0"
echo "  SAMPLE_INDEX=6517 ROLLOUT_N=0 $0"
echo "  NUM_SAMPLES=64 $0"
echo "  SELECT=most-tools NUM_SAMPLES=24 $0"
echo "  SELECT=random NUM_SAMPLES=24 $0"
echo "  SELECT=longest OUTPUT=/tmp/trajectory.html $0"
echo "  EMBED_IMAGES=0 $0  # smaller HTML that links to local image paths"
