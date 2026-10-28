#!/usr/bin/env bash
# Three full-benchmark jobs: Qwen with/without zoom, and Vero without tools.
# DRY_RUN=1 bash examples/submit_zoom_eval_qwen35_9b_vero_visual_probe_pinpoint_vstar.sh
set -euo pipefail
exec python3 "$(dirname "$0")/submit_zoom_eval_qwen35_9b_vero_visual_probe_pinpoint_vstar.py" "$@"
