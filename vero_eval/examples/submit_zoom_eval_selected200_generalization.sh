#!/usr/bin/env bash
# Eleven Visual Probe conditions on Pinpoint (433) and VStar (191) by default.
# Add --include-hrbench-8k to also evaluate HRBench-8K (800).
# DRY_RUN=1 bash examples/submit_zoom_eval_selected200_generalization.sh
# bash examples/submit_zoom_eval_selected200_generalization.sh --preflight
# EVAL_NAME=branchstrata20_fresh_lr1e5_critic bash examples/submit_zoom_eval_selected200_generalization.sh
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/submit_zoom_eval_selected200_generalization.py" "$@"
