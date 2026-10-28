#!/usr/bin/env bash
set -euo pipefail
VERO_EVAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec "${GEMMA_SUBMIT_PYTHON:-python}" "$VERO_EVAL_DIR/examples/submit_zoom_eval_checkpoints_gemma4.py" "$@"
