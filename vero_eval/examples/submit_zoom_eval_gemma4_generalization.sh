#!/usr/bin/env bash
# Both Gemma 4 E4B modes on full Pinpoint (433) and VStar (191) by default.
# Add --include-hrbench-8k to also evaluate HRBench-8K (800).
# --preflight / --dry-run never submit; --modes thinking selects just that mode.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${PYTHON:-python3}" "$SCRIPT_DIR/submit_zoom_eval_gemma4_generalization.py" "$@"
