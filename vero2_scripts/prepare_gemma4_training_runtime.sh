#!/usr/bin/env bash
# Creates only the dedicated training environment and frozen Transformers copy.
set -euo pipefail
REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
GEMMA4_ROOT="${ROOT_PATH:-/scratch/gpfs/ZHUANGL/gs1693}"
GEMMA4_ENV="$GEMMA4_ROOT/conda/vero2_gemma4_train"
GEMMA4_OVERLAY="$GEMMA4_ROOT/vero2/runtime/gemma4_train_transformers"
if [[ ! -x "$GEMMA4_ENV/bin/python" ]]; then
  "$HOME/miniconda3/bin/conda" create --offline --yes --prefix "$GEMMA4_ENV" --clone "$GEMMA4_ROOT/conda/vero2_eval"
fi
"$GEMMA4_ENV/bin/python" -m pip install 'pytest==8.4.2' 'pytest-asyncio==1.2.0'
"$GEMMA4_ENV/bin/python" "$REPO_ROOT/vero_eval/scripts/prepare_gemma_zoom_runtime.py" \
  --source "$GEMMA4_ROOT/vero2/runtime/gemma_zoom_transformers" --output "$GEMMA4_OVERLAY"
