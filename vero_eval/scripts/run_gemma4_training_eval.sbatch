#!/usr/bin/env bash
set -euo pipefail
: "${VERO_EVAL_DIR:?}" "${GEMMA_TRAINING_CONFIG:?}" "${GEMMA_RL_ACTOR_DIR:?}" "${CHECKPOINT_PATH:?}"
# Never merge into a training checkpoint or a base-model directory. A completed
# marker is written only after the merger and the HF shard validator succeed.
mkdir -p "$CHECKPOINT_PATH"
exec 8>"$CHECKPOINT_PATH/merge.lock"
flock 8
if [[ -f "$CHECKPOINT_PATH/merge_complete.json" ]]; then
  python -c 'import json,os; from pathlib import Path; assert json.loads((Path(os.environ["CHECKPOINT_PATH"])/"merge_complete.json").read_text())["actor"] == os.environ["GEMMA_RL_ACTOR_DIR"]'
fi
if [[ ! -f "$CHECKPOINT_PATH/merge_complete.json" ]]; then
  GEMMA_MERGE_PYTHON=/scratch/gpfs/ZHUANGL/gs1693/conda/vero2_gemma4_train/bin/python
  GEMMA_MERGE_OVERLAY=/scratch/gpfs/ZHUANGL/gs1693/vero2/runtime/gemma4_train_transformers
  PYTHONPATH="$GEMMA_MERGE_OVERLAY:$VERO_EVAL_DIR/.." "$GEMMA_MERGE_PYTHON" -m verl.model_merger merge \
    --backend fsdp --local_dir "$GEMMA_RL_ACTOR_DIR" --target_dir "$CHECKPOINT_PATH"
  PYTHONPATH="$VERO_EVAL_DIR/scripts" "$GEMMA_MERGE_PYTHON" -c \
    'import os; from pathlib import Path; from cache_gemma_zoom_models import validate; validate(Path(os.environ["CHECKPOINT_PATH"]))'
  printf '{"actor":"%s"}\n' "$GEMMA_RL_ACTOR_DIR" > "$CHECKPOINT_PATH/merge_complete.json"
fi
flock -u 8
exec bash "$VERO_EVAL_DIR/scripts/run_gemma_zoom_eval.sbatch"
