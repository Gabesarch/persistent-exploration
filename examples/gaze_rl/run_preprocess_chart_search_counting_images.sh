#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VERL_ROOT="${VERL_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"

DEFAULT_DATA_DIR="/scratch/gpfs/ZHUANGL/gs1693/vigorl_gaze_datasets/chart_search_counting_rl"
TRAIN_JSONL="${TRAIN_JSONL:-$DEFAULT_DATA_DIR/chart_search_counting_train.verl.jsonl}"
VAL_JSONL="${VAL_JSONL:-$DEFAULT_DATA_DIR/chart_search_counting_val.verl.jsonl}"
MAX_PIXELS="${MAX_PIXELS:-10240000}"
IMAGE_PATCH_SIZE="${IMAGE_PATCH_SIZE:-14}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-}"
PNG_COMPRESS_LEVEL="${PNG_COMPRESS_LEVEL:-1}"
OUTPUT_DIR="${OUTPUT_DIR:-$DEFAULT_DATA_DIR/preprocessed_max_pixels_${MAX_PIXELS}}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OVERWRITE="${OVERWRITE:-0}"
SKIP_MISSING="${SKIP_MISSING:-0}"
FAST_RESUME="${FAST_RESUME:-0}"
LIMIT="${LIMIT:-}"

export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

args=(
  --train-jsonl "$TRAIN_JSONL"
  --val-jsonl "$VAL_JSONL"
  --output-dir "$OUTPUT_DIR"
  --max-pixels "$MAX_PIXELS"
  --image-patch-size "$IMAGE_PATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --png-compress-level "$PNG_COMPRESS_LEVEL"
)

if [[ "$OVERWRITE" == "1" || "$OVERWRITE" == "true" || "$OVERWRITE" == "TRUE" ]]; then
  args+=(--overwrite)
fi

if [[ "$SKIP_MISSING" == "1" || "$SKIP_MISSING" == "true" || "$SKIP_MISSING" == "TRUE" ]]; then
  args+=(--skip-missing)
fi

if [[ "$FAST_RESUME" == "1" || "$FAST_RESUME" == "true" || "$FAST_RESUME" == "TRUE" ]]; then
  args+=(--fast-resume)
fi

if [[ -n "$LIMIT" ]]; then
  args+=(--limit "$LIMIT")
fi

if [[ -n "$CHUNK_SIZE" ]]; then
  args+=(--chunk-size "$CHUNK_SIZE")
fi

train_base="$(basename "$TRAIN_JSONL")"
val_base="$(basename "$VAL_JSONL")"
train_out="$OUTPUT_DIR/${train_base%.jsonl}.preprocessed.jsonl"
val_out="$OUTPUT_DIR/${val_base%.jsonl}.preprocessed.jsonl"
image_dir="$OUTPUT_DIR/images"
manifest="$OUTPUT_DIR/manifest.jsonl"

echo "Preprocessing chart/search/counting images"
echo "  train input: $TRAIN_JSONL"
echo "  val input:   $VAL_JSONL"
echo "  output dir:  $OUTPUT_DIR"
echo "  max_pixels:  $MAX_PIXELS"
echo "  workers:     $NUM_WORKERS"
echo "  png level:   $PNG_COMPRESS_LEVEL"
if [[ -n "$CHUNK_SIZE" ]]; then
  echo "  chunk size:  $CHUNK_SIZE"
fi

"$PYTHON_BIN" "$SCRIPT_DIR/preprocess_chart_search_counting_images.py" "${args[@]}"

echo
echo "Processed train JSONL: $train_out"
echo "Processed val JSONL:   $val_out"
echo "Image directory:       $image_dir"
echo "Manifest:              $manifest"
