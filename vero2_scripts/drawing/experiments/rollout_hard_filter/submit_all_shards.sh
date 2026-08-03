#!/bin/bash
# Submit every shard of the val-only difficulty measurement as its own pli-lc job.
# Run from the REPO ROOT after make_shards.py:
#   bash vero2_scripts/drawing/experiments/rollout_hard_filter/submit_all_shards.sh
#   FORCE=1 bash .../submit_all_shards.sh          # redo shards whose dump already exists
#   SHARD_DIR=/abs/other/shards bash .../submit_all_shards.sh
set -euo pipefail

[ -f ./set_paths.sh ] || { echo "ERROR: run from the repo root" >&2; exit 2; }
source ./set_paths.sh >/dev/null 2>&1 || true
SAVE_PATH="${SAVE_PATH:-/scratch/gpfs/ZHUANGL/lc1556}"

SHARD_DIR="${SHARD_DIR:-/scratch/gpfs/ZHUANGL/lc1556/vero2/hard_filter_runs/shards}"
LAUNCHER="vero2_scripts/drawing/experiments/rollout_hard_filter/run_valonly_basefilter_counting_drawonly_full.sh"

shopt -s nullglob
SHARDS=("$SHARD_DIR"/shard_*.verl.jsonl)
[ "${#SHARDS[@]}" -gt 0 ] || { echo "ERROR: no shard_*.verl.jsonl in $SHARD_DIR (run make_shards.py)" >&2; exit 2; }

for shard_file in "${SHARDS[@]}"; do
  tag="$(basename "$shard_file" .verl.jsonl | tr -d '_')"     # shard_00 -> shard00
  dump="${SAVE_PATH}/checkpoints/vero2_verl/basefilter_counting_drawonly_${tag}/validation_data/0.jsonl"
  if [ -e "$dump" ] && [ "${FORCE:-0}" != "1" ]; then
    echo "skip $tag: dump exists ($dump); FORCE=1 to redo"
    continue
  fi
  echo "submit $tag <- $shard_file"
  sbatch --export=ALL,SHARD_FILE="$shard_file",SHARD_TAG="$tag",FORCE="${FORCE:-0}" "$LAUNCHER"
done
