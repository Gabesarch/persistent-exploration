#!/bin/bash
#SBATCH --job-name=hstarbench_sft
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=12:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=8
#SBATCH --partition=cpu
#SBATCH --output=vero_eval/logs/build_hstarbench_sft_%j.out
#SBATCH --error=vero_eval/logs/build_hstarbench_sft_%j.err
#
# Export hstarbench SFT split JSONL + strip/view assets (resumable).
# Submit from repo root:
#   sbatch vero_eval/scripts/build_hstarbench_sft.sbatch

set -euo pipefail

SUBMIT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
if [[ -f "$SUBMIT_DIR/set_paths.sh.example" && "$(basename "$SUBMIT_DIR")" == "vero_eval" ]]; then
  REPO_ROOT="$(dirname "$SUBMIT_DIR")"
else
  REPO_ROOT="$SUBMIT_DIR"
fi

cd "$REPO_ROOT"
mkdir -p vero_eval/logs

PYTHON="${PYTHON:-/scratch/gpfs/ZHUANGL/yz5121/miniconda3/envs/qwentool/bin/python}"
BUILD_SCRIPT="vero_eval/lmms_eval/tasks/hstarbench/build_hstarbench_jsonl.py"
BENCH_ROOT="${HSTARBENCH_ROOT:-/scratch/gpfs/ZHUANGL/shared/Hstarbench}"
LOG="${BENCH_ROOT}/data/build_sft_export.log"
DATA_DIR="${BENCH_ROOT}/data"

export HSTAR_VAGEN_ROOT="${HSTAR_VAGEN_ROOT:-/home/yz5121/vero/thinkin360}"
export HSTARBENCH_ROOT="$BENCH_ROOT"

hos_jsonl="${DATA_DIR}/hstarbench_hos_sft.jsonl"
hps_jsonl="${DATA_DIR}/hstarbench_hps_sft.jsonl"
mixed_jsonl="${DATA_DIR}/hstarbench_mixed_sft.jsonl"

echo "[$(date)] hos_sft export (auto-resume)" | tee -a "$LOG"
"$PYTHON" "$BUILD_SCRIPT" --split sft --only hos_sft --auto-resume --skip-existing 2>&1 | tee -a "$LOG"

echo "[$(date)] hps_sft export (auto-resume)" | tee -a "$LOG"
"$PYTHON" "$BUILD_SCRIPT" --split sft --only hps_sft --auto-resume --skip-existing 2>&1 | tee -a "$LOG"

echo "[$(date)] concat mixed_sft" | tee -a "$LOG"
"$PYTHON" -c "
from pathlib import Path
import sys
sys.path.insert(0, 'vero_eval/lmms_eval/tasks/hstarbench')
from build_hstarbench_jsonl import _concat_jsonl
n = _concat_jsonl([Path('$hos_jsonl'), Path('$hps_jsonl')], Path('$mixed_jsonl'))
print(f'Wrote {n} -> $mixed_jsonl')
" 2>&1 | tee -a "$LOG"

echo "[$(date)] done" | tee -a "$LOG"
