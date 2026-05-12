#!/bin/bash
#SBATCH --job-name=verl_gaze_chart_meta
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --array=0-119%6
#SBATCH --cpus-per-task=64
#SBATCH --mem=968G
#SBATCH --time=24:00:00
#SBATCH --output=examples/gaze_rl/slurm/output/%x_%A_%a.log
#SBATCH --error=examples/gaze_rl/slurm/error/%x_%A_%a.log

set -euo pipefail

cd /home/gs1693/repo/gaze-vlm/verl

dispatcher="./run_gaze_rl_chart_meta_dispatch.sh"

if [ ! -f "$dispatcher" ]; then
  echo "error: missing dispatcher: $dispatcher" >&2
  exit 2
fi

bash "$dispatcher"
