#!/usr/bin/env bash
#SBATCH --job-name=gemma4_generation_diag
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --constraint=gpu80
#SBATCH --cpus-per-task=16
#SBATCH --mem=180G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log
set -euo pipefail
cd /home/gs1693/repo/vero2
export PYTHONPATH=/scratch/gpfs/ZHUANGL/gs1693/vero2/runtime/gemma4_train_transformers:$PWD
export OMP_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn NCCL_IB_DISABLE=1 NCCL_NVLS_ENABLE=0
export VLLM_USE_FLASHINFER_SAMPLER=0
export TMPDIR="$(mktemp -d /tmp/gemma_diag.XXXXXX)"
export VLLM_RPC_BASE_PATH="$TMPDIR"
diagnostic_python=/scratch/gpfs/ZHUANGL/gs1693/conda/vero2_gemma4_train/bin/python
diagnostic_root=/scratch/gpfs/ZHUANGL/gs1693/vero2/gemma4_training
diagnostic_output="$diagnostic_root/diagnostics/generation_${SLURM_JOB_ID}"
mkdir -p "$diagnostic_output"
export VLLM_CACHE_ROOT="$diagnostic_output/vllm_cache"
args=(--model /scratch/gpfs/ZHUANGL/gs1693/hf_cache/google/gemma-4-E4B-it
      --run "$diagnostic_root/checkpoints/branchstrata20_fresh_lr1e5_critic_gemma4_e4b_smoke2gpu_15806964"
      --output "$diagnostic_output")
cp vero2_scripts/diagnose_gemma4_generation.py verl/models/gemma4/sleep_diagnostics.py "$diagnostic_output/"
"$diagnostic_python" -B vero2_scripts/diagnose_gemma4_generation.py "${args[@]}" --audit-checkpoint 2>&1 | tee "$diagnostic_output/checkpoint_compare.log"
for mode in auto dummy; do
    "$diagnostic_python" -B vero2_scripts/diagnose_gemma4_generation.py "${args[@]}" --load-format "$mode" 2>&1 | tee "$diagnostic_output/generation_$mode.log"
done
