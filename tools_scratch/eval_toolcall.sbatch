#!/bin/bash
#SBATCH --job-name=3b_l_chart
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=10:00:00
#SBATCH --mem=80G
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --output=logs/eval_toolcall_%j.out
#SBATCH --error=logs/eval_toolcall_%j.err

# Create logs directory if it doesn't exist
mkdir -p /home/hw0171/reason-vlm-sft/tools_scratch/logs

# Activate conda environment
source ~/.bashrc
conda activate verovlm

# Set up SSH tunnel for proxy (connect to della9 for internet access)
echo "Setting up SSH tunnel for proxy..."
ssh -f -N -L 18888:localhost:18888 della9
sleep 2

# Set proxy environment variables
export HTTP_PROXY=http://localhost:18888
export HTTPS_PROXY=http://localhost:18888
export http_proxy=http://localhost:18888
export https_proxy=http://localhost:18888

# IMPORTANT: Exclude localhost from proxy (for local services like vLLM)
export no_proxy=localhost,127.0.0.1
export NO_PROXY=localhost,127.0.0.1

echo "Proxy configured: $HTTP_PROXY"

# Change to tools_scratch directory
cd /home/hw0171/reason-vlm-sft/tools_scratch

# Start vLLM servers (main model + judge)
echo "Starting vLLM servers..."
bash serve_qwen3vl.sh

# Check GPU usage
echo "=== GPU status after vLLM start ==="
nvidia-smi

# Change to eval directory and run evaluation
cd /home/hw0171/reason-vlm-sft/tools_scratch/eval

echo "Starting evaluation..."
bash examples/eval_domain.sh \
    --model-path Qwen3-VL-8B \
    --domain knowledge_rec \
    --variant toolcall

echo "Evaluation complete!"
