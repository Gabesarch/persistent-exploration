#!/usr/bin/env bash
# Setup Qwen 3.5 FSDP training with rllm
#
# Usage:
#   pip install -e ".[verl]"
#   bash scripts/setup_qwen35.sh
#
# Installs exact package versions tested with Qwen3.5-0.8B GRPO on B200 GPUs.
# Several packages have conflicting pins (vllm vs transformers, verl vs numpy)
# so they must be installed with --no-deps.
set -euo pipefail

echo "=== Installing Qwen 3.5 dependencies ==="

# Core packages (--no-deps to avoid conflicts between vllm<->transformers, verl<->numpy)
uv pip install --no-deps \
    "verl @ git+https://github.com/Zhang1Sheng/verl.git@main" \
    "vllm==0.18.0" \
    "transformers==5.3.0" \
    "huggingface-hub==1.8.0" \
    "numpy==2.2.6"

# Install vllm's remaining deps (excluding the conflicting ones we already installed)
uv pip install --no-deps \
    "torchvision==0.25.0" \
    "torchaudio==2.10.0"

uv pip install \
  anthropic \
  blake3 \
  cachetools \
  cbor2 \
  "compressed-tensors==0.13.0" \
  "depyf==0.20.0" \
  "diskcache==5.6.3" \
  "fastapi[standard]>=0.115.0" \
  "flashinfer-python==0.6.6" \
  "gguf>=0.17.0" \
  ijson \
  "lark==1.2.2" \
  "llguidance>=1.3.0,<1.4.0" \
  "lm-format-enforcer==0.11.3" \
  mcp \
  "mistral-common[image]>=1.10.0" \
  "model-hosting-container-standards>=0.1.13,<1.0.0" \
  ninja \
  "numba==0.61.2" \
  "nvidia-cudnn-frontend>=1.13.0,<1.19.0" \
  "nvidia-cutlass-dsl>=4.4.0.dev1" \
  "openai>=1.99.1,<2.25.0" \
  "openai-harmony>=0.0.3" \
  "opencv-python-headless>=4.13.0" \
  "opentelemetry-exporter-otlp>=1.27.0" \
  "opentelemetry-semantic-conventions-ai>=0.4.1" \
  "outlines-core==0.2.11" \
  partial-json-parser \
  "prometheus-fastapi-instrumentator>=7.0.0" \
  py-cpuinfo \
  pybase64 \
  python-json-logger \
  "pyzmq>=25.0.0" \
  "quack-kernels>=0.2.7" \
  sentencepiece \
  setproctitle \
  "tiktoken>=0.6.0" \
  watchfiles \
  "xgrammar>=0.1.32,<1.0.0"


echo "=== Verifying ==="
python3 -c "
import verl; print(f'verl: {verl.__version__}')
import vllm; print(f'vllm: {vllm.__version__}')
import torch; print(f'torch: {torch.__version__}')
import transformers; print(f'transformers: {transformers.__version__}')
from transformers import AutoConfig
cfg = AutoConfig.from_pretrained('Qwen/Qwen3.5-0.8B', trust_remote_code=True)
print(f'Qwen3.5 model_type: {cfg.model_type}')
print('Setup complete!')
"