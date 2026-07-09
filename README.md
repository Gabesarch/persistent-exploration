# vero2

Multi-turn **tool-use RL for vision-language models** (Qwen3-VL-8B), built on a fork of
[verl](https://github.com/volcengine/verl). The model reasons in `<think>`, calls a visual tool —
**zoom** (`image_zoom_in_tool` crops a bbox and feeds the crop back) or **drawing**
(annotate the image with points/boxes/lines) — and answers in `<answer>`, trained with GDPO
advantage + GSPO loss on chart / search / counting data. Runs on Princeton Della (Slurm).

| Where | What |
|---|---|
| [`vero2_scripts/`](vero2_scripts/README.md) | **the experiments** — launchers (`drawing/`, `zoom/`), configs, prompts; start at its README |
| `verl/` | the RL engine (trainer, async tool rollout, tools, rewards) |
| [`vero_eval/`](vero_eval/README.md) | evaluation harness (lmms_eval + MCP tools) |
| `dataset_filter/` | Gemini ground-truth filter for datasets |
| [`docs/vero2/`](docs/vero2/) | detailed docs (training pipeline, environment, forced-explore) |

## Quick Start

### Prerequisites
- Linux with CUDA-capable GPUs (tested on Princeton Della cluster).
- Python 3.12.
- Git, Conda/Miniconda.

### Clone Repositories
```bash
git clone https://github.com/gabesarch/vero2
```

### Create / Update Conda Environment
We use a single conda environment (`vero2`):
```bash
conda create -n vero2 python=3.12 -y

conda activate vero2
pip install uv
module load cudatoolkit/12.9
module load gcc/11

python -m pip install --no-cache-dir --force-reinstall \
  torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
  --index-url https://download.pytorch.org/whl/cu129

uv pip install -e ".[verl]"

bash scripts/setup_qwen35.sh

# uv pip install "transformers>=5.0.0" accelerate datasets peft safetensors "flash-linear-attention>=0.4.0" "qwen-vl-utils==0.0.14"

uv pip install safetensors "qwen-vl-utils==0.0.14" "flash-linear-attention>=0.4.0"

# pip install causal-conv1d --no-build-isolation --no-cache-dir
python -m pip install -U packaging psutil ninja wheel
FLASH_ATTENTION_FORCE_BUILD=TRUE \
FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE \
MAX_JOBS=2 \
python -m pip install --no-build-isolation flash-attn==2.8.3

uv pip install pybase64 msgspec math_verify langdetect nltk latex2sympy2_extended 
uv pip install liger-kernel
uv pip install flash-linear-attention

# Install lmms-eval dependencies for evaluation.
# If a wheel download fails with IncompleteRead, rerun the same command with
# --no-cache-dir so pip does not reuse the partial download.
python -m pip install --no-cache-dir \
    sacrebleu zss zstandard transformers-stream-generator yt-dlp \
    timm tqdm-multiprocess evaluate sqlitedict tenacity sentence_transformers \
    pytablewriter Levenshtein matplotlib

# The bundled lmms_eval tree is not a standalone pip project. Run it from its
# parent directory, or add that directory to PYTHONPATH before invoking it.
export PYTHONPATH="$PWD/vero_eval:${PYTHONPATH:-}"
cd vero_eval
python -m lmms_eval --help
cd ..
```

### Setup

1. **Paths.** `set_paths.sh` (git-tracked) activates the conda env and exports `VERL_ROOT`,
   `ROOT_PATH`, `HF_HOME`, `SAVE_PATH`, `IMAGE_ROOT`, …. Per-user overrides go in the git-ignored
   `set_paths.local.sh` next to it (e.g. point `VERO2_CONDA_ENV` at your env). Every launcher
   sources it from the repo root — **always submit jobs from the repo root**.

2. Cache the model to your HF directory:

Set cache directory:
```bash
export HF_HOME="$ROOT_PATH/hf_cache"
export HF_DATASETS_CACHE="$HF_HOME/datasets"
export TRANSFORMERS_CACHE="$HF_HOME/transformers"
```

Then cache the model we will train since no internet on Della:

```python
import os
os.environ['HF_HOME'] = os.environ.get('HF_HOME')
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE')
os.environ['HF_HUB_CACHE'] = os.environ.get('HF_HUB_CACHE')

from transformers import AutoModelForVision2Seq, AutoProcessor, AutoTokenizer
import torch

model_id = "Qwen/Qwen3-VL-8B-Instruct"
print(f"Downloading model: {model_id}")
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id=model_id,
    local_dir=os.path.join(os.environ.get('HF_HOME'), model_id),
    local_dir_use_symlinks=False,     # copy real files; no links back into cache
    # revision="main",                # or pin a tag/commit for reproducibility
    ignore_patterns=["*.gguf","*.ggml"]
)

print(f"Successfully cached to {os.environ.get('HF_HOME')}/{model_id}")
```

3. **Train.** All launchers, configs, and instructions live in
   [`vero2_scripts/README.md`](vero2_scripts/README.md) — the folder map (`drawing/` vs `zoom/`,
   `vanilla/` vs `force_explore/`), how to change the training data, the config-knob glossary,
   and the smoke → full launch workflow. Typical flow:

```bash
# smoke on gpu-test first (small batch, 2 steps, <1 h):
sbatch vero2_scripts/drawing/force_explore/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_smoke.sh
# then the full run on pli-lc (batch 256, 24 h self-resuming array):
sbatch vero2_scripts/drawing/force_explore/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_full_plilc.sh
```

Full training needs a whole GPU node (8× H100 / A100-80GB) and ~700–900 GB CPU RAM.

4. **Evaluate.** See [`vero_eval/README.md`](vero_eval/README.md) (served-vLLM + lmms_eval + MCP visual tools).
