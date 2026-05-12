## Quick Start

### Prerequisites
- Linux with CUDA-capable GPUs (tested on Princeton Della cluster).
- Python 3.10 (3.9 works but 3.10 recommended).
- Git, Conda/Miniconda.

### Clone Repositories
```bash
git clone https://github.com/nb0564/backfill.git
```

### Create / Update Conda Environment
We use a single environment:
```bash
conda create -n gaze-vlm python=3.11 -y
conda create -p /scratch/gpfs/ZHUANGL/gs1693/conda/reason-vlm-gemma4-new python=3.11 -y

conda activate gaze-vlm
pip install uv
module load cudatoolkit/12.8
module load gcc/11
# uv pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu129
python -m pip install --no-cache-dir --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128

cd gaze-rl
pip install -e ".[verl]"

bash scripts/setup_qwen35.sh

uv pip install "transformers>=5.0.0" accelerate datasets peft safetensors "flash-linear-attention>=0.4.0" "qwen-vl-utils==0.0.14"


# pip install causal-conv1d --no-build-isolation --no-cache-dir
# pip install -U packaging psutil ninja
git clone https://github.com/Dao-AILab/flash-attention
cd flash-attention
python setup.py install

uv pip install pybase64 msgspec math_verify langdetect nltk latex2sympy2_extended 
uv pip install liger-kernel
uv pip install flash-linear-attention
```

### Setup

1. Copy set_paths.sh.example and rename it to set_paths.sh in the gaze-rl folder

2. Set all paths to data and caches in set_paths.sh and in run_scripts/model_runs/run_gspo_qwen3vl_instruct_condition_grounded_fixations_thinkaloud.sh

examples:
```bash
export REPO_ROOT="/home/gs1693/repo/gaze-vlm/gaze-rl" # path to this gaze-rl part
export ROOT_PATH="/scratch/gpfs/ZHUANGL/gs1693" # root path for cache and data (I have mine on scratch on della)
export SAVE_PATH="/scratch/gpfs/ZHUANGL/gs1693/vigorl_gaze_datasets" (save path for checkpoints)
export IMAGE_ROOT="/scratch/gpfs/ZHUANGL/gs1693/vigorl_gaze_datasets/cohen/raw" # root path for trial stimuli images (path to trail-stimuli images)
```

3. Cache the model to your HF directory:

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

4. (optional) Split train and val of jsonls Nick processed:

```bash
python run_scripts/data_prep/split_gaze_jsonl.py \
  --input-dir FOLDER_OF_ALL_JSONLS \
  --output-dir OUTPUT_PATH \
  --val-ratio 0.1 \
  --seed 42 \
  --overwrite
```

5. Set `TRAIN_FILES` and `VAL_FILES` in run_scripts/model_runs/run_gspo_qwen3vl_instruct_condition_grounded_fixations_thinkaloud.sh to the jsonl paths you want to train.

6. Run the script:

Make sure you are inside the gaze-rl folder

If in interactive node:
```bash
bash run_scripts/model_runs/run_gspo_qwen3vl_instruct_condition_grounded_fixations_thinkaloud.sh
```

To queue it up in slurm (adjust sbatch params as needed)
```bash
sbatch run_scripts/model_runs/run_gspo_qwen3vl_instruct_condition_grounded_fixations_thinkaloud.sh
```
Recommended atleast 8 H100s or 4 H200s (more the better). And atleast 700-900GB cpu memory.

