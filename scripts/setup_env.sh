#!/usr/bin/env bash
# Create the Python environment used by the public training and evaluation runners.
# Usage: bash scripts/setup_env.sh qwen|gemma [--check]
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
profile="${1:-}"
case "$profile" in
  qwen|gemma) shift ;;
  -h|--help|'')
    cat <<'EOF'
Usage: bash scripts/setup_env.sh qwen|gemma [--check]

Creates a Python 3.12 Conda environment and installs the runtime for the
training and evaluation commands in this repository. --check verifies an
existing environment without installing anything.

Environment overrides:
  CONDA_ENV_NAME    Environment name (default: persistent-exploration-PROFILE)
  CONDA_ENV_PATH    Use an environment at this path instead of a named one
  TORCH_INDEX_URL   PyTorch wheel index (default: https://download.pytorch.org/whl/cu129)
  SKIP_FLASH_ATTN   Set to 1 for Qwen evaluation only; Qwen training needs it
  MAX_JOBS          Parallel jobs when building flash-attn (default: 4)
EOF
    [[ "$profile" != '' ]] && exit 0 || exit 2 ;;
  *) echo "Unknown profile: $profile" >&2; exit 2 ;;
esac

check_only=0
if (($#)); then
  [[ "$1" == --check && $# -eq 1 ]] || { echo "Unknown argument: $*" >&2; exit 2; }
  check_only=1
fi

if [[ -n "${CONDA_ENV_PATH:-}" ]]; then
  env_target="$CONDA_ENV_PATH"
  env_flag=-p
else
  env_target="${CONDA_ENV_NAME:-persistent-exploration-$profile}"
  env_flag=-n
fi

command -v conda >/dev/null 2>&1 || { echo "Conda is required (Miniconda is sufficient)." >&2; exit 2; }
eval "$(conda shell.bash hook)"
if (( !check_only )); then
  if [[ "$env_flag" == -p ]]; then
    [[ -x "$env_target/bin/python" ]] || conda create -y -p "$env_target" python=3.12
  elif ! conda env list | awk 'NF && $1 !~ /^#/' | awk '{print $1}' | grep -Fxq "$env_target"; then
    conda create -y -n "$env_target" python=3.12
  fi
fi
conda activate "$env_target"
python_bin="$(command -v python)"
"$python_bin" -c 'import sys; assert sys.version_info[:2] == (3, 12), "Python 3.12 is required"'

if (( !check_only )); then
  torch_index="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu129}"
  "$python_bin" -m pip install --upgrade pip setuptools wheel
  "$python_bin" -m pip install \
    torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0 \
    --index-url "$torch_index"

  # vLLM supplies the policy server, local judge, and most serving dependencies.
  "$python_bin" -m pip install vllm==0.23.0

  # Dependencies used by the local verl trainer and the bundled lmms_eval tree.
  "$python_bin" -m pip install \
    'accelerate==1.13.0' codetiming 'datasets==5.0.0' dill \
    'hydra-core==1.3.2' 'pandas==3.0.3' 'peft==0.19.1' \
    'pyarrow==24.0.0' pybind11 pylatexenc 'ray[default]==2.55.1' \
    torchdata 'tensordict>=0.8.0,<=0.10.0,!=0.9.0' wandb \
    'packaging>=20.0' tensorboard safetensors 'qwen-vl-utils==0.0.14' \
    math-verify nltk langdetect pybase64 msgspec \
    sacrebleu zss zstandard transformers-stream-generator yt-dlp \
    timm tqdm-multiprocess evaluate sqlitedict tenacity \
    'sentence-transformers==5.5.1' pytablewriter Levenshtein matplotlib

  # Qwen3-VL uses Transformers 5.3; Gemma 4 needs native support in 5.8.
  if [[ "$profile" == qwen ]]; then
    transformers_version=5.3.0
  else
    transformers_version=5.8.0
  fi
  "$python_bin" -m pip install --no-deps \
    "transformers==$transformers_version" 'numpy==2.2.6' \
    'huggingface-hub==1.8.0' 'tokenizers==0.22.2' \
    'fla-core==0.5.0' 'flash-linear-attention==0.5.0' \
    'xgrammar==0.2.1'
  "$python_bin" -m pip install 'mcp==1.27.2' 'openai==2.24.0'

  if [[ "$profile" == qwen ]]; then
    "$python_bin" -m pip install --no-deps liger-kernel
  fi

  # Register this checkout without changing the tested runtime versions.
  "$python_bin" -m pip install -e "$REPO_ROOT" --no-deps

  if [[ "$profile" == qwen && "${SKIP_FLASH_ATTN:-0}" != 1 ]]; then
    command -v nvcc >/dev/null 2>&1 || {
      echo "Qwen training needs flash-attn and nvcc. Load a CUDA toolkit, then rerun this script." >&2
      exit 2
    }
    "$python_bin" -m pip install packaging psutil ninja
    MAX_JOBS="${MAX_JOBS:-4}" FLASH_ATTENTION_FORCE_BUILD=TRUE \
      FLASH_ATTENTION_FORCE_CXX11_ABI=TRUE \
      "$python_bin" -m pip install --no-build-isolation 'flash-attn==2.8.3'
  fi
fi

export PYTHONPATH="$REPO_ROOT/persistent-exploration-eval:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
"$python_bin" - "$profile" "${SKIP_FLASH_ATTN:-0}" <<'PY'
import importlib.util
import sys
from importlib.metadata import version

profile, skip_flash_attn = sys.argv[1:]
expected_transformers = '5.3.0' if profile == 'qwen' else '5.8.0'
expected = {
    'torch': '2.11.0',
    'vllm': '0.23.0',
    'transformers': expected_transformers,
    'tokenizers': '0.22.2',
    'datasets': '5.0.0',
    'mcp': '1.27.2',
    'fla-core': '0.5.0',
    'xgrammar': '0.2.1',
}
for name, wanted in expected.items():
    actual = version(name)
    assert actual.split('+')[0] == wanted, f'{name}: expected {wanted}, found {actual}'
    print(f'{name}: {actual}')
if profile == 'gemma':
    assert importlib.util.find_spec('transformers.models.gemma4'), 'Gemma 4 support is missing'
if profile == 'qwen' and skip_flash_attn != '1':
    assert importlib.util.find_spec('flash_attn'), 'flash-attn is missing for Qwen training'
from lmms_eval.mcp.vero2_mcp import app  # noqa: F401
import verl.trainer.main_ppo  # noqa: F401
print('Training and evaluation imports: OK')
PY
"$python_bin" -m lmms_eval --help >/dev/null
echo "Environment ready: conda activate $env_target"
