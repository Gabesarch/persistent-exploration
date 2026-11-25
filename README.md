# Learning Persistent Exploration for Visual Search

Code for **“Learning Persistent Exploration for Visual Search.”** Persistent Exploration trains a vision-language model to inspect an image, answer a question, and keep searching with the zoom tool when its learned evaluator decides the answer needs another attempt.

This repository provides Qwen3-VL-8B and Gemma 4 E4B training recipes and evaluation code used for the paper. The data and checkpoint links are below. The implementation builds on [Vero](https://github.com/zlab-princeton/vero), [verl](https://github.com/volcengine/verl), and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

| Path | Purpose |
| --- | --- |
| [`persistent-exploration-rl/`](persistent-exploration-rl/README.md) | Qwen and Gemma training launchers, configurations, and prompts |
| [`verl/`](verl/) | Forked trainer, async tool rollout, reward and checkpoint code |
| [`persistent-exploration-eval/`](persistent-exploration-eval/README.md) | VisualProbe and generalization evaluation |
| [`tests/`](tests/) | Focused CPU checks for mass bounds and related loss logic |

## What is available

| Item | Location |
| --- | --- |
| Training data: 2,042 VisualProbe image–question pairs | [Persistent Exploration RL Data](https://huggingface.co/datasets/gsarch/Persistent-Exploration-RL-Data) |
| Trained Qwen3-VL-8B checkpoint | [Persistent Exploration Qwen](https://huggingface.co/gsarch/Persistent-Exploration-Qwen3VLi) |
| Trained Gemma 4 E4B checkpoint (step 60) | [Persistent Exploration Gemma](https://huggingface.co/gsarch/Persistent-Exploration-Gemma4-E4B) |
| Held-out evaluation questions | VisualProbe Easy, Medium, and Hard datasets loaded by `persistent-exploration-eval` |

The training dataset contains **only training examples**. The held-out VisualProbe questions are separate: 141 Easy, 268 Medium, and 106 Hard, for 515 total. The evaluation code loads those benchmark datasets from Hugging Face. The training launcher uses the same questions for validation; [`prepare_validation.py`](persistent-exploration-rl/prepare_validation.py) downloads their pinned revisions and exports a local validation JSONL and images.

## Requirements

- Linux, Python 3.12, and CUDA GPUs
- A local Qwen3-VL-8B-Instruct or Gemma 4 E4B base model for the selected training recipe; a Qwen3.5-27B judge for evaluation
- The training data and validation splits exported to local JSONL files
- A Python environment with PyTorch, vLLM, verl dependencies, and the evaluation dependencies

The reference training recipes use eight GPUs. You can launch either directly with Bash and pass Hydra overrides for a smaller smoke run.

The data download and export commands are below. See the [training guide](persistent-exploration-rl/README.md) for the recipe settings and data format.

## Install

For Qwen, create a Python 3.12 environment with a CUDA compatible PyTorch build and vLLM. Then install this checkout:

```bash
python -m pip install -e '.[vllm,geo,math]'
```

The bundled evaluation harness also needs its lmms-eval dependencies; see [`persistent-exploration-eval/README.md`](persistent-exploration-eval/README.md). GPU package versions must match the CUDA and driver stack on the machine.

## Train

Export paths or copy [`set_paths.local.sh.example`](set_paths.local.sh.example) to `set_paths.local.sh` and edit it. The local file is ignored by Git.

Download the training dataset to scratch and run its included exporter:

```bash
export ROOT_PATH=/path/to/runtime
hf download gsarch/Persistent-Exploration-RL-Data --repo-type dataset \
  --revision 07fcdb7be28759fe278507aca49d1a38382456d4 \
  --local-dir "$ROOT_PATH/datasets/persistent-exploration-release"
python "$ROOT_PATH/datasets/persistent-exploration-release/export_verl.py" \
  "$ROOT_PATH/datasets/persistent-exploration-train"
export TRAIN_FILES="$ROOT_PATH/datasets/persistent-exploration-train/chart_search_counting_visual_probe_train_rollout_score_leq05_all.verl.jsonl"
```

Prepare the held-out validation data, then launch training:

```bash
export ROOT_PATH=/path/to/runtime
python persistent-exploration-rl/prepare_validation.py "$ROOT_PATH/datasets/visual-probe-validation"
export VAL_FILES="$ROOT_PATH/datasets/visual-probe-validation/visual_probe_validation.verl.jsonl"
export IMAGE_ROOT="$ROOT_PATH/datasets/persistent-exploration-train"
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
bash persistent-exploration-rl/train.sh
```

Checkpoints are written under `$SAVE_PATH/checkpoints/`; `SAVE_PATH` defaults to `$ROOT_PATH/outputs`. The run has 200 optimizer updates, saves every ten steps, and resumes automatically when restarted with the same output directory. For a small launch validation, override `trainer.total_training_steps`, `data.train_batch_size`, and `actor_rollout_ref.rollout.n` on the command line.

For Gemma 4 E4B, use the same `TRAIN_FILES`, `VAL_FILES`, and `IMAGE_ROOT` with a local Gemma base model and a Gemma-compatible Python environment:

```bash
export GEMMA4_MODEL_PATH=/path/to/gemma-4-E4B-it
bash persistent-exploration-rl/train_gemma.sh
```

See the [Gemma training guide](persistent-exploration-rl/README.md#gemma-4-e4b) for its runtime and recipe settings. The package dependency constraints have not been changed for Gemma.

## Evaluate

On a GPU machine, run the published Qwen checkpoint on a benchmark:

```bash
export ROOT_PATH=/path/to/runtime
export HF_HOME=/path/to/hf_cache
# Set VERO2_EVAL_CONDA_ENV only if evaluation uses a separate conda environment.
bash persistent-exploration-eval/examples/evaluate_checkpoint.sh --task visual_probe \
  --judge /path/to/Qwen3.5-27B --output /path/to/results
```

This downloads the linked Qwen checkpoint when needed. Use `--task all` to evaluate VisualProbe Easy, Medium, Hard, Pinpoint, VStar, and O3Bench, or choose one benchmark. Select the published Gemma checkpoint with `--checkpoint gsarch/Persistent-Exploration-Gemma4-E4B`; its packaged profile supports VisualProbe. See the [evaluation guide](persistent-exploration-eval/README.md) for setup and options.

## Implementation and license

The implementation builds on Vero, verl, and lmms-eval. The `verl` and `lmms_eval` runtime trees provide the trainers, model adapters, tools, and benchmark tasks used by the entrypoints. Copyright and license notices are in [`LICENSE`](LICENSE), [`Notice.txt`](Notice.txt), and the [lmms-eval license notice](persistent-exploration-eval/LICENSE).
