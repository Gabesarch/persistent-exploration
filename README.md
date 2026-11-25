# Learning Persistent Exploration for Visual Search

Persistent Exploration trains a vision-language model to keep searching an image after a proposed answer fails a learned answer/retry check. The model explores through an image zoom tool; the training recipe learns from the additional search and bounds the contribution of retry branches to the policy update.

## Evaluate

On a GPU machine with a local Qwen3.5-27B judge, evaluate the [published Qwen checkpoint](https://huggingface.co/gsarch/Persistent-Exploration-Qwen3VLi):

```bash
export ROOT_PATH=/path/to/runtime
bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --task visual_probe --judge /path/to/Qwen3.5-27B --judge-tp 2 \
  --output "$ROOT_PATH/eval/qwen_visual_probe"
```

To evaluate the [published Gemma checkpoint](https://huggingface.co/gsarch/Persistent-Exploration-Gemma4-E4B), use a Gemma-compatible environment:

```bash
export ROOT_PATH=/path/to/runtime
export VERO2_EVAL_CONDA_ENV=/path/to/gemma-compatible/environment
bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --task visual_probe --checkpoint gsarch/Persistent-Exploration-Gemma4-E4B \
  --judge /path/to/Qwen3.5-27B --judge-tp 2 \
  --output "$ROOT_PATH/eval/gemma_visual_probe"
```

The script downloads the selected checkpoint if needed. `visual_probe` runs Easy, Medium, and Hard; choose one with `--task visual_probe_easy`, `visual_probe_medium`, or `visual_probe_hard`. Qwen also supports `--task all` for the paper's other benchmarks. Gemma's published profile supports VisualProbe. Use `--limit N --offset N` for a subset. See the [evaluation guide](persistent-exploration-eval/README.md) for setup and benchmark details.

## Train

The reference recipes use eight GPUs. For Qwen, install this checkout in a Python 3.12 environment with CUDA-compatible PyTorch and vLLM using `python -m pip install -e '.[vllm,geo,math]'`. Gemma requires a compatible Transformers and vLLM environment; install the checkout there with `python -m pip install -e . --no-deps`.

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

Prepare the separate VisualProbe validation data, then launch Qwen training:

```bash
export ROOT_PATH=/path/to/runtime
python persistent-exploration-rl/prepare_validation.py "$ROOT_PATH/datasets/visual-probe-validation"
export VAL_FILES="$ROOT_PATH/datasets/visual-probe-validation/visual_probe_validation.verl.jsonl"
export IMAGE_ROOT="$ROOT_PATH/datasets/persistent-exploration-train"
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
bash persistent-exploration-rl/train.sh
```

For Gemma 4 E4B, use the same `TRAIN_FILES`, `VAL_FILES`, and `IMAGE_ROOT` with a local Gemma base model and a Gemma-compatible Python environment:

```bash
export GEMMA4_MODEL_PATH=/path/to/gemma-4-E4B-it
bash persistent-exploration-rl/train_gemma.sh
```

Checkpoints are written under `$SAVE_PATH/checkpoints/`, with `SAVE_PATH` defaulting to `$ROOT_PATH/outputs`. Both recipes are configured for 200 optimizer updates; the published Gemma checkpoint is from step 60. See the [training guide](persistent-exploration-rl/README.md) for recipe settings and Hydra overrides.

## License

This code builds on [Vero](https://github.com/zlab-princeton/vero), [verl](https://github.com/volcengine/verl), and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Copyright and license notices are in [`LICENSE`](LICENSE), [`Notice.txt`](Notice.txt), and the [lmms-eval license notice](persistent-exploration-eval/LICENSE).
