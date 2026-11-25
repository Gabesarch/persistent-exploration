# Learning Persistent Exploration for Visual Search

This repository releases the training and evaluation code for Persistent Exploration, a method that teaches vision-language models to continue visual search when an answer is premature.

## Installation

Set up the environment for your model:

**Qwen3-VL-8B**

```bash
bash scripts/setup_env.sh qwen
conda activate persistent-exploration-qwen
```

**Gemma 4 E4B**

```bash
bash scripts/setup_env.sh gemma
conda activate persistent-exploration-gemma
```

## Evaluation

### Qwen3-VL-8B

Evaluate the [Qwen3-VL-8B checkpoint](https://huggingface.co/gsarch/Persistent-Exploration-Qwen3VLi):

```bash
export ROOT_PATH=/path/to/runtime
bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --task visual_probe --judge /path/to/Qwen3.5-27B --judge-tp 2 \
  --output "$ROOT_PATH/eval/qwen_visual_probe"
```

### Gemma 4 E4B

Evaluate the [Gemma 4 E4B checkpoint](https://huggingface.co/gsarch/Persistent-Exploration-Gemma4-E4B):

```bash
export ROOT_PATH=/path/to/runtime
bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --task visual_probe --checkpoint gsarch/Persistent-Exploration-Gemma4-E4B \
  --judge /path/to/Qwen3.5-27B --judge-tp 2 \
  --output "$ROOT_PATH/eval/gemma_visual_probe"
```

See the [evaluation guide](persistent-exploration-eval/README.md) for benchmarks and options.

## Training

Prepare the [training dataset](https://huggingface.co/datasets/gsarch/Persistent-Exploration-RL-Data) as described in the [training guide](persistent-exploration-rl/README.md).

**Qwen3-VL-8B**

```bash
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
bash persistent-exploration-rl/train.sh
```

**Gemma 4 E4B**

```bash
export GEMMA4_MODEL_PATH=/path/to/gemma-4-E4B-it
bash persistent-exploration-rl/train_gemma.sh
```

## License

This code builds on [Vero](https://github.com/zlab-princeton/vero), [verl](https://github.com/volcengine/verl), and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval). Copyright and license notices are in [`LICENSE`](LICENSE), [`Notice.txt`](Notice.txt), and the [lmms-eval license notice](persistent-exploration-eval/LICENSE).
