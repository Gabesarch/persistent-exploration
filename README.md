# Learning to Persist in Visual Search

Persistent Exploration is a reinforcement learning method that teaches vision-language models to continue searching after unsuccessful attempts. Models learn from failed and recovered searches, and learn to decide when further exploration is needed.

---

## Highlights

- Visual search with image zoom tools and continued exploration after failure.
- Training recipes and checkpoints for Qwen3-VL-8B and Gemma 4 E4B.
- Evaluation on VisualProbe, V*, PinpointBench, and O3Bench.

---

## Installation

### Environment Setup

Run the setup script from the repository root for your model.

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

The script installs the training and evaluation dependencies. See [scripts/setup_env.sh](scripts/setup_env.sh) for the setup flow.

---

## Quick Start: Evaluation

Evaluate the released Qwen checkpoint on VisualProbe:

```bash
conda activate persistent-exploration-qwen
export ROOT_PATH=/path/to/runtime

hf download Qwen/Qwen3.5-27B \
  --local-dir "$ROOT_PATH/models/Qwen3.5-27B"

bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --task visual_probe \
  --judge "$ROOT_PATH/models/Qwen3.5-27B" --judge-tp 2 \
  --output "$ROOT_PATH/eval/qwen_visual_probe"
```

The runner downloads the checkpoint and evaluates the Easy, Medium, and Hard splits. Evaluation does not require training data.

See the [evaluation guide](persistent-exploration-eval/README.md) for Gemma, individual benchmarks, and the full evaluation suite.

---

## Quick Start: Training

Prepare the [released training dataset](https://huggingface.co/datasets/gsarch/Persistent-Exploration-RL-Data) using the [data setup instructions](persistent-exploration-rl/README.md#data-setup), then launch training for your model.

**Qwen3-VL-8B**

```bash
conda activate persistent-exploration-qwen
export MODEL_PATH=/path/to/Qwen3-VL-8B-Instruct
bash persistent-exploration-rl/train.sh
```

**Gemma 4 E4B**

```bash
conda activate persistent-exploration-gemma
export GEMMA4_MODEL_PATH=/path/to/gemma-4-E4B-it
bash persistent-exploration-rl/train_gemma.sh
```

See the [training guide](persistent-exploration-rl/README.md) for data preparation, model downloads, and configuration.

---

## Model Checkpoints

| Model | Base Model | Checkpoint |
| --- | --- | --- |
| Persistent Exploration Qwen | [Qwen3-VL-8B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct) | [Hugging Face](https://huggingface.co/gsarch/Persistent-Exploration-Qwen3VLi) |
| Persistent Exploration Gemma | [Gemma 4 E4B IT](https://huggingface.co/google/gemma-4-E4B-it) | [Hugging Face](https://huggingface.co/gsarch/Persistent-Exploration-Gemma4-E4B) |

---

## Evaluation Benchmarks

| Benchmark | Task | Supported Models |
| --- | --- | --- |
| VisualProbe Easy, Medium, Hard | `visual_probe` | Qwen, Gemma |
| V* | `vstar` | Qwen, Gemma |
| PinpointBench | `pinpoint` | Qwen, Gemma |
| O3Bench | `o3_bench` | Qwen, Gemma |

See the [evaluation guide](persistent-exploration-eval/README.md#benchmarks) for dataset links and commands.

---

## Documentation

- [Evaluation guide](persistent-exploration-eval/README.md)
- [Training guide](persistent-exploration-rl/README.md)
- [Training data setup](persistent-exploration-rl/README.md#data-setup)

---

## Acknowledgements

This project builds on [Vero](https://github.com/zlab-princeton/vero), [verl](https://github.com/volcengine/verl), and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval).

## License

The code is licensed under [Apache 2.0](LICENSE). Third-party notices are preserved in [Notice.txt](Notice.txt) and the [evaluation harness license](persistent-exploration-eval/LICENSE). Model and dataset licenses are provided on their Hugging Face pages.
