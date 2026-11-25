# Evaluation

Evaluate Persistent Exploration checkpoints with image zoom tools and learned answer/retry decisions. The pipeline uses vLLM for inference and lmms-eval for benchmark evaluation.

Run all commands from the repository root after completing [environment setup](../README.md#installation).

## Quick Start

### Prepare the Judge

The examples below use two GPUs and a local Qwen3.5-27B judge. Download the judge once:

```bash
export ROOT_PATH=/path/to/runtime
export HF_HOME="$ROOT_PATH/hf_cache"

hf download Qwen/Qwen3.5-27B \
  --local-dir "$ROOT_PATH/models/Qwen3.5-27B"
```

### Qwen3-VL-8B

```bash
conda activate persistent-exploration-qwen
bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --task visual_probe \
  --judge "$ROOT_PATH/models/Qwen3.5-27B" --judge-tp 2 \
  --output "$ROOT_PATH/eval/qwen_visual_probe"
```

The default checkpoint is [Persistent Exploration Qwen](https://huggingface.co/gsarch/Persistent-Exploration-Qwen3VLi). The runner downloads it on first use.

### Gemma 4 E4B

```bash
conda activate persistent-exploration-gemma
bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --checkpoint gsarch/Persistent-Exploration-Gemma4-E4B \
  --task visual_probe \
  --judge "$ROOT_PATH/models/Qwen3.5-27B" --judge-tp 2 \
  --output "$ROOT_PATH/eval/gemma_visual_probe"
```

Both checkpoints can be evaluated on all supported benchmarks. Each released checkpoint supplies its evaluation profile, including tool use and answer/retry settings.

## Benchmarks

Select a benchmark with `--task`.

| Task | Benchmark | Dataset | Models |
| --- | --- | --- | --- |
| `visual_probe_easy` | VisualProbe Easy | [Mini-o3/VisualProbe_Easy](https://huggingface.co/datasets/Mini-o3/VisualProbe_Easy) | Qwen, Gemma |
| `visual_probe_medium` | VisualProbe Medium | [Mini-o3/VisualProbe_Medium](https://huggingface.co/datasets/Mini-o3/VisualProbe_Medium) | Qwen, Gemma |
| `visual_probe_hard` | VisualProbe Hard | [Mini-o3/VisualProbe_Hard](https://huggingface.co/datasets/Mini-o3/VisualProbe_Hard) | Qwen, Gemma |
| `vstar` | V* | [lmms-lab/vstar-bench](https://huggingface.co/datasets/lmms-lab/vstar-bench) | Qwen, Gemma |
| `pinpoint` | PinpointBench | [godx7/Pinpoint-Bench](https://huggingface.co/datasets/godx7/Pinpoint-Bench) | Qwen, Gemma |
| `o3_bench` | O3Bench | [m-Just/O3-Bench](https://huggingface.co/datasets/m-Just/O3-Bench) | Qwen, Gemma |

Use `--task visual_probe` for all three VisualProbe splits or `--task all` for the full suite.

### Run All Benchmarks

Prepare the PinpointBench and O3Bench image caches, then run the suite:

```bash
conda activate persistent-exploration-qwen
python persistent-exploration-eval/scripts/download_pinpoint_bench.py
python persistent-exploration-eval/scripts/cache_o3_bench.py

bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --task all \
  --judge "$ROOT_PATH/models/Qwen3.5-27B" --judge-tp 2 \
  --output "$ROOT_PATH/eval/qwen_all"
```

For Gemma, use the same dataset caches and select its checkpoint:

```bash
conda activate persistent-exploration-gemma
bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --checkpoint gsarch/Persistent-Exploration-Gemma4-E4B \
  --task all \
  --judge "$ROOT_PATH/models/Qwen3.5-27B" --judge-tp 2 \
  --output "$ROOT_PATH/eval/gemma_all"
```

## Evaluate a Subset

Use `--limit` to select the number of examples and `--offset` to choose the starting example. These options apply to each selected task.

```bash
conda activate persistent-exploration-qwen
bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --task visual_probe_easy --limit 30 --offset 0 \
  --judge "$ROOT_PATH/models/Qwen3.5-27B" --judge-tp 2 \
  --output "$ROOT_PATH/eval/qwen_easy_30"
```

## Evaluate a Local Checkpoint

Pass a local checkpoint directory with `--checkpoint`:

```bash
bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --checkpoint /path/to/checkpoint \
  --task visual_probe \
  --judge "$ROOT_PATH/models/Qwen3.5-27B" --judge-tp 2 \
  --output "$ROOT_PATH/eval/local_checkpoint"
```

Use the matching model environment. Local Gemma checkpoints must include the Gemma evaluation profile supplied with the released checkpoint.

## Results

The output directory contains lmms-eval aggregate metrics and per-example records. Choose a separate `--output` directory for each model or evaluation run.

For all command-line options, run:

```bash
bash persistent-exploration-eval/examples/evaluate_checkpoint.sh --help
```
