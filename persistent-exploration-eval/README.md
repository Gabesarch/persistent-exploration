# Evaluation

Install and activate the Qwen or Gemma environment with [`scripts/setup_env.sh`](../scripts/setup_env.sh) from the repository root before running an evaluation.

[`examples/evaluate_checkpoint.sh`](examples/evaluate_checkpoint.sh) evaluates the linked Qwen and Gemma checkpoints. It serves the model with vLLM, runs the bundled `lmms_eval` harness, and uses a local Qwen3.5-27B judge for benchmarks that need one. Run it on a GPU machine:

```bash
export ROOT_PATH=/path/to/runtime
export HF_HOME=/path/to/hf_cache

bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --task visual_probe --judge /path/to/Qwen3.5-27B \
  --judge-tp 2 --output "$ROOT_PATH/eval/visual_probe"
```

The default model is [`gsarch/Persistent-Exploration-Qwen3VLi`](https://huggingface.co/gsarch/Persistent-Exploration-Qwen3VLi), pinned to its release revision. The script downloads it into `$ROOT_PATH/models` if absent. Use `--checkpoint PATH` to evaluate a local Hugging Face checkpoint. Provide two GPUs with at least 40 GB each when using `--judge-tp 2`.

For the [published Gemma 4 E4B checkpoint](https://huggingface.co/gsarch/Persistent-Exploration-Gemma4-E4B), activate the Gemma environment and select its packaged VisualProbe profile:

```bash
conda activate persistent-exploration-gemma
bash persistent-exploration-eval/examples/evaluate_checkpoint.sh \
  --task visual_probe --checkpoint gsarch/Persistent-Exploration-Gemma4-E4B \
  --judge /path/to/Qwen3.5-27B --judge-tp 2 \
  --output "$ROOT_PATH/eval/gemma_visual_probe"
```

The Gemma profile supports VisualProbe Easy, Medium, and Hard. It uses the model's native tool calls and the inference settings supplied with the checkpoint. Use Transformers and vLLM builds with Gemma 4 support.

For Qwen, `--task` accepts `visual_probe` (Easy, Medium, and Hard), `visual_probe_easy`, `visual_probe_medium`, `visual_probe_hard`, `pinpoint`, `vstar`, `o3_bench`, or `all` (all six paper tasks). Gemma supports the VisualProbe tasks. Use `--limit N` for a smoke run, `--offset N` to resume at a particular example, and `--dry-run` to print the command. Full benchmarks can take hours because of multi-turn zoom trajectories and judging.

The VisualProbe datasets are loaded from `Mini-o3/VisualProbe_Easy`, `Mini-o3/VisualProbe_Medium`, and `Mini-o3/VisualProbe_Hard`. Pinpoint uses `godx7/Pinpoint-Bench`, VStar uses `lmms-lab/vstar-bench`, and O3Bench uses `m-Just/O3-Bench`. Dataset downloads are enabled by default. Before `--task all`, fetch the extra local files for Pinpoint and O3Bench:

```bash
python persistent-exploration-eval/scripts/download_pinpoint_bench.py
python persistent-exploration-eval/scripts/cache_o3_bench.py
```

Set `HF_HUB_OFFLINE=1`, `HF_DATASETS_OFFLINE=1`, and `TRANSFORMERS_OFFLINE=1` after caching if you want an offline run.

The canonical task configs use temperature 0.7, top-p 0.8, and top-k 20. The remaining files in `examples/` and `scripts/` are runtime helpers for this entrypoint.
