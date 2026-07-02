# hstarbench for vero_eval

Humanoid 360° visual search ([hstarbench](https://huggingface.co/datasets/humanoid-vstar/hstarbench)). Launch like any other benchmark — pick a task yaml and submit.

## Quick start

```bash
source vero_eval/set_paths.sh
TASKS=hstarbench_mixed_reasoning_zoom WITH_TOOLS=1 sbatch vero_eval/scripts/eval_tasks.sbatch
```

Or sweep all Hstar splits with the domain helper:

```bash
source vero_eval/examples/_domain_map.sh
TASKS=$(expand_domain hstarbench reasoning_zoom) WITH_TOOLS=1 sbatch vero_eval/scripts/eval_tasks.sbatch
```

## Tasks

| Task yaml | Use case |
|-----------|----------|
| `hstarbench_mixed_reasoning_zoom` | **Primary** — Vero RL checkpoint + `image_zoom_in_tool` on 5760×1792 strip |
| `hstarbench_mixed_reasoning` | Strip + zoom with generic tool schema |
| `hstarbench_mixed` | Original agentic rotate/submit env (needs `HSTAR_VAGEN_ROOT`) |
| `hstarbench_hos` / `hstarbench_hps` | Single-split agentic |
| `hstarbench_hos_reasoning` / `hstarbench_hps_reasoning` | Single-split strip + zoom agentic |

## Data

JSONL indices live on shared scratch (not in git):

```
$HSTARBENCH_ROOT/data/hstarbench_{mixed,hos,hps}_{test,rl,sft}.jsonl
$HSTARBENCH_ROOT/data/verl/hstarbench_{mixed,hos,hps}_{rl,sft,test}.verl.jsonl
$HSTARBENCH_ROOT/strips/   # 5760×1792 cylindrical panoramas
$HSTARBENCH_ROOT/views/    # initial view thumbnails
```

Regenerate lmms-eval JSONL + assets:

```bash
export HSTAR_VAGEN_ROOT=/path/to/thinkin360
python vero_eval/lmms_eval/tasks/hstarbench/build_hstarbench_jsonl.py --split rl
python vero_eval/lmms_eval/tasks/hstarbench/build_hstarbench_jsonl.py --split sft
# or: sbatch vero_eval/scripts/build_hstarbench_{rl,sft}.sbatch
```

Set `HSTARBENCH_ROOT` in `vero_eval/set_paths.sh` if your assets live elsewhere.

## Metrics

Geometric scoring (no LLM judge): `hstarbench_cone_success`, `hstarbench_bbox_iou_success`, `hstarbench_bbox_iou`, `hstarbench_has_answer_bbox`.

Agentic tasks also report `hstarbench_success` (env rotate/submit or zoom cone).

## RL training eval (different from benchmark test)

During verl RL training, validation uses **in-loop** logic that differs from offline `vero_eval`:

| | In-loop (`verl`) | Offline checkpoint eval (`vero_eval`) |
|--|------------------|---------------------------------------|
| Data | `.verl.jsonl` from `build_hstarbench_verl_jsonl.py` (RL + SFT) | Task yaml JSONL (`*_test.jsonl`, `*_rl.jsonl`, `*_sft.jsonl`) |
| Reward / metric | `reward_type: hstar_strip` → cone success on strip bbox | `hstarbench_*_reasoning_zoom` → same cone/IoU via lmms-eval |
| Sampling | `val_kwargs`: temp=0.7, n=1 (match `*_reasoning_zoom`) | Same in yaml `generation_kwargs` |
| Monitor split | RL train distribution | Use domain `hstarbench_train_monitor` (RL + held-out test) |

```bash
# Build verl JSONL (one merged file per split; train/val split is up to you)
python vero_eval/lmms_eval/tasks/hstarbench/build_hstarbench_verl_jsonl.py --source all

# In-loop val-only smoke (8 GPU)
bash vero2_scripts/run_valonly_hstar_qwen3vl_cropzoom.sh

# Post-training checkpoint sweep: RL distribution + benchmark generalization
source vero_eval/examples/_domain_map.sh
TASKS=$(expand_domain hstarbench_train_monitor reasoning_zoom) WITH_TOOLS=1 sbatch vero_eval/scripts/eval_checkpoints_array.sbatch
```

## Agentic tasks only

`hstarbench_mixed` and per-split agentic variants need the vagen HstarEnv simulator:

```bash
export HSTAR_VAGEN_ROOT=/path/to/thinkin360   # checkout with vagen.env.hstar
```

Strip+zoom tasks (`*_reasoning*`) do **not** require `HSTAR_VAGEN_ROOT`.
