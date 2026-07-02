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
$HSTARBENCH_ROOT/data/thinkin360_{mixed,hos,hps}_test.jsonl
$HSTARBENCH_ROOT/strips/   # 5760×1792 cylindrical panoramas
$HSTARBENCH_ROOT/views/    # initial view thumbnails
```

Set `HSTARBENCH_ROOT` in `vero_eval/set_paths.sh` if your assets live elsewhere.

## Metrics

Geometric scoring (no LLM judge): `hstarbench_cone_success`, `hstarbench_bbox_iou_success`, `hstarbench_bbox_iou`, `hstarbench_has_answer_bbox`.

Agentic tasks also report `hstarbench_success` (env rotate/submit or zoom cone).

## Agentic tasks only

`hstarbench_mixed` and per-split agentic variants need the vagen HstarEnv simulator:

```bash
export HSTAR_VAGEN_ROOT=/path/to/thinkin360   # checkout with vagen.env.hstar
```

Strip+zoom tasks (`*_reasoning*`) do **not** require `HSTAR_VAGEN_ROOT`.
