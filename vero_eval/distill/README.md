# Distillation: GPT-5.5 reasoning + tool-use traces

Collect a **teacher** model's chain-of-thought **and** multi-turn tool calls on a
task, then filter to correct traces and emit an SFT dataset to distill into a
student VLM. Tasks wired up so far:

| Task | Dataset | Toolset | Teacher variant |
|---|---|---|---|
| SketchVLM ball-drop | `loganbolton/sketchvlm-physics-ball-drop` | zoom / draw | `sketchvlm_ball_drop_gpt55_zoom` / `…_draw` |
| V* (visual search) | `lmms-lab/vstar-bench` | **zoom+draw combined** | `vstar_bench_gpt55_zoomdraw` |

## TL;DR

```bash
# V* with the combined zoom+draw toolset, full split, 8 workers:
TASKS="vstar_bench_gpt55_zoomdraw" bash vero_eval/distill/run_distill.sh

# ball-drop, both toolsets:
TASKS="sketchvlm_ball_drop_gpt55_zoom sketchvlm_ball_drop_gpt55_draw" bash vero_eval/distill/run_distill.sh

# small debug run:
TASKS="vstar_bench_gpt55_zoomdraw" LIMIT=20 WORKERS=6 bash vero_eval/distill/run_distill.sh
```

Output (per task, under `/scratch/gpfs/ZHUANGL/lc1556/distillation/<task-stem>/`):

```
images/<doc_id>.png                               # input image, ORIGINAL resolution (tools zoom into this)
tool_images/<doc_id>/turn<t>_call<k>_idx<n>.jpg   # every tool-returned image, persisted
traces/<task>.jsonl                               # one JSON line per doc (the trace)
traces/<task>.summary.json                        # accuracy / trainable / token / error summary
```

## Why a dedicated harness (not lmms_eval)

`gpt-5.5` tool-calling **with `reasoning_effort`** is rejected on
`/v1/chat/completions` (the API says *"use /v1/responses instead"*). So the teacher
must run through the OpenAI **Responses API**. lmms_eval's `async_openai` backend
speaks chat-completions, so `collect_traces.py` reimplements the **same** multi-turn
tool loop against the Responses API:

- reasoning items are echoed back between turns (with `encrypted_content`, `store=false`)
  so the reasoning model keeps its chain across tool calls;
- tool **images** are fed back as `input_image` user turns (a Responses
  `function_call_output` is text-only);
- the opening image is sent at **full resolution** to match what the student sees;
- images are addressed by **`img_idx`** (matching training) via the
  `LMMS_EVAL_IMAGE_REGISTRY` env passed to the MCP tool subprocess;
- the teacher sees the **same** system prompt and **same** tools as the student
  tool tasks (read from the task YAML), so the traces match the student interface.

The teacher is **remote** — no GPU, no `torch`. Runs on a login node or any CPU
node with internet, in an isolated `uv run --no-project` env.

## The modular task YAMLs (single source of truth)

Composition mirrors the rest of `vero_eval` (see `lmms_eval/tasks/_tool_templates/`):

```
vstar_bench_gpt55_zoomdraw.yaml
  include:
    - _default_template_yaml                     # dataset, doc_to_text/visual, split
    - ../_tool_templates/_toolset_zoomdraw_yaml  # metadata.allowed_tools + system_prompt_files
  metadata:
    distill: { teacher_model, reasoning_effort, reasoning_summary, max_output_tokens,
               max_tool_iterations, max_tool_images, adapter? }
```

`metadata` deep-merges across includes, so the toolset template contributes tools +
system fragments and the task adds only the `distill` block.

**Toolset templates** (`_tool_templates/`): `_toolset_zoom_yaml`, `_toolset_draw_yaml`,
and `_toolset_zoomdraw_yaml` (both sets concatenated — zoom + all draw tools, with
`system_base` + `system_zoom` + `system_draw`). NOTE: do not include the zoom and
draw templates together to "combine" them — metadata list-merge is last-writer-wins,
so use the explicit `_toolset_zoomdraw_yaml`.

### Add a new distillation task (recipe)

1. **Dataset task dir** `lmms_eval/tasks/<name>/` with a `_default_template_yaml`
   (dataset_path, test_split, doc_to_*, scoring) — typically already exists for eval.
2. **Torch-free adapter** `lmms_eval/tasks/<name>/distill_adapter.py` exposing four
   functions (the harness imports this by file path, so it must NOT import the heavy
   lmms_eval package):
   - `doc_to_text(doc, lmms_eval_specific_kwargs) -> str`
   - `doc_to_visual(doc) -> [PIL.Image]`
   - `doc_id(doc, index) -> str`  (names the image/trace files)
   - `score(doc, final_text) -> {"pred", "gt", "correct"[, "category"]}`
   (If the eval `utils.py` is already dependency-light, the adapter can just re-export it.)
3. **Teacher variant** `<name>_gpt55_<toolset>.yaml`: `include` the dataset template +
   a toolset template, add a `metadata.distill` block. The harness finds the adapter by
   the convention `module=distill_adapter` with the standard function names; override via
   `metadata.distill.adapter: {module, doc_to_text, doc_to_visual, doc_id, score, dataset_split}`.
4. Run: `TASKS="<name>_gpt55_<toolset>" bash run_distill.sh`, then `build_distill_dataset.py --root <out>`.

## Trace schema (`traces/<task>.jsonl`, one object per doc)

| field | meaning |
|---|---|
| `doc_id`, `run_id`, `index`, `category` | doc identity (image/trace files named by `doc_id`) |
| `gt`, `pred`, `correct`, `trainable` | ground truth, parsed pred, correctness; `trainable` = clean run AND final-turn answer == gt |
| `teacher_model`, `reasoning_effort`, `prompt_version` | teacher config |
| `system_prompt`, `user_text`, `allowed_tools`, `system_prompt_files` | exact prompt provenance (self-contained) |
| `input_image_path`, `image_registry` | input image + full img_idx→path map (all resolve on disk) |
| `turns[]` | per round: `reasoning_summary` (teacher CoT, summarized), `message_text`, `tool_calls[]`, `usage` |
| `turns[].tool_calls[]` | `name`, `arguments`, `result_text`, `result_img_idx`, `result_image_path` |
| `final_text`, `status` | final answer; `status` ∈ ok / incomplete / max_tool_iterations / error |
| `usage` | cumulative `input_tokens` / `output_tokens` / `reasoning_tokens` |

> gpt-5.5 does **not** expose verbatim chain-of-thought; `reasoning_summary` is the
> API's summarized CoT (`reasoning_effort=medium`, `summary=auto`).

## Build the distillation SFT dataset (filter to correct traces)

```bash
python vero_eval/distill/build_distill_dataset.py --root <out>            # filter=trainable (default)
python vero_eval/distill/build_distill_dataset.py --root <out> --require-tools   # drop no-tool convs
```

`--filter trainable` (default) keeps only traces that ran clean **and** whose final
answer equals ground truth (correct *and* well-formed for final-answer supervision).
Output under `<root>/sft/`: `<root-name>_distill.jsonl`, `.stats.json`, `.preview.md`.

### SFT record schema

Multimodal multi-turn conversation, assistant turns rendered in this project's
**hermes** style so it matches the student tool interface (Qwen3-VL / Vero):

```jsonc
{
  "id": "vstar_bench_gpt55_zoomdraw::direct_attributes_0",
  "source_task": "...", "toolset": "zoomdraw"|"zoom"|"draw",
  "teacher_model": "gpt-5.5", "reasoning_effort": "medium",
  "ground_truth": "A", "category": "direct_attributes", "answer": "\\boxed{A}",
  "images": ["<img_idx0 input .png>", "<img_idx1 tool .jpg>", ...],   // img_idx order, all resolve on disk
  "messages": [
    {"role": "system", "content": "<system_base + system_zoom + system_draw>"},
    {"role": "user",   "content": [{"type":"image","image":"<path>","img_idx":0}, {"type":"text","text":"<question>"}]},
    {"role": "assistant", "content": "<think>...</think>\n<tool_call>\n{\"name\":...,\"arguments\":{...}}\n</tool_call>"},
    {"role": "tool",   "content": [{"type":"text","text":"Registered ... img_idx 1."}, {"type":"image","image":"<path>","img_idx":1}]},
    {"role": "assistant", "content": "<think>...</think>\n<answer>\n\\boxed{A}\n</answer>"}
  ],
  "num_turns": 2, "num_tool_calls": 1
}
```

Convert to your trainer (verl / ms-swift / LLaMA-Factory): assistant turns are plain
hermes text; `user`/`tool` turns carry images as content parts, and the flat `images`
list (img_idx order) is provided for loaders that want `<image>` tokens + a separate
image list. The `<tool_call>`/`<tool_response>`-style markers match the verl
ToolAgentLoop / Qwen-Agent convention.

## AI Sandbox key

`collect_traces.py` imports `sandbox.get_client()` from `/home/lc1556/repo/AI-sandbox`,
which loads `AI_SANDBOX_KEY` from that repo's `.env` (gitignored) or the environment.
`gpt-5.5` is available on the key (`gpt-5.5-mini`/`-chat`/`-nano` are not). Rate limit
≈ 200K tok/min org-wide — keep `WORKERS` ≲ 8 and run tasks sequentially.
