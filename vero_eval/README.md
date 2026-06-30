# vero_eval

Agentic, **multi-turn tool-use** evaluation for vision-language models, built on a
vendored `lmms_eval`. The policy model is served by **vLLM** (OpenAI-compatible) with
Qwen3-VL tool-calling flags; the `async_openai` backend drives an MCP visual-tool loop
and scores the result (exact-match or an LLM judge). It mirrors the `reason-vlm-eval`
conventions (same `set_paths.sh`, task-yaml naming, judge) but serves the policy
*externally* and evaluates it agentically.

---

## 1. Layout
```
vero_eval/
├── set_paths.sh.example   # copy to set_paths.sh and edit (gitignored; account paths + secrets)
├── lmms_eval/             # the eval package (run via `python -m lmms_eval`)
├── examples/              # runners (see §5)
│   ├── eval_domain.sh        # the ONE canonical runner; everything else calls it
│   ├── eval_tasks.sh         # TYPE 1: a model on an explicit --tasks list
│   ├── eval_across_domains.sh# TYPE 3: one checkpoint across many domains
│   ├── submit_eval_checkpoints.sh  # TYPE 2 (submit): array over a run's checkpoints
│   └── _domain_map.sh        # shared domain -> task-list map (sourced by the above)
├── scripts/
│   ├── serve_qwen_vllm.sh    # start the policy vLLM server
│   ├── serve_judge_vllm.sh   # optional dedicated judge server
│   ├── eval_checkpoints_array.sbatch  # TYPE 2 (array body)
│   ├── eval_countqa_10.sbatch / eval_visual_probe.sbatch  # ready demos
│   └── probe_tool_image.py / test_countqa.py  # debug helpers
├── prompts/   # system prompts + chat templates
├── results/   # run outputs (gitignored; countqa_10/ kept as a reference)
└── logs/      # serve + slurm logs (gitignored)
```

## 2. Setup
```bash
cp set_paths.sh.example set_paths.sh   # edit REPO_ROOT / ROOT_PATH / conda env / CHECKPOINT_PATH
```
`set_paths.sh` (gitignored) activates the conda env — needs **vLLM ≥0.17** (Qwen3-VL),
`mcp`, `lmms_eval`, `matplotlib`, `cv2` — and sets HF caches, offline flags, `PORT`,
`CHECKPOINT_PATH`, `MODEL_NAME`. **Every script sources it automatically.**

---

## 3. How it works (architecture)

**Which server.** The policy model is served **externally** by
`scripts/serve_qwen_vllm.sh` →
`vllm serve $CHECKPOINT_PATH --served-model-name $MODEL_NAME --port $PORT
--enable-auto-tool-choice --tool-call-parser hermes --max-model-len 32768
--mm-processor-kwargs '{"max_pixels":...}' --api-key $API_KEY --enforce-eager`.
It runs as a background daemon and the eval talks to it at `http://localhost:$PORT/v1`.
The checkpoint is **baked in at launch** and the server holds 0.9 GPU util, so only one
checkpoint can be served per GPU at a time. (This is the key difference from
reason-vlm-eval, which loads the policy *in-process* via `--model vllm`.)

**Multi-turn / agentic loop.** Driven by the `async_openai` lmms_eval model
(`lmms_eval/models/chat/async_openai.py`, registered as `async_openai`). For each
question it runs an OpenAI chat-completions loop:
> `while finish_reason == "tool_calls" and tool_iteration < max_tool_iterations:`
> execute each requested tool → append the tool result (incl. returned images) as
> `role:"tool"` messages → re-query the model.

So **"multi-turn" = the tool-call loop**, bounded by `max_tool_iterations` (default 10)
and `max_tool_images` (default 8). vLLM's `hermes` parser turns the model's
`<tool_call>{…}</tool_call>` into structured `tool_calls`; the loop is protocol-correct
(one assistant message with `tool_calls` ids, one `role:"tool"` per call keyed by
`tool_call_id`). Per-yaml sampling (`top_p`/`top_k`/penalties) is forwarded.

**Tools.** An MCP stdio server, `lmms_eval/mcp/vero2_mcp.py`, launched as a subprocess
by `MCPClient`. It is the single canonical tool server (supersedes the old
`offline_image_tools.py`/`vision_tools.py`) and exposes:
- **offline-safe PIL core** (always available): `image_crop`, `image_resize`, `draw_box`;
- **richer cv2/matplotlib drawing tools** (lazy-imported): `draw_numbered_boxes`,
  `draw_numbered_points`, `draw_grid`, `draw_circle`, `draw_line`, `draw_arrow`,
  `draw_text`, `draw_polygon`, `draw_mask`, `draw_numbered_mask` (0–1000 coord scale).

---

## 4. Knobs — tools on/off, number of turns, system prompt

All set via env before any runner (consumed in `examples/eval_domain.sh`, forwarded as
`async_openai` model_args):

| Knob | Env var | Default | Effect |
|---|---|---|---|
| **Use tools?** | `MCP_SERVER_PATH` | `lmms_eval/mcp/vero2_mcp.py` | set **`MCP_SERVER_PATH=''`** to disable tools (no MCP client → single-turn, no tool loop) |
| **Number of turns** | `MAX_TOOL_ITERATIONS` | `10` | max tool-call rounds per question |
| Max tool images | (model_arg `max_tool_images`) | `8` | cap on returned images kept in context |
| System prompt | `SYSTEM_PROMPT` | none | path to a `prompts/*.txt` (e.g. `prompts/countqa_system_prompt.txt`) |
| Policy port | `PORT` | `9001` | vLLM server port |
| Judge | `JUDGE_BACKEND` | auto | judge tasks → dedicated Qwen3-32B on a 2nd GPU (see §7) |

```bash
# tools OFF, single-turn:
MCP_SERVER_PATH='' bash examples/eval_tasks.sh --tasks countqa_qwen3_zs
# 5 turns instead of 10, with the counting system prompt:
MAX_TOOL_ITERATIONS=5 SYSTEM_PROMPT=prompts/countqa_system_prompt.txt \
  bash examples/eval_tasks.sh --tasks countqa_reasoning
```

### 4b. Tool-calling tasks: modular toolsets + system prompts

A tool-calling task is composed from three independent, reusable pieces, so you
never copy prompt/tool config between tasks:

1. **A dataset template** — the existing per-benchmark template (`visual_probe_hard.yaml`,
   `_default_template_point_yaml`, …): dataset, scorer, `doc_to_*`.
2. **A toolset template** (`lmms_eval/tasks/_tool_templates/_toolset_*_yaml`) — the
   single source of truth for one toolset: which tools, and which system-prompt
   fragments.
3. **(optional) An answer-protocol template** (`_answerproto_*_yaml`) — for
   grounding tasks that answer by "calling" a non-MCP tool (ScreenSpot's `computer_use`).

A task yaml just `include`s these; their `metadata` blocks **deep-merge** (see
`load_yaml_config`), and `async_openai` reads the merged result:

| `metadata` key | Meaning | Set by |
|---|---|---|
| `allowed_tools` | MCP tool names / fnmatch patterns offered to the model (`[image_zoom_in_tool]`, `["draw_*"]`). Absent = all server tools; matching nothing is a hard error. | toolset template |
| `system_prompt_files` | files concatenated (in order, after any run-wide `SYSTEM_PROMPT`) into the turn-0 system message. Paths resolve against the `vero_eval/` root. | toolset template |
| `extra_tools` | OpenAI function specs offered but NOT backed by MCP (answer protocols). | answer-protocol template |
| `terminal_tools` | tool names whose call ENDS the episode — recorded as the final answer, never executed. | answer-protocol template |

**How the system prompt is built.** The final turn-0 system message is
`[run-wide SYSTEM_PROMPT?] + [system_prompt_files…]`, and the model's chat
template then appends its own `# Tools` / `<tools>` schema block. So for a zoom
task it is: `system_base.txt` ⊕ `system_zoom.txt` ⊕ (auto-rendered tool schemas).
The fragments live in `prompts/tools/` (`system_base.txt`, `system_zoom.txt`,
`system_draw.txt`) — edit those to change behavior across **all** tasks of a
toolset at once. The system message persists across every tool turn (the loop
only appends tool calls/results).

Toolsets exposed by `lmms_eval/mcp/vero2_mcp.py`:
- **zoom** (`_toolset_zoom_yaml`): `image_zoom_in_tool` — eval twin of the RL
  training tool (`verl/tools/image_zoom_in_tool.py`): `bbox_2d=[x1,y1,x2,y2]` on
  the 0-1000 scale + `label`, crop upscaled into the training pixel budget.
- **draw** (`_toolset_draw_yaml`): `draw_box, draw_numbered_boxes,
  draw_numbered_points, draw_grid, draw_circle, draw_line, draw_arrow, draw_text,
  draw_polygon`.

Ready-made tasks (Qwen3-VL Instruct; same sampling/scorers as the `_qwen3_zs`
baselines): `visual_probe_hard_qwen3_{zoom,draw}`,
`screenspotpro_point_in_box_qwen3_{zoom,draw}`.

**Submit all of them with one script** (judge-aware GRES, unique PORT per job):
```bash
bash vero_eval/examples/submit_tool_evals.sh                       # the 4 tasks
TASKS="visual_probe_hard_qwen3_zoom screenspotpro_point_in_box_qwen3_draw" \
  bash vero_eval/examples/submit_tool_evals.sh                     # a subset
DRY_RUN=1 bash vero_eval/examples/submit_tool_evals.sh             # preview
# defaults to pli/H100 (--account=vlmrl); for ailab/H200 instead:
#   PARTITION=ailab ACCOUNT=zhuangl GPU_TYPE=h200 bash vero_eval/examples/submit_tool_evals.sh
```

#### Authoring a new tool-calling task

1. **New toolset?** Add `prompts/tools/system_<name>.txt` and a template
   `lmms_eval/tasks/_tool_templates/_toolset_<name>_yaml` with `metadata.allowed_tools`
   + `metadata.system_prompt_files: [prompts/tools/system_base.txt, prompts/tools/system_<name>.txt]`.
   (The tools themselves must be registered in `lmms_eval/mcp/vero2_mcp.py`.)
2. **New task** — create `tasks/<bench>/<bench>_<model>_<toolset>.yaml`:
   ```yaml
   include:
     - <dataset_template>                          # e.g. visual_probe_hard.yaml
     - ../_tool_templates/_toolset_<name>_yaml
     # - ../_tool_templates/_answerproto_computer_use_yaml   # grounding only
   task: "<bench>_<model>_<toolset>"
   generation_kwargs: { … }                        # copy from the _qwen3_zs sibling
   lmms_eval_specific_kwargs:
     default: { pre_prompt: "<benchmark/answer framing>", post_prompt: "" }
   ```
   Put **tool behavior** in the system fragments and **answer framing** in
   `pre_prompt`. Do **not** add a `metadata` block unless you need an extra key —
   `allowed_tools`/`system_prompt_files`/`version` come from the includes.

   > **Wording matters for tool use.** Phrase `pre_prompt`'s answer constraint as
   > *"Your final answer should be a short phrase"*, **not** *"Respond with a short
   > phrase only"*. The latter reads as "answer directly/tersely" and suppresses
   > tool calls. `system_base.txt` already states that the answer-format applies to
   > the FINAL answer only and that the model should inspect with tools first — so
   > the `pre_prompt` only needs to describe the final answer's shape. (This is a
   > common trap when adapting a no-tool `_qwen3_zs` baseline yaml — its
   > "respond … only" wording is fine without tools but self-defeating with them.)
3. **Run** it: `MODEL=qwen3vl TOOLS=on TASKS=<task> sbatch --export=ALL vero_eval/run_eval.sbatch`
   (add it to `TASKS=...` in `submit_tool_evals.sh` for the batch path).

(NB: when passing a comma-list `TASKS`, give it as a real env var, not inside
`--export=ALL,TASKS=a,b` — sbatch splits that on the commas. `submit_tool_evals.sh`
submits one task per job to avoid this.)

---

## 5. Running

### Easiest — one sbatch entrypoint (`run_eval.sbatch`)
Pick MODEL / TASKS / TOOLS and submit from the **repo root** (see `instruction.md`):
```bash
sbatch --export=ALL,MODEL=qwen3vl,TASKS=countqa_reasoning,visual_probe_easy_qwen3_zs,TOOLS=on \
       vero_eval/run_eval.sbatch
sbatch --export=ALL,MODEL=vero,CHECKPOINT_PATH=/scratch/.../actor/huggingface,TASKS=all,TOOLS=on \
       vero_eval/run_eval.sbatch
```
It serves the model, runs the tasks, and launches the judge (for judge tasks) in one job.
`TASKS` accepts a comma-list **or** a domain name. Add `--gres=gpu:1` for exact-match-only runs.

### The three underlying runner types
All ultimately call `examples/eval_domain.sh`. Run interactive ones from the package root
(`cd vero_eval`); submit the array from the **repo root**.

### TYPE 1 — one model on an explicit list of task yamls
```bash
cd vero_eval
bash examples/eval_tasks.sh --tasks countqa_reasoning,chartqa_reasoning,charxiv_reasoning --limit 20
```
Auto-serves `CHECKPOINT_PATH` if nothing is on `$PORT`, then forwards the comma-list to
`eval_domain.sh`. Use any task name that exists under `lmms_eval/tasks/**` (the `task:`/
`group:` field). Judge tasks need 2 GPUs for the dedicated Qwen3-32B judge (see §7).

### TYPE 2 — an array of checkpoints on a domain (SLURM)
```bash
# from the repo root:
bash vero_eval/examples/submit_eval_checkpoints.sh \
  --train-dir /scratch/.../checkpoints/verl/<RUN> \
  --domain grounding_counting_search --model-suffix reasoning \
  --every 200 --array-chunk 4
```
Globs the run's `global_step_*`/`checkpoint_*` dirs, writes a `checkpoints.index`, and
submits `sbatch --array=0-(N-1) scripts/eval_checkpoints_array.sbatch`. **Each array task
= one checkpoint** and does the full cycle itself: resolve weights (merging FSDP shards →
`actor/huggingface` with `verl.model_merger` if needed) → serve that checkpoint → eval the
domain's tasks → stop the server. Results land in
`$ROOT_PATH/vero_eval/logs/<run-id>/raw/<global_step_N>/`. Flags: `--domain all` submits
one array per canonical domain; `--max-checkpoints N`, `--every X`, `--dry-run`.

**Focusing the sweep on VisualProbe** (full setting): a `visual_probe` domain
(`visual_probe_{easy,medium,hard}`, **not** in `all`) is provided so the sweep isn't diluted
by the other `grounding_counting_search` tasks. Carry the 10 MP setting via `MM_MAX_PIXELS`
(propagated by `--export=ALL`); the array auto-applies Vero's forced-`<think>` template for
any `reasoning*` suffix:
```bash
MM_MAX_PIXELS=10000000 bash vero_eval/examples/submit_eval_checkpoints.sh \
  --train-dir /scratch/.../checkpoints/verl/<RUN> \
  --domain visual_probe --model-suffix reasoning_zoom --every 200 --array-chunk 2
# (one checkpoint or the base model instead of a sweep -> use eval_visual_probe_full.sbatch below.)
```

### TYPE 3 — one checkpoint across multiple domains
```bash
cd vero_eval
bash examples/eval_across_domains.sh --domains all --model-suffix reasoning --limit 50
```
Serves `CHECKPOINT_PATH` **once**, then loops the domains (default `server` judge so judge
tasks reuse the policy endpoint). Results go to `results/<model>_<ts>/<domain>/`.
`--domains chart_ocr,stem` to pick a subset.

### Visual Probe at the full setting (`scripts/eval_visual_probe_full.sbatch`)

The canonical, reportable VisualProbe zoom configuration in one command:
**`max_pixels = 10,000,000`** served at launch, the **per-model** zoom task + sampling, and
the dedicated **Qwen3-32B** judge (needs `gpu:2`). Submit from the repo root:

```bash
# Vero RL checkpoint — TRAINING-FAITHFUL (forced-<think> chat template, temp 0.7/top_p 1/top_k -1,
# image_indexing: training_faithful); runs visual_probe_hard_reasoning_zoom:
sbatch --export=ALL,MODEL=vero,CHECKPOINT_PATH=/scratch/.../actor/huggingface \
       vero_eval/scripts/eval_visual_probe_full.sbatch

# Base Qwen3-VL-8B-Instruct — qwen3 sampling (temp 1.0/top_p 1/top_k 40/presence_penalty 2.0);
# runs visual_probe_hard_qwen3_zoom:
sbatch --export=ALL,MODEL=qwen3vl vero_eval/scripts/eval_visual_probe_full.sbatch

# all three difficulties; or a single-GPU smoke (8B self-judge, NOT reportable):
sbatch --export=ALL,MODEL=vero,CHECKPOINT_PATH=...,PROBE=all vero_eval/scripts/eval_visual_probe_full.sbatch
sbatch --gres=gpu:1 --export=ALL,MODEL=qwen3vl,LIMIT=10,START_VLLM_JUDGE_SERVER=0 \
       vero_eval/scripts/eval_visual_probe_full.sbatch
```

It picks the task by `MODEL` (`vero` → `visual_probe_<probe>_reasoning_zoom`, `qwen3vl` →
`visual_probe_<probe>_qwen3_zoom`), exports `MM_MAX_PIXELS=10000000`, and for Vero sets the
forced-`<think>` chat template automatically. `PROBE` = `hard` (default) | `easy` | `medium` | `all`.

> **`MM_MAX_PIXELS` is a serve-time knob** read only by `scripts/serve_qwen_vllm.sh` (vLLM
> `--mm-processor-kwargs max_pixels`); it caps the **initial** image resolution the model sees
> — the zoom tool re-crops the *original* PNG at native resolution regardless. It does not
> flow through `eval_domain.sh`, so set it in the environment that reaches the serve call (the
> full-setting script and `sbatch --export=ALL,MM_MAX_PIXELS=...` both do); default when unset
> is 10,000,000 (the full setting — on the ~8-13MP accuracy plateau). The reuse-if-up server
> check does not inspect a running server's resolution,
> so use a clean `PORT` (or let this script restart the server) when changing it.

---

## 6. Which task yaml per base model (`--model-suffix`)

Tool usage is **orthogonal** to the suffix (enable via `MCP_SERVER_PATH`, not the yaml);
the suffix selects the prompt/sampling/scoring for that base model.

**System prompts are model-dependent — important for Vero.** Vero's `chat_template.jinja`
**bakes in** its reasoning system prompt (`<think>/<answer>/\boxed`) and emits it every
turn (and, in the tools branch, the tool instructions too); its message loop only renders
`user`/`assistant`/`tool`, so a passed `system` message is **silently dropped**. So when
evaluating **Vero, do NOT set `SYSTEM_PROMPT`** — it has no effect (the runner default of
no system prompt is already correct). **Baseline** models (base Qwen3-VL-Instruct/Thinking,
Qwen2.5-VL) do *not* bake one in and *do* honor a passed `system` message, so set
`SYSTEM_PROMPT=prompts/<...>.txt` to control them. (`prompts/vero_system_prompt.txt` is a
reference of the format; it is close to but not byte-identical to Vero's baked-in text —
extract from the checkpoint's `chat_template.jinja` if you need exact parity for a baseline.)

| Base model / mode | suffix |
|---|---|
| **Vero** (our RL agent) | `_reasoning` |
| **Vero thinking** | `_reasoning_samplingq3` |
| **Vero + zoom** (training-faithful) | `_reasoning_zoom` |
| **Qwen3-VL-8B Instruct** (baseline) | `_qwen3_zs` |
| **Qwen3-VL-8B Instruct + zoom** | `_qwen3_zoom` |
| **Qwen3-VL-8B Thinking** (baseline) | `_qwen3_thinking_zs` |
| Qwen2.5-VL-7B Instruct | `_qwen25_zs` |
| MiMo-VL | `_mimo_zs` |
| GPT-5-nano (API; not these local runners) | `_gpt5nano_zs` |

**Zoom variants** (multi-turn `image_zoom_in_tool`, available for every benchmark): pick by
base model exactly like the non-tool suffixes. `_reasoning_zoom` (Vero) replays the RL
training input — forced `<think>`, temp 0.7 / top_p 1.0 / top_k -1, `image_indexing:
training_faithful`, and the exact training tool schema; `_qwen3_zoom` (base Qwen3-VL) uses
Qwen3-VL's sampling (temp 1.0 / top_p 1.0 / top_k 40 / presence_penalty 2.0). Per-model
sampling lives in each task yaml; both require tools on (`MCP_SERVER_PATH=lmms_eval/mcp/vero2_mcp.py`,
the default). For VisualProbe at the production resolution use the full-setting runner in §5.

## 6b. Domains → tasks (`examples/_domain_map.sh`)

| Domain | task stems (suffix appended at runtime) |
|---|---|
| `chart_ocr` | chartqa, chartqa_pro, infovqa_val, charxiv, chartmuseum, evochart |
| `stem` | mmmu_pro, mathvision_test, mathvista_testmini |
| `spatial_action` | cv_bench, embspatial, erqa, game_qa_lite, blink |
| `knowledge_rec` | realworldqa, simplevqa_en, fvqa, mmvetv2_group_img |
| `grounding_counting_search` | countbenchqa, countqa, mme_realworld_lite, vstar_bench, aerialvg_bbox, visual_probe_{easy,medium,hard}, screenspot_point_in_box, screenspotpro_point_in_box |
| `visual_probe` | visual_probe_{easy,medium,hard} — visual-search-only group (NOT in `all`; for a VisualProbe-focused checkpoint sweep, see §5 TYPE 2) |
| `instruction_following` | mia_bench, mm_mt_bench, mmifeval |
| `ball_drop` | balldrop |

`expand_domain` appends `_<suffix>` to each stem and **auto-drops** stems with no such
variant (e.g. charxiv has no `_qwen3_zs`). Edit this file to add/remove domain members.

---

## 7. Judge (judge-based tasks: visual_probe, simplevqa, mathvista, charxiv, …)

countqa is `exact_match` (no judge). Judge tasks are auto-detected; `eval_domain.sh` then
sets up the judge **by default = a dedicated Qwen3-32B on a 2nd GPU** (the strong,
reportable judge, like reason-vlm-eval — hence the runners default to `--gres=gpu:2`):

| Situation | Judge used | GPU |
|---|---|---|
| **judge task + ≥2 GPUs** (default) | dedicated **Qwen3-32B** server on GPU1 | +1 GPU |
| judge task + 1 GPU (fallback) | the **policy model (Qwen3-VL-8B)** judges itself — prints a **WARNING**; not for reporting | 0 extra |
| `JUDGE_BACKEND=engine` (≥2 GPUs) | in-process **Qwen3-32B** | +1 GPU |
| no judge task (countqa) | none set up | — |

```bash
# Strong 32B judge is automatic on 2 GPUs:
sbatch vero_eval/scripts/eval_visual_probe.sbatch
# Quick 8B self-judge (single GPU, not reportable):
sbatch --gres=gpu:1 --export=ALL,START_VLLM_JUDGE_SERVER=0 vero_eval/scripts/eval_visual_probe.sbatch
```
The 8B-self-judge is never silently the default — it only happens on a single GPU and warns.

---

## 8. Adding a new task

1. Create `lmms_eval/tasks/<name>/` with a base yaml + per-model variants, copying an
   existing task (e.g. `countqa/`): set `dataset_path`, `doc_to_visual`/`doc_to_text`/
   `doc_to_target`, `process_results`, `metric_list`, and a `task:` field per variant
   (`<name>_reasoning`, `<name>_qwen3_zs`, …). For LLM-judge tasks, reuse a
   `*_aggregate_judge_results` + `get_judge_engine` pattern (see `visual_probe/utils.py`).
2. **Offline datasets:** keep the upstream **HF id** in `dataset_path` (e.g.
   `dataset_path: lmms-lab/RealWorldQA`). The loader (`lmms_eval/api/task.py`
   `_resolve_local_dataset_path`) resolves an HF id to its local cache snapshot via
   `snapshot_download` (using `$HF_HOME` from `set_paths.sh`), so it loads under
   `HF_HUB_OFFLINE=1` as long as the dataset is in your HF cache. To force a *specific*
   local copy of one dataset, set `dataset_path` to an absolute dir — the loader passes
   existing paths through unchanged (`scripts/pin_datasets.py` can do this in bulk).
3. Smoke it: `bash examples/eval_tasks.sh --tasks <name>_reasoning --limit 2`.
4. To include it in a domain, append its stem to that domain in `examples/_domain_map.sh`.

---

## 9. Quick reference

- **Canonical runner:** `examples/eval_domain.sh --tasks <list> [--limit N]` (env:
  `PORT`, `MCP_SERVER_PATH`, `MAX_TOOL_ITERATIONS`, `SYSTEM_PROMPT`, `JUDGE_BACKEND`).
- **Serve only:** `bash scripts/serve_qwen_vllm.sh` (reads `CHECKPOINT_PATH`/`PORT`).
- **async_openai model_args:** `model_version, base_url, api_key, mcp_server_path,
  system_prompt, max_tool_iterations, max_tool_images, tool_image_as_user_turn`.
- **Validate tool-image rendering for a new checkpoint:**
  `sbatch scripts/probe_tool_image.sbatch`.
