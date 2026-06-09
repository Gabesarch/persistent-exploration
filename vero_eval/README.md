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
| Judge | `JUDGE_BACKEND` | `engine` | `engine` / `server` (see §7) |

```bash
# tools OFF, single-turn:
MCP_SERVER_PATH='' bash examples/eval_tasks.sh --tasks countqa_qwen3_zs
# 5 turns instead of 10, with the counting system prompt:
MAX_TOOL_ITERATIONS=5 SYSTEM_PROMPT=prompts/countqa_system_prompt.txt \
  bash examples/eval_tasks.sh --tasks countqa_reasoning
```

---

## 5. The three runner types

All three ultimately call `examples/eval_domain.sh`. Run interactive ones from the
package root (`cd vero_eval`); submit the array from the **repo root**.

### TYPE 1 — one model on an explicit list of task yamls
```bash
cd vero_eval
bash examples/eval_tasks.sh --tasks countqa_reasoning,chartqa_reasoning,charxiv_reasoning --limit 20
```
Auto-serves `CHECKPOINT_PATH` if nothing is on `$PORT`, then forwards the comma-list to
`eval_domain.sh`. Use any task name that exists under `lmms_eval/tasks/**` (the `task:`/
`group:` field). For judge tasks on 1 GPU add `JUDGE_BACKEND=server`.

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

### TYPE 3 — one checkpoint across multiple domains
```bash
cd vero_eval
bash examples/eval_across_domains.sh --domains all --model-suffix reasoning --limit 50
```
Serves `CHECKPOINT_PATH` **once**, then loops the domains (default `server` judge so judge
tasks reuse the policy endpoint). Results go to `results/<model>_<ts>/<domain>/`.
`--domains chart_ocr,stem` to pick a subset.

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
| **Qwen3-VL-8B Instruct** (baseline) | `_qwen3_zs` |
| **Qwen3-VL-8B Thinking** (baseline) | `_qwen3_thinking_zs` |
| Qwen2.5-VL-7B Instruct | `_qwen25_zs` |
| MiMo-VL | `_mimo_zs` |
| GPT-5-nano (API; not these local runners) | `_gpt5nano_zs` |

## 6b. Domains → tasks (`examples/_domain_map.sh`)

| Domain | task stems (suffix appended at runtime) |
|---|---|
| `chart_ocr` | chartqa, chartqa_pro, infovqa_val, charxiv, chartmuseum, evochart |
| `stem` | mmmu_pro, mathvision_test, mathvista_testmini |
| `spatial_action` | cv_bench, embspatial, erqa, game_qa_lite, blink |
| `knowledge_rec` | realworldqa, simplevqa_en, fvqa, mmvetv2_group_img |
| `grounding_counting_search` | countbenchqa, countqa, mme_realworld_lite, vstar_bench, aerialvg_bbox, visual_probe_{easy,medium,hard}, screenspot_point_in_box, screenspotpro_point_in_box |
| `instruction_following` | mia_bench, mm_mt_bench, mmifeval |
| `ball_drop` | balldrop |

`expand_domain` appends `_<suffix>` to each stem and **auto-drops** stems with no such
variant (e.g. charxiv has no `_qwen3_zs`). Edit this file to add/remove domain members.

---

## 7. Judge (judge-based tasks: visual_probe, simplevqa, mathvista, charxiv, …)

countqa is `exact_match` (no judge). Judge tasks score via `get_judge_engine`, chosen by
`JUDGE_BACKEND` in `eval_domain.sh`:

| `JUDGE_BACKEND` | judge | GPU | use when |
|---|---|---|---|
| `server` (best on 1 GPU) | reuses the policy server → judge **is** the Qwen3-VL policy | none extra | single H100; vision-capable |
| `engine` (default) | in-process **Qwen3-32B** (text-only) | own GPU (OOMs beside the policy) | 2-GPU alloc |
| `server` + `START_VLLM_JUDGE_SERVER=1` | dedicated sleep-mode Qwen3-32B (`scripts/serve_judge_vllm.sh`) | own GPU (`VLLM_JUDGE_CUDA_VISIBLE_DEVICES`) | strong judge + 2nd GPU |

```bash
# zero-extra-GPU judge run:
sbatch vero_eval/scripts/eval_visual_probe.sbatch        # JUDGE_BACKEND=server
# dedicated 32B judge on a 2-GPU node:
sbatch --gres=gpu:2 --export=ALL,JUDGE_BACKEND=server,START_VLLM_JUDGE_SERVER=1,VLLM_JUDGE_CUDA_VISIBLE_DEVICES=1 \
  vero_eval/scripts/eval_visual_probe.sbatch
```

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
