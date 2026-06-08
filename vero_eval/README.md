# vero_eval

Agentic, multi-turn **tool-use** evaluation for vision-language models, built on a
vendored `lmms_eval`. The policy model is served by **vLLM** (OpenAI-compatible) with
Qwen3-VL tool-calling flags; the `async_openai` backend drives an MCP tool loop
(`lmms_eval/mcp/vero2_mcp.py`) and scores the result. `vero2_mcp.py` is the single
canonical tool server (it supersedes the old `offline_image_tools.py` and
`vision_tools.py`): an **offline-safe PIL core** (`image_crop`, `image_resize`,
`draw_box`) that always works, plus 10 richer drawing tools rendered via the
optional cv2/matplotlib `Visualizer` (lazy-imported; if those deps are missing the
server still starts and the PIL core still works).

This is the cleaned-up successor to `tools_scratch/`. It mirrors the
`reason-vlm-eval` conventions (same `set_paths.sh` pattern, task-yaml naming, judge).

## Layout
```
vero_eval/
├── set_paths.sh.example   # copy to set_paths.sh and edit (gitignored; account paths)
├── lmms_eval/             # the eval package (run via `python -m lmms_eval`)
├── examples/eval_domain.sh# canonical runner
├── scripts/               # serve_qwen_vllm.sh, *.sbatch, probe_tool_image.py, test_countqa.py
├── prompts/               # countqa_system_prompt.txt, vero_system_prompt.txt, *.jinja
├── results/               # run outputs (gitignored; countqa_10/ kept as reference)
└── logs/                  # serve + slurm logs (gitignored)
```

## Setup
```bash
cp set_paths.sh.example set_paths.sh   # then edit REPO_ROOT / ROOT_PATH / conda env / CHECKPOINT_PATH
```
`set_paths.sh` activates the conda env (needs **vLLM ≥0.17** for Qwen3-VL, plus `mcp`,
`lmms_eval`, `matplotlib`, `cv2`) and sets HF caches, offline flags, `PORT`, `CHECKPOINT_PATH`.
It is auto-sourced by every script here.

## Run (interactively on a GPU node)
```bash
cd vero_eval
bash scripts/serve_qwen_vllm.sh                                   # start vLLM on $PORT (default 9001)
SYSTEM_PROMPT=prompts/countqa_system_prompt.txt \
  bash examples/eval_domain.sh --tasks countqa_reasoning --limit 10
```

## Run (SLURM)
```bash
sbatch vero_eval/scripts/eval_countqa_10.sbatch     # submit from the repo root
```
Serves once, then runs three countqa configs to `results/countqa_10/`.

## Picking the task yaml (per base model)
Tool usage is orthogonal to the yaml — enable it with `mcp_server_path` (default in
`eval_domain.sh`); choose the yaml suffix by the base model being evaluated:

| Base model / mode      | yaml suffix            |
|------------------------|------------------------|
| Vero (RL agent)        | `_reasoning`           |
| Vero thinking          | `_reasoning_samplingq3`|
| Qwen3-VL Instruct      | `_qwen3_zs`            |
| Qwen3-VL Thinking      | `_qwen3_thinking_zs`   |

## Multi-turn tool loop (`async_openai` model_args)
`model_version`, `base_url`, `api_key`, `mcp_server_path`, `system_prompt`,
`max_tool_iterations` (default 10), `max_tool_images` (default 8),
`tool_image_as_user_turn` (default false; fallback for templates that drop images in
`role:"tool"` messages). Per-yaml sampling (`top_p`/`top_k`/penalties) is forwarded.

Validate tool-image rendering for a new checkpoint:
`sbatch vero_eval/scripts/probe_tool_image.sbatch`.

## Judge (for judge-based tasks, e.g. visual_probe / simplevqa / mathvista)
countqa is `exact_match` (no judge). Judge tasks score via `get_judge_engine`, picked
by `JUDGE_BACKEND` in `eval_domain.sh`:

| `JUDGE_BACKEND` | judge | GPU | when |
|---|---|---|---|
| `server` (recommended on 1 GPU) | reuses the policy server on `$PORT` → judge **is** the Qwen3-VL policy | **0 extra** | single H100; vision-capable |
| `engine` (default) | in-process **Qwen3-32B** (text-only) | needs a free GPU (OOMs beside the 0.9-util policy) | 2-GPU alloc, or no resident policy |
| `server` + `START_VLLM_JUDGE_SERVER=1` | dedicated sleep-mode judge (Qwen3-32B) via `scripts/serve_judge_vllm.sh` | its own GPU (set `VLLM_JUDGE_CUDA_VISIBLE_DEVICES`) | want a strong judge + 2nd GPU |

Run a judge task (reuses the policy server, zero extra GPU):
```bash
sbatch vero_eval/scripts/eval_visual_probe.sbatch        # JUDGE_BACKEND=server by default
# or interactively, with the policy server already up:
JUDGE_BACKEND=server bash examples/eval_domain.sh --tasks visual_probe_easy_qwen3_zs --limit 20
```
For a dedicated Qwen3-32B judge on a 2-GPU node:
```bash
sbatch --gres=gpu:2 --export=ALL,JUDGE_BACKEND=server,START_VLLM_JUDGE_SERVER=1,VLLM_JUDGE_CUDA_VISIBLE_DEVICES=1 \
  vero_eval/scripts/eval_visual_probe.sbatch
```
