# vero_eval — how to run an evaluation

`vero_eval` evaluates a vision-language model with **multi-turn visual tool use**: a vLLM
server hosts the policy model, and the harness lets it call drawing/crop tools over several
turns before it answers; answers are scored by exact-match or an LLM judge.

This guide is the quick how-to. See `README.md` for the full reference.

---

## 0. One-time setup
```bash
cp vero_eval/set_paths.sh.example vero_eval/set_paths.sh
# edit: REPO_ROOT, ROOT_PATH (your /scratch), conda env, CHECKPOINT_PATH, HF_HOME
```
`set_paths.sh` is gitignored and is auto-sourced by every script. Your datasets just need
to be in your HF cache (`$HF_HOME/hub`); the loader finds them offline by their HF id.

---

## 1. The easy way — one entrypoint: `run_eval.sbatch`

Pick **MODEL**, **TASKS**, **TOOLS**, and submit **from the repo root**:

```bash
# A model on a few tasks, tools on:
sbatch --export=ALL,MODEL=qwen3vl,TASKS=countqa_reasoning,visual_probe_easy_qwen3_zs,TOOLS=on \
       vero_eval/run_eval.sbatch

# Your Vero checkpoint across a whole domain:
sbatch --export=ALL,MODEL=vero,CHECKPOINT_PATH=/scratch/.../actor/huggingface,TASKS=grounding_counting_search,TOOLS=on \
       vero_eval/run_eval.sbatch

# Everything (all 6 domains), tools on:
sbatch --export=ALL,MODEL=vero,CHECKPOINT_PATH=/scratch/.../actor/huggingface,TASKS=all,TOOLS=on \
       vero_eval/run_eval.sbatch

# Exact-match-only run (countqa) — no judge, so 1 GPU is enough:
sbatch --gres=gpu:1 --export=ALL,MODEL=qwen3vl,TASKS=countqa_reasoning,TOOLS=on \
       vero_eval/run_eval.sbatch
```

**Parameters** (all via `--export=ALL,KEY=VAL,...`):

| Key | Values | Meaning |
|-----|--------|---------|
| `MODEL` | `vero` · `vero-thinking` · `qwen3vl` · `qwen3vl-thinking` · `qwen25vl` · `<abs ckpt path>` | which model; sets the checkpoint + default task suffix |
| `CHECKPOINT_PATH` | a path | required for `vero*`/`qwen3vl-thinking`/`qwen25vl`; the HF weights dir |
| `TASKS` | `t1,t2,...` **or** a domain (`chart_ocr`…`all`) | what to run |
| `TOOLS` | `on` (default) · `off` | enable/disable the visual tools |
| `LIMIT` | N | quick test on N examples per task |
| `MAX_TOOL_ITERATIONS` | N (default 10) | number of tool-call turns |
| `SUFFIX` | `reasoning`/`qwen3_zs`/… | override the per-task suffix for a domain run |

`run_eval.sbatch` serves the model, runs the tasks, and (for judge tasks) launches the
judge — all in one job.

---

## 2. Picking MODEL ↔ task suffix

Each base model has its own task yaml flavor. `run_eval.sbatch` sets a sensible default;
override with `SUFFIX=` for a domain run.

| MODEL | suffix |
|-------|--------|
| `vero` (RL agent) | `reasoning` |
| `vero-thinking` | `reasoning_samplingq3` |
| `qwen3vl` (Instruct baseline) | `qwen3_zs` |
| `qwen3vl-thinking` | `qwen3_thinking_zs` |
| `qwen25vl` | `qwen25_zs` |

> **Vero & system prompts:** Vero bakes its reasoning system prompt into its chat template,
> so you do **not** pass a system prompt for it (the entrypoint handles this). Baselines do
> honor `SYSTEM_PROMPT=vero_eval/prompts/<file>.txt` if you want to steer them.

---

## 3. Tools — on/off and turns

- **Disable tools:** `TOOLS=off` (the model answers in a single turn, no tool calls).
- **Number of turns:** `MAX_TOOL_ITERATIONS=N` (default 10).
- The tools live in `vero_eval/lmms_eval/mcp/vero2_mcp.py` (crop, zoom, draw box/points/grid/…).

---

## 4. Judge (for tasks scored by an LLM, e.g. visual_probe / mathvista / charxiv)

`countqa` etc. are exact-match (no judge). Judge tasks **auto-launch a dedicated Qwen3-32B
judge on the 2nd GPU** — the strong, reportable judge (same as the previous repo). That's
why `run_eval.sbatch` defaults to `--gres=gpu:2`.

- 1 GPU + a judge task → it falls back to using the **policy model as the judge (8B)** and
  prints a **WARNING** (don't report those numbers). Give it 2 GPUs for the real judge.
- `JUDGE_BACKEND=engine` runs the 32B judge in-process instead (also needs 2 GPUs).

---

## 5. Other runners (when you need them)

`run_eval.sbatch` covers most cases. Underneath, it uses:

| Script | Use directly when … |
|--------|---------------------|
| `examples/eval_tasks.sh --tasks a,b` | interactive node, explicit task list |
| `examples/eval_across_domains.sh --domains all` | interactive node, one ckpt across domains |
| `examples/submit_eval_checkpoints.sh --train-dir DIR --domain D` | sweep **many checkpoints** of a training run (SLURM array) |
| `examples/eval_domain.sh --tasks …` | the lowest-level runner everything calls |

Example checkpoint sweep:
```bash
bash vero_eval/examples/submit_eval_checkpoints.sh \
  --train-dir /scratch/.../checkpoints/verl/<RUN> --domain grounding_counting_search \
  --model-suffix reasoning --every 200 --array-chunk 4
```

---

## 6. Add a new task

1. Make `vero_eval/lmms_eval/tasks/<name>/` (copy an existing task like `countqa/`); set
   `dataset_path` to the **HF id** (it resolves from your cache automatically) and add the
   per-model variant yamls (`<name>_reasoning.yaml`, `<name>_qwen3_zs.yaml`, …).
2. Smoke it: `sbatch --gres=gpu:1 --export=ALL,MODEL=qwen3vl,TASKS=<name>_qwen3_zs,LIMIT=2 vero_eval/run_eval.sbatch`.
3. To add it to a domain, append its stem to `examples/_domain_map.sh`.

---

## 7. Where results go
`vero_eval/results/<model>_<timestamp>/…` (per-task or per-domain). Logs in `vero_eval/logs/`.
Both are gitignored.
