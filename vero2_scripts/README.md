# vero2_scripts

The vero2 RL experiments: Slurm launchers (`run_*.sh`), Hydra configs, tool schemas, system
prompts. Everything drives `python -m verl.trainer.main_ppo` on Della.

> **Always launch from the repo root** (`sbatch vero2_scripts/<family>/<variant>/run_*.sh`).
> Launchers `source ./set_paths.sh` via a relative path. Cluster/Slurm details: top-level
> `CLAUDE.md` and `useful_command.sh` (§3 headers, §7 exact commands).

## Layout

```
vero2_scripts/
├── drawing/            draw-tool experiments (counting data)
│   ├── vanilla/        plain GDPO/GSPO training, no interruption
│   ├── force_explore/  answer-retry interruption ON (base + cluster wrappers + smoke)
│   └── experiments/    zoomdraw (combined zoom+draw toolset) + perf benches
├── zoom/               image_zoom_in_tool experiments (chart/search/counting visual-probe data)
│   ├── vanilla/        plain training (full + old-style debug256)
│   ├── force_explore/  answer-retry interruption ON (full + debug)
│   └── experiments/    text_explore, forced_explore, answer_retry_explore, llmjudge,
│                       multiidx, non-hardonly cropzoom, valonly hstarbench
├── archive/            superseded launchers, kept verbatim (see below)
├── config/             ALL live Hydra configs — FLAT, do not nest (see below)
│   ├── archive/        configs referenced only by archived launchers
│   └── tool_config/    tool JSON schemas (zoom / draw / zoom+draw / multiidx)
├── prompts/            system prompts, referenced by absolute path from configs
└── debug_scripts/      trajectory / crop / validation visualizers
```

**Why `config/` stays flat:** every launcher passes
`--config-path=$VERL_ROOT/vero2_scripts/config --config-name=<yaml>`. Hydra resolves configs from
that one directory, so moving a yaml into a subfolder silently breaks every launcher that names it.
Only `config/archive/` is an exception — those yamls are referenced by nothing live. **To revive an
archived launcher, copy its yaml from `config/archive/` back into `config/` first.**

## The 4 canonical experiments

| Family | Variant | Launcher | Config (in `config/`) | batch | mini | steps | per-turn cap | force-explore |
|---|---|---|---|---|---|---|---|---|
| drawing | vanilla | `drawing/vanilla/run_gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_debug256_cpueff.sh` | `gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_debug256_cpueff.yaml` | 256 | 128 | 1000 | 4096 | OFF |
| drawing | force_explore | `drawing/force_explore/run_..._drawonly_cpueff_hardonly_answer_retry_forcedprefill_full.sh` | `gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill.yaml` | 256 | 128 | 1000 | 4096 | ON |
| zoom | vanilla | `zoom/vanilla/run_..._cropzoom_cpueff_hardonly.sh` | `gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_cpueff_hardonly.yaml` | 256 | 128 | 400 | **none** | OFF |
| zoom | force_explore | `zoom/force_explore/run_..._cropzoom_cpueff_hardonly_answer_retry_forcedprefill_full.sh` | `gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_cpueff_hardonly_answer_retry_forcedprefill.yaml` | 128 | 64 | 400 | **none** | ON |

(batch = `data.train_batch_size`, mini = `ppo_mini_batch_size`, steps = `total_training_steps`,
per-turn cap = `max_assistant_response_length`. `rollout.n=8` everywhere → trajectories/step = batch×8.)

Note the drawing-vanilla launcher/config are named `debug256` but are the **full** run — see
[Name traps](#name-traps).

## Force-explore = conditional answer-retry (terminology!)

The `force_explore/` folders implement the **conditional answer-retry interruption**: when the model
emits a wrong `<answer>` (accuracy ≤ `correct_accuracy_threshold`), the answer is **deleted**, the
model is **forced into another tool call** (optionally with a prefilled assistant sentence), and it
re-answers. Rolled-back text is loss-masked (not trained on).

Config block (in the two `answer_retry_forcedprefill` yamls, under `rollout.multi_turn`):

| Key | drawing | zoom |
|---|---|---|
| `conditional_answer_retry_enabled` | true | true |
| `conditional_answer_retry_force_next_tool_call` | true | true |
| `conditional_answer_retry_suffix` | `""` | `""` |
| `conditional_answer_retry_correct_accuracy_threshold` | 0.0 (retry only when accuracy == 0) | 0.0 |
| `conditional_answer_retry_max_retries` | **2** | **null** (uncapped) |

The prefill text is injected via the launcher env **`ANSWER_RETRY_ASSISTANT_PREFILL`** (each
force_explore base sets a family-specific default; export your own before `sbatch` to override).

**Do not confuse with two older zoom/experiments mechanisms:**

- `forced_explore` (in `zoom/experiments/`) = an **older ban-the-answer-token** mechanism (logit-level), not answer-retry.
- `text_explore` = **prompt-nudge** exploration (no interruption at all).

Folder names `force_explore/` always mean the answer-retry interruption.

## How to change training data

**Option A — edit the config yaml:** change `data.train_files` / `data.val_files` in
`config/<name>.yaml`.

**Option B — no file edits (every launcher passes extra Hydra args through `"$@"`):**

```bash
sbatch vero2_scripts/<family>/<variant>/run_....sh vllm \
  data.train_files=/abs/path/train.jsonl data.val_files=/abs/path/val.jsonl
```

⚠️ The leading **`vllm` is required only when you pass overrides directly to a launcher**: training
launchers consume the first argument that doesn't start with `-` as the inference engine. Without
it, your first override (`data.train_files=...`) is eaten as the engine name and folded into a
bogus `actor_rollout_ref.rollout.name=…`, and the run **crashes at startup** (fail-loud, but your
override never took effect). Submitting a launcher **with no arguments** defaults to `vllm` — no
arg needed — and all shipped cluster/smoke wrappers already pass it correctly.

Current defaults:

| Experiment | train_files |
|---|---|
| drawing vanilla | `$VERL_ROOT/datasets/combined_counting_hq/train_hard_answer_lt10.verl.jsonl` — **not in git**, materialize once per checkout (below) |
| drawing force_explore | `/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync/filtered_correct_only/combined_counting_train.correct.partial.verl.jsonl` (Gemini-filtered correct-only) |
| zoom (both) | `/scratch/gpfs/ZHUANGL/gs1693/vero2_datasets/train_rl_datasets/chart_search_counting_visual_probe_train_rollout_score_leq05_all.verl.jsonl` |

Materialize the drawing-vanilla dataset symlink (per checkout/worktree):

```bash
mkdir -p "$VERL_ROOT/datasets" && \
  ln -sfn /scratch/gpfs/ZHUANGL/hw0171/vero2/datasets/combined_counting_hq "$VERL_ROOT/datasets/combined_counting_hq"
```

The readable copy lives in hw0171 scratch (gs1693's copy is permission-locked). The jsonl's image
paths are absolute into hw0171 — for long-term runs consider making a local copy.

## Knob glossary

| Knob | Meaning |
|---|---|
| `data.train_batch_size` | prompts per training step |
| `actor_rollout_ref.rollout.n` | rollouts per prompt (GRPO group); trajectories/step = batch × n (**8** everywhere here) |
| `actor_rollout_ref.actor.ppo_mini_batch_size` | prompts per gradient update; updates/step = batch ÷ mini |
| `...actor.ppo_micro_batch_size_per_gpu` | per-GPU fwd/bwd chunk — throughput only, **no training-math effect** |
| **`...rollout.multi_turn.max_assistant_response_length`** | ⚠️ **PER-TURN output token cap.** 4096 in both drawing configs; **absent in all zoom configs**, so a single zoom turn can run to the full response budget. This is the historically buried speed lever — check it first when a config is mysteriously slow. |
| `data.max_response_length` | TOTAL response budget per trajectory (drawing 16384; zoom vanilla 16384; zoom force_explore 24576) |
| `...rollout.multi_turn.max_assistant_turns` | max assistant turns (11 in all four) |
| `conditional_answer_retry_*` | the force-explore block (see above) |
| `trainer.total_training_steps` / `trainer.save_freq` | run length / checkpoint cadence (cluster wrappers override save_freq) |
| `CHECKPOINT_DATE_TIME` (env) | pins the checkpoint dir name → enables the `%1` resume chain. **A new run needs a NEW `CHECKPOINT_DATE_TIME`**, or it resumes an old run's checkpoints. |

When in doubt about any number, grep the yaml:

```bash
grep -nE 'train_batch_size|total_training_steps|max_assistant_response_length|max_response_length' \
  vero2_scripts/config/<name>.yaml
```

## Launch workflow

**1. Smoke first** on gpu-test (A100, ≤1 h): `--qos=gpu-test` (never `-p`), `--constraint=gpu80`.

- `drawing/vanilla/run_gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_smoke.sh` — real config + CLI overrides `data.train_batch_size=8` `trainer.total_training_steps=2`
- `drawing/force_explore/run_..._answer_retry_forcedprefill_smoke.sh` — points the base at the bs-32/3-step debug256 retry config, plus the same bs-8/2-step CLI shrink
- `zoom/force_explore/run_..._answer_retry_forcedprefill_debug.sh` — runs the bs-32/3-step debug256 retry config **as-is** (no CLI shrink); its `#SBATCH` header is a historical pli-c one — adapt it to gpu-test before submitting (see `zoom/README.md`)

**2. Full run** — pick the cluster wrapper (drawing/force_explore; all `exec bash` the same base launcher):

| Wrapper | Cluster | Notes |
|---|---|---|
| `..._full_plilc.sh` | pli-lc H100, `--account=pli_xl` | **primary training target**; `save_freq=5` |
| `..._full_pli.sh` (+ `_pli_resume.sh`) | pli H100, `--account=vlmrl` | `save_freq=5` |
| `..._full_ailab.sh` | ailab H200, default account | guest jobs get **reaped** → `save_freq=1` |

All full wrappers/launchers use `--time=24:00:00` + a `%1` **self-resume chain**: one array task at
a time, each auto-resumes the latest checkpoint from the `CHECKPOINT_DATE_TIME`-pinned dir. Chain
lengths vary by launcher (`1-5%1` for drawing force_explore + zoomdraw + zoom force_explore;
`1-3%1` drawing vanilla; `1-2%1` zoom vanilla; some zoom experiments `1-1%1`) — check the header,
and extend with more tasks if the run needs longer. Wrappers pass extra Hydra args through, so data
overrides (Option B above) work at every layer.

**CONFIG_NAME:** the drawing base launchers honor `CONFIG_NAME="${CONFIG_NAME:-<default>}"` — export
it (as the smoke wrappers do) to point the same base at any yaml in `config/`. The zoom base
launchers hardcode their config name.

## Name traps

| Trap | Reality |
|---|---|
| `gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_debug256_cpueff.{yaml,sh}` | **THE FULL drawing-vanilla run** (batch 256, 1000 steps). "debug256" is historical residue, not a debug config. |
| `zoom/vanilla/run_..._cropzoom_debug256_cpueff.sh` (old-style debug) | yaml keeps the **full batch 256 / 400 steps**; the launcher makes it a debug run by overriding `data.train_files` to the 256-sample `chart_search_counting_visual_probe_train_debug256.verl.jsonl`. Small *data*, full *batch math*. |
| `..._debug256_cpueff_answer_retry_forcedprefill.yaml` (new-style debug, zoom + drawing) | genuinely small: **batch 32, 3 steps** baked into the yaml. |
| `force_explore/` vs `forced_explore` | folder `force_explore/` = answer-retry interruption; launcher name `forced_explore` (zoom/experiments) = older answer-token-ban mechanism. |
| `archive/run_gdpo_..._cropzoom_debug256_cpueff_hardonly.sh` | near-duplicate of the zoom/vanilla debug launcher — same config yaml, but overrides train data to the debug256 **hardonly** (leq05) subset; use the zoom/vanilla one unless you need that subset. |
| `archive/run_counting_reward_ablation.sh` | finished study (PR #45); see `docs/vero2/counting-reward-ablation.md`. |
| `zoom/vanilla` full is 16384 total-response, `zoom/force_explore` is 24576 | don't assume the two zoom configs share budgets — retries need headroom. |

Rule of thumb: **never trust the filename — grep the yaml** (command in the knob glossary).
