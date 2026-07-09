# zoom/ — image_zoom_in_tool experiments

The zoom family trains Qwen3-VL-8B to reason in `<think>`, call **`image_zoom_in_tool`**
(a bbox crop: `bbox_2d` in normalized 0–1000 coordinates + `label` + `img_idx`; the crop is
fed back into the conversation as a new image), and answer in `<answer>` — GDPO advantage +
GSPO loss, async multi-turn rollout (see `docs/vero2/training-pipeline.md`).

**Data** is the chart / search / counting **visual-probe** set
(`/scratch/gpfs/ZHUANGL/gs1693/vero2_datasets/train_rl_datasets/`):

| tag | train file | meaning |
|---|---|---|
| *hardonly* | `chart_search_counting_visual_probe_train_rollout_score_leq05_all.verl.jsonl` | hard subset — examples whose pre-run rollout score was ≤ 0.5 (`leq05`) |
| full data | `chart_search_counting_visual_probe_train.verl.jsonl` | unfiltered train set (used only by `experiments/run_..._cropzoom_cpueff.sh`) |
| val | `chart_search_counting_visual_probe_val.verl.jsonl` | shared val set |

> **Terminology trap:** `force_explore/` here = the **conditional answer-retry** interruption
> (wrong `<answer>` is deleted, model is forced into another tool call, then re-answers;
> rolled-back text is loss-masked). `experiments/` also contains launchers literally named
> `forced_explore` — that is an **older, different** mechanism (tool calls forced every turn,
> answering banned until the final turn). Don't confuse them.

## Layout

```
zoom/
  vanilla/         full run (retry OFF) + old-style debug256 launcher
  force_explore/   answer-retry-interruption run (retry ON), full + new-style debug
  experiments/     one-off variants: text_explore, forced_explore, answer_retry_explore,
                   llmjudge, multiidx, non-hardonly full-data, valonly hstarbench
```

## Shared launcher conventions (read once)

- **Launch from the repo root** (`sbatch vero2_scripts/zoom/<dir>/<launcher>.sh`) — every
  launcher sources `./set_paths.sh` by relative path and reads configs from
  `--config-path=$VERL_ROOT/vero2_scripts/config`.
- **First positional arg is the inference engine** (default `vllm`). Any first arg that
  doesn't start with `-` is consumed as the engine, so when passing Hydra overrides you must
  say `vllm` explicitly first:
  `sbatch vero2_scripts/zoom/vanilla/run_..._hardonly.sh vllm data.train_files=/abs/train.jsonl`
- **`CHECKPOINT_DATE_TIME`** (default `20260524_160610` in every launcher) pins the checkpoint
  dir `${SAVE_PATH}/checkpoints/vero2_verl/<ALGORITHM_TAG>_<SUFFIX>_<DATE_TIME>` — the
  `--array=…%1` chain self-resumes from it. **A new run needs a new
  `CHECKPOINT_DATE_TIME`** or you resume (or clobber) the old one.
- **SBATCH headers are historical**: the active headers target `--partition=pli-c`
  (the original author's allocation; full runs: 1 node, `gpu:8`, 64 CPUs, 968G, 24 h).
  Adapt the header to your own targets per `useful_command.sh` §3 (gpu-test smoke /
  pli-lc full) — each launcher carries a commented-out gpu-test block as a template.
- **Stale dependency:** `vanilla/run_..._cropzoom_cpueff_hardonly.sh` still has
  `#SBATCH --dependency=afterany:9252967_3` from an old run — delete it before submitting.
- **Exception:** `experiments/run_valonly_hstar_qwen3vl_cropzoom.sh` follows its own rules —
  no `CHECKPOINT_DATE_TIME` (it timestamps a fresh output dir per run, no resume chain), the
  engine arg is consumed only if it is literally `vllm`/`sglang`/`trtllm`, and it has no
  gpu-test header template. The conventions above apply to the *training* launchers.

## vanilla/ — retry OFF

| launcher | config (`config/`) | batch / mini | steps | resp budget | data |
|---|---|---|---|---|---|
| `run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_cropzoom_cpueff_hardonly.sh` (full, `--array=1-2%1`) | `gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_cpueff_hardonly.yaml` | 256 / 128 | 400 (save_freq 40) | 16384 | hardonly |
| `run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_cropzoom_debug256_cpueff.sh` (debug, `--array=1-3%1`) | `gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_debug256_cpueff.yaml` | 256 / 128 | 400 | 16384 | debug-256 subset |

Both: rollout `n: 8` (2048 trajectories/step for batch 256), `max_prompt_length: 16384`,
`max_assistant_turns: 11`, GDPO reward keys `[accuracy, format, tool_efficiency]`
weights `[0.8, 0.1, 0.1]`.

**Old-style debug convention** (different from drawing's smoke-override style): the
`debug256` **config keeps the full training math** — batch 256, mini 128, 400 steps, full
node — and the *launcher* swaps in a **256-sample data subset** by CLI override
(`data.train_files=…/chart_search_counting_visual_probe_train_debug256.verl.jsonl`, same for
val). It's a "full-shaped run on tiny data", not a 2-step smoke. Ignore the dataset paths
inside the debug256 yaml itself (they point at an obsolete vigorl preprocessed set) — the
launcher override wins. The debug launcher also exposes the tool-efficiency reward knobs as
env vars (`TOOL_EFFICIENCY_REWARD_MODE=concave_length_penalty`, `_FREE_CALLS=1`,
`_CONCAVE_K=10.0`, `_CONCAVE_Q=0.95`).

## force_explore/ — the answer-retry interruption (retry ON)

When a rollout emits `<answer>` and the graded accuracy is ≤
`conditional_answer_retry_correct_accuracy_threshold` (**0.0** — i.e. only outright-wrong
answers), the answer turn is rolled back, an assistant **prefill** is injected
(env `ANSWER_RETRY_ASSISTANT_PREFILL`, default *"Let's see, I do not have enough information
to answer and need to continue exploring with additional tool calls."*), and the model is
**forced into another tool call** (`conditional_answer_retry_force_next_tool_call: true`)
before re-answering. Rolled-back text is loss-masked. In the zoom configs
`conditional_answer_retry_max_retries: null` (unbounded — limited only by the turn/token
budget) and `conditional_answer_retry_suffix: ""` (prefill-only, no user-side nudge).

| launcher | config | batch / mini | steps | resp budget |
|---|---|---|---|---|
| `run_..._cropzoom_cpueff_hardonly_answer_retry_forcedprefill_full.sh` (full, `--array=1-5%1`) | `..._cropzoom_cpueff_hardonly_answer_retry_forcedprefill.yaml` | **128 / 64** | 400 (save_freq 20) | **24576** |
| `run_..._answer_retry_forcedprefill_debug.sh` (debug, `gpu:2`, 1 h) | `..._cropzoom_debug256_cpueff_answer_retry_forcedprefill.yaml` | **32 / 32** | **3** (save_freq -1) | 24576 |

- **Batch 128 (vs vanilla's 256) is historical, for rollout memory**: retry trajectories are
  longer (response budget 24576 vs 16384, `max_model_len` 40960 vs 32768), so the batch was
  halved. Reward keys here are `[accuracy, format]` `[0.9, 0.1]` (no tool_efficiency term).
- **Debug is new-style**: a genuinely small config (batch 32, 3 steps), not a full-shaped
  run. The launcher overrides data to the debug-256 hardonly subset
  (`…train_debug256_rollout_score_leq05.verl.jsonl` train / `…train_debug256.verl.jsonl` val),
  defaults `VERO2_COMPREHENSIVE_DEBUG_LOGGING=1`, and — unless
  `RUN_ANSWER_RETRY_SMOKE_CHECK=0` — runs a **post-run smoke check** over
  `debug_logs/…/sample_traces.jsonl` asserting every retry is followed by a
  `<tool_call>` turn with no premature `<answer>`.

## experiments/ — one-off variants

All are cropzoom + hardonly data + 400 steps unless noted. `*_debug.sh` counterparts use the
new-style debug256 configs (batch 32/mini 32, 3 steps, resp 24576).

| launcher (full) | one line |
|---|---|
| `..._text_explore_full.sh` (+debug) | **Prompt-nudge exploration**: appends an "explore more / DFS+BFS zoom" suffix to tool responses for the first 10 turns (`tool_response_turn_suffix_enabled`), then a "now answer" suffix. Batch 256/128, resp 24576. |
| `..._forced_explore_full.sh` (+debug) | **Older ban-answer-token forcing** — `force_tool_calls_until_final_turn: true` + `force_final_answer_prefix: "\n\boxed{"`: tool calls are forced every turn, answering only allowed on the final turn. **NOT the same as `force_explore/`.** Batch 128/64, resp 24576. |
| `..._answer_retry_explore_full.sh` (+debug) | **Earlier retry flavor**: same conditional answer-retry, but with a long "keep exploring, test alternative hypotheses" `conditional_answer_retry_suffix` message in addition to the prefill. Batch 128/64, save_freq 40. |
| `..._answer_retry_llmjudge_forcedprefill_full.sh` | **LLM-judge-graded retry**: forcedprefill retry + `llm_judge_enabled: true` — an LLM judge (prompt `prompts/llm_judge_visual_search_prompt.txt`) grades `search`-type answers with partial credit 0.5, falling back to string reward. Batch 128/64. |
| `..._multiidx.sh` | **Multi-image-index zoom variant** — *intended* to let the model address previous crops as new `img_idx` values via `image_indexing: multi_index` in `image_zoom_in_tool_multiidx.yaml`. ⚠️ That key is currently **unwired in the training tool** (see box below), and the yaml is byte-identical to the base one, so in practice this differs from vanilla only by **1000 steps** (vs 400). Batch 256/128, resp 16384. |
| `run_..._cropzoom_cpueff.sh` | **Full-data (non-hardonly) vanilla**: identical shape to vanilla full (256/128, 400 steps, resp 16384) but trains on the unfiltered `…visual_probe_train.verl.jsonl`. |
| `run_valonly_hstar_qwen3vl_cropzoom.sh` | **Eval-only hstarbench validation**: `trainer.val_only=true`, 1 step, rollout `n: 1`, data `/scratch/gpfs/ZHUANGL/shared/Hstarbench/data/verl/hstarbench_mixed_rl.verl.jsonl`. No training. |

> **The `image_indexing: multi_index` key is currently INERT in training.**
> `config/tool_config/image_zoom_in_tool.yaml` is byte-identical to
> `image_zoom_in_tool_multiidx.yaml` (the key was added to the base file in `a436672`), but the
> training-side `ImageZoomInTool` only validates `image_indexing` in `__init__` and **never reads
> it again** — the tool response wording is always the training-faithful *"Zoomed in on the image
> to the region …"* with no new-index announcement. Training behavior is unchanged either way:
> **crops are always cut from the original-resolution source image**
> (`use_original_resolution_source: true` + a provenance store maps every crop back to
> root-original coordinates, so even a zoom into a prior crop re-composites from the original —
> no compounding resolution loss). The knob is only meaningful in `vero_eval`'s `*_multiidx`
> eval tasks. Net: the "multiidx experiment" below currently differs from vanilla only by its
> step count, since its tool yaml is identical and the key is unwired.

## Changing the training data

- **Option A** — edit `data.train_files` / `data.val_files` in the config yaml.
- **Option B** (no file edits) — pass Hydra overrides through the launcher, with the engine
  named explicitly first:
  ```bash
  sbatch vero2_scripts/zoom/vanilla/run_..._cropzoom_cpueff_hardonly.sh \
      vllm data.train_files=/abs/path/train.jsonl data.val_files=/abs/path/val.jsonl
  ```
- **Caveat for the two debug launchers** (`vanilla/…debug256…` and
  `force_explore/…_debug.sh`): they re-inject their own `data.train_files`/`data.val_files`
  overrides on every run, so editing the yaml's data block alone does nothing. CLI overrides
  passed after `vllm` still win (user args come last; Hydra last-override-wins) — or edit
  the `DEBUG_DATA` / `DEBUG_TRAIN_DATA` / `DEBUG_VAL_DATA` variables inside the launcher.

## Zoom-specific knob notes

- **No per-turn output cap in any zoom config.**
  `actor_rollout_ref.rollout.multi_turn.max_assistant_response_length` is absent from every
  zoom yaml, so a single assistant turn can run to the full remaining response budget
  (16384 vanilla / 24576 explore-retry variants). Both drawing configs cap it at 4096. **If
  zoom steps are slow, adding
  `actor_rollout_ref.rollout.multi_turn.max_assistant_response_length=4096` is the known
  speed lever** — pass it on the command line (after `vllm`) without editing yaml.
- `max_assistant_turns: 11` and rollout `n: 8` in every zoom training config;
  trajectories/step = batch × 8, gradient updates/step = batch / ppo_mini (= 2 in every
  full config; the bs32/mini32 debug configs do 1).
- `max_prompt_length: 16384` in all zoom training configs.
- Reward composition differs by variant: vanilla/multiidx/full-data include the
  `tool_efficiency` GDPO key (`[0.8, 0.1, 0.1]`); the explore/retry variants drop it
  (`[accuracy, format]` = `[0.9, 0.1]`).
