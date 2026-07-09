# `vero2_scripts/drawing/` — drawing-tool experiments (counting data)

Multi-turn **drawing-tool** RL on Qwen3-VL-8B over counting data: the model reasons in
`<think>`, annotates the image with drawing tools (`draw_points` / `draw_lines` / `draw_boxes` /
`draw_texts` / `draw_polygons`, 0–1000 normalized coordinates), gets each annotated image fed
back as a new `img_idx`, and answers in `<answer>`. The draw-only toolset is
[`config/tool_config/image_drawing_tools.yaml`](../config/tool_config/image_drawing_tools.yaml)
(tool classes `verl.tools.image_drawing_tools.*`); the `experiments/` zoomdraw variant uses the
combined [`config/tool_config/image_zoom_in_draw_tool.yaml`](../config/tool_config/image_zoom_in_draw_tool.yaml)
(zoom + the five draw tools). Trained with GDPO advantage
(`accuracy/format/tool_efficiency` = 0.8/0.1/0.1) + GSPO loss via `python -m verl.trainer.main_ppo`.

All configs live **flat** in [`../config/`](../config/) (launchers pass
`--config-path=$VERL_ROOT/vero2_scripts/config`). Always launch from the **repo root**
(launchers `source ./set_paths.sh` relatively). Workflow: **smoke on gpu-test first, then full**
(see the hub [`../README.md`](../README.md) and `useful_command.sh` §3/§7).

---

## 1. `vanilla/` — plain drawing run (no interruption)

### Full run — NAME TRAP, read this first

> **WARNING — "debug256" in the name is historical, this IS the full run.**
> `vanilla/run_gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_debug256_cpueff.sh` and its config
> `gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_debug256_cpueff.yaml` are the **full production
> config**: `train_batch_size=256`, `ppo_mini_batch_size=128`, `total_training_steps=1000`.
> Nothing about it is a 256-sample debug run. (Contrast with the zoom family, where
> "debug256" really does mean debug data.)

```bash
sbatch vero2_scripts/drawing/vanilla/run_gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_debug256_cpueff.sh
```

Header: `--partition=pli --account=vlmrl`, 8 GPUs, 64 CPUs, 968G, 24 h wall,
`--array=1-3%1` self-resume chain. `CHECKPOINT_DATE_TIME` (default `20260524_160610`) pins the
checkpoint dir so array tasks resume each other — **export a fresh `CHECKPOINT_DATE_TIME` for a
new run** or you resume the old one. `CONFIG_NAME` is env-overridable (default
`gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_debug256_cpueff`). Retry/force-explore: **OFF**
(no `conditional_answer_retry_*` block in the config).

### Dataset — must be materialized once per checkout

The vanilla config trains on `${VERL_ROOT}/datasets/combined_counting_hq/train_hard_answer_lt10.verl.jsonl`
(val: `combined_counting_val_answer_lt10.verl.jsonl`) — **NOT in git**. Materialize it once per checkout:

```bash
mkdir -p "$VERL_ROOT/datasets" && ln -sfn /scratch/gpfs/ZHUANGL/hw0171/vero2/datasets/combined_counting_hq "$VERL_ROOT/datasets/combined_counting_hq"
```

(The readable copy lives in hw0171 scratch; gs1693's copy is permission-locked. Image paths inside
the jsonl are absolute into hw0171 scratch — consider a real local copy for long-term runs.)
The smoke wrapper fail-fasts with this exact fix if the symlink is missing.

### Smoke

```bash
sbatch vero2_scripts/drawing/vanilla/run_gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_smoke.sh
```

`run_..._drawing_smoke.sh` execs the full launcher with the **same config** and only shrinks on
the CLI: `data.train_batch_size=8`, `ppo_mini_batch_size=8`, `rollout.n=8`,
`trainer.total_training_steps=2`, save/test off. Header: `--qos=gpu-test`, 4x A100-80GB
(`--gres=gpu:a100:4 --constraint=gpu80`), 59 min. So the smoke exercises exactly the settings the
full run trains with.

### Changing the training data (vanilla-specific gotcha)

Every launcher forwards extra Hydra args, and **the last override wins**:

```bash
sbatch vero2_scripts/drawing/vanilla/run_..._drawing_debug256_cpueff.sh vllm \
  data.train_files=/abs/path/train.jsonl data.val_files=/abs/path/val.jsonl
```

Note: the **vanilla launcher itself re-passes** `data.train_files`/`data.val_files`
(pointing at `$VERL_ROOT/datasets/combined_counting_hq/`) on every invocation, so for this
family **editing the yaml's data block alone does nothing** — use the CLI override above (your
args come after the launcher's and win) or edit the launcher's `DEBUG_TRAIN_DATA`/`DEBUG_VAL_DATA`.
The force_explore launchers do *not* re-pass data paths, so there either editing the yaml or the
CLI override works.

---

## 2. `force_explore/` — conditional answer-retry interruption

### What the interruption does (mechanics)

"Force-explore" here = the **conditional answer-retry** rollout interruption
(implemented in `verl/experimental/agent_loop/tool_agent_loop.py`):

1. The model produces an `<answer>`; the rollout grades it in-flight with the reward function.
2. If `accuracy > conditional_answer_retry_correct_accuracy_threshold` (**0.0** → retry fires
   only on accuracy == 0, i.e. a wrong answer), the answer stands.
3. Otherwise the wrong answer turn is **rolled back** — `prompt_ids`/`response_mask`/logprobs are
   truncated to the pre-answer snapshot, so the wrong `<answer>` is deleted from the trajectory.
4. `conditional_answer_retry_force_next_tool_call: true` forces the next generation to be a
   `<tool_call>` (another drawing action), after which the model re-answers.
5. The **`ANSWER_RETRY_ASSISTANT_PREFILL`** env var is the assistant prefill text injected at the
   retry point (wired via Hydra
   `actor_rollout_ref.rollout.multi_turn.conditional_answer_retry_assistant_prefill=${oc.env:ANSWER_RETRY_ASSISTANT_PREFILL}`).
   The injected prefill tokens are appended with `response_mask=0`, i.e. **loss-masked** — the
   model never gets gradient on text it didn't sample.
6. `conditional_answer_retry_max_retries: 2` bounds retries per trajectory (full config;
   the debug256 smoke config leaves it `null` = unbounded).

   **Deliberate asymmetry vs zoom:** the zoom force_explore config runs retries **uncapped**
   (`max_retries: null`, limited only by turn/token budget) while drawing is bounded at **2**.
   Current plan: test drawing with the bounded setting first; switching to uncapped later is a
   one-line change (`...conditional_answer_retry_max_retries=null` on the CLI after `vllm`, or
   edit the yaml).

Base-launcher default prefill (drawing-worded, override by exporting the env var):

> "Let's see, I do not have enough information to answer yet. My current annotations may be
> inaccurate or incomplete — let me use the drawing tools to correct and complete my markings,
> then re-examine the image before answering."

**Terminology caution:** this is distinct from `zoom/experiments`' older `forced_explore`
(ban-the-answer-token) and `text_explore` (prompt nudge) launchers.

### Config and data

Full config: `gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill.yaml`
— batch 256 / mini 128 / 1000 steps / per-turn cap 4096, same profile as vanilla (see section 5).
Data (baked into the config, no symlink needed): the Gemini-filtered **correct-only** counting set

```
/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync/filtered_correct_only/combined_counting_train.correct.partial.verl.jsonl
```

(val: `combined_counting_val.correct.partial.verl.jsonl` in the same dir).

### Launching: base launcher + `submit.sh` (partition = a runtime decision)

`run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_full.sh`
is the **base launcher** (a generic `main_ppo` runner): `CONFIG_NAME`-overridable, exports the
default `ANSWER_RETRY_ASSISTANT_PREFILL`, builds the checkpoint dir from
`CHECKPOINT_DATE_TIME` (default `20260524_160610` — new run → new value) and
`TRAINER_EXPERIMENT_SUFFIX`, and forwards all extra Hydra args. Its header is a pli-lc profile
(H100 x8, 96 CPUs, 960G, 24 h, `--array=1-5%1`), but you normally submit through **`submit.sh`**,
which picks the cluster at submit time (sbatch CLI flags override the `#SBATCH` header — no
per-partition files needed):

```bash
bash vero2_scripts/drawing/force_explore/submit.sh              # pli-lc (default, primary)
bash vero2_scripts/drawing/force_explore/submit.sh pli          # pli (vlmrl) — best allocation odds
bash vero2_scripts/drawing/force_explore/submit.sh ailab        # ailab H200 — reap-safe save_freq=1
bash vero2_scripts/drawing/force_explore/submit.sh plilc data.train_files=/abs/train.jsonl   # + overrides
```

| target | partition / account / GPUs | save_freq | notes |
|---|---|---|---|
| `plilc` | `pli-lc` / `pli_xl` / h100:8, 96 CPU, 960G | 5 | primary target (stable, PreemptMode=OFF); save_freq=5 because a 24 h task can't reach step 20, so save_freq=20 would make every resume restart from scratch |
| `pli` | `pli` / `vlmrl` / h100:8, 96 CPU, 960G | 5 | 38-node partition, better allocation odds |
| `ailab` | `ailab` / `zhuangl` (default acct) / h200:8, 64 CPU, 1450G | **1** | guest jobs get **reaped** (one run killed at ~5.75 h) → checkpoint **every step**; the `%1` chain resumes |

`submit.sh` extras (see its header for examples):
- **Fresh run:** `CHECKPOINT_DATE_TIME=<new> bash submit.sh …` (else it resumes the old checkpoint dir).
- **Resume a plilc run on pli** (the old `_pli_resume` behavior): set `TRAINER_EXPERIMENT_SUFFIX`
  to the plilc-tagged suffix before submitting to `pli` — same checkpoint dir → `resume_mode:auto`
  continues it. **Clobber hazard:** never run concurrently with a pli-lc array task on the same dir.
- **Rollout caching speedups** (inference-only; the old `_pli` wrapper had them baked in):
  `ROLLOUT_CACHING=1 bash submit.sh <target>` adds `enable_prefix_caching=true`,
  `mm_processor_cache_gb=4`, `max_num_batched_tokens=16384`.

All targets: 24 h wall + `--array=1-5%1` self-resume chain (from the base header).

### Smoke

```bash
sbatch vero2_scripts/drawing/force_explore/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill_smoke.sh
```

gpu-test, 4x A100-80GB, 59 min. Sets
`CONFIG_NAME=gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_drawonly_debug256_cpueff_answer_retry_forcedprefill`
(a bs32/3-step debug config — this one really is a debug config) then shrinks further on the CLI
to bs8 / 2 steps, with comprehensive debug logging on (traces + images under
`${TRAINER_DEFAULT_LOCAL_DIR}/debug_logs`). It exports its own draw-worded retry prefill and an
isolated `drawonly_arf_smoke_<jobid>` checkpoint dir (fresh dir → no accidental resume).

---

## 3. `experiments/` — variants and benches

| launcher | what it is |
|---|---|
| `run_..._zoomdraw_..._answer_retry_forcedprefill_full.sh` | **Combined zoom+draw toolset** (`tool_config/image_zoom_in_draw_tool.yaml`: `image_zoom_in_tool` + all five draw tools), answer-retry ON. **Why it exists:** a zoom-trained checkpoint reflexively calls `image_zoom_in_tool`; with a draw-only tool list that call is unknown and the trajectory terminates (`tool_agent_loop.py` `_call_tool`: the unknown tool name raises `KeyError`, and the handler returns a terminal error `ToolResponse` for `image_zoom_in_tool`) — registering zoom alongside draw keeps those calls productive while adding drawing. Trains on gs1693's chart/search/counting visual-probe hardonly data (not counting-only), config `..._zoomdraw_cpueff_hardonly_answer_retry_forcedprefill.yaml`: **batch 128 / mini 64 / 400 steps / no per-turn cap / 24576 response** (zoom-family profile), draw-aware system prompt `prompts/system_prompt_chatting_tool_calling_draw.txt`, retry prefill defaults to a "annotate ... (or zoom in)" wording. Note: its header says `--partition=pli-c` with no account — verify against `useful_command.sh` §3 before submitting. |
| `run_..._zoomdraw_..._answer_retry_forcedprefill_smoke.sh` | gpu-test smoke for the above: `CONFIG_NAME=..._zoomdraw_debug256_cpueff_answer_retry_forcedprefill` (bs32/3-step debug config) shrunk to bs8 / 2 steps, debug logging on. |
| `run_drawonly_fast_bench_smoke.sh` | Perf bench of the **FAST** drawonly config (per-turn cap 4096 + bounded retries): explicitly sets `CONFIG_NAME` to the hardonly retry config, bs16 / mini 16 / 2 steps on 4x A100 gpu-test; times a step to extrapolate to 8x H100 / bs256. |
| `run_drawonly_speed_bench_smoke.sh` | Same base launcher and (default) hardonly retry config, bs16 / 2 steps, **plus** the vLLM caching levers (`enable_prefix_caching=true`, `mm_processor_cache_gb=4`, `max_num_batched_tokens=16384`) — de-risks the caching flags used by the `_pli` wrapper and times a step. |

Both benches run over the `force_explore/` base launcher.

---

## 4. Drawing-specific knobs

The historically buried speed lever of this family is the **per-turn output cap**:

- `actor_rollout_ref.rollout.multi_turn.max_assistant_response_length: 4096` — each assistant
  turn is clamped to 4096 generated tokens, so no single turn can eat the whole trajectory
  budget. Present in **both** canonical drawing configs; **absent** in the zoom-family configs
  (and in the debug256/zoomdraw configs here), where a turn can run to the full budget.
- `data.max_response_length: 16384` — total response budget per trajectory (drawing canonical
  configs; the debug256 + zoomdraw configs use 24576).
- `actor_rollout_ref.rollout.multi_turn.max_assistant_turns: 11` (with `max_user_turns: 10`) —
  everywhere in this family.
- `actor_rollout_ref.rollout.n: 8` — rollouts per prompt (GRPO group) everywhere;
  trajectories/step = `train_batch_size` x 8.
- gradient updates/step = `train_batch_size / ppo_mini_batch_size`.

| config (in `../config/`) | used by | batch | mini | steps | per-turn cap | total resp | retry |
|---|---|---|---|---|---|---|---|
| `gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_debug256_cpueff` | vanilla full + smoke | 256 | 128 | 1000 | 4096 | 16384 | OFF |
| `..._visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill` | force_explore full + wrappers + benches | 256 | 128 | 1000 | 4096 | 16384 | ON, max_retries 2 |
| `..._visual_probe_drawonly_debug256_cpueff_answer_retry_forcedprefill` | force_explore smoke | 32 | 32 | 3 | none | 24576 | ON, max_retries null |
| `..._visual_probe_zoomdraw_cpueff_hardonly_answer_retry_forcedprefill` | experiments zoomdraw full | 128 | 64 | 400 | none | 24576 | ON, max_retries null |
| `..._visual_probe_zoomdraw_debug256_cpueff_answer_retry_forcedprefill` | experiments zoomdraw smoke | 32 | 32 | 3 | none | 24576 | ON, max_retries null |

(All numbers read from the yamls; smokes shrink batch/steps further on the CLI as described above.)

---

## 5. Vanilla vs force_explore: the config diff

`diff gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_debug256_cpueff.yaml \
     gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_drawonly_cpueff_hardonly_answer_retry_forcedprefill.yaml`
shows the two canonical configs are **identical in training math and batch profile**
(GDPO/GSPO, batch 256 / mini 128 / 1000 steps, per-turn cap 4096, response 16384, 11 turns,
lr 1e-6, same reward stack, save_freq 20 / test_freq 40). The only differences:

1. **Dataset** — vanilla: `$VERL_ROOT/datasets/combined_counting_hq/train_hard_answer_lt10.verl.jsonl`
   (symlink, section 1); force_explore: the Gemini-filtered correct-only set in lc1556 scratch
   (section 2).
2. **Retry block** — force_explore adds `force_tool_calls_until_final_turn: false`,
   `final_answer_token_margin: 2048`, and the `conditional_answer_retry_*` block
   (`enabled: true`, `force_next_tool_call: true`, `suffix: ""`,
   `correct_accuracy_threshold: 0.0`, `max_retries: 2`); vanilla has none of it.
3. **Debug logging detail** — vanilla saves tool images (`save_cropped_images: true`),
   force_explore doesn't (`false`).

Same tool config (`image_drawing_tools.yaml`), same system prompt
(`prompts/system_prompt_chatting_tool_calling.txt`), same model
(`$HF_HOME/Qwen/Qwen3-VL-8B-Instruct`).
