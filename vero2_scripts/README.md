# vero2_scripts

The vero2 RL experiments: launcher scripts (`run_*.sh`), Hydra configs, tool schemas, and system
prompts. Everything here drives `python -m verl.trainer.main_ppo` on Della Slurm.

> **Always launch from the repo root** (`sbatch vero2_scripts/<folder>/run_*.sh`). The launchers do
> `cd "$SLURM_SUBMIT_DIR"` then `source ./set_paths.sh` via relative paths, so the submit dir must be
> the repo root. See the top-level `CLAUDE.md` and `useful_command.sh` for cluster/Slurm details.

## Layout

| Path | What |
|---|---|
| **`zoom/`** | Launchers for the **zoom-tool** experiments (`image_zoom_in_tool`) — filenames contain `cropzoom`. The model crops/zooms into a bbox. |
| **`counting/`** | Launchers for the **draw-tool** experiments on counting data — `drawonly` (draw only), `zoomdraw` (zoom + draw), `countqa`. The model annotates the image with points/boxes/lines. |
| `config/` | **All Hydra configs** (`*.yaml`), shared by both families. Launchers select one by name via `--config-name`; the dir is referenced by the absolute `${VERL_ROOT}/vero2_scripts/config` path, so configs are **not** split per-folder. |
| `config/tool_config/` | Tool JSON schemas (`image_zoom_in_tool.yaml`, `image_drawing_tools.yaml`, `image_zoom_in_draw_tool.yaml`, …). Referenced by absolute path from inside the configs. |
| `prompts/` | System prompts referenced by absolute path from the configs. |
| `debug_scripts/` | Trajectory / crop / validation visualizers (not launched by the run scripts). |
| top-level `run_*.sh` | Cross-cutting launchers that are neither zoom nor draw: `run_counting_reward_ablation.sh` (counting-**reward** ablation — runs on a `cropzoom` config), `run_gspo_chart_qwen3vl_base.sh`, `run_gspo_qwen3vl_visual_probe_debug256.sh`, `run_valonly_*.sh`. |

## Launcher naming

`run_<algo>_<model>_visual_probe_<tool>_..._<variant>_<size>.sh`

- **tool**: `cropzoom` → `zoom/`; `drawonly` / `zoomdraw` / `countqa` → `counting/`.
- **variant**: `answer_retry_forcedprefill`, `forced_explore`, `text_explore`, `llmjudge`, `hardonly`, `multiidx`, …
- **size / role**: `full` = real training · `debug` = small subset · `smoke`/`bench_smoke` = quick start+timing check.

Many launchers are **thin wrappers** that `exec bash` a base `..._full.sh` launcher with a different
`#SBATCH` header + a `CONFIG_NAME` override. Example chain (`counting/`):

```
run_..._drawonly_..._full_plilc.sh   # pli-lc H100 header + CONFIG_NAME override
  └─ exec bash counting/run_..._drawonly_..._full.sh   # the base: builds args, calls main_ppo
       └─ --config-path=$VERL_ROOT/vero2_scripts/config  --config-name=$CONFIG_NAME
```

The base launchers honor `CONFIG_NAME="${CONFIG_NAME:-<default>}"`, so a wrapper (or your shell) can
point the same launcher at any config in `config/` without editing the base.

## Debug vs. full configs — the convention

Each experiment has a **full** training config and a smaller **debug** one for fast iteration. Watch
out: `debug256` in a *name* does **not** reliably mean small — there are two generations.

| Kind | How to recognize | batch / steps |
|---|---|---|
| **Full** (real training) | no `debug` in the name | `train_batch_size` 128–256, `total_training_steps` 400–1000 |
| **Debug — new-style** | `debug256` config with baked-in small values | batch **32**, steps **3** |
| **Debug — old-style** | `debug256` config keeps full batch/steps in the yaml, but the paired `run_..._debug256_*.sh` launcher overrides `data.train_files=$DEBUG_DATA` (the 256-sample subset `..._train_debug256.verl.jsonl`) | full batch in yaml, small data via launcher |
| **Smoke** | any config + a `_smoke` / `_bench_smoke` wrapper | wrapper additionally overrides `data.train_batch_size=8/16 trainer.total_training_steps=2` on the CLI |

⚠️ **Name traps:** `gdpo_hdpotool_gspo_countqa_qwen3vl_drawing_debug256_cpueff.yaml` is a **full**
config (batch 256, steps 1000) despite the `debug256` in its name — it's the counting hard subset,
not a debug run. When in doubt, grep the config:

```bash
grep -E 'train_batch_size|total_training_steps' vero2_scripts/config/<name>.yaml
```

## Typical workflow

1. **Smoke first** (gpu-test A100, ~1 h cap, small batch, 2 steps) — confirms the config starts and
   times one step. e.g. `sbatch vero2_scripts/counting/run_drawonly_fast_bench_smoke.sh`.
2. **Full run** on the real target — pick the wrapper for your cluster:
   - `..._full_plilc.sh` → pli-lc H100 (`--account=pli_xl`, primary training target)
   - `..._full_pli.sh` → pli H100 (`--account=vlmrl`, 38 nodes, best allocation odds)
   - `..._full_ailab.sh` → ailab H200 (default account; reaps guest jobs → `save_freq=1`)
   - Full-run wrappers use `--array=1-N%1` (a **resume chain**: one task at a time, each auto-resumes
     the last checkpoint from a fixed date-stamped dir) with `--time=24:00:00`.

See `useful_command.sh` §7 for the exact copy-paste training + smoke commands, and `docs/vero2/` for
the training-pipeline / environment deep dives.
