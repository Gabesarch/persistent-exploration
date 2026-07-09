# archive/ — cold storage

Superseded or finished launchers, kept verbatim. Nothing here is part of the 4 canonical
experiments (see [`../README.md`](../README.md)). Do not edit in place — revive (below) or leave alone.

## Archived launchers

| File | What it was | Why archived |
|---|---|---|
| `run_gspo_qwen3vl_visual_probe_cropzoom_cpueff.sh` | GSPO zoom-tool training on visual-probe data | Pre-GDPO algorithm (gspo trio); superseded by `zoom/vanilla/` GDPO launchers |
| `run_gspo_hdpotool_qwen3vl_visual_probe_cropzoom_cpueff.sh` | GSPO + hdpotool zoom variant | Pre-GDPO algorithm (gspo trio); superseded |
| `run_gspo_qwen3vl_visual_probe_cropzoom_debug256_cpueff.sh` | Debug run of the GSPO zoom setup | Pre-GDPO algorithm (gspo trio); superseded |
| `run_gspo_chart_qwen3vl_base.sh` | No-tool GSPO baseline (plain chart QA, no zoom tool) | Baseline study over; pre-GDPO |
| `run_gspo_qwen3vl_visual_probe_debug256.sh` | No-tool GSPO baseline on 256-sample debug visual-probe data (same `gspo_chart_qwen3vl_base` config, debug data overrides) | Debug companion of the no-tool baseline; pre-GDPO |
| `run_valonly_vero_qwen3i_visual_probe.sh` | Eval-only (val-only) run for the old vero qwen3i model | Utility for a retired model |
| `run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_cropzoom_debug256_cpueff_hardonly.sh` | GDPO zoom debug launcher | Near-duplicate: selects the exact same config yaml as the live `zoom/vanilla/` debug launcher but overrides train data to the debug256 **hardonly** (leq05) subset; use the zoom/vanilla one unless you need that subset |
| `run_counting_reward_ablation.sh` | Graded-vs-exact counting-reward ablation sweep | Study completed — results in `docs/vero2/counting-reward-ablation.md`, merged as PR #45 |

## config/archive/ — configs used only by archived launchers

Six yamls live in `../config/archive/`:

| Config yaml | Used by |
|---|---|
| `gspo_chart_qwen3vl_base.yaml` | nothing directly — `run_gspo_chart_qwen3vl_base.sh` and `run_gspo_qwen3vl_visual_probe_debug256.sh` read the **duplicate copy** in `examples/gaze_rl/config/` (see below) |
| `gspo_chart_qwen3vl_visual_probe_cropzoom_cpueff.yaml` | `run_gspo_qwen3vl_visual_probe_cropzoom_cpueff.sh` |
| `gspo_chart_qwen3vl_visual_probe_cropzoom_debug256_cpueff.yaml` | `run_gspo_qwen3vl_visual_probe_cropzoom_debug256_cpueff.sh` |
| `gspo_chart_qwen3vl_visual_probe_cropzoom_debug256.yaml` | nothing (orphan — no launcher references it) |
| `gspo_hdpotool_chart_qwen3vl_visual_probe_cropzoom_cpueff.yaml` | `run_gspo_hdpotool_qwen3vl_visual_probe_cropzoom_cpueff.sh` |
| `valonly_vero_qwen3i_visual_probe.yaml` | `run_valonly_vero_qwen3i_visual_probe.sh` |

`run_counting_reward_ablation.sh` is the exception: its default config
(`gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_cpueff_hardonly`) is the live zoom/vanilla
config and still sits in `../config/`.

## Reviving an archived launcher

Launchers resolve configs **by name** via `--config-path=$VERL_ROOT/vero2_scripts/config`, so an
archived launcher will not find its yaml until you put it back:

```bash
cp vero2_scripts/config/archive/<name>.yaml vero2_scripts/config/
sbatch vero2_scripts/archive/<launcher>.sh
```

Exception: `run_gspo_chart_qwen3vl_base.sh` and `run_gspo_qwen3vl_visual_probe_debug256.sh` use
`--config-path=$VERL_ROOT/examples/gaze_rl/config` (a pre-vero2 location that still exists), so
they run as-is with no copy step.

Also sanity-check the Slurm header against `useful_command.sh` §3 — these scripts predate the
current account/partition conventions.
