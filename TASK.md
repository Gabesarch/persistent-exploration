# TASK: Counting-reward ablation (offline-first)

Branch: `counting-reward-ablation` · Worktree: `/home/lc1556/repo/reason-vlm/vero2-counting-reward-ablation`
Written 2026-07-02 by the coordinating session (facts below verified against the repo on that date).

## Context

PR #40 (commit `9ac5e40`, on `main`) replaced the counting reward — previously an exact match — with a **scale-aware graded (decaying band) reward**. We want to ablate this: old vs new, band width, decay speed/shape. Decision from Chris: **offline analysis first** — no training runs until the offline study says which variants are worth GPU time.

## Pre-investigated facts (don't re-derive; do spot-check)

**Current formula** — `counting_reward_value(pred, gt, tolerance)` at
`verl/utils/reward_score/math_verify_reward_type_boxed.py:938-960`, wrapped by `_counting_reward` at `:963-970`. With `d = |pred − gt|`:
1. `d == 0` → 1.0
2. intrinsic full-credit band `band = floor(0.05·|gt|)` (0 for gt<20)
3. per-row `extra_info["tolerance"]` may widen: `band = max(band, min(tol, 0.15·|gt|))` — currently **always null** in our data
4. `d ≤ band` → 1.0
5. else Gaussian decay `exp(−((d−band)/σ)²)`, `σ = max(0.5, 0.15·|gt|)`

Reference values (use as harness sanity checks): gt=100,pred=110 → ≈0.895 · gt=7,pred=8 → ≈0.40 · gt=1,pred=0 → ≈0.018.

**Knobs — all hardcoded module constants, no config plumbing exists:**

| Constant | Default | Location |
|---|---|---|
| `_COUNTING_REL_TOL` (band fraction) | 0.05 | `math_verify_reward_type_boxed.py:927` |
| `_COUNTING_REL_BAND_CAP` | 0.15 | `:930` |
| `_COUNTING_REL_SIGMA` (decay width fraction) | 0.15 | `:932` |
| `_COUNTING_MIN_SIGMA` (width floor) | 0.5 | `:935` |
| decay shape (Gaussian, exp 2) | fixed | `:960` |

**Old behavior (the baseline arm):** the training entrypoint used `_short_string_reward` (`math_verify_reward_type_fixations_no_think_point_boxed.py:288`): both parse as int → 1.0 iff exactly equal; else normalized string equality; else 0.0. Sibling modules (`math_verify_reward_type_multiturn_tool_calling.py:156`, the fixations handler maps) still use exact match — a ready-made reference implementation.

**Routing:** per-row `extra_info["reward_type"] == "counting"` → handler map at `math_verify_reward_type_multiturn_tool_calling_optional_think.py:173-182` (this module is the configured `reward.custom_reward_function` in all vero2 training YAMLs). Final blend: `(1−format_score−tool_eff)·acc + …` at `:250-255`.

**Tests:** `tests/utils/reward_score/test_counting_reward_on_cpu.py` (12 tests). Known-working interpreter:
`/scratch/gpfs/ZHUANGL/lc1556/conda/envs/reason-vlm/bin/python -m pytest tests/utils/reward_score/test_counting_reward_on_cpu.py`
(pre-existing unrelated failure elsewhere in suite: `test_search_reward_integration_uses_partial_accuracy`).

**Gotchas:**
- `counting_reward_formula.md` referenced by CLAUDE.md **does not exist** — the code docstring is the only formula doc. Recreating it is a deliverable.
- The standard train/val jsonls contain **zero** counting rows; any future training arm must mix in `/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync/filtered/combined_counting_train.filtered.partial.verl.jsonl` (+val).
- Realistic (pred, gt) pairs for offline replay: `/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync/decisions.jsonl` — 10,586 records with `blind.count` (a strong VLM's count, never saw GT) vs `ground_truth`. Also check `<checkpoint>/debug_logs/reward_assignments/` of past runs under `/scratch/gpfs/ZHUANGL/lc1556/vero2/checkpoints/vero2_verl/` for real policy predictions.

## Scope (in order)

**Phase A — offline replay harness.** A script (suggest `vero2_scripts/debug_scripts/counting_reward_ablation.py`) that replays (pred, gt) pairs through reward variants and emits distribution plots + tables: reward histograms, fraction-exact / in-band / in-decay / zero, mean reward vs gt magnitude buckets. Variants to sweep:
1. old exact-match vs new graded (headline)
2. `_COUNTING_REL_TOL ∈ {0, 0.02, 0.05, 0.10}`
3. `_COUNTING_REL_SIGMA ∈ {0.05, 0.15, 0.30}` × `_COUNTING_MIN_SIGMA ∈ {0.25, 0.5, 1.0}`
4. band-only (0 outside band) vs graded — isolates "gradient signal" from "leniency"
5. Gaussian (exp 2) vs Laplacian (exp 1) decay

**Phase B — plumb the knobs.** Thread the four constants through `reward.custom_reward_function.reward_kwargs` (config → `async_compute_score_from_data_source` → `compute_score` → `acc_reward` → `_counting_reward` → `counting_reward_value`) with the current values as defaults, so training arms become Hydra overrides instead of code edits. Extend the CPU test file to cover the kwargs path. **Do not change default behavior** — all existing tests must still pass unmodified.

**Phase C — deliverables.**
- `counting_reward_formula.md` at repo root (the real formula doc CLAUDE.md already points to).
- An analysis memo (`docs/vero2/counting-reward-ablation.md`): plots, findings, and a ranked shortlist of 2-3 variants worth training runs, each with its exact Hydra override line and a proposed `trainer.experiment_name`.
- Prepared (not launched) config/launcher deltas for those arms. **Check with Chris before launching any multi-GPU training.**

## Verify the result

1. Harness reproduces the reference values above to 3 decimals for the default config.
2. `pytest tests/utils/reward_score/test_counting_reward_on_cpu.py` passes; new kwargs tests pass; run the whole `tests/utils/reward_score/` dir and confirm no new failures vs the known pre-existing one.
3. Phase B with no overrides is bit-identical to current behavior (add a test asserting defaults == module constants).
4. Memo plots regenerate from the checked-in script with one command.

## Workflow (per CLAUDE.md)

Work only on this branch. Commit every substantive chunk; `git push` as you go (upstream is set). When done: `gh pr create` into `main`. For any GPU work: `source ./set_paths.sh` from THIS worktree root (verify `$VERL_ROOT` prints this path), Slurm headers from `useful_command.sh` §3, smokes on gpu-test (<1h), and a distinct `trainer.experiment_name` per arm.
