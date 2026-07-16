# Postmortem: the 3.4 h/step drawing-RL slowdown (vLLM attention backend)

**TL;DR** — Counting-reward drawing runs (Qwen3-VL-8B, batch 256 × n 8 = 2048 rollouts/step) took
**3.4–4 h/step on 8×H100** while a colleague's identical-knob runs took **22 min/step**. Root cause:
our configs never pinned the vLLM attention backend, so vLLM 0.23 **auto-selected FlashInfer**,
which handles this image-heavy multi-turn workload pathologically. Pinning
**FlashAttention-2** restores expected speed — verified at **23.1 min/step** with our env/data.

```yaml
# the fix (actor_rollout_ref.rollout.engine_kwargs.vllm)
attention_config:
  backend: FLASH_ATTN
  flash_attn_version: 2
```

## Symptom

- graded/exact ablation runs (jobs 10882641/10882642, pli H100): step 1 = 12,177 s, degrading to
  ~14,400 s by step 8. 60 steps ≈ 10 days/arm — unusable.
- Rollout generation was 92–94 % of each step; the slowest of 2048 rollouts gated the batch, with
  stragglers decoding at **0.18 tok/s** (2,106 text tokens in 3.2 h).

## What it was NOT (each ruled out by experiment or hard evidence)

| Hypothesis | Verdict | Evidence |
|---|---|---|
| Slow matplotlib draw tool | ❌ | pure-PIL swap changed nothing (tool_calls mean 0.54 s); tool file md5-identical to the fast runs' |
| High-count data / straggler tail | ❌ (minor, ~1.2–1.5×) | colleague runs the **same full-range tier** (his `correct_only` ⊂ our `filtered`, same 4.4 MP images) at 22 min/step |
| Env (vllm/torch/packages) | ❌ | **diag B**: his script+config in OUR env → 23.1 min/step |
| Hardware (H200 vs H100) | ❌ | his H100 runs also fast |
| Turn/token caps | ❌ | caps byte-identical in both configs and respected in traces |

## Localization

Phase breakdown (ours job 10882641 vs his wandb-verified job 11142137, both batch 256):

| Phase | ours (H100) | his (H200) |
|---|---|---|
| update_actor | 711 s | 659 s |
| old_log_prob | 263 s | 235 s |
| **generation** | **11,189 s** | **428 s** |

Training identical; **generation 26×**. Same rollout shapes (turns mean 4.7, response mean ~2,400
tok). A field-by-field diff of the resolved configs left exactly one engine-level difference: his
`attention_config` pin. Our slurm log confirms the FlashInfer autotuner firing at engine start
(env carries flashinfer 0.6.6); with the pin, the log prints `Using FlashAttention version 2`.

## Verification (diag B, job 11182509, 2026-07-16)

His launcher+config verbatim, **our** conda env (`vllm 0.23.0`), our scratch data, ailab H200:
`Using AttentionBackendEnum.FLASH_ATTN backend` → **step 1 = 1,385 s (23.1 min), gen = 432 s** —
matching his 32-step mean (1,332 s / 428 s). A twin diagnostic on our exact ablation config
(+pin, pli H100) queued as final confirmation; the relaunched ablation doubles as that test.

## Fix applied

- All 17 active `vero2_scripts/config/*.yaml` now carry the pin (`archive/` untouched).
- Expected ablation cost after fix: ~25–40 min/step → 60 steps ≈ 1 day/arm (was ~10 days).

## Lessons

1. **Pin engine-critical choices.** Backend auto-selection is environment-dependent and silently
   catastrophic on unusual workloads (multi-image, multi-turn, 2048-way concurrency).
2. **Run vLLM at INFO log level in at least one smoke** — the backend-selection line is the
   difference between a 5-minute diagnosis and a week of data/tool theories.
3. **Compare resolved configs, not launch scripts** — `debug_logs/config_resolved.yaml` is ground
   truth, and colleagues' group-readable checkpoint dirs (incl. wandb `output.log`) are gold.
4. Plausible mechanisms (image-token re-prefill, straggler tails) can be real *amplifiers* yet not
   the root cause — falsify against a working baseline before optimizing around them.
