# Training-speed investigation (Phase B)

Where does a training step go, and what is the cheapest way to make it faster? This
quantifies the step-time breakdown from **real wandb metrics** of the most recent real
run, cross-checks it against the Phase-A drawing smoke, and gives a prioritized fix list.

**Headline:** a step takes **~3445 s (57 min)** and the **rollout (`timing_s/gen`) is 58 %**
of it. The rollout is a **barrier that waits for the slowest of 1024 trajectories** — the
gen wall-clock (1998 s) is essentially the slowest single trajectory (1926 s), which runs
**1.82× the mean** (1060 s) and **31× the fastest** (62 s). The two cheapest levers are
(1) stop re-prefilling the image-heavy context every turn (prefix / mm caching — currently
**off**), and (2) abort the rollout long tail (`over_sample_rate`, currently a **dead knob**).

---

## 1. Data sources (every number below is traceable)

| What | Where |
|---|---|
| **Real run** = `GDPO_HDPOtool_GSPO_…_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_answer_retry_forcedprefill_20260524_160610` (the direct predecessor of the Phase-A zoom+draw config) | checkpoints: `…/gs1693/vero2/checkpoints/vero2_verl/…answer_retry_forcedprefill_20260524_160610/` (trained past `global_step_104`) |
| Real-run per-step metrics | wandb offline runs `…/gs1693/wandb/offline-run-20260702_053355-wo6vyn1x` (steps 105–109) + `…-20260701_052807-qm1z9j6w` (steps 83–105) → **28 steady-state steps** aggregated |
| Real-run resolved config | `…answer_retry_forcedprefill_20260524_160610/debug_logs/config_resolved.yaml` |
| **Phase-A smoke** = `zoomdraw_answer_retry_forcedprefill_smoke_a100` (fresh vllm-0.23 env, 4×A100, debug256) | `…/lc1556/vero2/checkpoints/vero2_verl/zoomdraw_answer_retry_forcedprefill_smoke_a100/` + `smoke_10608595.log` |

wandb offline history is binary; it was read with `wandb.sdk.internal.datastore.DataStore` +
`wandb_internal_pb2` (script kept at `…/tmp/wbread.py`). Runs resume per `--array` segment, so
history is spread across several `offline-run-*` dirs of the same experiment.

---

## 2. Step-time breakdown — real run (mean of 28 steps)

`timing_s/step` = **3444.9 s**. Stacked share of the step:

| Stage (`timing_s/…`) | seconds | share |
|---|---:|---:|
| **gen** (async multi-turn rollout) | **1998.0** | **58.0 %** |
| **update_actor** (FSDP2 fwd+bwd+optim) | 1085.4 | 31.5 % |
| **old_log_prob** (recompute logprobs for IS) | 352.9 | 10.2 % |
| update_weights (vLLM weight sync) | 6.6 | 0.2 % |
| adv (GDPO advantage) | 0.76 | 0.02 % |
| reward | 0.035 | 0.001 % |

So **gen + update_actor = 89.5 %** of the step; the reward/advantage math is free. Anything
that matters is in the rollout or the optimizer.

## 3. The rollout is a long-tail barrier

`generate_sequences` is the per-trajectory wall-clock (all 1024 run concurrently on the vLLM
engine); the step blocks on `asyncio.gather` of all of them (`agent_loop.py:651`).

| `timing_s/agent_loop/generate_sequences/…` | seconds |
|---|---:|
| min | 61.9 |
| mean | 1060.0 |
| **max (slowest trajectory)** | **1925.9** |
| **straggler ratio `max ÷ mean`** | **1.82** (per-step range 1.61 – 2.04, n=28) |

`timing_s/gen` (1998 s) ≈ `generate_sequences/max` (1926 s) + ~72 s of gather/postprocess
overhead. **The whole batch pays for its slowest member.** The slow members are the long,
many-turn trajectories: `response_length/mean` = 6777, **`/max` = 24576 (the cap)**,
`response_length/clip_ratio` = **4.3 %** (i.e. 4.3 % of responses are truncated at
`max_response_length`), and `num_turns/mean` = 16.3, **`/max` = 22** (= 2 × `max_assistant_turns`
11 — the tail runs every allowed turn). Tool time is small: `tool_calls/mean` = 11.0 s,
`compute_score/mean` = 0.12 s.

**If the barrier finished at the mean instead of the max**, gen → ≈ 1060 + 72 ≈ **1132 s**,
cutting **~866 s off a 3445 s step ⇒ ~25 % faster**. That is the ceiling for straggler
mitigation at this straggler ratio.

## 4. Config confirms the two flagged suspicions

From `config_resolved.yaml` (active rollout block):

```
enable_prefix_caching: false          # ← every turn re-prefills the whole conversation
engine_kwargs.vllm.mm_processor_cache_gb: 0   # ← multimodal processor cache disabled
over_sample_rate: 0                    # ← straggler-abort knob exists but is unused
gpu_memory_utilization: 0.6 · max_num_seqs: 728 · max_model_len: 40960
train_batch_size: 128 · n: 8  ⇒ 1024 trajectories/step · max_response_length: 24576
```

With `num_turns/mean` ≈ 16 (≈ 8 assistant turns), **prefix caching off means each turn
re-prefills the full, growing, image-heavy context** (a 4 MP image ≈ thousands of tokens).
This inflates the *mean* `generate_sequences` (1060 s), i.e. it slows *every* trajectory, not
just the tail. These flags were almost certainly old-vLLM workarounds; the fresh vllm-0.23
env is the place to retest them (§7).

## 5. Cross-check — Phase-A smoke (fresh vllm-0.23, 4×A100, debug256)

Small scale (32 trajectories, `max_pixels` 4.19 MP, step 2), but the **shape is identical**:

| `timing_s/…` | seconds | share |
|---|---:|---:|
| step | 114.5 | — |
| gen | 60.7 | 53.0 % |
| update_actor | 39.8 | 34.8 % |
| old_log_prob | 9.4 | 8.2 % |
| update_weights | 4.5 | 3.9 % |

`generate_sequences` min/mean/max = 2.6 / 17.2 / 57.1 s → straggler ratio **3.32** (higher, but
32-trajectory small-batch noise). `response_length` mean/max = 3423 / 11658, clip 0.0;
`num_turns` mean/max = 9.3 / 22. Same story: gen-dominated, tail-bound. **This smoke still had
`enable_prefix_caching:false` + `mm_processor_cache_gb:0`**, so it is the caching-**off**
baseline for the A/B in §7.

---

## 6. Prioritized fix list

Ordered by (win ÷ effort). Wins are share-of-step against the 3445 s real step.

| # | Fix | Mechanism | Expected win | Effort | Risk |
|---|---|---|---|---|---|
| **1** | **Enable prefix caching + mm processor cache** on the fresh vllm-0.23 env (`enable_prefix_caching:true`, `mm_processor_cache_gb>0`) | Stop re-prefilling the growing image-heavy context each of ~16 turns → lowers **mean** `generate_sequences`, hence gen (58 % of step) | Potentially large (attacks the mean, not just the tail). **A/B measures it — §7.** | **Trivial** (2 flags, already in schema, default-off) | Was off on old vLLM; verify correctness/compat + numerics on the fresh stack before flipping the default. |
| **2** | **`over_sample_rate` straggler abort** (APRIL, arXiv 2509.18521), default-off flag | Over-provision rollouts; when `(1−rate)·N` complete, abort the tail via `abort_request`; makes gen finish near the **mean** not the **max** | ~**25 %** of step (gen 1998→~1132 s), bounded by straggler ratio 1.82 | **Medium** (code; §8) | Aborting GRPO group members breaks advantage math — must abort whole groups or resume partials; gate behind default-off flag. |
| **3** | **`rollout_correction` bypass** (skip the `old_log_prob` forward; use rollout logprobs + truncated-IS) | Removes the 10.2 % `old_log_prob` recompute | ~**10 %** of step (353 s) | Low–medium (in-tree, unused) | Changes the IS correction; validate advantage/gradient equivalence. |
| 4 | Tighten `max_response_length` / `max_assistant_turns` tail | 4.3 % of responses hit the 24576 cap; the tail runs all 11 turns | Small–moderate (shrinks the tail directly) | Trivial config | Quality/accuracy trade-off — measure reward impact first. |
| — | `update_actor` (31.5 %) | FSDP2 fwd+bwd+optim | (largest single block) | **High** — needs parallelism/precision/offload changes | Separate project, not a quick win. |

**Combined ceiling** of the two rollout fixes (#1 mean + #2 tail are partly overlapping; #3 is
additive): #2 alone ≈ −25 %, #3 alone ≈ −10 %, together ≈ **−33 %** step time; #1 could add more
on top by cutting the mean that #2's floor depends on.

**Recommendation:** ship #1 first (it is nearly free and, if it cuts the mean, it *also* lowers
the floor #2 can reach). Then implement #2 for the tail, and #3 for the log-prob pass.

---

## 7. A/B — prefix + mm caching (fresh vllm-0.23 env)

Two gpu-test smokes, identical `zoomdraw_debug256` config (3 steps, batch 8, n=4), differing only
in the caching flags. Baseline = shipped default (caching off). Run from the isolated worktree
pinned to the Phase-A commit; `over_sample_rate` stays 0 in both arms.

| Arm | `enable_prefix_caching` | `mm_processor_cache_gb` | `timing_s/gen` (mean) | `timing_s/step` (mean) | Δ gen |
|---|---|---|---:|---:|---:|
| OFF (baseline) | false | 0 | _pending (job 10609421)_ | _pending_ | — |
| ON | true | 4 | _pending (job 10609422)_ | _pending_ | _pending_ |

_(Filled from `ab_*_<jobid>.log` when both smokes finish; both are default-off in the shipped
configs, so this is a measurement, not a default change.)_

---

## 8. `over_sample_rate` design sketch (fix #2, not yet implemented)

The knob already exists — `verl/workers/config/rollout.py:239-240`:

```python
# Abort remaining requests when (1 - over_sample_rate) * total_requests are completed.
over_sample_rate: float = 0.0
```

…but **nothing consumes it** in this fork. The barrier to replace is the plain gather in
`AgentLoopWorker.generate_sequences` (`verl/experimental/agent_loop/agent_loop.py:651`):

```python
outputs = await asyncio.gather(*tasks)     # waits for ALL 1024 trajectories
```

Concrete change (default-off, so `over_sample_rate == 0` keeps today's exact behavior):

1. Launch all tasks as today. Iterate with `asyncio.as_completed`, counting completions.
2. When `ceil((1 - over_sample_rate) * N)` have completed, `abort_request` (server:
   `vllm_async_server.py:693`; manager fan-out `:1011`) the still-running tail, then let the
   aborted tasks return a sentinel.
3. **GRPO group integrity** (the one real hazard — advantage is computed within each
   `index` group of `n` samples): pick one of
   - **abort whole groups** — only count a group "done" when all `n` of its members finish, and
     abort groups that are entirely unstarted/incomplete; keeps every kept group full; or
   - **over-provision + keep first-full-groups** — request `N·(1+rate)` prompts, keep the first
     `N/n` groups that complete in full, abort the rest; or
   - **resume partials next step** — the real APRIL behaviour; `verl/experimental/fully_async_policy/`
     already implements partial-rollout resume and is the reference if we want zero wasted compute.

Start with "abort whole groups" (simplest, no wasted *kept* data, no cross-step state), behind
`over_sample_rate > 0`, and A/B it (gen delta) exactly like §7. Expected ≈ −25 % gen at the
observed 1.82 straggler ratio.

---

## 9. Reproduce

```bash
# pull real-run timing from the wandb offline segments:
python3 <repo>/…/tmp/wbread.py \
  /scratch/gpfs/ZHUANGL/gs1693/wandb/offline-run-20260702_053355-wo6vyn1x \
  /scratch/gpfs/ZHUANGL/gs1693/wandb/offline-run-20260701_052807-qm1z9j6w
# smoke timing is printed per step in the trainer console (grep 'step:N - .*timing_s/').
```
