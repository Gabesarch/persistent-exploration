# TASK: Drawing-tool interruption (answer-retry) + training-speed investigation

Branch: `drawing-interruption` · Worktree: `/home/lc1556/repo/reason-vlm/vero2-drawing-interruption`
Written 2026-07-02 by the coordinating session (facts below verified against the repo on that date).
Two phases, **sequential** (Chris's decision): Phase A first; Phase B after Phase A's job has run.

## Phase A — port the interruption mechanism from zoom to drawing

### Context

PR #43 merged the forced-explore / conditional answer-retry machinery for the zoom tool (read `docs/vero2/forced-explore.md` first — it's accurate). The reference launcher is
`vero2_scripts/run_gdpo_hdpotool_gspo_qwen3vl_visual_probe_cropzoom_cpueff_hardonly_answer_retry_forcedprefill_full.sh` → config `vero2_scripts/config/gdpo_hdpotool_gspo_chart_qwen3vl_visual_probe_cropzoom_cpueff_hardonly_answer_retry_forcedprefill.yaml` (retry block at `:221-237`; assistant-prefill text injected by the launcher at `:44-46`, `:118-123`). We want the same conditional answer-retry behavior in a run where the model uses **drawing tools** (annotate the image, get the annotated image back) instead of / alongside zoom.

### Pre-investigated facts

**The interruption machinery is tool-agnostic — no changes needed** in `verl/experimental/agent_loop/tool_agent_loop.py`, `verl/utils/forced_action_tags.py`, the vLLM logits processor, or `vllm_async_server.py`. It forces the generic `\n<tool_call>` tag (never a tool name), grades via the task reward fn, and rolls back wrong answer turns (snapshot `:610` / rollback `:620-628` / retry decision `_maybe_retry_incorrect_answer` `:630` / skip reasons `:955-972` / masked prefill append `:597-608`).

**Training side has NO drawing tool.** `verl/tools/` only has `image_zoom_in_tool.py`; tool_config only zoom YAMLs. The `drawing-eval-merge` branch is eval-only.

**Eval side has a complete PIL toolset to port from:** `vero_eval/lmms_eval/mcp/vero2_mcp.py` — `draw_points` `:347`, `draw_lines` `:375`, `draw_boxes` `:403`, `draw_texts` `:439`, `draw_polygons` `:469` (0–1000 normalized coords; each returns the annotated image, which eval registers as a new addressable `img_idx` — mirror that semantics). System-prompt reference: `vero_eval/prompts/tools/system_draw.txt`.

### Work items

1. **`verl/tools/image_draw_tool.py`** — subclass `BaseTool`, mirroring `ImageZoomInTool` (`__init__` `:155`, `create` `:298`, `execute` `:539`, `release` `:670`); port the PIL logic from `vero2_mcp.py:347-500`. `execute` returns `ToolResponse(image=<annotated PIL image>, text=...)`; the agent loop already appends returned images to `image_data` (`tool_agent_loop.py:1333-1338`) and the shipped `custom_chat_template` renders them inside tool messages.
2. **Tool schema YAML** — `vero2_scripts/config/tool_config/image_draw_tool.yaml` (and/or a combined zoom+draw file; the `tools:` list supports multiple entries). Set BOTH `data.tool_config_path` and `actor_rollout_ref.rollout.multi_turn.tool_config_path`.
3. **Config + launcher** — clone the `..._answer_retry_forcedprefill` YAML + `_full.sh` launcher into drawing variants. All `conditional_answer_retry_*` flags carry over. Keep the `reward.custom_reward_function` block (without it every answer grades 0.0 and retries fire until the caps — `docs/vero2/forced-explore.md:172`). Keep `gdpo_reward_keys: [accuracy, format]`.
4. **Suffix/prefill text** — if you enable Mechanism 2, you MUST override `tool_response_explore_suffix` (its default in `verl/workers/config/rollout.py:97-102` says "keep zooming"). Consider tailoring `ANSWER_RETRY_ASSISTANT_PREFILL` to drawing ("…need to annotate the image to verify").

### Critical caveats

- **Lazy multimodal cache**: the zoom tool emits `crop_image_refs` (`image_zoom_in_tool.py:663`, schema `verl/tools/schemas.py:100`) so crops can be lazily rebuilt. A draw tool must either emit an analogous deterministic ref (source img idx + draw args) or the run must set `lazy_multi_modal_inputs: false` — otherwise the lazy path has nothing to rebuild annotated images from. Decide explicitly and say which in the PR.
- Forcing `<tool_call>` doesn't choose WHICH tool; with zoom+draw both registered the model picks. That's fine for v1; forcing a specific tool would need a new `force_tool_call_prefix`-style knob (precedent: `force_final_answer_prefix`, `tool_agent_loop.py:399-405`).

## Phase B — training-speed bottleneck investigation (after Phase A's job)

Training is slow; find out where the time goes and what to do. A prior investigation (2026-07-02) established:

- **All the instrumentation already exists**: wandb `timing_s/{step,gen,old_log_prob,update_actor,update_weights,…}` + per-trajectory `agent_loop/generate_sequences/{min,max,mean}`, `agent_loop/tool_calls/*`, `agent_loop/compute_score/*`, `response_length/clip_ratio`, `num_turns/*` (emitted from `verl/trainer/ppo/ray_trainer.py` + `metric_utils.py:271` + `agent_loop.py:1297`). `debug_logs/` has per-stage GPU-memory/duration events every 40 steps.
- **No straggler mitigation is active**: the step barrier waits for the slowest of 2048 trajectories (`agent_loop.py:1243-1287`). `rollout.over_sample_rate` exists in the schema (`verl/workers/config/rollout.py:239`) but **nothing consumes it in this fork** (abort plumbing exists: `vllm_async_server.py:693`, `:1011`). `verl/experimental/fully_async_policy/` implements real partial-rollout resume but is unused. `algorithm.rollout_correction` (truncated-IS + bypass mode skipping the `old_log_prob` forward) is in-tree, unused.
- **Suspicious config**: `enable_prefix_caching: false` + `mm_processor_cache_gb: 0` → every turn re-prefills the whole image-heavy conversation (4MP images ≈ 4k tokens each, 11 turns). Likely old-vLLM workarounds; retest on the fresh vllm-0.23 env.
- **slime's answer** = APRIL (arXiv 2509.18521): over-provision rollouts, abort the tail when the target count completes, resume partials next step under new weights; 20-44% throughput gain. verl-upstream equivalents: `over_sample_rate` (SGLang path), `fully_async_policy`.

**Do:** pull the wandb metrics of the most recent real training run (plus Phase A's run) and produce `docs/vero2/training-perf.md`: stacked `timing_s/*` share-of-step, straggler ratio `generate_sequences/max ÷ mean`, `clip_ratio`, and a prioritized fix list with expected win and effort. If (and only if) the numbers point at the rollout long tail, prototype the cheapest mitigation (e.g. wire `over_sample_rate` into `AgentLoopManager` using the existing `abort_request`, handling GRPO groups: abort whole groups or mask aborted members) behind a default-off config flag, and A/B it in two gpu-test smokes.

## Verify the result

**Phase A:** 4×A100 gpu-test smoke (recipe: `useful_command.sh` §7) with the drawing config: job completes ≥2 steps; `<checkpoint>/debug_logs/sample_traces` show (a) `draw_*` tool calls parsed, (b) annotated images returned into the conversation, (c) `conditional_answer_retry_count` > 0 somewhere with rolled-back turns absent from the final trace; rewards non-degenerate. Use a `_smoke` `trainer.experiment_name`.
**Phase B:** every claim in the report cites an actual metric value from a named run; any implemented optimization shows a measured before/after `timing_s/gen` (or step) delta from the A/B smokes, and is default-off.

## Workflow (per CLAUDE.md)

Work only on this branch; commit each work item separately and `git push` as you go (upstream set). PR into `main` when Phase A is verified — Phase B can be a second PR from this branch or a follow-up branch, your call (state it in the first PR). `source ./set_paths.sh` from THIS worktree root (verify `$VERL_ROOT`); Slurm headers from `useful_command.sh` §3; distinct `trainer.experiment_name` per run.
