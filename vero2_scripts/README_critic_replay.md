# Replay critic with prompt-group answer scheduling

The new run is isolated in branch `feat/critic-replay-schedule-v1`, in `/home/gs1693/repo/vero2-critic-replay`. Its launcher pins `REPO_ROOT`, `VERL_ROOT`, and the first Python import path to this checkout even if Slurm is invoked from another directory. `VERO2_REPO_ROOT` can select another checkout containing the feature config; invalid paths fail immediately. The original checkout and previous run configs retain their behavior.

## Full run

```bash
cd /home/gs1693/repo/vero2-critic-replay
mkdir -p slurm/output slurm/error
sbatch --chdir="$PWD" vero2_scripts/run_branchstrata20_fresh_lr1e5_critic_replay_schedule_v1.sh
```

The config copies the base Qwen3-VL-8B run: learning rate `1e-5`, batch `128`, eight samples per prompt, the task datasets/domain mixture, GDPO stability, GSPO clipping, and 20% task branch-strata weighting. Its distinct experiment is `branchstrata20_fresh_lr1e5_critic_replay_schedule_v1_20261010`. The checkpoint directory is `$SAVE_PATH/checkpoints/vero2_verl/<experiment>`. Auto-resume continues this run only; a different experiment name creates a fresh run. Hydra overrides may be passed after the launcher filename.

Replay targets 20% of prompts: 26 judges and 102 fresh task prompts when the batch-128 cache has enough candidates. Judges contain 13 full-credit accepts and 13 incorrect/partial rejects; shortages use available balanced pairs and leave more fresh task prompts. Each judge prompt is sampled eight times from the current policy. Historical answers supply input evidence, never policy likelihoods or training targets for SFT. Judges use accuracy/format GDPO weights `0.8/0.2` and the shared GSPO update; critic NLL and oracle-token supervision are disabled. Judge response tokens are capped at `4096 + 1024`, with the last 1024 tokens receiving the configured overlong penalty. The injected `<think>` prefix has no policy loss.

The cache lives under `<checkpoint root>/trajectory_replay`: at most 8192 candidates, balanced between `A` and `R`, with sampling across retained turn counts. Images are persisted by content and remain available after temporary crop cleanup. Each update loads the prior committed cache; new scored candidates are committed only after a successful update. `global_step_N/trajectory_replay.json` snapshots the matching cache. A checkpoint resume restores its snapshot rather than newer live cache state. Cache image cleanup keeps assets referenced by retained checkpoint snapshots.

A direct first-turn answer without a tool call is judged from its initial images. Later candidates use only their ordered tool outputs. Both training and validation judges receive the question and candidate answer, with previous reasoning and reference answers excluded. Validation samples one deterministic reasoning decision; malformed responses and generation failures retain the candidate and are logged. Validation never adds cache entries.

Fresh training prompts are apportioned 85% normal, 10% with one answer deadline, and 5% with two. The selected 15% force their first turn to call a tool. Deadlines are uniformly sampled from turns 2–9; double deadlines differ by at least two retained turns. The eight samples of a prompt share the same schedule. Scheduled deadlines are consumed before rollback, accepted answers terminate immediately, and final-turn/token limits take priority. Validation has no prompt-group schedule.

## GPU smoke and acceptance

```bash
cd /home/gs1693/repo/vero2-critic-replay
mkdir -p slurm/output slurm/error
sbatch --chdir="$PWD" vero2_scripts/run_branchstrata20_fresh_lr1e5_critic_replay_schedule_v1_smoke4gpu.sh
```

The smoke requests four 80GB GPUs on `gpu-test`. It uses batch 20 with eight samples, debug training data, and the original held-out validation file (20 validation examples). Phase one runs three updates, validating and checkpointing each. A second Python process resumes `global_step_3` and performs update four. `REPLAY_SMOKE_TRAIN_FILE` may override the debug training input. Artifacts, logs, and all four full checkpoints are under `$SAVE_PATH/replay_smoke/branchstrata20_fresh_lr1e5_critic_replay_schedule_v1_smoke4gpu_<job id>` on scratch. `REPLAY_SMOKE_RUN_ROOT` overrides the parent directory; retain a scratch destination because checkpoints can consume substantial disk space.

The launcher runs the acceptance validator automatically. To repeat it on saved artifacts:

```bash
python3 vero2_scripts/validate_replay_critic_smoke.py --run-dir /absolute/path/to/smoke_run
```

A pass requires an empty-cache warmup, balanced replay-bearing updates with active reasoning and nonzero advantages, a masked injected prefix, no supervised critic loss, preserved cache assets in all four checkpoints, restoration of the step-3 cache, sampled judge reasoning, and observed validation reasoning decisions. Missing runtime evidence fails the validator. The smoke does not guarantee that both scheduled deadlines are reached: early correct answers legitimately terminate a group. Scheduling and rollback invariants are separately tested on CPU.

## CPU checks

```bash
python3 -m pytest -q \
  tests/utils/test_trajectory_replay_on_cpu.py \
  tests/trainer/ppo/test_trajectory_replay_on_cpu.py \
  tests/experimental/agent_loop/test_replay_judge_agent_on_cpu.py \
  tests/experimental/agent_loop/test_prompt_group_replay_on_cpu.py \
  tests/experimental/agent_loop/test_replay_critic_launch_config_on_cpu.py \
  tests/experimental/agent_loop/test_replay_critic_smoke_validator_on_cpu.py \
  tests/experimental/agent_loop/test_answer_retry_branch_on_cpu.py \
  tests/experimental/agent_loop/test_candidate_answer_schedule_on_cpu.py \
  tests/experimental/agent_loop/test_forced_action_tags_on_cpu.py \
  tests/experimental/agent_loop/test_candidate_schedule_smoke_validator_on_cpu.py \
  tests/trainer/ppo/test_branch_advantage_mass_on_cpu.py
```

These checks cover replay capture, balancing, persistence and resume, mixed GDPO/GSPO advantages and gradients, judge evidence and token masks, prompt-group schedules and rollback, copied-config parity, launcher path isolation, smoke evidence, and existing branch/retry/candidate behavior. CPU checks establish wiring. A submitted job establishes queue registration; only observed GPU updates and a passing artifact validator establish GPU smoke acceptance.

Implementation verification on 2026-10-10: **221 passed, 1 skipped**. Ruff, shell syntax, compilation, base/feature Hydra composition, and typed rollout conversion in the training environment passed. No GPU job was submitted or run.

The separate existing `test_agent_loop_extra_fields_schema_on_cpu.py` file has four failures that also reproduce in the untouched original checkout: two async tests need a pytest async plugin, and two dummy-worker fixtures omit the existing expansion method. The new mixed-worker tests exercise actual worker expansion, concatenation, and trainer union with both synchronous and asynchronous reward paths.
