The selected step-200 generalization suite evaluates exactly the eight requested
checkpoint conditions and three Visual Probe baselines. It submits one H100 job
per condition, with Pinpoint and VStar in that job by default. Add
`--include-hrbench-8k` to also evaluate HRBench-8K. The checkpoint sweep script
`submit_zoom_eval_checkpoints_qwen35_27b_judge.sh` is preserved.

```bash
# May be invoked from any directory. Neither of these submits jobs.
bash /home/gs1693/repo/vero2/vero_eval/examples/submit_zoom_eval_selected200_generalization.sh --preflight
DRY_RUN=1 bash /home/gs1693/repo/vero2/vero_eval/examples/submit_zoom_eval_selected200_generalization.sh

# Submit all eleven jobs when ready.
bash /home/gs1693/repo/vero2/vero_eval/examples/submit_zoom_eval_selected200_generalization.sh

# Select one condition, or comma-separated condition keys.
EVAL_NAME=branchstrata20_fresh_lr1e5_critic \
  bash /home/gs1693/repo/vero2/vero_eval/examples/submit_zoom_eval_selected200_generalization.sh
```

`--conditions`, `--output-root`, `--time HH:MM:SS`, and `--manifest FILE` are also
supported. `DRY_RUN`, `EVAL_NAME`, `OUTPUT_ROOT`, and `TIME_OVERRIDE` are their
environment equivalents. Input preflight checks model/indexed weight readiness,
prompt/template availability and cached sample counts. It does not run inference.
The implementation uses `run_eval.sbatch` and the canonical evaluation runner.
Other inherited training, critic, prompt and Slurm overrides are cleared so a
previous experiment cannot silently change these profiles. Resources are fixed
to `pli-c`, account `pli`, one H100, 8 CPUs, 180G RAM, 10 workers. No explicit
QoS is requested, so the jobs use the partition/account defaults. The scheduler
accepts the full time limits through this route; explicit `pli-cp` is capped at 8h.

`--config examples/configs/selected200_requested_step200.json` selects the eight
requested comparison rows, including Mini-o3 rewards at step 200. Its dedicated
`minio3_step200_zoom` tasks preserve Mini-o3's saved zoom schema (no bbox items or
length constraints, and optional `img_idx`), JSON/Hermes template, exact system
prompt, ten tool rounds, forced final answer and no critic. The dual-loss
continuation uses local step 120, which is cumulative step 200. Existing full
jobs retain their original protocol identities with `--include-hrbench-8k`
and are skipped when queued.

By default there are 624 samples per condition (6,864 total across 11 jobs).
With `--include-hrbench-8k`, there are 1,424 per condition (15,664 total):

| Benchmark | Split | Samples | Scorer |
|---|---|---:|---|
| Pinpoint-Bench | validation | 433 | Visual Probe rubric, Qwen3.5-27B judge |
| VStarBench | test | 191 | Existing `vstar_overall_acc` |
| HRBench-8K (opt-in) | hrbench_8k | 800 | Existing `hrbench_overall_acc` |

Pinpoint uses the existing validation metadata and cached images.
Each benchmark retains its dataset question/choice construction and scoring
implementation. These are generalization inputs, with the condition's Visual
Probe inference protocol applied to them.

The frozen condition registry is `configs/selected200_generalization.json`.
`lmms_eval/tasks/selected200_generalization/` supplies three selected tasks for
each of the three protocol families; older task variants remain available. The task names include `selected200` to keep old
results/configurations separate. The tool family pins Visual Probe's 24,576
generation tokens, temperature .7, top-p .8, top-k 20, repetition penalty 1,
presence penalty 0, exact zoom schema, training image indexing and restored
assistant prefill. The selected task variants apply the 24,576-token Visual
Probe cap consistently, including VStar and HRBench-8K.

| Condition key | Checkpoint | Tool rounds/images | Answer policy |
|---|---|---:|---|
| `qwen35_vero2_step200` (Vero2 zoom-in, Qwen3.5 XML) | standalone step 200 | 10 | Natural tools, forced final at cap; XML prompt/template, `qwen3_coder` |
| `gdpo_fresh_lr1e5_noexplore` | 200 | 10 | Natural tools, forced final at cap; no critic |
| `gdpo_fresh_lr1e5_noexplore_accformat` | 200 | 10 | Natural tools, forced final at cap; no critic |
| `gdpo_fresh_lr1e5_noexplore_accformat_forced10_no_final_prefill` | accformat 200 | 10 | Force ten tool rounds, then force an answer; empty final textual prefill; no critic |
| `gdpo_fresh_lr1e5_noexplore_accformat_always_retry_answers_no_critic_final_prefill` | accformat 200 | 10 | Retry every feasible candidate answer with persist prefill, no critic; standard final prefill |
| `branchstrata20_fresh_lr1e5_restart_step80` | **local 120 = cumulative 200** | 10 | Natural tools, forced final at cap; no critic |
| `branchstrata20_fresh_lr1e5_critic` | 200 | 10 | Learned policy token critic, threshold .5, persist retry prefill |
| `branchstrata20_fresh_lr1e5_critic_maxcalls5` | 200 | **5** | Same learned critic, threshold .5, persist retry prefill |
| `qwen3vl_8b_instruct_normal` | fixed base model | 10 | Vero zoom task contract with native Qwen template; no forced final, no critic |
| `qwen3vl_8b_instruct_no_tools` | fixed base model | tools off | Native Qwen template; 16,384 tokens, temperature 1, top-p 1, top-k 40, presence penalty 2 |
| `vero_qwen3i_8b_no_tools` | fixed Vero model | tools off | Vero no-think template; 16,384 tokens, temperature .6; checkpoint defaults for omitted sampling fields |

All solver images use the same 10M-pixel maximum. JSON checkpoints use their
exported templates and Hermes parser. The 2,980-byte JSON system prompt matches
Visual Probe's task prompt exactly (SHA-256 `c86c158c...`); the XML prompt is 3,372
bytes (SHA-256 `f1802c9d...`). The critic cutoff is uniform across all three
benchmarks, matching selected9 Visual Probe's .5 cutoff. The judge is separate
from the learned policy critic. Pinpoint uses Qwen3.5-27B with the no-think judge
template, TP=1, deterministic sampling, eager execution and Triton GDN prefill;
policy and judge share the allocated GPU sequentially.

The unconditional retry controller is now an explicit `always_retry_answers`
model option, default off. It uses the saved ablation's controller: discard a
candidate, force a zoom using persist prefill while budget remains, record the
decision, and never query a critic. It retains final-answer handling at the cap.
An explicitly empty `FORCE_FINAL_ANSWER_THINK_PREFILL` now survives both canonical
shell runners. Pinpoint is also recognized as a judge task so policy sleep/wake
works when no Visual Probe task is present.

Walltime estimates use the exact selected9 step-200/baseline result files, each
with 515 Visual Probe samples. Most inputs have complete evaluation time. The
two ablation jobs failed during judge startup, so their policy-generation timing
is used explicitly; their recovered scoring files do not contain end-to-end time.

| Condition | Measured wall seconds/sample | 1,424-sample projection | Configured limit |
|---|---:|---:|---:|
| Qwen3.5 XML | 31.27 | 12.4h | 20h |
| GDPO noexplore | 15.50 | 6.1h | 10h |
| GDPO accformat | 17.20 | 6.8h | 11h |
| Forced ten, no final prefill | 23.72 | 9.4h | 20h |
| Always retry, no critic | 25.93 | 10.3h | 22h |
| Restart from 80 | 27.78 | 11.0h | 17h |
| Critic ten rounds | 37.27 | 14.7h | 23h |
| Critic five rounds | 18.89 | 7.5h | 12h |
| Qwen normal tools | 15.90 | 6.3h | 10h |
| Qwen no tools | 6.85 | 2.7h | 5h |
| Vero no tools | 7.85 | 3.1h | 6h |

Formula: `ceil(samples * prior_wall_seconds / 515 / 3600 * factor + 0.5)` hours.
The factor is 1.5 normally, and 2.0 for the ablations, whose generation used two
GPUs and omitted successful judging. These are aggregate wall seconds per sample
with concurrent workers, not individual request latency. New benchmark image
sizes and difficulty can change throughput; these are estimates, not measured
suite runtimes. All limits are below the verified `pli-c` maximum of 72h.

Outputs and policy/Slurm logs go to scratch under
`/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_qwen35_27b_judge/selected200_generalization_v2/`.
Checkpoint outputs retain `global_step_200`, including the local-120 restart;
fixed models use `baseline`. Queued/running jobs are checked using unique hashed
protocol identities. Complete results are skipped only with full task/sample
coverage and no sample limit. Conflicting saved submission plans are rejected.
Prejudge samples are saved for recovery if scoring fails. The script refuses
actual submission if the scheduler cannot be queried.

The setup audit is saved in `analysis/selected200_pinpoint_vstar_hr8k_20261018/`:
`plan.json`, `timing.csv`, `dry_run.log`, and `task_preflight.json`. All 9 selected task
variants were instantiated from the real cache, complete split counts checked,
and representative native images decoded. Controller/runner tests cover empty
prefill, unconditional retry without critic requests, cumulative checkpoint
mapping and completion checks. CPU tests and dry runs do not establish GPU
inference or suite completion. No jobs were submitted during setup.
