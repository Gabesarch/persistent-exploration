# Gemma 4 E4B generalization evaluation

Run both **nonthinking and thinking** Gemma 4 E4B on the exact benchmark scope of
`submit_zoom_eval_selected200_generalization.sh`:

| Benchmark | Split | Samples | Scorer |
|---|---|---:|---|
| Pinpoint-Bench | validation | 433 | VisualProbe rubric, Qwen3.5-27B no-think judge |
| VStarBench | test | 191 | Existing `vstar_overall_acc` |
| HRBench-8K (opt-in) | hrbench_8k | 800 | Existing `hrbench_overall_acc` |

By default there are **624 examples per mode**, 1,248 across the two jobs.
Add `--include-hrbench-8k` for 1,424 per mode, 2,848 across the two jobs.
There is no sample limit. Each mode gets one H100 on `pli-c`, account `pli`, 8 CPUs, 180G RAM, and
a 16-hour limit; no explicit QoS is set. This allows time for larger images than
VisualProbe. It is a resource allowance, not a measured suite runtime.

```bash
cd /home/gs1693/repo/vero2/vero_eval
# No jobs are submitted by either check.
bash examples/submit_zoom_eval_gemma4_generalization.sh --preflight
bash examples/submit_zoom_eval_gemma4_generalization.sh --dry-run

# Submit both full modes.
bash examples/submit_zoom_eval_gemma4_generalization.sh

# Select one mode, or a separate output root/time limit.
bash examples/submit_zoom_eval_gemma4_generalization.sh --modes thinking
bash examples/submit_zoom_eval_gemma4_generalization.sh --time 24:00:00 --output-root /scratch/path/new_run
```

`--manifest FILE` saves a reviewable plan. Repeated calls skip matching queued
jobs or results with complete task/sample coverage and no sample limit. A changed
protocol in an existing output directory is rejected; use a new output root.
Inherited Slurm, model, prompt, critic and judge overrides are cleared before
submission. Dataset counts come from the selected200 config and actual cache.

The new YAMLs in `lmms_eval/tasks/gemma4_generalization/` inherit the exact
selected200 dataset/split, question/choice construction, generation kwargs, and
scorers. They replace Vero's prompt/tool protocol with the established Gemma
VisualProbe protocol. The source selected200 task definitions remain unchanged.

Both modes use the same `prompts/tools/system_gemma_zoom.txt`, cached native
Gemma template, `--tool-call-parser gemma4`, and model-facing `[y1,x1,y2,x2]`
coordinates on a 1000x1000 grid relative to the selected image dimensions. The
adapter converts to x/y only at crop execution. There are 10 voluntary tool
rounds/10 crop images, no forced reasoning prefix, no forced final answer and
no critic retry. Final answers use `<answer>...</answer>`; choice extraction
for VStar/HRBench uses the existing benchmark scorers. Pinpoint uses the same
VisualProbe rubric as the reference suite.

Thinking mode enables `enable_thinking=true`, `preserve_thinking=true`, and
`--reasoning-parser gemma4`, and replays separate API reasoning in tool history.
Nonthinking uses `enable_thinking=false` and no reasoning parser. Sampling is
24,576 max output tokens/request, temperature 0.7, top-p 0.8, top-k 20,
repetition penalty 1 and presence penalty 0. Both modes use the existing Gemma
runtime: 49,152 context tokens, 1,120 soft image tokens/image, two workers and
an isolated Transformers overlay over CUDA `vero2_eval`. Model-native processing
replaces Qwen's max-pixel settings. Pinpoint judging shares the policy GPU
sequentially, using eager execution and Triton GDN prefill.

The shared CPU preflight now accepts `--tasks`, supports the Pinpoint custom
dataset builder, verifies expected full split counts, and validates each selected
image, question, resolved prompt/schema and native first/after-crop rendering.
It runs inside each GPU job before serving. A separate submission-time CPU
check can use `LIMIT=2` to inspect representative images while checking all split
counts; submission always exports `LIMIT=''` for full coverage.

Outputs default to:

```text
/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_gemma_zoom/generalization_v1/
  gemma4_e4b/
  gemma4_e4b_thinking/
```

Each contains `submission_plan.json`, `submission.json`, `job_id.txt`, Slurm
logs, `protocol/manifest.json`, exact prompts/schema/templates, request traces
with images and parsed calls, prejudge samples, scored sample JSONLs, final
result JSON, and `zoom_test_summary.json`. The summary is shared with Gemma
VisualProbe and applies to all three benchmarks despite its historical name.
Submission and CPU checks do not establish GPU completion; verify all three
task counts in the saved results and samples.

## Thinking only, with no tools

```bash
bash examples/submit_zoom_eval_gemma4_generalization.sh --modes thinking_only \
  --output-root /scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_gemma_zoom/thinking_only_generalization_v1
```

This submits one full 1,424-example Gemma 4 E4B job. It enables native thinking
and the reasoning parser, with MCP, crop calls and server tool parsing disabled.
The new task variants retain the selected200 Qwen baseline populations, question
formatting and scorers, with Qwen3-VL thinking-only sampling (16,384 output
tokens including reasoning, temperature 1, top-p 0.95, top-k 20 and presence
penalty 1.5). There is no added system prompt or tool/answer-tag instruction.
`--preflight`, `--dry-run` and `--manifest` apply to this mode as well.
