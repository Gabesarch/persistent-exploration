# Gemma VisualProbe normal zoom evaluation

For full Gemma 4 thinking/nonthinking Pinpoint, VStar and HRBench-8K runs,
see [README_gemma4_generalization.md](README_gemma4_generalization.md).

`submit_zoom_eval_gemma_visual_probe.sh` submits one job per policy model. Defaults:
`gputest`, `gpu80`, one GPU, eight CPUs, 180G host RAM, one hour, and **15 examples
per model: the first five Easy, first five Medium, and first five Hard examples**.
Both models see the same population. They choose whether and when to zoom.
On Della the submission plugin rejects an explicit `--partition=gputest`.
For this target the launcher omits `--partition`; the one-hour GPU request is
automatically assigned to `Partition=gputest`, `QOS=gpu-test` by the plugin.
There is no forced tool call, forced reasoning prefix, forced final answer, or
critic retry. These are new base-model conditions, not Vero training-protocol runs.

## Cache and runtime preparation

Run from `/home/gs1693/repo/vero2/vero_eval` on a login node with network access:

```bash
PY=/scratch/gpfs/ZHUANGL/gs1693/conda/vero2_eval/bin/python
HF_HUB_OFFLINE=0 TRANSFORMERS_OFFLINE=0 HF_DATASETS_OFFLINE=0 \
  HF_HOME=/scratch/gpfs/ZHUANGL/gs1693/hf_cache \
  "$PY" scripts/cache_gemma_zoom_models.py
"$PY" scripts/cache_gemma_zoom_models.py --verify-only
"$PY" scripts/prepare_gemma_zoom_runtime.py
```

For Gemma 3, accept Google's model terms and authenticate with `hf auth login` if
your Hugging Face account does not already have access. Use the `hf` executable
in the same environment. The cache command resumes interrupted downloads and
leaves the model's native template unchanged. To download just one model, add
`--model gemma3_4b` or `--model gemma4_e4b`. Model locations:

- `/scratch/gpfs/ZHUANGL/gs1693/hf_cache/google/gemma-3-4b-it`
- `/scratch/gpfs/ZHUANGL/gs1693/hf_cache/google/gemma-4-E4B-it`

The existing CUDA `vero2_eval` environment has vLLM 0.23.0, Torch 2.11.0+cu129,
and Transformers 5.3.0. Its vLLM knows Gemma 4, but its Transformers does not.
`prepare_gemma_zoom_runtime.py` snapshots **only** the installed pure-Python
Transformers 5.8.0.dev0 distribution from `reason-vlm-gemma4` into
`/scratch/gpfs/ZHUANGL/gs1693/vero2/runtime/gemma_zoom_transformers`.
The jobs prepend this isolated overlay to `PYTHONPATH`. They retain CUDA Torch
and vLLM from `vero2_eval`; neither Conda environment is modified. The source
and distribution name are recorded in `runtime_source.json`.

VisualProbe Easy/Medium/Hard and the Qwen3.5-27B judge are already in the shared
cache. GPU jobs run offline. Their CPU preflight opens every selected image
before starting the policy server.

## Submission

```bash
DRY_RUN=1 bash examples/submit_zoom_eval_gemma_visual_probe.sh
bash examples/submit_zoom_eval_gemma_visual_probe.sh
```

To select one model, set `GEMMA_MODELS=gemma3_4b` or `GEMMA_MODELS=gemma4_e4b`.
For a separate Gemma 4 thinking test, run:

```bash
GEMMA_MODELS=gemma4_e4b GEMMA4_ENABLE_THINKING=true \
  bash examples/submit_zoom_eval_gemma_visual_probe.sh
```

This sets `enable_thinking=true`, `preserve_thinking=true`, and
`--reasoning-parser gemma4`, alongside `--tool-call-parser gemma4`. The driver
replays the API's separate `reasoning` field in assistant tool-call history.
Exact API reasoning is retained in request traces; the summary counts reasoning
turns. The variant gets a separate `_thinking` output directory and job name.
The system instruction, sample population, coordinates, sampling and judge match
the non-thinking test. Native thinking activation is supplied by the chat template.
`PARTITION`, `CONSTRAINT`, `TIME_LIMIT`, `ACCOUNT`, `OUTPUT_BASE`,
`GEMMA3_MODEL_PATH`, and `GEMMA4_MODEL_PATH` are overridable. `LIMIT_PER_TASK=5`
means 15 total samples per model, not 15 per difficulty. For full evaluation:

```bash
LIMIT_PER_TASK='' PARTITION=pli-c TIME_LIMIT=06:00:00 \
  bash examples/submit_zoom_eval_gemma_visual_probe.sh
```

## Exact prompts and output contracts

For **both** models the base system instruction is exactly the UTF-8 contents of
`prompts/tools/system_gemma_zoom.txt`, stripped of surrounding whitespace.
For **Gemma 3 only**, the stripped contents of
`prompts/tools/system_gemma3_zoom_format.txt` are appended with **two newlines**.
There are no other system fragments. The exact effective system string and its
SHA-256 are saved in each job's `protocol/system_prompt.txt` and
`protocol/manifest.json`. The user message contains the image, a newline, the
original dataset question, and `Image 0 is available as img_idx=0.`.

Gemma 3 uses `prompts/gemma3_zoom_tools.jinja`, preserving Gemma's native
`<start_of_turn>user/model`, `<end_of_turn>`, and `<start_of_image>` tokens.
Text parts use Google's native whitespace trimming. A leading newline after an
image marker otherwise merges with the expanded image suffix and prevents vLLM
0.23 from finding the image span. The CPU preflight exercises vLLM preprocessing
with one original image and an original-plus-crop conversation, using both text
and token-ID prompts, and verifies 256 embedding tokens per image.
The logical system message is folded into the first user turn, then the template
appends `# Tools\n<tools>\n`, one JSON OpenAI function schema per line, and
`</tools>\n\n`. Tool-call responses and crop images are merged into one user turn.
vLLM's **Hermes** parser interprets model-generated calls:

```text
<tool_call>{"name":"image_zoom_in_tool","arguments":{"bbox_2d":[100,200,500,600],"label":"region of interest","img_idx":0}}</tool_call>
```

Gemma 4 uses the **native `chat_template.jinja` from its cached E4B checkpoint**,
vLLM's **gemma4** parser, and `enable_thinking=false`. The native template renders
the instruction in a system turn and serializes the same function schema in
Gemma's native format. An example raw model-generated call is:

```text
<|tool_call>call:image_zoom_in_tool{bbox_2d:[100,200,500,600],img_idx:0,label:<|"|>region of interest<|"|>}<tool_call|>
```

For both models, the server converts calls into the same OpenAI response shape:

```json
{"tool_calls":[{"type":"function","function":{"name":"image_zoom_in_tool","arguments":"{\"bbox_2d\":[100,200,500,600],\"label\":\"region of interest\",\"img_idx\":0}"}}]}
```

The driver executes the existing MCP crop tool. Model-facing bounding boxes are `[y1,x1,y2,x2]`
on a **1000×1000 grid relative to the selected image dimensions**. The adapter
converts to the existing crop tool's `[x1,y1,x2,y2]` order before cropping;
model-visible call logs and tool responses retain y/x order. A crop into a previous crop is composited back onto the
original image at native resolution. The text tool response is:

```text
Zoomed in on the image to the region [100, 200, 500, 600] with label region of interest. The cropped view is now available as img_idx 1.
```

The crop image follows in a user message with
`Above are the tool-returned image(s).`. Gemma 3 wraps the text in
`<tool_response>...</tool_response>`; Gemma 4 uses its native tool-response format.
The normal tool loop offers up to ten tool rounds and ten crop images.
It does not force the model to use the tool. Both models finish with:

```text
<answer>short, specific answer</answer>
```

Sampling matches the source normal-zoom tasks: temperature 0.7, top_p 0.8,
top_k 20, repetition penalty 1, presence penalty 0, up to 24,576 output tokens
per request. Policy context is 49,152 tokens. Gemma 3 uses its native 896x896
image processor with pan-and-scan disabled; Gemma 4 uses 1,120 soft image tokens.
Full original images are sent losslessly; native image processors determine the
model's actual representation. Qwen's `max_pixels`/`min_pixels` options are not
passed to Gemma. The single GPU is reused sequentially for the unchanged
Qwen3.5-27B no-think judge and the original VisualProbe judging rubric.

## Inspecting a test

Outputs default to
`/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_gemma_zoom/test_<UTC timestamp>/<model>/`:

- `job_id.txt`, `slurm_<job>.out/.err`, and `runtime_logs/`: submission and runtime.
- `protocol/`: exact prompts, schema, templates, versions, selected sample IDs,
  image sizes, question text, and rendered first/after-crop prompt examples.
- `request_traces/<sample>/`: one JSON per actual API turn, retaining lossless
  image data URLs, conversation messages, parsed calls, completions and usage.
- `prejudge_samples/`: policy outputs retained before judge execution.
- Standard lmms result JSON and sample JSONL files.
- `zoom_test_summary.json`: observed tool calls, crop responses, answer tags,
  truncations and result-file paths. Regenerate it even after a partial failure:

```bash
python scripts/summarize_gemma_zoom_test.py /path/to/model/output
```

CPU preflight and successful submission do not establish GPU inference success.
Check that all 15 samples completed, crops were returned, final answers exist,
and final judge result artifacts were saved. A run with zero voluntary zooms
does not demonstrate crop round-trip behavior.

## Separate Gemma4 E4B plots

```bash
bash /home/gs1693/repo/vero2/vero_eval/examples/plot_eval_accuracy_over_steps_qwen35_27b_judge_gemma4_e4b.sh
```

This produces separate accuracy and average-tool-call PNG/CSV files in
`examples/plots`, each with easy, medium, hard and sample-weighted overall panels.
Only full Visual Probe results (141/268/106 examples, with no sample limit) enter
the plots; smoke tests, Gemma3 and generalization benchmarks are excluded.
The current pretrained thinking/non-thinking results appear as comparisons.
Once checkpoint results under `global_step_N` folders exist beneath
`EVAL_OUTPUT_ROOT`, they appear as training curves with dotted pretrained
references. Baselines have blank CSV steps and are never invented step-zero
training evaluations. No checkpoint evaluations are submitted by plotting.

The default root is `/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_gemma_zoom`.
`OUTPUT_DIR`, `OUTPUT_BASENAME`, `TOOL_CALLS_OUTPUT_BASENAME`, `METRIC`, `PERCENT`,
`LATEST_BY`, `TITLE`, `TOOL_CALLS_TITLE` and `PYTHON` are overridable. To use
checkpoint results outside that root, set `CONDITION_GDPO_DIR` and/or
`CONDITION_CRITIC_DIR`, or pass `--condition 'Label=/path/to/run'`. Optional
`BASELINE_NONTHINKING_DIR` / `BASELINE_THINKING_DIR` replace the matching
automatically discovered reference. The accuracy `_sources.json` records the
selected result files and tool-count sources. When final logs omit `resps`,
tool counts use pre-judge trajectories matched by both document ID and hash;
missing trajectories are reported instead of counted as zero.

The job creates a private temporary directory at `/tmp/gemma.<jobid>.<suffix>`
and sets both `TMPDIR` and `VLLM_RPC_BASE_PATH` to it. vLLM appends UUIDs to
its IPC socket paths; using the long evaluation output path as the temporary
directory exceeds Linux's 107-byte Unix socket path limit and prevents startup.
Logs, request traces, and evaluation results remain in the output directory.

## Thinking only, with no tools

For a full Gemma 4 E4B thinking-only Visual Probe run:

```bash
GEMMA_EVAL_MODE=thinking_only LIMIT_PER_TASK='' PARTITION=pli-c ACCOUNT=pli \
  CONSTRAINT=h100 TIME_LIMIT=06:00:00 \
  OUTPUT_BASE=/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_gemma_zoom/thinking_only_visual_probe_v1 \
  bash examples/submit_zoom_eval_gemma_visual_probe.sh
```

This selects only Gemma 4, activates its native thinking template and reasoning
parser, disables MCP and server tool parsing, and permits zero crop calls.
The `*_gemma4_thinking_only` tasks inherit Qwen3-VL thinking-only question
prompts and sampling: 16,384 output tokens including reasoning, temperature 1,
top-p 0.95, top-k 20, presence penalty 1.5. There is no added system text,
zoom instruction, image-index hint, answer-tag instruction, forced final or retry.
The native Gemma template supplies the thinking activation. Full Visual Probe
contains 141 easy, 268 medium and 106 hard examples (515 total).
