#!/usr/bin/env bash
# Run one paper benchmark directly on a GPU machine.
# Example:
#   bash persistent-exploration-eval/examples/evaluate_checkpoint.sh --task visual_probe \
#     --judge /path/to/Qwen3.5-27B --output /path/to/results
set -euo pipefail

EVAL_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
REPO_ROOT="$(cd -- "$EVAL_DIR/.." && pwd)"

usage() {
  cat <<'EOF'
Usage: evaluate_checkpoint.sh --task TASK [options]

TASK: all, visual_probe, visual_probe_easy, visual_probe_medium,
  visual_probe_hard, pinpoint, vstar, or o3_bench

Options:
  --checkpoint PATH  Local checkpoint directory or published model ID
  --judge PATH       Qwen3.5-27B model directory
  --judge-tp N       Tensor parallel GPUs for the judge (default: 1)
  --output PATH      Result directory (default: runtime/outputs/eval/TASK)
  --limit N          Evaluate N examples per selected task (smoke test)
  --offset N         Start at example N in each selected task (default: 0)
  --dry-run          Print the planned command without running it
  -h, --help         Show this help

The default checkpoint is gsarch/Persistent-Exploration-Qwen3VLi at its
pinned revision. It downloads to ROOT_PATH/models when needed.
Optionally set VERO2_EVAL_CONDA_ENV for a separate eval environment and
HF_HOME for the benchmark cache. Run on a GPU machine.
EOF
}

task=""
checkpoint="gsarch/Persistent-Exploration-Qwen3VLi"
judge=""
judge_tp="${CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE:-1}"
output=""
limit=""
offset="${OFFSET:-0}"
dry_run=0
while (($#)); do
  case "$1" in
    --task|--checkpoint|--judge|--judge-tp|--output|--limit|--offset)
      (($# >= 2)) || { echo "Missing value for $1" >&2; exit 2; }
      case "$1" in
        --task) task="$2";;
        --checkpoint) checkpoint="$2";;
        --judge) judge="$2";;
        --judge-tp) judge_tp="$2";;
        --output) output="$2";;
        --limit) limit="$2";;
        --offset) offset="$2";;
      esac
      shift 2;;
    --dry-run) dry_run=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "Unknown option: $1" >&2; usage >&2; exit 2;;
  esac
done

[[ -n "$task" ]] || { usage >&2; exit 2; }
case "$task" in
  all) tasks=visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom,pinpoint_bench_reasoning_zoom,vstar_bench_reasoning_zoom,o3_bench_reasoning_zoom;;
  visual_probe) tasks=visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom;;
  visual_probe_easy) tasks=visual_probe_easy_reasoning_zoom;;
  visual_probe_medium) tasks=visual_probe_medium_reasoning_zoom;;
  visual_probe_hard) tasks=visual_probe_hard_reasoning_zoom;;
  pinpoint) tasks=pinpoint_bench_reasoning_zoom;;
  vstar) tasks=vstar_bench_reasoning_zoom;;
  o3_bench) tasks=o3_bench_reasoning_zoom;;
  *) echo "Unknown task: $task" >&2; usage >&2; exit 2;;
esac
[[ -z "$limit" || "$limit" =~ ^[1-9][0-9]*$ ]] || { echo "--limit must be a positive integer" >&2; exit 2; }
[[ "$offset" =~ ^(0|[1-9][0-9]*)$ ]] || { echo "--offset must be a nonnegative integer" >&2; exit 2; }
[[ "$judge_tp" =~ ^[1-9][0-9]*$ ]] || { echo "--judge-tp must be a positive integer" >&2; exit 2; }

root_path="${ROOT_PATH:-$REPO_ROOT/runtime}"
hf_home="${HF_HOME:-$root_path/hf_cache}"
judge="${judge:-$hf_home/Qwen/Qwen3.5-27B}"
output="${output:-$root_path/outputs/eval/$task}"
export PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
if ((!dry_run)) && [[ -n "${VERO2_EVAL_CONDA_ENV:-}" ]]; then
  export PATH="$VERO2_EVAL_CONDA_ENV/bin:$PATH"
  export CONDA_PREFIX="$VERO2_EVAL_CONDA_ENV"
  # set_paths.sh.example would otherwise try to activate again.
  unset VERO2_EVAL_CONDA_ENV
fi
release_model=0
download_pending=0
case "$checkpoint" in
  https://huggingface.co/gsarch/Persistent-Exploration-Qwen3VLi|gsarch/Persistent-Exploration-Qwen3VLi)
    release_model=1
    model_dir="$root_path/models/Persistent-Exploration-Qwen3VLi"
    if [[ -s "$model_dir/model.safetensors" && -r "$model_dir/model.safetensors" \
          && -f "$model_dir/eval/run_visual_probe.sh" \
          && -f "$model_dir/eval/judge_no_think.jinja" \
          && -f "$model_dir/chat_template.jinja" ]]; then
      echo "Using downloaded model: $model_dir"
    elif ((dry_run)); then
      echo "Would download gsarch/Persistent-Exploration-Qwen3VLi to $model_dir"
      download_pending=1
    else
      mkdir -p "$model_dir"
      HF_HOME="$hf_home" HF_HUB_OFFLINE=0 python - "$model_dir" <<'PY'
import sys
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="gsarch/Persistent-Exploration-Qwen3VLi",
    revision="aeda0df1c4ce394ed7ce0d42719bce8e2641bc08",
    local_dir=sys.argv[1],
)
PY
    fi
    checkpoint="$model_dir";;
esac
if ((download_pending == 0)); then
  [[ -d "$checkpoint" ]] || { echo "Missing checkpoint: $checkpoint" >&2; exit 2; }
fi
[[ -f "$judge/config.json" && -f "$judge/tokenizer_config.json" ]] || {
  echo "Missing judge model metadata: $judge" >&2; exit 2;
}

# The training checkpoint can contain an unreadable HF export but readable
# FSDP shards. Merge the shards into a directory owned by the caller.
has_readable_weights() {
  local file
  for file in "$1"/model.safetensors "$1"/model.safetensors-* "$1"/pytorch_model.bin; do
    [[ -f "$file" && -s "$file" && -r "$file" ]] && return 0
  done
  return 1
}

has_readable_weights "$judge" || { echo "No readable judge weights in $judge" >&2; exit 2; }

if ((download_pending == 0)) && ! has_readable_weights "$checkpoint"; then
  case "$checkpoint" in
    */global_step_*/actor/huggingface) actor_dir="${checkpoint%/huggingface}";;
    */global_step_*/actor) actor_dir="$checkpoint";;
    */global_step_*) actor_dir="$checkpoint/actor";;
    *) echo "No readable model weights in $checkpoint" >&2; exit 2;;
  esac
  if [[ -n "$actor_dir" ]]; then
    shards=("$actor_dir"/model_world_size_*_rank_*.pt)
    [[ -r "${shards[0]}" ]] || { echo "No readable FSDP shards in $actor_dir" >&2; exit 2; }
    merged="$output/merged_checkpoint"
    if has_readable_weights "$merged"; then
      echo "Reusing merged checkpoint: $merged"
    elif ((dry_run)); then
      echo "Would merge FSDP shards from $actor_dir to $merged"
    else
      mkdir -p "$output"
      python -m verl.model_merger merge --backend fsdp --local_dir "$actor_dir" \
        --target_dir "$merged" --trust-remote-code
    fi
    checkpoint="$merged"
  fi
fi

if ((!dry_run)); then
  [[ -f "$checkpoint/config.json" && -f "$checkpoint/tokenizer_config.json" ]] || {
    echo "Missing checkpoint model metadata: $checkpoint" >&2; exit 2;
  }
  command -v nvidia-smi >/dev/null && nvidia-smi -L >/dev/null 2>&1 || {
    echo "No GPU detected. Run this script on a GPU machine." >&2; exit 2;
  }
  mkdir -p "$output"
fi

export VERO_EVAL_DIR="$EVAL_DIR" ROOT_PATH="$root_path" HF_HOME="$hf_home"
export MODEL=vero CHECKPOINT_PATH="$checkpoint" TASKS="$tasks" TOOLS=on
export OUTPUT_ROOT="$output" JUDGE_BACKEND=engine API_TYPE=vllm
export JUDGE_MODEL_PATH="$judge" VLLM_JUDGE_MODEL_PATH="$judge"
export VLLM_JUDGE_CHAT_TEMPLATE="$judge/chat_template_no_think.jinja"
export CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE="$judge_tp"
export SYSTEM_PROMPT="$REPO_ROOT/persistent-exploration-rl/prompts/system_prompt_chatting_tool_calling.txt"
export RUN_SYSTEM_PROMPT_MODE=run_env TOOL_CALL_PARSER=hermes
export FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS=true
export CRITIC_GUIDED_ANSWER_RETRY=true CRITIC_GUIDED_ANSWER_RETRY_SOURCE=policy
export CRITIC_GUIDED_ANSWER_RETRY_MODE=token
export CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD="${CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD:-0.5}"
export CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE=persist
export MAX_TOOL_ITERATIONS=10 MAX_TOOL_IMAGES=10
export VLLM_JUDGE_ENFORCE_EAGER=true VLLM_JUDGE_GDN_PREFILL_BACKEND=triton
unset LIMIT
if ((release_model)) || [[ -f "$checkpoint/eval/run_visual_probe.sh" ]]; then
  # Match the published checkpoint's packaged evaluation profile.
  export CHAT_TEMPLATE="$checkpoint/chat_template.jinja"
  export LMMS_EVAL_HF_TOKENIZER="$checkpoint"
  export VLLM_JUDGE_CHAT_TEMPLATE="$checkpoint/eval/judge_no_think.jinja"
  export RUN_SYSTEM_PROMPT_MODE=task_metadata
  export VLLM_JUDGE_ENFORCE_EAGER=false VLLM_JUDGE_GDN_PREFILL_BACKEND=auto
  export MODEL_NAME=qwen_vllm MODEL_VERSION=qwen_vllm WORKERS=10
  export MM_PROCESSOR_KWARGS='{"max_pixels":10000000}'
  export TOOL_IMAGE_AS_USER_TURN=false LMMS_EVAL_PRESERVE_REASONING=false
  export VLLM_USE_FLASHINFER_SAMPLER=0 BATCH_SIZE=1 ALWAYS_RETRY_ANSWERS=false
  export FORCE_TOOL_CALLS_UNTIL_FINAL_TURN=false
  export FORCE_FINAL_ANSWER_PREFIX='' FIRST_CANDIDATE_ANSWER_TURN=0 FOLLOWUP_CANDIDATE_ANSWER_TURN=0
  unset SYSTEM_PROMPT VERO_TRAINING_CONFIG GEMMA_TRAINING_CONFIG CUSTOM_CHAT_TEMPLATE
  unset DEFAULT_CHAT_TEMPLATE_KWARGS REASONING_PARSER LIMIT_MM_PER_PROMPT VLLM_MAX_NUM_SEQS
  unset VERO_EVAL_CRITIC_DOMAIN_THRESHOLDS VERO_EVAL_DISABLE_CRITIC VERO_EVAL_CRITIC_THRESHOLD
fi
[[ -z "$limit" ]] || export LIMIT="$limit"
export OFFSET="$offset"

if ((dry_run)); then
  if ((release_model)) || [[ -f "$checkpoint/eval/run_visual_probe.sh" ]]; then
    echo "Profile: published checkpoint eval settings"
  fi
  printf 'TASKS=%q CHECKPOINT_PATH=%q JUDGE_MODEL_PATH=%q JUDGE_TP=%q OUTPUT_ROOT=%q' \
    "$TASKS" "$CHECKPOINT_PATH" "$JUDGE_MODEL_PATH" "$judge_tp" "$OUTPUT_ROOT"
  [[ -z "$limit" ]] || printf ' LIMIT=%q' "$limit"
  printf ' OFFSET=%q' "$offset"
  printf ' bash %q\n' "$EVAL_DIR/scripts/run_eval.sh"
else
  bash "$EVAL_DIR/scripts/run_eval.sh"
  result_file="$(find "$output" -type f -name '*_results.json' -print | sort | tail -n 1)"
  if [[ -n "$result_file" ]]; then
    python - "$result_file" <<'PY'
import json
import sys

path = sys.argv[1]
data = json.load(open(path))
correct = total = 0.0
print(f"Results: {path}")
for name, metrics in sorted(data.get("results", {}).items()):
    value = metrics.get("judge_accuracy,none", metrics.get("vstar_overall_acc,none"))
    if value is None:
        continue
    count = data.get("n-samples", {}).get(name, {}).get("effective", 0)
    print(f"  {name}: {100 * value:.1f}% ({count} examples)")
    if name.startswith("visual_probe_"):
        correct += value * count
        total += count
if total:
    print(f"  visual_probe_overall: {100 * correct / total:.1f}% ({int(total)} examples)")
PY
  fi
fi
