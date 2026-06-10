#!/bin/bash
#SBATCH --job-name=eval_ckpt_array
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=4:00:00
#SBATCH --mem=80G
#SBATCH --cpus-per-task=10
#SBATCH --gres=gpu:2
#SBATCH --partition=pli
#SBATCH --account=vlmrl
#SBATCH --output=vero_eval/logs/eval_ckpt_%A_%a.out
#SBATCH --error=vero_eval/logs/eval_ckpt_%A_%a.err
#
# TYPE 2 — one SLURM array task per checkpoint: resolve -> serve -> eval one domain ->
# stop. Submit via examples/submit_eval_checkpoints.sh (sets --array + the env below).
# Each task owns its own external vLLM server (a checkpoint is baked in at serve time
# and pins 0.9 GPU util, so checkpoints cannot share a GPU).
#
# Required env: CHECKPOINT_INDEX_PATH (one checkpoint dir per line), DOMAIN, MODEL_SUFFIX,
#   RUN_ROOT. Optional: PORT(9001), NUM_GPUS(1), JUDGE_BACKEND(server), LIMIT,
#   MCP_SERVER_PATH, SYSTEM_PROMPT, MAX_TOOL_ITERATIONS(10), MERGE_BACKEND(fsdp), SKIP_IF_EXISTS(1).
set -euo pipefail

VERO_EVAL_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}/vero_eval"
[[ -f "$VERO_EVAL_DIR/set_paths.sh.example" ]] || { echo "ERROR: $VERO_EVAL_DIR not found (submit from repo root)" >&2; exit 1; }
set +u
[[ -f "$VERO_EVAL_DIR/set_paths.sh" ]] && source "$VERO_EVAL_DIR/set_paths.sh" || source "$VERO_EVAL_DIR/set_paths.sh.example"
set -u
source "$VERO_EVAL_DIR/examples/_domain_map.sh"

: "${CHECKPOINT_INDEX_PATH:?required}"; : "${DOMAIN:?required}"; : "${MODEL_SUFFIX:?required}"; : "${RUN_ROOT:?required}"
PORT="${PORT:-9001}"; API_KEY="${API_KEY:-qwen}"; NUM_GPUS="${NUM_GPUS:-1}"
export JUDGE_BACKEND="${JUDGE_BACKEND:-server}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-1}"; MERGE_BACKEND="${MERGE_BACKEND:-fsdp}"
cd "$VERO_EVAL_DIR"

# 1) Index -> this task's checkpoint dir.
TASK_ID="${SLURM_ARRAY_TASK_ID:-0}"
CKPT_DIR="$(sed -n "$((TASK_ID + 1))p" "$CHECKPOINT_INDEX_PATH" || true)"
[[ -z "$CKPT_DIR" || ! -d "$CKPT_DIR" ]] && { echo "No checkpoint for array index $TASK_ID"; exit 0; }
CKPT_NAME="$(basename "$CKPT_DIR")"
RAW_DIR="$RUN_ROOT/raw/$CKPT_NAME"; mkdir -p "$RAW_DIR" "$VERO_EVAL_DIR/logs"
if [[ "$SKIP_IF_EXISTS" == 1 ]] && ls "$RAW_DIR"/*/*results*.json >/dev/null 2>&1; then
  echo "[skip] results already exist for $CKPT_NAME"; exit 0
fi

# 2) Resolve a Hugging Face weights dir (merge FSDP shards if needed).
HF_ACTOR="$CKPT_DIR/actor"; HF_TARGET="$HF_ACTOR/huggingface"
if ls "$HF_TARGET"/*.safetensors "$HF_TARGET"/*.bin >/dev/null 2>&1; then
  CHECKPOINT_PATH="$HF_TARGET"
elif [[ -d "$HF_ACTOR" ]]; then
  echo "[INFO] Merging FSDP shards -> $HF_TARGET (backend=$MERGE_BACKEND)"
  ( flock 200
    if ! ls "$HF_TARGET"/*.safetensors >/dev/null 2>&1; then
      python -m verl.model_merger merge --backend "$MERGE_BACKEND" --local_dir "$HF_ACTOR" --target_dir "$HF_TARGET" \
        || { echo "[ERROR] merge failed; pre-export with scripts/export_verl_checkpoints.py" >&2; exit 1; }
    fi
  ) 200>"$HF_ACTOR/.merge.lock"
  CHECKPOINT_PATH="$HF_TARGET"
else
  CHECKPOINT_PATH="$CKPT_DIR"   # already an HF dir
fi
export CHECKPOINT_PATH
echo "[INFO] index=$TASK_ID checkpoint=$CKPT_NAME path=$CHECKPOINT_PATH domain=$DOMAIN suffix=$MODEL_SUFFIX"

# 3) Domain -> tasks.
TASKS="$(expand_domain "$DOMAIN" "$MODEL_SUFFIX")" || { echo "[ERROR] empty task set"; exit 1; }
echo "[INFO] tasks: $TASKS"

# 4) Serve this checkpoint.
NUM_GPUS="$NUM_GPUS" CHECKPOINT_PATH="$CHECKPOINT_PATH" PORT="$PORT" bash scripts/serve_qwen_vllm.sh
curl --max-time 5 -s -o /dev/null -H "Authorization: Bearer ${API_KEY}" "http://localhost:${PORT}/v1/models" \
  || { echo "[ERROR] vLLM not up for $CKPT_NAME"; exit 1; }

# 5) Eval the domain's tasks against the served checkpoint.
OUTPUT_PATH="$RAW_DIR" bash examples/eval_domain.sh --tasks "$TASKS" ${LIMIT:+--limit "$LIMIT"} || true

# 6) Stop this checkpoint's server before the node frees for the next array task.
pkill -f "vllm serve" || true; sleep 5
echo "[DONE] $CKPT_NAME -> $RAW_DIR"
