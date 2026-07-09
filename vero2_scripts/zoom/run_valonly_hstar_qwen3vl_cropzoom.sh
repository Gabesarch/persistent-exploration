#!/bin/bash
#SBATCH --job-name=valonly_hstar
#SBATCH --partition=pli-c
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=968G
#SBATCH --time=24:00:00
#SBATCH --output=slurm/output/%x_%j.log
#SBATCH --error=slurm/error/%x_%j.log
#
# In-loop validation smoke for hstarbench RL split (panorama strip + zoom tool).
# Uses reward_type=hstar_strip (cone success) and val_kwargs aligned with
# hstarbench_mixed_rl_reasoning_zoom (temp=0.7, n=1).

set -x
set -o pipefail

SET_PATHS_SH="./set_paths.sh"
if [ ! -f "$SET_PATHS_SH" ]; then
  echo "error: set_paths.sh not found at $SET_PATHS_SH" >&2
  exit 2
fi
source "$SET_PATHS_SH"

ENGINE="vllm"
case "${1:-}" in
  vllm|sglang|trtllm)
    ENGINE="$1"
    shift
    ;;
esac
HYDRA_ARGS=("$@")

export HF_CACHE="${HF_CACHE:-$HF_HOME}"
export MODEL_PATH="${MODEL_PATH:-$HF_CACHE/Qwen/Qwen3-VL-8B-Instruct}"
VAL_FILE="/scratch/gpfs/ZHUANGL/shared/Hstarbench/data/verl/hstarbench_mixed_rl.verl.jsonl"
SYSTEM_PROMPT="$VERL_ROOT/vero2_scripts/prompts/system_prompt_chatting_tool_calling.txt"

if [ ! -f "$VAL_FILE" ]; then
  echo "Building verl JSONL from hstarbench RL split..."
  python3 "$VERO_EVAL_DIR/lmms_eval/tasks/hstarbench/build_hstarbench_verl_jsonl.py" --source rl
fi

for required_path in "$MODEL_PATH" "$VAL_FILE" "$SYSTEM_PROMPT"; do
  if [ ! -e "$required_path" ]; then
    echo "error: required path does not exist: $required_path" >&2
    exit 2
  fi
done

DATE_TIME="${DATE_TIME:-$(date +%Y%m%d_%H%M%S)}"
RUN_UNIQUE_ID="${RUN_UNIQUE_ID:-${SLURM_JOB_ID:-pid_$$}}"
NUM_GPUS="${NUM_GPUS:-${SLURM_GPUS_ON_NODE:-8}}"

TRAINER_PROJECT_NAME="${TRAINER_PROJECT_NAME:-vero2_verl}"
TRAINER_EXPERIMENT_SUFFIX="${TRAINER_EXPERIMENT_SUFFIX:-valonly_hstar_rl}"
TRAINER_EXPERIMENT_NAME="${TRAINER_EXPERIMENT_NAME:-${TRAINER_EXPERIMENT_SUFFIX}_${DATE_TIME}_${RUN_UNIQUE_ID}}"
TRAINER_DEFAULT_LOCAL_DIR="${TRAINER_DEFAULT_LOCAL_DIR:-${SAVE_PATH}/checkpoints/${TRAINER_PROJECT_NAME}/${TRAINER_EXPERIMENT_NAME}}"

mkdir -p slurm/output slurm/error logs
export PYTHONPATH="$VERL_ROOT${PYTHONPATH:+:$PYTHONPATH}"

python3 -m verl.trainer.main_ppo \
  --config-path="$VERL_ROOT/vero2_scripts/config" \
  --config-name=valonly_hstar_qwen3vl_cropzoom \
  trainer.project_name="$TRAINER_PROJECT_NAME" \
  trainer.experiment_name="$TRAINER_EXPERIMENT_NAME" \
  trainer.default_local_dir="$TRAINER_DEFAULT_LOCAL_DIR" \
  trainer.n_gpus_per_node="$NUM_GPUS" \
  actor_rollout_ref.actor.fsdp_config.fsdp_size="$NUM_GPUS" \
  actor_rollout_ref.rollout.agent.num_workers="$NUM_GPUS" \
  data.val_files="$VAL_FILE" \
  data.train_files="$VAL_FILE" \
  "${HYDRA_ARGS[@]}"
