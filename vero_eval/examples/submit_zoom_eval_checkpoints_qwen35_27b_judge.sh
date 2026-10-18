#!/usr/bin/env bash
# Submit one SLURM eval job per checkpoint for Vero zoom-tool training runs,
# using the cached Qwen3.5-27B model as an in-process sequential LLM judge.
#
# Usage, from /home/gs1693/repo/vero2/vero_eval:
#   DRY_RUN=1 bash examples/submit_zoom_eval_checkpoints_qwen35_27b_judge.sh
#   bash examples/submit_zoom_eval_checkpoints_qwen35_27b_judge.sh
#
# Common overrides:
#   EVAL_NAME=hardonly TASK_GROUP=visual_probe MAX_CHECKPOINTS=5 bash ...
#   EVAL_NAME=all TASK_GROUP=visual_probe MAX_CHECKPOINTS=5 bash ...
#   EVAL_NAME=branch_comparison EVAL_PROFILE=full bash ...
#   TRAIN_DIR=/path/to/run TASKS=task1,task2 MAX_CHECKPOINTS=5 bash ...
#   MAX_STEPS_EVAL=200 bash ...   # only evaluate checkpoints with step <= 200
#   ONLY_EVAL_EVERY_X=40 ONLY_EVAL_GLOBAL_STEPS=80,120 bash ...
#   EVAL_PROFILE=debug bash ...   # default GPU test queue/gpu80, 1h, limit=100, 1 GPU
#   EVAL_PROFILE=full bash ...    # default pli-c/h100 config
#   EVAL_PROFILE=vlmrl bash ...   # pli partition with vlmrl account
#   EVAL_PROFILE=ailab bash ...   # ailab partition with no account
#   MERGE_BACKEND=fsdp bash ...   # convert FSDP shards to HF if actor/huggingface lacks weights
#   PORT_OFFSET=200 bash ...      # optional stable offset added to all per-job ports
#   FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS=true bash ...  # force one final <answer> call at tool cap
#   EVAL_NAME=branchstrata20_fresh_lr1e5_critic \
#     CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD=0.5 ONLY_EVAL_GLOBAL_STEPS=10,20 bash ...
#   Candidate schedule v1 defaults to first candidate by turn 5, global critic
#   threshold 0.5, and recorded candidate answers. FIRST_CANDIDATE_ANSWER_TURN=0
#   restores natural timing; an explicit critic threshold overrides 0.5.
#   Completed candidate checkpoints in the previous timing's output directory
#   are also skipped. Use SKIP_IF_EXISTS=false for an intentional reevaluation.
#   SKIP_IF_EXISTS=false SKIP_IF_QUEUED=false DRY_RUN=1 bash ...
set -euo pipefail

# Also accept the repository-root invocation used by the submission skill.
# Re-enter with an eval-relative $0 so grouped child invocations keep working.
if [[ -f "vero_eval/run_eval.sbatch" && -d "vero_eval/lmms_eval" ]]; then
  cd vero_eval
  exec bash examples/submit_zoom_eval_checkpoints_qwen35_27b_judge.sh "$@"
fi

VERO_EVAL_DIR="$(pwd)"
if [[ ! -f "$VERO_EVAL_DIR/run_eval.sbatch" || ! -d "$VERO_EVAL_DIR/lmms_eval" ]]; then
  echo "ERROR: run this script from /home/gs1693/repo/vero2/vero_eval:" >&2
  echo "  bash examples/submit_zoom_eval_checkpoints_qwen35_27b_judge.sh" >&2
  exit 1
fi

# -------------------------
# Runtime/config knobs
# -------------------------
EVAL_NAME_EXPLICIT="${EVAL_NAME+x}"
EVAL_NAME="${EVAL_NAME:-branch_comparison}"
TASK_GROUP_EXPLICIT="${TASK_GROUP+x}"
TASK_GROUP="${TASK_GROUP:-visual_probe}"
TRAIN_DIR_EXPLICIT="${TRAIN_DIR+x}"
TASKS_EXPLICIT="${TASKS+x}"
OUTPUT_ROOT_EXPLICIT="${OUTPUT_ROOT+x}"
LIMIT="${LIMIT:-}"
STRICT_TASKS="${STRICT_TASKS:-1}"
EVAL_PROFILE="${EVAL_PROFILE:-full}"
RUN_SYSTEM_PROMPT_MODE_EXPLICIT="${RUN_SYSTEM_PROMPT_MODE+x}"
RUN_SYSTEM_PROMPT_MODE="${RUN_SYSTEM_PROMPT_MODE:-task_metadata}"  # task_metadata|run_env|none

MAX_IMAGE_PIXELS="${MAX_IMAGE_PIXELS:-10000000}"
export WORKERS="${WORKERS:-10}"
ACCELERATE_PORT_BASE="${ACCELERATE_PORT_BASE:-12346}"
POLICY_PORT_BASE="${POLICY_PORT_BASE:-10001}"
PORT_OFFSET="${PORT_OFFSET:-}"
MAX_CHECKPOINTS="${MAX_CHECKPOINTS:-}"
MAX_STEPS_EVAL="${MAX_STEPS_EVAL:-}"
USE_MOST_RECENT_MAX_CHECKPOINTS="${USE_MOST_RECENT_MAX_CHECKPOINTS:-false}"
ONLY_EVAL_EVERY_X="${ONLY_EVAL_EVERY_X:-10}"
ONLY_EVAL_GLOBAL_STEPS="${ONLY_EVAL_GLOBAL_STEPS:-}"
OVERRIDE_EXISTING="${OVERRIDE_EXISTING:-true}"
SKIP_IF_EXISTS="${SKIP_IF_EXISTS:-true}"
SKIP_IF_QUEUED="${SKIP_IF_QUEUED:-true}"
DRY_RUN="${DRY_RUN:-0}"
MEM_OVERRIDE="${MEM_OVERRIDE:-}"
TIME_OVERRIDE="${TIME_OVERRIDE:-}"
DEFAULT_MEM="${DEFAULT_MEM:-180G}"
MERGE_BACKEND="${MERGE_BACKEND:-fsdp}"
FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS="${FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS:-true}"
FORCE_FINAL_ANSWER_PREFIX="${FORCE_FINAL_ANSWER_PREFIX:-}"
DEFAULT_FORCE_FINAL_ANSWER_THINK_PREFILL="This is the final turn, so I need to answer from the evidence gathered so far."
FORCE_FINAL_ANSWER_THINK_PREFILL="${FORCE_FINAL_ANSWER_THINK_PREFILL:-$DEFAULT_FORCE_FINAL_ANSWER_THINK_PREFILL}"
CRITIC_GUIDED_ANSWER_RETRY_EXPLICIT="${CRITIC_GUIDED_ANSWER_RETRY+x}"
CRITIC_GUIDED_ANSWER_RETRY="${CRITIC_GUIDED_ANSWER_RETRY:-false}"
CRITIC_GUIDED_ANSWER_RETRY_SOURCE="${CRITIC_GUIDED_ANSWER_RETRY_SOURCE:-policy}"
JUDGE_BACKEND="${JUDGE_BACKEND:-engine}"
CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD="${CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD:-0.5}"
VLLM_JUDGE_MODEL_PATH="${VLLM_JUDGE_MODEL_PATH:-/scratch/gpfs/ZHUANGL/gs1693/hf_cache/Qwen/Qwen3.5-27B}"
VLLM_JUDGE_CHAT_TEMPLATE="${VLLM_JUDGE_CHAT_TEMPLATE:-$VLLM_JUDGE_MODEL_PATH/chat_template_no_think.jinja}"
# Sampling and GDN attention have separate FlashInfer controls.
VLLM_USE_FLASHINFER_SAMPLER="${VLLM_USE_FLASHINFER_SAMPLER:-0}"
VLLM_JUDGE_ENFORCE_EAGER="${VLLM_JUDGE_ENFORCE_EAGER:-true}"
VLLM_JUDGE_GDN_PREFILL_BACKEND="${VLLM_JUDGE_GDN_PREFILL_BACKEND:-triton}"

TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING="/home/gs1693/repo/vero2/vero2_scripts/prompts/system_prompt_chatting_tool_calling.txt"
TRAINING_SYSTEM_PROMPT_QWEN3_XML="$VERO_EVAL_DIR/prompts/tools/system_vero_reasoning_qwen3_xml.txt"
# Exact minio3-yp prompt at 8013b08821be150825978eb89fb7c05d0434bc09:
# 3372 UTF-8 bytes; SHA-256 f1802c9d603f78c4b192c3dbd8fc0a832e522970c72eba80bead3a56460c5b2d.

declare -A TRAIN_DIR_BY_EVAL_NAME=(
  # Standalone downloaded checkpoints; paths are model directories, not run roots.
  [qwen35_vero2_step200]="/scratch/gpfs/ZHUANGL/yz5121/vero-home/vero2-ablations/qwen3.5-vero2at200step"
  # Repaired local metadata: Qwen3-VL-8B config/tokenizer and JSON/Hermes template.
  [minio3_vero2_step100]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/standalone_eval_20261007/minio3-vero2at100step"
  [fixjson_vero2_step200]="/scratch/gpfs/ZHUANGL/yz5121/vero-home/vero2-ablations/22_5_fixjson-step_200"
  # Step-200 shard links plus verified Qwen3-VL metadata; merge in our own directory.
  [minio3_vero2_step200]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/standalone_eval_20261017/minio3-vero2-ckpt-step200"
  # Continuation arrays 14444045 and 14446126; tasks 1-10 share each root.
  [gdpo_fresh_lr1e5_noexplore]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/gdpo_fresh_lr1e5_noexplore_20260925"
  [gdpo_fresh_lr1e5_noexplore_accformat]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/gdpo_fresh_lr1e5_noexplore_accformat_20260925"
  [answer_retry_branch_donly_llmjudge_forcedprefill]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/GDPO_HDPOtool_GSPO_fp16_math_verify_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_answer_retry_branch_donly_llmjudge_forcedprefill_20260524_160610"
  [branchstrata20]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/GDPO_HDPOtool_GSPO_fp16_math_verify_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_answer_retry_branch_donly_llmjudge_forcedprefill_branchstrata20_20260524_160610"
  [branchstrata20_unmaskedprefill]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/GDPO_HDPOtool_GSPO_fp16_math_verify_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_answer_retry_branch_donly_llmjudge_forcedprefill_branchstrata20_unmaskedprefill_20260524_160610"
  [branchstrata20_restart20]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/GDPO_HDPOtool_GSPO_fp16_math_verify_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_answer_retry_branch_donly_llmjudge_forcedprefill_branchstrata20_restart20_te015_turn4k_inferenceval_20260725"
  [branchstrata20_unmasked_restart20]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/GDPO_HDPOtool_GSPO_fp16_math_verify_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_answer_retry_branch_donly_llmjudge_forcedprefill_branchstrata20_unmaskedprefill_restart20_te015_turn4k_inferenceval_20260725"
  [branchstrata20_base_noforceanswer]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/GDPO_HDPOtool_GSPO_fp16_math_verify_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_answer_retry_branch_donly_llmjudge_forcedprefill_branchstrata20_base_noforceanswer_te015_turn4k_inferenceval_20260727"
  [branchstrata20_base_noforceanswer_unmaskedprefill]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/GDPO_HDPOtool_GSPO_fp16_math_verify_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_answer_retry_branch_donly_llmjudge_forcedprefill_branchstrata20_base_noforceanswer_unmaskedprefill_te015_turn4k_inferenceval_20260727"
  # Training SLURM arrays 11827594 and 11808446, respectively.
  [teacheronly_retry_sft05_forceanswer]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/teacheronly_retry_sft05_forceanswer_20260730"
  [teacheronly_retry_sft05]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/teacheronly_retry_sft05_20260730"
  # Training SLURM array task 11974343_1.
  [branchstrata20_fresh]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_20260803"
  # Training SLURM arrays 11974347 and 12095251.
  [branchstrata20_fresh_prefixremoved_gspo]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_prefixremoved_gspo_20260803"
  # Training SLURM array task 11981184_1.
  [branchstrata20_fresh_randomturncap_p05_x5_q075]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_randomturncap_p05_x5_q075_20260803"
  # Training SLURM array 12095040.
  [branchstrata20_fresh_lr1e5]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_lr1e5_20260806"
  # Training continuation array 14651867; fresh updates after original step 80.
  [branchstrata20_fresh_lr1e5_restart_step80]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_lr1e5_restart_step80_20260928"
  [branchstrata20_fresh_lr1e5_critic]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_lr1e5_critic_20260810"
  # Continuation arrays 14777121 (5 tool rounds) and 14777129 (20 tool rounds).
  [branchstrata20_fresh_lr1e5_critic_maxcalls5]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_lr1e5_critic_maxcalls5_20260930"
  [branchstrata20_fresh_lr1e5_critic_maxcalls20]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_lr1e5_critic_maxcalls20_20260930"
  # Training continuation array 14785215; tasks share this checkpoint root.
  [branchstrata20_fresh_lr1e5_critic_candidate_schedule_v1]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_lr1e5_critic_candidate_schedule_v1_20260930"
  # Array 15328860, including its append-batch continuation; one shared run.
  [branchstrata20_fresh_lr1e5_critic_replay_schedule_v1]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_lr1e5_critic_replay_schedule_v1_20261010"
  # Training SLURM continuation array 12794189 (tasks 1-10 share this directory).
  [branchstrata20_fresh_prefixremoved_gspo_lr1e5_critic]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_prefixremoved_gspo_lr1e5_critic_20260810"
  # Training SLURM array 12095042.
  [branchstrata30_fresh]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata30_fresh_20260806"
  # Training SLURM array task 12095041_1.
  [branchstrata10_fresh]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata10_fresh_20260806"
  # Training SLURM array task 12095043_1.
  [branchstrata20_fresh_clip0005_0007]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/branchstrata20_fresh_clip0005_0007_20260806"
  [hardonly_llmjudge]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/GDPO_HDPOtool_GSPO_fp16_math_verify_chart_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_llmjudge_20260524_160610"
  [hardonly_llmjudge_lr5e6]="/scratch/gpfs/ZHUANGL/gs1693/vero2/checkpoints/vero2_verl/GDPO_HDPOtool_GSPO_fp16_math_verify_chart_qwen3vl_visual_probe_cropzoom_gdpo_hdpotool_hardonly_llmjudge_lr5e6_20260524_160610"
)

# GDPO prompt audit 2026-09-26: all 2042 train and 515 validation records
# omit system messages, so RLHFDataset inserts the file below verbatim.
# It equals prompts/tools/system_vero_reasoning.txt (task_metadata mode):
# 2980 UTF-8 bytes; SHA-256 c86c158cbbbc751309b1f20671c457dd59a3d887a69d9eb552d2394a544431c6.
# Both GDPO runs disable critic retries; retain the comparison evaluation protocol.
# Restart-step80 audit: all 2042 train and 515 validation records omit system
# messages; the same 2980-byte prompt/hash above is inserted verbatim. Answer
# retries are training-only; retain the standard comparison evaluation policy.
# Candidate-schedule audit: all 2042 train and 515 validation records omit
# system messages; the same prompt bytes/hash above apply. Candidate deadlines
# are training-only. Validation uses learned critic retries with persist prefill.
# Replay audit: the same 2042/515 records and 2980-byte solver prompt apply.
# The dedicated judge prompt has 1174 UTF-8 bytes; SHA-256
# a0d2a2024d5bbc61189f0570bc7f53c790c88c57384984bdfa713aa08efe699e.
declare -A SYSTEM_PROMPT_BY_EVAL_NAME=(
  [qwen35_vero2_step200]="$TRAINING_SYSTEM_PROMPT_QWEN3_XML"
  [minio3_vero2_step100]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [fixjson_vero2_step200]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [minio3_vero2_step200]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [gdpo_fresh_lr1e5_noexplore]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [gdpo_fresh_lr1e5_noexplore_accformat]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [answer_retry_branch_donly_llmjudge_forcedprefill]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_unmaskedprefill]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_restart20]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_unmasked_restart20]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_base_noforceanswer]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_base_noforceanswer_unmaskedprefill]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [teacheronly_retry_sft05_forceanswer]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [teacheronly_retry_sft05]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh_prefixremoved_gspo]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh_randomturncap_p05_x5_q075]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh_lr1e5]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh_lr1e5_restart_step80]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh_lr1e5_critic]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh_lr1e5_critic_maxcalls5]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh_lr1e5_critic_maxcalls20]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh_lr1e5_critic_candidate_schedule_v1]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh_lr1e5_critic_replay_schedule_v1]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh_prefixremoved_gspo_lr1e5_critic]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata30_fresh]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata10_fresh]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [branchstrata20_fresh_clip0005_0007]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [hardonly_llmjudge]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
  [hardonly_llmjudge_lr5e6]="$TRAINING_SYSTEM_PROMPT_CHATTING_TOOL_CALLING"
)

declare -A EVAL_NAMES_BY_GROUP=(
  [branch_comparison]="qwen35_vero2_step200 minio3_vero2_step100 gdpo_fresh_lr1e5_noexplore gdpo_fresh_lr1e5_noexplore_accformat hardonly_llmjudge hardonly_llmjudge_lr5e6 branchstrata20 branchstrata20_unmaskedprefill answer_retry_branch_donly_llmjudge_forcedprefill branchstrata20_restart20 branchstrata20_unmasked_restart20 branchstrata20_base_noforceanswer branchstrata20_base_noforceanswer_unmaskedprefill teacheronly_retry_sft05_forceanswer teacheronly_retry_sft05 branchstrata20_fresh branchstrata20_fresh_prefixremoved_gspo branchstrata20_fresh_randomturncap_p05_x5_q075 branchstrata20_fresh_lr1e5 branchstrata20_fresh_lr1e5_restart_step80 branchstrata20_fresh_lr1e5_critic branchstrata20_fresh_lr1e5_critic_candidate_schedule_v1 branchstrata20_fresh_prefixremoved_gspo_lr1e5_critic branchstrata30_fresh branchstrata10_fresh branchstrata20_fresh_clip0005_0007"
)
EVAL_NAMES_BY_GROUP[branch_comparison]+=" branchstrata20_fresh_lr1e5_critic_maxcalls5 branchstrata20_fresh_lr1e5_critic_maxcalls20"
EVAL_NAMES_BY_GROUP[branch_comparison]+=" branchstrata20_fresh_lr1e5_critic_replay_schedule_v1"
EVAL_NAMES_BY_GROUP[branch_comparison]+=" fixjson_vero2_step200 minio3_vero2_step200"

# Single model directories get numbered output identities for filters, dedup and plots.
declare -A SINGLE_CHECKPOINT_STEP_BY_EVAL_NAME=(
  [qwen35_vero2_step200]=200
  [minio3_vero2_step100]=100
  [fixjson_vero2_step200]=200
  [minio3_vero2_step200]=200
)
declare -A RUN_SYSTEM_PROMPT_MODE_BY_EVAL_NAME=(
  # Replace the task's JSON prompt, rather than appending an incompatible fragment.
  [qwen35_vero2_step200]=run_env
  # Exact JSON training prompt, including its trailing newline; no task fragments.
  [fixjson_vero2_step200]=run_env
  [minio3_vero2_step200]=run_env
  # Use the training file verbatim, independent of a benchmark's prompt fragments.
  [branchstrata20_fresh_lr1e5_critic_maxcalls5]=run_env
  [branchstrata20_fresh_lr1e5_critic_maxcalls20]=run_env
  [branchstrata20_fresh_lr1e5_critic_replay_schedule_v1]=run_env
)
declare -A TOOL_CALL_PARSER_BY_EVAL_NAME=(
  [qwen35_vero2_step200]=qwen3_coder
  [fixjson_vero2_step200]=hermes
  [minio3_vero2_step200]=hermes
)
declare -A CHAT_TEMPLATE_BY_EVAL_NAME=(
  [qwen35_vero2_step200]="$VERO_EVAL_DIR/prompts/vero_qwen3_xml.jinja"
  # Identical to the exported fixjson and mini-o3 step-164 templates (990af54b...).
  [fixjson_vero2_step200]="$VERO_EVAL_DIR/prompts/vero_json_step200.jinja"
  [minio3_vero2_step200]="$VERO_EVAL_DIR/prompts/vero_json_step200.jinja"
)
if [[ -z "$RUN_SYSTEM_PROMPT_MODE_EXPLICIT" ]]; then
  RUN_SYSTEM_PROMPT_MODE="${RUN_SYSTEM_PROMPT_MODE_BY_EVAL_NAME[$EVAL_NAME]:-$RUN_SYSTEM_PROMPT_MODE}"
fi
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-${TOOL_CALL_PARSER_BY_EVAL_NAME[$EVAL_NAME]:-hermes}}"
CHAT_TEMPLATE="${CHAT_TEMPLATE:-${CHAT_TEMPLATE_BY_EVAL_NAME[$EVAL_NAME]:-}}"

# A directory can appear before a download has all its weight shards. Check
# model artifacts before dispatching standalone conditions; never convert here.
standalone_checkpoint_ready() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
hf = root / "actor/huggingface"
if not hf.is_dir():
    hf = root
try:
    if not (hf / "config.json").is_file():
        raise ValueError("config.json is not available yet")
    json.loads((hf / "config.json").read_text())
    indexes = list(hf.glob("*.safetensors.index.json")) + list(hf.glob("pytorch_model.bin.index.json"))
    if indexes:
        for index in indexes:
            metadata = json.loads(index.read_text())
            shards = set(metadata.get("weight_map", {}).values())
            if not shards or any(not (hf / name).is_file() or (hf / name).stat().st_size == 0 for name in shards):
                raise ValueError("indexed weight shards are missing or empty")
            if sum((hf / name).stat().st_size for name in shards) < metadata.get("metadata", {}).get("total_size", 0):
                raise ValueError("indexed weight shards are still incomplete")
    elif not any(p.stat().st_size > 0 for pattern in ("*.safetensors", "pytorch_model*.bin") for p in hf.glob(pattern)):
        # Existing run_eval.sbatch can merge an actor's downloaded FSDP shards.
        if not any(p.stat().st_size > 0 for p in (root / "actor").glob("model_world_size_*_rank_*.pt")):
            raise ValueError("model weights are not available yet")
except (OSError, ValueError, TypeError) as exc:
    print(f"[INFO] Standalone checkpoint not ready: {root}: {exc}", file=sys.stderr)
    raise SystemExit(1)
PY
}

declare -A CRITIC_RETRY_PROFILE_BY_EVAL_NAME=(
  [branchstrata20_fresh_lr1e5_critic]="persist"
  [branchstrata20_fresh_lr1e5_critic_maxcalls5]="persist"
  [branchstrata20_fresh_lr1e5_critic_maxcalls20]="persist"
  [branchstrata20_fresh_lr1e5_critic_candidate_schedule_v1]="persist"
  [branchstrata20_fresh_lr1e5_critic_replay_schedule_v1]="persist"
  [branchstrata20_fresh_prefixremoved_gspo_lr1e5_critic]="teacher_only"
)

CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE="${CRITIC_RETRY_PROFILE_BY_EVAL_NAME[$EVAL_NAME]:-}"
# Critic-trained eval configurations should reproduce their answer-retry policy
# without requiring an extra flag. An explicit true/false override still wins.
if [[ -z "$CRITIC_GUIDED_ANSWER_RETRY_EXPLICIT" && -n "$CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE" ]]; then
  CRITIC_GUIDED_ANSWER_RETRY=true
fi

CRITIC_RETRY_ENABLED=false
case "$CRITIC_GUIDED_ANSWER_RETRY_SOURCE" in
  policy|ground_truth) ;;
  *) echo "ERROR: critic source must be policy or ground_truth." >&2; exit 1 ;;
esac
case "${CRITIC_GUIDED_ANSWER_RETRY,,}" in
  1|true|yes|on) CRITIC_RETRY_ENABLED=true ;;
  0|false|no|off) ;;
  *)
    echo "ERROR: CRITIC_GUIDED_ANSWER_RETRY must be true or false (got '$CRITIC_GUIDED_ANSWER_RETRY')." >&2
    exit 1
    ;;
esac
if [[ "$CRITIC_RETRY_ENABLED" == true ]] && ! python3 - "$CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD" <<'PY'
import math
import sys
try:
    value = float(sys.argv[1])
except ValueError:
    raise SystemExit(1)
raise SystemExit(0 if math.isfinite(value) and 0.0 <= value <= 1.0 else 1)
PY
then
  echo "ERROR: CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD must be a finite number in [0, 1]." >&2
  exit 1
fi
if [[ "$CRITIC_RETRY_ENABLED" == true && -z "$CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE" ]]; then
  echo "ERROR: critic-guided answer retry is only supported for mapped critic EVAL_NAME values: ${!CRITIC_RETRY_PROFILE_BY_EVAL_NAME[*]} (got '$EVAL_NAME')." >&2
  exit 1
fi

eval_group_key="${EVAL_NAME,,}"
if [[ -n "${EVAL_NAMES_BY_GROUP[$eval_group_key]:-}" ]]; then
  eval_group_members="${EVAL_NAMES_BY_GROUP[$eval_group_key]:-}"
  if [[ -n "$TRAIN_DIR_EXPLICIT" ]]; then
    echo "ERROR: TRAIN_DIR cannot be combined with grouped EVAL_NAME=$EVAL_NAME." >&2
    exit 1
  fi

  read -r -a GROUP_EVAL_NAMES <<< "$eval_group_members"
  failed_eval_names=()
  skipped_eval_names=()
  submitted_eval_names=()
  for eval_name in "${GROUP_EVAL_NAMES[@]}"; do
    train_dir="${TRAIN_DIR_BY_EVAL_NAME[$eval_name]:-}"
    if [[ ! -d "$train_dir" ]]; then
      echo "[INFO] Skipping EVAL_NAME=$eval_name (checkpoint directory is missing or inaccessible: $train_dir)."
      skipped_eval_names+=("$eval_name")
      continue
    fi
    if [[ -n "${SINGLE_CHECKPOINT_STEP_BY_EVAL_NAME[$eval_name]:-}" ]]; then
      if ! standalone_checkpoint_ready "$train_dir"; then
        skipped_eval_names+=("$eval_name")
        continue
      fi
      group_checkpoints=("$train_dir")
    else
      mapfile -t group_checkpoints < <(find "$train_dir" -maxdepth 1 -type d \( \
        -regex '.*/checkpoint-[0-9]+$' -o \
        -regex '.*/checkpoint_[0-9]+$' -o \
        -regex '.*/global_step_[0-9]+$' \
      \) -print -quit)
    fi
    if [[ ${#group_checkpoints[@]} -eq 0 ]]; then
      echo "[INFO] Skipping EVAL_NAME=$eval_name (no checkpoints available yet under $train_dir)."
      skipped_eval_names+=("$eval_name")
      continue
    fi

    echo
    echo "========== EVAL_NAME=$eval_group_key MEMBER=$eval_name =========="
    child_env=(EVAL_NAME="$eval_name" EVAL_PROFILE="$EVAL_PROFILE")
    if [[ -n "$OUTPUT_ROOT_EXPLICIT" && -n "$OUTPUT_ROOT" ]]; then
      child_env+=(OUTPUT_ROOT="${OUTPUT_ROOT%/}/$eval_name")
    fi

    if env "${child_env[@]}" bash "$0"; then
      submitted_eval_names+=("$eval_name")
    else
      failed_eval_names+=("$eval_name")
    fi
  done

  if [[ ${#failed_eval_names[@]} -gt 0 ]]; then
    echo "ERROR: eval submission failed for group members: ${failed_eval_names[*]}" >&2
    exit 1
  fi
  echo "Completed grouped EVAL_NAME=$eval_group_key: processed ${#submitted_eval_names[@]}, skipped unavailable ${#skipped_eval_names[@]}."
  exit 0
fi

if [[ "$EVAL_NAME" == "all" ]]; then
  if [[ -n "$TRAIN_DIR_EXPLICIT" ]]; then
    echo "ERROR: TRAIN_DIR cannot be combined with EVAL_NAME=all; 'all' uses every entry in TRAIN_DIR_BY_EVAL_NAME." >&2
    exit 1
  fi

  mapfile -t ALL_EVAL_NAMES < <(printf '%s\n' "${!TRAIN_DIR_BY_EVAL_NAME[@]}" | sort)
  failed_eval_names=()
  for eval_name in "${ALL_EVAL_NAMES[@]}"; do
    echo
    echo "========== EVAL_NAME=$eval_name =========="

    child_env=(EVAL_NAME="$eval_name")
    # Keep an explicit output root collision-free when dispatching multiple runs.
    if [[ -n "$OUTPUT_ROOT_EXPLICIT" && -n "$OUTPUT_ROOT" ]]; then
      child_env+=(OUTPUT_ROOT="${OUTPUT_ROOT%/}/$eval_name")
    fi

    if ! env "${child_env[@]}" bash "$0"; then
      failed_eval_names+=("$eval_name")
    fi
  done

  if [[ ${#failed_eval_names[@]} -gt 0 ]]; then
    echo "ERROR: eval submission failed for: ${failed_eval_names[*]}" >&2
    exit 1
  fi
  echo "Completed checkpoint checks/submissions for ${#ALL_EVAL_NAMES[@]} eval configurations."
  exit 0
fi

# Resolve per-run defaults only in leaf invocations, so a group does not pass
# one member's timing policy to the other members. Explicit values still win.
declare -A MAX_TOOL_ITERATIONS_BY_EVAL_NAME=(
  # Training allows N tool rounds and N+1 assistant turns (including the answer).
  [branchstrata20_fresh_lr1e5_critic_maxcalls5]=5
  [branchstrata20_fresh_lr1e5_critic_maxcalls20]=20
)
MAX_TOOL_ITERATIONS="${MAX_TOOL_ITERATIONS:-${MAX_TOOL_ITERATIONS_BY_EVAL_NAME[$EVAL_NAME]:-10}}"
# Every zoom returns an image; a lower image cap would truncate the 20-round run.
MAX_TOOL_IMAGES="${MAX_TOOL_IMAGES:-${MAX_TOOL_ITERATIONS_BY_EVAL_NAME[$EVAL_NAME]:-10}}"
if [[ "$EVAL_NAME" == branchstrata20_fresh_lr1e5_critic_candidate_schedule_v1 ]]; then
  FIRST_CANDIDATE_ANSWER_TURN="${FIRST_CANDIDATE_ANSWER_TURN:-5}"
  RECORD_ANSWER_CANDIDATES="${RECORD_ANSWER_CANDIDATES:-true}"
else
  FIRST_CANDIDATE_ANSWER_TURN="${FIRST_CANDIDATE_ANSWER_TURN:-0}"
  RECORD_ANSWER_CANDIDATES="${RECORD_ANSWER_CANDIDATES:-false}"
fi

# Replay deadlines are training-only. Validation generates a full reasoning
# critic response over sanitized evidence, using the exact training recipe.
if [[ "$EVAL_NAME" == branchstrata20_fresh_lr1e5_critic_replay_schedule_v1 ]]; then
  CRITIC_GUIDED_ANSWER_RETRY_MODE="${CRITIC_GUIDED_ANSWER_RETRY_MODE:-reasoning}"
  REPLAY_CRITIC_SYSTEM_PROMPT="${REPLAY_CRITIC_SYSTEM_PROMPT:-$VERO_EVAL_DIR/../vero2_scripts/prompts/reasoning_replay_critic.txt}"
  VERO_TRAINING_CONFIG="${VERO_TRAINING_CONFIG:-$VERO_EVAL_DIR/../vero2_scripts/config/branchstrata20_fresh_lr1e5_critic_replay_schedule_v1.yaml}"
  REPLAY_CRITIC_TEMPERATURE="${REPLAY_CRITIC_TEMPERATURE:-0.6}"
  REPLAY_CRITIC_TOP_K="${REPLAY_CRITIC_TOP_K:-20}"
  REPLAY_CRITIC_TOP_P="${REPLAY_CRITIC_TOP_P:-0.95}"
else
  CRITIC_GUIDED_ANSWER_RETRY_MODE="${CRITIC_GUIDED_ANSWER_RETRY_MODE:-token}"
fi
case "$CRITIC_GUIDED_ANSWER_RETRY_MODE" in
  token|reasoning) ;;
  *) echo "ERROR: critic mode must be token or reasoning." >&2; exit 1 ;;
esac

declare -A TASKS_BY_TASK_GROUP=(
  [visual_probe]="visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom"
  [visual_probe_hstar_pinpoint]="visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom,hstarbench_mixed_reasoning_zoom,pinpoint_bench_reasoning_zoom,screenspotpro_point_in_box_reasoning_zoom"
  [visual_probe_vstar]="visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom,vstar_bench_reasoning_zoom"
  [visual_probe_vstar_hrbench]="visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom,vstar_bench_reasoning_zoom,hrbench_4k_reasoning_zoom,hrbench_8k_reasoning_zoom"
  [visual_probe_vstar_hstar_hrbench]="visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom,vstar_bench_reasoning_zoom,hstarbench_mixed_reasoning_zoom,hrbench_4k_reasoning_zoom,hrbench_8k_reasoning_zoom"
  [grounding_counting_search]="countbenchqa_reasoning_zoom,countqa_reasoning_zoom,mme_realworld_lite_reasoning_zoom,vstar_bench_reasoning_zoom,aerialvg_bbox_reasoning_zoom,visual_probe_easy_reasoning_zoom,visual_probe_medium_reasoning_zoom,visual_probe_hard_reasoning_zoom,screenspot_point_in_box_reasoning_zoom,screenspotpro_point_in_box_reasoning_zoom"
  [chart_ocr]="chartqa_reasoning_zoom,chartqa_pro_reasoning_zoom,infovqa_val_reasoning_zoom,chartmuseum_reasoning_zoom,evochart_reasoning_zoom"
  [stem]="mathvision_test_reasoning_zoom,mathvista_testmini_reasoning_zoom,mmmu_pro_standard_reasoning_zoom,mmmu_pro_vision_reasoning_zoom"
  [knowledge_rec]="realworldqa_reasoning_zoom,simplevqa_en_reasoning_zoom,fvqa_reasoning_zoom,mmvetv2_group_img_reasoning_zoom"
  [instruction_following]="mia_bench_reasoning_zoom,mm_mt_bench_reasoning_zoom,mmifeval_reasoning_zoom"
)

if [[ -n "$TRAIN_DIR_EXPLICIT" ]]; then
  TRAIN_DIR="$TRAIN_DIR"
else
  TRAIN_DIR="${TRAIN_DIR_BY_EVAL_NAME[$EVAL_NAME]:-}"
  if [[ -z "$TRAIN_DIR" ]]; then
    echo "ERROR: unknown EVAL_NAME '$EVAL_NAME'." >&2
    echo "Valid EVAL_NAME values: all ${!TRAIN_DIR_BY_EVAL_NAME[*]}" >&2
    exit 1
  fi
fi

SYSTEM_PROMPT="${SYSTEM_PROMPT:-${SYSTEM_PROMPT_BY_EVAL_NAME[$EVAL_NAME]:-}}"
if [[ -z "$TASKS_EXPLICIT" ]]; then
  TASKS="${TASKS_BY_TASK_GROUP[$TASK_GROUP]:-}"
  if [[ -z "$TASKS" ]]; then
    echo "ERROR: unknown TASK_GROUP '$TASK_GROUP'." >&2
    echo "Valid TASK_GROUP values: ${!TASKS_BY_TASK_GROUP[*]}" >&2
    exit 1
  fi
elif [[ -z "$TASK_GROUP_EXPLICIT" ]]; then
  TASK_GROUP="custom"
fi

OUTPUT_ROOT_TOKEN="$EVAL_NAME"
if [[ -n "$TRAIN_DIR_EXPLICIT" && -z "$EVAL_NAME_EXPLICIT" ]]; then
  OUTPUT_ROOT_TOKEN="$(basename "$TRAIN_DIR")"
fi
OUTPUT_ROOT="${OUTPUT_ROOT:-/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_qwen35_27b_judge/zoom_eval_${OUTPUT_ROOT_TOKEN}_${TASK_GROUP}}"
CRITIC_RETRY_JOB_TAG=""
if [[ "$CRITIC_RETRY_ENABLED" == true ]]; then
  CRITIC_THRESHOLD_TAG="${CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD//./p}"
  CRITIC_THRESHOLD_TAG="${CRITIC_THRESHOLD_TAG//[^A-Za-z0-9]/_}"
  CRITIC_RETRY_JOB_TAG="critic_retry_thr${CRITIC_THRESHOLD_TAG}"
  if [[ "$CRITIC_GUIDED_ANSWER_RETRY_MODE" == reasoning ]]; then
    CRITIC_RETRY_JOB_TAG="reasoning_critic"
  fi
  if [[ "$CRITIC_GUIDED_ANSWER_RETRY_SOURCE" == ground_truth ]]; then
    CRITIC_RETRY_JOB_TAG="gt_critic"
  fi
  OUTPUT_ROOT="${OUTPUT_ROOT}_${CRITIC_RETRY_JOB_TAG}"
fi
LEGACY_CANDIDATE_OUTPUT_ROOT=""
LEGACY_CANDIDATE_JOB_TAG=""
FOLLOWUP_CANDIDATE_ANSWER_TURN="${FOLLOWUP_CANDIDATE_ANSWER_TURN:-0}"
if [[ "$FIRST_CANDIDATE_ANSWER_TURN" != 0 ]]; then
  if [[ "$EVAL_NAME" == branchstrata20_fresh_lr1e5_critic_candidate_schedule_v1 ]]; then
    LEGACY_CANDIDATE_OUTPUT_ROOT="$OUTPUT_ROOT"
    LEGACY_CANDIDATE_JOB_TAG="$CRITIC_RETRY_JOB_TAG"
  fi
  FIRST_CANDIDATE_JOB_TAG="first_candidate_turn${FIRST_CANDIDATE_ANSWER_TURN}"
  OUTPUT_ROOT="${OUTPUT_ROOT}_${FIRST_CANDIDATE_JOB_TAG}"
  CRITIC_RETRY_JOB_TAG="${CRITIC_RETRY_JOB_TAG:+${CRITIC_RETRY_JOB_TAG}_}turn${FIRST_CANDIDATE_ANSWER_TURN}"
fi
if [[ "$FOLLOWUP_CANDIDATE_ANSWER_TURN" != 0 ]]; then
  if [[ "$FIRST_CANDIDATE_ANSWER_TURN" -lt 2 || "$FOLLOWUP_CANDIDATE_ANSWER_TURN" -le "$FIRST_CANDIDATE_ANSWER_TURN" || "$FOLLOWUP_CANDIDATE_ANSWER_TURN" -gt "$MAX_TOOL_ITERATIONS" ]]; then
    echo "ERROR: followup candidate turn requires 2 <= first < followup <= max_tool_iterations." >&2
    exit 1
  fi
  OUTPUT_ROOT="${OUTPUT_ROOT}_followup_turn${FOLLOWUP_CANDIDATE_ANSWER_TURN}"
  CRITIC_RETRY_JOB_TAG="${CRITIC_RETRY_JOB_TAG}_followup${FOLLOWUP_CANDIDATE_ANSWER_TURN}"
  LEGACY_CANDIDATE_OUTPUT_ROOT=""
  LEGACY_CANDIDATE_JOB_TAG=""
fi

case "${EVAL_PROFILE,,}" in
  full)
    PARTITION="${PARTITION:-pli-c}"
    ACCOUNT="${ACCOUNT:-}"
    CONSTRAINT="${CONSTRAINT:-}"
    GPU_TYPE="${GPU_TYPE:-}"
    NUM_GPUS="${NUM_GPUS:-1}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
    DEFAULT_TIME="${DEFAULT_TIME:-6:00:00}"
    ;;
  vlmrl)
    PARTITION="${PARTITION:-pli}"
    ACCOUNT="${ACCOUNT:-vlmrl}"
    CONSTRAINT="${CONSTRAINT:-}"
    GPU_TYPE="${GPU_TYPE:-}"
    NUM_GPUS="${NUM_GPUS:-1}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-12}"
    DEFAULT_TIME="${DEFAULT_TIME:-6:00:00}"
    ;;
  ailab)
    PARTITION="${PARTITION:-ailab}"
    ACCOUNT="${ACCOUNT:-}"
    CONSTRAINT="${CONSTRAINT:-}"
    GPU_TYPE="${GPU_TYPE:-}"
    NUM_GPUS="${NUM_GPUS:-1}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
    DEFAULT_TIME="${DEFAULT_TIME:-6:00:00}"
    ;;
  debug)
    PARTITION="gpu-test"
    ACCOUNT="${ACCOUNT:-}"
    CONSTRAINT="${CONSTRAINT:-gpu80}"
    GPU_TYPE="${GPU_TYPE:-}"
    NUM_GPUS="${NUM_GPUS:-1}"
    CPUS_PER_TASK="${CPUS_PER_TASK:-8}"
    DEFAULT_TIME="${DEFAULT_TIME:-1:00:00}"
    LIMIT="${LIMIT:-100}"
    # Debug default: a single task so LIMIT caps the whole run at 100 samples
    # (--limit is applied per task). Respect an explicit TASKS= override.
    [[ -z "$TASKS_EXPLICIT" ]] && TASKS="visual_probe_easy_reasoning_zoom"
    # Tag debug outputs with _DEBUG so they don't collide with real runs.
    # Respect an explicit OUTPUT_ROOT= override.
    [[ -z "$OUTPUT_ROOT_EXPLICIT" ]] && OUTPUT_ROOT="${OUTPUT_ROOT}_DEBUG"
    ;;
  *)
    echo "ERROR: EVAL_PROFILE must be 'full', 'vlmrl', 'ailab', or 'debug' (got '$EVAL_PROFILE')" >&2
    exit 1
    ;;
esac
# Preserve recent-model sampling and controller settings; only pin the tool
# schema after the profile has selected its tasks (including the debug task).
# Mini-o3's saved validation prompt has no bbox items/length constraints.
if [[ -z "$TASKS_EXPLICIT" && "$TASK_GROUP" == visual_probe ]]; then
  case "$EVAL_NAME" in
    fixjson_vero2_step200) TASKS="${TASKS//reasoning_zoom/reasoning_zoom_fixjson_step200}" ;;
    minio3_vero2_step200) TASKS="${TASKS//reasoning_zoom/reasoning_zoom_minio3_step200}" ;;
  esac
fi
if [[ "$CRITIC_GUIDED_ANSWER_RETRY_SOURCE" == ground_truth ]]; then
  if [[ "$CRITIC_RETRY_ENABLED" != true || "$JUDGE_BACKEND" != server || "$NUM_GPUS" -lt 2 ]]; then
    echo "ERROR: ground_truth critic requires retry=true, JUDGE_BACKEND=server and NUM_GPUS>=2." >&2
    exit 1
  fi
fi
SBATCH_OPTS_RAW="${SBATCH_OPTS:-}"
SBATCH_OPTS=()
if [[ -n "$SBATCH_OPTS_RAW" ]]; then
  read -r -a SBATCH_OPTS <<< "$SBATCH_OPTS_RAW"
fi

extract_checkpoint_step() {
  local checkpoint_name="$1"
  local step_raw=""
  if [[ "$checkpoint_name" =~ checkpoint-([0-9]+)$ ]]; then
    step_raw="${BASH_REMATCH[1]}"
  elif [[ "$checkpoint_name" =~ checkpoint_([0-9]+)$ ]]; then
    step_raw="${BASH_REMATCH[1]}"
  elif [[ "$checkpoint_name" =~ global_step_([0-9]+)$ ]]; then
    step_raw="${BASH_REMATCH[1]}"
  else
    return 1
  fi
  printf '%d' "$((10#$step_raw))"
}

checkpoint_eval_name() {
  if [[ -n "${SINGLE_CHECKPOINT_STEP_BY_EVAL_NAME[$EVAL_NAME]:-}" ]]; then
    printf 'global_step_%s' "${SINGLE_CHECKPOINT_STEP_BY_EVAL_NAME[$EVAL_NAME]}"
  else
    basename "$1"
  fi
}

sanitize_job_token() {
  local value="$1"
  value="${value//[^A-Za-z0-9_.-]/_}"
  printf '%s' "${value:0:80}"
}

checkpoint_eval_complete() {
  local out_dir="$1"
  [[ -d "$out_dir" ]] || return 1

  TASKS_ENV="$TASKS" OUT_DIR_ENV="$out_dir" python3 - <<'PY'
import json
import os
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR_ENV"])
requested = [t.strip() for t in os.environ.get("TASKS_ENV", "").split(",") if t.strip()]

for path in sorted(out_dir.rglob("*_results.json")):
    try:
        payload = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        continue
    results = payload.get("results")
    if not isinstance(results, dict):
        continue
    if all(task in results for task in requested):
        raise SystemExit(0)

raise SystemExit(1)
PY
}

has_sbatch_option() {
  local kind="$1"
  local opt
  for opt in "${SBATCH_OPTS[@]}"; do
    case "$kind:$opt" in
      mem:--mem=*|mem:--mem-per-cpu=*|mem:--mem-per-gpu=*) return 0 ;;
      time:--time|time:--time=*|time:-t|time:-t=*|time:-t[0-9]*) return 0 ;;
      gres:--gres=*|gres:--gpus=*|gres:--gpus-per-node=*|gres:--gpus-per-task=*) return 0 ;;
      partition:--partition=*|partition:-p|partition:-p*) return 0 ;;
      constraint:--constraint=*|constraint:-C|constraint:-C*) return 0 ;;
      account:--account=*|account:-A|account:-A*) return 0 ;;
      cpus:--cpus-per-task=*|cpus:-c|cpus:-c*) return 0 ;;
      job-name:--job-name=*|job-name:-J|job-name:-J*) return 0 ;;
    esac
  done
  return 1
}

validate_positive_int_or_empty() {
  local name="$1"
  local value="$2"
  [[ -z "$value" ]] && return 0
  if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1 ]]; then
    echo "$name must be a positive integer when set" >&2
    exit 1
  fi
}

validate_port() {
  local name="$1"
  local value="$2"
  if [[ ! "$value" =~ ^[0-9]+$ || "$value" -lt 1024 || "$value" -gt 65535 ]]; then
    echo "$name must be an integer port in [1024, 65535] (got '$value')" >&2
    exit 1
  fi
}

[[ -d "$TRAIN_DIR" ]] || { echo "ERROR: TRAIN_DIR missing or inaccessible: $TRAIN_DIR" >&2; exit 1; }
validate_positive_int_or_empty MAX_CHECKPOINTS "$MAX_CHECKPOINTS"
validate_positive_int_or_empty MAX_STEPS_EVAL "$MAX_STEPS_EVAL"
validate_positive_int_or_empty ONLY_EVAL_EVERY_X "$ONLY_EVAL_EVERY_X"
validate_positive_int_or_empty MAX_TOOL_ITERATIONS "$MAX_TOOL_ITERATIONS"
if [[ ! "$FIRST_CANDIDATE_ANSWER_TURN" =~ ^(0|[1-9][0-9]*)$ ]] || \
   [[ "$FIRST_CANDIDATE_ANSWER_TURN" != 0 && \
      ( "$FIRST_CANDIDATE_ANSWER_TURN" -lt 2 || "$FIRST_CANDIDATE_ANSWER_TURN" -gt "$MAX_TOOL_ITERATIONS" ) ]]; then
  echo "ERROR: FIRST_CANDIDATE_ANSWER_TURN must be 0 (disabled) or an integer between 2 and MAX_TOOL_ITERATIONS." >&2
  exit 1
fi
if [[ "$FIRST_CANDIDATE_ANSWER_TURN" != 0 ]]; then
  case "${FORCE_TOOL_CALLS_UNTIL_FINAL_TURN:-false}" in
    1|[Tt][Rr][Uu][Ee]|[Yy][Ee][Ss]|[Oo][Nn])
      echo "ERROR: FIRST_CANDIDATE_ANSWER_TURN requires FORCE_TOOL_CALLS_UNTIL_FINAL_TURN=false; set FIRST_CANDIDATE_ANSWER_TURN=0 for a tool-only schedule." >&2
      exit 1
      ;;
  esac
fi
validate_port POLICY_PORT_BASE "$POLICY_PORT_BASE"
validate_port ACCELERATE_PORT_BASE "$ACCELERATE_PORT_BASE"
case "${RUN_SYSTEM_PROMPT_MODE,,}" in
  task_metadata|run_env|none) ;;
  *)
    echo "ERROR: RUN_SYSTEM_PROMPT_MODE must be 'task_metadata', 'run_env', or 'none' (got '$RUN_SYSTEM_PROMPT_MODE')" >&2
    exit 1
    ;;
esac
if [[ -n "$SYSTEM_PROMPT" && ! -f "$SYSTEM_PROMPT" ]]; then
  echo "ERROR: configured SYSTEM_PROMPT not found: $SYSTEM_PROMPT" >&2
  exit 1
fi
if [[ -n "$CHAT_TEMPLATE" && ! -f "$CHAT_TEMPLATE" ]]; then
  echo "ERROR: configured CHAT_TEMPLATE not found: $CHAT_TEMPLATE" >&2
  exit 1
fi
if [[ ! -d "$VLLM_JUDGE_MODEL_PATH" ]]; then
  echo "ERROR: judge model directory not found: $VLLM_JUDGE_MODEL_PATH" >&2
  exit 1
fi
if [[ ! -f "$VLLM_JUDGE_CHAT_TEMPLATE" ]]; then
  echo "ERROR: judge chat template not found: $VLLM_JUDGE_CHAT_TEMPLATE" >&2
  exit 1
fi
declare -A ONLY_EVAL_GLOBAL_STEP_SET=()
ONLY_EVAL_GLOBAL_STEPS_NORMALIZED=""
if [[ -n "$ONLY_EVAL_GLOBAL_STEPS" ]]; then
  IFS=',' read -r -a ONLY_EVAL_GLOBAL_STEPS_LIST <<< "$ONLY_EVAL_GLOBAL_STEPS"
  for raw_step in "${ONLY_EVAL_GLOBAL_STEPS_LIST[@]}"; do
    step="${raw_step//[[:space:]]/}"
    if [[ -z "$step" || ! "$step" =~ ^[0-9]+$ || "$step" -lt 1 ]]; then
      echo "ONLY_EVAL_GLOBAL_STEPS must be a comma-separated list of positive integers" >&2
      exit 1
    fi
    ONLY_EVAL_GLOBAL_STEP_SET["$((10#$step))"]=1
  done
  mapfile -t ONLY_EVAL_GLOBAL_STEPS_KEYS < <(printf '%s\n' "${!ONLY_EVAL_GLOBAL_STEP_SET[@]}" | sort -n)
  ONLY_EVAL_GLOBAL_STEPS_NORMALIZED=$(IFS=,; echo "${ONLY_EVAL_GLOBAL_STEPS_KEYS[*]}")
fi

_TASKNAMES="$(command grep -rhoE '^(task|group):[[:space:]]*["'\'']?[A-Za-z0-9_]+' \
  "$VERO_EVAL_DIR/lmms_eval/tasks"/*/*.yaml 2>/dev/null \
  | sed -E 's/^(task|group):[[:space:]]*//; s/["'\'']//g' | sort -u)"

missing=()
for task in ${TASKS//,/ }; do
  command grep -qxF "$task" <<< "$_TASKNAMES" || missing+=("$task")
done
if [[ ${#missing[@]} -gt 0 ]]; then
  echo "WARN: missing task yaml(s): ${missing[*]}" >&2
  if [[ "$STRICT_TASKS" == 1 ]]; then
    echo "ERROR: refusing to submit partial run. Set STRICT_TASKS=0 or pass TASKS=... explicitly." >&2
    exit 1
  fi
fi

if [[ -n "${SINGLE_CHECKPOINT_STEP_BY_EVAL_NAME[$EVAL_NAME]:-}" ]]; then
  if ! standalone_checkpoint_ready "$TRAIN_DIR"; then
    echo "[INFO] Skipping EVAL_NAME=$EVAL_NAME until the checkpoint download is ready."
    exit 0
  fi
  CHECKPOINTS=("$TRAIN_DIR")
else
  mapfile -t CHECKPOINTS < <(find "$TRAIN_DIR" -maxdepth 1 -type d \( \
    -regex '.*/checkpoint-[0-9]+$' -o \
    -regex '.*/checkpoint_[0-9]+$' -o \
    -regex '.*/global_step_[0-9]+$' \
  \) | sort -V)
fi

FOUND_CHECKPOINTS=${#CHECKPOINTS[@]}
[[ $FOUND_CHECKPOINTS -gt 0 ]] || { echo "ERROR: no checkpoints under $TRAIN_DIR" >&2; exit 1; }

declare -i SKIPPED_EXISTING=0
declare -i SKIPPED_QUEUED=0
declare -i SKIPPED_MAX_STEP=0
declare -i SKIPPED_INTERVAL=0
declare -i SKIPPED_EXPLICIT=0

if [[ -n "$MAX_STEPS_EVAL" ]]; then
  FILTERED_CHECKPOINTS=()
  for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name="$(checkpoint_eval_name "$ckpt")"
    step="$(extract_checkpoint_step "$ckpt_name" || true)"
    if [[ -n "$step" && "$step" -le "$MAX_STEPS_EVAL" ]]; then
      FILTERED_CHECKPOINTS+=("$ckpt")
    else
      SKIPPED_MAX_STEP+=1
    fi
  done
  CHECKPOINTS=("${FILTERED_CHECKPOINTS[@]}")
fi

if [[ "${SKIP_IF_EXISTS,,}" == "true" ]]; then
  FILTERED_CHECKPOINTS=()
  for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name="$(checkpoint_eval_name "$ckpt")"
    out="$OUTPUT_ROOT/$ckpt_name"
    if checkpoint_eval_complete "$out"; then
      echo "[INFO] Skipping $ckpt_name (complete results detected under $out)."
      SKIPPED_EXISTING+=1
      continue
    fi
    if [[ -n "$LEGACY_CANDIDATE_OUTPUT_ROOT" ]] && \
       checkpoint_eval_complete "$LEGACY_CANDIDATE_OUTPUT_ROOT/$ckpt_name"; then
      echo "[INFO] Skipping $ckpt_name (previous evaluation complete under $LEGACY_CANDIDATE_OUTPUT_ROOT/$ckpt_name)."
      SKIPPED_EXISTING+=1
      continue
    fi
    FILTERED_CHECKPOINTS+=("$ckpt")
  done
  CHECKPOINTS=("${FILTERED_CHECKPOINTS[@]}")
fi

declare -A QUEUED_JOB_SET=()
if [[ "${SKIP_IF_QUEUED,,}" == "true" ]]; then
  if command -v squeue >/dev/null 2>&1; then
    QUEUE_USER="${USER:-$(whoami)}"
    while IFS= read -r job_name; do
      [[ -n "$job_name" ]] && QUEUED_JOB_SET["$job_name"]=1
    done < <(squeue -u "$QUEUE_USER" -h -r -o "%j" -t PD,R)
  else
    echo "[WARN] SKIP_IF_QUEUED enabled but squeue not found; skipping queued check."
  fi
fi

if [[ "${SKIP_IF_QUEUED,,}" == "true" && ${#QUEUED_JOB_SET[@]} -gt 0 ]]; then
  FILTERED_CHECKPOINTS=()
  for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name="$(checkpoint_eval_name "$ckpt")"
    if [[ -n "$CRITIC_RETRY_JOB_TAG" ]]; then
      job_name="zoom-eval-$(sanitize_job_token "${CRITIC_RETRY_JOB_TAG}-${ckpt_name}-${EVAL_NAME}-${TASK_GROUP}")"
    else
      job_name="zoom-eval-$(sanitize_job_token "${EVAL_NAME}-${TASK_GROUP}-${ckpt_name}")"
    fi
    legacy_job_name="zoom-eval-$(sanitize_job_token "$ckpt_name")"
    queued_match=""
    if [[ -n "${QUEUED_JOB_SET[$job_name]:-}" ]]; then
      queued_match="$job_name"
    elif [[ -z "$CRITIC_RETRY_JOB_TAG" && -n "${QUEUED_JOB_SET[$legacy_job_name]:-}" ]]; then
      queued_match="$legacy_job_name"
    fi
    if [[ -z "$queued_match" && -n "$LEGACY_CANDIDATE_OUTPUT_ROOT" ]]; then
      if [[ -n "$LEGACY_CANDIDATE_JOB_TAG" ]]; then
        previous_timing_job_name="zoom-eval-$(sanitize_job_token "${LEGACY_CANDIDATE_JOB_TAG}-${ckpt_name}-${EVAL_NAME}-${TASK_GROUP}")"
      else
        previous_timing_job_name="zoom-eval-$(sanitize_job_token "${EVAL_NAME}-${TASK_GROUP}-${ckpt_name}")"
      fi
      if [[ -n "${QUEUED_JOB_SET[$previous_timing_job_name]:-}" ]]; then
        queued_match="$previous_timing_job_name"
      fi
    fi
    if [[ -n "$queued_match" ]]; then
      echo "[INFO] Skipping $ckpt_name (job $queued_match already queued/running)."
      SKIPPED_QUEUED+=1
      continue
    fi
    FILTERED_CHECKPOINTS+=("$ckpt")
  done
  CHECKPOINTS=("${FILTERED_CHECKPOINTS[@]}")
fi

if [[ -n "$ONLY_EVAL_EVERY_X" ]]; then
  FILTERED_CHECKPOINTS=()
  for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name="$(checkpoint_eval_name "$ckpt")"
    step="$(extract_checkpoint_step "$ckpt_name" || true)"
    if [[ -n "$step" && $((step % ONLY_EVAL_EVERY_X)) -eq 0 ]]; then
      FILTERED_CHECKPOINTS+=("$ckpt")
    else
      SKIPPED_INTERVAL+=1
    fi
  done
  CHECKPOINTS=("${FILTERED_CHECKPOINTS[@]}")
fi

if [[ ${#ONLY_EVAL_GLOBAL_STEP_SET[@]} -gt 0 ]]; then
  FILTERED_CHECKPOINTS=()
  for ckpt in "${CHECKPOINTS[@]}"; do
    ckpt_name="$(checkpoint_eval_name "$ckpt")"
    step="$(extract_checkpoint_step "$ckpt_name" || true)"
    if [[ -n "$step" && -n "${ONLY_EVAL_GLOBAL_STEP_SET[$step]:-}" ]]; then
      FILTERED_CHECKPOINTS+=("$ckpt")
    else
      SKIPPED_EXPLICIT+=1
    fi
  done
  CHECKPOINTS=("${FILTERED_CHECKPOINTS[@]}")
fi

SELECTABLE_CHECKPOINTS=${#CHECKPOINTS[@]}
if [[ -n "$MAX_CHECKPOINTS" && $SELECTABLE_CHECKPOINTS -gt "$MAX_CHECKPOINTS" ]]; then
  if [[ "${USE_MOST_RECENT_MAX_CHECKPOINTS,,}" == "true" ]]; then
    CHECKPOINTS=("${CHECKPOINTS[@]:SELECTABLE_CHECKPOINTS-MAX_CHECKPOINTS:MAX_CHECKPOINTS}")
  else
    CHECKPOINTS=("${CHECKPOINTS[@]:0:MAX_CHECKPOINTS}")
  fi
fi

if [[ ${#CHECKPOINTS[@]} -eq 0 ]]; then
  echo "[INFO] No checkpoints to submit after applying filters."
  exit 0
fi

# Avoid every invocation's first selected checkpoint using the same policy port.
# This matters on shared nodes: two independent submitter runs can both
# start with PORT=10001 and collide if Slurm places them together.
if [[ -z "$PORT_OFFSET" ]]; then
  printf -v SUBMIT_TIME_TOKEN '%(%H%M%S)T' -1
  PORT_OFFSET=$(( (10#$SUBMIT_TIME_TOKEN + $$) % 20000 ))
else
  validate_positive_int_or_empty PORT_OFFSET "$PORT_OFFSET"
fi

MEM_REQUEST="${MEM_OVERRIDE:-$DEFAULT_MEM}"
TIME_REQUEST="${TIME_OVERRIDE:-$DEFAULT_TIME}"

BASE_SBATCH_OPTS=("${SBATCH_OPTS[@]}")
has_sbatch_option partition || [[ -z "$PARTITION" ]] || BASE_SBATCH_OPTS+=("--partition=$PARTITION")
has_sbatch_option account || [[ -z "$ACCOUNT" ]] || BASE_SBATCH_OPTS+=("--account=$ACCOUNT")
has_sbatch_option constraint || [[ -z "$CONSTRAINT" ]] || BASE_SBATCH_OPTS+=("--constraint=$CONSTRAINT")
has_sbatch_option cpus || BASE_SBATCH_OPTS+=("--cpus-per-task=$CPUS_PER_TASK")
has_sbatch_option mem || BASE_SBATCH_OPTS+=("--mem=$MEM_REQUEST")
has_sbatch_option time || BASE_SBATCH_OPTS+=("--time=$TIME_REQUEST")
if ! has_sbatch_option gres; then
  if [[ -n "$GPU_TYPE" ]]; then
    BASE_SBATCH_OPTS+=("--gres=gpu:${GPU_TYPE}:${NUM_GPUS}")
  else
    BASE_SBATCH_OPTS+=("--gres=gpu:${NUM_GPUS}")
  fi
fi

CHECKPOINT_INFO="${#CHECKPOINTS[@]} selected from $FOUND_CHECKPOINTS found"
(( SKIPPED_EXISTING > 0 )) && CHECKPOINT_INFO+="; skipped $SKIPPED_EXISTING existing"
(( SKIPPED_QUEUED > 0 )) && CHECKPOINT_INFO+="; skipped $SKIPPED_QUEUED queued/running"
[[ -n "$MAX_STEPS_EVAL" ]] && CHECKPOINT_INFO+="; max_steps_eval=$MAX_STEPS_EVAL skipped $SKIPPED_MAX_STEP"
[[ -n "$ONLY_EVAL_EVERY_X" ]] && CHECKPOINT_INFO+="; only_eval_every_x=$ONLY_EVAL_EVERY_X skipped $SKIPPED_INTERVAL"
[[ -n "$ONLY_EVAL_GLOBAL_STEPS_NORMALIZED" ]] && CHECKPOINT_INFO+="; only_eval_global_steps=$ONLY_EVAL_GLOBAL_STEPS_NORMALIZED skipped $SKIPPED_EXPLICIT"
[[ -n "$MAX_CHECKPOINTS" ]] && CHECKPOINT_INFO+="; max_checkpoints=$MAX_CHECKPOINTS most_recent=${USE_MOST_RECENT_MAX_CHECKPOINTS}"

cat <<INFO
Submitting Vero zoom checkpoint evals
  Eval name:       $EVAL_NAME
  Train dir:       $TRAIN_DIR
  Task group:      $TASK_GROUP
  Profile:         ${EVAL_PROFILE,,}
  Tasks:           $TASKS
  System prompt:   ${SYSTEM_PROMPT:-<none>} (mode=${RUN_SYSTEM_PROMPT_MODE,,})
  Tool parser:     $TOOL_CALL_PARSER
  Chat template:   ${CHAT_TEMPLATE:-<checkpoint default>}
  Output root:     $OUTPUT_ROOT
  Checkpoints:     $CHECKPOINT_INFO
  Image pixels:    max=$MAX_IMAGE_PIXELS
  Max tool images: $MAX_TOOL_IMAGES
  Candidate timing: first_answer_turn=$FIRST_CANDIDATE_ANSWER_TURN (0=natural) max_tool_iterations=$MAX_TOOL_ITERATIONS record_candidates=$RECORD_ANSWER_CANDIDATES
  Judge model:     $VLLM_JUDGE_MODEL_PATH
  Judge template:  $VLLM_JUDGE_CHAT_TEMPLATE
  FlashInfer sampler: $VLLM_USE_FLASHINFER_SAMPLER
  Judge runtime:   enforce_eager=$VLLM_JUDGE_ENFORCE_EAGER gdn_prefill_backend=$VLLM_JUDGE_GDN_PREFILL_BACKEND
  Forced final:    after_max_tool_iterations=$FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS prefix=${FORCE_FINAL_ANSWER_PREFIX:-<empty>} think_prefill=${FORCE_FINAL_ANSWER_THINK_PREFILL:-<empty>}
  Critic retry:    enabled=$CRITIC_RETRY_ENABLED mode=$CRITIC_GUIDED_ANSWER_RETRY_MODE source=$CRITIC_GUIDED_ANSWER_RETRY_SOURCE threshold=$CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD prefill_profile=${CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE:-<none>} tag=${CRITIC_RETRY_JOB_TAG:-<none>}
  Replay critic sampling: temperature=${REPLAY_CRITIC_TEMPERATURE:-0} top_k=${REPLAY_CRITIC_TOP_K:--1} top_p=${REPLAY_CRITIC_TOP_P:-1}
  Workers:         $WORKERS
  Port bases:      policy=$POLICY_PORT_BASE accelerate=$ACCELERATE_PORT_BASE offset=$PORT_OFFSET
  Sbatch options:  ${BASE_SBATCH_OPTS[*]}
INFO

submitted=0
for i in "${!CHECKPOINTS[@]}"; do
  CKPT_DIR="${CHECKPOINTS[$i]}"
  CKPT_NAME="$(checkpoint_eval_name "$CKPT_DIR")"
  HF_PATH="$CKPT_DIR/actor/huggingface"
  [[ -d "$HF_PATH" ]] || HF_PATH="$CKPT_DIR"

  PORT=$((POLICY_PORT_BASE + PORT_OFFSET + i))
  MASTER_PORT=$((ACCELERATE_PORT_BASE + PORT_OFFSET + i))
  validate_port PORT "$PORT"
  validate_port MASTER_PORT "$MASTER_PORT"
  OUT="$OUTPUT_ROOT/$CKPT_NAME"
  if [[ -n "$CRITIC_RETRY_JOB_TAG" ]]; then
    JOB_NAME="zoom-eval-$(sanitize_job_token "${CRITIC_RETRY_JOB_TAG}-${CKPT_NAME}-${EVAL_NAME}-${TASK_GROUP}")"
  else
    JOB_NAME="zoom-eval-$(sanitize_job_token "${EVAL_NAME}-${TASK_GROUP}-${CKPT_NAME}")"
  fi

  env_list=(
    MODEL=vero
    TOOL_CALL_PARSER="$TOOL_CALL_PARSER"
    SUFFIX=reasoning_zoom
    TOOLS=on
    MM_MAX_PIXELS="$MAX_IMAGE_PIXELS"
    MM_MIN_PIXELS=
    MAX_TOOL_ITERATIONS="$MAX_TOOL_ITERATIONS"
    MAX_TOOL_IMAGES="$MAX_TOOL_IMAGES"
    CHECKPOINT_PATH="$HF_PATH"
    TASKS="$TASKS"
    PORT="$PORT"
    JUDGE_BACKEND="$JUDGE_BACKEND"
    API_TYPE=vllm
    JUDGE_MODEL_PATH="$VLLM_JUDGE_MODEL_PATH"
    VLLM_JUDGE_MODEL_PATH="$VLLM_JUDGE_MODEL_PATH"
    VLLM_JUDGE_CHAT_TEMPLATE="$VLLM_JUDGE_CHAT_TEMPLATE"
    VLLM_USE_FLASHINFER_SAMPLER="$VLLM_USE_FLASHINFER_SAMPLER"
    VLLM_JUDGE_ENFORCE_EAGER="$VLLM_JUDGE_ENFORCE_EAGER"
    VLLM_JUDGE_GDN_PREFILL_BACKEND="$VLLM_JUDGE_GDN_PREFILL_BACKEND"
    CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE=1
    OUTPUT_ROOT="$OUT"
    NUM_GPUS=1
    WORKERS="$WORKERS"
    MASTER_PORT="$MASTER_PORT"
    OVERRIDE_EXISTING="$OVERRIDE_EXISTING"
    MERGE_BACKEND="$MERGE_BACKEND"
    TRAINING_SYSTEM_PROMPT="${SYSTEM_PROMPT:-}"
    RUN_SYSTEM_PROMPT_MODE="${RUN_SYSTEM_PROMPT_MODE,,}"
    FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS="$FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS"
    FORCE_FINAL_ANSWER_PREFIX="$FORCE_FINAL_ANSWER_PREFIX"
    FORCE_FINAL_ANSWER_THINK_PREFILL="$FORCE_FINAL_ANSWER_THINK_PREFILL"
    CRITIC_GUIDED_ANSWER_RETRY="$CRITIC_RETRY_ENABLED"
    CRITIC_GUIDED_ANSWER_RETRY_SOURCE="$CRITIC_GUIDED_ANSWER_RETRY_SOURCE"
    CRITIC_GUIDED_ANSWER_RETRY_MODE="$CRITIC_GUIDED_ANSWER_RETRY_MODE"
    CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD="$CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD"
    CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE="${CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE:-persist}"
    FIRST_CANDIDATE_ANSWER_TURN="$FIRST_CANDIDATE_ANSWER_TURN"
    FOLLOWUP_CANDIDATE_ANSWER_TURN="$FOLLOWUP_CANDIDATE_ANSWER_TURN"
    RECORD_ANSWER_CANDIDATES="$RECORD_ANSWER_CANDIDATES"
  )
  if [[ "$EVAL_NAME" == branchstrata20_fresh_lr1e5_critic_replay_schedule_v1 ]]; then
    env_list+=(REPLAY_CRITIC_SYSTEM_PROMPT="$REPLAY_CRITIC_SYSTEM_PROMPT" VERO_TRAINING_CONFIG="$VERO_TRAINING_CONFIG")
    env_list+=(REPLAY_CRITIC_TEMPERATURE="$REPLAY_CRITIC_TEMPERATURE" REPLAY_CRITIC_TOP_K="$REPLAY_CRITIC_TOP_K" REPLAY_CRITIC_TOP_P="$REPLAY_CRITIC_TOP_P")
    # Training alignment also configures the model; honor an explicit disable.
    if [[ "$CRITIC_RETRY_ENABLED" == false ]]; then
      env_list+=(VERO_EVAL_DISABLE_CRITIC=true)
    fi
  fi
  [[ -n "$CHAT_TEMPLATE" ]] && env_list+=(CHAT_TEMPLATE="$CHAT_TEMPLATE")
  if [[ -n "${SINGLE_CHECKPOINT_STEP_BY_EVAL_NAME[$EVAL_NAME]:-}" ]]; then
    env_list+=(LMMS_EVAL_HF_TOKENIZER="$HF_PATH")
  fi
  [[ -n "$LIMIT" ]] && env_list+=(LIMIT="$LIMIT")
  if [[ "${RUN_SYSTEM_PROMPT_MODE,,}" == "run_env" && -n "$SYSTEM_PROMPT" ]]; then
    env_list+=(SYSTEM_PROMPT="$SYSTEM_PROMPT")
  fi

  sbatch_opts=("${BASE_SBATCH_OPTS[@]}")
  if ! has_sbatch_option job-name; then
    sbatch_opts+=("--job-name=$JOB_NAME")
  fi
  cmd=(sbatch "${sbatch_opts[@]}" --export=ALL run_eval.sbatch)

  echo "[$CKPT_NAME] checkpoint=$HF_PATH"
  echo "  tasks=$TASKS"
  echo "  port=$PORT judge=$JUDGE_BACKEND master_port=$MASTER_PORT -> $OUT"
  if [[ "$DRY_RUN" == 1 ]]; then
    printf '  '; printf '%q ' env "${env_list[@]}" "${cmd[@]}"; printf '\n'
  else
    env "${env_list[@]}" "${cmd[@]}"
  fi
  submitted=$((submitted + 1))
done

echo "Submitted ${submitted} job(s). Watch with: squeue -u \$USER"
