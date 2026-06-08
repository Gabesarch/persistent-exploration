#!/bin/bash
# TYPE 2 (submit side) — enumerate a training run's checkpoints and submit a SLURM
# array that evaluates each on a domain (one array task per checkpoint). Submit from
# the repo root so the sbatch's SLURM_SUBMIT_DIR resolves vero_eval/.
#
#   bash vero_eval/examples/submit_eval_checkpoints.sh \
#     --train-dir /scratch/.../checkpoints/verl/<RUN> --domain grounding_counting_search \
#     --model-suffix reasoning --every 200 --array-chunk 4
#   # --domain all submits one array job per canonical domain. --dry-run prints, no submit.
set -euo pipefail

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; VERO_EVAL_DIR="$(cd "$SELF/.." && pwd)"
REPO_ROOT_DIR="$(cd "$VERO_EVAL_DIR/.." && pwd)"
set +u
[[ -f "$VERO_EVAL_DIR/set_paths.sh" ]] && source "$VERO_EVAL_DIR/set_paths.sh" || source "$VERO_EVAL_DIR/set_paths.sh.example"
set -u
source "$VERO_EVAL_DIR/examples/_domain_map.sh"

TRAIN_DIR=""; DOMAIN=""; SUFFIX="reasoning"; RUN_ID=""; MAX_CKPTS=0; EVERY=0
NUM_GPUS="${NUM_GPUS:-1}"; JUDGE_BACKEND="${JUDGE_BACKEND:-server}"; LIMIT="${LIMIT:-}"
ARRAY_CHUNK=""; DRYRUN=0; SBATCH_OPTS=()
while [[ $# -gt 0 ]]; do case "$1" in
  --train-dir) TRAIN_DIR="$2"; shift 2;; --domain) DOMAIN="$2"; shift 2;;
  --model-suffix) SUFFIX="$2"; shift 2;; --run-id) RUN_ID="$2"; shift 2;;
  --max-checkpoints) MAX_CKPTS="$2"; shift 2;; --every) EVERY="$2"; shift 2;;
  --num-gpus) NUM_GPUS="$2"; shift 2;; --judge-backend) JUDGE_BACKEND="$2"; shift 2;;
  --limit) LIMIT="$2"; shift 2;; --array-chunk) ARRAY_CHUNK="$2"; shift 2;;
  --dry-run) DRYRUN=1; shift;;
  --time|--mem|--cpus-per-task|--partition|--account|--gres) SBATCH_OPTS+=("$1" "$2"); shift 2;;
  *) echo "Unknown arg: $1" >&2; shift;;
esac; done
[[ -z "$TRAIN_DIR" || -z "$DOMAIN" ]] && { echo "Usage: submit_eval_checkpoints.sh --train-dir DIR --domain NAME [--model-suffix S] [--every X] [--max-checkpoints N] [--array-chunk N] [--limit N] [--dry-run]"; exit 1; }
[[ -d "$TRAIN_DIR" ]] || { echo "ERROR: train-dir not found: $TRAIN_DIR" >&2; exit 1; }

if [[ "$DOMAIN" == "all" ]]; then DOMAIN_LIST=("${VERO_DOMAINS[@]}"); else DOMAIN_LIST=("$DOMAIN"); fi

submit_one() {
  local domain="$1"
  local run_id="${RUN_ID:-$(basename "$TRAIN_DIR")__${domain}}"
  local run_root="$ROOT_PATH/vero_eval/logs/$run_id"
  mkdir -p "$run_root/raw"
  # enumerate global_step_* / checkpoint_* dirs
  mapfile -t CKPTS < <(find "$TRAIN_DIR" -maxdepth 1 -type d -regextype posix-extended \
                          -regex '.*/(checkpoint|global_step)_[0-9]+$' | sort -V)
  # filter --every (step % X == 0)
  if [[ "$EVERY" -gt 0 ]]; then
    local keep=(); for c in "${CKPTS[@]}"; do
      local s="${c##*_}"; (( s % EVERY == 0 )) && keep+=("$c"); done
    CKPTS=("${keep[@]}")
  fi
  # filter --max-checkpoints (most recent N)
  [[ "$MAX_CKPTS" -gt 0 && ${#CKPTS[@]} -gt $MAX_CKPTS ]] && CKPTS=("${CKPTS[@]: -$MAX_CKPTS}")
  [[ ${#CKPTS[@]} -eq 0 ]] && { echo "ERROR: no checkpoints under $TRAIN_DIR for domain $domain" >&2; return 1; }

  local index="$run_root/checkpoints.index"
  printf '%s\n' "${CKPTS[@]}" > "$index"
  local total=${#CKPTS[@]}
  local spec="0-$((total-1))"
  [[ -n "$ARRAY_CHUNK" ]] && spec="${spec}%${ARRAY_CHUNK}"
  echo "[$domain] $total checkpoints -> $index  (array $spec, suffix _$SUFFIX)"

  local env_list=(CHECKPOINT_INDEX_PATH="$index" DOMAIN="$domain" MODEL_SUFFIX="$SUFFIX"
                  RUN_ROOT="$run_root" NUM_GPUS="$NUM_GPUS" JUDGE_BACKEND="$JUDGE_BACKEND")
  [[ -n "$LIMIT" ]] && env_list+=(LIMIT="$LIMIT")
  local cmd=(sbatch "${SBATCH_OPTS[@]}" --array="$spec" --export=ALL
             "$VERO_EVAL_DIR/scripts/eval_checkpoints_array.sbatch")
  if [[ "$DRYRUN" == 1 ]]; then
    echo "DRY-RUN: ( cd $REPO_ROOT_DIR && ${env_list[*]} ${cmd[*]} )"
  else
    ( cd "$REPO_ROOT_DIR" && env "${env_list[@]}" "${cmd[@]}" )
  fi
}

for d in "${DOMAIN_LIST[@]}"; do submit_one "$d"; done
