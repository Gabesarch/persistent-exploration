#!/bin/bash
# CountQA-drawing LEARNING CURVE driver — run on the LOGIN node from the repo root.
# For each requested step: merge the FSDP checkpoint to HF (skipped if already merged),
# then chain the matched CountQA drawing eval on gpu-test. Optionally adds the base model.
# This automates the merge->eval pipeline used for every drawing curve so far, with the
# hard-won safeguards baked in (distinct vLLM port per eval, per-turn gen cap, afterok chaining).
#
#   bash vero_eval/scripts/eval_countqa_drawing_curve.sh \
#       --ckpt-root /scratch/.../checkpoints/vero2_verl/<RUN> \
#       --steps "15 40 65 140" [--base] [--limit 400]
#
# Results: vero_eval/results/cqa_drawing/<label>_<jobid>/**/*results*.json
# Read the curve when done:
#   grep -r '"exact_match,none"' vero_eval/results/cqa_drawing/*/ | sort
set -euo pipefail
CKPT_ROOT=""; STEPS=""; WITH_BASE=0; LIMIT=400
while [[ $# -gt 0 ]]; do case "$1" in
  --ckpt-root) CKPT_ROOT="$2"; shift 2;;
  --steps)     STEPS="$2"; shift 2;;
  --base)      WITH_BASE=1; shift;;
  --limit)     LIMIT="$2"; shift 2;;
  *) echo "unknown arg $1" >&2; exit 2;;
esac; done
[[ -n "$CKPT_ROOT" && -n "$STEPS" ]] || { grep -m12 '^#' "$0"; exit 2; }
[[ -f vero_eval/set_paths.sh.example ]] || { echo "ERROR: run from the repo root" >&2; exit 2; }
set +u; source ./set_paths.sh >/dev/null 2>&1; set -u   # for ROOT_PATH (merged-HF location)

RUNBASE=$(basename "$CKPT_ROOT")
i=0
echo "step | merge job | eval job | merged HF"
for STEP in $STEPS; do
  i=$((i+1))
  OUT="$ROOT_PATH/vero2_merged_hf/${RUNBASE}_step_${STEP}"
  DEP=""
  MJOB="-"
  if [[ ! -f "$OUT/model.safetensors" ]]; then
    MJOB=$(sbatch --parsable --export=ALL,CKPT_ROOT="$CKPT_ROOT",STEP="$STEP",OUT="$OUT" \
           vero_eval/scripts/merge_ckpt_to_hf.sbatch)
    DEP="--dependency=afterok:$MJOB"
  fi
  EJOB=$(sbatch --parsable $DEP \
         --export=ALL,CHECKPOINT_PATH="$OUT",LIMIT="$LIMIT",PORT=$((9401+i)) \
         vero_eval/scripts/eval_countqa_drawing.sbatch)
  echo "$STEP | $MJOB | $EJOB | $OUT"
done
if [[ "$WITH_BASE" == 1 ]]; then
  BJOB=$(sbatch --parsable --export=ALL,MODEL=base,LIMIT="$LIMIT",PORT=9400 \
         vero_eval/scripts/eval_countqa_drawing.sbatch)
  echo "base | - | $BJOB | (Qwen3-VL-8B-Instruct under vero template)"
fi
echo "Watch: squeue -u \$USER | grep -E 'merge_hf|cqa_draw'"
