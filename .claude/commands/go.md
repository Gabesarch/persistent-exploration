---
description: Filter the combined_counting_hq dataset for ground-truth correctness with Gemini
argument-hint: "[N] [--strategy smart|blind|verify] [--split train|val] [--apply]"
allowed-tools: Bash, Read, Write, Edit, TaskCreate, TaskUpdate
---

You are running the **counting-dataset ground-truth filter**. Goal: keep
hard-but-correct counting examples; drop examples whose ground-truth count is
WRONG, whose question is AMBIGUOUS/ill-posed, or that are UNANSWERABLE. NEVER drop
an example just for being hard — that is what the PLAUSIBLE verdict is for.
Stage-2 verify overlays the dataset's GT points (where available, ~50%) as an
audit guide; `--no-points` disables that. keep = CORRECT/PLAUSIBLE.

## Fixed paths
- code:    `/home/lc1556/repo/reason-vlm/vero2/dataset_filter`
- python:  `/scratch/gpfs/ZHUANGL/lc1556/conda/envs/reason-vlm/bin/python`  (has portkey + PIL)
- dataset: `/scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq`  (images symlinked to gs1693)
- runs:    `/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs`
- model:   `gemini-3.1-pro-preview` (Princeton AI-sandbox; key in ../../AI-sandbox/.env)

## Arguments
`$ARGUMENTS` — interpret leniently:
- a bare integer => `--n` (number of records; `0` = the whole split, ~10.3k train)
- `--strategy {smart|blind|verify}` (default **smart**)
- `--split {train|val}` (default **train**)
- `--apply` => after filtering, also write the filtered verl jsonl (filter_apply.py)
- pass through `--seed`, `--workers`, `--max-side`, `--abs-tol`, `--rel-tol` if given
If no N is given, default to **200**.

Keep/verdict tolerance (already baked into a fresh run): a WRONG verdict whose own
count is within `max(abs_tol, ceil(rel_tol*gt))` of GT **and > 0** is kept as
PLAUSIBLE (off-by-1-2 on hard images is not a real error). Defaults abs_tol=2,
rel_tol=0.15. To re-tune these WITHOUT re-calling Gemini, use `rescore.py` on an
existing decisions file, then `compare_runs.py` to see what flipped.

## Steps
1. Create/refresh a short task list (TaskCreate) for: run filter, summarize,
   build report, spot-check, (optional) apply.
2. Run the filter (background if N is large). Output file name encodes split+strategy,
   e.g. `runs/<split>_<strategy>.jsonl`. The run is **resumable** — re-running with a
   larger N reuses earlier results (same seed shuffle, append to the same file):
   ```bash
   cd /home/lc1556/repo/reason-vlm/vero2/dataset_filter
   PY=/scratch/gpfs/ZHUANGL/lc1556/conda/envs/reason-vlm/bin/python
   RUNS=/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs
   $PY filter_counting.py --n <N> --strategy <S> --split <SP> --workers 14 \
       --reasoning-effort medium --out "$RUNS/<SP>_<S>.jsonl"
   ```
   ALWAYS pass an explicit `--reasoning-effort` (consistency + reproducibility;
   it is recorded per row). Do not rely on the provider default. `reasoning_effort`
   controls hidden "thinking" tokens, which dominate $ cost. Pick the SAME level for
   the whole dataset. Cost model: gpt-5 $0.625/$5 per 1M in/out (~$105 batch for the
   full train), gemini-3.1-pro $2/$12 (~$258 batch); batch = 50% off, 24h.
   For N over ~400, launch it with run_in_background and poll the output file.
3. When done, report the **SUMMARY** block: total, keep %, drop %, verdict counts
   (CORRECT/PLAUSIBLE/WRONG/UNANSWERABLE), and blind-difficulty distribution.
4. Build the before/after HTML report (overlay GT points so you can see what the
   verifier audited):
   ```bash
   $PY make_report.py --decisions "$RUNS/<SP>_<S>.jsonl" --k 40 \
       --points-from /scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq \
       --out "$RUNS/<SP>_<S>_report.html"
   ```
   Optional: `$PY cost_estimate.py --decisions "$RUNS/<SP>_<S>.jsonl" --n-full 10289`
   to project the $ cost of filtering the whole split from the measured tokens.
5. **Spot-check quality**: open ~10 decisions across verdicts (esp. every WRONG and a
   few PLAUSIBLE). For 3-4 of them, actually Read the underlying image
   (`r["image"]`) and confirm Gemini's count/verdict is right. Report agreement and
   flag any case where a hard-but-correct example was wrongly dropped (a WRONG that
   should be PLAUSIBLE) — that is the failure mode we care about most.
6. If `--apply` was passed, write the filtered dataset:
   ```bash
   $PY filter_apply.py --decisions "$RUNS/<SP>_<S>.jsonl" --split <SP> \
       --out-dir /scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq
   ```
   This writes `combined_counting_<SP>.filtered.verl.jsonl`, `keep_ids_<SP>.json`,
   and `dropped_<SP>.jsonl`. (Undecided records are kept by default — never silently
   delete data.)
7. Finish with: paths written, keep/drop counts, the report path, and your spot-check
   verdict on whether the filter is trustworthy at this N or needs prompt tuning.

## Strategy notes (why "smart" is the default)
- **blind**: count with image only; keep if GT in [count_low, count_high]. No
  sycophancy, but over-rejects very hard images.
- **verify**: show GT and ask a fair auditor (hard != wrong); risk of agreeing too
  readily, mitigated by the strict prompt.
- **smart**: blind count first; exact match => CORRECT in 1 call; on mismatch run the
  answer-aware auditor (seeded with the blind count as a hint, not as truth). Best
  accuracy/cost balance and the recommended default.
