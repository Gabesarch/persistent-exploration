---
description: Audit a dataset's ground-truth labels with a strong VLM and filter out wrong/ambiguous examples (reusable across datasets)
argument-hint: "[--dataset NAME] [--split train|val] [N] [--effort high|medium] [--apply]"
allowed-tools: Bash, Read, Write, Edit, TaskCreate, TaskUpdate
---

You are running the **ground-truth (GT) correctness filter** in `dataset_filter/`. It
audits a dataset's labels with a strong VLM (default Gemini) and keeps only examples
whose GT is correct-or-plausible, dropping WRONG / AMBIGUOUS / UNANSWERABLE ones.
NEVER drop an example just for being hard — that is what the **PLAUSIBLE** verdict is
for. Read `dataset_filter/README.md` first; it is the source of truth for the method,
the reusability contract, and the output schema.

## This tool is dataset-agnostic
Everything keys off `--dataset <PREFIX>` + `--split`, reading
`<data-dir>/<PREFIX>_<split>.verl.jsonl` (verl schema: `images[0]`, `extra_info.id`,
`reward_model.ground_truth`, optional `extra_info.metadata.original_points`). To filter
a NEW dataset, drop its verl jsonl(s) in a data dir and pass its prefix — no code
changes. Default prefix is `combined_counting`.

## Fixed paths
- code:    `/home/lc1556/repo/reason-vlm/vero2/dataset_filter`
- python:  `/scratch/gpfs/ZHUANGL/lc1556/conda/envs/reason-vlm/bin/python` (portkey + PIL)
- runs:    `/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs`
- model:   `gemini-3.1-pro-preview` (Princeton AI-sandbox; keys `AI_SANDBOX_KEY`,
           `AI_SANDBOX_KEY2` in `../../AI-sandbox/.env`, never committed)

## Arguments (`$ARGUMENTS`, interpret leniently)
- `--dataset NAME`  metadata prefix (default `combined_counting`)
- `--split {train|val}` (default `train`); a bare integer => `--n` (0 = whole split)
- `--effort {high|medium|...}` => `--reasoning-effort` (ALWAYS pass one explicitly)
- `--apply` => after filtering, write the filtered dataset (fail-closed, see step 5)
- pass through `--data-dir --workers --max-side --abs-tol --rel-tol --no-points --seed`

## Method (two-phase "smart", per example)
1. **Blind count** — image only, GT hidden (no sycophancy); visible-instances only.
2. **Answer-aware audit** — only when blind ≠ GT; shows GT + blind hint + GT points
   overlaid (where available) + an ambiguity check. keep = CORRECT/PLAUSIBLE;
   drop = WRONG/AMBIGUOUS/UNANSWERABLE. If blind == GT, short-circuit CORRECT (1 call).
Tolerance guard: a WRONG whose own count is within `max(abs_tol, floor(rel_tol*gt))`
of GT **and > 0** is kept as PLAUSIBLE. Defaults `abs_tol=0, rel_tol=0.15` (small
counts must match ~exactly; the 15 % band only opens on large counts). Re-tune WITHOUT
re-calling the model via `rescore.py` on an existing decisions file.

## Steps
1. TaskCreate a short list: (localize), run filter, summarize, report, spot-check, (apply).
2. **One-time per dataset** — make it self-contained + fail closed on missing images:
   ```bash
   PY=/scratch/gpfs/ZHUANGL/lc1556/conda/envs/reason-vlm/bin/python
   cd /home/lc1556/repo/reason-vlm/vero2/dataset_filter
   $PY localize_paths.py --data-dir <DATA> --dataset <PREFIX>   # -> *_<split>.local.verl.jsonl
   ```
   (Skip if the verl paths already point at readable local images; then omit `--local-meta` below.)
3. Run the filter. ALWAYS pass an explicit `--reasoning-effort` (recorded per row;
   it drives hidden "thinking" tokens = the dominant $ cost — keep it the SAME for the
   whole dataset). The run is **resumable + config-guarded** (`<out>.config.json`; refuses
   to mix incompatible settings) and does **2-key failover** on quota. For large N launch
   with run_in_background and poll the out file.
   ```bash
   RUNS=/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs
   $PY filter_counting.py --data-dir <DATA> --dataset <PREFIX> --local-meta \
       --split <SP> --n <N> --strategy smart --reasoning-effort <E> \
       --abs-tol 0 --rel-tol 0.15 --workers 8 --out "$RUNS/<PREFIX>_<SP>.jsonl"
   ```
   Batch API (~50 % cheaper) is NOT available on the sandbox key — it requires a
   single-target provider/virtual-key from Princeton (see README). `batch_runner.py` is
   ready for when that lands; until then use the synchronous path above.
4. Report the SUMMARY: total, keep %, drop %, verdict counts, blind-difficulty. Build the
   GT-neutral zoomable report and spot-check (open ~10 across verdicts, esp. every WRONG
   and some PLAUSIBLE; Read a few underlying images and confirm the call):
   ```bash
   $PY make_exam_report.py --run "gemini=$RUNS/<PREFIX>_<SP>.jsonl" \
       --points-from <DATA> --out "$RUNS/<PREFIX>_<SP>_exam.html"
   ```
   Flag the failure mode we care about most: a hard-but-correct example wrongly DROPPED.
5. If `--apply`: write the filtered dataset. **Fail-closed** — it ERRORS if any example is
   undecided (missing / ERROR / PARSE_ERROR) rather than silently emitting a partial set.
   Finish the run first, or pass `--allow-partial` (excludes + lists undecided). Only an
   explicit `keep is False` is ever dropped.
   ```bash
   $PY filter_apply.py --data-dir <DATA> --dataset <PREFIX> --local-meta \
       --decisions "$RUNS/<PREFIX>_<SP>.jsonl" --splits <SP> --out-dir "$RUNS/filtered"
   ```
6. Finish with: paths written, keep/drop counts, report path, cost (from measured
   tokens), and your spot-check verdict on whether the filter is trustworthy at this N.
