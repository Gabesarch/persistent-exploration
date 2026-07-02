# Counting-dataset ground-truth filter

Filter `combined_counting_hq` so we keep **hard-but-correct** counting examples and
drop examples whose **ground-truth count is wrong / unanswerable**. Judging is done
by a strong multimodal model (**Gemini `gemini-3.1-pro-preview`**) via the Princeton
AI-sandbox (Portkey gateway; key auto-loaded from `../../AI-sandbox/.env`).

## The core idea (why not just "ask the model")

This dataset's value is its hard examples, so the filter must not reject an example
merely because it is hard. Two naive approaches each fail:

| approach | failure |
|---|---|
| blind count (image only), require `== GT` | over-rejects correct-but-hard images the model slightly miscounts |
| show the GT and ask "is this right?" | sycophancy — rubber-stamps wrong GTs |

**`smart` (default) combines them.** An independent **blind count** (the model never
sees the GT) is LLM-as-extraction + our own exact-match metric. If it matches GT
exactly → `CORRECT` in one call. On mismatch we run an **answer-aware auditor** that
is explicitly told *hard ≠ wrong*: it returns one of

- `CORRECT` — its recount matches GT
- `PLAUSIBLE` — too dense to count exactly, but GT is in a believable range, no
  evidence it's wrong → **KEEP** (these are the valuable hard examples)
- `WRONG` — concrete evidence the true count differs → drop
- `UNANSWERABLE` — ambiguous question/category or unusable image → drop

`keep = verdict in {CORRECT, PLAUSIBLE}`.

Guardrails learned from the pilots (see "Pilot findings") that keep the filter
fair on hard examples:
- **Count only visible instances.** The prompts forbid inferring hidden objects
  (a solid 3D stack's interior, background buildings) that the point-annotated GT
  excludes — otherwise the model over-counts and calls a correct GT "too low".
- **Numeric tolerance override.** If the auditor says WRONG but its own count is
  within `max(abs_tol, ceil(rel_tol*gt))` of GT *and is > 0*, we override to
  PLAUSIBLE+keep — so off-by-1-2 on a hard image is not dropped. The `> 0` guard
  is essential: `model_count == 0` means "category absent / mismatch" (geese ≠
  cranes), a real drop signal the tolerance must never paper over.
- **GT points as a stage-2 audit guide.** ~50% of records (fsc147, rec8k) ship
  point annotations where `GT == len(points)`. We draw them as magenta rings on
  the stage-2 image and tell the model they are a noisy guide to audit (each ring
  on a distinct instance? any missed/spurious/doubled?), not truth. This recovers
  hard dense examples the model alone slightly miscounts. The independent stage-1
  blind count still runs WITHOUT the points, preserving an unbiased signal.
- **AMBIGUOUS verdict (drop).** Ill-posed / subjective questions ("innermost
  stripe", "all the windows" across a painting) are dropped even when a count
  exists — we don't want ambiguity in the dataset. Verdicts: keep = CORRECT /
  PLAUSIBLE; drop = WRONG / AMBIGUOUS / UNANSWERABLE.

## Files
- `prompts.py` — blind + auditor prompts (anti-laziness / anti-sycophancy / visible-only)
- `filter_counting.py` — run the filter; emits a resumable per-example JSONL of decisions
- `rescore.py` — re-derive keep/verdict from stored model outputs under new
  `--abs-tol`/`--rel-tol` **without new API calls** (tune the decision layer for free)
- `compare_runs.py` — diff two decision files (keep rate, verdict flips, overrides)
- `cost_estimate.py` — $ cost to filter the full split, from a pilot's measured tokens
- `make_report.py` — self-contained before/after HTML (`--points-from <dir>` overlays GT marks)
- `filter_apply.py` — turn decisions into `*.filtered.verl.jsonl` + `keep_ids.json`
- `../.claude/commands/go.md` — the `/go` slash command that drives the whole thing

## Quick use
```bash
PY=/scratch/gpfs/ZHUANGL/lc1556/conda/envs/reason-vlm/bin/python
RUNS=/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs

# 1. filter (resumable; grow N to reuse earlier results with the same --seed)
$PY filter_counting.py --n 200 --strategy smart --out "$RUNS/train_smart.jsonl"

# 2. before/after report
$PY make_report.py --decisions "$RUNS/train_smart.jsonl" --k 40 --out "$RUNS/report.html"

# 3. apply -> filtered dataset (undecided records are kept, never silently dropped)
$PY filter_apply.py --decisions "$RUNS/train_smart.jsonl" --split train \
    --out-dir /scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq
```
Or just run `/go 200` (then `/go 500`, `/go 0 --apply` for the full split).

## Decision record schema (one JSON line per example)
`id, index, image, question, ground_truth, category, source_dataset, original_size,
strategy, model, blind{count,count_low,count_high,difficulty,confidence,reasoning,raw},
verify{your_count,verdict,confidence,reasoning,raw}, verdict, best_count, keep,
short_circuit`

## Pilot findings (train, smart strategy, gemini-3.1-pro-preview)
- 200-sample pilot keep rates: **v1 78.5% → v2/v3 ≈ 87-89%** after adding the two
  guardrails. The recovered ~24/200 were genuine hard-but-correct examples wrongly
  dropped over ±1-2 (oranges 47 vs GT 48 with one green fruit; biscuits; the "blue"
  keyboard where the blue/periwinkle boundary is fuzzy; dense FSC147 piles).
- Correct drops are dominated by: **category mismatch** (question asks "cranes",
  image is geese → model counts 0 → drop) and **large genuine over/under-counts**.
- Hand-checked 15+ images against the verdicts; CORRECT/PLAUSIBLE are reliable,
  and the remaining WRONG drops are defensible. Re-verify by hand at any new scale.

## Tuning the decision layer for free
```bash
# try a tighter / looser tolerance without re-calling Gemini:
$PY rescore.py --in "$RUNS/train_smart_v2.jsonl" --out "$RUNS/train_smart_tight.jsonl" \
    --abs-tol 1 --rel-tol 0.10
$PY compare_runs.py "$RUNS/train_smart_v2.jsonl" "$RUNS/train_smart_tight.jsonl"
```

## Notes
- Images are upscaled on disk (e.g. 384→2240); we downscale the long side to
  `--max-side` (default 1536) — never below the real source resolution's worth of detail.
- Concurrency `--workers` ≤ ~16 (org limit ≈ 200K tokens/min, shared).
- `gemini-3.1-pro-preview` is the only Gemini on this key; it has strong vision.
- For reported keep/drop numbers, spot-check a handful of images by hand (the `/go`
  command does this automatically) and watch specifically for hard examples wrongly
  marked `WRONG`.
