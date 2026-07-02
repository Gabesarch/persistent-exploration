# Dataset ground-truth (GT) filter

A **reusable** toolkit that audits a dataset's ground-truth labels with a strong
multimodal model and keeps only the examples whose GT is correct-or-plausible,
dropping ones that are **wrong / ambiguous / unanswerable**. Judging uses
**Gemini `gemini-3.1-pro-preview`** via the Princeton AI-sandbox (Portkey gateway; keys
auto-loaded from `../../AI-sandbox/.env`, never committed).

First use case: `combined_counting_hq` (a counting dataset). Nothing below is
counting-specific except the prompts in `prompts.py` — the pipeline reads the standard
**verl** record schema and is driven entirely by `--dataset <PREFIX>` + `--split`, so
the next dataset reuses the same tooling (see **[Filter a new dataset](#filter-a-new-dataset)**).

The reason to filter at all: **GT is not trusted** — we audit it. So the whole pipeline
is *fail-closed* and every decision is traceable (verdict + both counts + reasoning).

## The core idea (why not just "ask the model")

A dataset's value is often its hard examples, so the filter must not reject an example
merely because it is hard. Two naive approaches each fail:

| approach | failure |
|---|---|
| blind judge (input only), require `== GT` | over-rejects correct-but-hard examples the model slightly misjudges |
| show the GT and ask "is this right?" | sycophancy — rubber-stamps wrong GTs |

**`smart` (default) combines them.** An independent **blind** pass (the model never sees
the GT) → if it matches GT exactly, `CORRECT` in one call. On mismatch, an **answer-aware
auditor** that is explicitly told *hard ≠ wrong* returns:

- `CORRECT` — recount matches GT
- `PLAUSIBLE` — can't pin exact, but GT is believable with no evidence it's wrong → **KEEP** (the valuable hard examples)
- `WRONG` — concrete evidence the true value differs → drop
- `AMBIGUOUS` — ill-posed / subjective question ("innermost stripe", "all the windows" across a painting) → drop
- `UNANSWERABLE` — requested category absent / unusable input → drop

`keep = verdict in {CORRECT, PLAUSIBLE}`.

### Guardrails (from the pilots)
- **Judge only visible instances.** Prompts forbid inferring hidden objects (a solid
  3-D stack's interior, background buildings) that point-annotated GT excludes.
- **Numeric-tolerance override, small-count-safe.** If the auditor says WRONG but its own
  count is within `max(abs_tol, floor(rel_tol*gt))` of GT **and > 0**, override to
  PLAUSIBLE+keep. Defaults `abs_tol=0, rel_tol=0.15` → tolerance is **0 for gt ≤ 6**
  (small counts must match ~exactly: GT=1/model=2 and GT=2/model=4 are NOT rescued), and
  the 15 % band only opens on large counts (gt=100 → ±15). The `> 0` guard is essential:
  `model_count == 0` means "category absent/mismatch" (geese ≠ cranes), a real drop.
- **GT points as a stage-2 audit guide.** Where records ship point annotations
  (`extra_info.metadata.original_points`, `GT == len(points)`), stage-2 draws them as
  magenta rings and tells the model they're a *noisy* guide to audit — not truth. Stage-1
  blind still runs WITHOUT points, preserving an unbiased signal.

## Files
| file | role |
|---|---|
| `prompts.py` | blind + auditor prompts (anti-laziness / anti-sycophancy / visible-only / ambiguity) — the only dataset-specific piece |
| `localize_paths.py` | rewrite metadata image paths to a local copy → `*_<split>.local.verl.jsonl`; **fails closed** if any image is missing |
| `filter_counting.py` | synchronous filter; resumable per-example decisions JSONL + `<out>.config.json` guard + self-healing multi-key failover |
| `batch_runner.py` | same pipeline via the Portkey **batch API** (~50 % cheaper) — ready but blocked on the sandbox key (see [Batch](#batch-status)) |
| `filter_apply.py` | decisions → `<dataset>_<split>.filtered.verl.jsonl` (+ keep_ids / dropped / undecided); **fail-closed** |
| `rescore.py` | re-derive keep/verdict under new `--abs-tol`/`--rel-tol` from stored outputs — **no new API calls** |
| `compare_runs.py` | diff two decision files (keep rate, verdict flips, overrides) |
| `cost_estimate.py` | project $ to filter a full split from measured tokens |
| `make_report.py` / `make_exam_report.py` / `make_compare_report.py` | HTML reports; `make_exam_report` is GT-neutral + zoomable for manual audit |
| `../.claude/commands/gt-filter.md` | the `/gt-filter` slash command that drives the whole flow |

## Quick use (synchronous — the working path today)
```bash
PY=/scratch/gpfs/ZHUANGL/lc1556/conda/envs/reason-vlm/bin/python
DATA=/scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq
RUNS=/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs
cd /home/lc1556/repo/reason-vlm/vero2/dataset_filter

# 0. one-time: self-contained local images + fail-closed coverage check
$PY localize_paths.py --data-dir "$DATA" --dataset combined_counting

# 1. filter (resumable; ALWAYS pin --reasoning-effort; 2-key failover on quota)
$PY filter_counting.py --data-dir "$DATA" --dataset combined_counting --local-meta \
    --split train --n 0 --strategy smart --reasoning-effort high \
    --abs-tol 0 --rel-tol 0.15 --workers 8 --out "$RUNS/combined_counting_train.jsonl"

# 2. GT-neutral, zoomable report for manual spot-check
$PY make_exam_report.py --run "gemini=$RUNS/combined_counting_train.jsonl" \
    --points-from "$DATA" --out "$RUNS/combined_counting_train_exam.html"

# 3. apply -> filtered dataset (FAILS if anything is undecided; --allow-partial to override)
$PY filter_apply.py --data-dir "$DATA" --dataset combined_counting --local-meta \
    --decisions "$RUNS/combined_counting_train.jsonl" --splits train --out-dir "$RUNS/filtered"
```
Or just run `/gt-filter 200` → `/gt-filter 0 --effort high --apply` for the full split.

## Filter a new dataset
No code changes — the pipeline is dataset-agnostic. To filter `mydata`:
1. Put verl JSONL(s) at `<DATA>/mydata_<split>.verl.jsonl`. Required per record:
   `images[0]` (path), `extra_info.id` (unique), `reward_model.ground_truth`.
   Optional: `extra_info.metadata.original_points` + `original_size` (enables the point
   overlay), `extra_info.question`, `extra_info.metadata.annotation_category`.
2. `localize_paths.py --data-dir <DATA> --dataset mydata` (skip if paths already point at
   readable local images, then omit `--local-meta`).
3. `filter_counting.py --dataset mydata --split <sp> ...` then `filter_apply.py --dataset mydata ...`.
4. If the task isn't counting, adapt `prompts.py` (the judging rubric) — that's the only
   task-specific file; the harness, resume, tolerance, apply, and reports are generic.

## Robustness (fail-closed everywhere — GT is not trusted)
- **Resumable + config-guarded.** Re-running the same `--out` continues where it left off.
  A `<out>.config.json` sidecar records the run settings; resuming with an incompatible
  setting (model / effort / tolerance / split / dataset …) is **refused** (`--force` to
  override). New config fields never spuriously block older runs.
- **Terminal-only "done".** A row counts as done only with an explicit `keep ∈ {True,False}`
  and a valid verdict. API/parse failures (`ERROR` / `PARSE_ERROR`) are **re-queued** on
  the next run — never silently treated as a drop, never skipped.
- **Fail-closed apply.** `filter_apply.py` classifies every source id keep / drop /
  **undecided** (missing, `keep is None`, or `ERROR`/`PARSE_ERROR`) and **errors out** if
  any is undecided, rather than emitting a partial dataset that looks complete.
  `--allow-partial` excludes + lists undecided (`undecided_<split>.txt`); output is named
  `*.filtered.partial.verl.jsonl`. Which verdicts count as keep is set by `--keep-verdicts`
  (default `CORRECT,PLAUSIBLE`); pass `--keep-verdicts CORRECT --out-tag correct` for a
  strict **correct-only** tier — PLAUSIBLE then classifies as drop and the output is named
  `*_<split>.correct[.partial].verl.jsonl`. Undecided is never kept.
- **Multi-key failover (self-healing).** `AI_SANDBOX_KEY` (primary) → `AI_SANDBOX_KEY2…`
  (backups) on quota / auth exhaustion; transient per-minute rate-limits are backed-off, not
  failed-over. A key that returns a usage/auth error is marked dead and skipped for the rest
  of the run; if **every** key looks dead the set is cleared and re-probed from the top
  (usage windows may have cleared). It never permanently gives up on the whole run — a row
  only becomes `ERROR` after its own retry budget and stays resumable on the next run.

## Batch status
Provider-native batch (`batch_runner.py`, ~50 % cheaper) is **not usable on the sandbox
key**: it routes through a multi-target Portkey config, and `/v1/files`+`/v1/batches`
require a **single** target. Client-side overrides don't help — it needs Princeton to
provide a single-target **config id** or **virtual key** for the Gemini/Vertex provider
(Vertex batch also needs a same-region GCS bucket). `batch_runner.py` is complete and
`--selftest`-gated for the moment that lands; until then use the synchronous path.

## Decision record schema (one JSON line per example)
`id, index, image, question, ground_truth, category, source_dataset, original_size,
n_points, strategy, model, reasoning_effort, abs_tol, rel_tol,
blind{count,count_low,count_high,difficulty,confidence,reasoning[,raw]},
verify{your_count,verdict,confidence,reasoning[,raw]}, verdict, best_count, keep,
short_circuit, used_points, override, usage_total{prompt,completion,total,calls}`

## Cost
Driven by `--reasoning-effort` (hidden thinking tokens dominate). Measured on
`combined_counting_hq` at **high** effort: ~6.2k tok/example, ~1.5 calls/example ⇒
**~$630 synchronous** for train+val (10,589) at gemini list price ($2/$12 per 1M);
batch would be ~half but is blocked (above). The org limit is ~200K tok/min shared, so
keep `--workers ≤ ~12`. Use `cost_estimate.py` to project from a pilot.

## Pilot findings & model choice
- By hand-eval on disagreement cases (GT treated as unverified), **gemini-3.1-pro is the
  best auditor** (5/5) vs gemini-3.5-flash 4/5, gpt-5 3/5, gpt-5.5 2/5 — the Gemini family
  beats GPT-5 variants at this visual-audit task. Use gemini-3.1-pro-preview.
- Full-run keep rate ≈ **80 %** (drop ≈ 20 %: WRONG ~15 %, AMBIGUOUS ~3 %, UNANSWERABLE ~1 %).
- Recurring GT failure the filter catches: dense FSC147/rec8k annotations that **under-count**
  (mark ~half the objects). Always spot-check a handful by hand at any new scale, watching
  specifically for a hard-but-correct example wrongly marked `WRONG`.

## Tune the decision layer for free
```bash
$PY rescore.py --in "$RUNS/combined_counting_train.jsonl" \
    --out "$RUNS/combined_counting_train_tight.jsonl" --abs-tol 1 --rel-tol 0.10
$PY compare_runs.py "$RUNS/combined_counting_train.jsonl" "$RUNS/combined_counting_train_tight.jsonl"
```
