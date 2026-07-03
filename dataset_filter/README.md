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
| `make_review_site.py` | builds the **PLAUSIBLE-tier manual-review site** — one self-contained HTML per reviewer with keep/drop + corrected-count capture and JSON export ([below](#manual-review-of-the-plausible-tier)) |
| `serve_review.py` | stdlib server for the review site with a `POST /submit` upload endpoint, so reviewers submit straight to scratch instead of downloading + re-uploading |
| `merge_reviews.py` | merges the reviewers' exported JSONs → an override decisions file `filter_apply.py` consumes |
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

## Manual review of the PLAUSIBLE tier

`PLAUSIBLE` is the "GT is believable but unverifiable" middle band the auto-filter keeps —
the valuable hard examples that separate the standard tier (`CORRECT` + `PLAUSIBLE`) from a
strict correct-only tier. To put a **human** verdict on those, `make_review_site.py` builds a
manual-review website: it reads the run's `decisions.jsonl`, keeps `verdict == PLAUSIBLE`,
splits the ids deterministically across the reviewers, and emits **one self-contained HTML
page per reviewer** + an `index.html` hub. Each page embeds its images as base64 JPEGs (no
server needed to view), shows the question, the neutrally-labeled count (the thing being
audited — *not* trusted GT), the source/category, and both model passes (blind + verify), with
a clean/GT-points toggle and a click-to-zoom lightbox. The reviewer records **keep / drop**
(required to count as reviewed), an optional **corrected count**, and a note; decisions persist
in `localStorage` keyed by record id (refresh-safe) and download via **Export decisions JSON**.
`merge_reviews.py` folds the exports back into a decisions file `filter_apply.py` re-applies.

```bash
PY=/scratch/gpfs/ZHUANGL/lc1556/conda/envs/reason-vlm/bin/python
RUNS=/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync
DATA=/scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq
cd /home/lc1556/repo/reason-vlm/vero2/dataset_filter

# 1. generate the 4-reviewer site (defaults target the full_sync run + deploy dir)
$PY make_review_site.py                    # -> $RUNS/manual_review/{index,review_*}.html
chmod -R g+rX "$RUNS/manual_review"        # group zhuangl can read/serve it

# 2. reviewers each open their shard (see "Two ways to open your shard" below),
#    keep/drop every card, then click "Export decisions JSON" and send back review_<name>.json

# 3. merge the exports -> an override decisions file, then re-apply to regenerate the tier
$PY merge_reviews.py review_chris-cai.json review_gabe-sarch.json \
    review_yipeng-zhang.json review_haoyang-wu.json \
    --base "$RUNS/decisions.jsonl" --decisions-out "$RUNS/decisions.human_reviewed.jsonl"
$PY filter_apply.py --data-dir "$DATA" --dataset combined_counting --local-meta \
    --decisions "$RUNS/decisions.human_reviewed.jsonl" --splits train val \
    --out-dir "$RUNS/filtered_human_reviewed" --allow-partial
```

The default split is **round-robin over sorted ids** (`--split-mode`), which balances the
source datasets (fsc147/rec8k/rpine/pixmo) across reviewers; 801 → shards of **201/200/200/200**.
The generator asserts the shards are disjoint and cover every id, and `--help` documents every
flag (`--decisions`, `--reviewers`, `--side`, `--limit`, …). Regeneration is idempotent.

`merge_reviews.py` maps each human **keep** → `keep=True`/verdict `PLAUSIBLE` (still classified
KEEP) and each **drop** → `keep=False`/verdict `HUMAN_DROP` (classified DROP), leaving unreviewed
rows on the auto verdict, so re-applying flips exactly the human-dropped ids. Like the rest of the
filter it is **fail-closed**: it errors out if a reviewed id isn't a PLAUSIBLE row in `--base`
(the exports don't match the base) or if the tier isn't fully reviewed — pass `--allow-incomplete`
to proceed on a partial review (the output is then tagged `*.partial*` and unreviewed rows keep
the auto verdict). Reviewer-supplied **corrected counts** are written to `gt_corrections.jsonl`
(with the labeled count) so a wrong-GT-but-keep example isn't silently shipped on the old GT;
`filter_apply.py` keeps each source record's GT as-is, so apply those corrections as a separate step.

### Two ways for a reviewer to open their shard
Everything (images included) is embedded, so a reviewer needs **no GPU** to review. A concise
reviewer-facing guide lives in `REVIEWING.md` (copied into the deploy dir next to the HTML).

**A. Download the single HTML file (offline).** From VS Code's Della explorer (or `scp`), download
`manual_review/review_<name>.html` and open it in any local browser. Progress lives in that
browser's `localStorage`. When done, click **Export JSON** and send the file back for merging.

**B. Serve from a Della login node + VS Code port-forward (submit straight back).**
```bash
ssh <netid>@della.princeton.edu
cd /scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync/manual_review
python serve_review.py                     # prints the localhost URL + port to forward
# VS Code (Remote-SSH to della): Ports panel → Forward the printed port,
# open http://localhost:<port>/ → click your name. When done, click "Submit to Della".
```
`serve_review.py` is a stdlib server: it serves the site **and** accepts `POST /submit`, writing
`exports/review_<name>.json` straight into the folder — no download/re-upload. It binds to
localhost (SSH-tunnel-only) and the submit filename is sanitized to `review_<slug>.json`, so a
submit can never escape `exports/`. The site is **group `zhuangl`-readable** and `exports/` is
setgid + group-writable, so all four reviewers (each running their own `serve_review.py`) submit
into one place; the coordinator collects them and runs `merge_reviews.py`. `index.html` restates
these recipes; the page's **Export JSON** button is the offline fallback if there's no server.

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
