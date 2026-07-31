# review_site — PLAUSIBLE-tier manual review

The human-review layer that sits **on top of** the GT filter in `../` (the filter decides
CORRECT/PLAUSIBLE/WRONG/…; this lets people re-review the PLAUSIBLE tier by hand). Kept in its
own folder so the filter pipeline and the review tooling don't tangle.

| file | role |
|---|---|
| `make_review_site.py` | build one self-contained HTML per reviewer from a `decisions.jsonl` (keep/drop + optional 100%-confident count + JSON export) + an `index.html` hub |
| `serve_review.py` | stdlib server for the generated site with a `POST /submit` upload endpoint (reviewers submit straight to scratch); reviewers run it from the deploy dir |
| `merge_reviews.py` | merge the reviewers' exported JSONs → `confident_counts.jsonl` + an override decisions file that `../filter_apply.py` consumes |
| `REVIEWING.md` | concise reviewer-facing guide (copied next to the HTML at deploy time) |

Run the generator/merge from the parent `dataset_filter/` dir (so `filter_apply.py` is a
sibling), e.g. `python review_site/make_review_site.py`. Full workflow, flags, deploy, and the
two access recipes: **[../README.md → Manual review of the PLAUSIBLE tier](../README.md#manual-review-of-the-plausible-tier)**.
Dependency-free (stdlib + PIL, vanilla JS).
