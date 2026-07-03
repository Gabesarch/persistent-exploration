#!/usr/bin/env python
"""Merge the exported human-review JSONs into a decisions file `filter_apply.py` consumes.

Each reviewer's site (`make_review_site.py`) exports a JSON like
  {"reviewer": "...", "exported_at": "...", "decisions": {"<id>": {"decision":"keep|drop",
   "corrected_count": <int|null>, "note": "..."}}}
This tool combines the (typically 4) exports, reports coverage + any conflicts, and writes:

  1. a merged human-decisions map keyed by id (`--merged-out`, an audit/portable summary),
  2. `gt_corrections.jsonl` — every reviewer-supplied corrected count (kept, dropped, or
     count-only rows), so a wrong-GT-but-keep example isn't silently shipped on the old GT, and
  3. the **override decisions JSONL** (`--decisions-out`) that `filter_apply.py` reads: it is the
     base `decisions.jsonl` with every human-reviewed PLAUSIBLE row overridden —
        keep -> keep=True,  verdict stays PLAUSIBLE      (still classified KEEP)
        drop -> keep=False, verdict="HUMAN_DROP"         (classified DROP)
     Re-running filter_apply on it regenerates the filtered tier with the humans' verdicts; the
     only ids whose keep/drop flips vs the auto run are the human DROPs.

Fail-closed, like the rest of the filter: conflicting decisions error out (--allow-conflicts),
reviewed ids that aren't a PLAUSIBLE tier row in --base error out (the exports don't match the
base), and an incompletely-reviewed tier errors out unless --allow-incomplete (which then tags
the output *.partial* and passes unreviewed rows through on the auto verdict). corrected_count
is captured but not auto-written into the dataset GT (filter_apply stays generic — apply
gt_corrections.jsonl separately).

  python merge_reviews.py review_chris-cai.json review_gabe-sarch.json \
      review_yipeng-zhang.json review_haoyang-wu.json \
      --base .../full_sync/decisions.jsonl --decisions-out .../decisions.human_reviewed.jsonl

Dependency-free (stdlib only).
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

TERMINAL = {"keep", "drop"}


def load_review(path: str):
    """-> (reviewer, {id: {decision, corrected_count, note}})."""
    obj = json.load(open(path))
    if not isinstance(obj, dict) or "decisions" not in obj:
        sys.exit(f"[merge] FATAL: {path} is not a review export (no 'decisions' key)")
    reviewer = obj.get("reviewer") or Path(path).stem
    dec = obj.get("decisions") or {}
    if not isinstance(dec, dict):
        sys.exit(f"[merge] FATAL: {path} 'decisions' is not an object")
    return reviewer, dec


def norm_count(v):
    if v is None or v == "":
        return None
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("reviews", nargs="+", help="exported review JSON files (one per reviewer)")
    ap.add_argument("--base",
                    default="/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/"
                            "full_sync/decisions.jsonl",
                    help="the full decisions JSONL to override onto (default full_sync). "
                         "Pass --no-base to emit only the reviewed rows.")
    ap.add_argument("--no-base", action="store_true",
                    help="emit override-only decisions (reviewed ids only; needs "
                         "filter_apply --allow-partial downstream)")
    ap.add_argument("--verdict", default="PLAUSIBLE",
                    help="the tier that was reviewed (defines the coverage universe; default PLAUSIBLE)")
    ap.add_argument("--drop-verdict", default="HUMAN_DROP",
                    help="verdict stamped on human DROPs (any non-keep, non-ERROR label; default HUMAN_DROP)")
    ap.add_argument("--decisions-out", default=None,
                    help="output override decisions JSONL (default: alongside --base as "
                         "decisions.human_reviewed.jsonl)")
    ap.add_argument("--merged-out", default=None,
                    help="output merged human-decisions map keyed by id (default: alongside "
                         "--decisions-out as reviews_merged.json)")
    ap.add_argument("--allow-conflicts", action="store_true",
                    help="if two exports give the same id different decisions, take the last "
                         "file's (default: error out)")
    ap.add_argument("--allow-incomplete", action="store_true",
                    help="proceed when the tier is not fully reviewed (default: error out, "
                         "fail-closed). Emits a *.partial.* decisions file; unreviewed rows "
                         "pass through on the auto verdict.")
    ap.add_argument("--gt-corrections-out", default=None,
                    help="output JSONL of reviewer-supplied corrected counts (default: "
                         "alongside --decisions-out as gt_corrections.jsonl)")
    args = ap.parse_args()

    # ---- 1. load + merge exports (track provenance + incomplete signals) ----
    merged = {}                      # id -> {reviewer, decision, corrected_count, note}
    incomplete = {}                  # id -> {reviewer, corrected_count, note}  (count/note, no keep/drop)
    per_reviewer = defaultdict(lambda: {"keep": 0, "drop": 0, "incomplete": 0, "file": None})
    conflicts = []
    for path in args.reviews:
        reviewer, dec = load_review(path)
        per_reviewer[reviewer]["file"] = path
        for rid, d in dec.items():
            d = d or {}
            decision = d.get("decision")
            cc = norm_count(d.get("corrected_count"))
            note = d.get("note") or ""
            if decision not in TERMINAL:
                # touched but not decided (a note/count with no keep/drop) — captured,
                # never silently folded into the auto keep
                if cc is not None or note:
                    per_reviewer[reviewer]["incomplete"] += 1
                    incomplete[rid] = {"reviewer": reviewer, "corrected_count": cc, "note": note}
                continue
            per_reviewer[reviewer][decision] += 1
            if rid in merged and merged[rid]["decision"] != decision:
                conflicts.append((rid, merged[rid], {"reviewer": reviewer, "decision": decision}))
            merged[rid] = {"reviewer": reviewer, "decision": decision,
                           "corrected_count": cc, "note": note}

    # ---- 2. report ----
    n_keep = sum(1 for v in merged.values() if v["decision"] == "keep")
    n_drop = sum(1 for v in merged.values() if v["decision"] == "drop")
    n_corr = sum(1 for v in merged.values() if v["corrected_count"] is not None)
    print(f"[merge] {len(args.reviews)} export(s) -> {len(merged)} reviewed ids "
          f"(keep={n_keep}, drop={n_drop}, with corrected_count={n_corr})")
    for reviewer, s in per_reviewer.items():
        inc = f", incomplete={s['incomplete']}" if s["incomplete"] else ""
        print(f"    {reviewer}: keep={s['keep']} drop={s['drop']}{inc}  [{Path(s['file']).name}]")
    if incomplete:
        print(f"[merge] {len(incomplete)} id(s) have a corrected_count/note but no keep/drop "
              f"(NOT counted as reviewed; captured in gt_corrections)")

    # conflicts are decided regardless of --base
    if conflicts:
        print(f"[merge] {len(conflicts)} CONFLICT(s) — same id, different decisions:")
        for rid, a, b in conflicts[:20]:
            print(f"    {rid}: {a['reviewer']}={a['decision']} vs {b['reviewer']}={b['decision']}")
        if not args.allow_conflicts:
            sys.exit("[merge] FATAL: conflicting decisions. Resolve the overlap, or pass "
                     "--allow-conflicts to take the last file's decision.")
        print("[merge] --allow-conflicts: taking the last file's decision for each.")

    # ---- 3. validate against --base + coverage gate (fail-closed) ----
    base_rows = None
    base_by_id = {}
    partial = False
    if not args.no_base:
        if not Path(args.base).exists():
            sys.exit(f"[merge] FATAL: --base not found: {args.base}")
        base_rows = [json.loads(l) for l in open(args.base) if l.strip()]
        base_by_id = {str(r.get("id")): r for r in base_rows}
        universe = {rid for rid, r in base_by_id.items() if r.get("verdict") == args.verdict}
        reviewed = set(merged)
        unknown = sorted(reviewed - set(base_by_id))
        off_tier = sorted((reviewed & set(base_by_id)) - universe)
        n_unrev = len(universe) - len(reviewed & universe)
        print(f"[merge] coverage: {len(reviewed & universe)}/{len(universe)} of the "
              f"{args.verdict} tier reviewed ({n_unrev} unreviewed)")
        # exports must match the base they are applied to — a reviewed id that is not a
        # tier row in --base means the wrong base or a stale export; a human 'keep' on it
        # would launder an off-tier (e.g. WRONG) record into the kept set. Fail closed.
        if unknown or off_tier:
            for rid in unknown[:10]:
                print(f"    NOT-IN-BASE: {rid}")
            for rid in off_tier[:10]:
                print(f"    OFF-TIER({base_by_id[rid].get('verdict')}): {rid}")
            sys.exit(f"[merge] FATAL: {len(unknown)} reviewed id(s) absent from --base and "
                     f"{len(off_tier)} not in the {args.verdict} tier. The exports don't match "
                     f"--base (wrong base file or stale shard). Refusing to apply.")
        # coverage gate: don't silently ship a 'human_reviewed' file full of unreviewed rows
        if n_unrev > 0:
            if not args.allow_incomplete:
                sys.exit(f"[merge] FATAL: {n_unrev}/{len(universe)} {args.verdict} ids unreviewed. "
                         f"Refusing to write a would-be-complete human-reviewed file. Finish the "
                         f"review, or pass --allow-incomplete (emits a *.partial* file; unreviewed "
                         f"rows pass through on the auto verdict).")
            partial = True
            print(f"[merge] --allow-incomplete: {n_unrev} unreviewed {args.verdict} row(s) pass "
                  f"through on the auto verdict; output tagged .partial")

    # ---- 4. resolve output paths (tag .partial when coverage is incomplete) ----
    decisions_out = Path(args.decisions_out) if args.decisions_out else \
        Path(args.base).with_name("decisions.human_reviewed.jsonl")
    if partial:
        decisions_out = decisions_out.with_suffix(".partial" + decisions_out.suffix)
    merged_out = Path(args.merged_out) if args.merged_out else \
        decisions_out.with_name("reviews_merged.json")
    gt_out = Path(args.gt_corrections_out) if args.gt_corrections_out else \
        decisions_out.with_name("gt_corrections.jsonl")
    decisions_out.parent.mkdir(parents=True, exist_ok=True)

    # ---- 5. write merged human-decisions map (keyed by id) ----
    merged_out.write_text(json.dumps(
        {"n_reviewed": len(merged), "keep": n_keep, "drop": n_drop, "partial": partial,
         "reviewers": {r: {k: s[k] for k in ("keep", "drop", "incomplete")}
                       for r, s in per_reviewer.items()},
         "decisions": merged, "incomplete": incomplete}, indent=2))
    print(f"[merge] wrote merged human decisions -> {merged_out}")

    # ---- 6. write GT-correction candidates (kept/dropped/incomplete rows w/ a corrected count) ----
    gt_rows = []
    for rid, hv in merged.items():
        if hv["corrected_count"] is not None:
            gt_rows.append({"id": rid, "labeled_count": base_by_id.get(rid, {}).get("ground_truth"),
                            "corrected_count": hv["corrected_count"], "decision": hv["decision"],
                            "reviewer": hv["reviewer"], "note": hv["note"]})
    for rid, iv in incomplete.items():
        if iv["corrected_count"] is not None:
            gt_rows.append({"id": rid, "labeled_count": base_by_id.get(rid, {}).get("ground_truth"),
                            "corrected_count": iv["corrected_count"], "decision": None,
                            "reviewer": iv["reviewer"], "note": iv["note"]})
    if gt_rows:
        with open(gt_out, "w") as f:
            for row in gt_rows:
                f.write(json.dumps(row) + "\n")
        print(f"[merge] wrote {len(gt_rows)} GT-correction candidate(s) -> {gt_out}")

    # ---- 7. write the filter_apply-consumable override decisions JSONL ----
    flipped = []  # human DROPs = the ids whose keep/drop differs from the auto run
    with open(decisions_out, "w") as f:
        if args.no_base:
            for rid, hv in merged.items():
                keep = hv["decision"] == "keep"
                row = {"id": rid, "keep": keep,
                       "verdict": args.verdict if keep else args.drop_verdict,
                       "human_review": hv}
                if not keep:
                    flipped.append(rid)
                f.write(json.dumps(row) + "\n")
        else:
            for r in base_rows:
                rid = str(r.get("id"))
                hv = merged.get(rid)
                if hv is not None:
                    r = dict(r)
                    r["verdict_auto"] = r.get("verdict")
                    r["keep_auto"] = r.get("keep")
                    r["human_review"] = hv
                    if hv["decision"] == "keep":
                        r["keep"] = True
                        r["verdict"] = args.verdict
                    else:
                        r["keep"] = False
                        r["verdict"] = args.drop_verdict
                        flipped.append(rid)
                f.write(json.dumps(r) + "\n")
    n_lines = len(merged) if args.no_base else len(base_rows)
    print(f"[merge] wrote override decisions -> {decisions_out}  ({n_lines} rows)")
    print(f"[merge] human DROPs flip {len(flipped)} id(s) kept->dropped vs the auto run"
          + (f": {flipped[:8]}{'…' if len(flipped) > 8 else ''}" if flipped else ""))
    if gt_rows:
        print(f"[merge] NOTE: {len(gt_rows)} row(s) carry a corrected_count. filter_apply keeps "
              f"each kept record's source GT unchanged — apply {gt_out.name} to the dataset as a "
              f"separate GT-correction step if desired.")
    if args.no_base:
        print("[merge] NOTE: --no-base output omits ground_truth/question/etc, so filter_apply's "
              "dropped_*.jsonl audit will be sparse. Prefer base mode for a full drop audit.")
    print("[merge] feed the override decisions into filter_apply.py to regenerate the filtered tier.")


if __name__ == "__main__":
    main()
