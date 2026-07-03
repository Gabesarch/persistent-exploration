#!/usr/bin/env python
"""Merge the exported human-review JSONs into a decisions file `filter_apply.py` consumes.

Each reviewer's site (`make_review_site.py`) exports a JSON like
  {"reviewer": "...", "exported_at": "...", "decisions": {"<id>": {"decision":"keep|drop",
   "corrected_count": <int|null>, "note": "..."}}}
This tool combines the (typically 4) exports, reports coverage + any conflicts, and writes:

  1. a merged human-decisions map keyed by id (`--merged-out`, an audit/portable summary), and
  2. the **override decisions JSONL** (`--decisions-out`) that `filter_apply.py` reads: it is the
     base `decisions.jsonl` with every human-reviewed PLAUSIBLE row overridden —
        keep -> keep=True,  verdict stays PLAUSIBLE      (still classified KEEP)
        drop -> keep=False, verdict="HUMAN_DROP"         (classified DROP)
     Rows nobody reviewed are copied through unchanged (still terminal, so apply never
     chokes). Re-running filter_apply on this file regenerates the filtered tier with the
     humans' verdicts; the only ids whose keep/drop flips vs the auto run are the human DROPs.

  python merge_reviews.py review_chris-cai.json review_gabe-sarch.json \
      review_yipeng-zhang.json review_haoyang-wu.json \
      --base .../full_sync/decisions.jsonl --decisions-out .../decisions.human_reviewed.jsonl

Dependency-free (stdlib only). Fail-loud on conflicting decisions unless --allow-conflicts.
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
    args = ap.parse_args()

    # ---- 1. load + merge exports (track provenance for conflict reporting) ----
    merged = {}                      # id -> {reviewer, decision, corrected_count, note}
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
                # touched but not decided (e.g. a note/count with no keep/drop)
                if cc is not None or note:
                    per_reviewer[reviewer]["incomplete"] += 1
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

    base_rows = None
    universe = None
    if not args.no_base:
        if not Path(args.base).exists():
            sys.exit(f"[merge] FATAL: --base not found: {args.base}")
        base_rows = [json.loads(l) for l in open(args.base) if l.strip()]
        base_by_id = {str(r.get("id")): r for r in base_rows}
        universe = {rid for rid, r in base_by_id.items() if r.get("verdict") == args.verdict}
        reviewed = set(merged)
        unknown = sorted(reviewed - set(base_by_id))
        off_tier = sorted((reviewed & set(base_by_id)) - universe)
        print(f"[merge] coverage: {len(reviewed & universe)}/{len(universe)} of the "
              f"{args.verdict} tier reviewed ({len(universe) - len(reviewed & universe)} unreviewed)")
        if unknown:
            print(f"[merge] WARNING: {len(unknown)} reviewed id(s) not in --base at all: "
                  f"{unknown[:5]}{'…' if len(unknown) > 5 else ''}")
        if off_tier:
            print(f"[merge] WARNING: {len(off_tier)} reviewed id(s) are not {args.verdict} in "
                  f"--base (still applied): {off_tier[:5]}{'…' if len(off_tier) > 5 else ''}")

    if conflicts:
        print(f"[merge] {len(conflicts)} CONFLICT(s) — same id, different decisions:")
        for rid, a, b in conflicts[:20]:
            print(f"    {rid}: {a['reviewer']}={a['decision']} vs {b['reviewer']}={b['decision']}")
        if not args.allow_conflicts:
            sys.exit("[merge] FATAL: conflicting decisions. Resolve the overlap, or pass "
                     "--allow-conflicts to take the last file's decision.")
        print("[merge] --allow-conflicts: taking the last file's decision for each.")

    # ---- 3. write merged human-decisions map (keyed by id) ----
    decisions_out = Path(args.decisions_out) if args.decisions_out else \
        Path(args.base).with_name("decisions.human_reviewed.jsonl")
    merged_out = Path(args.merged_out) if args.merged_out else \
        decisions_out.with_name("reviews_merged.json")
    decisions_out.parent.mkdir(parents=True, exist_ok=True)
    merged_out.write_text(json.dumps(
        {"n_reviewed": len(merged), "keep": n_keep, "drop": n_drop,
         "reviewers": {r: {k: s[k] for k in ("keep", "drop", "incomplete")}
                       for r, s in per_reviewer.items()},
         "decisions": merged}, indent=2))
    print(f"[merge] wrote merged human decisions -> {merged_out}")

    # ---- 4. write the filter_apply-consumable override decisions JSONL ----
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
    if n_corr:
        print(f"[merge] NOTE: {n_corr} kept row(s) carry a corrected_count (GT-correction "
              f"candidates); filter_apply keeps the source record's GT unchanged — apply "
              f"corrections as a separate step if desired.")
    print("[merge] feed --decisions-out into filter_apply.py to regenerate the filtered tier.")


if __name__ == "__main__":
    main()
