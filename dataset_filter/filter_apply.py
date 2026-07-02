#!/usr/bin/env python
"""Apply filter decisions to produce a filtered verl jsonl + keep-id list.

FAIL-CLOSED. A record is only written to the filtered dataset when it has an
explicit terminal decision:
  * keep   -> keep is True   (verdict CORRECT/PLAUSIBLE)
  * drop   -> keep is False  (verdict WRONG/AMBIGUOUS/UNANSWERABLE)
Anything else is **undecided**: id missing from the decisions file, keep is null,
or a non-terminal verdict (ERROR / PARSE_ERROR from a failed API/parse). Undecided
records are NEVER silently kept or dropped.

By default, if ANY requested record is undecided, this errors out and writes
nothing — so applying a partial/pilot decisions file can't masquerade as a full
filtered dataset. Pass --allow-partial to proceed anyway: undecided records are
then EXCLUDED, the output is named `*.filtered.partial.verl.jsonl`, and their ids
are written to `undecided_<split>.txt`.

  python filter_apply.py --decisions runs/full_run/decisions.jsonl \
      --data-dir .../combined_counting_hq --local-meta --splits train val \
      --out-dir runs/full_run/filtered
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Keep the terminal/keep semantics identical to the filter (single source of truth
# if importable; otherwise fall back to a local copy so apply needs no API client).
try:
    from filter_counting import is_terminal  # noqa: F401
except Exception:  # pragma: no cover
    NONTERMINAL_VERDICTS = {"ERROR", "PARSE_ERROR"}

    def is_terminal(row: dict) -> bool:
        return (row.get("keep") in (True, False)
                and row.get("verdict") not in NONTERMINAL_VERDICTS)


def classify(d):
    """-> 'keep' | 'drop' | 'undecided' for a decision row (or None if absent)."""
    if d is None or not is_terminal(d):
        return "undecided"
    return "keep" if d["keep"] is True else "drop"


def source_path(data_dir, split, local):
    name = (f"combined_counting_{split}.local.verl.jsonl" if local
            else f"combined_counting_{split}.verl.jsonl")
    return os.path.join(data_dir, name)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--data-dir",
                    default="/scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq")
    ap.add_argument("--local-meta", action="store_true",
                    help="read combined_counting_<split>.local.verl.jsonl")
    ap.add_argument("--splits", nargs="+", default=["train"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--allow-partial", action="store_true",
                    help="proceed even if some records are undecided (excluded + listed)")
    args = ap.parse_args()

    decisions = {}
    for line in open(args.decisions):
        line = line.strip()
        if not line:
            continue
        d = json.loads(line)
        decisions[d["id"]] = d          # last write wins

    # ---- pass 1: classify everything up front; do not write until validated ----
    plans = []  # (split, src, [(rec, rid, cls)], counts)
    grand_undecided = 0
    for split in args.splits:
        src = source_path(args.data_dir, split, args.local_meta)
        if not os.path.exists(src):
            sys.exit(f"[apply] FATAL: source not found: {src}")
        rows = []
        counts = {"keep": 0, "drop": 0, "undecided": 0}
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                rid = rec.get("extra_info", {}).get("id")
                cls = classify(decisions.get(rid))
                counts[cls] += 1
                rows.append((rec, rid, cls))
        grand_undecided += counts["undecided"]
        plans.append((split, src, rows, counts))
        print(f"[apply] {split}: total={len(rows)}  keep={counts['keep']}  "
              f"drop={counts['drop']}  undecided={counts['undecided']}")

    if grand_undecided and not args.allow_partial:
        sys.exit(
            f"[apply] FATAL: {grand_undecided} undecided record(s) (missing / ERROR / "
            f"PARSE_ERROR). Refusing to write a partial dataset. Finish the run, or pass "
            f"--allow-partial to exclude and list them.")

    # ---- pass 2: write ----
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    partial = bool(grand_undecided)  # only true when --allow-partial got us here
    suffix = "filtered.partial" if partial else "filtered"
    for split, src, rows, counts in plans:
        out_path = os.path.join(args.out_dir, f"combined_counting_{split}.{suffix}.verl.jsonl")
        drop_path = os.path.join(args.out_dir, f"dropped_{split}.jsonl")
        ids_path = os.path.join(args.out_dir, f"keep_ids_{split}.json")
        undec_path = os.path.join(args.out_dir, f"undecided_{split}.txt")

        keep_ids, drop_ids, undecided = [], [], []
        with open(out_path, "w") as fout, open(drop_path, "w") as fdrop:
            for rec, rid, cls in rows:
                if cls == "keep":
                    keep_ids.append(rid)
                    fout.write(json.dumps(rec) + "\n")
                elif cls == "drop":
                    drop_ids.append(rid)
                    d = decisions.get(rid) or {}
                    fdrop.write(json.dumps({
                        "id": rid, "ground_truth": d.get("ground_truth"),
                        "verdict": d.get("verdict"), "best_count": d.get("best_count"),
                        "question": d.get("question"),
                        "reasoning": (d.get("verify") or d.get("blind") or {}).get("reasoning"),
                    }) + "\n")
                else:  # undecided -> excluded, listed (never written to filtered)
                    undecided.append(rid)

        with open(undec_path, "w") as fu:
            fu.write("\n".join(str(x) for x in undecided))
        meta = {
            "split": split, "source": src, "decisions_file": args.decisions,
            "partial": partial, "local_meta": args.local_meta,
            "n_total": len(rows), "n_kept": len(keep_ids),
            "n_dropped": len(drop_ids), "n_undecided": len(undecided),
        }
        json.dump({"meta": meta, "keep": keep_ids, "drop": drop_ids, "undecided": undecided},
                  open(ids_path, "w"), indent=2)
        print(json.dumps(meta, indent=2))
        print(f"wrote {out_path}")
        if undecided:
            print(f"wrote {undec_path}  ({len(undecided)} excluded)")

    if partial:
        print("[apply] NOTE: partial output — undecided records were EXCLUDED, not kept.")


if __name__ == "__main__":
    main()
