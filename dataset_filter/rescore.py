#!/usr/bin/env python
"""Re-apply the keep/verdict DECISION logic to an existing decisions JSONL.

The expensive part (Gemini calls) is already stored per example (blind count,
verify verdict + count). This re-derives `verdict` / `keep` / `override` from
those stored signals under new tolerances WITHOUT any new API calls — so you can
tune --abs-tol / --rel-tol for free.

  python rescore.py --in runs/train_smart_v2.jsonl --out runs/train_smart_v3.jsonl \
      --abs-tol 2 --rel-tol 0.15
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from filter_counting import combine_verify, num_ok  # decision logic, single source of truth


def rescore_row(r, abs_tol, rel_tol):
    gt = r.get("ground_truth")
    strat = r.get("strategy")
    if r.get("short_circuit") is True:  # blind exactly matched GT
        r["verdict"], r["keep"], r["override"] = "CORRECT", True, None
        r["best_count"] = (r.get("blind") or {}).get("count")
    elif r.get("verify"):
        v = r["verify"]
        r["best_count"] = v.get("your_count")
        r["verdict"], r["keep"], r["override"] = combine_verify(
            v.get("verdict"), v.get("your_count"), gt, abs_tol, rel_tol)
    elif strat == "blind" and r.get("blind"):
        b = r["blind"]
        lo, hi, c = b.get("count_low"), b.get("count_high"), b.get("count")
        in_band = (lo is not None and hi is not None and gt is not None and lo <= gt <= hi)
        exact = (c == gt)
        r["verdict"] = "CORRECT" if exact else ("PLAUSIBLE" if in_band else "WRONG")
        r["keep"] = bool(exact or in_band)
        r["best_count"] = c
    # rows with errors / no signals are left untouched
    r["abs_tol"], r["rel_tol"] = abs_tol, rel_tol
    return r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--abs-tol", type=int, default=2)
    ap.add_argument("--rel-tol", type=float, default=0.15)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    out = [rescore_row(r, args.abs_tol, args.rel_tol) for r in rows]
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    n = len(out)
    keep = sum(1 for r in out if r.get("keep") is True)
    print(f"rescored {n} -> {args.out}")
    print(f"keep={keep} ({keep/max(1,n):.1%})  verdicts={dict(Counter(r.get('verdict') for r in out))}")
    print(f"overrides={dict(Counter(r.get('override') for r in out if r.get('override')))}")


if __name__ == "__main__":
    main()
