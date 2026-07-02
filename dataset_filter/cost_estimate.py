#!/usr/bin/env python
"""Estimate $ cost of filtering the whole dataset from a pilot run's MEASURED tokens.

Reads a decisions JSONL that has per-row `usage_total` (prompt/completion tokens),
computes average input/output tokens per example, and scales to the full split.

  python cost_estimate.py --decisions runs/train_points.jsonl --n-full 10289

Prices are list $ / 1M tokens (June 2026), input/output:
  gemini-3.1-pro-preview  2.00 / 12.00   (batch 50% off)
  gpt-5                   0.625 / 5.00
  gpt-5.4                 2.50 / 15.00
Token COUNTS are taken from whatever model produced the decisions file; applying a
different model's prices to them is an approximation (tokenizers differ a little).
"""
from __future__ import annotations

import argparse
import json

PRICES = {  # $ per 1M tokens: (input, output)  [list price, June 2026]
    "gemini-3.1-pro-preview": (2.00, 12.00),
    "gemini-3.5-flash": (1.50, 9.00),
    "gpt-5": (0.625, 5.00),
    "gpt-5.4": (2.50, 15.00),
    "gpt-5.5": (5.00, 30.00),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--n-full", type=int, default=10289, help="examples in the full split")
    ap.add_argument("--budget", type=float, default=250.0, help="$ / month budget")
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.decisions) if l.strip()]
    rows = [r for r in rows if (r.get("usage_total") or {}).get("total")]
    n = len(rows)
    pin = sum((r["usage_total"]).get("prompt", 0) for r in rows)
    pout = sum((r["usage_total"]).get("completion", 0) for r in rows)
    calls = sum((r["usage_total"]).get("calls", 0) for r in rows)
    in_pe, out_pe = pin / n, pout / n
    run_model = rows[0].get("model", "?")

    print(f"measured from {n} examples (model={run_model}):")
    print(f"  input  tok/example = {in_pe:,.0f}")
    print(f"  output tok/example = {out_pe:,.0f}")
    print(f"  calls/example      = {calls/n:.2f}")
    print(f"  scaling to full split N = {args.n_full:,}\n")

    full_in = in_pe * args.n_full / 1e6   # millions
    full_out = out_pe * args.n_full / 1e6
    print(f"full-split tokens: input={full_in:.1f}M  output={full_out:.1f}M\n")

    print(f"{'model':24} {'standard $':>11} {'batch(-50%) $':>14} {'fits $%d/mo?' % args.budget:>14}")
    for model, (pi, po) in PRICES.items():
        cost = full_in * pi + full_out * po
        fits = "yes" if cost <= args.budget else f"NO ({cost/args.budget:.1f}x)"
        note = " <- token counts native" if model == run_model else " (approx)"
        print(f"{model:24} {cost:>11.2f} {cost/2:>14.2f} {fits:>14}{note}")


if __name__ == "__main__":
    main()
