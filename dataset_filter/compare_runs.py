#!/usr/bin/env python
"""Compare two decision JSONLs over their shared ids (e.g. prompt v1 vs v2).

  python compare_runs.py A.jsonl B.jsonl
"""
import json
import sys
from collections import Counter


def load(p):
    d = {}
    for line in open(p):
        line = line.strip()
        if line:
            r = json.loads(line)
            d[r["id"]] = r
    return d


def main():
    a, b = load(sys.argv[1]), load(sys.argv[2])
    ids = sorted(set(a) & set(b))
    print(f"A={sys.argv[1]}  ({len(a)})")
    print(f"B={sys.argv[2]}  ({len(b)})")
    print(f"shared ids: {len(ids)}")
    ka = sum(1 for i in ids if a[i].get("keep"))
    kb = sum(1 for i in ids if b[i].get("keep"))
    print(f"keep:  A={ka} ({ka/len(ids):.1%})   B={kb} ({kb/len(ids):.1%})")
    print(f"A verdicts: {dict(Counter(a[i]['verdict'] for i in ids))}")
    print(f"B verdicts: {dict(Counter(b[i]['verdict'] for i in ids))}")
    flipped = [(i, a[i]['verdict'], b[i]['verdict']) for i in ids
               if bool(a[i].get('keep')) != bool(b[i].get('keep'))]
    print(f"\nkeep-decision flips: {len(flipped)}")
    a2b_keep = [f for f in flipped if not a[f[0]].get('keep')]  # drop->keep
    a2b_drop = [f for f in flipped if a[f[0]].get('keep')]      # keep->drop
    print(f"  drop->keep (recovered): {len(a2b_keep)}")
    print(f"  keep->drop (newly cut): {len(a2b_drop)}")
    ov = Counter(b[i].get('override') for i in ids if b[i].get('override'))
    print(f"B overrides: {dict(ov)}")

    def show(label, fl):
        print(f"\n--- {label} ---")
        for i, va, vb in fl:
            r = b[i]
            mc = r.get('best_count')
            print(f"  GT={r['ground_truth']!s:>4} model={mc!s:>4}  {va:>12}->{vb:<12} "
                  f"{(r.get('source_dataset') or '')[:14]:14} {r['question'][:46]}")
    show("DROP->KEEP (were these wrongly dropped before?)", a2b_keep)
    show("KEEP->DROP (newly cut by v2)", a2b_drop)


if __name__ == "__main__":
    main()
