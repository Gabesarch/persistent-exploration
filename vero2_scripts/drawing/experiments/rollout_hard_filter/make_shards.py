#!/usr/bin/env python3
"""Split the counting source jsonl into round-robin shards for the val-only difficulty runs.

Also emits a small GT-stratified smoke file (so a gpu-test smoke exercises the long
big-count draw chains, not just easy rows) and a manifest.json used by the aggregator
to assert full id coverage.

Usage:
  python make_shards.py \
    --source /scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync/filtered_correct_only/combined_counting_train.correct.partial.verl.jsonl \
    --out-dir /scratch/gpfs/ZHUANGL/lc1556/vero2/hard_filter_runs/shards \
    --num-shards 4 --smoke-size 64
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

GT_BUCKETS = (
    ("lt10", lambda v: v < 10),
    ("10to19", lambda v: 10 <= v < 20),
    ("20to49", lambda v: 20 <= v < 50),
    ("ge50", lambda v: v >= 50),
)


def bucket_of(gt_value: float) -> str:
    for name, pred in GT_BUCKETS:
        if pred(gt_value):
            return name
    raise AssertionError(f"unbucketed ground truth {gt_value!r}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--num-shards", type=int, default=4)
    parser.add_argument("--smoke-size", type=int, default=64)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    ids: set[str] = set()
    with open(args.source, encoding="utf-8") as f:
        for line_no, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            row_id = row["extra_info"]["id"]
            if row_id in ids:
                raise SystemExit(f"duplicate extra_info.id {row_id!r} at line {line_no}")
            ids.add(row_id)
            rows.append(row)

    shards: list[list[dict]] = [[] for _ in range(args.num_shards)]
    for i, row in enumerate(rows):
        shards[i % args.num_shards].append(row)

    manifest = {"source": str(args.source), "total_rows": len(rows), "files": {}}
    for shard_idx, shard_rows in enumerate(shards):
        name = f"shard_{shard_idx:02d}.verl.jsonl"
        path = out_dir / name
        with open(path, "w", encoding="utf-8") as f:
            for row in shard_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        manifest["files"][name] = {
            "rows": len(shard_rows),
            "ids": sorted(row["extra_info"]["id"] for row in shard_rows),
        }

    # Stratified smoke: equal share per GT-magnitude bucket, first-N within each bucket.
    per_bucket = max(1, args.smoke_size // len(GT_BUCKETS))
    smoke_rows: list[dict] = []
    for bucket_name, _ in GT_BUCKETS:
        picked = [
            row
            for row in rows
            if bucket_of(int(float(str(row["reward_model"]["ground_truth"]).strip()))) == bucket_name
        ][:per_bucket]
        if len(picked) < per_bucket:
            print(f"warning: bucket {bucket_name} has only {len(picked)} rows")
        smoke_rows.extend(picked)
    smoke_name = f"smoke_{len(smoke_rows)}.verl.jsonl"
    with open(out_dir / smoke_name, "w", encoding="utf-8") as f:
        for row in smoke_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest["files"][smoke_name] = {
        "rows": len(smoke_rows),
        "ids": sorted(row["extra_info"]["id"] for row in smoke_rows),
    }

    assert sum(len(s) for s in shards) == len(rows)
    with open(out_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"{len(rows)} rows -> {args.num_shards} shards + {smoke_name} in {out_dir}")
    for name, info in manifest["files"].items():
        print(f"  {name}: {info['rows']} rows")


if __name__ == "__main__":
    main()
