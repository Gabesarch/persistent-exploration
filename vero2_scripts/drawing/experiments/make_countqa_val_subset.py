#!/usr/bin/env python3
"""Build a small CountQA validation subset in verl.jsonl format for counting training runs.

Flattens Jayant-Sravan/CountQA (1,001 images, each with a list of QA pairs) into one row
per question, keeps integer-answer questions only, stratifies by count magnitude, and
writes images + a verl.jsonl that matches the training data schema (reward_type counting;
no GT boxes, so point_box_applicable is 0 on these rows — the eligibility gate handles it).

Usage:
  HF_HUB_OFFLINE=1 python make_countqa_val_subset.py \
    --out-dir /scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/countqa_val --size 256
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

GT_BUCKETS = (
    ("1to5", lambda v: 1 <= v <= 5),
    ("6to10", lambda v: 6 <= v <= 10),
    ("11to19", lambda v: 11 <= v < 20),
    ("20to49", lambda v: 20 <= v < 50),
    ("ge50", lambda v: v >= 50),
)


def bucket_of(value: int) -> str | None:
    for name, pred in GT_BUCKETS:
        if pred(value):
            return name
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from datasets import load_dataset

    out_dir = Path(args.out_dir)
    images_dir = out_dir / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset("Jayant-Sravan/CountQA")["test"]

    # Flatten: one candidate per (image, question) with an integer answer.
    candidates: dict[str, list[tuple[int, int, str, int]]] = defaultdict(list)
    for img_idx in range(len(ds)):
        row = ds[img_idx]
        for q_idx, (question, answer) in enumerate(zip(row["questions"], row["answers"])):
            try:
                value = int(str(answer).strip())
            except (TypeError, ValueError):
                continue
            if value < 1:
                continue
            bucket = bucket_of(value)
            if bucket is None:
                continue
            candidates[bucket].append((img_idx, q_idx, str(question).strip(), value))

    rng = random.Random(args.seed)
    per_bucket = max(1, args.size // len(GT_BUCKETS))
    picked: list[tuple[int, int, str, int]] = []
    leftovers: list[tuple[int, int, str, int]] = []
    for bucket, _ in GT_BUCKETS:
        pool = candidates.get(bucket, [])
        rng.shuffle(pool)
        picked.extend(pool[:per_bucket])
        leftovers.extend(pool[per_bucket:])
        print(f"bucket {bucket}: {len(pool)} candidates, taking {min(per_bucket, len(pool))}")
    # Top up to the target size from leftovers if some bucket ran short.
    if len(picked) < args.size:
        rng.shuffle(leftovers)
        picked.extend(leftovers[: args.size - len(picked)])
    picked = picked[: args.size]

    needed_images = sorted({img_idx for img_idx, *_ in picked})
    image_paths: dict[int, str] = {}
    for img_idx in needed_images:
        path = images_dir / f"countqa_{img_idx:04d}.png"
        if not path.exists():
            ds[img_idx]["image"].convert("RGB").save(path)
        image_paths[img_idx] = str(path)

    out_path = out_dir / f"countqa_val_{len(picked)}.verl.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for index, (img_idx, q_idx, question, value) in enumerate(sorted(picked)):
            row = {
                "data_source": "countqa",
                "prompt": [{"role": "user", "content": f"<image>\n{question}"}],
                "images": [image_paths[img_idx]],
                "ability": "counting",
                "reward_model": {"style": "rule", "ground_truth": str(value)},
                "extra_info": {
                    "split": "val",
                    "index": index,
                    "answer": str(value),
                    "question": f"<image>\n{question}",
                    "tolerance": None,
                    "id": f"countqa_{img_idx:04d}_{q_idx}",
                    "reward_type": "counting",
                    "source_dataset": "countqa",
                    "metadata": {},
                },
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dist = defaultdict(int)
    for _, _, _, value in picked:
        dist[bucket_of(value)] += 1
    print(f"wrote {out_path} ({len(picked)} rows, {len(needed_images)} images)")
    print("bucket distribution:", dict(dist))


if __name__ == "__main__":
    main()
