#!/usr/bin/env python
"""Make the counting dataset self-contained under our own folder.

The verl metadata stores ABSOLUTE image paths that point at the original owner's
directory (e.g. .../gs1693/.../images/...). We keep our own copy of the images in
`<data-dir>/images/`, so this script:

  1. verifies EVERY referenced image already exists in our local images dir
     (fail closed — nonzero exit, no output written — if any is missing), and
  2. writes `combined_counting_<split>.local.verl.jsonl` next to the originals with
     each `images[...]` entry rebased onto `<data-dir>/images/...`.

After this, the filter/apply/batch tools read the `.local.` metadata with `--local-meta`
and have ZERO dependency on the original owner's folder.

  python localize_paths.py --data-dir /scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

MARKER = "/images/"


def rebase(path: str, images_dir: str) -> str:
    """Rebase a stored image path onto our local images dir, preserving any subpath."""
    i = path.rfind(MARKER)
    suffix = path[i + len(MARKER):] if i != -1 else os.path.basename(path)
    return os.path.join(images_dir, suffix)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir",
                    default="/scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--images-subdir", default="images")
    args = ap.parse_args()

    images_dir = os.path.join(args.data_dir, args.images_subdir)
    if not os.path.isdir(images_dir):
        sys.exit(f"[localize] FATAL: local images dir not found: {images_dir}")

    # First pass over all splits: verify coverage BEFORE writing anything.
    plans = []  # (split, src_path, out_path, [rewritten records])
    total_missing = 0
    for split in args.splits:
        src = os.path.join(args.data_dir, f"combined_counting_{split}.verl.jsonl")
        if not os.path.exists(src):
            sys.exit(f"[localize] FATAL: metadata not found: {src}")
        out = os.path.join(args.data_dir, f"combined_counting_{split}.local.verl.jsonl")
        recs, missing = [], []
        with open(src) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                imgs = rec.get("images") or []
                new_imgs = []
                for p in imgs:
                    lp = rebase(p, images_dir)
                    if not os.path.exists(lp):
                        missing.append((rec.get("extra_info", {}).get("id"), lp))
                    new_imgs.append(lp)
                rec["images"] = new_imgs
                recs.append(rec)
        print(f"[localize] {split}: {len(recs)} records, {len(missing)} missing locally")
        for eid, lp in missing[:10]:
            print(f"    MISSING {eid}: {lp}")
        total_missing += len(missing)
        plans.append((split, src, out, recs))

    if total_missing:
        sys.exit(f"[localize] FATAL: {total_missing} image(s) not present locally — "
                 f"NOT writing localized metadata (fail closed). Fix the image copy first.")

    # All covered -> write localized metadata.
    for split, src, out, recs in plans:
        with open(out, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec) + "\n")
        print(f"[localize] wrote {out}  ({len(recs)} records, all images local)")


if __name__ == "__main__":
    main()
