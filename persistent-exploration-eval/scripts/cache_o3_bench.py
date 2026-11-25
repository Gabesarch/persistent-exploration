"""Cache pinned O3-Bench files on scratch and fully decode every referenced image."""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hf-home", default=os.getenv("HF_HOME", str(Path.home() / ".cache/huggingface")))
    parser.add_argument("--verify-only", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    os.environ.update(HF_HOME=args.hf_home, HF_HUB_CACHE=str(Path(args.hf_home) / "hub"),
                      HF_HUB_OFFLINE="1" if args.verify_only else "0")
    from huggingface_hub import hf_hub_download
    from PIL import Image, ImageFile
    from lmms_eval.tasks.o3_bench.utils import REPO_ID, REVISION, image_filename, load_rows
    ImageFile.LOAD_TRUNCATED_IMAGES = False

    def fetch(filename):
        return Path(hf_hub_download(REPO_ID, filename, repo_type="dataset", revision=REVISION,
                                   local_files_only=args.verify_only))

    metadata = fetch("test/metadata.jsonl")
    rows = load_rows(metadata)
    filenames = sorted({image_filename(row) for row in rows})
    print(f"O3-Bench {REVISION}: {len(rows)} questions, {len(filenames)} images", flush=True)

    def verify(filename):
        path = fetch(filename)
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            image.load()
            size = list(image.size)
        return {"filename": filename, "bytes": path.stat().st_size, "size": size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        images = list(pool.map(verify, filenames))
    by_name = {image["filename"]: image for image in images}
    mismatches = [i for i, row in enumerate(rows) if tuple(by_name[image_filename(row)]["size"]) !=
                  (row["img_width"], row["img_height"])]
    if mismatches:
        raise ValueError(f"Image dimensions disagree with annotations for rows {mismatches}")
    report = {"repo_id": REPO_ID, "revision": REVISION, "metadata": str(metadata),
              "metadata_sha256": hashlib.sha256(metadata.read_bytes()).hexdigest(),
              "questions": len(rows), "subsets": dict(Counter(row["subset"] for row in rows)),
              "images": images, "total_image_bytes": sum(image["bytes"] for image in images),
              "all_images_decoded": True}
    manifest = Path(args.hf_home) / "o3_bench_cache_manifest.json"
    # A read-only preflight verifies the manifest too, without modifying scratch.
    if args.verify_only:
        if json.loads(manifest.read_text()) != report:
            raise ValueError("O3-Bench cache manifest differs from verified files")
    else:
        manifest.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items() if key != "images"}, indent=2))
    print(f"Verified manifest: {manifest}")


if __name__ == "__main__":
    main()
