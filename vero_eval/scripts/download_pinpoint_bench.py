#!/usr/bin/env python3
"""Download and validate every image referenced by Pinpoint-Bench.

This intentionally downloads the files into the Hugging Face Hub cache used by
the evaluator.  It does not build a second copy of the dataset.

Typical use (from ``vero_eval``):

    source set_paths.sh
    python scripts/download_pinpoint_bench.py

The command exits non-zero unless every image referenced by ``val.json`` is
present in the cache and can be fully decoded by Pillow.  Run it on a node with
internet access before launching an evaluation with ``HF_HUB_OFFLINE=1``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ID = "godx7/Pinpoint-Bench"
METADATA_FILE = "val.json"
REQUIRED_FIELDS = {"images", "doc_id", "problem", "solution", "data_source"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download and decode-check all Pinpoint-Bench validation images."
    )
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument(
        "--revision",
        default="main",
        help="Hub branch, tag, or commit to download (default: main).",
    )
    parser.add_argument(
        "--cache-dir",
        default=None,
        help=(
            "Hugging Face Hub cache directory. By default, use HF_HUB_CACHE or "
            "$HF_HOME/hub, exactly like the evaluator."
        ),
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--retries", type=int, default=5)
    parser.add_argument(
        "--force",
        action="store_true",
        help="Redownload files even if Hugging Face considers them cached.",
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Do not access the network; only verify the existing cache.",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be at least 1")
    if args.retries < 1:
        parser.error("--retries must be at least 1")
    if args.force and args.verify_only:
        parser.error("--force and --verify-only cannot be used together")
    return args


@dataclass(frozen=True)
class ImageResult:
    filename: str
    path: str | None
    error: str | None


def normalize_hub_path(value: Any, row_index: int) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"row {row_index} contains an invalid image path: {value!r}")
    normalized = value.strip().replace("\\", "/")
    while normalized.startswith("./"):
        normalized = normalized[2:]
    path = PurePosixPath(normalized)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise ValueError(f"row {row_index} contains an unsafe image path: {value!r}")
    return path.as_posix()


def load_image_references(metadata_path: Path) -> tuple[list[dict[str, Any]], list[str]]:
    with metadata_path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{METADATA_FILE} must contain a non-empty JSON list")

    references: list[str] = []
    seen_doc_ids: set[str] = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"row {index} is not a JSON object")
        missing = sorted(REQUIRED_FIELDS.difference(row))
        if missing:
            raise ValueError(f"row {index} is missing fields: {', '.join(missing)}")
        images = row["images"]
        if not isinstance(images, list) or not images:
            raise ValueError(f"row {index} does not contain at least one image")
        doc_id = str(row["doc_id"])
        if doc_id in seen_doc_ids:
            raise ValueError(f"duplicate doc_id {doc_id!r} at row {index}")
        seen_doc_ids.add(doc_id)
        references.extend(normalize_hub_path(value, index) for value in images)

    # Preserve metadata order while avoiding redundant downloads.
    return rows, list(dict.fromkeys(references))


def validate_image(path: Path) -> None:
    from PIL import Image

    # verify() checks structure/checksums without decoding all pixels. Reopening
    # and load() additionally catches truncated pixel data.
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        image.load()
        if image.width <= 0 or image.height <= 0:
            raise ValueError(f"invalid dimensions {image.size}")


def download_image(
    *,
    hf_hub_download: Any,
    repo_id: str,
    revision: str,
    filename: str,
    cache_dir: str | None,
    retries: int,
    force: bool,
    verify_only: bool,
) -> ImageResult:
    last_error: Exception | None = None
    force_this_attempt = force
    attempts = 1 if verify_only else retries

    for attempt in range(1, attempts + 1):
        try:
            local_path = hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                repo_type="dataset",
                revision=revision,
                cache_dir=cache_dir,
                force_download=force_this_attempt,
                local_files_only=verify_only,
            )
            validate_image(Path(local_path))
            return ImageResult(filename, local_path, None)
        except Exception as exc:  # report any per-file decode/download failure
            last_error = exc
            # A cached file that fails to decode must be fetched afresh.
            force_this_attempt = True
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))

    error = f"{type(last_error).__name__}: {last_error}"
    return ImageResult(filename, None, error)


def main() -> int:
    args = parse_args()

    # set_paths.sh.example intentionally enables offline evaluation. Override it
    # before importing huggingface_hub when this command needs to populate cache.
    os.environ["HF_HUB_OFFLINE"] = "1" if args.verify_only else "0"
    if not args.verify_only:
        os.environ["HF_DATASETS_OFFLINE"] = "0"

    try:
        from huggingface_hub import constants, hf_hub_download
    except ImportError as exc:
        print(f"ERROR: missing dependency: {exc}", file=sys.stderr)
        print("Run this script in the same Python environment as vero_eval.", file=sys.stderr)
        return 2

    cache_dir = os.path.expanduser(os.path.expandvars(args.cache_dir)) if args.cache_dir else None
    effective_cache = cache_dir or constants.HF_HUB_CACHE
    mode = "Verifying" if args.verify_only else "Downloading"
    print(f"{mode} {args.repo_id}@{args.revision}")
    print(f"Hub cache: {effective_cache}")

    metadata_attempts = 1 if args.verify_only else args.retries
    metadata_path: str | None = None
    metadata_error: Exception | None = None
    force_metadata = args.force
    rows: list[dict[str, Any]] = []
    image_names: list[str] = []
    for attempt in range(1, metadata_attempts + 1):
        try:
            metadata_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=METADATA_FILE,
                repo_type="dataset",
                revision=args.revision,
                cache_dir=cache_dir,
                force_download=force_metadata,
                local_files_only=args.verify_only,
            )
            rows, image_names = load_image_references(Path(metadata_path))
            metadata_error = None
            break
        except Exception as exc:
            metadata_error = exc
            force_metadata = True
            if attempt < metadata_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))

    if metadata_error is not None or metadata_path is None:
        print(
            f"ERROR: could not download/parse {METADATA_FILE}: "
            f"{type(metadata_error).__name__}: {metadata_error}",
            file=sys.stderr,
        )
        return 1

    # hf_hub_download returns .../snapshots/<commit>/<filename>. Pin every image
    # to that same commit so metadata and samples cannot drift during this run.
    snapshot_dir = Path(metadata_path).parent
    commit = snapshot_dir.name
    if snapshot_dir.parent.name != "snapshots":
        print(
            f"ERROR: unexpected Hub cache path for metadata: {metadata_path}",
            file=sys.stderr,
        )
        return 1

    print(
        f"Metadata: {len(rows)} rows, {sum(len(row['images']) for row in rows)} "
        f"references, {len(image_names)} unique images"
    )
    print(f"Resolved commit: {commit}")

    failures: list[ImageResult] = []
    completed = 0
    worker_args = {
        "hf_hub_download": hf_hub_download,
        "repo_id": args.repo_id,
        "revision": commit,
        "cache_dir": cache_dir,
        "retries": args.retries,
        "force": args.force,
        "verify_only": args.verify_only,
    }
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(download_image, filename=filename, **worker_args): filename
            for filename in image_names
        }
        try:
            for future in as_completed(futures):
                result = future.result()
                completed += 1
                if result.error is not None:
                    failures.append(result)
                if completed % 25 == 0 or completed == len(image_names):
                    print(
                        f"Checked {completed}/{len(image_names)} images "
                        f"({len(failures)} failed)",
                        flush=True,
                    )
        except KeyboardInterrupt:
            for future in futures:
                future.cancel()
            print("\nInterrupted; cache remains safe to resume.", file=sys.stderr)
            return 130

    if failures:
        print(f"\nERROR: {len(failures)} images are unavailable or invalid:", file=sys.stderr)
        displayed_failures = sorted(failures, key=lambda item: item.filename)[:50]
        for result in displayed_failures:
            print(f"  {result.filename}: {result.error}", file=sys.stderr)
        if len(failures) > len(displayed_failures):
            print(
                f"  ... and {len(failures) - len(displayed_failures)} more failures",
                file=sys.stderr,
            )
        print("Rerun the same command to resume/retry failed files.", file=sys.stderr)
        return 1

    print(
        f"\nSUCCESS: all {len(image_names)} unique images for all {len(rows)} "
        "Pinpoint-Bench samples are cached and decodable."
    )
    print("You can now run the evaluation with HF_HUB_OFFLINE=1.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
