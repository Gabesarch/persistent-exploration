#!/usr/bin/env python3
"""Preprocess Gaze RL JSONL image paths with Qwen vision resizing."""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
import queue
import re
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image
from tqdm import tqdm

Image.MAX_IMAGE_PIXELS = None


DEFAULT_DATA_DIR = Path("/scratch/gpfs/ZHUANGL/gs1693/vigorl_gaze_datasets/chart_search_counting_rl")
DEFAULT_TRAIN_JSONL = DEFAULT_DATA_DIR / "chart_search_counting_train.verl.jsonl"
DEFAULT_VAL_JSONL = DEFAULT_DATA_DIR / "chart_search_counting_val.verl.jsonl"
DEFAULT_MAX_PIXELS = 16_777_216
DEFAULT_IMAGE_PATCH_SIZE = 14
DEFAULT_NUM_WORKERS = 8
DEFAULT_CHUNK_SIZE_MULTIPLIER = 128


@dataclass(frozen=True)
class SplitSpec:
    name: str
    source_path: Path
    output_path: Path


@dataclass(frozen=True)
class ImageTask:
    source_path: str
    output_path: str
    max_pixels: int
    image_patch_size: int
    overwrite: bool
    png_compress_level: int
    fast_resume: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Resize/preprocess every unique image referenced by chart_search_counting VERL JSONL files, "
            "then write new JSONL files pointing to the processed images."
        )
    )
    parser.add_argument("--train-jsonl", type=Path, default=DEFAULT_TRAIN_JSONL)
    parser.add_argument("--val-jsonl", type=Path, default=DEFAULT_VAL_JSONL)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. Defaults to a sibling of the train JSONL named preprocessed_max_pixels_${max_pixels}.",
    )
    parser.add_argument("--max-pixels", type=int, default=DEFAULT_MAX_PIXELS)
    parser.add_argument("--image-patch-size", type=int, default=DEFAULT_IMAGE_PATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=1,
        help="PNG compression level for processed images. 1 is faster, 6 is PIL's default, 0 is least compressed.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=None,
        help=(
            "Number of image tasks submitted to one process pool before recycling it. "
            "Defaults to max(num_workers * 128, 128)."
        ),
    )
    parser.add_argument("--overwrite", action="store_true", help="Reprocess images even if the output PNG exists.")
    parser.add_argument(
        "--fast-resume",
        action="store_true",
        help="When a processed image already exists, reuse it without opening source/output images to record sizes.",
    )
    parser.add_argument(
        "--skip-missing",
        action="store_true",
        help="Do not abort on missing/failed images; leave those image paths unchanged in rewritten JSONL.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only read and rewrite the first N rows from each split. Intended for smoke tests.",
    )
    return parser.parse_args()


def count_limited_lines(path: Path, limit: int | None) -> int:
    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for count, _ in enumerate(handle, start=1):
            if limit is not None and count >= limit:
                break
    return count


def output_jsonl_path(output_dir: Path, source_path: Path) -> Path:
    name = source_path.name
    if name.endswith(".jsonl"):
        name = name[: -len(".jsonl")]
    return output_dir / f"{name}.preprocessed.jsonl"


def processed_image_path(images_dir: Path, source_path: str) -> Path:
    digest = hashlib.sha1(os.path.abspath(source_path).encode("utf-8")).hexdigest()
    stem = Path(source_path).stem or "image"
    safe_stem = re.sub(r"[^A-Za-z0-9._-]+", "_", stem)[:160]
    return images_dir / f"{digest}_{safe_stem}.png"


def read_image_size(path: str | Path) -> tuple[int, int] | None:
    try:
        with Image.open(path) as image:
            return image.size
    except Exception:
        return None


def validate_image(path: str | Path) -> tuple[int, int] | None:
    """Return the image size only if PIL can fully decode the file."""
    try:
        with Image.open(path) as image:
            image.load()
            return image.size
    except Exception:
        return None


def load_json_line(line: str, path: Path, line_number: int) -> dict[str, Any]:
    try:
        row = json.loads(line)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in {path}:{line_number}: {exc}") from exc
    if not isinstance(row, dict):
        raise ValueError(f"Expected object row in {path}:{line_number}, got {type(row).__name__}")
    return row


def iter_limited_rows(path: Path, limit: int | None):
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if limit is not None and line_number > limit:
                break
            if not line.strip():
                continue
            yield line_number, load_json_line(line, path, line_number)


def scan_inputs(
    split_specs: list[SplitSpec], images_dir: Path, limit: int | None
) -> tuple[dict[str, Path], dict[str, dict[str, int]], dict[str, int]]:
    image_outputs: dict[str, Path] = {}
    image_split_counts: dict[str, dict[str, int]] = {}
    row_counts: dict[str, int] = {}
    total_rows = sum(count_limited_lines(spec.source_path, limit) for spec in split_specs)

    with tqdm(total=total_rows, desc="Scanning JSONL", unit="row") as progress:
        for spec in split_specs:
            row_counts[spec.name] = 0
            for _, row in iter_limited_rows(spec.source_path, limit):
                row_counts[spec.name] += 1
                images = row.get("images") or []
                if not isinstance(images, list):
                    raise ValueError(f"Expected row 'images' to be a list in {spec.source_path}")
                for image_path in images:
                    if not isinstance(image_path, str):
                        raise TypeError(
                            f"Only string image paths are supported; got {type(image_path).__name__} in {spec.source_path}"
                        )
                    image_outputs.setdefault(image_path, processed_image_path(images_dir, image_path))
                    split_counts = image_split_counts.setdefault(image_path, {})
                    split_counts[spec.name] = split_counts.get(spec.name, 0) + 1
                progress.update(1)

    return image_outputs, image_split_counts, row_counts


def process_one_image(task: ImageTask) -> dict[str, Any]:
    source_path = Path(task.source_path)
    output_path = Path(task.output_path)

    if not source_path.exists():
        return {
            "source_path": task.source_path,
            "processed_path": task.output_path,
            "status": "missing",
            "original_size": None,
            "processed_size": None,
            "error": f"source image does not exist: {task.source_path}",
        }

    if output_path.exists() and not task.overwrite:
        processed_size = validate_image(output_path)
        if processed_size is not None:
            return {
                "source_path": task.source_path,
                "processed_path": task.output_path,
                "status": "reused",
                "original_size": None if task.fast_resume else read_image_size(source_path),
                "processed_size": processed_size,
                "error": None,
            }

    original_size = read_image_size(source_path)

    try:
        from qwen_vl_utils import process_vision_info

        output_path.parent.mkdir(parents=True, exist_ok=True)
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": task.source_path,
                        "max_pixels": task.max_pixels,
                    }
                ],
            }
        ]
        vision_result = process_vision_info(
            messages,
            return_video_metadata=True,
            image_patch_size=task.image_patch_size,
        )
        images = vision_result[0] if isinstance(vision_result, tuple) else None
        if not images:
            raise RuntimeError("process_vision_info did not return an image")

        processed_image = images[0]
        if not isinstance(processed_image, Image.Image):
            raise TypeError(f"Expected PIL.Image from process_vision_info, got {type(processed_image).__name__}")
        if processed_image.mode != "RGB":
            processed_image = processed_image.convert("RGB")
        fd, tmp_name = tempfile.mkstemp(prefix=f"{output_path.stem}.", suffix=".tmp", dir=output_path.parent)
        os.close(fd)
        tmp_path = Path(tmp_name)
        try:
            processed_image.save(tmp_path, format="PNG", compress_level=task.png_compress_level)
            os.replace(tmp_path, output_path)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

        return {
            "source_path": task.source_path,
            "processed_path": task.output_path,
            "status": "processed",
            "original_size": original_size,
            "processed_size": processed_image.size,
            "error": None,
        }
    except Exception as exc:
        return {
            "source_path": task.source_path,
            "processed_path": task.output_path,
            "status": "error",
            "original_size": original_size,
            "processed_size": read_image_size(output_path) if output_path.exists() else None,
            "error": repr(exc),
        }


def image_error_result(task: ImageTask, status: str, error: str) -> dict[str, Any]:
    return {
        "source_path": task.source_path,
        "processed_path": task.output_path,
        "status": status,
        "original_size": read_image_size(task.source_path),
        "processed_size": read_image_size(task.output_path) if Path(task.output_path).exists() else None,
        "error": error,
    }


def isolated_process_entry(task: ImageTask, result_queue) -> None:
    try:
        result_queue.put(process_one_image(task))
    except BaseException as exc:
        result_queue.put(image_error_result(task, "error", repr(exc)))


def process_one_image_isolated(task: ImageTask) -> dict[str, Any]:
    """Run one task in a throwaway child process so worker death cannot kill the main run."""
    context = mp.get_context("spawn")
    result_queue = context.Queue(maxsize=1)
    process = context.Process(target=isolated_process_entry, args=(task, result_queue))
    process.start()
    process.join()

    try:
        if process.exitcode == 0:
            return result_queue.get_nowait()
        return image_error_result(
            task,
            "worker_exit",
            f"isolated image worker exited with code {process.exitcode}",
        )
    except queue.Empty:
        return image_error_result(task, "worker_exit", "isolated image worker exited without returning a result")
    finally:
        result_queue.close()
        result_queue.join_thread()


def process_task_chunk(
    chunk: list[ImageTask],
    num_workers: int,
    image_outputs: dict[str, Path],
    progress: tqdm,
) -> dict[str, dict[str, Any]]:
    chunk_results: dict[str, dict[str, Any]] = {}
    executor = ProcessPoolExecutor(max_workers=num_workers)
    future_to_task = {executor.submit(process_one_image, task): task for task in chunk}
    broken_pool_error: BrokenProcessPool | None = None

    try:
        for future in as_completed(future_to_task):
            task = future_to_task[future]
            try:
                result = future.result()
            except BrokenProcessPool as exc:
                broken_pool_error = exc
                break
            except Exception as exc:
                result = {
                    "source_path": task.source_path,
                    "processed_path": str(image_outputs[task.source_path]),
                    "status": "error",
                    "original_size": None,
                    "processed_size": None,
                    "error": repr(exc),
                }
            chunk_results[task.source_path] = result
            progress.update(1)
    finally:
        executor.shutdown(wait=broken_pool_error is None, cancel_futures=broken_pool_error is not None)

    if broken_pool_error is None:
        return chunk_results

    remaining_tasks = [task for task in chunk if task.source_path not in chunk_results]
    tqdm.write(
        "Process pool broke while processing images; retrying "
        f"{len(remaining_tasks)} in-flight task(s) in isolated subprocesses. Error: {broken_pool_error}"
    )
    for task in remaining_tasks:
        chunk_results[task.source_path] = process_one_image_isolated(task)
        progress.update(1)

    return chunk_results


def process_images(
    image_outputs: dict[str, Path],
    max_pixels: int,
    image_patch_size: int,
    num_workers: int,
    overwrite: bool,
    chunk_size: int | None,
    png_compress_level: int,
    fast_resume: bool,
) -> dict[str, dict[str, Any]]:
    tasks = [
        ImageTask(
            source_path=source_path,
            output_path=str(output_path),
            max_pixels=max_pixels,
            image_patch_size=image_patch_size,
            overwrite=overwrite,
            png_compress_level=png_compress_level,
            fast_resume=fast_resume,
        )
        for source_path, output_path in sorted(image_outputs.items())
    ]
    results: dict[str, dict[str, Any]] = {}

    if not tasks:
        return results

    if num_workers <= 1:
        iterator = (process_one_image(task) for task in tasks)
        for result in tqdm(iterator, total=len(tasks), desc="Processing images", unit="image"):
            results[result["source_path"]] = result
        return results

    chunk_size = chunk_size or max(num_workers * DEFAULT_CHUNK_SIZE_MULTIPLIER, DEFAULT_CHUNK_SIZE_MULTIPLIER)
    chunk_size = max(chunk_size, num_workers)
    with tqdm(total=len(tasks), desc="Processing images", unit="image") as progress:
        for start in range(0, len(tasks), chunk_size):
            chunk = tasks[start : start + chunk_size]
            results.update(process_task_chunk(chunk, num_workers, image_outputs, progress))

    return results


def rewrite_jsonl(
    split_specs: list[SplitSpec],
    image_results: dict[str, dict[str, Any]],
    skip_missing: bool,
    limit: int | None,
) -> dict[str, int]:
    total_rows = sum(count_limited_lines(spec.source_path, limit) for spec in split_specs)
    rewritten_counts: dict[str, int] = {}

    with tqdm(total=total_rows, desc="Rewriting JSONL", unit="row") as progress:
        for spec in split_specs:
            written = 0
            with spec.output_path.open("w", encoding="utf-8") as output:
                for _, row in iter_limited_rows(spec.source_path, limit):
                    rewritten_images = []
                    for image_path in row.get("images") or []:
                        result = image_results.get(image_path)
                        if result is None:
                            if skip_missing:
                                rewritten_images.append(image_path)
                                continue
                            raise RuntimeError(f"No processing result found for image: {image_path}")
                        if result["status"] in {"processed", "reused"}:
                            rewritten_images.append(result["processed_path"])
                        elif skip_missing:
                            rewritten_images.append(image_path)
                        else:
                            raise RuntimeError(f"Image preprocessing failed for {image_path}: {result.get('error')}")
                    row["images"] = rewritten_images
                    output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    written += 1
                    progress.update(1)
            rewritten_counts[spec.name] = written

    return rewritten_counts


def write_manifest(
    manifest_path: Path,
    image_results: dict[str, dict[str, Any]],
    image_split_counts: dict[str, dict[str, int]],
    max_pixels: int,
    image_patch_size: int,
) -> None:
    with manifest_path.open("w", encoding="utf-8") as output:
        for source_path in sorted(image_results):
            result = image_results[source_path]
            row = {
                "source_path": source_path,
                "processed_path": result.get("processed_path"),
                "status": result.get("status"),
                "original_size": result.get("original_size"),
                "processed_size": result.get("processed_size"),
                "max_pixels": max_pixels,
                "image_patch_size": image_patch_size,
                "splits": sorted(image_split_counts.get(source_path, {})),
                "row_counts": image_split_counts.get(source_path, {}),
                "error": result.get("error"),
            }
            output.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def summarize(
    row_counts: dict[str, int],
    rewritten_counts: dict[str, int],
    image_results: dict[str, dict[str, Any]],
    split_specs: list[SplitSpec],
    images_dir: Path,
    manifest_path: Path,
) -> None:
    status_counts: dict[str, int] = {}
    for result in image_results.values():
        status = str(result.get("status"))
        status_counts[status] = status_counts.get(status, 0) + 1

    print("", file=sys.stderr)
    print("Preprocessing complete.", file=sys.stderr)
    print(f"Rows read: {row_counts}", file=sys.stderr)
    print(f"Rows written: {rewritten_counts}", file=sys.stderr)
    print(f"Unique images: {len(image_results)}", file=sys.stderr)
    print(f"Image status counts: {status_counts}", file=sys.stderr)
    for spec in split_specs:
        print(f"{spec.name} JSONL: {spec.output_path}", file=sys.stderr)
    print(f"Image directory: {images_dir}", file=sys.stderr)
    print(f"Manifest: {manifest_path}", file=sys.stderr)


def main() -> int:
    args = parse_args()
    if args.max_pixels <= 0:
        raise ValueError(f"--max-pixels must be positive, got {args.max_pixels}")
    if args.image_patch_size <= 0:
        raise ValueError(f"--image-patch-size must be positive, got {args.image_patch_size}")
    if args.num_workers <= 0:
        raise ValueError(f"--num-workers must be positive, got {args.num_workers}")
    if args.png_compress_level < 0 or args.png_compress_level > 9:
        raise ValueError(f"--png-compress-level must be in [0, 9], got {args.png_compress_level}")
    if args.chunk_size is not None and args.chunk_size <= 0:
        raise ValueError(f"--chunk-size must be positive when set, got {args.chunk_size}")
    if args.limit is not None and args.limit <= 0:
        raise ValueError(f"--limit must be positive when set, got {args.limit}")

    output_dir = args.output_dir or (args.train_jsonl.parent / f"preprocessed_max_pixels_{args.max_pixels}")
    output_dir = output_dir.resolve()
    images_dir = output_dir / "images"
    manifest_path = output_dir / "manifest.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)

    split_specs = [
        SplitSpec("train", args.train_jsonl.resolve(), output_jsonl_path(output_dir, args.train_jsonl)),
        SplitSpec("val", args.val_jsonl.resolve(), output_jsonl_path(output_dir, args.val_jsonl)),
    ]
    for spec in split_specs:
        if not spec.source_path.exists():
            raise FileNotFoundError(f"{spec.name} JSONL does not exist: {spec.source_path}")

    image_outputs, image_split_counts, row_counts = scan_inputs(split_specs, images_dir, args.limit)
    image_results = process_images(
        image_outputs=image_outputs,
        max_pixels=args.max_pixels,
        image_patch_size=args.image_patch_size,
        num_workers=args.num_workers,
        overwrite=args.overwrite,
        chunk_size=args.chunk_size,
        png_compress_level=args.png_compress_level,
        fast_resume=args.fast_resume,
    )

    failed = [result for result in image_results.values() if result["status"] not in {"processed", "reused"}]
    if failed and not args.skip_missing:
        write_manifest(manifest_path, image_results, image_split_counts, args.max_pixels, args.image_patch_size)
        sample_errors = "\n".join(
            f"- {result['source_path']}: {result.get('error')}" for result in failed[:10]
        )
        raise RuntimeError(
            f"Image preprocessing failed for {len(failed)} image(s). "
            f"Manifest with failures was written to {manifest_path}.\n{sample_errors}"
        )

    rewritten_counts = rewrite_jsonl(split_specs, image_results, args.skip_missing, args.limit)
    write_manifest(manifest_path, image_results, image_split_counts, args.max_pixels, args.image_patch_size)
    summarize(row_counts, rewritten_counts, image_results, split_specs, images_dir, manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
