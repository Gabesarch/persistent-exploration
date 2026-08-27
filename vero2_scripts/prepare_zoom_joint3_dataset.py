#!/usr/bin/env python3
"""Build and validate the three-source zoom/retry/critic training mixture."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
from collections import Counter
from pathlib import Path
from typing import Any, Iterator

from PIL import Image

DEFAULT_VISUAL_PROBE = Path(
    "/scratch/gpfs/ZHUANGL/gs1693/vero2_datasets/train_rl_datasets/"
    "chart_search_counting_visual_probe_train_rollout_score_leq05_all.verl.jsonl"
)
DEFAULT_HSTAR = Path(
    "/scratch/gpfs/ZHUANGL/gs1693/vero2/datasets/hstar/data/"
    "hstarbench_mixed.verl.jsonl"
)
DEFAULT_SCREENSPOT_LIKE = Path(
    "/scratch/gpfs/ZHUANGL/gs1693/vero2/datasets/screenspot_pro_like_v1/data/"
    "screenspot_pro_like_train.verl.jsonl"
)
DEFAULT_OUTPUT = Path(
    "/scratch/gpfs/ZHUANGL/gs1693/vero2/datasets/"
    "zoom_answer_retry_critic_joint3/data/"
    "zoom_answer_retry_critic_joint3_train.verl.jsonl"
)

POINT_SUFFIX = (
    "Use coordinates relative to the original full image: x and y must each be "
    "normalized to the 0-1000 range. In the <answer> block, output only this JSON "
    'array: [{"point_2d": [x, y]}].'
)
_OLD_POINT_SUFFIX = re.compile(
    r"\s*Output\s+the\s+point\s+in\s+a\s+JSON\s+array\s+format:\s*"
    r"\[\s*\{\s*\"point_2d\"\s*:\s*\[\s*x\s*,\s*y\s*\]\s*\}\s*\]\.?\s*$",
    flags=re.IGNORECASE,
)


def _iter_rows(path: Path) -> Iterator[tuple[int, dict[str, Any]]]:
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_number}: row must be an object")
            yield line_number, row


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _validate_common(row: dict[str, Any], location: str) -> None:
    prompt = row.get("prompt")
    images = row.get("images")
    reward_model = row.get("reward_model")
    extra_info = row.get("extra_info")
    _require(isinstance(prompt, list) and len(prompt) == 1, f"{location}: prompt must have one message")
    _require(isinstance(prompt[0], dict), f"{location}: prompt message must be an object")
    _require(prompt[0].get("role") == "user", f"{location}: prompt role must be user")
    _require(
        isinstance(prompt[0].get("content"), str) and prompt[0]["content"].strip(),
        f"{location}: prompt content must be a non-empty string",
    )
    _require(isinstance(images, list) and len(images) == 1, f"{location}: exactly one image is required")
    _require(isinstance(images[0], str) and images[0], f"{location}: image path must be a string")
    _require(Path(images[0]).is_absolute(), f"{location}: image path must be absolute")
    _require(row.get("ability") == "search", f"{location}: ability must remain 'search'")
    _require(isinstance(row.get("data_source"), str), f"{location}: data_source must be a string")
    _require(isinstance(reward_model, dict), f"{location}: reward_model must be an object")
    _require(reward_model.get("style") == "rule", f"{location}: reward_model.style must be 'rule'")
    _require(
        isinstance(reward_model.get("ground_truth"), str) and reward_model["ground_truth"].strip(),
        f"{location}: reward_model.ground_truth must be a non-empty string",
    )
    _require(isinstance(extra_info, dict), f"{location}: extra_info must be an object")
    _require(
        isinstance(extra_info.get("id"), str) and extra_info["id"].strip(),
        f"{location}: extra_info.id must be a non-empty string",
    )


def _read_image_size(image_path: str, location: str) -> tuple[int, int]:
    path = Path(image_path)
    _require(path.is_file(), f"{location}: image does not exist: {path}")
    try:
        with Image.open(path) as image:
            width, height = image.size
    except Exception as exc:
        raise ValueError(f"{location}: cannot read image header {path}: {exc}") from exc
    _require(width > 0 and height > 0, f"{location}: image has invalid dimensions {width}x{height}")
    return int(width), int(height)


def _metadata_image_size(row: dict[str, Any], location: str) -> tuple[int, int]:
    image_sizes = row["extra_info"].get("image_sizes")
    _require(
        isinstance(image_sizes, list) and len(image_sizes) == 1,
        f"{location}: pointing row must have one image_sizes entry",
    )
    size = image_sizes[0]
    _require(isinstance(size, list) and len(size) == 2, f"{location}: image size must be [width,height]")
    width, height = size
    _require(
        isinstance(width, int) and not isinstance(width, bool) and width > 0,
        f"{location}: invalid image width {width!r}",
    )
    _require(
        isinstance(height, int) and not isinstance(height, bool) and height > 0,
        f"{location}: invalid image height {height!r}",
    )
    return width, height


def _validate_point_truth(row: dict[str, Any], location: str, width: int, height: int) -> None:
    truth = row["reward_model"]["ground_truth"]
    try:
        items = json.loads(truth)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{location}: pointing ground truth is not JSON: {exc}") from exc
    _require(isinstance(items, list) and items, f"{location}: pointing truth must be a non-empty array")
    for item_index, item in enumerate(items):
        _require(isinstance(item, dict), f"{location}: truth item {item_index} must be an object")
        bbox = item.get("bbox_2d")
        _require(
            isinstance(bbox, list) and len(bbox) == 4,
            f"{location}: truth item {item_index} must have four bbox_2d coordinates",
        )
        _require(
            all(
                isinstance(value, int | float)
                and not isinstance(value, bool)
                and math.isfinite(value)
                for value in bbox
            ),
            f"{location}: truth item {item_index} has non-finite/non-numeric coordinates",
        )
        x1, y1, x2, y2 = (float(value) for value in bbox)
        _require(0.0 <= x1 < x2 <= width, f"{location}: x bounds {bbox} exceed width {width}")
        _require(0.0 <= y1 < y2 <= height, f"{location}: y bounds {bbox} exceed height {height}")


def _canonical_point_prompt(prompt: str) -> str:
    base = _OLD_POINT_SUFFIX.sub("", prompt).rstrip()
    if not base:
        raise ValueError("pointing prompt became empty while replacing its output instruction")
    return f"{base} {POINT_SUFFIX}"


def _normalize_row(
    raw_row: dict[str, Any],
    *,
    source_name: str,
    expected_reward_type: str,
    global_index: int,
    location: str,
    verify_image_headers: bool,
) -> dict[str, Any]:
    row = copy.deepcopy(raw_row)
    _validate_common(row, location)
    extra_info = row["extra_info"]
    _require(
        extra_info.get("reward_type") == expected_reward_type,
        f"{location}: expected reward_type={expected_reward_type!r}, got {extra_info.get('reward_type')!r}",
    )

    row["mixture_source"] = source_name
    extra_info["mixture_source"] = source_name
    extra_info["source_index"] = extra_info.get("index")
    extra_info["index"] = global_index

    if expected_reward_type == "clicking":
        width, height = _metadata_image_size(row, location)
        if verify_image_headers:
            actual_size = _read_image_size(row["images"][0], location)
            _require(
                actual_size == (width, height),
                f"{location}: metadata size {(width, height)} != image size {actual_size}",
            )
        prompt = _canonical_point_prompt(row["prompt"][0]["content"])
        row["prompt"][0]["content"] = prompt
        extra_info["question"] = prompt
        extra_info["answer"] = row["reward_model"]["ground_truth"]
        extra_info["normalize_bbox_to_1000"] = True
        extra_info["coordinate_format"] = "point_2d_normalized_1000"
        _validate_point_truth(row, location, width, height)
    elif verify_image_headers:
        width, height = _read_image_size(row["images"][0], location)
        existing_sizes = extra_info.get("image_sizes")
        if existing_sizes is not None:
            _require(
                existing_sizes == [[width, height]],
                f"{location}: metadata image_sizes {existing_sizes!r} != image size {(width, height)}",
            )
        else:
            extra_info["image_sizes"] = [[width, height]]

    return row


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.partial.{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(args: argparse.Namespace) -> dict[str, Any]:
    sources = (
        ("visual_probe", args.visual_probe.resolve(), "search"),
        ("hstarbench", args.hstar.resolve(), "clicking"),
        ("screenspot_pro_like", args.screenspot_like.resolve(), "clicking"),
    )
    output = args.output.resolve()
    _require(str(output).startswith("/scratch/"), "production output must be below /scratch")
    _require(all(path.is_file() for _, path, _ in sources), "one or more source JSONL files do not exist")
    _require(output not in {path for _, path, _ in sources}, "output must not overwrite a source dataset")
    output.parent.mkdir(parents=True, exist_ok=True)

    temporary = output.with_name(f".{output.name}.partial.{os.getpid()}")
    counts: Counter[str] = Counter()
    reward_counts: Counter[str] = Counter()
    seen_ids: set[str] = set()
    total = 0
    try:
        with temporary.open("w", encoding="utf-8") as destination:
            source_iterators = [
                (source_name, source_path, expected_reward_type, iter(_iter_rows(source_path)))
                for source_name, source_path, expected_reward_type in sources
            ]
            while True:
                wrote_row = False
                for source_name, source_path, expected_reward_type, rows in source_iterators:
                    try:
                        line_number, raw_row = next(rows)
                    except StopIteration:
                        continue
                    wrote_row = True
                    location = f"{source_path}:{line_number}"
                    row = _normalize_row(
                        raw_row,
                        source_name=source_name,
                        expected_reward_type=expected_reward_type,
                        global_index=total,
                        location=location,
                        verify_image_headers=not args.skip_image_header_check,
                    )
                    stable_id = row["extra_info"]["id"]
                    _require(stable_id not in seen_ids, f"{location}: duplicate global extra_info.id {stable_id!r}")
                    seen_ids.add(stable_id)
                    destination.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
                    counts[source_name] += 1
                    reward_counts[expected_reward_type] += 1
                    total += 1
                if not wrote_row:
                    break
            destination.flush()
            os.fsync(destination.fileno())
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)

    manifest = {
        "format_version": 2,
        "output": str(output),
        "output_sha256": _sha256(output),
        "row_count": total,
        "counts_by_mixture_source": dict(counts),
        "counts_by_reward_type": dict(reward_counts),
        "mixture_sampling_field": "mixture_source",
        "row_order": "round_robin_sources_for_arrow_schema_inference",
        "point_coordinate_format": "point_2d_normalized_1000",
        "point_ground_truth_format": "absolute_pixel_bbox_2d_json_string",
        "image_headers_verified": not args.skip_image_header_check,
        "sources": [
            {
                "mixture_source": source_name,
                "path": str(source_path),
                "sha256": _sha256(source_path),
                "expected_reward_type": reward_type,
            }
            for source_name, source_path, reward_type in sources
        ],
    }
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    _atomic_json(manifest_path, manifest)
    manifest["manifest"] = str(manifest_path)
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--visual-probe", type=Path, default=DEFAULT_VISUAL_PROBE)
    parser.add_argument("--hstar", type=Path, default=DEFAULT_HSTAR)
    parser.add_argument("--screenspot-like", type=Path, default=DEFAULT_SCREENSPOT_LIKE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--skip-image-header-check",
        action="store_true",
        help="Skip existence/dimension checks against the actual image headers.",
    )
    args = parser.parse_args()
    manifest = build(args)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
