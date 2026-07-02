#!/usr/bin/env python3
"""Render a verl/Vero2 debug trajectory as a local HTML page."""

from __future__ import annotations

import argparse
import base64
import io
import json
import math
import mimetypes
import os
import random
from html import escape
from pathlib import Path
from typing import Any

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - optional visual fallback
    Image = None
    ImageDraw = None
    ImageFont = None


EVENT_ORDER = {
    "sample_start": 0,
    "assistant_turn": 1,
    "tool_call": 2,
    "tool_response": 3,
    "sample_final": 4,
}

SUCCESS_SCORE_THRESHOLD = 1.0

# --- Crop reconstruction constants -------------------------------------------------
# These mirror the cropzoom multi-index tool config used in training, see
# vero2_scripts/config/tool_config/image_zoom_in_tool_multiidx.yaml and
# verl/tools/image_zoom_in_tool.py (ImageZoomInTool). The cropzoom setup runs with
# coordinate_system=normalized_1000 and use_original_resolution_source=true, so every
# zoom (including zooms into a previously produced crop, img_idx>0) is composited back
# onto the ROOT original image's native pixels. Keep these in sync with the tool config,
# or override via env when visualizing runs that used different values.
SMART_RESIZE_FACTOR = 32
CROP_MIN_DIMENSION = 32
CROPPED_MIN_PIXELS = int(os.getenv("VERO2_VIZ_CROPPED_MIN_PIXELS", "518400"))
CROPPED_MAX_PIXELS = int(os.getenv("VERO2_VIZ_CROPPED_MAX_PIXELS", "1048576"))


def _round_by_factor(number: float, factor: int) -> int:
    return round(number / factor) * factor


def _ceil_by_factor(number: float, factor: int) -> int:
    return math.ceil(number / factor) * factor


def _floor_by_factor(number: float, factor: int) -> int:
    return math.floor(number / factor) * factor


def smart_resize(
    height: float,
    width: float,
    factor: int = SMART_RESIZE_FACTOR,
    min_pixels: int = CROPPED_MIN_PIXELS,
    max_pixels: int = CROPPED_MAX_PIXELS,
) -> tuple[int, int]:
    """Port of ImageZoomInTool.smart_resize so reconstructed crops match the model's view."""
    h_bar = max(factor, _round_by_factor(height, factor))
    w_bar = max(factor, _round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = _floor_by_factor(height / beta, factor)
        w_bar = _floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = _ceil_by_factor(height * beta, factor)
        w_bar = _ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def composite_bbox_to_native(region: list[float], bbox_2d: list[float]) -> list[float]:
    """Map a 0-1000 bbox drawn on an image representing ``region`` (native pixels of the root
    original) back into native pixels: native = region_origin + (bbox/1000) * region_span.
    For a full-image region [0,0,W,H] this reduces to (bbox/1000)*size. Mirrors
    ImageZoomInTool._composite_bbox_to_native."""
    rx1, ry1, rx2, ry2 = region
    rw, rh = rx2 - rx1, ry2 - ry1
    return [
        rx1 + float(bbox_2d[0]) / 1000.0 * rw,
        ry1 + float(bbox_2d[1]) / 1000.0 * rh,
        rx1 + float(bbox_2d[2]) / 1000.0 * rw,
        ry1 + float(bbox_2d[3]) / 1000.0 * rh,
    ]


def validate_and_clamp_bbox_px(bbox_2d: list[float], image_width: int, image_height: int) -> list[float] | None:
    """Port of ImageZoomInTool._validate_and_clamp_bbox; returns None instead of raising."""
    try:
        left, top, right, bottom = [float(v) for v in bbox_2d]
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(v) for v in (left, top, right, bottom)):
        return None
    if right <= left or bottom <= top:
        return None
    left = max(0.0, left)
    top = max(0.0, top)
    right = min(float(image_width), right)
    bottom = min(float(image_height), bottom)
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def maybe_resize_bbox_px(bbox_2d: list[float], image_width: int, image_height: int) -> list[float]:
    """Port of ImageZoomInTool._maybe_resize_bbox (expand tiny boxes to MIN_DIMENSION)."""
    left, top, right, bottom = bbox_2d
    left = max(0, left)
    top = max(0, top)
    right = min(image_width, right)
    bottom = min(image_height, bottom)

    height = bottom - top
    width = right - left
    if height < CROP_MIN_DIMENSION or width < CROP_MIN_DIMENSION:
        center_x = (left + right) / 2.0
        center_y = (top + bottom) / 2.0
        ratio = CROP_MIN_DIMENSION / min(height, width)
        new_half_height = math.ceil(height * ratio * 0.5)
        new_half_width = math.ceil(width * ratio * 0.5)
        new_left = math.floor(center_x - new_half_width)
        new_right = math.ceil(center_x + new_half_width)
        new_top = math.floor(center_y - new_half_height)
        new_bottom = math.ceil(center_y + new_half_height)

        new_left = max(0, new_left)
        new_top = max(0, new_top)
        new_right = min(image_width, new_right)
        new_bottom = min(image_height, new_bottom)

        if (new_bottom - new_top) > CROP_MIN_DIMENSION and (new_right - new_left) > CROP_MIN_DIMENSION:
            return [new_left, new_top, new_right, new_bottom]

    return [left, top, right, bottom]


class CropReconstructor:
    """Replays ImageZoomInTool's provenance crop logic so the visualizer can faithfully
    reconstruct what the model saw, including zooms into a previously produced crop
    (img_idx>0).

    For the cropzoom config (coordinate_system=normalized_1000,
    use_original_resolution_source=true, image_indexing=multi_index) the tool:
      * keeps a per-trajectory provenance map {img_idx: (root_orig_idx, region_in_native_px)},
      * composites the model's 0-1000 bbox through that region back to the ROOT original's
        native pixels, clamps + expands tiny boxes, crops from the original, then smart-resizes,
      * registers the resulting crop as the NEXT img_idx (image_data grows by one per zoom),
        so a later zoom into that crop resolves through the same provenance map.

    Events MUST be fed in chronological order so the appended-crop indices line up with the
    model's img_idx values (the forced-think grammar emits one tool call per turn)."""

    def __init__(self, source_paths: list[str]):
        self.source_paths = source_paths
        self.num_originals = len(source_paths)
        # Crops are appended to agent_data.image_data, so the next crop's img_idx is the
        # current image count, which starts at the number of original prompt images.
        self.next_index = self.num_originals
        self.provenance: dict[int, tuple[int, list[float]]] = {}
        self._orig_cache: dict[int, Any] = {}

    def _load_original(self, orig_idx: int):
        if orig_idx not in self._orig_cache:
            with Image.open(self.source_paths[orig_idx]) as image:
                self._orig_cache[orig_idx] = image.convert("RGB")
        return self._orig_cache[orig_idx]

    def process(self, img_idx: int, bbox_2d: list[float]) -> dict[str, Any] | None:
        if Image is None or not self.source_paths or bbox_2d is None:
            return None

        if img_idx in self.provenance:
            orig_idx, region = self.provenance[img_idx]
            region = list(region)
        elif 0 <= img_idx < self.num_originals:
            orig_idx = img_idx
            region = None  # full-image region filled in after the original is loaded
        else:
            # img_idx refers to a crop we never registered (out-of-order / earlier failure).
            return None

        try:
            image = self._load_original(orig_idx)
        except OSError:
            return None
        width, height = image.size

        if region is None:
            region = [0.0, 0.0, float(width), float(height)]
            self.provenance[img_idx] = (orig_idx, list(region))

        pixel_bbox = composite_bbox_to_native(region, bbox_2d)
        clamped = validate_and_clamp_bbox_px(pixel_bbox, width, height)
        if clamped is None:
            return None
        resized_bbox = maybe_resize_bbox_px(clamped, width, height)

        crop = image.crop(tuple(resized_bbox))
        left, top, right, bottom = resized_bbox
        new_width, new_height = smart_resize(right - left, bottom - top)
        try:
            crop = crop.resize((int(new_width), int(new_height)), resample=Image.BICUBIC)
        except (ValueError, OSError):
            pass

        # Register the crop at the index it will occupy in agent_data.image_data, so a later
        # zoom into it composites from the original again.
        new_img_idx = self.next_index
        self.provenance[new_img_idx] = (orig_idx, [float(c) for c in resized_bbox])
        self.next_index += 1

        return {
            "input_img_idx": img_idx,
            "orig_idx": orig_idx,
            "source_path": self.source_paths[orig_idx],
            "parent_region": [float(c) for c in region],
            "pixel_bbox": [float(c) for c in pixel_bbox],
            "native_region": [float(c) for c in resized_bbox],
            "crop": crop,
            "crop_size": list(crop.size),
            "new_img_idx": new_img_idx,
        }


def reconstruct_crops(events: list[dict[str, Any]], source_paths: list[str]) -> dict[int, dict[str, Any]]:
    """Replay every successful zoom in order, returning {id(event): crop record}."""
    recon_by_event: dict[int, dict[str, Any]] = {}
    if Image is None or not source_paths:
        return recon_by_event
    reconstructor = CropReconstructor(source_paths)
    for event in collect_zoom_events(events):
        info = event.get("tool_info") or {}
        # Failed crops are not appended to image_data, so they don't consume an index.
        if info.get("success") is False:
            continue
        bbox = bbox_from_event(event)
        record = reconstructor.process(img_idx_from_event(event), bbox)
        if record is None:
            continue
        record["bbox_2d"] = bbox
        record["label"] = label_from_event(event)
        recon_by_event[id(event)] = record
    return recon_by_event


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def first_scalar(value: Any) -> Any:
    if isinstance(value, list) and value:
        return first_scalar(value[0])
    return value


def flatten_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, str):
        paths.append(value)
    elif isinstance(value, list):
        for item in value:
            paths.extend(flatten_paths(item))
    return paths


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            chunks.append(str(item["text"]))
    return "\n".join(chunks)


def image_paths_from_content(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    paths = []
    for item in content:
        if not isinstance(item, dict):
            continue
        for key in ("image", "image_url"):
            if key in item and isinstance(item[key], str):
                paths.append(item[key])
    return paths


def original_image_paths(sample: dict[str, Any]) -> list[str]:
    paths = flatten_paths(sample.get("image_references"))
    paths.extend(flatten_paths((sample.get("extra_info") or {}).get("images")))
    for message in sample.get("raw_prompt") or sample.get("messages") or []:
        paths.extend(image_paths_from_content(message.get("content")))

    deduped = []
    seen = set()
    for path in paths:
        if path and path not in seen:
            seen.add(path)
            deduped.append(path)
    return deduped


def image_src(path: str, embed_images: bool) -> str:
    if path.startswith(("http://", "https://", "data:")):
        return path
    if not embed_images:
        return Path(path).as_posix()

    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError:
        return p.as_posix()
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def pil_image_src(image: Any, image_format: str = "PNG") -> str:
    buf = io.BytesIO()
    image.save(buf, format=image_format)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    mime = "image/png" if image_format.upper() == "PNG" else "image/jpeg"
    return f"data:{mime};base64,{encoded}"


def resized_preview(image: Any, max_side: int) -> Any:
    if max(image.width, image.height) <= max_side:
        return image
    preview = image.copy()
    preview.thumbnail((max_side, max_side))
    return preview


def parse_tool_arguments(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def bbox_from_event(event: dict[str, Any]) -> list[float] | None:
    info = event.get("tool_info") or {}
    args = parse_tool_arguments(event.get("tool_arguments"))
    bbox = info.get("bbox_2d") or args.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        return [float(v) for v in bbox]
    except (TypeError, ValueError):
        return None


def img_idx_from_event(event: dict[str, Any]) -> int:
    info = event.get("tool_info") or {}
    args = parse_tool_arguments(event.get("tool_arguments"))
    try:
        return int(info.get("img_idx", args.get("img_idx", 0)))
    except (TypeError, ValueError):
        return 0


def label_from_event(event: dict[str, Any]) -> str:
    info = event.get("tool_info") or {}
    args = parse_tool_arguments(event.get("tool_arguments"))
    return str(info.get("label") or args.get("label") or "")


def shorten(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + "\n\n...[truncated]...\n\n" + text[-keep:]


def find_sample(
    sample_path: Path,
    request_id: str | None,
    sample_index: int | None,
    rollout_n: int | None,
    select: str,
) -> dict[str, Any]:
    samples = [r for r in read_jsonl(sample_path) if r.get("event") == "sample_final"]
    if not samples:
        raise SystemExit(f"No sample_final records found in {sample_path}")

    if request_id:
        for sample in samples:
            if sample.get("request_id") == request_id:
                return sample
        raise SystemExit(f"request_id not found: {request_id}")

    if sample_index is not None:
        matches = [
            s
            for s in samples
            if s.get("index") == sample_index
            or s.get("trajectory", {}).get("sample_index") == sample_index
        ]
        if rollout_n is not None:
            matches = [s for s in matches if s.get("trajectory", {}).get("rollout_n") == rollout_n]
        if not matches:
            raise SystemExit(f"No sample found for sample_index={sample_index}, rollout_n={rollout_n}")
        return sorted(matches, key=sample_score, reverse=True)[0]

    if select == "first":
        return samples[0]
    if select == "last":
        return samples[-1]
    if select == "most-tools":
        return sorted(samples, key=lambda s: (len(s.get("tool_rewards") or []), sample_score(s)), reverse=True)[0]
    if select == "longest":
        return sorted(samples, key=sample_score, reverse=True)[0]
    if select == "random":
        return random.choice(samples)
    raise SystemExit(f"Unsupported --select value: {select}")


def read_reward_assignments(reward_path: Path) -> dict[str, dict[str, Any]]:
    if not reward_path.exists():
        return {}
    rewards: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(reward_path):
        request_id = row.get("request_id")
        if request_id:
            rewards[str(request_id)] = row
    return rewards


def reward_score(reward: dict[str, Any] | None) -> float | None:
    if not reward:
        return None
    value = reward.get("score", reward.get("final_reward"))
    if value is None:
        raw_result = reward.get("raw_result") or reward.get("reward_extra_info") or {}
        value = raw_result.get("score") if isinstance(raw_result, dict) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def reward_extra(reward: dict[str, Any] | None, key: str) -> Any:
    if not reward:
        return None
    if key in reward:
        return reward.get(key)
    for field in ("raw_result", "reward_extra_info"):
        values = reward.get(field)
        if isinstance(values, dict) and key in values:
            return values.get(key)
    return None


def reward_outcome(reward: dict[str, Any] | None) -> str | None:
    score = reward_score(reward)
    if score is None:
        return None
    return "successful" if score >= SUCCESS_SCORE_THRESHOLD else "incorrect"


def attach_reward_assignments(samples: list[dict[str, Any]], rewards: dict[str, dict[str, Any]]) -> None:
    for sample in samples:
        request_id = sample.get("request_id")
        if request_id and str(request_id) in rewards:
            sample["_reward_assignment"] = rewards[str(request_id)]


def filter_by_outcome(samples: list[dict[str, Any]], outcome: str) -> list[dict[str, Any]]:
    if outcome == "all":
        return samples
    return [
        sample
        for sample in samples
        if reward_outcome(sample.get("_reward_assignment")) == outcome
    ]


def find_samples(
    sample_path: Path,
    reward_path: Path | None,
    request_id: str | None,
    sample_index: int | None,
    rollout_n: int | None,
    select: str,
    num_samples: int,
    outcome: str,
) -> list[dict[str, Any]]:
    samples = [r for r in read_jsonl(sample_path) if r.get("event") == "sample_final"]
    if not samples:
        raise SystemExit(f"No sample_final records found in {sample_path}")
    total_samples = len(samples)

    rewards: dict[str, dict[str, Any]] = {}
    if reward_path is not None and (outcome != "all" or reward_path.exists()):
        rewards = read_reward_assignments(reward_path)
        attach_reward_assignments(samples, rewards)
    if outcome != "all":
        if reward_path is None or not reward_path.exists():
            raise SystemExit(f"--outcome={outcome} requires reward assignments, but {reward_path} is missing")
        samples = filter_by_outcome(samples, outcome)
        if not samples:
            raise SystemExit(
                f"No {outcome} samples found in {sample_path}; "
                f"checked {total_samples} samples and {len(rewards)} reward assignments"
            )

    if request_id:
        matches = [sample for sample in samples if sample.get("request_id") == request_id]
        if not matches:
            suffix = f" with outcome={outcome}" if outcome != "all" else ""
            raise SystemExit(f"request_id not found{suffix}: {request_id}")
        return matches[:1]

    if sample_index is not None:
        matches = [
            s
            for s in samples
            if s.get("index") == sample_index
            or s.get("trajectory", {}).get("sample_index") == sample_index
        ]
        if rollout_n is not None:
            matches = [s for s in matches if s.get("trajectory", {}).get("rollout_n") == rollout_n]
        if not matches:
            suffix = f", outcome={outcome}" if outcome != "all" else ""
            raise SystemExit(f"No sample found for sample_index={sample_index}, rollout_n={rollout_n}{suffix}")
        return sorted(matches, key=sample_score, reverse=True)[: max(1, num_samples)]

    if select == "first":
        return samples[: max(1, num_samples)]
    if select == "last":
        return samples[-max(1, num_samples) :]
    if select == "most-tools":
        return sorted(samples, key=lambda s: (len(s.get("tool_rewards") or []), sample_score(s)), reverse=True)[
            : max(1, num_samples)
        ]
    if select == "longest":
        return sorted(samples, key=sample_score, reverse=True)[: max(1, num_samples)]
    if select == "random":
        count = min(max(1, num_samples), len(samples))
        return random.sample(samples, count)
    raise SystemExit(f"Unsupported --select value: {select}")


def sample_score(sample: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(sample.get("num_turns") or 0),
        len(sample.get("tool_rewards") or []),
        len(sample.get("final_response") or ""),
    )


def events_for_request(agent_path: Path, request_id: str) -> list[dict[str, Any]]:
    return events_for_requests(agent_path, [request_id]).get(request_id, [])


def events_for_requests(agent_path: Path, request_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    events_by_request_id = {request_id: [] for request_id in request_ids}
    if not agent_path.exists() or not request_ids:
        return events_by_request_id

    wanted = set(request_ids)
    for seq, event in enumerate(read_jsonl(agent_path)):
        request_id = str(event.get("request_id"))
        if request_id in wanted:
            event["_seq"] = seq
            events_by_request_id[request_id].append(event)

    for request_id, events in events_by_request_id.items():
        events_by_request_id[request_id] = sorted(
            events,
            key=lambda e: (e.get("assistant_turn") or 0, EVENT_ORDER.get(e.get("event"), 99), e["_seq"]),
        )
    return events_by_request_id


def render_meta(sample: dict[str, Any], debug_dir: Path) -> str:
    extra = sample.get("extra_info") or {}
    traj = sample.get("trajectory") or {}
    reward = sample.get("_reward_assignment")
    rows = [
        ("request_id", sample.get("request_id")),
        ("sample index", sample.get("index") or traj.get("sample_index")),
        ("rollout_n", traj.get("rollout_n")),
        ("step", traj.get("step")),
        ("dataset id", extra.get("id")),
        ("question", extra.get("question")),
        ("ground_truth", extra.get("ground_truth") or (sample.get("reward_model") or {}).get("ground_truth")),
        ("answer", extra.get("answer")),
        ("num_turns", sample.get("num_turns")),
        ("tool_calls", len(sample.get("tool_rewards") or [])),
        ("reward_outcome", reward_outcome(reward)),
        ("score", reward_score(reward)),
        ("format", reward_extra(reward, "format")),
        ("accuracy", reward_extra(reward, "accuracy")),
        ("tool_call_bonus", reward_extra(reward, "tool_call_bonus")),
        ("tool_call_count", reward_extra(reward, "tool_call_count")),
        ("debug_dir", str(debug_dir)),
    ]
    body = "\n".join(
        f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>"
        for k, v in rows
        if v is not None
    )
    return f"<table>{body}</table>"


def render_images(paths: list[str], embed_images: bool) -> str:
    blocks = []
    seen = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        exists = Path(path).exists() if not path.startswith(("http://", "https://", "data:")) else True
        status = "" if exists else " missing"
        blocks.append(
            "<figure>"
            f'<a href="{escape(Path(path).as_posix())}">'
            f'<img src="{escape(image_src(path, embed_images))}" alt="{escape(path)}">'
            "</a>"
            f"<figcaption class=\"path{status}\">{escape(path)}</figcaption>"
            "</figure>"
        )
    return "<div class=\"images\">" + "\n".join(blocks) + "</div>" if blocks else ""


def render_image_blocks(items: list[dict[str, str]]) -> str:
    blocks = []
    for item in items:
        src = item.get("src")
        if not src:
            continue
        href = item.get("href") or src
        caption = item.get("caption") or href
        blocks.append(
            "<figure>"
            f'<a href="{escape(href)}">'
            f'<img src="{escape(src)}" alt="{escape(caption)}">'
            "</a>"
            f"<figcaption>{escape(caption)}</figcaption>"
            "</figure>"
        )
    return "<div class=\"images\">" + "\n".join(blocks) + "</div>" if blocks else ""


def render_message(role: str, text: str, image_paths: list[str], embed_images: bool, max_text_chars: int | None) -> str:
    text_html = f"<pre>{escape(shorten(text, max_text_chars))}</pre>" if text else ""
    images_html = render_images(image_paths, embed_images)
    return f'<section class="message {escape(role)}"><h3>{escape(role)}</h3>{text_html}{images_html}</section>'


def crop_from_record(record: dict[str, Any] | None) -> dict[str, str] | None:
    """Build a figure for a reconstructed crop produced by CropReconstructor."""
    if record is None or Image is None:
        return None
    crop = record.get("crop")
    if crop is None:
        return None
    native_region = [round(c, 1) for c in record.get("native_region", [])]
    caption = (
        f"reconstructed crop (multi-index) from img_idx={record.get('input_img_idx')} "
        f"-> root original img_idx={record.get('orig_idx')}, bbox_2d={record.get('bbox_2d')}, "
        f"native_region={native_region}, crop_size={record.get('crop_size')}, "
        f"label={record.get('label')}, new img_idx={record.get('new_img_idx')}"
    )
    return {
        "src": pil_image_src(resized_preview(crop, 1400)),
        "href": record.get("source_path", ""),
        "caption": caption,
    }


def render_tool_response(
    event: dict[str, Any],
    source_paths: list[str],
    recon_by_event: dict[int, dict[str, Any]],
    embed_images: bool,
    max_text_chars: int | None,
) -> str:
    info = event.get("tool_info") or {}
    paths = flatten_paths(info.get("saved_images"))
    crop_ref = info.get("crop_image_ref") or {}
    if isinstance(crop_ref, dict) and crop_ref.get("path"):
        paths.append(str(crop_ref["path"]))
    paths = [path for path in paths if path]

    dynamic_images = []
    if not paths or not any(Path(path).exists() for path in paths if not path.startswith(("http://", "https://", "data:"))):
        reconstructed = crop_from_record(recon_by_event.get(id(event)))
        if reconstructed is not None:
            dynamic_images.append(reconstructed)

    meta = {
        "tool": event.get("tool_name") or info.get("tool"),
        "assistant_turn": event.get("assistant_turn"),
        "label": info.get("label"),
        "bbox_2d": info.get("bbox_2d"),
        "parameters": info.get("parameters") or parse_tool_arguments(event.get("tool_arguments")),
        "pixel_bbox": info.get("pixel_bbox"),
        "source_image_size": info.get("source_image_size"),
        "cropped_image_size": info.get("cropped_image_size"),
        "crop_image_path": crop_ref.get("path") if isinstance(crop_ref, dict) else None,
        "success": info.get("success"),
    }
    meta_text = json.dumps({k: v for k, v in meta.items() if v is not None}, indent=2)
    text = (event.get("tool_response_text") or "") + "\n\n" + meta_text
    text_html = f"<pre>{escape(shorten(text, max_text_chars))}</pre>" if text else ""
    images_html = render_images(paths, embed_images) + render_image_blocks(dynamic_images)
    return f'<section class="message tool response"><h3>tool response</h3>{text_html}{images_html}</section>'


def collect_zoom_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    zooms = []
    seen = set()
    for event in events:
        if event.get("event") != "tool_response":
            continue
        bbox = bbox_from_event(event)
        if bbox is None:
            continue
        key = (event.get("assistant_turn"), tuple(bbox), img_idx_from_event(event), label_from_event(event))
        if key in seen:
            continue
        seen.add(key)
        zooms.append(event)
    return zooms


def render_zoom_overlays(
    events: list[dict[str, Any]],
    source_paths: list[str],
    recon_by_event: dict[int, dict[str, Any]],
) -> str:
    if Image is None or ImageDraw is None:
        return "<p>Pillow is not installed, so zoom box overlays could not be rendered.</p>"

    zooms = collect_zoom_events(events)
    if not zooms:
        return ""

    colors = [
        "#e11d48",
        "#2563eb",
        "#16a34a",
        "#f59e0b",
        "#7c3aed",
        "#0891b2",
        "#db2777",
        "#65a30d",
        "#dc2626",
        "#0f766e",
    ]
    # Group by the ROOT original image. A zoom into a crop (img_idx>0) is composited back to
    # native pixels of its root original, so all boxes for a trajectory land on the original.
    by_orig_idx: dict[int, list[tuple[int, dict[str, Any], dict[str, Any]]]] = {}
    order = 0
    for event in zooms:
        record = recon_by_event.get(id(event))
        if record is None:
            continue
        order += 1
        by_orig_idx.setdefault(record["orig_idx"], []).append((order, event, record))

    items = []
    for orig_idx, ordered in by_orig_idx.items():
        if orig_idx < 0 or orig_idx >= len(source_paths):
            continue
        source_path = source_paths[orig_idx]
        try:
            with Image.open(source_path) as image:
                image = image.convert("RGB")
                draw = ImageDraw.Draw(image)
                line_width = max(3, min(image.size) // 250)
                font = ImageFont.load_default() if ImageFont is not None else None
                legend_rows = []
                for order_num, event, record in ordered:
                    region = record["native_region"]
                    pixel_box = tuple(int(round(c)) for c in region)
                    color = colors[(order_num - 1) % len(colors)]
                    draw.rectangle(pixel_box, outline=color, width=line_width)
                    text_xy = (pixel_box[0] + line_width, max(0, pixel_box[1] - 14))
                    draw.text(text_xy, str(order_num), fill=color, font=font)
                    legend_rows.append(
                        f"{order_num}: turn={event.get('assistant_turn')}, "
                        f"input img_idx={record['input_img_idx']}, bbox_2d={record['bbox_2d']}, "
                        f"native_region={[round(c, 1) for c in region]}, "
                        f"label={record['label']}, new img_idx={record['new_img_idx']}"
                    )
        except OSError:
            continue

        caption = (
            f"root original img_idx={orig_idx}: all zoom regions composited onto the original image\n"
            + "\n".join(legend_rows)
        )
        items.append({"src": pil_image_src(resized_preview(image, 1800)), "href": source_path, "caption": caption})

    return render_image_blocks(items)


def build_html(
    sample: dict[str, Any],
    events: list[dict[str, Any]],
    debug_dir: Path,
    embed_images: bool,
    max_text_chars: int | None,
) -> str:
    request_id = sample.get("request_id", "")
    source_paths = original_image_paths(sample)
    recon_by_event = reconstruct_crops(events, source_paths)
    blocks = [render_meta(sample, debug_dir)]

    for message in sample.get("raw_prompt") or sample.get("messages") or []:
        role = message.get("role", "message")
        content = message.get("content")
        blocks.append(
            render_message(
                role,
                text_from_content(content),
                image_paths_from_content(content),
                embed_images,
                max_text_chars,
            )
        )

    overlay_html = render_zoom_overlays(events, source_paths, recon_by_event)
    if overlay_html:
        blocks.append("<h2>Zoom Boxes On Original Images</h2>")
        blocks.append(overlay_html)

    event_blocks = []
    for event in events:
        name = event.get("event")
        if name == "sample_start":
            continue
        if name == "assistant_turn":
            event_blocks.append(
                render_message(
                    f"assistant turn {event.get('assistant_turn')}",
                    event.get("text") or "",
                    [],
                    embed_images,
                    max_text_chars,
                )
            )
        elif name == "tool_call":
            text = json.dumps(event.get("tool_arguments") or event, indent=2, ensure_ascii=False)
            event_blocks.append(render_message("tool call", text, [], embed_images, max_text_chars))
        elif name == "tool_response":
            event_blocks.append(
                render_tool_response(event, source_paths, recon_by_event, embed_images, max_text_chars)
            )
        elif name == "sample_final":
            # sample_traces has a richer final record; render that below instead.
            continue

    if event_blocks:
        blocks.append("<h2>Interleaved Events</h2>")
        blocks.extend(event_blocks)

    blocks.append("<h2>Final Transcript</h2>")
    blocks.append(render_message("final_response", sample.get("final_response") or "", [], embed_images, max_text_chars))

    if source_paths:
        blocks.append("<h2>Referenced Original Images</h2>")
        blocks.append(render_images(source_paths, embed_images))

    css = """
    body { color: #1f2933; font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f8fa; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 24px; margin: 0 0 16px; }
    h2 { border-top: 1px solid #d0d7de; margin-top: 28px; padding-top: 20px; }
    h3 { margin: 0 0 10px; text-transform: uppercase; letter-spacing: .04em; font-size: 12px; color: #57606a; }
    table { border-collapse: collapse; width: 100%; background: white; margin: 12px 0 20px; }
    th, td { border: 1px solid #d0d7de; padding: 8px 10px; vertical-align: top; text-align: left; }
    th { width: 160px; background: #f0f3f6; }
    .message { background: white; border: 1px solid #d0d7de; border-radius: 8px; margin: 14px 0; padding: 14px; }
    .system { border-left: 5px solid #6e7781; }
    .user { border-left: 5px solid #0969da; }
    .assistant, .final_response { border-left: 5px solid #1a7f37; }
    .tool, .tool\\ response, .tool\\ call { border-left: 5px solid #bc4c00; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .images { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 12px; }
    figure { margin: 0; border: 1px solid #d0d7de; border-radius: 6px; overflow: hidden; background: #f6f8fa; }
    img { display: block; max-width: 100%; max-height: 620px; margin: 0 auto; object-fit: contain; background: #111; }
    figcaption { border-top: 1px solid #d0d7de; padding: 8px; color: #57606a; font: 12px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }
    figcaption { white-space: pre-wrap; }
    .missing { color: #cf222e; }
    """
    title = f"Trajectory {request_id}"
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{escape(title)}</title><style>{css}</style></head><body><main>"
        f"<h1>{escape(title)}</h1>"
        + "\n".join(blocks)
        + "</main></body></html>\n"
    )


def build_multi_sample_html(
    samples: list[dict[str, Any]],
    events_by_request_id: dict[str, list[dict[str, Any]]],
    debug_dir: Path,
    embed_images: bool,
    max_text_chars: int | None,
) -> str:
    if len(samples) == 1:
        request_id = str(samples[0].get("request_id"))
        return build_html(samples[0], events_by_request_id.get(request_id, []), debug_dir, embed_images, max_text_chars)

    sections = []
    nav_items = []
    for idx, sample in enumerate(samples, start=1):
        request_id = str(sample.get("request_id"))
        anchor = f"sample-{idx}"
        traj = sample.get("trajectory") or {}
        sample_index = sample.get("index") or traj.get("sample_index")
        rollout_n = traj.get("rollout_n")
        tool_count = len(sample.get("tool_rewards") or [])
        reward = sample.get("_reward_assignment")
        score = reward_score(reward)
        outcome = reward_outcome(reward)
        reward_label = ""
        if score is not None:
            reward_label = f", {outcome}, score {score:g}"
        nav_items.append(
            f'<li><a href="#{anchor}">Sample {idx}: '
            f'{escape(str(sample_index))} / rollout {escape(str(rollout_n))} '
            f'({tool_count} tools{escape(reward_label)})</a></li>'
        )
        sample_html = build_html(
            sample,
            events_by_request_id.get(request_id, []),
            debug_dir,
            embed_images,
            max_text_chars,
        )
        inner = sample_html.split("<main>", 1)[1].rsplit("</main>", 1)[0]
        sections.append(f'<section id="{anchor}" class="sample-section">{inner}</section>')

    css = """
    body { color: #1f2933; font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f8fa; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 24px; margin: 0 0 16px; }
    .sample-nav, .sample-section { background: white; border: 1px solid #d0d7de; border-radius: 8px; padding: 14px; margin: 14px 0 28px; }
    .sample-section { background: #f6f8fa; }
    .sample-section > h1 { border-top: 3px solid #0969da; padding-top: 18px; }
    h2 { border-top: 1px solid #d0d7de; margin-top: 28px; padding-top: 20px; }
    h3 { margin: 0 0 10px; text-transform: uppercase; letter-spacing: .04em; font-size: 12px; color: #57606a; }
    table { border-collapse: collapse; width: 100%; background: white; margin: 12px 0 20px; }
    th, td { border: 1px solid #d0d7de; padding: 8px 10px; vertical-align: top; text-align: left; }
    th { width: 160px; background: #f0f3f6; }
    .message { background: white; border: 1px solid #d0d7de; border-radius: 8px; margin: 14px 0; padding: 14px; }
    .system { border-left: 5px solid #6e7781; }
    .user { border-left: 5px solid #0969da; }
    .assistant, .final_response { border-left: 5px solid #1a7f37; }
    .tool, .tool\\ response, .tool\\ call { border-left: 5px solid #bc4c00; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .images { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 12px; }
    figure { margin: 0; border: 1px solid #d0d7de; border-radius: 6px; overflow: hidden; background: #f6f8fa; }
    img { display: block; max-width: 100%; max-height: 620px; margin: 0 auto; object-fit: contain; background: #111; }
    figcaption { border-top: 1px solid #d0d7de; padding: 8px; color: #57606a; font: 12px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }
    figcaption { white-space: pre-wrap; }
    .missing { color: #cf222e; }
    """
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>Trajectory Samples</title><style>{css}</style></head><body><main>"
        f"<h1>Trajectory Samples ({len(samples)})</h1>"
        f'<nav class="sample-nav"><ol>{"".join(nav_items)}</ol></nav>'
        + "\n".join(sections)
        + "</main></body></html>\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-log-dir", type=Path, required=True, help="Directory containing sample_traces.jsonl and agent_events.jsonl")
    parser.add_argument("--request-id", help="Exact rollout request_id to render")
    parser.add_argument("--sample-index", type=int, help="Dataset/sample index to render")
    parser.add_argument("--rollout-n", type=int, help="Disambiguate --sample-index by rollout number")
    parser.add_argument(
        "--select",
        choices=("longest", "most-tools", "first", "last", "random"),
        default="most-tools",
        help="Selection strategy when --request-id is omitted",
    )
    parser.add_argument("--output", type=Path, help="Output HTML path")
    parser.add_argument("--max-text-chars", type=int, default=0, help="Truncate long text blocks; 0 means no truncation")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to render when not selecting an exact request")
    parser.add_argument("--embed-images", action="store_true", help="Embed images as base64 data URIs; produces large HTML")
    parser.add_argument(
        "--outcome",
        choices=("all", "successful", "incorrect"),
        default="all",
        help=(
            "Filter by reward outcome using reward_assignments.jsonl. "
            "successful means score >= 1.0; incorrect means score < 1.0."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_dir = args.debug_log_dir.expanduser().resolve()
    sample_path = debug_dir / "sample_traces.jsonl"
    agent_path = debug_dir / "agent_events.jsonl"
    reward_path = debug_dir / "reward_assignments.jsonl"
    if not sample_path.exists():
        raise SystemExit(f"Missing {sample_path}")

    samples = find_samples(
        sample_path,
        reward_path,
        args.request_id,
        args.sample_index,
        args.rollout_n,
        args.select,
        args.num_samples,
        args.outcome,
    )
    request_ids = [str(sample.get("request_id")) for sample in samples]
    events_by_request_id = events_for_requests(agent_path, request_ids)

    output = args.output
    if output is None:
        sample = samples[0]
        request_id = str(sample.get("request_id"))
        output = debug_dir / f"trajectory_{request_id}.html"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    max_text_chars = args.max_text_chars if args.max_text_chars > 0 else None
    html = build_multi_sample_html(samples, events_by_request_id, debug_dir, args.embed_images, max_text_chars)
    output.write_text(html, encoding="utf-8")

    print(f"Wrote {output}")
    print(f"num_samples={len(samples)}")
    print(f"outcome={args.outcome}")
    for sample in samples:
        request_id = str(sample.get("request_id"))
        reward = sample.get("_reward_assignment")
        print(f"request_id={request_id}")
        print(f"sample_index={sample.get('index') or sample.get('trajectory', {}).get('sample_index')}")
        print(f"rollout_n={sample.get('trajectory', {}).get('rollout_n')}")
        print(f"tool_calls={len(sample.get('tool_rewards') or [])}")
        if reward:
            print(f"score={reward_score(reward)}")
            print(f"reward_outcome={reward_outcome(reward)}")
            print(f"format={reward_extra(reward, 'format')}")
            print(f"accuracy={reward_extra(reward, 'accuracy')}")


if __name__ == "__main__":
    main()
