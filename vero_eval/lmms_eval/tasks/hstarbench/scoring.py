"""HStarBench scoring for env actions, panorama bboxes, and point clicks."""

from __future__ import annotations

import json
import math
import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

from panorama_strip import bbox_center_to_yaw_pitch, target_cone_to_strip_bboxes

DEFAULT_IOU_SUCCESS_THRESHOLD = float(os.environ.get("HSTARBENCH_IOU_THRESHOLD", "0.5"))

BBOX_PATTERN = re.compile(
    r"bbox_2d\s*=\s*\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]",
    re.IGNORECASE,
)
SUBMIT_BBOX_PATTERN = re.compile(
    r"submit_bbox\s*\(\s*bbox_2d\s*=\s*\[([^\]]+)\]\s*\)",
    re.IGNORECASE,
)
ANSWER_BLOCK_PATTERN = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
BBOX_LIST_PATTERN = re.compile(
    r"\[\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\]"
)


def compute_bbox_iou(
    box_a: Sequence[float],
    box_b: Sequence[float],
) -> float:
    """Standard IoU for xyxy boxes on the 0–1000 panorama strip."""
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]
    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)
    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def _as_bbox_tuple(box: Sequence[float]) -> Tuple[float, float, float, float]:
    return (float(box[0]), float(box[1]), float(box[2]), float(box[3]))


def get_doc_ground_bboxes(doc: Dict[str, Any]) -> List[Tuple[float, float, float, float]]:
    """Ground-truth strip bboxes derived from annotation yaw/pitch cone (+ strip yaw offset)."""
    target_yaw = doc.get("target_yaw") or [0, 360]
    target_pitch = doc.get("target_pitch") or [-90, 90]
    return target_cone_to_strip_bboxes(target_yaw, target_pitch)


def _extract_last_boxed_content(text: str) -> Optional[str]:
    r"""Return the payload of the last balanced ``\boxed{...}`` expression."""
    cursor = 0
    last_content: Optional[str] = None
    while True:
        start = text.find(r"\boxed", cursor)
        if start < 0:
            return last_content
        brace = start + len(r"\boxed")
        while brace < len(text) and text[brace].isspace():
            brace += 1
        if brace >= len(text) or text[brace] != "{":
            cursor = brace + 1
            continue

        depth = 1
        content_start = brace + 1
        cursor = content_start
        while cursor < len(text) and depth:
            if text[cursor] == "{":
                depth += 1
            elif text[cursor] == "}":
                depth -= 1
            cursor += 1
        if depth:
            return last_content
        candidate = text[content_start : cursor - 1].strip()
        if candidate:
            last_content = candidate


def parse_final_answer_point(text: str) -> Optional[Tuple[float, float]]:
    r"""Parse the strict joint3 clicking contract from the last answer block.

    The accepted payload is one JSON-array item whose only key is ``point_2d``.
    As in the training reward, either the raw answer or the last balanced
    ``\boxed{...}`` payload may contain that JSON.
    """
    answer_matches = list(ANSWER_BLOCK_PATTERN.finditer(text or ""))
    if not answer_matches:
        return None
    answer_text = answer_matches[-1].group(1).strip()
    candidate = _extract_last_boxed_content(answer_text) or answer_text
    try:
        prediction = json.loads(candidate)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(prediction, list) or len(prediction) != 1:
        return None
    item = prediction[0]
    if not isinstance(item, dict) or set(item) != {"point_2d"}:
        return None
    point = item["point_2d"]
    if not isinstance(point, list) or len(point) != 2:
        return None
    try:
        x, y = (float(value) for value in point)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
        return None
    return x, y


def score_final_answer_point(doc: Dict[str, Any], response_text: str) -> Dict[str, Any]:
    """Score joint3-style HStar output as point-in-any-target-box accuracy."""
    ground_bboxes = get_doc_ground_bboxes(doc)
    answer_point = parse_final_answer_point(response_text)
    correct = bool(
        answer_point is not None
        and any(
            float(x1) <= answer_point[0] <= float(x2)
            and float(y1) <= answer_point[1] <= float(y2)
            for x1, y1, x2, y2 in ground_bboxes
        )
    )
    return {
        "success": correct,
        "point_in_box_success": correct,
        "has_answer_point": answer_point is not None,
        "answer_point": list(answer_point) if answer_point is not None else None,
        "ground_bboxes": [list(box) for box in ground_bboxes],
        "score_mode": "point_in_any_target_box",
    }


def bbox_iou_vs_ground(
    bbox: Tuple[float, float, float, float],
    ground_bboxes: Sequence[Tuple[float, float, float, float]],
) -> float:
    if not ground_bboxes:
        return 0.0
    return max(compute_bbox_iou(bbox, gt) for gt in ground_bboxes)


def normalize_yaw(yaw: float) -> float:
    return float(yaw) % 360.0


def yaw_in_target_range(yaw: float, target_yaw: Sequence[float]) -> bool:
    y = normalize_yaw(yaw)
    t0, t1 = float(target_yaw[0]), float(target_yaw[1])
    if t0 > t1:
        return t0 <= y <= 360.0 or 0.0 <= y <= t1
    return t0 <= y <= t1


def pitch_in_target_range(pitch: float, target_pitch: Sequence[float]) -> bool:
    p = float(pitch)
    return float(target_pitch[0]) <= p <= float(target_pitch[1])


def angles_in_target_cone(
    yaw: float,
    pitch: float,
    target_yaw: Sequence[float],
    target_pitch: Sequence[float],
) -> bool:
    return yaw_in_target_range(yaw, target_yaw) and pitch_in_target_range(pitch, target_pitch)


def parse_bboxes_from_text(text: str) -> List[Tuple[float, float, float, float]]:
    out: List[Tuple[float, float, float, float]] = []
    for m in BBOX_PATTERN.finditer(text or ""):
        try:
            box = tuple(float(m.group(i)) for i in range(1, 5))
            out.append(box)
        except ValueError:
            continue
    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", text or "", re.DOTALL):
        try:
            data = json.loads(m.group(1))
            if data.get("name") == "image_zoom_in_tool":
                bbox = data.get("arguments", {}).get("bbox_2d")
                if isinstance(bbox, list) and len(bbox) == 4:
                    out.append(tuple(float(x) for x in bbox))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return out


def _parse_bbox_list_match(match: re.Match) -> Optional[Tuple[float, float, float, float]]:
    try:
        return tuple(float(match.group(i)) for i in range(1, 5))
    except ValueError:
        return None


def parse_final_answer_bbox(text: str) -> Optional[Tuple[float, float, float, float]]:
    """Extract the target bbox from the model's final <answer> block on the panorama strip."""
    if not text:
        return None

    answer_matches = list(ANSWER_BLOCK_PATTERN.finditer(text))
    if not answer_matches:
        return None

    # Use the last <answer> block — earlier matches may come from prompt examples.
    answer_text = answer_matches[-1].group(1)
    submit_bbox = parse_submit_bbox(answer_text)
    if submit_bbox is not None:
        return submit_bbox

    for pattern in (BBOX_PATTERN, BBOX_LIST_PATTERN):
        matches = list(pattern.finditer(answer_text))
        if matches:
            bbox = _parse_bbox_list_match(matches[-1])
            if bbox is not None:
                return bbox

    for m in re.finditer(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", answer_text, re.DOTALL):
        try:
            data = json.loads(m.group(1))
            if data.get("name") != "image_zoom_in_tool":
                continue
            bbox = data.get("arguments", {}).get("bbox_2d")
            if isinstance(bbox, list) and len(bbox) == 4:
                return tuple(float(x) for x in bbox)
        except (json.JSONDecodeError, TypeError, ValueError):
            continue
    return None


def score_final_answer_bbox(
    doc: Dict[str, Any],
    response_text: str,
    *,
    iou_threshold: float = DEFAULT_IOU_SUCCESS_THRESHOLD,
) -> Dict[str, Any]:
    """Score zoom-only workflow: IoU vs ground strip bbox AND center-in-target-cone."""
    ground_bboxes = get_doc_ground_bboxes(doc)
    target_yaw = doc.get("target_yaw") or [0, 360]
    target_pitch = doc.get("target_pitch") or [-90, 90]
    answer_bbox = parse_final_answer_bbox(response_text)
    if answer_bbox is None:
        return {
            "success": False,
            "answer_bbox_success": False,
            "cone_success": False,
            "iou_success": False,
            "has_answer_bbox": False,
            "answer_bbox": None,
            "answer_yaw": None,
            "answer_pitch": None,
            "bbox_iou": 0.0,
            "ground_bboxes": [list(box) for box in ground_bboxes],
            "score_mode": "answer_bbox_dual",
        }

    iou = bbox_iou_vs_ground(answer_bbox, ground_bboxes)
    cone_ok, yaw, pitch = bbox_aligns_target(answer_bbox, target_yaw, target_pitch)
    iou_ok = iou >= iou_threshold
    return {
        "success": cone_ok,
        "answer_bbox_success": iou_ok,
        "cone_success": cone_ok,
        "iou_success": iou_ok,
        "has_answer_bbox": True,
        "answer_bbox": list(answer_bbox),
        "answer_yaw": yaw,
        "answer_pitch": pitch,
        "bbox_iou": iou,
        "ground_bboxes": [list(box) for box in ground_bboxes],
        "score_mode": "answer_bbox_dual",
    }


def parse_submit_bbox(text: str) -> Optional[Tuple[float, float, float, float]]:
    m = SUBMIT_BBOX_PATTERN.search(text or "")
    if not m:
        return None
    try:
        parts = [float(x.strip()) for x in m.group(1).split(",")]
        if len(parts) == 4:
            return (parts[0], parts[1], parts[2], parts[3])
    except ValueError:
        return None
    return None


def bbox_aligns_target(
    bbox: Tuple[float, float, float, float],
    target_yaw: Sequence[float],
    target_pitch: Sequence[float],
) -> Tuple[bool, float, float]:
    """Return (aligned, yaw, pitch) for bbox center on panorama strip (0–1000 coords)."""
    yaw, pitch = bbox_center_to_yaw_pitch(bbox)
    ok = angles_in_target_cone(yaw, pitch, target_yaw, target_pitch)
    return ok, yaw, pitch


def best_zoom_align_from_trace(
    doc: Dict[str, Any],
    trace: Optional[List[str]],
    *,
    iou_threshold: float = DEFAULT_IOU_SUCCESS_THRESHOLD,
) -> Dict[str, Any]:
    """Pick last zoom bbox in trace; report IoU against ground-truth strip bbox."""
    ground_bboxes = get_doc_ground_bboxes(doc)
    last_bbox: Optional[Tuple[float, float, float, float]] = None
    all_bboxes: List[Tuple[float, float, float, float]] = []
    for turn in trace or []:
        for box in parse_bboxes_from_text(turn):
            all_bboxes.append(box)
            last_bbox = box

    if last_bbox is None:
        return {
            "zoom_align_success": False,
            "zoom_bbox": None,
            "zoom_yaw": None,
            "zoom_pitch": None,
            "zoom_bbox_count": 0,
            "bbox_iou": 0.0,
            "ground_bboxes": [list(box) for box in ground_bboxes],
        }

    iou = bbox_iou_vs_ground(last_bbox, ground_bboxes)
    yaw, pitch = bbox_center_to_yaw_pitch(last_bbox)
    ok = iou >= iou_threshold
    return {
        "zoom_align_success": ok,
        "zoom_bbox": list(last_bbox),
        "zoom_yaw": yaw,
        "zoom_pitch": pitch,
        "zoom_bbox_count": len(all_bboxes),
        "bbox_iou": iou,
        "ground_bboxes": [list(box) for box in ground_bboxes],
    }


def inject_submit_from_bbox(text: str, target_yaw: Sequence[float], target_pitch: Sequence[float]) -> str:
    """Replace submit_bbox(...) with submit(yaw,pitch) using bbox center (for env simulator)."""
    bbox = parse_submit_bbox(text)
    if bbox is None:
        return text
    _, yaw, pitch = bbox_aligns_target(bbox, target_yaw, target_pitch)
    yaw_i, pitch_i = int(round(yaw)), int(round(pitch))
    replacement = f"submit({yaw_i},{pitch_i})"
    converted = SUBMIT_BBOX_PATTERN.sub(replacement, text, count=1)
    if "<answer>" in converted:
        return converted
    return f"<think>submit from panorama bbox</think><answer>{replacement}</answer>"


def evaluate_episode_outcome(
    doc: Dict[str, Any],
    env_success: bool,
    trace: Optional[List[str]],
    *,
    score_mode: str = "env",
    use_strip: bool = False,
) -> Dict[str, Any]:
    """Combine simulator success with panorama zoom-bbox alignment."""
    target_yaw = doc.get("target_yaw") or [0, 360]
    target_pitch = doc.get("target_pitch") or [-90, 90]
    zoom_info = best_zoom_align_from_trace(doc, trace) if use_strip else {
        "zoom_align_success": False,
        "zoom_bbox": None,
        "zoom_yaw": None,
        "zoom_pitch": None,
        "zoom_bbox_count": 0,
        "bbox_iou": 0.0,
        "ground_bboxes": [list(box) for box in get_doc_ground_bboxes(doc)],
    }

    submit_bbox = None
    for turn in reversed(trace or []):
        submit_bbox = parse_submit_bbox(turn)
        if submit_bbox is not None:
            break
    submit_bbox_align = False
    submit_yaw = submit_pitch = None
    submit_bbox_iou = 0.0
    if submit_bbox is not None:
        ground_bboxes = get_doc_ground_bboxes(doc)
        submit_bbox_iou = bbox_iou_vs_ground(submit_bbox, ground_bboxes)
        submit_bbox_align = submit_bbox_iou >= DEFAULT_IOU_SUCCESS_THRESHOLD
        submit_yaw, submit_pitch = bbox_center_to_yaw_pitch(submit_bbox)

    zoom_align = bool(zoom_info.get("zoom_align_success"))
    # Primary success for strip+zoom workflows
    if score_mode == "zoom_align":
        primary = zoom_align or submit_bbox_align
    elif score_mode == "combined_or":
        primary = env_success or zoom_align or submit_bbox_align
    elif score_mode == "combined_and":
        primary = env_success and (zoom_align or not use_strip)
    else:
        primary = env_success

    return {
        "success": primary,
        "env_success": env_success,
        "zoom_align_success": zoom_align,
        "submit_bbox_align_success": submit_bbox_align,
        "submit_bbox": list(submit_bbox) if submit_bbox else None,
        "submit_yaw": submit_yaw,
        "submit_pitch": submit_pitch,
        "submit_bbox_iou": submit_bbox_iou,
        "score_mode": score_mode,
        "use_strip": use_strip,
        **zoom_info,
    }
