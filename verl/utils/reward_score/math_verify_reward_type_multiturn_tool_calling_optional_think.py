"""Reward helpers for multi-turn tool-calling outputs with mandatory think tags."""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from numbers import Integral
from typing import Any

from verl.utils.reward_score.hstar_strip_reward import hstar_strip_reward
from verl.utils.reward_score.math_verify_reward_type_boxed import _counting_reward
from verl.utils.reward_score.math_verify_reward_type_fixations_no_think_point_boxed import (
    _extract_reward_answer,
    _extract_reward_type,
    _grounding_reward,
    _multiple_choice_reward,
    _number_list_reward,
    _numeric_reward,
    _short_string_reward,
)
from verl.utils.reward_score.text_normalization import search_string_reward

__all__ = [
    "format_reward",
    "acc_reward",
    "extract_answer_for_accuracy",
    "assistant_turn_overlong_reward",
    "tool_repeat_overlap_call_diagnostics",
    "tool_repeat_overlap_reward",
    "compute_score",
    "compute_score_from_data_source",
    "async_compute_score_from_data_source",
]


_THINK_OPEN = "<think>"
_THINK_CLOSE = "</think>"
_TOOL_CALL_OPEN = "<tool_call>"
_TOOL_CALL_CLOSE = "</tool_call>"
_TOOL_RESPONSE_OPEN = "<tool_response>"
_TOOL_RESPONSE_CLOSE = "</tool_response>"
_ANSWER_OPEN = "<answer>"
_ANSWER_CLOSE = "</answer>"
_RESERVED_TAGS = (
    _THINK_OPEN,
    _THINK_CLOSE,
    _TOOL_CALL_OPEN,
    _TOOL_CALL_CLOSE,
    _TOOL_RESPONSE_OPEN,
    _TOOL_RESPONSE_CLOSE,
    _ANSWER_OPEN,
    _ANSWER_CLOSE,
)
_TOOL_CALL_TO_RESPONSE_ROLE_RE = re.compile(
    rf"({re.escape(_TOOL_CALL_CLOSE)})\s*user\s*({re.escape(_TOOL_RESPONSE_OPEN)})"
)
_TOOL_RESPONSE_TO_ASSISTANT_RE = re.compile(rf"({re.escape(_TOOL_RESPONSE_CLOSE)})\s*assistant\s*")


@dataclass(frozen=True)
class AnswerExtraction:
    prediction: str | None
    has_boxed: bool
    used_unboxed_fallback: bool


def _normalize_chat_template_role_markers(text: str) -> str:
    """Remove role labels emitted by decoded chat templates at tool boundaries."""
    text = _TOOL_CALL_TO_RESPONSE_ROLE_RE.sub(r"\1\n\2", text)
    return _TOOL_RESPONSE_TO_ASSISTANT_RE.sub(r"\1\n", text)


def _skip_ws(text: str, pos: int) -> int:
    while pos < len(text) and text[pos].isspace():
        pos += 1
    return pos


def _contains_reserved_tag(text: str) -> bool:
    return any(tag in text for tag in _RESERVED_TAGS)


def _consume_tagged_block(text: str, pos: int, open_tag: str, close_tag: str) -> tuple[str | None, int]:
    if not text.startswith(open_tag, pos):
        return None, pos

    content_start = pos + len(open_tag)
    content_end = text.find(close_tag, content_start)
    if content_end == -1:
        return None, pos

    content = text[content_start:content_end]
    if not content.strip() or _contains_reserved_tag(content):
        return None, pos

    return content, content_end + len(close_tag)


def _consume_think_block(text: str, pos: int) -> int | None:
    _, next_pos = _consume_tagged_block(text, pos, _THINK_OPEN, _THINK_CLOSE)
    if next_pos == pos:
        return None
    return _skip_ws(text, next_pos)


def _parse_tool_call_payload(payload: str) -> dict[str, Any] | None:
    try:
        tool_call = json.loads(payload.strip())
    except json.JSONDecodeError:
        return None

    if not isinstance(tool_call, dict):
        return None

    name = tool_call.get("name")
    arguments = tool_call.get("arguments")
    if not (isinstance(name, str) and bool(name.strip()) and isinstance(arguments, dict)):
        return None
    return tool_call


def _parse_multiturn_tool_calling_output(predict_str: str) -> tuple[bool, str | None, int, list[dict[str, Any]]]:
    text = _normalize_chat_template_role_markers(str(predict_str)).strip()
    if not text:
        return False, None, 0, []

    pos = 0
    final_answer: str | None = None
    tool_call_count = 0
    tool_calls: list[dict[str, Any]] = []

    while pos < len(text):
        pos = _skip_ws(text, pos)
        pos = _consume_think_block(text, pos)
        if pos is None:
            return False, None, 0, []

        tool_call, next_pos = _consume_tagged_block(text, pos, _TOOL_CALL_OPEN, _TOOL_CALL_CLOSE)
        if tool_call is not None:
            parsed_tool_call = _parse_tool_call_payload(tool_call)
            if parsed_tool_call is None:
                return False, None, 0, []
            tool_call_count += 1
            tool_calls.append(parsed_tool_call)
            pos = _skip_ws(text, next_pos)
            while text.startswith(_TOOL_RESPONSE_OPEN, pos):
                tool_response, next_pos = _consume_tagged_block(
                    text,
                    pos,
                    _TOOL_RESPONSE_OPEN,
                    _TOOL_RESPONSE_CLOSE,
                )
                if tool_response is None:
                    return False, None, 0, []
                pos = _skip_ws(text, next_pos)
            continue

        answer, next_pos = _consume_tagged_block(text, pos, _ANSWER_OPEN, _ANSWER_CLOSE)
        if answer is None:
            return False, None, 0, []

        pos = _skip_ws(text, next_pos)
        if pos != len(text):
            return False, None, 0, []
        final_answer = answer.strip()
        break

    return final_answer is not None, final_answer, tool_call_count, tool_calls


def _extract_last_valid_boxed_content(text: str) -> str | None:
    if not text:
        return None

    idx = 0
    last_content: str | None = None
    while True:
        start = text.find(r"\boxed", idx)
        if start == -1:
            break
        cursor = start + len(r"\boxed")
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        if cursor >= len(text) or text[cursor] != "{":
            idx = cursor + 1
            continue
        cursor += 1
        depth = 1
        content_start = cursor
        while cursor < len(text) and depth > 0:
            ch = text[cursor]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            cursor += 1
        if depth == 0:
            last_content = text[content_start : cursor - 1].strip()
            idx = cursor
        else:
            break
    return last_content or None


def extract_answer_for_accuracy(
    answer_text: str,
    reward_type: str | None,
    allow_unboxed_answer_for_accuracy: bool = True,
) -> AnswerExtraction:
    candidate = str(answer_text).strip()
    if not candidate:
        return AnswerExtraction(prediction=None, has_boxed=False, used_unboxed_fallback=False)

    has_boxed = _extract_last_valid_boxed_content(candidate) is not None
    if reward_type is None:
        return AnswerExtraction(prediction=None, has_boxed=has_boxed, used_unboxed_fallback=False)

    if has_boxed:
        prediction = _extract_reward_answer(candidate, reward_type)
        return AnswerExtraction(
            prediction=prediction,
            has_boxed=True,
            used_unboxed_fallback=False,
        )

    if not allow_unboxed_answer_for_accuracy:
        return AnswerExtraction(prediction=None, has_boxed=False, used_unboxed_fallback=False)

    return AnswerExtraction(
        prediction=candidate,
        has_boxed=False,
        used_unboxed_fallback=True,
    )


def _compute_formatting_score(
    is_valid: bool,
    answer_info: AnswerExtraction,
    require_boxed_answer_for_full_format: bool = False,
    unboxed_answer_format_score: float = 0.0,
) -> float:
    if not is_valid:
        return 0.0
    if not require_boxed_answer_for_full_format:
        return 1.0
    if answer_info.has_boxed:
        return 1.0
    if answer_info.used_unboxed_fallback and answer_info.prediction:
        return float(unboxed_answer_format_score)
    return 0.0


def format_reward(
    predict_str: str,
    extra_info: Any | None = None,
    allow_unboxed_answer_for_accuracy: bool = True,
    require_boxed_answer_for_full_format: bool = False,
    unboxed_answer_format_score: float = 0.0,
) -> float:
    is_valid, _, _, _ = _parse_multiturn_tool_calling_output(predict_str)
    if not is_valid or not require_boxed_answer_for_full_format:
        return 1.0 if is_valid else 0.0
    reward_type = _extract_reward_type(extra_info)
    _, final_answer_text, _, _ = _parse_multiturn_tool_calling_output(predict_str)
    answer_info = extract_answer_for_accuracy(
        final_answer_text or "",
        reward_type,
        allow_unboxed_answer_for_accuracy=allow_unboxed_answer_for_accuracy,
    )
    return _compute_formatting_score(
        is_valid,
        answer_info,
        require_boxed_answer_for_full_format=require_boxed_answer_for_full_format,
        unboxed_answer_format_score=unboxed_answer_format_score,
    )


def _build_counting_reward_kwargs(
    *,
    rel_tol: float | None = None,
    rel_band_cap: float | None = None,
    rel_sigma: float | None = None,
    min_sigma: float | None = None,
    decay_exp: float | None = None,
    graded: bool = True,
) -> dict[str, Any]:
    """Collect non-default counting knobs into a kwargs dict for ``counting_reward_value``.

    Returns ``{}`` when every knob is at its default, so the counting reward path is
    bit-identical to the shipped behavior unless a training arm overrides a knob via
    ``reward.custom_reward_function.reward_kwargs.counting_*``.
    """
    kwargs: dict[str, Any] = {}
    if rel_tol is not None:
        kwargs["rel_tol"] = float(rel_tol)
    if rel_band_cap is not None:
        kwargs["rel_band_cap"] = float(rel_band_cap)
    if rel_sigma is not None:
        kwargs["rel_sigma"] = float(rel_sigma)
    if min_sigma is not None:
        kwargs["min_sigma"] = float(min_sigma)
    if decay_exp is not None:
        kwargs["decay_exp"] = float(decay_exp)
    if not graded:
        kwargs["graded"] = False
    return kwargs


def acc_reward(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = False,  # noqa: ARG001
    extra_info=None,
    allow_unboxed_answer_for_accuracy: bool = True,
    counting_reward_kwargs: dict[str, Any] | None = None,
) -> float:
    reward_type = _extract_reward_type(extra_info)
    if reward_type is None:
        return 0.0

    is_valid, final_answer_text, _, _ = _parse_multiturn_tool_calling_output(predict_str)
    if not is_valid or not final_answer_text:
        return 0.0

    answer_info = extract_answer_for_accuracy(
        final_answer_text,
        reward_type,
        allow_unboxed_answer_for_accuracy=allow_unboxed_answer_for_accuracy,
    )
    if answer_info.prediction is None:
        return 0.0

    truth = str(ground_truth).strip()
    reward_handlers = {
        "short_string": _short_string_reward,
        "string_match": _short_string_reward,
        "multiple_choice": _multiple_choice_reward,
        "number_list": _number_list_reward,
        "numeric": _numeric_reward,
        "counting": _counting_reward,
        "search": search_string_reward,
        "grounding": _grounding_reward,
        "hstar_strip": hstar_strip_reward,
    }
    handler = reward_handlers.get(reward_type)
    if handler is None:
        return 0.0
    handler_kwargs: dict[str, Any] = {"extra_info": extra_info}
    if reward_type == "counting" and counting_reward_kwargs:
        handler_kwargs["counting_reward_kwargs"] = counting_reward_kwargs
    return handler(answer_info.prediction, truth, **handler_kwargs)


def tool_efficiency_reward(
    tool_call_count: int,
    mode: str = "inverse_count",
    free_calls: int = 0,
    concave_k: float = 10.0,
    concave_q: float = 0.95,
) -> float:
    effective_tool_calls = max(float(tool_call_count) - float(free_calls), 0.0)
    if mode == "inverse_count":
        return 1.0 / (effective_tool_calls + 1.0)
    if mode == "concave_length_penalty":
        k = float(concave_k)
        q = float(concave_q)
        if k <= 0.0:
            raise ValueError(f"tool_efficiency_concave_k must be > 0, got {k}.")
        if abs(q - 1.0) <= 1e-8:
            return -math.log1p(k * effective_tool_calls) / k
        return -((1.0 + k * effective_tool_calls) ** (1.0 - q) - 1.0) / (k * (1.0 - q))
    raise ValueError(f"Unsupported tool_efficiency_reward_mode: {mode}")


def assistant_turn_overlong_reward(
    assistant_turns: int,
    *,
    max_turns: int,
    buffer_turns: int,
    penalty_factor: float = 1.0,
) -> float:
    """Return a bounded linear penalty over the final assistant-turn buffer."""
    if isinstance(assistant_turns, bool) or not isinstance(assistant_turns, Integral):
        raise ValueError(f"assistant_turns must be an integer, got {assistant_turns!r}.")
    if isinstance(max_turns, bool) or not isinstance(max_turns, Integral) or max_turns <= 0:
        raise ValueError(f"assistant_turn_overlong_max_turns must be a positive integer, got {max_turns!r}.")
    if isinstance(buffer_turns, bool) or not isinstance(buffer_turns, Integral) or buffer_turns <= 0:
        raise ValueError(
            "assistant_turn_overlong_buffer_turns must be a positive integer, "
            f"got {buffer_turns!r}."
        )
    if buffer_turns > max_turns:
        raise ValueError(
            "assistant_turn_overlong_buffer_turns must not exceed "
            f"assistant_turn_overlong_max_turns, got {buffer_turns} > {max_turns}."
        )
    factor = float(penalty_factor)
    if not math.isfinite(factor) or factor <= 0.0:
        raise ValueError(
            "assistant_turn_overlong_penalty_factor must be finite and positive, "
            f"got {penalty_factor!r}."
        )
    if assistant_turns < 0 or assistant_turns > max_turns:
        raise ValueError(f"assistant_turns must be in [0, {max_turns}], got {assistant_turns}.")

    expected_turns = max_turns - buffer_turns
    raw_penalty = -(assistant_turns - expected_turns) / buffer_turns * factor
    return max(-factor, min(raw_penalty, 0.0))


def _extract_zoom_call_box(tool_call: dict[str, Any]) -> tuple[int, tuple[float, float, float, float]] | None:
    if tool_call.get("name") != "image_zoom_in_tool":
        return None
    arguments = tool_call.get("arguments")
    if not isinstance(arguments, dict):
        return None
    bbox = arguments.get("bbox_2d")
    if not isinstance(bbox, list) or len(bbox) != 4 or any(isinstance(value, bool) for value in bbox):
        return None
    raw_img_idx = arguments.get("img_idx")
    if isinstance(raw_img_idx, bool):
        return None
    try:
        x1, y1, x2, y2 = (float(value) for value in bbox)
        img_idx_value = float(raw_img_idx)
    except (TypeError, ValueError):
        return None
    if (
        not all(math.isfinite(value) for value in (x1, y1, x2, y2, img_idx_value))
        or not img_idx_value.is_integer()
        or img_idx_value < 0
        or not x1 < x2
        or not y1 < y2
    ):
        return None
    return int(img_idx_value), (x1, y1, x2, y2)


def _bbox_iou(box_a: tuple[float, float, float, float], box_b: tuple[float, float, float, float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b
    inter_w = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    inter_h = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = inter_w * inter_h
    if inter <= 0.0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    union = area_a + area_b - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _bbox_area(box: tuple[float, float, float, float]) -> float:
    return (box[2] - box[0]) * (box[3] - box[1])


def _bbox_containment(
    inner: tuple[float, float, float, float],
    outer: tuple[float, float, float, float],
) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    intersection = max(0.0, min(ix2, ox2) - max(ix1, ox1)) * max(0.0, min(iy2, oy2) - max(iy1, oy1))
    inner_area = _bbox_area(inner)
    return intersection / inner_area if inner_area > 0.0 else 0.0


def _extract_executed_tool_calls_best_effort(predict_str: str) -> list[dict[str, Any]]:
    """Recover well-formed tool calls that have a following tool response.

    Accuracy and format scoring intentionally retain the strict trajectory parser. This
    recovery path is used only for tool-quality rewards so a malformed final answer does
    not erase penalties for tool calls that were already executed.
    """
    text = _normalize_chat_template_role_markers(str(predict_str))
    tool_calls: list[dict[str, Any]] = []
    pos = 0
    while True:
        start = text.find(_TOOL_CALL_OPEN, pos)
        if start < 0:
            break
        payload_start = start + len(_TOOL_CALL_OPEN)
        payload_end = text.find(_TOOL_CALL_CLOSE, payload_start)
        if payload_end < 0:
            break
        parsed = _parse_tool_call_payload(text[payload_start:payload_end])
        response_pos = _skip_ws(text, payload_end + len(_TOOL_CALL_CLOSE))
        response, next_pos = _consume_tagged_block(text, response_pos, _TOOL_RESPONSE_OPEN, _TOOL_RESPONSE_CLOSE)
        if parsed is not None and response is not None:
            tool_calls.append(parsed)
        pos = next_pos if next_pos > response_pos else payload_end + len(_TOOL_CALL_CLOSE)
    return tool_calls


def tool_repeat_overlap_reward(
    tool_calls: list[dict[str, Any]],
    iou_threshold: float = 0.9,
    penalty_per_repeat: float | None = None,  # Legacy compatibility; intentionally unused.
    max_penalty: float | None = None,  # Legacy compatibility; intentionally unused.
    *,
    small_crop_area_threshold: float = 0.0002,
    repeat_weight: float = 1.0,
    small_crop_weight: float = 0.5,
    free_depth_refinements: int = 2,
    depth_containment_threshold: float = 0.95,
    excess_depth_weight: float = 0.5,
) -> float:
    """Return a normalized composite crop-quality penalty under the existing key."""
    del penalty_per_repeat, max_penalty
    diagnostics = tool_repeat_overlap_call_diagnostics(
        tool_calls,
        iou_threshold=iou_threshold,
        small_crop_area_threshold=small_crop_area_threshold,
        free_depth_refinements=free_depth_refinements,
        depth_containment_threshold=depth_containment_threshold,
    )
    valid_diagnostics = [item for item in diagnostics if item["valid"]]
    num_calls = len(valid_diagnostics)
    if num_calls == 0:
        return 0.0

    repeat_count = sum(bool(item["is_repeat"]) for item in valid_diagnostics)
    small_crop_count = sum(bool(item["is_small_crop"]) for item in valid_diagnostics)
    excess_depth_count = sum(bool(item["is_excess_depth"]) for item in valid_diagnostics)
    repeat_fraction = repeat_count / num_calls
    small_crop_fraction = small_crop_count / num_calls
    excess_depth_fraction = excess_depth_count / num_calls
    return -(
        max(float(repeat_weight), 0.0) * repeat_fraction
        + max(float(small_crop_weight), 0.0) * small_crop_fraction
        + max(float(excess_depth_weight), 0.0) * excess_depth_fraction
    )


def tool_repeat_overlap_call_diagnostics(
    tool_calls: list[dict[str, Any]],
    iou_threshold: float = 0.9,
    *,
    small_crop_area_threshold: float = 0.0002,
    free_depth_refinements: int = 2,
    depth_containment_threshold: float = 0.95,
) -> list[dict[str, Any]]:
    """Return per-call crop-quality flags using the reward's exact state machine.

    Invalid or non-zoom calls retain their position in the returned list with
    ``valid=False``. Reward aggregation ignores those entries, while rollout
    filtering can fail closed on them.
    """
    threshold = min(max(float(iou_threshold), 0.0), 1.0)
    area_threshold = max(float(small_crop_area_threshold), 0.0)
    containment_threshold = min(max(float(depth_containment_threshold), 0.0), 1.0)
    free_depth = max(int(free_depth_refinements), 0)
    depth_run = 0
    previous_by_img_idx: dict[int, list[tuple[float, float, float, float]]] = {}
    previous_call: tuple[int, tuple[float, float, float, float]] | None = None
    diagnostics: list[dict[str, Any]] = []
    for tool_call in tool_calls:
        parsed = _extract_zoom_call_box(tool_call)
        if parsed is None:
            diagnostics.append(
                {
                    "valid": False,
                    "is_repeat": False,
                    "is_small_crop": False,
                    "is_excess_depth": False,
                    "has_penalty": False,
                }
            )
            continue

        img_idx, box = parsed
        previous_boxes = previous_by_img_idx.setdefault(img_idx, [])
        is_repeat = any(_bbox_iou(box, previous_box) >= threshold for previous_box in previous_boxes)

        area = _bbox_area(box)
        is_small_crop = area / 1_000_000.0 < area_threshold

        is_depth_refinement = False
        if previous_call is not None and previous_call[0] == img_idx and not is_repeat:
            previous_box = previous_call[1]
            is_depth_refinement = (
                area < _bbox_area(previous_box)
                and _bbox_containment(box, previous_box) >= containment_threshold
            )
        if is_depth_refinement:
            depth_run += 1
        else:
            depth_run = 0
        is_excess_depth = depth_run > free_depth

        previous_boxes.append(box)
        previous_call = (img_idx, box)
        diagnostics.append(
            {
                "valid": True,
                "img_idx": img_idx,
                "bbox": box,
                "is_repeat": is_repeat,
                "is_small_crop": is_small_crop,
                "is_excess_depth": is_excess_depth,
                "has_penalty": is_repeat or is_small_crop or is_excess_depth,
            }
        )
    return diagnostics


def compute_score(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = False,
    format_score: float = 0.5,
    min_tool_calls_for_bonus: int | None = None,
    tool_call_bonus: float = 0.0,
    tool_efficiency_reward_mode: str = "inverse_count",
    tool_efficiency_free_calls: int = 0,
    tool_efficiency_concave_k: float = 10.0,
    tool_efficiency_concave_q: float = 0.95,
    tool_efficiency_score_weight: float = 0.0,
    tool_repeat_overlap_iou_threshold: float = 0.9,
    tool_repeat_overlap_penalty_per_repeat: float | None = None,
    tool_repeat_overlap_max_penalty: float | None = None,
    tool_repeat_overlap_small_crop_area_threshold: float = 0.0002,
    tool_repeat_overlap_repeat_weight: float = 1.0,
    tool_repeat_overlap_small_crop_weight: float = 0.5,
    tool_repeat_overlap_free_depth_refinements: int = 2,
    tool_repeat_overlap_depth_containment_threshold: float = 0.95,
    tool_repeat_overlap_excess_depth_weight: float = 0.5,
    assistant_turn_overlong_enabled: bool = False,
    assistant_turn_overlong_max_turns: int | None = None,
    assistant_turn_overlong_buffer_turns: int | None = None,
    assistant_turn_overlong_penalty_factor: float = 1.0,
    allow_unboxed_answer_for_accuracy: bool = True,
    require_boxed_answer_for_full_format: bool = False,
    unboxed_answer_format_score: float = 0.0,
    counting_rel_tol: float | None = None,
    counting_rel_band_cap: float | None = None,
    counting_rel_sigma: float | None = None,
    counting_min_sigma: float | None = None,
    counting_decay_exp: float | None = None,
    counting_graded: bool = True,
    extra_info=None,
) -> dict[str, Any]:
    predict_str = str(predict_str)
    ground_truth = str(ground_truth)

    counting_reward_kwargs = _build_counting_reward_kwargs(
        rel_tol=counting_rel_tol,
        rel_band_cap=counting_rel_band_cap,
        rel_sigma=counting_rel_sigma,
        min_sigma=counting_min_sigma,
        decay_exp=counting_decay_exp,
        graded=counting_graded,
    )

    is_valid, final_answer_text, tool_call_count, tool_calls = _parse_multiturn_tool_calling_output(predict_str)
    reward_type = _extract_reward_type(extra_info)
    answer_info = extract_answer_for_accuracy(
        final_answer_text or "",
        reward_type,
        allow_unboxed_answer_for_accuracy=allow_unboxed_answer_for_accuracy,
    )
    formatting = _compute_formatting_score(
        is_valid,
        answer_info,
        require_boxed_answer_for_full_format=require_boxed_answer_for_full_format,
        unboxed_answer_format_score=unboxed_answer_format_score,
    )
    if not is_valid:
        accuracy = 0.0
    else:
        accuracy = acc_reward(
            predict_str,
            ground_truth,
            use_boxed=use_boxed,
            extra_info=extra_info,
            allow_unboxed_answer_for_accuracy=allow_unboxed_answer_for_accuracy,
            counting_reward_kwargs=counting_reward_kwargs,
        )

    tool_call_bonus_score = 0.0
    if (
        formatting == 1.0
        and min_tool_calls_for_bonus is not None
        and float(tool_call_bonus) != 0.0
        and tool_call_count >= int(min_tool_calls_for_bonus)
    ):
        tool_call_bonus_score = float(tool_call_bonus)

    tool_efficiency_score = tool_efficiency_reward(
        tool_call_count,
        mode=tool_efficiency_reward_mode,
        free_calls=tool_efficiency_free_calls,
        concave_k=tool_efficiency_concave_k,
        concave_q=tool_efficiency_concave_q,
    )
    penalty_tool_calls = tool_calls if is_valid else _extract_executed_tool_calls_best_effort(predict_str)
    tool_repeat_overlap_score = tool_repeat_overlap_reward(
        penalty_tool_calls,
        iou_threshold=tool_repeat_overlap_iou_threshold,
        penalty_per_repeat=tool_repeat_overlap_penalty_per_repeat,
        max_penalty=tool_repeat_overlap_max_penalty,
        small_crop_area_threshold=tool_repeat_overlap_small_crop_area_threshold,
        repeat_weight=tool_repeat_overlap_repeat_weight,
        small_crop_weight=tool_repeat_overlap_small_crop_weight,
        free_depth_refinements=tool_repeat_overlap_free_depth_refinements,
        depth_containment_threshold=tool_repeat_overlap_depth_containment_threshold,
        excess_depth_weight=tool_repeat_overlap_excess_depth_weight,
    )
    score = (
        (1.0 - format_score - float(tool_efficiency_score_weight)) * accuracy
        + format_score * formatting
        + float(tool_efficiency_score_weight) * tool_efficiency_score
        + tool_call_bonus_score
    )
    result = {
        "score": score,
        "accuracy": accuracy,
        "format": formatting,
        "tool_call_count": tool_call_count,
        "tool_efficiency": tool_efficiency_score,
        "tool_repeat_overlap": tool_repeat_overlap_score,
        "tool_call_bonus": tool_call_bonus_score,
    }
    if assistant_turn_overlong_enabled:
        if not isinstance(extra_info, dict) or "assistant_turns" not in extra_info:
            raise ValueError(
                "assistant_turns must be provided in extra_info when assistant-turn overlong reward is enabled."
            )
        if assistant_turn_overlong_max_turns is None:
            raise ValueError(
                "assistant_turn_overlong_max_turns must be provided when assistant-turn overlong reward is enabled."
            )
        if assistant_turn_overlong_buffer_turns is None:
            raise ValueError(
                "assistant_turn_overlong_buffer_turns must be provided when assistant-turn overlong reward is enabled."
            )
        result["assistant_turn_overlong"] = assistant_turn_overlong_reward(
            extra_info["assistant_turns"],
            max_turns=assistant_turn_overlong_max_turns,
            buffer_turns=assistant_turn_overlong_buffer_turns,
            penalty_factor=assistant_turn_overlong_penalty_factor,
        )
    if require_boxed_answer_for_full_format:
        result["answer_has_boxed"] = answer_info.has_boxed
        result["answer_used_unboxed_fallback"] = answer_info.used_unboxed_fallback
    return result


def compute_score_from_data_source(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    use_boxed: bool = False,
    format_score: float = 0.5,
    min_tool_calls_for_bonus: int | None = None,
    tool_call_bonus: float = 0.0,
    tool_efficiency_reward_mode: str = "inverse_count",
    tool_efficiency_free_calls: int = 0,
    tool_efficiency_concave_k: float = 10.0,
    tool_efficiency_concave_q: float = 0.95,
    tool_efficiency_score_weight: float = 0.0,
    tool_repeat_overlap_iou_threshold: float = 0.9,
    tool_repeat_overlap_penalty_per_repeat: float | None = None,
    tool_repeat_overlap_max_penalty: float | None = None,
    tool_repeat_overlap_small_crop_area_threshold: float = 0.0002,
    tool_repeat_overlap_repeat_weight: float = 1.0,
    tool_repeat_overlap_small_crop_weight: float = 0.5,
    tool_repeat_overlap_free_depth_refinements: int = 2,
    tool_repeat_overlap_depth_containment_threshold: float = 0.95,
    tool_repeat_overlap_excess_depth_weight: float = 0.5,
    assistant_turn_overlong_enabled: bool = False,
    assistant_turn_overlong_max_turns: int | None = None,
    assistant_turn_overlong_buffer_turns: int | None = None,
    assistant_turn_overlong_penalty_factor: float = 1.0,
    allow_unboxed_answer_for_accuracy: bool = True,
    require_boxed_answer_for_full_format: bool = False,
    unboxed_answer_format_score: float = 0.0,
    counting_rel_tol: float | None = None,
    counting_rel_band_cap: float | None = None,
    counting_rel_sigma: float | None = None,
    counting_min_sigma: float | None = None,
    counting_decay_exp: float | None = None,
    counting_graded: bool = True,
    **_: dict,
) -> dict[str, Any]:
    """Adapter for custom_reward_function configs that expect default signature."""
    return compute_score(
        solution_str,
        ground_truth,
        use_boxed=use_boxed,
        format_score=format_score,
        min_tool_calls_for_bonus=min_tool_calls_for_bonus,
        tool_call_bonus=tool_call_bonus,
        tool_efficiency_reward_mode=tool_efficiency_reward_mode,
        tool_efficiency_free_calls=tool_efficiency_free_calls,
        tool_efficiency_concave_k=tool_efficiency_concave_k,
        tool_efficiency_concave_q=tool_efficiency_concave_q,
        tool_efficiency_score_weight=tool_efficiency_score_weight,
        tool_repeat_overlap_iou_threshold=tool_repeat_overlap_iou_threshold,
        tool_repeat_overlap_penalty_per_repeat=tool_repeat_overlap_penalty_per_repeat,
        tool_repeat_overlap_max_penalty=tool_repeat_overlap_max_penalty,
        tool_repeat_overlap_small_crop_area_threshold=tool_repeat_overlap_small_crop_area_threshold,
        tool_repeat_overlap_repeat_weight=tool_repeat_overlap_repeat_weight,
        tool_repeat_overlap_small_crop_weight=tool_repeat_overlap_small_crop_weight,
        tool_repeat_overlap_free_depth_refinements=tool_repeat_overlap_free_depth_refinements,
        tool_repeat_overlap_depth_containment_threshold=tool_repeat_overlap_depth_containment_threshold,
        tool_repeat_overlap_excess_depth_weight=tool_repeat_overlap_excess_depth_weight,
        assistant_turn_overlong_enabled=assistant_turn_overlong_enabled,
        assistant_turn_overlong_max_turns=assistant_turn_overlong_max_turns,
        assistant_turn_overlong_buffer_turns=assistant_turn_overlong_buffer_turns,
        assistant_turn_overlong_penalty_factor=assistant_turn_overlong_penalty_factor,
        allow_unboxed_answer_for_accuracy=allow_unboxed_answer_for_accuracy,
        require_boxed_answer_for_full_format=require_boxed_answer_for_full_format,
        unboxed_answer_format_score=unboxed_answer_format_score,
        counting_rel_tol=counting_rel_tol,
        counting_rel_band_cap=counting_rel_band_cap,
        counting_rel_sigma=counting_rel_sigma,
        counting_min_sigma=counting_min_sigma,
        counting_decay_exp=counting_decay_exp,
        counting_graded=counting_graded,
        extra_info=extra_info,
    )


async def async_compute_score_from_data_source(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    use_boxed: bool = False,
    format_score: float = 0.5,
    min_tool_calls_for_bonus: int | None = None,
    tool_call_bonus: float = 0.0,
    tool_efficiency_reward_mode: str = "inverse_count",
    tool_efficiency_free_calls: int = 0,
    tool_efficiency_concave_k: float = 10.0,
    tool_efficiency_concave_q: float = 0.95,
    tool_efficiency_score_weight: float = 0.0,
    tool_repeat_overlap_iou_threshold: float = 0.9,
    tool_repeat_overlap_penalty_per_repeat: float | None = None,
    tool_repeat_overlap_max_penalty: float | None = None,
    tool_repeat_overlap_small_crop_area_threshold: float = 0.0002,
    tool_repeat_overlap_repeat_weight: float = 1.0,
    tool_repeat_overlap_small_crop_weight: float = 0.5,
    tool_repeat_overlap_free_depth_refinements: int = 2,
    tool_repeat_overlap_depth_containment_threshold: float = 0.95,
    tool_repeat_overlap_excess_depth_weight: float = 0.5,
    assistant_turn_overlong_enabled: bool = False,
    assistant_turn_overlong_max_turns: int | None = None,
    assistant_turn_overlong_buffer_turns: int | None = None,
    assistant_turn_overlong_penalty_factor: float = 1.0,
    allow_unboxed_answer_for_accuracy: bool = True,
    require_boxed_answer_for_full_format: bool = False,
    unboxed_answer_format_score: float = 0.0,
    counting_rel_tol: float | None = None,
    counting_rel_band_cap: float | None = None,
    counting_rel_sigma: float | None = None,
    counting_min_sigma: float | None = None,
    counting_decay_exp: float | None = None,
    counting_graded: bool = True,
    **_: dict,
) -> dict[str, Any]:
    """Async adapter for reward loops."""

    return compute_score(
        solution_str,
        ground_truth,
        use_boxed=use_boxed,
        format_score=format_score,
        min_tool_calls_for_bonus=min_tool_calls_for_bonus,
        tool_call_bonus=tool_call_bonus,
        tool_efficiency_reward_mode=tool_efficiency_reward_mode,
        tool_efficiency_free_calls=tool_efficiency_free_calls,
        tool_efficiency_concave_k=tool_efficiency_concave_k,
        tool_efficiency_concave_q=tool_efficiency_concave_q,
        tool_efficiency_score_weight=tool_efficiency_score_weight,
        tool_repeat_overlap_iou_threshold=tool_repeat_overlap_iou_threshold,
        tool_repeat_overlap_penalty_per_repeat=tool_repeat_overlap_penalty_per_repeat,
        tool_repeat_overlap_max_penalty=tool_repeat_overlap_max_penalty,
        tool_repeat_overlap_small_crop_area_threshold=tool_repeat_overlap_small_crop_area_threshold,
        tool_repeat_overlap_repeat_weight=tool_repeat_overlap_repeat_weight,
        tool_repeat_overlap_small_crop_weight=tool_repeat_overlap_small_crop_weight,
        tool_repeat_overlap_free_depth_refinements=tool_repeat_overlap_free_depth_refinements,
        tool_repeat_overlap_depth_containment_threshold=tool_repeat_overlap_depth_containment_threshold,
        tool_repeat_overlap_excess_depth_weight=tool_repeat_overlap_excess_depth_weight,
        assistant_turn_overlong_enabled=assistant_turn_overlong_enabled,
        assistant_turn_overlong_max_turns=assistant_turn_overlong_max_turns,
        assistant_turn_overlong_buffer_turns=assistant_turn_overlong_buffer_turns,
        assistant_turn_overlong_penalty_factor=assistant_turn_overlong_penalty_factor,
        allow_unboxed_answer_for_accuracy=allow_unboxed_answer_for_accuracy,
        require_boxed_answer_for_full_format=require_boxed_answer_for_full_format,
        unboxed_answer_format_score=unboxed_answer_format_score,
        counting_rel_tol=counting_rel_tol,
        counting_rel_band_cap=counting_rel_band_cap,
        counting_rel_sigma=counting_rel_sigma,
        counting_min_sigma=counting_min_sigma,
        counting_decay_exp=counting_decay_exp,
        counting_graded=counting_graded,
        extra_info=extra_info,
    )
