"""Experiment-local reward routing for the three-source zoom mixture.

Visual Probe ``search`` rows keep the base reward path so the tool agent can
replace their string score with its LLM-judge score. Pointing rows are handled
here: their strict JSON point prediction is compared with every ground-truth box
after float-preserving conversion from image pixels to the 0-1000 coordinate
space. Keeping this logic local avoids changing existing experiments.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from typing import Any

from verl.utils.reward_score.math_verify_reward_type_boxed import _extract_grounding_clicking_answer
from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
    _parse_multiturn_tool_calling_output,
)
from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
    compute_score_from_data_source as _base_compute_score,
)

_TOOL_REPEAT_OVERLAP_DOMAIN_OVERRIDES_KEY = "tool_repeat_overlap_domain_overrides"
_SUPPORTED_TOOL_REPEAT_OVERLAP_OVERRIDE_KEYS = frozenset(
    {
        "tool_repeat_overlap_iou_threshold",
        "tool_repeat_overlap_small_crop_area_threshold",
        "tool_repeat_overlap_repeat_weight",
        "tool_repeat_overlap_small_crop_weight",
        "tool_repeat_overlap_free_depth_refinements",
        "tool_repeat_overlap_depth_containment_threshold",
        "tool_repeat_overlap_excess_depth_weight",
    }
)


def _reward_kwargs_for_domain(extra_info: Any, reward_kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Return reward kwargs with any configured mixture-source override applied."""
    routed_kwargs = dict(reward_kwargs)
    domain_overrides = routed_kwargs.pop(_TOOL_REPEAT_OVERLAP_DOMAIN_OVERRIDES_KEY, None)
    if domain_overrides is None:
        return routed_kwargs
    if not isinstance(domain_overrides, Mapping):
        raise ValueError(f"{_TOOL_REPEAT_OVERLAP_DOMAIN_OVERRIDES_KEY} must be a mapping")

    for domain, overrides in domain_overrides.items():
        if not isinstance(domain, str) or not domain:
            raise ValueError(f"{_TOOL_REPEAT_OVERLAP_DOMAIN_OVERRIDES_KEY} domain keys must be non-empty strings")
        if not isinstance(overrides, Mapping):
            raise ValueError(
                f"{_TOOL_REPEAT_OVERLAP_DOMAIN_OVERRIDES_KEY}[{domain!r}] must be a mapping"
            )
        unsupported = sorted(set(overrides) - _SUPPORTED_TOOL_REPEAT_OVERLAP_OVERRIDE_KEYS)
        if unsupported:
            raise ValueError(
                f"unsupported {domain!r} tool-repeat-overlap override keys: {unsupported}; "
                f"supported keys: {sorted(_SUPPORTED_TOOL_REPEAT_OVERLAP_OVERRIDE_KEYS)}"
            )

    mixture_source = extra_info.get("mixture_source") if isinstance(extra_info, Mapping) else None
    if not isinstance(mixture_source, str):
        return routed_kwargs
    selected_overrides = domain_overrides.get(mixture_source)
    if selected_overrides is not None:
        routed_kwargs.update(dict(selected_overrides))
    return routed_kwargs


def _normalized_click_inputs(
    ground_truth: Any,
    extra_info: Any,
) -> tuple[Any, Any]:
    if not isinstance(extra_info, dict) or extra_info.get("reward_type") != "clicking":
        return ground_truth, extra_info
    if extra_info.get("normalize_bbox_to_1000") is not True:
        return ground_truth, extra_info

    image_sizes = extra_info.get("image_sizes")
    if not isinstance(image_sizes, list) or len(image_sizes) != 1:
        raise ValueError("clicking rows require exactly one image_sizes entry")
    size = image_sizes[0]
    if not isinstance(size, list | tuple) or len(size) != 2:
        raise ValueError("clicking image size must be [width, height]")
    width, height = size
    if (
        not isinstance(width, int | float)
        or isinstance(width, bool)
        or not math.isfinite(width)
        or width <= 0
        or not isinstance(height, int | float)
        or isinstance(height, bool)
        or not math.isfinite(height)
        or height <= 0
    ):
        raise ValueError(f"invalid clicking image size: {size!r}")

    try:
        truth_items = json.loads(str(ground_truth))
    except json.JSONDecodeError as exc:
        raise ValueError(f"clicking ground truth is not valid JSON: {exc}") from exc
    if not isinstance(truth_items, list) or not truth_items:
        raise ValueError("clicking ground truth must be a non-empty JSON array")

    normalized_items: list[dict[str, Any]] = []
    for item_index, item in enumerate(truth_items):
        if not isinstance(item, dict):
            raise ValueError(f"clicking ground-truth item {item_index} is not an object")
        bbox = item.get("bbox_2d")
        if not isinstance(bbox, list) or len(bbox) != 4:
            raise ValueError(f"clicking ground-truth item {item_index} has invalid bbox_2d")
        try:
            x1, y1, x2, y2 = (float(value) for value in bbox)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"clicking ground-truth item {item_index} has non-numeric bbox_2d") from exc
        if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
            raise ValueError(f"clicking ground-truth item {item_index} has non-finite bbox_2d")
        if not (0.0 <= x1 < x2 <= float(width) and 0.0 <= y1 < y2 <= float(height)):
            raise ValueError(
                f"clicking ground-truth item {item_index} bbox_2d is outside {width}x{height}"
            )
        normalized_item = dict(item)
        normalized_item["bbox_2d"] = [
            x1 * 1000.0 / float(width),
            y1 * 1000.0 / float(height),
            x2 * 1000.0 / float(width),
            y2 * 1000.0 / float(height),
        ]
        normalized_items.append(normalized_item)

    adapted_extra_info = dict(extra_info)
    adapted_extra_info["image_sizes"] = [[1000, 1000]]
    adapted_extra_info["normalize_bbox_to_1000"] = False
    adapted_extra_info["click_ground_truth_pre_normalized"] = True
    adapted_ground_truth = json.dumps(normalized_items, ensure_ascii=False, separators=(",", ":"))
    return adapted_ground_truth, adapted_extra_info


def _clicking_accuracy(solution_str: str, normalized_ground_truth: str) -> float:
    is_valid, final_answer, _, _ = _parse_multiturn_tool_calling_output(solution_str)
    if not is_valid or not final_answer:
        return 0.0

    prediction_json = _extract_grounding_clicking_answer(final_answer)
    if prediction_json is None:
        return 0.0
    try:
        predictions = json.loads(prediction_json)
        truth_items = json.loads(normalized_ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0
    if not isinstance(predictions, list) or len(predictions) != 1:
        return 0.0
    prediction = predictions[0]
    if not isinstance(prediction, dict) or set(prediction) != {"point_2d"}:
        return 0.0
    point = prediction["point_2d"]
    if not isinstance(point, list) or len(point) != 2:
        return 0.0
    try:
        x, y = (float(value) for value in point)
    except (TypeError, ValueError):
        return 0.0
    if not (math.isfinite(x) and math.isfinite(y) and 0.0 <= x <= 1000.0 and 0.0 <= y <= 1000.0):
        return 0.0

    for item in truth_items:
        x1, y1, x2, y2 = item["bbox_2d"]
        if float(x1) <= x <= float(x2) and float(y1) <= y <= float(y2):
            return 1.0
    return 0.0


def _clicking_score(
    data_source: str,
    solution_str: str,
    normalized_ground_truth: str,
    adapted_extra_info: dict[str, Any],
    reward_kwargs: dict[str, Any],
) -> dict[str, Any]:
    # Ask the base scorer for trajectory formatting and tool diagnostics using a
    # non-point route, then replace only its task accuracy and aggregate score.
    diagnostic_extra_info = dict(adapted_extra_info)
    diagnostic_extra_info["reward_type"] = "search"
    base_kwargs = dict(reward_kwargs)
    requested_boxed_diagnostics = bool(base_kwargs.get("require_boxed_answer_for_full_format", False))
    base_kwargs["require_boxed_answer_for_full_format"] = False
    result = _base_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=normalized_ground_truth,
        extra_info=diagnostic_extra_info,
        **base_kwargs,
    )

    accuracy = _clicking_accuracy(solution_str, normalized_ground_truth)
    format_score = float(reward_kwargs.get("format_score", 0.5))
    tool_efficiency_weight = float(reward_kwargs.get("tool_efficiency_score_weight", 0.0))
    formatting = float(result.get("format", 0.0))
    tool_efficiency = float(result.get("tool_efficiency", 0.0))
    tool_call_bonus = float(result.get("tool_call_bonus", 0.0))
    result["accuracy"] = accuracy
    result["score"] = (
        (1.0 - format_score - tool_efficiency_weight) * accuracy
        + format_score * formatting
        + tool_efficiency_weight * tool_efficiency
        + tool_call_bonus
    )
    if requested_boxed_diagnostics:
        result["answer_has_boxed"] = False
        result["answer_used_unboxed_fallback"] = bool(formatting)
    return result


def compute_score_from_data_source(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    routed_reward_kwargs = _reward_kwargs_for_domain(extra_info, kwargs)
    adapted_ground_truth, adapted_extra_info = _normalized_click_inputs(ground_truth, extra_info)
    is_clicking = isinstance(extra_info, dict) and extra_info.get("reward_type") == "clicking"
    if is_clicking:
        if adapted_extra_info is extra_info:
            raise ValueError("joint-three clicking rows must set normalize_bbox_to_1000=true")
        return _clicking_score(
            data_source=data_source,
            solution_str=solution_str,
            normalized_ground_truth=adapted_ground_truth,
            adapted_extra_info=adapted_extra_info,
            reward_kwargs=routed_reward_kwargs,
        )
    return _base_compute_score(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=adapted_ground_truth,
        extra_info=adapted_extra_info,
        **routed_reward_kwargs,
    )


async def async_compute_score_from_data_source(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: Any = None,
    **kwargs: Any,
) -> dict[str, Any]:
    return compute_score_from_data_source(
        data_source=data_source,
        solution_str=solution_str,
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
