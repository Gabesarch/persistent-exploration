"""Reward helpers for multi-turn tool-calling outputs with mandatory think tags."""

from __future__ import annotations

import json
import math
import re
from typing import Any

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


def _validate_tool_call_payload(payload: str) -> bool:
    try:
        tool_call = json.loads(payload.strip())
    except json.JSONDecodeError:
        return False

    if not isinstance(tool_call, dict):
        return False

    name = tool_call.get("name")
    arguments = tool_call.get("arguments")
    return isinstance(name, str) and bool(name.strip()) and isinstance(arguments, dict)


def _parse_multiturn_tool_calling_output(predict_str: str) -> tuple[bool, str | None, int]:
    text = _normalize_chat_template_role_markers(str(predict_str)).strip()
    if not text:
        return False, None, 0

    pos = 0
    final_answer: str | None = None
    tool_call_count = 0

    while pos < len(text):
        pos = _skip_ws(text, pos)
        pos = _consume_think_block(text, pos)
        if pos is None:
            return False, None, 0

        tool_call, next_pos = _consume_tagged_block(text, pos, _TOOL_CALL_OPEN, _TOOL_CALL_CLOSE)
        if tool_call is not None:
            if not _validate_tool_call_payload(tool_call):
                return False, None, 0
            tool_call_count += 1
            pos = _skip_ws(text, next_pos)
            while text.startswith(_TOOL_RESPONSE_OPEN, pos):
                tool_response, next_pos = _consume_tagged_block(
                    text,
                    pos,
                    _TOOL_RESPONSE_OPEN,
                    _TOOL_RESPONSE_CLOSE,
                )
                if tool_response is None:
                    return False, None, 0
                pos = _skip_ws(text, next_pos)
            continue

        answer, next_pos = _consume_tagged_block(text, pos, _ANSWER_OPEN, _ANSWER_CLOSE)
        if answer is None:
            return False, None, 0

        pos = _skip_ws(text, next_pos)
        if pos != len(text):
            return False, None, 0
        final_answer = answer.strip()
        break

    return final_answer is not None, final_answer, tool_call_count


def format_reward(predict_str: str, extra_info: Any | None = None) -> float:  # noqa: ARG001
    is_valid, _, _ = _parse_multiturn_tool_calling_output(predict_str)
    return 1.0 if is_valid else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = False, extra_info=None) -> float:  # noqa: ARG001
    reward_type = _extract_reward_type(extra_info)
    if reward_type is None:
        return 0.0

    is_valid, final_answer_text, _ = _parse_multiturn_tool_calling_output(predict_str)
    if not is_valid or not final_answer_text:
        return 0.0

    final_answer = _extract_reward_answer(final_answer_text, reward_type)
    if final_answer is None:
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
    }
    handler = reward_handlers.get(reward_type)
    if handler is None:
        return 0.0
    return handler(final_answer, truth, extra_info=extra_info)


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
    extra_info=None,
) -> dict[str, float]:
    predict_str = str(predict_str)
    ground_truth = str(ground_truth)

    is_valid, _, tool_call_count = _parse_multiturn_tool_calling_output(predict_str)
    formatting = 1.0 if is_valid else 0.0
    if formatting == 0.0:
        accuracy = 0.0
    else:
        accuracy = acc_reward(predict_str, ground_truth, use_boxed=use_boxed, extra_info=extra_info)

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
    score = (
        (1.0 - format_score - float(tool_efficiency_score_weight)) * accuracy
        + format_score * formatting
        + float(tool_efficiency_score_weight) * tool_efficiency_score
        + tool_call_bonus_score
    )
    return {
        "score": score,
        "accuracy": accuracy,
        "format": formatting,
        "tool_call_count": tool_call_count,
        "tool_efficiency": tool_efficiency_score,
        "tool_call_bonus": tool_call_bonus_score,
    }


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
    **_: dict,
) -> dict[str, float]:
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
    **_: dict,
) -> dict[str, float]:
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
        extra_info=extra_info,
    )
