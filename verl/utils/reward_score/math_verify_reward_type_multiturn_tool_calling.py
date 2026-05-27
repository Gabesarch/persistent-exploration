"""Reward helpers for multi-turn tool-calling outputs."""

from __future__ import annotations

import json
from typing import Any

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


def _parse_multiturn_tool_calling_output(predict_str: str) -> tuple[bool, str | None]:
    text = str(predict_str).strip()
    if not text.startswith(_THINK_OPEN):
        return False, None

    pos = 0
    saw_final_answer = False
    final_answer: str | None = None

    while pos < len(text):
        pos = _skip_ws(text, pos)
        think, pos = _consume_tagged_block(text, pos, _THINK_OPEN, _THINK_CLOSE)
        if think is None:
            return False, None

        pos = _skip_ws(text, pos)
        tool_call, next_pos = _consume_tagged_block(text, pos, _TOOL_CALL_OPEN, _TOOL_CALL_CLOSE)
        if tool_call is not None:
            if not _validate_tool_call_payload(tool_call):
                return False, None
            pos = _skip_ws(text, next_pos)
            while text.startswith(_TOOL_RESPONSE_OPEN, pos):
                tool_response, next_pos = _consume_tagged_block(text, pos, _TOOL_RESPONSE_OPEN, _TOOL_RESPONSE_CLOSE)
                if tool_response is None:
                    return False, None
                pos = next_pos
                pos = _skip_ws(text, pos)
            continue

        answer, next_pos = _consume_tagged_block(text, pos, _ANSWER_OPEN, _ANSWER_CLOSE)
        if answer is None:
            return False, None

        pos = _skip_ws(text, next_pos)
        if pos != len(text):
            return False, None
        saw_final_answer = True
        final_answer = answer.strip()
        break

    return saw_final_answer, final_answer


def format_reward(predict_str: str, extra_info: Any | None = None) -> float:  # noqa: ARG001
    is_valid, _ = _parse_multiturn_tool_calling_output(predict_str)
    return 1.0 if is_valid else 0.0


def acc_reward(predict_str: str, ground_truth: str, use_boxed: bool = False, extra_info=None) -> float:  # noqa: ARG001
    reward_type = _extract_reward_type(extra_info)
    if reward_type is None:
        return 0.0

    is_valid, final_answer_text = _parse_multiturn_tool_calling_output(predict_str)
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
        "counting": _short_string_reward,
        "search": search_string_reward,
        "grounding": _grounding_reward,
    }
    handler = reward_handlers.get(reward_type)
    if handler is None:
        return 0.0
    return handler(final_answer, truth, extra_info=extra_info)


def compute_score(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = False,
    format_score: float = 0.5,
    extra_info=None,
) -> dict[str, float]:
    predict_str = str(predict_str)
    ground_truth = str(ground_truth)

    formatting = format_reward(predict_str, extra_info=extra_info)
    if formatting == 0.0:
        accuracy = 0.0
    else:
        accuracy = acc_reward(predict_str, ground_truth, use_boxed=use_boxed, extra_info=extra_info)

    score = (1.0 - format_score) * accuracy + format_score * formatting
    return {
        "score": score,
        "accuracy": accuracy,
        "format": formatting,
    }


def compute_score_from_data_source(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    use_boxed: bool = False,
    format_score: float = 0.5,
    **_: dict,
) -> dict[str, float]:
    """Adapter for custom_reward_function configs that expect default signature."""
    return compute_score(
        solution_str,
        ground_truth,
        use_boxed=use_boxed,
        format_score=format_score,
        extra_info=extra_info,
    )


async def async_compute_score_from_data_source(
    data_source: str,  # noqa: ARG001
    solution_str: str,
    ground_truth: str,
    extra_info=None,
    use_boxed: bool = False,
    format_score: float = 0.5,
    **_: dict,
) -> dict[str, float]:
    """Async adapter for reward loops."""

    return compute_score(
        solution_str,
        ground_truth,
        use_boxed=use_boxed,
        format_score=format_score,
        extra_info=extra_info,
    )
