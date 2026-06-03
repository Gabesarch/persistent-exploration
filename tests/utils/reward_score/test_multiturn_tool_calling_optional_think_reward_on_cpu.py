import asyncio
import inspect

import pytest

from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
    async_compute_score_from_data_source,
    compute_score_from_data_source,
    format_reward,
)


def test_optional_think_accepts_plain_reasoning_then_answer():
    solution = (
        "The image shows a white table with a wire basket, but I do not see the requested yellow basket.\n\n"
        "<answer>\nThere is no yellow basket on the left table in the image.\n</answer>"
    )

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="Spoons and forks.",
        extra_info={"reward_type": "search"},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["accuracy"] == 0.0
    assert result["score"] == pytest.approx(0.2)


def test_optional_think_accepts_plain_reasoning_tool_rounds_then_answer():
    solution = """
I need a closer view of the legend.
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 300, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>
The crop shows the legend labels.
</tool_response>
The matching category is now clear.
<answer>The matching category is \\boxed{Memory Care}.</answer>
""".strip()

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="memory care",
        extra_info={"reward_type": "string_match"},
        format_score=0.2,
    )

    assert result == {
        "score": 1.0,
        "accuracy": 1.0,
        "format": 1.0,
        "tool_call_count": 1,
        "tool_call_bonus": 0.0,
    }


def test_optional_think_accepts_decoded_role_markers_around_tool_response():
    solution = """
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [751, 187, 924, 298], "label": "painting of a baseball jersey", "img_idx": 0}}
</tool_call>user
<tool_response>
Zoomed in on the image to the region [751, 187, 924, 298] with label painting of a baseball jersey.
</tool_response>
assistant
<answer>
BALL CLUB
</answer>
""".strip()

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="BALL CLUB.",
        extra_info={"reward_type": "search"},
        format_score=0.2,
        min_tool_calls_for_bonus=1,
        tool_call_bonus=0.05,
    )

    assert result == {
        "score": pytest.approx(1.05),
        "accuracy": 1.0,
        "format": 1.0,
        "tool_call_count": 1,
        "tool_call_bonus": 0.05,
    }


def test_optional_think_accepts_decoded_assistant_marker_before_reasoning_and_answer():
    solution = """
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [403, 138, 563, 225], "label": "banner with text", "img_idx": 0}}
</tool_call>user
<tool_response>
Zoomed in on the image to the region [403, 138, 563, 225] with label banner with text.
</tool_response>
assistant
Based on the zoomed-in image, the word to the right of KOMPAS is visible.

<answer>
citybeats
</answer>
""".strip()

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="citybeats",
        extra_info={"reward_type": "search"},
        format_score=0.2,
    )

    assert result == {
        "score": 1.0,
        "accuracy": 1.0,
        "format": 1.0,
        "tool_call_count": 1,
        "tool_call_bonus": 0.0,
    }


def test_optional_think_accepts_multiple_decoded_role_marker_tool_rounds():
    solution = """
I need a closer view of the sign.
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 300, 400], "label": "sign", "img_idx": 0}}
</tool_call>user
<tool_response>
The crop shows the left half of the sign.
</tool_response>
assistant
I need the right half too.
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [300, 20, 600, 400], "label": "right sign", "img_idx": 0}}
</tool_call>
user
<tool_response>
The crop shows GRAND.
</tool_response>assistant
The answer is clear.
<answer>GRAND</answer>
""".strip()

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="GRAND",
        extra_info={"reward_type": "search"},
        format_score=0.2,
        min_tool_calls_for_bonus=2,
        tool_call_bonus=0.05,
    )

    assert result == {
        "score": pytest.approx(1.05),
        "accuracy": 1.0,
        "format": 1.0,
        "tool_call_count": 2,
        "tool_call_bonus": 0.05,
    }


def test_optional_think_still_accepts_think_wrapped_rounds():
    solution = "<think>I can answer directly.</think><answer>It is \\boxed{A}.</answer>"

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="A",
        extra_info={"reward_type": "multiple_choice"},
        format_score=0.2,
    )

    assert result == {
        "score": 1.0,
        "accuracy": 1.0,
        "format": 1.0,
        "tool_call_count": 0,
        "tool_call_bonus": 0.0,
    }


def test_optional_think_async_entrypoint_matches_sync_reward():
    assert not inspect.iscoroutinefunction(compute_score_from_data_source)
    assert inspect.iscoroutinefunction(async_compute_score_from_data_source)

    kwargs = {
        "data_source": "qwen3vl_filtering",
        "solution_str": "I can answer directly.<answer>It is \\boxed{6}.</answer>",
        "ground_truth": "6",
        "extra_info": {"reward_type": "numeric"},
        "format_score": 0.2,
    }

    sync_result = compute_score_from_data_source(**kwargs)
    async_result = asyncio.run(async_compute_score_from_data_source(**kwargs))

    assert async_result == sync_result == {
        "score": 1.0,
        "accuracy": 1.0,
        "format": 1.0,
        "tool_call_count": 0,
        "tool_call_bonus": 0.0,
    }


@pytest.mark.parametrize(
    "solution",
    [
        "plain reasoning only",
        "reasoning <tool_call>{bad json}</tool_call>",
        "reasoning <tool_call>{\"name\": \"image_zoom_in_tool\"}</tool_call>",
        "reasoning <tool_call>{\"name\": \"image_zoom_in_tool\", \"arguments\": []}</tool_call>",
        "reasoning <answer></answer>",
        (
            '<tool_call>{"name": "image_zoom_in_tool", "arguments": {}}</tool_call>user'
            "<tool_response>ok</tool_response>assistant <answer>missing close"
        ),
        (
            "<tool_call>{bad json}</tool_call>user"
            "<tool_response>ok</tool_response>assistant<answer>6</answer>"
        ),
        "reasoning <answer>\\boxed{6}</answer> trailing text",
        "reasoning <answer>\\boxed{6}</answer><tool_call>{\"name\": \"x\", \"arguments\": {}}</tool_call>",
        "<think></think><answer>\\boxed{6}</answer>",
        "reasoning </think><answer>\\boxed{6}</answer>",
        "reasoning <tool_response>unexpected</tool_response><answer>\\boxed{6}</answer>",
    ],
)
def test_optional_think_invalid_formats_score_zero(solution):
    assert format_reward(solution) == 0.0
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="6",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )
    assert result == {
        "score": 0.0,
        "accuracy": 0.0,
        "format": 0.0,
        "tool_call_count": 0,
        "tool_call_bonus": 0.0,
    }


def test_optional_think_decoded_role_markers_without_answer_remain_invalid():
    solution = (
        '<tool_call>{"name": "image_zoom_in_tool", "arguments": {}}</tool_call>user'
        "<tool_response>ok</tool_response>assistant"
    )

    assert format_reward(solution) == 0.0
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="6",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )
    assert result == {
        "score": 0.0,
        "accuracy": 0.0,
        "format": 0.0,
        "tool_call_count": 1,
        "tool_call_bonus": 0.0,
    }


def test_optional_think_terminal_tool_error_without_answer_scores_zero():
    solution = """
I need to zoom.
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 7000, 20], "label": "line", "img_idx": 0}}
</tool_call>
<tool_response>
Error processing image zoom-in: cropped image aspect ratio 300.0 exceeds max_cropped_aspect_ratio=199.0
</tool_response>
""".strip()

    assert format_reward(solution) == 0.0
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="6",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )
    assert result == {
        "score": 0.0,
        "accuracy": 0.0,
        "format": 0.0,
        "tool_call_count": 1,
        "tool_efficiency": 0.5,
        "tool_call_bonus": 0.0,
    }


def test_optional_think_tool_call_bonus_applies_at_threshold():
    solution = """
I need a closer view.
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 300, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>
The crop shows the legend labels.
</tool_response>
The answer is clear.
<answer>The matching category is \\boxed{Memory Care}.</answer>
""".strip()

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="memory care",
        extra_info={"reward_type": "string_match"},
        format_score=0.2,
        min_tool_calls_for_bonus=1,
        tool_call_bonus=0.05,
    )

    assert result == {
        "score": pytest.approx(1.05),
        "accuracy": 1.0,
        "format": 1.0,
        "tool_call_count": 1,
        "tool_call_bonus": 0.05,
    }


def test_optional_think_tool_call_bonus_requires_threshold():
    solution = """
I need a closer view.
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 300, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>
The crop shows the legend labels.
</tool_response>
The answer is clear.
<answer>The matching category is \\boxed{Memory Care}.</answer>
""".strip()

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="memory care",
        extra_info={"reward_type": "string_match"},
        format_score=0.2,
        min_tool_calls_for_bonus=2,
        tool_call_bonus=0.05,
    )

    assert result["score"] == pytest.approx(1.0)
    assert result["tool_call_count"] == 1
    assert result["tool_call_bonus"] == 0.0


def test_optional_think_tool_call_bonus_bypasses_when_disabled_or_zero():
    solution = """
I need a closer view.
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 300, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>
The crop shows the legend labels.
</tool_response>
The answer is clear.
<answer>The matching category is \\boxed{Memory Care}.</answer>
""".strip()

    disabled_by_threshold = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="memory care",
        extra_info={"reward_type": "string_match"},
        format_score=0.2,
        min_tool_calls_for_bonus=None,
        tool_call_bonus=0.05,
    )
    disabled_by_zero_bonus = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="memory care",
        extra_info={"reward_type": "string_match"},
        format_score=0.2,
        min_tool_calls_for_bonus=1,
        tool_call_bonus=0.0,
    )

    assert disabled_by_threshold["score"] == pytest.approx(1.0)
    assert disabled_by_threshold["tool_call_bonus"] == 0.0
    assert disabled_by_zero_bonus["score"] == pytest.approx(1.0)
    assert disabled_by_zero_bonus["tool_call_bonus"] == 0.0


def test_optional_think_tool_call_bonus_requires_valid_format():
    solution = 'reasoning <tool_call>{"name": "image_zoom_in_tool", "arguments": {}}</tool_call>'

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="6",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
        min_tool_calls_for_bonus=1,
        tool_call_bonus=0.05,
    )

    assert result == {
        "score": 0.0,
        "accuracy": 0.0,
        "format": 0.0,
        "tool_call_count": 1,
        "tool_call_bonus": 0.0,
    }
