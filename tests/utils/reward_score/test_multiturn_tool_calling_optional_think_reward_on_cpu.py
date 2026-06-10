import asyncio
import inspect

import pytest

from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
    async_compute_score_from_data_source,
    compute_score_from_data_source,
    format_reward,
)


def test_mandatory_think_accepts_think_wrapped_answer():
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
        "tool_efficiency": 1.0,
        "tool_call_bonus": 0.0,
    }


def test_mandatory_think_accepts_think_wrapped_tool_rounds_then_answer():
    solution = """
<think>I need a closer view of the legend.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 300, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>
The crop shows the legend labels.
</tool_response>
<think>The matching category is now clear.</think>
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
        "tool_efficiency": 0.5,
        "tool_call_bonus": 0.0,
    }


def test_mandatory_think_accepts_decoded_role_markers_around_tool_response():
    solution = """
<think>I need to zoom into the jersey painting.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [751, 187, 924, 298], "label": "painting of a baseball jersey", "img_idx": 0}}
</tool_call>user
<tool_response>
Zoomed in on the image to the region [751, 187, 924, 298] with label painting of a baseball jersey.
</tool_response>
assistant
<think>The words on the jersey are now readable.</think>
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
        "tool_efficiency": 0.5,
        "tool_call_bonus": 0.05,
    }


def test_mandatory_think_accepts_multiple_decoded_role_marker_tool_rounds():
    solution = """
<think>I need a closer view of the sign.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 300, 400], "label": "sign", "img_idx": 0}}
</tool_call>user
<tool_response>
The crop shows the left half of the sign.
</tool_response>
assistant
<think>I need the right half too.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [300, 20, 600, 400], "label": "right sign", "img_idx": 0}}
</tool_call>
user
<tool_response>
The crop shows GRAND.
</tool_response>assistant
<think>The answer is clear.</think>
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
        "tool_efficiency": pytest.approx(1.0 / 3.0),
        "tool_call_bonus": 0.05,
    }


def test_mandatory_think_async_entrypoint_matches_sync_reward():
    assert not inspect.iscoroutinefunction(compute_score_from_data_source)
    assert inspect.iscoroutinefunction(async_compute_score_from_data_source)

    kwargs = {
        "data_source": "qwen3vl_filtering",
        "solution_str": "<think>I can answer directly.</think><answer>It is \\boxed{6}.</answer>",
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
        "tool_efficiency": 1.0,
        "tool_call_bonus": 0.0,
    }


@pytest.mark.parametrize(
    "solution, expected_tool_calls",
    [
        ("plain reasoning only", 0),
        ("reasoning <answer>\\boxed{6}</answer>", 0),
        ("reasoning <tool_call>{\"name\": \"image_zoom_in_tool\", \"arguments\": {}}</tool_call>", 0),
        ("bare text <think>reason</think><answer>\\boxed{6}</answer>", 0),
        ("<think>reason</think> bare text <answer>\\boxed{6}</answer>", 0),
        ("<think>reason</think><answer>\\boxed{6}</answer> trailing text", 0),
        ("<think></think><answer>\\boxed{6}</answer>", 0),
        ("<think>reason</think><answer></answer>", 0),
        ("<think>reason</think><tool_call>{bad json}</tool_call>", 0),
        ("<think>reason</think><tool_call>{\"name\": \"image_zoom_in_tool\"}</tool_call>", 0),
        ("<think>reason</think><tool_call>{\"name\": \"image_zoom_in_tool\", \"arguments\": []}</tool_call>", 0),
        ("<think>reason <answer>x</answer></think><answer>\\boxed{6}</answer>", 0),
        ("<think>reason</think><tool_response>unexpected</tool_response><answer>\\boxed{6}</answer>", 0),
        (
            '<think>zoom</think><tool_call>{"name": "image_zoom_in_tool", "arguments": {}}</tool_call>'
            "<tool_response></tool_response>",
            0,
        ),
        (
            '<think>zoom</think><tool_call>{"name": "image_zoom_in_tool", "arguments": {}}</tool_call>'
            "<tool_response>ok</tool_response>bare reasoning<answer>\\boxed{6}</answer>",
            0,
        ),
        (
            '<think>zoom</think><tool_call>{"name": "image_zoom_in_tool", "arguments": {}}</tool_call>'
            "<tool_response>ok</tool_response><answer>\\boxed{6}</answer>",
            0,
        ),
        (
            '<think>zoom</think><tool_call>{"name": "image_zoom_in_tool", "arguments": {}}</tool_call>'
            "<tool_response>ok</tool_response><think>missing close<answer>\\boxed{6}</answer>",
            0,
        ),
    ],
)
def test_mandatory_think_invalid_formats_score_zero(solution, expected_tool_calls):
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
        "tool_call_count": expected_tool_calls,
        "tool_efficiency": 1.0,
        "tool_call_bonus": 0.0,
    }


def test_mandatory_think_decoded_role_markers_without_answer_remain_invalid():
    solution = (
        '<think>zoom</think><tool_call>{"name": "image_zoom_in_tool", "arguments": {}}</tool_call>user'
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
        "tool_efficiency": 0.5,
        "tool_call_bonus": 0.0,
    }


def test_mandatory_think_terminal_tool_error_without_answer_scores_zero():
    solution = """
<think>I need to zoom.</think>
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
        format_score=0.1,
    )
    assert result == {
        "score": 0.0,
        "accuracy": 0.0,
        "format": 0.0,
        "tool_call_count": 1,
        "tool_efficiency": 0.5,
        "tool_call_bonus": 0.0,
    }


def test_mandatory_think_tool_call_bonus_applies_at_threshold():
    solution = """
<think>I need a closer view.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 300, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>
The crop shows the legend labels.
</tool_response>
<think>The answer is clear.</think>
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
        "tool_efficiency": 0.5,
        "tool_call_bonus": 0.05,
    }


def test_mandatory_think_tool_call_bonus_requires_threshold():
    solution = """
<think>I need a closer view.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 300, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>
The crop shows the legend labels.
</tool_response>
<think>The answer is clear.</think>
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


def test_mandatory_think_tool_call_bonus_bypasses_when_disabled_or_zero():
    solution = """
<think>I need a closer view.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 300, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>
The crop shows the legend labels.
</tool_response>
<think>The answer is clear.</think>
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


def test_mandatory_think_tool_call_bonus_requires_valid_format():
    solution = '<think>reasoning</think><tool_call>{"name": "image_zoom_in_tool", "arguments": {}}</tool_call>'

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
        "tool_efficiency": 0.5,
        "tool_call_bonus": 0.0,
    }
