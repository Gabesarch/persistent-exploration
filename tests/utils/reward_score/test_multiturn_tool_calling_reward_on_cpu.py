import asyncio
import inspect

import pytest

from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling import (
    async_compute_score_from_data_source,
    compute_score_from_data_source,
    format_reward,
)


def test_multiturn_tool_calling_single_round_answer_scores():
    solution = (
        "<think>I can read the label directly.</think>\n"
        "<answer>The requested value is \\boxed{6}.</answer>"
    )

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="6",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )

    assert result == {"score": 1.0, "accuracy": 1.0, "format": 1.0}


def test_multiturn_tool_calling_multiple_tool_rounds_then_answer_scores():
    solution = """
<think>I need a closer view of the legend.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [10, 20, 300, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>
The crop shows the legend labels.
</tool_response>
<think>I need the bar label too.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [300, 100, 700, 500], "label": "bar", "img_idx": 0}}
</tool_call>
<tool_response>
The crop shows Memory Care.
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

    assert result == {"score": 1.0, "accuracy": 1.0, "format": 1.0}


def test_multiturn_tool_calling_async_entrypoint_matches_sync_reward():
    assert not inspect.iscoroutinefunction(compute_score_from_data_source)
    assert inspect.iscoroutinefunction(async_compute_score_from_data_source)

    kwargs = {
        "data_source": "qwen3vl_filtering",
        "solution_str": "<think>I can answer directly.</think><answer>It is \\boxed{A}.</answer>",
        "ground_truth": "A",
        "extra_info": {"reward_type": "multiple_choice"},
        "format_score": 0.2,
    }

    sync_result = compute_score_from_data_source(**kwargs)
    async_result = asyncio.run(async_compute_score_from_data_source(**kwargs))

    assert async_result == sync_result == {"score": 1.0, "accuracy": 1.0, "format": 1.0}


@pytest.mark.parametrize(
    "solution",
    [
        "bare text <think>reason</think><answer>\\boxed{6}</answer>",
        "<think></think><answer>\\boxed{6}</answer>",
        "<think>reason</think><answer></answer>",
        "<think>reason</think><answer>\\boxed{6}</answer><think>more</think>",
        "<think>reason</think><tool_call>{bad json}</tool_call>",
        "<think>reason</think><tool_call>{\"name\": \"image_zoom_in_tool\"}</tool_call>",
        "<think>reason</think><tool_call>{\"arguments\": {}}</tool_call>",
        "<think>reason</think><tool_call>{\"name\": \"image_zoom_in_tool\", \"arguments\": []}</tool_call>",
        "<think>reason <answer>x</answer></think><answer>\\boxed{6}</answer>",
        "<think>reason</think><tool_call>{\"name\": \"image_zoom_in_tool\", \"arguments\": {}}</tool_call>",
        (
            "<think>reason</think>"
            '<tool_call>{"name": "image_zoom_in_tool", "arguments": {}}</tool_call>'
            "<tool_response></tool_response>"
        ),
    ],
)
def test_multiturn_tool_calling_invalid_formats_score_zero(solution):
    assert format_reward(solution) == 0.0
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="6",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )
    assert result == {"score": 0.0, "accuracy": 0.0, "format": 0.0}


def test_multiturn_tool_calling_wrong_answer_keeps_format_score():
    solution = "<think>I can read the label directly.</think><answer>The value is \\boxed{7}.</answer>"

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="6",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["accuracy"] == 0.0
    assert result["score"] == pytest.approx(0.2)
