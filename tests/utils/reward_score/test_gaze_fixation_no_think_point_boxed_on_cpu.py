import asyncio
import inspect

import pytest

from verl.utils.reward_score.math_verify_reward_type_fixations_no_think_point_boxed import (
    async_compute_score_from_data_source,
    compute_score_from_data_source,
    format_reward,
)


@pytest.mark.parametrize(
    ("reward_type", "solution", "truth"),
    [
        (
            "numeric",
            r"I inspect the tallest bar near [420, 310]. The answer is \boxed{6}",
            "6",
        ),
        (
            "string_match",
            r"<think>I check the legend at [112, 85].</think> The matching category is \boxed{Memory Care}",
            "memory care",
        ),
    ],
)
def test_no_think_point_boxed_reward_correct_answer_types(reward_type, solution, truth):
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth=truth,
        extra_info={"reward_type": reward_type},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["score"] == 1.0


def test_no_think_point_boxed_reward_thread_safe_numeric_literal():
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=r"I inspect the tallest bar near [420, 310]. The answer is \boxed{8}",
        ground_truth="8",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["score"] == 1.0


def test_no_think_point_boxed_async_entrypoint_is_coroutine_but_sync_entrypoint_stays_sync():
    assert not inspect.iscoroutinefunction(compute_score_from_data_source)
    assert inspect.iscoroutinefunction(async_compute_score_from_data_source)


def test_no_think_point_boxed_async_entrypoint_matches_sync_reward():
    kwargs = {
        "data_source": "qwen3vl_filtering",
        "solution_str": r"I inspect the midpoint near [420, 310]. The answer is \boxed{1/2}",
        "ground_truth": "0.5",
        "extra_info": {"reward_type": "numeric"},
        "format_score": 0.2,
    }

    sync_result = compute_score_from_data_source(**kwargs)
    async_result = asyncio.run(async_compute_score_from_data_source(**kwargs))

    assert async_result == sync_result == {"score": 1.0, "accuracy": 1.0, "format": 1.0}


def test_no_think_point_boxed_reward_wrong_answer_keeps_format_score():
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=r"I inspect the tallest bar near [420, 310]. The answer is \boxed{7}",
        ground_truth="6",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["accuracy"] == 0.0
    assert result["score"] == pytest.approx(0.2)


@pytest.mark.parametrize(
    "solution",
    [
        r"I inspect the chart and conclude \boxed{6}",
        r"I inspect [1.5, 2] and conclude \boxed{6}",
        r"I inspect [1, 2, 3] and conclude \boxed{6}",
        r"I inspect x_min=1 and conclude \boxed{6}",
        r"I inspect [420, 310] and conclude 6",
        r"I inspect [420, 310] and conclude \boxed{}",
        r"I inspect [420, 310] and conclude \boxed[6]",
        r"I inspect the chart. The answer is \boxed{[420, 310]}",
        r"I inspect [420, 310] and conclude \boxed{6} then add text",
        r"I inspect [420, 310] and conclude \boxed{6} or \boxed{7}",
    ],
)
def test_no_think_point_boxed_reward_invalid_formats_score_zero(solution):
    assert format_reward(solution) == 0.0
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="6",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )
    assert result == {"score": 0.0, "accuracy": 0.0, "format": 0.0}
