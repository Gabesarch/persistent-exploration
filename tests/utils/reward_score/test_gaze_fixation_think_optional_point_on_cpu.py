import pytest

from verl.utils.reward_score.math_verify_reward_type_fixations_think_optional_point import (
    compute_score_from_data_source,
    format_reward,
)


def _solution(answer: str) -> str:
    return f"<think>No grounded point here.</think> {answer}"


def test_optional_point_reward_correct_answer_without_points():
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=_solution(r"\boxed{6}"),
        ground_truth="6",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["score"] == 1.0


def test_optional_point_reward_thread_safe_numeric_literal():
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=_solution(r"\boxed{8}"),
        ground_truth="8",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["score"] == 1.0


def test_optional_point_reward_wrong_answer_keeps_format_score():
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=_solution(r"\boxed{7}"),
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
        r"\boxed{6}",
        r"preface <think>No grounded point here.</think> \boxed{6}",
        r"<think>Bad float [1.5, 2] in trace.</think> \boxed{6}",
        r"<think>Bad named coordinate x_min=1 in trace.</think> \boxed{6}",
        r"<think>No grounded point here.</think>",
    ],
)
def test_optional_point_reward_invalid_formats_score_zero(solution):
    assert format_reward(solution) == 0.0
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="6",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )
    assert result == {"score": 0.0, "accuracy": 0.0, "format": 0.0}
