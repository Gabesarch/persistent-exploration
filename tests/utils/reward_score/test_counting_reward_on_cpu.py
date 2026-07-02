import pytest

from verl.utils.reward_score.math_verify_reward_type_boxed import (
    _counting_reward,
    counting_reward_value,
)
from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
    compute_score_from_data_source,
)


def test_exact_count_gets_full_reward():
    assert _counting_reward("7", "7", extra_info={"reward_type": "counting"}) == 1.0


def test_small_count_stays_strict_despite_large_absolute_tolerance():
    extra_info = {"reward_type": "counting", "tolerance": 2}
    assert _counting_reward("1", "1", extra_info=extra_info) == 1.0
    assert _counting_reward("0", "1", extra_info=extra_info) < 0.1
    assert _counting_reward("2", "1", extra_info=extra_info) < 0.1


def test_intrinsic_relative_band_gives_full_credit_on_large_counts():
    extra_info = {"reward_type": "counting"}
    assert _counting_reward("101", "100", extra_info=extra_info) == 1.0
    assert _counting_reward("105", "100", extra_info=extra_info) == 1.0
    assert _counting_reward("106", "100", extra_info=extra_info) < 1.0


def test_intrinsic_band_scales_with_ground_truth():
    # floor(0.05 * gt): 0 below 20, so off-by-one only earns full credit from gt=20 up.
    assert counting_reward_value(20, 19) < 1.0
    assert counting_reward_value(21, 20) == 1.0
    assert counting_reward_value(22, 20) < 1.0


def test_row_tolerance_widens_band_only_up_to_relative_cap():
    extra_info = {"reward_type": "counting", "tolerance": 10}
    assert _counting_reward("110", "100", extra_info=extra_info) == 1.0
    assert _counting_reward("111", "100", extra_info=extra_info) < 1.0
    extra_info = {"reward_type": "counting", "tolerance": 50}
    assert _counting_reward("115", "100", extra_info=extra_info) == 1.0
    assert _counting_reward("116", "100", extra_info=extra_info) < 1.0


def test_reward_decays_monotonically_with_error_beyond_band():
    rewards = [counting_reward_value(pred, 100) for pred in (105, 106, 110, 120, 150)]
    assert rewards[0] == 1.0
    assert all(a > b for a, b in zip(rewards, rewards[1:]))
    assert rewards[-1] < 0.01


def test_relative_scale_makes_same_absolute_error_score_higher_on_larger_counts():
    assert counting_reward_value(21, 20) > counting_reward_value(3, 2)


def test_zero_ground_truth_requires_exact_answer():
    assert counting_reward_value(0, 0) == 1.0
    assert counting_reward_value(1, 0) < 0.05


def test_number_words_and_thousand_separators_parse():
    assert _counting_reward("seven", "7", extra_info={"reward_type": "counting"}) == 1.0
    assert _counting_reward("1,234", "1234", extra_info={"reward_type": "counting"}) == 1.0


def test_non_numeric_prediction_scores_zero():
    assert _counting_reward("about 7", "7", extra_info={"reward_type": "counting"}) == 0.0


def test_non_numeric_ground_truth_falls_back_to_string_match():
    extra_info = {"reward_type": "counting"}
    assert _counting_reward("many", "many", extra_info=extra_info) == 1.0
    assert _counting_reward("few", "many", extra_info=extra_info) == 0.0


def test_multiturn_compute_score_uses_graded_counting_reward():
    solution = "<think>Counting the objects.</think><answer>\\boxed{101}</answer>"
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="100",
        extra_info={"reward_type": "counting"},
        format_score=0.1,
    )
    assert result["accuracy"] == 1.0

    solution = "<think>Counting the objects.</think><answer>\\boxed{0}</answer>"
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="1",
        extra_info={"reward_type": "counting", "tolerance": 2},
        format_score=0.1,
    )
    assert result["accuracy"] < 0.1
