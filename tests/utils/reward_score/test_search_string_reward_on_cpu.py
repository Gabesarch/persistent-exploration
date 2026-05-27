import pytest

from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
    compute_score_from_data_source,
)
from verl.utils.reward_score.text_normalization import search_string_reward


@pytest.mark.parametrize(
    ("predicted", "truth"),
    [
        ("4", "Four."),
        ("!", "!"),
        ("586-0380", "5860380"),
        ("fork and knife", "Knife and fork"),
        ("3\u00b794", "394"),
    ],
)
def test_search_string_reward_full_credit_equivalences(predicted, truth):
    assert search_string_reward(predicted, truth) == 1.0


@pytest.mark.parametrize(
    ("predicted", "truth", "expected"),
    [
        ("188", "888", 2 / 3),
        ("black", "white and black", 0.75),
        ("IDROTT'S BOKEN 1973", "IDROTTS BOKEN", 0.75),
    ],
)
def test_search_string_reward_partial_credit(predicted, truth, expected):
    assert search_string_reward(predicted, truth) == pytest.approx(expected)


@pytest.mark.parametrize(
    ("predicted", "truth"),
    [
        ("", "!"),
        ("A", "B"),
        ("abc", "xyz"),
    ],
)
def test_search_string_reward_zero_credit_guardrails(predicted, truth):
    assert search_string_reward(predicted, truth) == 0.0


@pytest.mark.parametrize(
    ("predicted", "truth", "max_score"),
    [
        ("IDROTTS BOKEN 1973 1974 1975", "IDROTTS BOKEN", 0.70),
        ("fork / knife / spoon / plate / cup", "fork and knife", 0.60),
        ("black black black black", "black", 0.60),
    ],
)
def test_search_string_reward_caps_overcredit(predicted, truth, max_score):
    score = search_string_reward(predicted, truth)
    assert 0.0 <= score <= max_score
    assert score < 1.0


def test_search_reward_integration_uses_partial_accuracy():
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str="<answer>188</answer>",
        ground_truth="888",
        extra_info={"reward_type": "search"},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["accuracy"] == pytest.approx(2 / 3)
    assert result["score"] == pytest.approx(0.2 + 0.8 * (2 / 3))
