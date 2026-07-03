import pytest

from verl.utils.reward_score import math_verify_reward_type_boxed as boxed
from verl.utils.reward_score.math_verify_reward_type_boxed import (
    _COUNTING_DECAY_EXP,
    _COUNTING_MIN_SIGMA,
    _COUNTING_REL_BAND_CAP,
    _COUNTING_REL_SIGMA,
    _COUNTING_REL_TOL,
    _counting_reward,
    counting_reward_value,
)
from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
    _build_counting_reward_kwargs,
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


# --- Reference values (harness sanity checks; see TASK.md / counting_reward_formula.md) ---


@pytest.mark.parametrize(
    "pred, gt, expected",
    [
        (110, 100, 0.8948),  # TASK.md quotes ~0.895
        (8, 7, 0.4037),  # TASK.md quotes ~0.40
        (0, 1, 0.0183),  # TASK.md quotes ~0.018
    ],
)
def test_reference_values_reproduce_to_three_decimals(pred, gt, expected):
    assert counting_reward_value(pred, gt) == pytest.approx(expected, abs=5e-4)


# --- Phase B: the knobs are plumbed but defaults stay bit-identical ---


def test_module_constants_have_documented_defaults():
    assert _COUNTING_REL_TOL == 0.05
    assert _COUNTING_REL_BAND_CAP == 0.15
    assert _COUNTING_REL_SIGMA == 0.15
    assert _COUNTING_MIN_SIGMA == 0.5
    assert _COUNTING_DECAY_EXP == 2.0


def test_defaults_equal_module_constants_across_grid():
    # An unadorned call must equal an explicit call that passes the module constants,
    # over the band, the decay, and the zero-gt edge -- the "no override => shipped
    # behavior" guarantee (TASK.md Verify #3).
    for gt in (0, 1, 2, 7, 19, 20, 50, 100, 512, 1912):
        for delta in (0, 1, 2, 3, 5, 8, 13, 40, 200):
            for pred in {gt + delta, max(gt - delta, 0)}:
                default = counting_reward_value(pred, gt)
                explicit = counting_reward_value(
                    pred,
                    gt,
                    rel_tol=_COUNTING_REL_TOL,
                    rel_band_cap=_COUNTING_REL_BAND_CAP,
                    rel_sigma=_COUNTING_REL_SIGMA,
                    min_sigma=_COUNTING_MIN_SIGMA,
                    decay_exp=_COUNTING_DECAY_EXP,
                    graded=True,
                )
                assert default == explicit


def test_none_knobs_fall_back_to_constants():
    # Passing None for every knob is the same as passing nothing.
    assert counting_reward_value(
        110, 100, rel_tol=None, rel_band_cap=None, rel_sigma=None, min_sigma=None, decay_exp=None
    ) == counting_reward_value(110, 100)


def test_build_counting_reward_kwargs_empty_when_all_default():
    # Empty dict => acc_reward never forwards it => bit-identical counting path.
    assert _build_counting_reward_kwargs() == {}
    assert _build_counting_reward_kwargs(
        rel_tol=None, rel_band_cap=None, rel_sigma=None, min_sigma=None, decay_exp=None, graded=True
    ) == {}
    assert _build_counting_reward_kwargs(rel_tol=0.1, graded=False) == {"rel_tol": 0.1, "graded": False}


# --- Phase B: the knobs actually change the reward when overridden ---


def test_rel_tol_override_widens_band():
    # gt=100 default band = floor(0.05*100) = 5, so pred=106 is outside.
    assert counting_reward_value(106, 100) < 1.0
    # rel_tol=0.10 -> band = floor(0.10*100) = 10, so pred=106..110 get full credit.
    assert counting_reward_value(106, 100, rel_tol=0.10) == 1.0
    assert counting_reward_value(110, 100, rel_tol=0.10) == 1.0
    assert counting_reward_value(111, 100, rel_tol=0.10) < 1.0
    # rel_tol=0 -> no intrinsic band, only exact answers get full credit.
    assert counting_reward_value(101, 100, rel_tol=0.0) < 1.0


def test_rel_sigma_and_min_sigma_control_decay_width():
    # Wider relative sigma => a fixed absolute error decays less.
    narrow = counting_reward_value(120, 100, rel_sigma=0.05)
    wide = counting_reward_value(120, 100, rel_sigma=0.30)
    assert wide > narrow
    # min_sigma is the floor that matters for small counts (rel_sigma*gt is tiny).
    tight = counting_reward_value(3, 1, min_sigma=0.25)
    loose = counting_reward_value(3, 1, min_sigma=1.0)
    assert loose > tight


def test_band_only_mode_is_hard_zero_outside_band():
    # graded=False collapses the Gaussian tail to a hard band.
    assert counting_reward_value(100, 100, graded=False) == 1.0
    assert counting_reward_value(105, 100, graded=False) == 1.0  # inside band (floor(0.05*100)=5)
    assert counting_reward_value(106, 100, graded=False) == 0.0  # just outside band
    assert counting_reward_value(150, 100, graded=False) == 0.0


def test_laplacian_decay_has_fatter_tail_than_gaussian():
    # exp 1 (Laplacian) decays slower than exp 2 (Gaussian) far outside the band.
    gaussian = counting_reward_value(160, 100, decay_exp=2.0)
    laplacian = counting_reward_value(160, 100, decay_exp=1.0)
    assert laplacian > gaussian


def _counting_solution(answer: str) -> str:
    return f"<think>Counting.</think><answer>\\boxed{{{answer}}}</answer>"


def test_compute_score_threads_counting_kwargs_end_to_end():
    # Default: gt=100 pred=106 is outside the band, so accuracy < 1.
    baseline = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=_counting_solution("106"),
        ground_truth="100",
        extra_info={"reward_type": "counting"},
        format_score=0.1,
    )
    assert baseline["accuracy"] < 1.0
    # Widening rel_tol via reward_kwargs must reach counting_reward_value.
    widened = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=_counting_solution("106"),
        ground_truth="100",
        extra_info={"reward_type": "counting"},
        format_score=0.1,
        counting_rel_tol=0.10,
    )
    assert widened["accuracy"] == 1.0


def test_compute_score_counting_sigma_matches_direct_call():
    # The end-to-end accuracy for a graded (decayed) answer must equal a direct
    # counting_reward_value call with the same knob.
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=_counting_solution("120"),
        ground_truth="100",
        extra_info={"reward_type": "counting"},
        format_score=0.1,
        counting_rel_sigma=0.30,
    )
    assert result["accuracy"] == pytest.approx(counting_reward_value(120, 100, rel_sigma=0.30))


def test_compute_score_no_counting_kwargs_is_unchanged():
    # No overrides => the counting accuracy equals the shipped counting_reward_value.
    for gt, pred in [("100", "110"), ("7", "8"), ("20", "22"), ("1", "0")]:
        result = compute_score_from_data_source(
            data_source="qwen3vl_filtering",
            solution_str=_counting_solution(pred),
            ground_truth=gt,
            extra_info={"reward_type": "counting"},
            format_score=0.1,
        )
        assert result["accuracy"] == pytest.approx(counting_reward_value(int(pred), int(gt)))


def test_band_only_mode_end_to_end():
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=_counting_solution("150"),
        ground_truth="100",
        extra_info={"reward_type": "counting"},
        format_score=0.1,
        counting_graded=False,
    )
    assert result["accuracy"] == 0.0
