import asyncio
import inspect
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.reward_score.math_verify_reward_type_boxed import grade_numeric
from verl.utils.reward_score.math_verify_reward_type_fixations_think_nonpoint import (
    acc_reward,
    async_compute_score_from_data_source,
    compute_score_from_data_source,
    format_reward,
)


def _solution(answer: str) -> str:
    return f"<think>Plan at [1, 2] then verify the mark at [3, 4] carefully.</think> {answer}"


def test_grade_numeric_exact_literals_on_main_thread():
    assert grade_numeric("8", "8", tolerance=None) == 1


def test_grade_numeric_exact_literals_in_thread_executor():
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(grade_numeric, "8", "8", 6, None).result()

    assert result == 1


def test_grade_numeric_escaped_percent_literal_in_thread_executor():
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(grade_numeric, "27.2", r"27.2\%", 6, None).result()

    assert result == 1


def test_grade_numeric_tolerance_path_still_works():
    assert grade_numeric("30500", "31000", tolerance=0.05) == 1


@pytest.mark.parametrize(
    ("reward_type", "prediction", "truth"),
    [
        ("numeric", r"\boxed{6}", "6"),
        ("string_match", r"\boxed{Memory Care}", "memory care"),
        ("short_string", r"\boxed{Alpha}", "alpha"),
        ("multiple_choice", r"\boxed{C}", "C"),
        ("counting", r"\boxed{3}", "3"),
        ("search", r"\boxed{left panel}", "left panel"),
    ],
)
def test_gaze_fixation_reward_correct_answer_types(reward_type, prediction, truth):
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=_solution(prediction),
        ground_truth=truth,
        extra_info={"reward_type": reward_type},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["accuracy"] == 1.0
    assert result["score"] == 1.0


def test_gaze_fixation_reward_thread_safe_numeric_literal():
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


def test_gaze_fixation_reward_thread_safe_numeric_fraction():
    kwargs = {
        "data_source": "qwen3vl_filtering",
        "solution_str": _solution(r"\boxed{1/2}"),
        "ground_truth": "0.5",
        "extra_info": {"reward_type": "numeric"},
        "format_score": 0.2,
    }
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(lambda: compute_score_from_data_source(**kwargs)).result()

    assert result == {"score": 1.0, "accuracy": 1.0, "format": 1.0}


def test_async_reward_entrypoint_is_coroutine_but_sync_entrypoint_stays_sync():
    assert not inspect.iscoroutinefunction(compute_score_from_data_source)
    assert inspect.iscoroutinefunction(async_compute_score_from_data_source)


def test_async_reward_entrypoint_matches_sync_reward():
    kwargs = {
        "data_source": "qwen3vl_filtering",
        "solution_str": _solution(r"\boxed{1/2}"),
        "ground_truth": "0.5",
        "extra_info": {"reward_type": "numeric"},
        "format_score": 0.2,
    }

    sync_result = compute_score_from_data_source(**kwargs)
    async_result = asyncio.run(async_compute_score_from_data_source(**kwargs))

    assert async_result == sync_result == {"score": 1.0, "accuracy": 1.0, "format": 1.0}


def test_reward_loader_preserves_async_custom_reward_detection():
    module_path = Path(__file__).parents[3] / "verl/utils/reward_score/math_verify_reward_type_fixations_think_nonpoint.py"
    config = OmegaConf.create(
        {
            "reward": {
                "custom_reward_function": {
                    "path": str(module_path),
                    "name": "async_compute_score_from_data_source",
                    "reward_kwargs": {"format_score": 0.2},
                }
            }
        }
    )

    reward_fn = get_custom_reward_fn(config)

    assert inspect.iscoroutinefunction(reward_fn)
    result = asyncio.run(
        reward_fn(
            data_source="qwen3vl_filtering",
            solution_str=_solution(r"\boxed{6}"),
            ground_truth="6",
            extra_info={"reward_type": "numeric"},
        )
    )
    assert result == {"score": 1.0, "accuracy": 1.0, "format": 1.0}


def test_gaze_fixation_reward_valid_format_wrong_answer_keeps_format_score():
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
        r"<think>No grounded point here.</think> \boxed{6}",
        r"<think>[1, 2] [3, 4]</think> \boxed{6}",
        r"<think>Bad float [1.5, 2] then text [3, 4].</think> \boxed{6}",
        r"preface <think>Look [1, 2] then [3, 4].</think> \boxed{6}",
    ],
)
def test_gaze_fixation_reward_invalid_formats_score_zero(solution):
    assert format_reward(solution) == 0.0
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="6",
        extra_info={"reward_type": "numeric"},
        format_score=0.2,
    )
    assert result == {"score": 0.0, "accuracy": 0.0, "format": 0.0}


def test_gaze_fixation_grounding_reward_exact_bbox():
    prediction = '[{"bbox_2d": [100, 100, 200, 200]}]'
    truth = '[{"bbox_2d": [100, 100, 200, 200]}]'
    extra_info = {
        "reward_type": "grounding",
        "image_sizes": [(1000, 1000)],
        "normalize_bbox_to_1000": True,
    }

    assert acc_reward(_solution(prediction), truth, extra_info=extra_info) == 1.0


def test_gaze_fixation_reward_missing_reward_type_scores_zero():
    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=_solution(r"\boxed{6}"),
        ground_truth="6",
        extra_info={},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["accuracy"] == 0.0
    assert result["score"] == pytest.approx(0.2)
