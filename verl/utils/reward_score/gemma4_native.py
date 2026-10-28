"""Strict native Gemma solver scoring; numerical reward semantics are shared."""

import inspect

from verl.models.gemma4.protocol import parse_trajectory
from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import compute_score


def compute_score_from_data_source(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    options = {k: v for k, v in kwargs.items() if k in inspect.signature(compute_score).parameters}
    options["require_boxed_answer_for_full_format"] = False
    return compute_score(
        solution_str, ground_truth, extra_info=extra_info, parsed_output=parse_trajectory(solution_str), **options
    )


async def async_compute_score_from_data_source(**kwargs):
    return compute_score_from_data_source(**kwargs)
