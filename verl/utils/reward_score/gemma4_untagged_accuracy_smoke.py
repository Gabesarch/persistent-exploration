"""Smoke-only accuracy relaxation; format and all other rewards remain strict."""

import inspect
import re

from verl.models.gemma4 import protocol as native
from verl.utils.reward_score.gemma4_native import compute_score_from_data_source as strict_score
from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import compute_score


def parse_accuracy_trajectory(text):
    """Accept a complete native final without answer tags, never feedback/thoughts.

    A temporary parsing view delegates history validation to the strict parser.
    Generated IDs, log probabilities, training text and format scoring are untouched.
    Partial tags and incomplete turns remain invalid.
    """
    strict = native.parse_trajectory(text)
    if strict[0]:
        return strict
    offset = 0
    for turn in text.split(native.MODEL):
        thought = re.match(r"\s*<\|channel>thought\n(.*?)<channel\|>", turn, re.S)
        if thought is None or not thought[1].strip():
            return strict
        action = turn[thought.end() :]
        if action.lstrip().startswith("<|tool_call>"):
            offset += len(turn) + len(native.MODEL)
            continue
        final = re.fullmatch(r"\s*(.*?)\s*(<turn\|>|<eos>)\s*(?:<\|turn>user\n.*)?", action, re.S)
        answer = final[1].strip() if final else ""
        if not answer or re.search(r"<\||<[^>]*\||</?answer\b|<eos>", answer):
            return strict
        parsing_view = text[: offset + thought.end()] + f"<answer>{answer}</answer>{final[2]}"
        return native.parse_trajectory(parsing_view)
    return strict


def compute_score_from_data_source(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    result = strict_score(data_source, solution_str, ground_truth, extra_info=extra_info, **kwargs)
    parsed = parse_accuracy_trajectory(solution_str)
    if not parsed[0] or native.parse_trajectory(solution_str)[0]:
        return result
    options = {k: v for k, v in kwargs.items() if k in inspect.signature(compute_score).parameters}
    options["require_boxed_answer_for_full_format"] = False
    relaxed = compute_score(solution_str, ground_truth, extra_info=extra_info, parsed_output=parsed, **options)
    accuracy_weight = (
        1.0 - float(options.get("format_score", 0.5)) - float(options.get("tool_efficiency_score_weight", 0.0))
    )
    result["score"] += accuracy_weight * (relaxed["accuracy"] - result["accuracy"])
    result["accuracy"] = relaxed["accuracy"]
    return result


async def async_compute_score_from_data_source(**kwargs):
    return compute_score_from_data_source(**kwargs)
