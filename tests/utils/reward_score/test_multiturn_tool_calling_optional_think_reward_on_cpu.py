import asyncio
import inspect
import json

import pytest

from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
    assistant_turn_overlong_reward,
    async_compute_score_from_data_source,
    compute_score_from_data_source,
    extract_answer_for_accuracy,
    format_reward,
    point_box_reward,
    tool_repeat_overlap_call_diagnostics,
    tool_repeat_overlap_reward,
)


def zoom_call(bbox, img_idx=0):
    return {
        "name": "image_zoom_in_tool",
        "arguments": {"bbox_2d": bbox, "label": "region", "img_idx": img_idx},
    }


@pytest.mark.parametrize(
    ("assistant_turns", "expected"),
    [(0, 0.0), (9, 0.0), (10, -0.5), (11, -1.0)],
)
def test_assistant_turn_overlong_reward_ramps_over_final_two_turns(assistant_turns, expected):
    assert assistant_turn_overlong_reward(
        assistant_turns,
        max_turns=11,
        buffer_turns=2,
        penalty_factor=1.0,
    ) == pytest.approx(expected)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"assistant_turns": 11.0, "max_turns": 11, "buffer_turns": 2},
        {"assistant_turns": 12, "max_turns": 11, "buffer_turns": 2},
        {"assistant_turns": 11, "max_turns": 0, "buffer_turns": 2},
        {"assistant_turns": 11, "max_turns": 11, "buffer_turns": 0},
        {"assistant_turns": 11, "max_turns": 2, "buffer_turns": 3},
        {"assistant_turns": 11, "max_turns": 11, "buffer_turns": 2, "penalty_factor": 0.0},
    ],
)
def test_assistant_turn_overlong_reward_rejects_invalid_inputs(kwargs):
    with pytest.raises(ValueError):
        assistant_turn_overlong_reward(**kwargs)


def test_assistant_turn_overlong_component_requires_metadata_when_enabled():
    with pytest.raises(ValueError, match="assistant_turns"):
        compute_score_from_data_source(
            data_source="unit",
            solution_str="<think>done</think><answer>A</answer>",
            ground_truth="A",
            extra_info={"reward_type": "multiple_choice"},
            assistant_turn_overlong_enabled=True,
            assistant_turn_overlong_max_turns=11,
            assistant_turn_overlong_buffer_turns=2,
        )


def test_assistant_turn_overlong_component_is_returned_without_changing_scalar_score():
    kwargs = {
        "data_source": "unit",
        "solution_str": "<think>done</think><answer>A</answer>",
        "ground_truth": "A",
        "extra_info": {"reward_type": "multiple_choice", "assistant_turns": 10},
        "assistant_turn_overlong_enabled": True,
        "assistant_turn_overlong_max_turns": 11,
        "assistant_turn_overlong_buffer_turns": 2,
        "assistant_turn_overlong_penalty_factor": 1.0,
    }

    sync_result = compute_score_from_data_source(**kwargs)
    async_result = asyncio.run(async_compute_score_from_data_source(**kwargs))

    assert sync_result["assistant_turn_overlong"] == pytest.approx(-0.5)
    assert async_result["assistant_turn_overlong"] == pytest.approx(-0.5)
    assert sync_result["score"] == pytest.approx(1.0)
    assert async_result["score"] == pytest.approx(1.0)


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
        "tool_repeat_overlap": 0.0,
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
        "tool_repeat_overlap": 0.0,
        "tool_call_bonus": 0.0,
    }


def test_mandatory_think_accepts_decoded_role_markers_around_tool_response():
    solution = """
<think>I need to zoom into the jersey painting.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {
  "bbox_2d": [751, 187, 924, 298], "label": "painting of a baseball jersey", "img_idx": 0
}}
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
        "tool_repeat_overlap": 0.0,
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
        "tool_repeat_overlap": 0.0,
        "tool_call_bonus": 0.05,
    }


def test_strict_boxed_format_gives_unboxed_correct_answer_partial_format_credit():
    solution = "<think>The sign text is readable.</think><answer>GRAND</answer>"

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="GRAND",
        extra_info={"reward_type": "search"},
        format_score=0.2,
        allow_unboxed_answer_for_accuracy=True,
        require_boxed_answer_for_full_format=True,
        unboxed_answer_format_score=0.5,
    )

    assert result == {
        "score": pytest.approx(0.9),
        "accuracy": 1.0,
        "format": 0.5,
        "tool_call_count": 0,
        "tool_efficiency": 1.0,
        "tool_repeat_overlap": 0.0,
        "tool_call_bonus": 0.0,
        "answer_has_boxed": False,
        "answer_used_unboxed_fallback": True,
    }


def test_strict_boxed_format_gives_boxed_correct_answer_full_format_credit():
    solution = "<think>The sign text is readable.</think><answer>It says \\boxed{GRAND}.</answer>"

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="GRAND",
        extra_info={"reward_type": "search"},
        format_score=0.2,
        allow_unboxed_answer_for_accuracy=True,
        require_boxed_answer_for_full_format=True,
        unboxed_answer_format_score=0.5,
    )

    assert result == {
        "score": pytest.approx(1.0),
        "accuracy": 1.0,
        "format": 1.0,
        "tool_call_count": 0,
        "tool_efficiency": 1.0,
        "tool_repeat_overlap": 0.0,
        "tool_call_bonus": 0.0,
        "answer_has_boxed": True,
        "answer_used_unboxed_fallback": False,
    }


def test_strict_boxed_format_gives_unboxed_wrong_answer_partial_format_only():
    solution = "<think>The sign text is readable.</think><answer>SMALL</answer>"

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="GRAND",
        extra_info={"reward_type": "search"},
        format_score=0.2,
        allow_unboxed_answer_for_accuracy=True,
        require_boxed_answer_for_full_format=True,
        unboxed_answer_format_score=0.5,
    )

    assert result == {
        "score": pytest.approx(0.1),
        "accuracy": 0.0,
        "format": 0.5,
        "tool_call_count": 0,
        "tool_efficiency": 1.0,
        "tool_repeat_overlap": 0.0,
        "tool_call_bonus": 0.0,
        "answer_has_boxed": False,
        "answer_used_unboxed_fallback": True,
    }


def test_strict_boxed_format_keeps_invalid_structure_zero():
    solution = "plain answer GRAND"

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="GRAND",
        extra_info={"reward_type": "search"},
        format_score=0.2,
        allow_unboxed_answer_for_accuracy=True,
        require_boxed_answer_for_full_format=True,
        unboxed_answer_format_score=0.5,
    )

    assert result == {
        "score": 0.0,
        "accuracy": 0.0,
        "format": 0.0,
        "tool_call_count": 0,
        "tool_efficiency": 1.0,
        "tool_repeat_overlap": 0.0,
        "tool_call_bonus": 0.0,
        "answer_has_boxed": False,
        "answer_used_unboxed_fallback": False,
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
        "tool_repeat_overlap": 0.0,
        "tool_call_bonus": 0.0,
    }


def test_strict_boxed_format_async_entrypoint_matches_sync_reward():
    kwargs = {
        "data_source": "qwen3vl_filtering",
        "solution_str": "<think>The sign text is readable.</think><answer>GRAND</answer>",
        "ground_truth": "GRAND",
        "extra_info": {"reward_type": "search"},
        "format_score": 0.2,
        "allow_unboxed_answer_for_accuracy": True,
        "require_boxed_answer_for_full_format": True,
        "unboxed_answer_format_score": 0.5,
    }

    sync_result = compute_score_from_data_source(**kwargs)
    async_result = asyncio.run(async_compute_score_from_data_source(**kwargs))

    assert async_result == sync_result
    assert async_result["accuracy"] == 1.0
    assert async_result["format"] == 0.5
    assert async_result["answer_used_unboxed_fallback"] is True


def test_shared_answer_extraction_uses_unboxed_content_for_judge_prediction():
    result = extract_answer_for_accuracy("GRAND", "search", allow_unboxed_answer_for_accuracy=True)

    assert result.prediction == "GRAND"
    assert result.has_boxed is False
    assert result.used_unboxed_fallback is True


def test_shared_answer_extraction_uses_boxed_payload_for_judge_prediction():
    result = extract_answer_for_accuracy(
        "The sign says \\boxed{GRAND}.", "search", allow_unboxed_answer_for_accuracy=True
    )

    assert result.prediction == "GRAND"
    assert result.has_boxed is True
    assert result.used_unboxed_fallback is False


def test_shared_answer_extraction_rejects_empty_answer():
    result = extract_answer_for_accuracy("", "search", allow_unboxed_answer_for_accuracy=True)

    assert result.prediction is None
    assert result.has_boxed is False
    assert result.used_unboxed_fallback is False


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
        "tool_repeat_overlap": 0.0,
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
        "tool_repeat_overlap": 0.0,
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
        "tool_repeat_overlap": 0.0,
        "tool_call_bonus": 0.0,
    }


def test_tool_repeat_overlap_penalizes_same_img_idx_high_iou_repeat():
    solution = """
<think>Zoom into the legend.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [100, 100, 400, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>ok</tool_response>
<think>The crop was unclear, zoom nearly the same place again.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [105, 105, 395, 395], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>ok</tool_response>
<think>The answer is clear.</think>
<answer>Memory Care</answer>
""".strip()

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="memory care",
        extra_info={"reward_type": "string_match"},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["tool_call_count"] == 2
    assert result["tool_repeat_overlap"] == pytest.approx(-0.5)


def test_tool_repeat_overlap_normalizes_multiple_same_img_idx_repeats():
    solution = """
<think>Zoom into the legend.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [100, 100, 400, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>ok</tool_response>
<think>Repeat once.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [101, 101, 399, 399], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>ok</tool_response>
<think>Repeat twice.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [102, 102, 398, 398], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>ok</tool_response>
<think>The answer is clear.</think>
<answer>Memory Care</answer>
""".strip()

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="memory care",
        extra_info={"reward_type": "string_match"},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["tool_call_count"] == 3
    assert result["tool_repeat_overlap"] == pytest.approx(-2.0 / 3.0)


def test_tool_repeat_overlap_ignores_high_iou_across_different_img_idx():
    solution = """
<think>Zoom into the full image.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [100, 100, 400, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>ok</tool_response>
<think>Zoom into the same normalized region of a different crop.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [100, 100, 400, 400], "label": "legend", "img_idx": 1}}
</tool_call>
<tool_response>ok</tool_response>
<think>The answer is clear.</think>
<answer>Memory Care</answer>
""".strip()

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="memory care",
        extra_info={"reward_type": "string_match"},
        format_score=0.2,
    )

    assert result["format"] == 1.0
    assert result["tool_call_count"] == 2
    assert result["tool_repeat_overlap"] == 0.0


def test_tool_repeat_overlap_has_no_early_saturation():
    calls = [zoom_call([100 + offset, 100 + offset, 400 - offset, 400 - offset]) for offset in range(5)]

    assert tool_repeat_overlap_reward(calls) == pytest.approx(-4.0 / 5.0)


def test_tool_repeat_overlap_accepts_legacy_penalty_kwargs_without_using_old_clamp():
    calls = [zoom_call([100 + offset, 100 + offset, 400 - offset, 400 - offset]) for offset in range(3)]

    result = tool_repeat_overlap_reward(calls, penalty_per_repeat=-10.0, max_penalty=-0.1)

    assert result == pytest.approx(-2.0 / 3.0)


@pytest.mark.parametrize(
    ("bbox", "expected"),
    [
        ([0, 0, 19.9, 10], -0.5),  # Just below 200 / 1,000,000.
        ([0, 0, 20, 10], 0.0),  # The threshold is strict.
        ([0, 0, 20.1, 10], 0.0),
    ],
)
def test_tool_repeat_overlap_small_crop_threshold(bbox, expected):
    assert tool_repeat_overlap_reward([zoom_call(bbox)]) == pytest.approx(expected)


@pytest.mark.parametrize(
    "call",
    [
        zoom_call([0, 0, 10]),
        zoom_call([0, 0, float("nan"), 10]),
        zoom_call([10, 0, 0, 10]),
        zoom_call([0, 0, 10, 10], img_idx=1.5),
        {"name": "other_tool", "arguments": {}},
    ],
)
def test_tool_repeat_overlap_ignores_malformed_and_non_zoom_calls(call):
    assert tool_repeat_overlap_reward([call]) == 0.0


def test_tool_repeat_overlap_allows_two_depth_refinements_then_penalizes_excess():
    two_refinements = [
        zoom_call([0, 0, 1000, 1000]),
        zoom_call([100, 100, 900, 900]),
        zoom_call([200, 200, 800, 800]),
    ]
    three_refinements = [*two_refinements, zoom_call([250, 250, 750, 750])]

    assert tool_repeat_overlap_reward(two_refinements) == 0.0
    assert tool_repeat_overlap_reward(three_refinements) == pytest.approx(-0.5 / 4.0)

    diagnostics = tool_repeat_overlap_call_diagnostics(three_refinements)
    assert [item["is_excess_depth"] for item in diagnostics] == [False, False, False, True]
    assert diagnostics[-1]["has_penalty"] is True


def test_tool_repeat_overlap_depth_run_resets_after_breadth_switch():
    calls = [
        zoom_call([0, 0, 1000, 1000]),
        zoom_call([50, 50, 700, 700]),
        zoom_call([100, 100, 600, 600]),
        zoom_call([700, 700, 1000, 1000]),
        zoom_call([725, 725, 975, 975]),
        zoom_call([750, 750, 950, 950]),
    ]

    assert tool_repeat_overlap_reward(calls) == 0.0


def test_tool_repeat_overlap_depth_run_resets_after_image_switch():
    calls = [
        zoom_call([0, 0, 1000, 1000], img_idx=0),
        zoom_call([100, 100, 900, 900], img_idx=0),
        zoom_call([200, 200, 800, 800], img_idx=0),
        zoom_call([250, 250, 750, 750], img_idx=1),
    ]

    assert tool_repeat_overlap_reward(calls) == 0.0


def test_tool_repeat_overlap_does_not_double_count_repeats_as_depth():
    calls = [zoom_call([100 + offset, 100 + offset, 400 - offset, 400 - offset]) for offset in range(4)]

    assert tool_repeat_overlap_reward(calls) == pytest.approx(-3.0 / 4.0)


def test_tool_repeat_overlap_combines_small_crop_and_excess_depth_penalties():
    calls = [
        zoom_call([0, 0, 1000, 1000]),
        zoom_call([400, 400, 500, 500]),
        zoom_call([430, 430, 460, 460]),
        zoom_call([440, 440, 450, 450]),
    ]

    # The final crop is tiny and is the third consecutive nested refinement.
    assert tool_repeat_overlap_reward(calls) == pytest.approx(-0.25)


def test_tool_repeat_overlap_survives_malformed_final_answer():
    solution = """
<think>Zoom into the legend.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [100, 100, 400, 400], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>ok</tool_response>
<think>Repeat the crop.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [105, 105, 395, 395], "label": "legend", "img_idx": 0}}
</tool_call>
<tool_response>ok</tool_response>
<think>Malformed final response.</think><answer>Memory Care
""".strip()

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="memory care",
        extra_info={"reward_type": "string_match"},
    )

    assert result["format"] == 0.0
    assert result["accuracy"] == 0.0
    assert result["tool_repeat_overlap"] == pytest.approx(-0.5)


def test_tool_repeat_overlap_custom_weights_match_sync_and_async_entrypoints():
    solution = """
<think>Zoom once.</think>
<tool_call>
{"name": "image_zoom_in_tool", "arguments": {"bbox_2d": [0, 0, 10, 10], "label": "detail", "img_idx": 0}}
</tool_call>
<tool_response>ok</tool_response>
<think>Answer.</think><answer>Memory Care</answer>
""".strip()
    kwargs = {
        "data_source": "qwen3vl_filtering",
        "solution_str": solution,
        "ground_truth": "memory care",
        "extra_info": {"reward_type": "string_match"},
        "tool_repeat_overlap_small_crop_weight": 0.25,
    }

    sync_result = compute_score_from_data_source(**kwargs)
    async_result = asyncio.run(async_compute_score_from_data_source(**kwargs))

    assert async_result == sync_result
    assert sync_result["tool_repeat_overlap"] == pytest.approx(-0.25)


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
        "tool_repeat_overlap": 0.0,
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
        "tool_repeat_overlap": 0.0,
        "tool_call_bonus": 0.0,
    }


def draw_points_solution(points, include_answer=True, answer_count=2):
    parts = [
        "<think>I will mark each object I can see.</think>",
        "<tool_call>",
        json.dumps({"name": "draw_points", "arguments": {"img_idx": 0, "points": points}}),
        "</tool_call>",
        "<tool_response>",
        "Annotated image returned.",
        "</tool_response>",
    ]
    if include_answer:
        parts.append("<think>All objects are marked now.</think>")
        parts.append(f"<answer>I count \\boxed{{{answer_count}}} objects.</answer>")
    return "\n".join(parts)


def test_point_box_reward_rescales_points_into_processed_size_hit():
    solution = draw_points_solution([[500, 500]])
    extra_info = {"metadata": {"boxes": [[90, 40, 110, 60]], "processed_size": [200, 100]}}

    score, applicable, gt_boxes, predicted, matched_boxes, matched_points = point_box_reward(solution, extra_info)

    assert score == pytest.approx(1.0)
    assert applicable == 1.0
    assert (gt_boxes, predicted, matched_boxes, matched_points) == (1, 1, 1, 1)


def test_point_box_reward_rescaled_point_outside_box_misses():
    solution = draw_points_solution([[500, 500]])
    extra_info = {"metadata": {"boxes": [[0, 0, 50, 50]], "processed_size": [200, 100]}}

    score, applicable, gt_boxes, predicted, matched_boxes, matched_points = point_box_reward(solution, extra_info)

    assert score == pytest.approx(0.0)
    assert applicable == 1.0
    assert (gt_boxes, predicted, matched_boxes, matched_points) == (1, 1, 0, 0)


def test_point_box_reward_duplicate_points_match_one_box_only():
    solution = draw_points_solution([[100, 100], [100, 100]])
    extra_info = {
        "metadata": {"boxes": [[0, 0, 40, 40], [150, 80, 190, 95]], "processed_size": [200, 100]}
    }

    score, applicable, gt_boxes, predicted, matched_boxes, matched_points = point_box_reward(solution, extra_info)

    assert score == pytest.approx(0.5)  # recall 1/2, precision 1/2
    assert applicable == 1.0
    assert (gt_boxes, predicted, matched_boxes, matched_points) == (2, 2, 1, 1)


def test_point_box_reward_without_gt_boxes_is_not_applicable():
    solution = draw_points_solution([[500, 500]])

    score, applicable, gt_boxes, predicted, matched_boxes, matched_points = point_box_reward(
        solution, {"metadata": {}}
    )

    assert score == 0.0
    assert applicable == 0.0
    assert (gt_boxes, predicted, matched_boxes, matched_points) == (0, 1, 0, 0)


def test_point_box_reward_with_boxes_but_no_draw_calls_scores_zero_but_applicable():
    solution = "<think>No tools needed.</think>\n<answer>I count \\boxed{1} objects.</answer>"
    extra_info = {"metadata": {"boxes": [[90, 40, 110, 60]], "processed_size": [200, 100]}}

    score, applicable, gt_boxes, predicted, matched_boxes, matched_points = point_box_reward(solution, extra_info)

    assert score == 0.0
    assert applicable == 1.0
    assert (gt_boxes, predicted, matched_boxes, matched_points) == (1, 0, 0, 0)


def test_point_box_reward_tolerance_px_expands_match_window():
    solution = draw_points_solution([[500, 500]])  # rescales to pixel (100, 50)
    extra_info = {"metadata": {"boxes": [[104, 54, 110, 60]], "processed_size": [200, 100]}}

    strict_score, strict_applicable, *_ = point_box_reward(solution, extra_info, tolerance_px=0.0)
    tolerant_score, tolerant_applicable, *_ = point_box_reward(solution, extra_info, tolerance_px=5.0)

    assert strict_score == pytest.approx(0.0)
    assert tolerant_score == pytest.approx(1.0)
    assert strict_applicable == tolerant_applicable == 1.0


def test_point_box_component_returned_only_when_enabled():
    solution = draw_points_solution([[500, 500]], answer_count=1)
    extra_info = {
        "reward_type": "counting",
        "metadata": {"boxes": [[90, 40, 110, 60]], "processed_size": [200, 100]},
    }

    enabled = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="1",
        extra_info=extra_info,
        format_score=0.1,
        point_box_reward_enabled=True,
    )
    disabled = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="1",
        extra_info=extra_info,
        format_score=0.1,
    )

    assert enabled["accuracy"] == pytest.approx(1.0)
    assert enabled["point_box"] == pytest.approx(1.0)
    assert enabled["point_box_applicable"] == 1.0
    assert enabled["point_box_gt_boxes"] == 1.0
    assert enabled["point_box_predicted_points"] == 1.0
    assert enabled["point_box_matched_boxes"] == 1.0
    assert enabled["point_box_matched_points"] == 1.0
    assert all(not key.startswith("point_box") for key in disabled)


def test_point_box_component_zeroed_on_invalid_trajectory_but_keeps_applicable():
    solution = draw_points_solution([[500, 500]], include_answer=False)
    extra_info = {
        "reward_type": "counting",
        "metadata": {"boxes": [[90, 40, 110, 60]], "processed_size": [200, 100]},
    }

    result = compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=solution,
        ground_truth="1",
        extra_info=extra_info,
        format_score=0.1,
        point_box_reward_enabled=True,
    )

    assert result["format"] == 0.0
    assert result["point_box"] == 0.0
    assert result["point_box_applicable"] == 1.0
    assert result["point_box_matched_boxes"] == 1.0  # diagnostics keep the raw match


def test_point_box_component_async_entrypoint_matches_sync():
    solution = draw_points_solution([[500, 500]], answer_count=1)
    extra_info = {
        "reward_type": "counting",
        "metadata": {"boxes": [[90, 40, 110, 60]], "processed_size": [200, 100]},
    }
    kwargs = {
        "data_source": "qwen3vl_filtering",
        "solution_str": solution,
        "ground_truth": "1",
        "extra_info": extra_info,
        "format_score": 0.1,
        "point_box_reward_enabled": True,
    }

    sync_result = compute_score_from_data_source(**kwargs)
    async_result = asyncio.run(async_compute_score_from_data_source(**kwargs))

    assert async_result == sync_result
