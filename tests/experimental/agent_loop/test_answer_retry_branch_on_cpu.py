from types import SimpleNamespace

import torch

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopMetrics,
    AgentLoopOutput,
    AgentLoopWorker,
    _InternalAgentLoopOutput,
)
from verl.experimental.agent_loop.tool_agent_loop import AgentData, ToolAgentLoop
from verl.protocol import DataProto


def _make_loop() -> ToolAgentLoop:
    loop = object.__new__(ToolAgentLoop)
    loop.conditional_answer_retry_branch_enabled = True
    loop.conditional_answer_retry_branch_enable_validation = False
    loop.conditional_answer_retry_branch_mask_mode = "divergence_only"
    loop.conditional_answer_retry_branch_reward_components = {
        "accuracy": 0.0,
        "format": 0.0,
        "tool_repeat_overlap": 0.0,
    }
    loop.conditional_answer_retry_branch_use_graded_rewards = False
    loop.conditional_answer_retry_branch_baseline_weight_key = "gdpo_baseline_weight"
    loop.response_length = 32
    return loop


def _make_agent_data() -> AgentData:
    agent_data = AgentData(
        [],
        None,
        None,
        {},
        "request-1",
        {},
    )
    agent_data.prompt_ids = [10, 11, 12, 13, 14, 15, 16]
    agent_data.response_mask = [1, 1, 0, 0, 1]
    agent_data.response_logprobs = [-0.1, -0.2, 0.0, 0.0, -0.3]
    agent_data.user_turns = 2
    agent_data.assistant_turns = 2
    return agent_data


def test_wrong_answer_branch_uses_divergence_only_mask_and_zero_components():
    loop = _make_loop()
    agent_data = _make_agent_data()
    snapshot = {"response_mask_len": 4}

    loop._maybe_append_wrong_answer_branch(
        agent_data=agent_data,
        generation_snapshot=snapshot,
        grade={"score": 0.1, "accuracy": 0.0, "format": 1.0, "tool_repeat_overlap": 0.25},
        retry_count=1,
        removed_assistant_text="<think>bad</think><answer>wrong</answer>",
        rollout_kwargs={"_debug_trajectory": {"validate": False}},
    )

    assert len(agent_data.branch_outputs) == 1
    branch = agent_data.branch_outputs[0]
    assert branch.prompt_ids == [10, 11]
    assert branch.response_ids == [12, 13, 14, 15, 16]
    assert branch.response_mask == [0, 0, 0, 0, 1]
    assert branch.response_logprobs == [-0.1, -0.2, 0.0, 0.0, -0.3]
    assert branch.reward_score == 0.0
    assert branch.extra_fields["assistant_turns"] == 2
    assert branch.extra_fields["is_conditional_answer_retry_branch"] is True
    assert branch.extra_fields["gdpo_baseline_weight"] == 0.0
    reward_extra_info = branch.extra_fields["reward_extra_info"]
    assert reward_extra_info["accuracy"] == 0.0
    assert reward_extra_info["format"] == 0.0
    assert reward_extra_info["tool_repeat_overlap"] == 0.0


def test_wrong_answer_branch_can_preserve_complete_graded_reward():
    loop = _make_loop()
    loop.conditional_answer_retry_branch_use_graded_rewards = True
    agent_data = _make_agent_data()
    grade = {
        "score": 0.05,
        "accuracy": 0.0,
        "format": 0.5,
        "tool_repeat_overlap": -0.25,
        "tool_efficiency": 0.75,
        "tool_call_bonus": 0.1,
        "future_reward_component": 0.4,
    }

    loop._maybe_append_wrong_answer_branch(
        agent_data=agent_data,
        generation_snapshot={"response_mask_len": 4},
        grade=grade,
        retry_count=1,
        removed_assistant_text="<think>bad</think><answer>wrong</answer>",
        rollout_kwargs={"_debug_trajectory": {"validate": False}},
    )

    branch = agent_data.branch_outputs[0]
    assert branch.reward_score == grade["score"]
    assert branch.extra_fields["reward_extra_info"] == grade
    assert branch.extra_fields["gdpo_baseline_weight"] == 0.0

    zero_format_grade = {**grade, "score": 0.0, "format": 0.0}
    assert loop._branch_reward_extra_info(zero_format_grade) == zero_format_grade


def test_wrong_answer_branch_not_emitted_for_validation_by_default():
    loop = _make_loop()
    agent_data = _make_agent_data()

    loop._maybe_append_wrong_answer_branch(
        agent_data=agent_data,
        generation_snapshot={"response_mask_len": 4},
        grade={"score": 0.0, "accuracy": 0.0},
        retry_count=1,
        removed_assistant_text="<answer>wrong</answer>",
        rollout_kwargs={"_debug_trajectory": {"validate": True}},
    )

    assert agent_data.branch_outputs == []


def test_wrong_answer_branch_keeps_forced_prefill_masked_after_divergence():
    loop = _make_loop()
    agent_data = _make_agent_data()
    agent_data.response_mask = [1, 1, 0, 0, 0, 1, 1]
    agent_data.prompt_ids = [10, 11, 12, 13, 14, 15, 16, 17, 18]
    agent_data.response_logprobs = [-0.1, -0.2, 0.0, 0.0, 0.0, -0.3, -0.4]

    loop._maybe_append_wrong_answer_branch(
        agent_data=agent_data,
        generation_snapshot={"response_mask_len": 4},
        grade={"score": 0.0, "accuracy": 0.0},
        retry_count=1,
        removed_assistant_text="<think>bad</think><answer>wrong</answer>",
        rollout_kwargs={"_debug_trajectory": {"validate": False}},
    )

    branch = agent_data.branch_outputs[0]
    assert branch.response_ids == [12, 13, 14, 15, 16, 17, 18]
    assert branch.response_mask == [0, 0, 0, 0, 0, 1, 1]


def test_wrong_answer_branch_snapshots_multimodal_state():
    loop = _make_loop()
    agent_data = _make_agent_data()
    image_0 = object()
    image_1 = object()
    image_2 = object()
    agent_data.image_data = [image_0, image_1]
    agent_data.crop_image_refs = [{"path": "crop_1.png", "image_index": 1}]

    loop._maybe_append_wrong_answer_branch(
        agent_data=agent_data,
        generation_snapshot={"response_mask_len": 4},
        grade={"score": 0.0, "accuracy": 0.0},
        retry_count=1,
        removed_assistant_text="<think>bad</think><answer>wrong</answer>",
        rollout_kwargs={"_debug_trajectory": {"validate": False}},
    )

    branch = agent_data.branch_outputs[0]
    agent_data.image_data.append(image_2)
    agent_data.crop_image_refs.append({"path": "crop_2.png", "image_index": 2})

    assert branch.multi_modal_data["images"] == [image_0, image_1]
    assert branch.extra_fields["crop_image_refs"] == [{"path": "crop_1.png", "image_index": 1}]


def _make_internal_output(extra_fields: dict) -> _InternalAgentLoopOutput:
    return _InternalAgentLoopOutput(
        prompt_ids=torch.tensor([[1, 2]], dtype=torch.long),
        response_ids=torch.tensor([[3, 4, 0]], dtype=torch.long),
        response_mask=torch.tensor([[1, 1, 0]], dtype=torch.long),
        input_ids=torch.tensor([[1, 2, 3, 4, 0]], dtype=torch.long),
        attention_mask=torch.tensor([[1, 1, 1, 1, 0]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2, 3, 4]], dtype=torch.long),
        reward_score=0.0,
        num_turns=1,
        metrics=AgentLoopMetrics(),
        extra_fields=extra_fields,
    )


def test_branch_extra_fields_are_stable_across_worker_concatenation():
    worker = object.__new__(AgentLoopWorker)
    worker.reward_loop_worker_handles = None
    worker.distillation_enabled = False
    worker.stream_teacher_with_rollout = False

    canonical = _make_internal_output(
        {
            "is_conditional_answer_retry_branch": False,
            "gdpo_baseline_weight": 1.0,
        }
    )
    branch = _make_internal_output(
        {
            "is_conditional_answer_retry_branch": True,
            "gdpo_baseline_weight": 0.0,
            "conditional_answer_retry_branch_removed_assistant_text": "<answer>wrong</answer>",
        }
    )

    no_branch_worker_output = worker._postprocess([canonical])
    branch_worker_output = worker._postprocess([branch])
    concatenated = DataProto.concat([no_branch_worker_output, branch_worker_output])

    assert len(concatenated) == 2
    assert concatenated.non_tensor_batch["conditional_answer_retry_branch_removed_assistant_text"].tolist() == [
        None,
        "<answer>wrong</answer>",
    ]


def test_rollout_row_id_survives_async_reward_postprocess_for_branch_rows():
    worker = object.__new__(AgentLoopWorker)
    worker.reward_loop_worker_handles = [object()]
    worker.distillation_enabled = False
    worker.stream_teacher_with_rollout = False

    canonical_0 = _make_internal_output({"__agent_loop_input_row": 0})
    branch_0 = _make_internal_output(
        {
            "__agent_loop_input_row": 0,
            "is_conditional_answer_retry_branch": True,
            "gdpo_baseline_weight": 0.0,
            "conditional_answer_retry_branch_removed_assistant_text": "<answer>wrong</answer>",
        }
    )
    canonical_1 = _make_internal_output({"__agent_loop_input_row": 1})

    output = worker._postprocess(
        [canonical_0, branch_0, canonical_1],
        input_non_tensor_batch={"__rollout_row_id": torch.tensor([0, 1]).numpy()},
    )

    assert len(output) == 3
    assert output.non_tensor_batch["__rollout_row_id"].tolist() == [0, 0, 1]


def _finalize_sft_mask(selection: str, final_accuracy=1.0):
    worker = object.__new__(AgentLoopWorker)
    worker.rollout_config = SimpleNamespace(
        multi_turn=SimpleNamespace(
            conditional_answer_retry_sft_enabled=True,
            conditional_answer_retry_sft_target_selection=selection,
        )
    )
    candidates = [
        {
            "response_start": 1,
            "response_end": 3,
            "rejected_accuracy": 0.0,
            "static_rejection_reasons": [],
            "tool_executed_successfully": True,
        },
        {
            "response_start": 5,
            "response_end": 8,
            "rejected_accuracy": 0.0,
            "static_rejection_reasons": [],
            "tool_executed_successfully": True,
        },
    ]
    output = AgentLoopOutput(
        prompt_ids=[1],
        response_ids=list(range(10)),
        response_mask=[1] * 10,
        metrics=AgentLoopMetrics(),
        extra_fields={
            "__agent_loop_retry_sft_candidates": candidates,
            "reward_extra_info": {"accuracy": final_accuracy},
        },
    )
    return worker._finalize_retry_sft_mask(output), output


def test_retry_sft_target_selection_defaults_support_all_first_and_last():
    all_mask, _ = _finalize_sft_mask("all")
    first_mask, _ = _finalize_sft_mask("first")
    last_mask, _ = _finalize_sft_mask("last")

    assert all_mask == [0, 1, 1, 0, 0, 1, 1, 1, 0, 0]
    assert first_mask == [0, 1, 1, 0, 0, 0, 0, 0, 0, 0]
    assert last_mask == [0, 0, 0, 0, 0, 1, 1, 1, 0, 0]


def test_retry_sft_final_accuracy_must_strictly_improve():
    mask, output = _finalize_sft_mask("all", final_accuracy=0.0)

    assert mask == [0] * 10
    assert output.extra_fields["retry_sft_rejection_counts"] == {"accuracy_not_improved": 2}
