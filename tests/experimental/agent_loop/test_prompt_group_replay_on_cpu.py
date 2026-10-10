"""Prompt-group scheduling, candidate capture, and sanitized reasoning validation."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image
from test_answer_retry_branch_on_cpu import _make_internal_output
from test_candidate_answer_schedule_on_cpu import (
    TRAJECTORY,
    complete,
    make_case,
)
from test_forced_action_tags_on_cpu import _make_agent_data

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopManager,
    AgentLoopWorker,
    _align_replay_output_columns,
)
from verl.experimental.agent_loop.tool_agent_loop import TERMINAL_TOOL_EXTRA_FIELD_DEFAULTS, AgentState
from verl.protocol import DataProto
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.ppo.trajectory_replay import object_array
from verl.workers.config.rollout import MultiTurnConfig
from verl.workers.rollout.replica import TokenOutput


def group_case(*, turns=(4,), grades=(1.0,), enabled=True, validate=False):
    loop, data, server = make_case(enabled=False, grades=grades)
    # Existing fake AgentData starts with no messages and populates them later.
    data.replay_question = "question"
    loop.prompt_group_schedule_enabled = enabled
    loop.conditional_answer_retry_accept_partial_credit = False
    assignment = {"group_id": "prompt-0", "force_first_tool_call": True, "answer_turns": list(turns)}
    loop._initialize_prompt_group_schedule(data, assignment, {**TRAJECTORY, "validate": validate})
    return loop, data, server


def test_group_schedule_forces_only_first_tool_and_assigned_answer():
    loop, data, server = group_case(turns=(5,))
    complete(loop, data)
    assert [call["tag"] for call in server.calls] == ["<tool_call>", None, None, None, "<answer>"]
    assert data.prompt_group_schedule.consumed_answer_turns == {5}
    assert 72 not in server.calls[-1]["prompt"]  # No final-budget prefill on intermediate deadlines.


def test_group_assignment_is_shared_but_consumption_is_sample_local():
    loop, _, _ = group_case(turns=(2, 8))
    assignment = {"group_id": "shared", "force_first_tool_call": True, "answer_turns": [2, 8]}
    samples = [_make_agent_data() for _ in range(8)]
    for index, data in enumerate(samples):
        loop._initialize_prompt_group_schedule(data, assignment, {**TRAJECTORY, "rollout_n": index})
    samples[0].prompt_group_schedule.consumed_answer_turns.add(2)
    assignment["answer_turns"].append(9)
    assert all(data.prompt_group_schedule.answer_turns == (2, 8) for data in samples)
    assert all(not data.prompt_group_schedule.consumed_answer_turns for data in samples[1:])


def test_wrong_scheduled_answer_consumption_survives_rollback():
    loop, data, server = group_case(turns=(4, 7), grades=(0.0, 1.0))
    complete(loop, data)
    assert [call["tag"] for call in server.calls] == [
        "<tool_call>",
        None,
        None,
        "<answer>",
        "<tool_call>",
        None,
        None,
        "<answer>",
    ]
    assert data.assistant_turns == 7
    assert data.prompt_group_schedule.consumed_answer_turns == {4, 7}
    assert data.branch_outputs[0].extra_fields["prompt_group_schedule"]["consumed_answer_turns"] == [4]
    assert data.branch_outputs[0].extra_fields["replay_candidate_entries"] == []


def test_early_voluntary_wrong_answer_does_not_consume_deadline():
    loop, data, server = group_case(turns=(5, 8), grades=(0.0, 1.0))
    server.voluntary = {(0, 2)}
    complete(loop, data)
    assert [event["answer_turn"] for event in data.prompt_group_schedule.history] == [5]
    assert data.prompt_group_schedule.consumed_answer_turns == {5}
    assert data.assistant_turns == 5  # Accepted candidate prevents the second deadline.
    assert server.calls[1]["tag"] is None
    assert server.calls[2]["tag"] == "<tool_call>"  # Retry reuses retained turn two.
    assert server.calls[-1]["tag"] == "<answer>"


def test_early_voluntary_accept_stops_before_all_deadlines():
    loop, data, server = group_case(turns=(5, 8), grades=(1.0,))
    server.voluntary = {(0, 2)}
    complete(loop, data)
    assert len(server.calls) == 2
    assert not data.prompt_group_schedule.consumed_answer_turns


@pytest.mark.parametrize("enabled,validate", [(False, False), (True, True)])
def test_group_schedule_disabled_or_validation_preserves_legacy(enabled, validate):
    loop, data, server = group_case(enabled=enabled, validate=validate)
    complete(loop, data, validate=validate)
    assert data.prompt_group_schedule is None
    assert len(server.calls) == 11
    assert all(call["tag"] is None for call in server.calls[:-1])
    assert server.calls[-1]["tag"] == "<answer>"


@pytest.mark.parametrize("turns", [(1,), (10,), (3, 4), (5, 3), (3, 3), (2, 4, 6), (True,)])
def test_group_schedule_rejects_invalid_deadlines(turns):
    with pytest.raises(ValueError):
        group_case(turns=turns)


def configure_replay(loop, root):
    loop.config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(root),
                "trajectory_replay": {"enabled": True, "normal_budget": 4096, "overlong_buffer": 1024},
            }
        }
    )


@pytest.mark.parametrize("turn,tools,expected", [(1, 0, "initial"), (3, 2, "tool_outputs")])
def test_capture_uses_initial_exception_or_ordered_tool_outputs(tmp_path, turn, tools, expected):
    loop, data, _ = group_case()
    configure_replay(loop, tmp_path)
    initial = Image.new("RGB", (20, 10), "red")
    crops = [Image.new("RGB", (7, 5), "green"), Image.new("RGB", (9, 6), "blue")]
    data.initial_processed_images = (initial,)
    data.tool_output_images = crops if tools else []
    data.executed_tool_calls = [{}] * tools
    data.assistant_turns = turn
    asyncio.run(
        loop._capture_replay_candidate(
            data,
            "<think>PRIVATE SOURCE REASONING</think><answer>\\boxed{42}</answer>",
            {"accuracy": 0.5},
            {"_debug_trajectory": TRAJECTORY, "uid": "task-uid"},
        )
    )
    (entry,) = data.replay_candidate_entries
    assert entry["evidence_mode"] == expected
    assert entry["answer"] == "\\boxed{42}"
    assert entry["label"] == "R"
    assert "PRIVATE" not in str(entry)
    expected_images = [initial] if expected == "initial" else crops
    assert [(ref["width"], ref["height"]) for ref in entry["images"]] == [image.size for image in expected_images]
    for ref, original in zip(entry["images"], expected_images, strict=True):
        with Image.open(ref["path"]) as saved:
            assert saved.tobytes() == original.tobytes()


def test_capture_excludes_validation_and_failed_grade(tmp_path):
    loop, data, _ = group_case()
    configure_replay(loop, tmp_path)
    data.initial_processed_images = (Image.new("RGB", (3, 3)),)
    data.assistant_turns = 1

    async def capture():
        await loop._capture_replay_candidate(
            data,
            "<answer>42</answer>",
            {"accuracy": 1.0},
            {
                "_debug_trajectory": {**TRAJECTORY, "validate": True},
            },
        )
        await loop._capture_replay_candidate(data, "<answer>42</answer>", {"accuracy": 0.0, "error": "timeout"}, {})

    asyncio.run(capture())
    assert not data.replay_candidate_entries
    assert not (tmp_path / "trajectory_replay").exists()


def test_rejected_and_accepted_candidates_are_captured_before_rollback(tmp_path):
    loop, data, _ = group_case(turns=(4, 7), grades=(0.0, 1.0))
    configure_replay(loop, tmp_path)
    data.tool_output_images = [Image.new("RGB", (3, 3))]
    data.executed_tool_calls = [{}]
    complete(loop, data)
    assert [entry["label"] for entry in data.replay_candidate_entries] == ["R", "A"]
    assert len(data.branch_outputs) == 1
    assert data.branch_outputs[0].extra_fields["replay_candidate_entries"] == []


@pytest.mark.parametrize("accuracy", [0.0, 1.0])
@pytest.mark.parametrize("replay_enabled", [False, True])
def test_replay_preserves_terminal_grade_without_oracle_sidecar(tmp_path, accuracy, replay_enabled):
    loop, data, _ = group_case(grades=(accuracy,))
    configure_replay(loop, tmp_path)
    loop.config.trainer.trajectory_replay.enabled = replay_enabled
    loop.conditional_answer_retry_critic_oracle_supervision_enabled = False
    data.assistant_turns = 11 if accuracy == 0.0 else 1
    snapshot = loop._snapshot_generation_state(data)
    snapshot["assistant_turns"] = data.assistant_turns - 1
    state = asyncio.run(
        loop._maybe_retry_incorrect_answer(
            agent_data=data,
            generation_snapshot=snapshot,
            assistant_message="<answer>42</answer>",
            rollout_kwargs={"_debug_trajectory": TRAJECTORY},
            ignore_termination=False,
        )
    )
    assert state is None
    assert not any(data.retry_critic_mask)
    if replay_enabled:
        assert (
            data.extra_fields["__agent_loop_precritic_grade"]
            == (data.extra_fields["conditional_answer_retry_last_grade"])
        )
        assert data.extra_fields["__agent_loop_precritic_grade"]["accuracy"] == accuracy
    else:
        assert "__agent_loop_precritic_grade" not in data.extra_fields


def reasoning_validation_case(tmp_path, decision="R", malformed=False):
    loop, data, _ = group_case()
    configure_replay(loop, tmp_path)
    data.assistant_turns = 4
    data.prompt_ids = [900, 901]  # Full rollout context must never reach the critic.
    data.response_mask = [1]
    data.initial_processed_images = (Image.new("RGB", (20, 10), "red"),)
    data.tool_output_images = [Image.new("RGB", (7, 5), "green")]
    data.executed_tool_calls = [{}]
    loop.conditional_answer_retry_critic_mode = "reasoning"
    loop.tokenizer.text_to_ids["<think>\n"] = [30, 2]
    loop.tokenizer.id_to_text[80] = (
        "</think><answer>R</answer>" if malformed else f"Consider visible evidence.</think><answer>{decision}</answer>"
    )
    loop.server_manager.generate = AsyncMock(return_value=TokenOutput(token_ids=[80], log_probs=None))
    loop.apply_chat_template = AsyncMock(return_value=[100])
    return loop, data


def test_reasoning_validation_uses_current_policy_sanitized_context_and_retry_gate(tmp_path):
    loop, data = reasoning_validation_case(tmp_path)
    snapshot = loop._snapshot_generation_state(data)
    snapshot["assistant_turns"] = 3
    state = asyncio.run(
        loop._maybe_retry_answer_with_critic(
            agent_data=data,
            generation_snapshot=snapshot,
            assistant_message="<think>PRIVATE</think><answer>\\boxed{42}</answer>",
            ignore_termination=False,
        )
    )
    assert state == AgentState.GENERATING
    assert data.force_next_tool_call
    call = loop.server_manager.generate.call_args.kwargs
    assert call["sampling_params"] == {
        "n": 1,
        "max_tokens": 5120,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "logprobs": False,
    }
    assert 900 not in call["prompt_ids"] and 901 not in call["prompt_ids"]
    assert call["image_data"] == data.tool_output_images
    messages = loop.apply_chat_template.call_args_list[0].args[0]
    assert "PRIVATE" not in str(messages)
    assert "42" in str(messages)
    (record,) = data.extra_fields["retry_critic_validation_decision_log"]
    assert record["critic_mode"] == "reasoning"
    assert record["critic_evidence_mode"] == "tool_outputs"
    assert record["controller_action"] == "retry"
    assert record["p_retry"] is None
    assert not data.replay_candidate_entries


def test_reasoning_validation_malformed_output_retains_candidate(tmp_path):
    loop, data = reasoning_validation_case(tmp_path, malformed=True)
    old_ids = list(data.prompt_ids)
    state = asyncio.run(
        loop._maybe_retry_answer_with_critic(
            agent_data=data,
            generation_snapshot=loop._snapshot_generation_state(data),
            assistant_message="<answer>42</answer>",
            ignore_termination=False,
        )
    )
    assert state is None
    assert data.prompt_ids == old_ids
    assert data.extra_fields["retry_critic_validation_decision_log"][0]["controller_action"] == "accept_fallback"


def test_reasoning_validation_reject_at_final_cap_is_forced_accept(tmp_path):
    loop, data = reasoning_validation_case(tmp_path)
    data.assistant_turns = 11
    snapshot = loop._snapshot_generation_state(data)
    snapshot["assistant_turns"] = 10
    state = asyncio.run(
        loop._maybe_retry_answer_with_critic(
            agent_data=data,
            generation_snapshot=snapshot,
            assistant_message="<answer>42</answer>",
            ignore_termination=False,
        )
    )
    assert state is None
    record = data.extra_fields["retry_critic_validation_decision_log"][0]
    assert record["controller_action"] == "forced_accept"
    assert record["skip_or_error_reason"] == "max_assistant_turns"


def test_new_multi_turn_features_are_disabled_by_default():
    config = MultiTurnConfig()
    assert config.prompt_group_schedule_enabled is False
    assert config.conditional_answer_retry_critic_mode == "legacy"


@pytest.mark.parametrize("async_rewards", [False, True])
def test_mixed_worker_postprocess_manager_concat_and_trainer_union(async_rewards):
    """Exercise real row expansion, sparse schemas, reward-key union, and strict flag dtypes."""
    config = OmegaConf.create(
        {
            "trainer": {"trajectory_replay": {"enabled": True, "prompt_fraction": 0.5}},
            "actor_rollout_ref": {
                "actor": {"policy_loss": {"retry_critic": {"enabled": False}}},
                "rollout": {
                    "agent": {"default_agent_loop": "tool_agent"},
                    "multi_turn": {
                        "prompt_group_schedule_enabled": True,
                    },
                },
            },
        }
    )
    candidates = [
        {
            "entry_id": label,
            "label": label,
            "question": "question",
            "answer": "\\boxed{42}",
            "images": [{"path": "/tmp/preflight-only-evidence.png", "width": 28, "height": 28}],
        }
        for label in ("A", "R")
    ]
    trainer = object.__new__(RayPPOTrainer)
    trainer.config = config
    trainer.global_steps = 3
    trainer._replay_restore_step = -1
    trainer.train_dataset = SimpleNamespace(per_batch_domain_weights=None)
    trainer._replay_prompt_fits = lambda candidate: True

    def sample_candidates(count, step, eligible=None):
        return [entry for entry in candidates if eligible is None or eligible(entry)]

    trainer.trajectory_replay_cache = SimpleNamespace(
        entries=candidates,
        reload=lambda: None,
        sample=sample_candidates,
    )
    batch = DataProto.from_dict(
        tensors={"dummy_tensor": torch.zeros((4, 1), dtype=torch.uint8)},
        non_tensors={
            "raw_prompt": object_array([[{"role": "user", "content": f"question {i}"}] for i in range(4)]),
            "data_source": object_array(["task"] * 4),
            "reward_model": object_array([{"ground_truth": "42"} for _ in range(4)]),
            "extra_info": object_array([{} for _ in range(4)]),
            "index": np.arange(4),
        },
    )
    batch = trainer._prepare_trajectory_replay_batch(batch, {})
    batch.non_tensor_batch["uid"] = object_array([f"group-{i}" for i in range(4)])
    gen_batch = trainer._get_gen_batch(batch).repeat(2, interleave=True)
    gen_batch.non_tensor_batch["__rollout_row_id"] = np.arange(8)
    worker = object.__new__(AgentLoopWorker)
    worker.config = config
    worker.reward_loop_worker_handles = [object()] if async_rewards else None
    worker.distillation_enabled = False
    worker.stream_teacher_with_rollout = False
    chunks = gen_batch.chunk(2)
    outputs = []
    for is_judge, chunk in zip((False, True), chunks, strict=True):
        internal = []
        for row in range(len(chunk)):
            common = {
                "__agent_loop_input_row": row,
                "request_id": f"{'judge' if is_judge else 'task'}-{row}",
                "is_replay_judge": is_judge,
                "is_conditional_answer_retry_branch": False,
                "gdpo_baseline_weight": 1.0,
                "assistant_turns": 1,
                "crop_image_refs": [],
                "turn_scores": [],
                "tool_rewards": [],
            }
            if is_judge:
                common.update(
                    {
                        "replay_judge_rewards": {"accuracy": 1.0, "format": 1.0},
                        "reward_extra_info": {"accuracy": 1.0, "format": 1.0, "raw_format": 1.0},
                        "replay_candidate_entry_id": "A" if row < 2 else "R",
                        "replay_judge_trace": {"policy_tokens": 2},
                        "judge_reasoning_tokens": 1,
                        "judge_prefill_active_tokens": 0,
                    }
                )
            else:
                common.update(
                    {
                        **TERMINAL_TOOL_EXTRA_FIELD_DEFAULTS,
                        "reward_extra_info": {"accuracy": 1.0, "format": 1.0, "tool_efficiency": 0.5},
                        "prompt_group_schedule": {
                            **chunk.non_tensor_batch["prompt_group_schedule"][row],
                            "consumed_answer_turns": [],
                        },
                        "replay_candidate_entries": [{"entry_id": f"captured-{row}"}],
                        "comprehensive_debug": {"event": "sample_final"},
                    }
                )
            internal.append(_make_internal_output(common))
            if not is_judge:
                internal.append(
                    _make_internal_output(
                        {
                            **common,
                            "is_conditional_answer_retry_branch": True,
                            "gdpo_baseline_weight": 0.0,
                            "replay_candidate_entries": [],
                        }
                    )
                )
        outputs.append(worker._postprocess(internal, input_non_tensor_batch=chunk.non_tensor_batch))
    _align_replay_output_columns(outputs)
    for key in outputs[0].non_tensor_batch:
        assert outputs[0].non_tensor_batch[key].ndim == outputs[1].non_tensor_batch[key].ndim, (
            key,
            outputs[0].non_tensor_batch[key].shape,
            outputs[1].non_tensor_batch[key].shape,
        )
    direct = DataProto.concat(outputs)
    assert len(direct) == 12
    assert direct.non_tensor_batch["__rollout_row_id"].tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4, 5, 6, 7]
    assert direct.non_tensor_batch["terminal_reason"].shape == (12,)
    assert direct.non_tensor_batch["comprehensive_debug"].tolist()[-4:] == [None] * 4
    assert direct.non_tensor_batch["raw_format"].tolist() == [None] * 8 + [1.0] * 4
    if not async_rewards:
        raw_prompts = direct.non_tensor_batch["raw_prompt"]
        assert raw_prompts.ndim == 1
        assert raw_prompts[0] == [{"role": "user", "content": "question 0"}]
        assert raw_prompts[-1] == chunks[1].non_tensor_batch["raw_prompt"][-1]
        assert raw_prompts[-1][1]["content"][0]["text"].startswith("Original question:\nquestion\n")
        assert raw_prompts[-1][1]["content"][-1]["text"] == "\nCandidate answer:\n\\boxed{42}"

    class FakeActor:
        def __init__(self, output):
            async def remote(chunk):
                assert len(chunk) == 4
                return output

            self.generate_sequences = SimpleNamespace(remote=remote)

    manager = object.__new__(AgentLoopManager)
    manager.config = config
    manager.stream_teacher_with_rollout = False
    manager.agent_loop_workers = [FakeActor(output) for output in outputs]
    manager._performance_metrics = lambda metrics, output: {"test_rows": len(output)}

    async def generate():
        return await manager.generate_sequences(gen_batch)

    generated = asyncio.run(generate())
    assert set(generated.meta_info["reward_extra_keys"]) == {"accuracy", "format", "tool_efficiency", "raw_format"}
    assert generated.meta_info["timing"]["test_rows"] == 12
    row_ids = generated.non_tensor_batch.pop("__rollout_row_id")
    trained = batch.repeat(2, interleave=True).select_idxs(row_ids).union(generated)
    assert len(trained) == 12
    assert trained.non_tensor_batch["is_replay_judge"].tolist() == [False] * 8 + [True] * 4
    assert trained.non_tensor_batch["is_replay_judge"].dtype == object
    assert all(len(value) == len(trained) for value in trained.non_tensor_batch.values())
