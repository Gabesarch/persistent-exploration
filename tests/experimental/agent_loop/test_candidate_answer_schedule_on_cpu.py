"""Exercise candidate deadlines with real generation, grading, branching and rollback."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from test_forced_action_tags_on_cpu import (
    _FakeTokenizer,
    _make_agent_data,
    _make_conditional_retry_loop,
)

from verl.experimental.agent_loop.agent_loop import AgentLoopMetrics, AgentLoopOutput
from verl.experimental.agent_loop.tool_agent_loop import AgentState
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.utils.forced_action_tags import FORCE_ACTION_REQUIRED_TAG_KEY
from verl.workers.config.rollout import MultiTurnConfig
from verl.workers.rollout.replica import TokenOutput

ANSWER = [30, 31, 1, 20, 21, 22, 50, 23]
TOOL = [30, 31, 1, 10, 40, 11]
TRAJECTORY = {"step": 1, "sample_index": 0, "rollout_n": 8, "validate": False}


class PolicyServer:
    def __init__(self, data):
        self.data = data
        self.calls = []
        self.voluntary = set()
        self.malformed = False

    async def generate(self, *, prompt_ids, sampling_params, **kwargs):
        del kwargs
        tag = sampling_params.get("extra_args", {}).get(FORCE_ACTION_REQUIRED_TAG_KEY)
        schedule = self.data.candidate_answer_schedule
        candidate_count = len(schedule.history) if schedule else 0
        voluntary = (candidate_count, self.data.assistant_turns + 1) in self.voluntary
        answer = tag == "<answer>" or (tag is None and voluntary)
        ids = ([30, 31, 1] if self.malformed else ANSWER) if answer else TOOL
        self.calls.append({"tag": tag, "prompt": list(prompt_ids), "ids": ids})
        return TokenOutput(token_ids=list(ids), log_probs=[-0.1] * len(ids), extra_fields={})


class ToolParser:
    async def extract_tool_calls(self, ids, tools=None):
        del tools
        return "", [FunctionCall(name="image_zoom_in_tool", arguments="{}")] if 10 in ids else []


def make_case(*, first=7, grades=(0.0, 1.0), trajectory=None, enabled=True, mask_prefix=False):
    data = _make_agent_data()
    server = PolicyServer(data)
    tokenizer = _FakeTokenizer()
    tokenizer.text_to_ids["\n</answer><|im_end|>\n"] = [23, 2]
    loop = _make_conditional_retry_loop(tokenizer, server, ToolParser(), None)
    loop.response_length = 10000
    loop.force_answer_on_final_turn = True
    loop.force_final_answer_think_prefill = (
        "This is the final turn, so I need to answer from the evidence gathered so far."
    )
    loop.conditional_answer_retry_suffix = ""
    loop.conditional_answer_retry_enable_validation = False
    loop.conditional_answer_retry_mask_canonical_prefix = mask_prefix
    loop.randomized_max_assistant_turns_enabled = False
    loop.candidate_answer_schedule_enabled = enabled
    loop.candidate_answer_first_turns = [first] if first else [4, 5, 6, 7]
    loop.candidate_answer_followup_min_turn_gap = 2
    loop.candidate_answer_schedule_seed = 42
    loop._validate_candidate_answer_schedule_config()
    loop.conditional_answer_retry_branch_enabled = True
    loop.conditional_answer_retry_branch_enable_validation = False
    loop.conditional_answer_retry_branch_mask_mode = "divergence_only"
    loop.conditional_answer_retry_branch_use_graded_rewards = True
    loop.conditional_answer_retry_branch_baseline_weight_key = "gdpo_baseline_weight"
    loop.conditional_answer_retry_critic_enabled = True
    loop.conditional_answer_retry_critic_oracle_supervision_enabled = True
    loop.conditional_answer_retry_critic_enable_validation = False
    loop._retry_critic_accept_token_id = 32
    loop._retry_critic_retry_token_id = 49
    loop._retry_critic_prompt_prefix_ids = AsyncMock(return_value=[90, 91])
    loop._grade_current_response = AsyncMock(
        side_effect=[{"accuracy": grade, "score": 0.9 * grade + 0.1, "format": 1.0} for grade in grades]
    )
    loop._initialize_candidate_answer_schedule(data, trajectory or TRAJECTORY)
    return loop, data, server


async def advance(loop, data, *, validate=False):
    state = await loop._handle_generating_state(
        data, {"temperature": 1.0}, rollout_kwargs={"_debug_trajectory": {"validate": validate}}
    )
    if state == AgentState.PROCESSING_TOOLS:
        # Substitute only the external tool execution; retain all generation and retry logic.
        data.messages.append({"role": "tool", "content": "observed region"})
        data.prompt_ids.append(100)
        data.response_mask.append(0)
        data.retry_sft_mask.append(0)
        data.retry_critic_mask.append(0)
        data.response_logprobs.append(0.0)
        data.user_turns += 1
        loop._record_last_user_tool_prompt_block(
            agent_data=data,
            prompt_start=len(data.prompt_ids) - 1, prompt_end=len(data.prompt_ids),
            response_mask_start=len(data.response_mask) - 1, response_mask_end=len(data.response_mask),
            response_logprobs_start=len(data.response_logprobs) - 1,
            response_logprobs_end=len(data.response_logprobs),
            block_start_index=len(data.messages) - 1, block_images=[], block_videos=[],
        )
    return state


def complete(loop, data, **kwargs):
    async def drive():
        for _ in range(30):
            if await advance(loop, data, **kwargs) == AgentState.TERMINATED:
                return
        pytest.fail("rollout did not terminate")

    asyncio.run(drive())


@pytest.mark.parametrize("first", [4, 5, 6, 7])
@pytest.mark.parametrize("credit", [0.5, 1.0])
def test_first_deadline_forces_and_accepts_positive_credit(first, credit):
    loop, data, server = make_case(first=first, grades=(credit,))
    complete(loop, data)
    history = data.candidate_answer_schedule.history
    assert len(history) == 1
    assert history[0]["retained_turn"] == first
    assert history[0]["sampled_deadline"] == first
    assert history[0]["action"] == "accept"
    assert history[0]["accuracy"] == credit
    assert history[0]["forcing_reason"] == "candidate_deadline"
    assert all(call["tag"] is None for call in server.calls[:-1])
    assert server.calls[-1]["tag"] == "<answer>"
    assert 72 not in server.calls[-1]["prompt"]  # No misleading final-turn prefill.
    assert data.extra_fields["retry_critic_label"] == "A"
    assert sum(data.retry_critic_mask) == 1
    assert not any(a and b for a, b in zip(data.response_mask, data.retry_critic_mask, strict=True))
    assert not data.branch_outputs


def test_turn_seven_rollback_preserves_schedule_branch_and_cost():
    loop, data, server = make_case(grades=(0.0, 0.0, 1.0))
    complete(loop, data)
    schedule = data.candidate_answer_schedule
    first, second, final = schedule.history
    assert first["retained_turn"] == 7
    assert second["retained_turn"] == schedule.followup_deadline in (9, 10)
    assert second["sampled_deadline"] == schedule.followup_deadline
    assert final["retained_turn"] == 11
    assert final["sampled_deadline"] is None
    assert final["forcing_reason"] == "budget"
    assert schedule.next_deadline is None
    assert server.calls[7]["tag"] == "<tool_call>"  # Retry reuses turn 7, never the first deadline.
    assert schedule.generated_policy_turns == len(server.calls) == 13
    assert schedule.generated_policy_tokens == sum(len(call["ids"]) for call in server.calls)
    assert server.calls[-1]["prompt"][-1] == 72
    assert data.assistant_turns == 11
    assert len(data.branch_outputs) == 2
    for ordinal, branch in enumerate(data.branch_outputs, 1):
        assert branch.extra_fields["retry_critic_label"] == "R"
        assert branch.extra_fields["gdpo_baseline_weight"] == 0.0
        assert sum(branch.retry_critic_mask) == 1
        assert sum(branch.response_mask) == len(ANSWER)
        assert not any(a and b for a, b in zip(branch.response_mask, branch.retry_critic_mask, strict=True))
        metadata = branch.extra_fields["candidate_answer_schedule"]
        assert metadata["candidate_count"] == ordinal  # Immutable snapshot of history.
        assert metadata["history"][-1]["action"] == "retry"
        assert len(branch.response_ids) == len(branch.response_mask) == len(branch.retry_critic_mask)


@pytest.mark.parametrize("final_credit", [0.0, 0.5, 1.0])
def test_canonical_mask_advances_on_actual_retries_and_preserves_context_and_critic(final_credit):
    grades = (0.0, 0.0, final_credit)
    loop, data, server = make_case(grades=grades, mask_prefix=True)
    pivots = []

    async def drive():
        while len(data.branch_outputs) < 2:
            previous_retries = len(data.branch_outputs)
            await advance(loop, data)
            if len(data.branch_outputs) > previous_retries:
                pivots.append(len(data.response_mask))
                assert not any(data.response_mask)
        return len(data.response_mask)

    last_pivot = asyncio.run(drive())
    assert pivots[0] < pivots[1]
    complete(loop, data)
    legacy_loop, legacy_data, legacy_server = make_case(grades=grades)
    complete(legacy_loop, legacy_data)

    assert server.calls == legacy_server.calls
    assert data.prompt_ids == legacy_data.prompt_ids
    assert data.response_logprobs == legacy_data.response_logprobs
    assert data.retry_critic_mask == legacy_data.retry_critic_mask
    assert data.retry_sft_mask == legacy_data.retry_sft_mask
    assert data.candidate_answer_schedule.history == legacy_data.candidate_answer_schedule.history
    assert data.assistant_turns == legacy_data.assistant_turns == 11
    assert data.branch_outputs == legacy_data.branch_outputs
    assert not any(data.response_mask[:last_pivot])
    assert data.response_mask[last_pivot:] == legacy_data.response_mask[last_pivot:]
    assert sum(data.response_mask) > 0
    metadata = loop._canonical_prefix_mask_metadata(data)["canonical_prefix_mask"]
    assert metadata["prefix_tokens"] == last_pivot  # No pivot advance on an unretryable final failure.
    assert metadata["active_prefix_policy_tokens"] == 0
    assert metadata["masked_policy_tokens"] == sum(legacy_data.response_mask[:last_pivot])
    assert metadata["active_suffix_policy_tokens"] == sum(data.response_mask)


@pytest.mark.parametrize("validate", [False, True])
def test_prefix_mask_does_not_change_no_retry_or_validation_rollouts(validate):
    loop, data, _ = make_case(
        grades=(0.5,), mask_prefix=True, trajectory={**TRAJECTORY, "validate": validate}
    )
    complete(loop, data, validate=validate)
    assert sum(data.response_mask) > 0
    assert loop._canonical_prefix_mask_metadata(data) == {}
    # Even if oracle retries were explicitly enabled for validation, the new mask stays off.
    original_mask = list(data.response_mask)
    loop._mask_canonical_retry_prefix(data, {"_debug_trajectory": {"validate": True}})
    assert data.response_mask == original_mask


def test_voluntary_first_and_followup_consume_deadlines_without_resampling():
    loop, data, server = make_case(grades=(0.0, 0.0, 1.0))
    server.voluntary = {(0, 2), (1, 3)}
    complete(loop, data)
    schedule = data.candidate_answer_schedule
    first, second, final = schedule.history
    assert first["retained_turn"] == 2 and not first["forced"]
    assert 4 <= schedule.followup_deadline <= 10
    assert second["retained_turn"] == 3 and not second["forced"]
    assert final["retained_turn"] == 11
    assert final["forcing_reason"] == "budget"
    assert len([call for call in server.calls if call["tag"] == "<answer>"]) == 1


def test_sampling_is_deterministic_and_reaches_all_allowed_deadlines():
    loop, _, _ = make_case(first=None)
    firsts, followups = set(), set()
    for index in range(200):
        trajectory = {**TRAJECTORY, "sample_index": index}
        a, b = _make_agent_data(), _make_agent_data()
        b.request_id = "different-random-request-id"
        loop._initialize_candidate_answer_schedule(a, trajectory)
        loop._initialize_candidate_answer_schedule(b, trajectory)
        assert a.candidate_answer_schedule.first_deadline == b.candidate_answer_schedule.first_deadline
        firsts.add(a.candidate_answer_schedule.first_deadline)
        a.assistant_turns = b.assistant_turns = 7
        for data in (a, b):
            loop._record_candidate_answer(data, {"accuracy": 0.0}, action="retry", reason="incorrect_answer")
        assert a.candidate_answer_schedule.followup_deadline == b.candidate_answer_schedule.followup_deadline
        followups.add(a.candidate_answer_schedule.followup_deadline)
    assert firsts == {4, 5, 6, 7}
    assert followups == {9, 10}


@pytest.mark.parametrize("first, expected", [(8, 10), (9, None), (10, None)])
def test_followup_range_at_total_cap_boundary(first, expected):
    loop, data, _ = make_case(first=first)
    data.assistant_turns = first
    loop._record_candidate_answer(data, {"accuracy": 0.0}, action="retry", reason="incorrect_answer")
    assert data.candidate_answer_schedule.followup_deadline == expected


@pytest.mark.parametrize("reason", ["max_assistant_turns", "max_user_turns", "response_length"])
def test_actual_budget_exhaustion_preempts_schedule(reason):
    loop, data, server = make_case(grades=(0.0,))
    if reason == "max_assistant_turns":
        data.assistant_turns = 10
    elif reason == "max_user_turns":
        data.user_turns = 10
    else:
        loop.response_length = 7  # Inside the configured four-token answer margin after this turn.
        data.response_mask = [1] * 3
        data.prompt_ids += [31] * 3
        data.retry_sft_mask = data.retry_critic_mask = [0] * 3
    complete(loop, data)
    history = data.candidate_answer_schedule.history
    assert len(history) == 1
    assert history[0]["action"] == "terminate"
    assert history[0]["reason"] == reason
    assert data.candidate_answer_schedule.followup_deadline is None
    assert not data.branch_outputs
    assert len(server.calls) == 1


def test_malformed_forced_candidate_terminates_without_retry():
    loop, data, server = make_case()
    server.malformed = True
    complete(loop, data)
    assert data.extra_fields["forced_structure_error_reason"] == "missing_final_answer"
    assert data.assistant_turns == 7
    assert not data.candidate_answer_schedule.history
    loop._grade_current_response.assert_not_called()


@pytest.mark.parametrize("enabled,validate", [(False, False), (True, True)])
def test_disabled_and_validation_rollouts_preserve_legacy_forcing(enabled, validate):
    loop, data, server = make_case(enabled=enabled, trajectory={**TRAJECTORY, "validate": validate}, grades=(1.0,))
    assert data.candidate_answer_schedule is None
    assert loop._candidate_answer_schedule_metadata(data) == {}
    complete(loop, data, validate=validate)
    assert len(server.calls) == 11
    assert all(call["tag"] is None for call in server.calls[:-1])
    assert server.calls[-1]["tag"] == "<answer>"


def test_validation_still_uses_learned_critic():
    loop, data, server = make_case(trajectory={**TRAJECTORY, "validate": True})
    loop.conditional_answer_retry_critic_enable_validation = True
    loop._maybe_retry_answer_with_critic = AsyncMock(return_value=None)
    server.voluntary = {(0, 2)}
    complete(loop, data, validate=True)
    loop._maybe_retry_answer_with_critic.assert_awaited_once()
    loop._grade_current_response.assert_not_called()
    assert data.candidate_answer_schedule is None


@pytest.mark.parametrize(
    "key,value",
    [
        ("max_assistant_turns", None),
        ("max_assistant_turns", True),
        ("candidate_answer_first_turns", []),
        ("candidate_answer_first_turns", [0]),
        ("candidate_answer_first_turns", [11]),
        ("candidate_answer_first_turns", [4, 4]),
        ("candidate_answer_first_turns", [True]),
        ("candidate_answer_first_turns", 4),
        ("candidate_answer_followup_min_turn_gap", 1),
        ("candidate_answer_followup_min_turn_gap", 11),
        ("candidate_answer_followup_min_turn_gap", 2.5),
        ("candidate_answer_schedule_seed", True),
        ("randomized_max_assistant_turns_enabled", True),
        ("force_tool_calls_until_final_turn", True),
        ("conditional_answer_retry_enabled", False),
        ("conditional_answer_retry_force_next_tool_call", False),
        ("force_answer_on_final_turn", False),
    ],
)
def test_invalid_schedule_configs_fail_early(key, value):
    loop, _, _ = make_case()
    setattr(loop, key, value)
    with pytest.raises(ValueError):
        loop._validate_candidate_answer_schedule_config()


def test_final_trace_keeps_candidate_history_removed_from_live_trajectory(tmp_path):
    loop, data, _ = make_case(mask_prefix=True)
    complete(loop, data)
    loop._debug_dir = lambda: str(tmp_path)
    loop._debug_base_record = lambda *args: {"request_id": data.request_id}
    loop._debug_max_text_chars = lambda: None
    loop._debug_append_event = lambda event: None
    output = AgentLoopOutput(
        prompt_ids=[100],
        response_ids=data.prompt_ids[-len(data.response_mask) :],
        response_mask=data.response_mask,
        retry_critic_mask=data.retry_critic_mask,
        num_turns=data.assistant_turns + data.user_turns + 1,
        metrics=AgentLoopMetrics(),
        extra_fields=dict(data.extra_fields),
        branch_outputs=data.branch_outputs,
    )
    asyncio.run(loop._debug_log_sample_final(output, data, {"raw_prompt": []}, TRAJECTORY))
    trace = json.loads((tmp_path / "sample_traces.jsonl").read_text())
    history = trace["candidate_answer_schedule"]["history"]
    assert [record["accuracy"] for record in history] == [0.0, 1.0]
    assert trace["conditional_answer_retry_branch_outputs"][0]["candidate_answer_schedule"]["candidate_count"] == 1
    assert trace["canonical_prefix_mask"]["active_prefix_policy_tokens"] == 0
    assert trace["canonical_prefix_mask"]["masked_policy_tokens"] > 0
    assert trace["canonical_prefix_mask"]["active_suffix_policy_tokens"] > 0


def test_new_recipe_composes_with_schedule_mask_diagnostics_and_half_critic_weight(monkeypatch):
    root = Path(__file__).resolve().parents[3]
    for key, value in {
        "VERL_ROOT": str(root),
        "HF_HOME": "/tmp/hf",
        "IMAGE_ROOT": "/tmp/images",
        "SAVE_PATH": "/tmp/save",
        "VERO2_CROP_CACHE_ROOT": "/tmp/crops",
    }.items():
        monkeypatch.setenv(key, value)
    base_name = "branchstrata20_fresh_lr1e5_critic"
    name = base_name + "_candidate_schedule_v1"
    with initialize_config_dir(config_dir=str(root / "vero2_scripts/config"), version_base=None):
        base = OmegaConf.to_container(compose(config_name=base_name), resolve=False)
        new = OmegaConf.to_container(compose(config_name=name), resolve=False)
    multi_turn = new["actor_rollout_ref"]["rollout"]["multi_turn"]
    assert multi_turn["max_assistant_turns"] == 11
    assert multi_turn["candidate_answer_schedule_enabled"] is True
    assert multi_turn["candidate_answer_first_turns"] == [4, 5, 6, 7]
    assert multi_turn["candidate_answer_followup_min_turn_gap"] == 2
    assert multi_turn["candidate_answer_schedule_seed"] == 42
    assert multi_turn["randomized_max_assistant_turns_enabled"] is False
    assert multi_turn["conditional_answer_retry_mask_canonical_prefix"] is True
    strata = new["actor_rollout_ref"]["actor"]["policy_loss"]["branch_strata"]
    assert strata["log_advantage_mass"] is True
    assert new["trainer"]["experiment_name"] == name + "_20260930"
    assert new["trainer"]["total_training_steps"] == 200
    assert float(new["actor_rollout_ref"]["actor"]["optim"]["lr"]) == 1e-5
    critic = new["actor_rollout_ref"]["actor"]["policy_loss"]["retry_critic"]
    assert critic["loss_fraction"] == 0.005
    assert base["actor_rollout_ref"]["actor"]["policy_loss"]["retry_critic"]["loss_fraction"] == 0.01
    critic["loss_fraction"] = 0.01
    multi_turn["candidate_answer_schedule_enabled"] = False
    multi_turn["conditional_answer_retry_mask_canonical_prefix"] = False
    strata["log_advantage_mass"] = False
    new["trainer"]["experiment_name"] = base["trainer"]["experiment_name"]
    assert new == base
    defaults = MultiTurnConfig()
    assert defaults.candidate_answer_schedule_enabled is False
    assert defaults.candidate_answer_first_turns == [4, 5, 6, 7]
    assert defaults.conditional_answer_retry_mask_canonical_prefix is False
