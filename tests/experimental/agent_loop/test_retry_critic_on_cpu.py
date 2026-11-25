import asyncio
import math
from unittest.mock import AsyncMock, Mock

from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.workers.rollout.replica import TokenOutput


def _agent_data():
    data = AgentData([], None, None, {}, "request", {})
    data.prompt_ids = [10, 11, 12]
    data.response_mask = [1, 1]
    data.retry_sft_mask = [0, 0]
    data.retry_critic_mask = [0, 0]
    data.response_logprobs = [-0.1, -0.2]
    data.assistant_turns = 1
    data.debug_logging_enabled = False
    return data


def _oracle_loop():
    loop = object.__new__(ToolAgentLoop)
    loop.conditional_answer_retry_correct_accuracy_threshold = 0.0
    loop.conditional_answer_retry_suffix = ""
    loop.conditional_answer_retry_force_next_tool_call = True
    loop.response_length = 100
    loop._append_retry_critic_supervision = AsyncMock(return_value=True)
    loop._log_incorrect_answer_retry = Mock()
    loop._log_incorrect_answer_retry_skipped = Mock()
    return loop


def test_oracle_correct_answer_gets_accept_supervision():
    loop = _oracle_loop()
    loop._grade_current_response = AsyncMock(return_value={"accuracy": 1.0, "score": 1.0})
    data = _agent_data()
    result = asyncio.run(
        loop._maybe_retry_incorrect_answer(data, {"assistant_turns": 0}, "candidate", {}, False)
    )
    assert result is None
    loop._append_retry_critic_supervision.assert_awaited_once_with(
        data, grade={"accuracy": 1.0, "score": 1.0}, label="A"
    )
    assert data.extra_fields["__agent_loop_precritic_grade"]["accuracy"] == 1.0


def test_oracle_wrong_retried_answer_gets_retry_supervision_on_branch():
    loop = _oracle_loop()
    loop._grade_current_response = AsyncMock(return_value={"accuracy": 0.0, "score": 0.0})
    loop._incorrect_answer_retry_skipped_reason = Mock(return_value=None)
    loop._maybe_append_wrong_answer_branch = Mock()
    loop._rollback_generation_state = Mock()
    loop._append_conditional_retry_suffix_to_previous_message = AsyncMock()
    data = _agent_data()
    result = asyncio.run(
        loop._maybe_retry_incorrect_answer(data, {"assistant_turns": 0}, "candidate", {}, False)
    )
    assert result is AgentState.GENERATING
    loop._append_retry_critic_supervision.assert_awaited_once_with(
        data, grade={"accuracy": 0.0, "score": 0.0}, label="R"
    )
    loop._maybe_append_wrong_answer_branch.assert_called_once()


def test_oracle_wrong_budget_exhausted_answer_still_gets_retry_supervision():
    loop = _oracle_loop()
    loop._grade_current_response = AsyncMock(return_value={"accuracy": 0.0, "score": 0.0})
    loop._incorrect_answer_retry_skipped_reason = Mock(return_value="max_assistant_turns")
    data = _agent_data()
    result = asyncio.run(
        loop._maybe_retry_incorrect_answer(data, {"assistant_turns": 1}, "candidate", {}, False)
    )
    assert result is None
    loop._append_retry_critic_supervision.assert_awaited_once_with(
        data, grade={"accuracy": 0.0, "score": 0.0}, label="R"
    )
    assert data.extra_fields["__agent_loop_precritic_grade"]["accuracy"] == 0.0


def test_missing_or_error_grade_is_excluded_from_supervision():
    loop = object.__new__(ToolAgentLoop)
    loop.conditional_answer_retry_critic_enabled = True
    loop.conditional_answer_retry_critic_oracle_supervision_enabled = True
    assert loop._retry_critic_grade_is_supervisable({"accuracy": 0.0})
    assert not loop._retry_critic_grade_is_supervisable({"accuracy": 0.0, "error": "failed"})
    assert not loop._retry_critic_grade_is_supervisable({})


class _Server:
    def __init__(self, output=None, error=None):
        self.output = output
        self.error = error
        self.calls = []

    async def generate(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        return self.output


def _validation_loop(output=None, error=None, skipped_reason=None):
    loop = object.__new__(ToolAgentLoop)
    loop.server_manager = _Server(output=output, error=error)
    loop._retry_critic_prompt_prefix_ids = AsyncMock(return_value=[70, 71])
    loop._retry_critic_accept_token_id = 32
    loop._retry_critic_retry_token_id = 49
    loop.conditional_answer_retry_critic_decision_threshold = 0.5
    loop.conditional_answer_retry_max_retries = None
    loop.conditional_answer_retry_suffix = ""
    loop.conditional_answer_retry_force_next_tool_call = True
    loop.response_length = 100
    loop.image_data = None
    loop._incorrect_answer_retry_skipped_reason = Mock(return_value=skipped_reason)
    loop._rollback_generation_state = Mock()
    loop._append_conditional_retry_suffix_to_previous_message = AsyncMock()
    return loop


def test_validation_accept_discards_sidecar_and_keeps_candidate():
    loop = _validation_loop(TokenOutput(token_ids=[32], log_probs=[math.log(0.8)]))
    data = _agent_data()
    original_prompt = list(data.prompt_ids)
    result = asyncio.run(
        loop._maybe_retry_answer_with_critic(
            agent_data=data,
            generation_snapshot={"assistant_turns": 0},
            assistant_message="candidate",
            ignore_termination=False,
        )
    )
    assert result is None
    assert data.prompt_ids == original_prompt
    assert data.extra_fields["retry_critic_validation_decision_log"][-1]["controller_action"] == "accept"
    params = loop.server_manager.calls[0]["sampling_params"]
    assert params["max_tokens"] == 1
    assert params["temperature"] == 0.0
    assert params["logprobs"] is True
    assert params["allowed_token_ids"] == [32, 49]


def test_validation_feasible_retry_rolls_back_and_forces_tool_path():
    loop = _validation_loop(TokenOutput(token_ids=[49], log_probs=[math.log(0.9)]))
    data = _agent_data()
    result = asyncio.run(
        loop._maybe_retry_answer_with_critic(
            agent_data=data,
            generation_snapshot={"assistant_turns": 0},
            assistant_message="candidate",
            ignore_termination=False,
        )
    )
    assert result is AgentState.GENERATING
    loop._rollback_generation_state.assert_called_once()
    loop._append_conditional_retry_suffix_to_previous_message.assert_awaited_once()
    assert data.force_next_tool_call is True
    assert data.extra_fields["retry_critic_validation_retries"] == 1


def test_validation_limit_blocked_retry_forces_accept():
    loop = _validation_loop(
        TokenOutput(token_ids=[49], log_probs=[math.log(0.9)]), skipped_reason="max_assistant_turns"
    )
    data = _agent_data()
    result = asyncio.run(
        loop._maybe_retry_answer_with_critic(
            agent_data=data,
            generation_snapshot={"assistant_turns": 0},
            assistant_message="candidate",
            ignore_termination=False,
        )
    )
    assert result is None
    assert data.extra_fields["retry_critic_validation_forced_accepts"] == 1
    assert (
        data.extra_fields["retry_critic_validation_decision_log"][-1]["skip_or_error_reason"]
        == "max_assistant_turns"
    )


def test_validation_generation_failure_accepts_candidate():
    loop = _validation_loop(error=RuntimeError("boom"))
    data = _agent_data()
    original_prompt = list(data.prompt_ids)
    result = asyncio.run(
        loop._maybe_retry_answer_with_critic(
            agent_data=data,
            generation_snapshot={"assistant_turns": 0},
            assistant_message="candidate",
            ignore_termination=False,
        )
    )
    assert result is None
    assert data.prompt_ids == original_prompt
    assert data.extra_fields["retry_critic_validation_failures"] == 1
    assert data.extra_fields["retry_critic_validation_decision_log"][-1]["controller_action"] == "accept_fallback"
