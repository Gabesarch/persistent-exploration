import asyncio
from typing import Any, Optional

import pytest
import torch
from vllm import SamplingParams

from verl.experimental.agent_loop.tool_agent_loop import (
    AgentData,
    AgentState,
    ToolAgentLoop,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.utils.forced_action_tags import (
    FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY,
    FORCE_ACTION_LOGITS_PROCESSOR_FQCN,
    FORCE_ACTION_REQUIRED_TAG_KEY,
    FORCE_ACTION_TAG_IDS_KEY,
    ForceActionTagAfterThinkRequestProcessor,
    ensure_forced_action_logits_processor,
)
from verl.utils.vllm_forced_action_logits_processor import ForceActionTagAfterThinkLogitsProcessor
from verl.workers.rollout.replica import TokenOutput


class _FakeTokenizer:
    def __init__(self):
        self.text_to_ids = {
            "</think>": [1],
            "</think>\n": [1, 2],
            "</think>\n\n": [1, 2, 2],
            "<tool_call>": [10],
            "\n<tool_call>": [2, 10],
            "<answer>": [20, 21, 22],
            "\n<answer>": [2, 20, 21, 22],
        }
        self.id_to_text = {
            1: "</think>",
            2: "\n",
            10: "<tool_call>",
            11: "</tool_call>",
            20: "<",
            21: "answer",
            22: ">",
            30: "<think>",
            31: "reason",
            40: '{"name":"image_zoom_in_tool","arguments":{}}',
            50: "final",
        }

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text in self.text_to_ids:
            return list(self.text_to_ids[text])
        raise AssertionError(f"unexpected encode text: {text}")

    def decode(self, ids: list[int], skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(self.id_to_text.get(int(token_id), "") for token_id in ids)


def _processor_for_tag(tokenizer: _FakeTokenizer, required_tag: str) -> ForceActionTagAfterThinkRequestProcessor:
    return ForceActionTagAfterThinkRequestProcessor(
        after_think_ids_options=[
            tokenizer.encode("</think>\n\n", add_special_tokens=False),
            tokenizer.encode("</think>\n", add_special_tokens=False),
            tokenizer.encode("</think>", add_special_tokens=False),
        ],
        tag_ids=tokenizer.encode(required_tag, add_special_tokens=False),
    )


def _forced_token(processor: ForceActionTagAfterThinkRequestProcessor, token_ids: list[int]) -> int:
    logits = torch.zeros(64)
    processed = processor(token_ids, logits)
    finite = torch.isfinite(processed).nonzero().flatten().tolist()
    assert len(finite) == 1
    return finite[0]


def test_force_action_tag_after_think_leaves_unmatched_logits_unchanged():
    processor = _processor_for_tag(_FakeTokenizer(), "<tool_call>")
    logits = torch.arange(64, dtype=torch.float32)

    processed = processor([30, 31], logits.clone())

    assert torch.equal(processed, logits)


def test_force_action_tag_after_think_forces_tool_call_after_think_suffix():
    processor = _processor_for_tag(_FakeTokenizer(), "<tool_call>")

    assert _forced_token(processor, [30, 31, 1]) == 10
    assert _forced_token(processor, [30, 31, 1, 2]) == 10
    assert _forced_token(processor, [30, 31, 1, 2, 2]) == 10


def test_force_action_tag_after_think_forces_multi_token_answer_prefix():
    processor = _processor_for_tag(_FakeTokenizer(), "<answer>")

    assert _forced_token(processor, [30, 31, 1, 2]) == 20
    assert _forced_token(processor, [30, 31, 1, 2, 20]) == 21
    assert _forced_token(processor, [30, 31, 1, 2, 20, 21]) == 22


def test_vllm_adapter_validates_and_builds_request_processor():
    params = SamplingParams(
        extra_args={
            FORCE_ACTION_REQUIRED_TAG_KEY: "<tool_call>",
            FORCE_ACTION_TAG_IDS_KEY: [10],
            FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY: [[1, 2], [1]],
        }
    )

    ForceActionTagAfterThinkLogitsProcessor.validate_params(params)
    processor = ForceActionTagAfterThinkLogitsProcessor(None, torch.device("cpu"), False)
    request_processor = processor.new_req_logits_processor(params)

    assert isinstance(request_processor, ForceActionTagAfterThinkRequestProcessor)
    assert _forced_token(request_processor, [30, 31, 1, 2]) == 10


def test_vllm_adapter_is_disabled_without_extra_args():
    params = SamplingParams()
    processor = ForceActionTagAfterThinkLogitsProcessor(None, torch.device("cpu"), False)

    ForceActionTagAfterThinkLogitsProcessor.validate_params(params)

    assert processor.new_req_logits_processor(params) is None


def test_vllm_adapter_rejects_invalid_payload():
    params = SamplingParams(
        extra_args={
            FORCE_ACTION_REQUIRED_TAG_KEY: "<bad>",
            FORCE_ACTION_TAG_IDS_KEY: [10],
            FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY: [[1, 2]],
        }
    )

    with pytest.raises(ValueError, match="unsupported forced action tag"):
        ForceActionTagAfterThinkLogitsProcessor.validate_params(params)


class _FakeServerManager:
    def __init__(self, token_ids: list[int]):
        self.token_ids = token_ids
        self.last_sampling_params: Optional[dict[str, Any]] = None

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        del request_id, prompt_ids, image_data, video_data
        self.last_sampling_params = sampling_params
        return TokenOutput(token_ids=self.token_ids, extra_fields={})


class _FakeToolParser:
    def __init__(self, tool_calls: list[FunctionCall]):
        self.tool_calls = tool_calls

    async def extract_tool_calls(self, responses_ids, tools=None):
        del responses_ids, tools
        return "", self.tool_calls


class _InlineExecutorLoop:
    def run_in_executor(self, executor, func):
        del executor
        future = asyncio.get_running_loop().create_future()
        future.set_result(func())
        return future


def _make_loop(tokenizer: _FakeTokenizer, server_manager: _FakeServerManager, tool_parser: _FakeToolParser):
    loop = object.__new__(ToolAgentLoop)
    loop.force_tool_calls_until_final_turn = True
    loop.final_answer_token_margin = 2048
    loop.max_assistant_turns = 11
    loop.max_user_turns = 10
    loop.response_length = 24576
    loop.server_manager = server_manager
    loop.tokenizer = tokenizer
    loop.loop = _InlineExecutorLoop()
    loop.tools = {}
    loop.tool_parser = tool_parser
    loop.interaction_config_file = None
    loop._debug_enabled_for_agent = lambda agent_data: False
    return loop


def _make_agent_data() -> AgentData:
    agent_data = AgentData(
        messages=[],
        image_data=[],
        video_data=[],
        metrics={},
        request_id="request-1",
        tools_kwargs={},
    )
    agent_data.prompt_ids = [100]
    return agent_data


def test_forced_tool_turn_adds_extra_args_without_bad_words_or_request_logits_processor():
    tokenizer = _FakeTokenizer()
    server_manager = _FakeServerManager([30, 31, 1, 2, 10, 40, 11])
    tool_parser = _FakeToolParser([FunctionCall(name="image_zoom_in_tool", arguments="{}")])

    async def run_test():
        loop = _make_loop(tokenizer, server_manager, tool_parser)
        return await loop._handle_generating_state(
            _make_agent_data(),
            {"temperature": 1.0, "extra_args": {"existing": "kept"}},
        )

    state = asyncio.run(run_test())

    assert state == AgentState.PROCESSING_TOOLS
    assert "bad_words" not in server_manager.last_sampling_params
    assert "logits_processors" not in server_manager.last_sampling_params
    extra_args = server_manager.last_sampling_params["extra_args"]
    assert extra_args["existing"] == "kept"
    assert extra_args[FORCE_ACTION_REQUIRED_TAG_KEY] == "<tool_call>"
    assert extra_args[FORCE_ACTION_TAG_IDS_KEY] == [2, 10]
    assert extra_args[FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY] == [[1]]


def test_forced_tool_turn_marks_missing_tool_call_invalid():
    tokenizer = _FakeTokenizer()
    server_manager = _FakeServerManager([30, 31, 1, 2, 20, 21, 22, 50])
    tool_parser = _FakeToolParser([])
    agent_data = _make_agent_data()

    async def run_test():
        loop = _make_loop(tokenizer, server_manager, tool_parser)
        return await loop._handle_generating_state(agent_data, {"temperature": 1.0})

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    assert agent_data.extra_fields["forced_structure_error"] is True
    assert agent_data.extra_fields["forced_structure_error_reason"] == "missing_tool_call"


def test_forced_final_turn_requires_answer_tag():
    tokenizer = _FakeTokenizer()
    server_manager = _FakeServerManager([30, 31, 1, 2, 50])
    tool_parser = _FakeToolParser([])
    agent_data = _make_agent_data()
    agent_data.assistant_turns = 10

    async def run_test():
        loop = _make_loop(tokenizer, server_manager, tool_parser)
        return await loop._handle_generating_state(agent_data, {"temperature": 1.0})

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    assert server_manager.last_sampling_params["extra_args"][FORCE_ACTION_REQUIRED_TAG_KEY] == "<answer>"
    assert server_manager.last_sampling_params["extra_args"][FORCE_ACTION_TAG_IDS_KEY] == [2, 20, 21, 22]
    assert agent_data.extra_fields["forced_structure_error"] is True
    assert agent_data.extra_fields["forced_structure_error_reason"] == "missing_final_answer"


def test_vllm_server_helper_appends_forced_action_processor_when_force_mode_enabled():
    engine_kwargs = {"logits_processors": ["existing.module:Processor"]}

    ensure_forced_action_logits_processor(engine_kwargs)

    assert engine_kwargs["logits_processors"] == ["existing.module:Processor", FORCE_ACTION_LOGITS_PROCESSOR_FQCN]


def test_vllm_server_helper_does_not_duplicate_forced_action_processor():
    engine_kwargs = {"logits_processors": [FORCE_ACTION_LOGITS_PROCESSOR_FQCN]}

    ensure_forced_action_logits_processor(engine_kwargs)

    assert engine_kwargs["logits_processors"] == [FORCE_ACTION_LOGITS_PROCESSOR_FQCN]
