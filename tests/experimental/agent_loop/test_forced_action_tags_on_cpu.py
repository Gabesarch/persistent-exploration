import asyncio
import json
from types import SimpleNamespace
from typing import Any, Optional

import pytest
import torch
from omegaconf import OmegaConf
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
    VLLM_OPENAI_EXTRA_ARGS_KEY,
    ForceActionTagAfterThinkRequestProcessor,
    build_forced_action_extra_args,
    build_forced_action_openai_extra_body,
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
            "</": [80],
            "think": [81],
            ">": [82],
            "<": [83],
            "/": [84],
            "/think": [85],
            "<tool_call>": [10],
            "\n<tool_call>": [2, 10],
            "<answer>": [20, 21, 22],
            "\n<answer>": [2, 20, 21, 22],
            "\n<answer>\n\\boxed{": [2, 20, 21, 22, 2, 60, 61],
            "retry prefill": [70, 71],
            "This is the final turn, so I need to answer from the evidence gathered so far.": [72],
        }
        self.id_to_text = {
            1: "</think>",
            2: "\n",
            10: "<tool_call>",
            11: "</tool_call>",
            20: "<",
            21: "answer",
            22: ">",
            23: "</answer>",
            30: "<think>",
            31: "reason",
            40: '{"name":"image_zoom_in_tool","arguments":{}}',
            41: (
                '{"name":"image_zoom_in_tool","arguments":'
                '{"label":"shelf","bbox_2d":[100,100,500,500],"img_idx":0}}'
            ),
            50: "final",
            60: "\\boxed",
            61: "{",
            70: "retry ",
            71: "prefill",
            72: "This is the final turn, so I need to answer from the evidence gathered so far.",
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
    processed = processor([], token_ids, logits)
    finite = torch.isfinite(processed).nonzero().flatten().tolist()
    assert len(finite) == 1
    return finite[0]


def test_force_action_tag_after_think_leaves_unmatched_logits_unchanged():
    processor = _processor_for_tag(_FakeTokenizer(), "<tool_call>")
    logits = torch.arange(64, dtype=torch.float32)

    processed = processor([], [30, 31], logits.clone())

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


def test_force_action_tag_after_think_matches_across_prompt_output_boundary():
    processor = _processor_for_tag(_FakeTokenizer(), "<answer>")
    logits = torch.zeros(64)

    processed = processor([30, 31, 1], [], logits)
    assert torch.isfinite(processed).nonzero().flatten().tolist() == [20]

    processed = processor([30, 31, 1], [20, 21], torch.zeros(64))
    assert torch.isfinite(processed).nonzero().flatten().tolist() == [22]


def test_force_action_tag_after_think_is_only_forced_once_per_request():
    processor = _processor_for_tag(_FakeTokenizer(), "<answer>")

    assert _forced_token(processor, [30, 31, 1]) == 20
    assert _forced_token(processor, [30, 31, 1, 20]) == 21
    assert _forced_token(processor, [30, 31, 1, 20, 21]) == 22

    # The next decoding step observes the completed forced tag. A later
    # spurious </think> inside the answer must not inject a second <answer>.
    unchanged = torch.arange(64, dtype=torch.float32)
    processed = processor([], [30, 31, 1, 20, 21, 22], unchanged.clone())
    assert torch.equal(processed, unchanged)
    processed = processor([], [30, 31, 1, 20, 21, 22, 50, 1], unchanged.clone())
    assert torch.equal(processed, unchanged)


def test_shared_payload_accepts_special_and_decomposed_after_think_encodings():
    payload = build_forced_action_extra_args(
        _FakeTokenizer(),
        "<answer>",
        forced_suffix="\n\\boxed{",
    )

    assert payload[FORCE_ACTION_TAG_IDS_KEY] == [2, 20, 21, 22, 2, 60, 61]
    assert payload[FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY] == [
        [1],
        [80, 81, 82],
        [83, 84, 81, 82],
        [83, 85, 82],
    ]


def test_openai_payload_uses_vllm_xargs_not_internal_extra_args_field():
    body = build_forced_action_openai_extra_body(_FakeTokenizer(), "<answer>")

    assert set(body) == {VLLM_OPENAI_EXTRA_ARGS_KEY}
    assert body[VLLM_OPENAI_EXTRA_ARGS_KEY][FORCE_ACTION_REQUIRED_TAG_KEY] == "<answer>"
    assert body[VLLM_OPENAI_EXTRA_ARGS_KEY][FORCE_ACTION_TAG_IDS_KEY] == [2, 20, 21, 22]
    assert json.loads(body[VLLM_OPENAI_EXTRA_ARGS_KEY][FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY]) == [
        [1],
        [80, 81, 82],
        [83, 84, 81, 82],
        [83, 85, 82],
    ]


def test_vllm_adapter_decodes_openai_json_marker_options():
    body = build_forced_action_openai_extra_body(_FakeTokenizer(), "<answer>")
    params = SamplingParams(extra_args=body[VLLM_OPENAI_EXTRA_ARGS_KEY])
    processor = ForceActionTagAfterThinkLogitsProcessor(None, torch.device("cpu"), False)

    ForceActionTagAfterThinkLogitsProcessor.validate_params(params)
    request_processor = processor.new_req_logits_processor(params)

    assert _forced_token(request_processor, [30, 31, 1]) == 2
    assert _forced_token(request_processor, [30, 31, 1, 2]) == 20


def test_vllm_openai_schema_accepts_serialized_forced_action_payload():
    protocol = pytest.importorskip("vllm.entrypoints.openai.chat_completion.protocol")
    body = build_forced_action_openai_extra_body(_FakeTokenizer(), "<answer>")

    request = protocol.ChatCompletionRequest(
        model="qwen_vllm",
        messages=[{"role": "user", "content": "question"}],
        **body,
    )

    assert request.vllm_xargs == body[VLLM_OPENAI_EXTRA_ARGS_KEY]


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
        del request_id, image_data, video_data
        self.last_prompt_ids = list(prompt_ids)
        self.last_sampling_params = sampling_params
        return TokenOutput(token_ids=self.token_ids, extra_fields={})


class _SequencedFakeServerManager:
    def __init__(self, token_ids_by_call: list[list[int]]):
        self.token_ids_by_call = list(token_ids_by_call)
        self.sampling_params_by_call: list[dict[str, Any]] = []
        self.prompt_ids_by_call: list[list[int]] = []

    async def generate(
        self,
        request_id: str,
        *,
        prompt_ids: list[int],
        sampling_params: dict[str, Any],
        image_data: Optional[list[Any]] = None,
        video_data: Optional[list[Any]] = None,
    ) -> TokenOutput:
        del request_id, image_data, video_data
        self.prompt_ids_by_call.append(list(prompt_ids))
        self.sampling_params_by_call.append(sampling_params)
        token_ids = self.token_ids_by_call.pop(0)
        return TokenOutput(token_ids=token_ids, extra_fields={}, log_probs=[0.0] * len(token_ids))


class _FakeToolParser:
    def __init__(self, tool_calls: list[FunctionCall]):
        self.tool_calls = tool_calls

    async def extract_tool_calls(self, responses_ids, tools=None):
        del responses_ids, tools
        return "", self.tool_calls


class _SequencedFakeToolParser:
    def __init__(self, tool_calls_by_call: list[list[FunctionCall]]):
        self.tool_calls_by_call = list(tool_calls_by_call)

    async def extract_tool_calls(self, responses_ids, tools=None):
        del responses_ids, tools
        return "", self.tool_calls_by_call.pop(0)


class _InlineExecutorLoop:
    def run_in_executor(self, executor, func):
        del executor
        future = asyncio.get_running_loop().create_future()
        future.set_result(func())
        return future


def _make_loop(tokenizer: _FakeTokenizer, server_manager: _FakeServerManager, tool_parser: _FakeToolParser):
    loop = object.__new__(ToolAgentLoop)
    loop.force_tool_calls_until_final_turn = True
    loop.force_answer_on_final_turn = False
    loop.force_answer_on_final_turn_validation = None
    loop.final_answer_token_margin = 2048
    loop.force_final_answer_think_prefill = ""
    loop.force_final_answer_think_prefill_trainable = False
    loop.force_final_answer_prefix = ""
    loop.max_assistant_turns = 11
    loop.max_assistant_tokens_per_turn = None
    loop.max_user_turns = 10
    loop.response_length = 24576
    loop.server_manager = server_manager
    loop.tokenizer = tokenizer
    loop.loop = _InlineExecutorLoop()
    loop.tools = {}
    loop.tool_parser = tool_parser
    loop.interaction_config_file = None
    loop.response_display_prefix = ""
    loop._debug_enabled_for_agent = lambda agent_data: False
    return loop


def _make_conditional_retry_loop(
    tokenizer: _FakeTokenizer,
    server_manager: _SequencedFakeServerManager,
    tool_parser: _SequencedFakeToolParser,
    reward_fn,
):
    loop = object.__new__(ToolAgentLoop)
    loop.force_tool_calls_until_final_turn = False
    loop.force_answer_on_final_turn = False
    loop.force_answer_on_final_turn_validation = None
    loop.final_answer_token_margin = 4
    loop.force_final_answer_think_prefill = ""
    loop.force_final_answer_think_prefill_trainable = False
    loop.force_final_answer_prefix = ""
    loop.max_assistant_turns = 11
    loop.max_assistant_tokens_per_turn = None
    loop.max_user_turns = 10
    loop.response_length = 64
    loop.server_manager = server_manager
    loop.tokenizer = tokenizer
    loop.loop = _InlineExecutorLoop()
    loop.tools = {}
    loop.tool_schemas = []
    loop.tool_parser = tool_parser
    loop.interaction_config_file = None
    loop.response_display_prefix = ""
    loop.conditional_answer_retry_enabled = True
    loop.conditional_answer_retry_enable_validation = True
    loop.conditional_answer_retry_force_next_tool_call = True
    loop.conditional_answer_retry_suffix = "retry suffix"
    loop.conditional_answer_retry_assistant_prefill = ""
    loop.conditional_answer_retry_assistant_prefill_trainable = False
    loop.conditional_answer_retry_teacher_only_prefill = False
    loop.conditional_answer_retry_sft_enabled = False
    loop.conditional_answer_retry_sft_target_selection = "all"
    loop.conditional_answer_retry_correct_accuracy_threshold = 0.0
    loop.conditional_answer_retry_max_retries = None
    loop.conditional_answer_retry_reward_fn = reward_fn
    loop.conditional_answer_retry_branch_enabled = False
    loop.llm_judge_enabled = False
    loop.llm_judge_prompt_template = None
    loop.llm_judge_reward_types = {"search"}
    loop.llm_judge_max_tokens = 48
    loop.llm_judge_temperature = 0.0
    loop.llm_judge_top_p = 1.0
    loop.llm_judge_partial_credit_score = 0.5
    loop.llm_judge_timeout_s = None
    loop.llm_judge_fallback_to_string_reward = True
    loop.llm_judge_assistant_prefill = "{"
    loop.llm_judge_format_score = 0.1
    loop.llm_judge_tool_efficiency_score_weight = 0.0
    loop.tool_repeat_overlap_iou_threshold = 0.9
    loop.tool_repeat_overlap_small_crop_area_threshold = 0.0002
    loop.tool_repeat_overlap_free_depth_refinements = 2
    loop.tool_repeat_overlap_depth_containment_threshold = 0.95
    loop._debug_enabled_for_agent = lambda agent_data: False

    async def apply_chat_template(
        messages,
        tools=None,
        images=None,
        videos=None,
        remove_system_prompt=False,
    ):
        del tools, images, videos, remove_system_prompt
        return [100, 101] if "retry suffix" in str(messages) else [100]

    loop.apply_chat_template = apply_chat_template
    return loop


def test_tool_agent_loop_init_tolerates_missing_optional_multi_turn_keys(monkeypatch):
    multi_turn = OmegaConf.create(
        {
            "max_user_turns": 10,
            "max_assistant_turns": 11,
            "max_parallel_calls": 1,
            "max_tool_response_length": 256,
            "tool_response_truncate_side": "middle",
            "tool_config_path": None,
            "format": "hermes",
            "interaction_config_path": None,
        }
    )
    OmegaConf.set_struct(multi_turn, True)
    loop = object.__new__(ToolAgentLoop)
    loop.rollout_config = SimpleNamespace(
        multi_turn=multi_turn,
        get=lambda key, default=None: default,
        prompt_length=16,
        response_length=64,
    )
    loop.config = {}
    loop.tokenizer = _FakeTokenizer()
    monkeypatch.setattr(
        "verl.experimental.agent_loop.tool_agent_loop.initialize_tools_from_config",
        lambda path: [],
    )
    monkeypatch.setattr(
        "verl.experimental.agent_loop.tool_agent_loop.ToolParser.get_tool_parser",
        lambda fmt, tokenizer: _FakeToolParser([]),
    )
    monkeypatch.setattr(
        "verl.experimental.agent_loop.tool_agent_loop.AgentLoopBase.__init__",
        lambda self, *args, **kwargs: None,
    )

    ToolAgentLoop.__init__(loop)

    assert loop.force_final_answer_prefix == ""
    assert loop.force_final_answer_think_prefill == ""
    assert loop.force_final_answer_think_prefill_trainable is False
    assert loop.force_tool_calls_until_final_turn is False
    assert loop.force_answer_on_final_turn is False
    assert loop.force_answer_on_final_turn_validation is None
    assert loop.max_assistant_tokens_per_turn is None
    assert loop.conditional_answer_retry_enabled is False
    assert loop.conditional_answer_retry_enable_validation is True
    assert loop.conditional_answer_retry_force_next_tool_call is True
    assert loop.conditional_answer_retry_assistant_prefill == ""
    assert loop.conditional_answer_retry_assistant_prefill_trainable is False
    assert loop.llm_judge_enabled is False
    assert loop.llm_judge_reward_types == {"search"}
    assert loop.llm_judge_assistant_prefill == "{"


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
    agent_data.messages = [{"role": "user", "content": "question"}]
    agent_data.last_user_tool_prompt_start = 0
    agent_data.last_user_tool_prompt_end = 1
    agent_data.last_user_tool_response_mask_start = 0
    agent_data.last_user_tool_response_mask_end = 0
    agent_data.last_user_tool_response_logprobs_start = 0
    agent_data.last_user_tool_response_logprobs_end = 0
    agent_data.last_user_tool_message_index = 0
    agent_data.last_user_tool_block_start_index = 0
    return agent_data


def test_assistant_turn_token_cap_uses_configured_existing_and_remaining_minimum():
    loop = object.__new__(ToolAgentLoop)
    loop.max_assistant_tokens_per_turn = 4096
    loop.response_length = 10000
    agent_data = _make_agent_data()
    agent_data.response_mask = [1] * 3500

    params = loop._with_assistant_turn_token_cap({"temperature": 1.0, "max_tokens": 5000}, agent_data)

    assert params["max_tokens"] == 4096

    agent_data.response_mask = [1] * 7500
    params = loop._with_assistant_turn_token_cap({"temperature": 1.0, "max_tokens": 5000}, agent_data)
    assert params["max_tokens"] == 2500


def test_assistant_turn_token_cap_default_does_not_change_sampling_params():
    loop = object.__new__(ToolAgentLoop)
    loop.max_assistant_tokens_per_turn = None
    loop.response_length = 10000
    agent_data = _make_agent_data()
    sampling_params = {"temperature": 1.0}

    assert loop._with_assistant_turn_token_cap(sampling_params, agent_data) is sampling_params


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
    assert extra_args[FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY] == [
        [1],
        [80, 81, 82],
        [83, 84, 81, 82],
        [83, 85, 82],
    ]


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


def test_force_answer_on_final_turn_does_not_force_early_turns():
    tokenizer = _FakeTokenizer()
    server_manager = _FakeServerManager([30, 31, 1, 2, 20, 21, 22, 50])
    tool_parser = _FakeToolParser([])
    agent_data = _make_agent_data()

    async def run_test():
        loop = _make_loop(tokenizer, server_manager, tool_parser)
        loop.force_tool_calls_until_final_turn = False
        loop.force_answer_on_final_turn = True
        return await loop._handle_generating_state(agent_data, {"temperature": 1.0})

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    assert FORCE_ACTION_REQUIRED_TAG_KEY not in server_manager.last_sampling_params.get("extra_args", {})
    assert "forced_structure_error" not in agent_data.extra_fields


def test_force_answer_on_final_turn_forces_only_final_turn_answer():
    tokenizer = _FakeTokenizer()
    server_manager = _FakeServerManager([30, 31, 1, 2, 20, 21, 22, 50])
    tool_parser = _FakeToolParser([])
    agent_data = _make_agent_data()
    agent_data.assistant_turns = 10

    async def run_test():
        loop = _make_loop(tokenizer, server_manager, tool_parser)
        loop.force_tool_calls_until_final_turn = False
        loop.force_answer_on_final_turn = True
        return await loop._handle_generating_state(agent_data, {"temperature": 1.0})

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    assert server_manager.last_sampling_params["extra_args"][FORCE_ACTION_REQUIRED_TAG_KEY] == "<answer>"
    assert server_manager.last_sampling_params["extra_args"][FORCE_ACTION_TAG_IDS_KEY] == [2, 20, 21, 22]
    assert "forced_structure_error" not in agent_data.extra_fields


def test_validation_final_answer_override_can_enable_forcing_independently():
    loop = object.__new__(ToolAgentLoop)
    loop.force_answer_on_final_turn = False
    loop.force_answer_on_final_turn_validation = True

    assert loop._force_answer_on_final_turn_active(None) is False
    assert loop._force_answer_on_final_turn_active({"_debug_trajectory": {"validate": True}}) is True


def test_validation_final_answer_override_can_disable_forcing_independently():
    loop = object.__new__(ToolAgentLoop)
    loop.force_answer_on_final_turn = True
    loop.force_answer_on_final_turn_validation = False

    assert loop._force_answer_on_final_turn_active(None) is True
    assert loop._force_answer_on_final_turn_active({"_debug_trajectory": {"validate": True}}) is False


def test_validation_final_answer_override_null_inherits_shared_setting():
    loop = object.__new__(ToolAgentLoop)
    loop.force_answer_on_final_turn_validation = None

    loop.force_answer_on_final_turn = False
    assert loop._force_answer_on_final_turn_active({"_debug_trajectory": {"validate": True}}) is False

    loop.force_answer_on_final_turn = True
    assert loop._force_answer_on_final_turn_active({"_debug_trajectory": {"validate": True}}) is True


def test_validation_final_answer_override_does_not_force_early_turns():
    loop = object.__new__(ToolAgentLoop)
    loop.force_answer_on_final_turn = False
    loop.force_answer_on_final_turn_validation = True
    loop.max_assistant_turns = 11
    loop.max_user_turns = 10
    loop.final_answer_token_margin = 4
    loop.response_length = 64
    agent_data = _make_agent_data()

    assert loop._force_answer_on_final_turn_active({"_debug_trajectory": {"validate": True}}) is True
    assert loop._must_answer_now(agent_data) is False


def test_forced_final_turn_forces_boxed_answer_prefix():
    tokenizer = _FakeTokenizer()
    server_manager = _FakeServerManager([30, 31, 1, 2, 20, 21, 22, 2, 60, 61, 50])
    tool_parser = _FakeToolParser([])
    agent_data = _make_agent_data()
    agent_data.assistant_turns = 10

    async def run_test():
        loop = _make_loop(tokenizer, server_manager, tool_parser)
        loop.force_final_answer_prefix = "\n\\boxed{"
        return await loop._handle_generating_state(agent_data, {"temperature": 1.0})

    state = asyncio.run(run_test())

    extra_args = server_manager.last_sampling_params["extra_args"]
    assert extra_args[FORCE_ACTION_REQUIRED_TAG_KEY] == "<answer>"
    # `<answer>` tag (2, 20, 21, 22) followed by the forced boxed prefix (2, 60, 61).
    assert extra_args[FORCE_ACTION_TAG_IDS_KEY] == [2, 20, 21, 22, 2, 60, 61]
    assert state == AgentState.TERMINATED
    assert "forced_structure_error" not in agent_data.extra_fields


def test_forced_final_turn_think_prefill_is_masked_and_in_prompt():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager([[31, 1, 2, 20, 21, 22, 50]])
    tool_parser = _FakeToolParser([])
    agent_data = _make_agent_data()
    agent_data.assistant_turns = 10
    prefill = "This is the final turn, so I need to answer from the evidence gathered so far."

    async def run_test():
        loop = _make_loop(tokenizer, server_manager, tool_parser)
        loop.force_final_answer_think_prefill = prefill
        return await loop._handle_generating_state(agent_data, {"temperature": 1.0})

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    assert server_manager.prompt_ids_by_call[0][-1:] == [72]
    assert agent_data.response_mask == [0] + [1] * 7
    assert agent_data.response_logprobs == [0.0] + [0.0] * 7
    assert agent_data.prompt_ids[-8:] == [72, 31, 1, 2, 20, 21, 22, 50]
    extra_args = server_manager.sampling_params_by_call[0]["extra_args"]
    assert extra_args[FORCE_ACTION_REQUIRED_TAG_KEY] == "<answer>"
    assert extra_args[FORCE_ACTION_TAG_IDS_KEY] == [2, 20, 21, 22]
    assert "forced_structure_error" not in agent_data.extra_fields


def test_forced_final_turn_think_prefill_is_trainable_when_enabled():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager([[31, 1, 2, 20, 21, 22, 50]])
    tool_parser = _FakeToolParser([])
    agent_data = _make_agent_data()
    agent_data.assistant_turns = 10

    async def run_test():
        loop = _make_loop(tokenizer, server_manager, tool_parser)
        loop.force_final_answer_think_prefill = (
            "This is the final turn, so I need to answer from the evidence gathered so far."
        )
        loop.force_final_answer_think_prefill_trainable = True
        return await loop._handle_generating_state(agent_data, {"temperature": 1.0})

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    assert server_manager.prompt_ids_by_call[0][-1:] == [72]
    assert agent_data.response_mask == [1] * 8
    assert agent_data.response_logprobs == [0.0] * 8
    assert len(agent_data.response_mask) == len(agent_data.response_logprobs)
    assert agent_data.prompt_ids[-8:] == [72, 31, 1, 2, 20, 21, 22, 50]
    assert "forced_structure_error" not in agent_data.extra_fields


def test_forced_final_turn_think_prefill_is_in_debug_text():
    tokenizer = _FakeTokenizer()
    server_manager = _FakeServerManager([31, 1, 2, 20, 21, 22, 50])
    tool_parser = _FakeToolParser([])
    agent_data = _make_agent_data()
    agent_data.assistant_turns = 10
    prefill = "This is the final turn, so I need to answer from the evidence gathered so far."

    async def run_test():
        loop = _make_loop(tokenizer, server_manager, tool_parser)
        loop.force_final_answer_think_prefill = prefill
        loop._debug_enabled_for_agent = lambda agent_data: True
        loop._debug_append_event = lambda event: None
        return await loop._handle_generating_state(agent_data, {"temperature": 1.0})

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    assert agent_data.debug_events[0]["text"].startswith(f"{prefill}reason")


def test_final_answer_think_prefill_is_not_applied_to_forced_tool_turn():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager([[30, 31, 1, 2, 10, 40, 11]])
    tool_parser = _FakeToolParser([FunctionCall(name="image_zoom_in_tool", arguments="{}")])
    agent_data = _make_agent_data()

    async def run_test():
        loop = _make_loop(tokenizer, server_manager, tool_parser)
        loop.force_final_answer_think_prefill = (
            "This is the final turn, so I need to answer from the evidence gathered so far."
        )
        return await loop._handle_generating_state(agent_data, {"temperature": 1.0})

    state = asyncio.run(run_test())

    assert state == AgentState.PROCESSING_TOOLS
    assert server_manager.prompt_ids_by_call[0] == [100]
    assert agent_data.response_mask == [1] * 7


def test_final_answer_think_prefill_is_not_applied_to_retry_forced_tool_call():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager([[30, 31, 1, 2, 10, 40, 11]])
    tool_parser = _SequencedFakeToolParser([[FunctionCall(name="image_zoom_in_tool", arguments="{}")]])
    agent_data = _make_agent_data()
    agent_data.force_next_tool_call = True

    async def run_test():
        loop = _make_conditional_retry_loop(
            tokenizer,
            server_manager,
            tool_parser,
            lambda **kwargs: {"accuracy": 0.0, "score": 0.0},
        )
        loop.force_final_answer_think_prefill = (
            "This is the final turn, so I need to answer from the evidence gathered so far."
        )
        return await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )

    state = asyncio.run(run_test())

    assert state == AgentState.PROCESSING_TOOLS
    assert server_manager.prompt_ids_by_call[0] == [100]
    assert agent_data.response_mask == [1] * 7
    extra_args = server_manager.sampling_params_by_call[0]["extra_args"]
    assert extra_args[FORCE_ACTION_REQUIRED_TAG_KEY] == "<tool_call>"


def test_conditional_answer_retry_keeps_correct_first_answer_without_forcing():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager([[30, 31, 1, 2, 20, 21, 22, 50]])
    tool_parser = _SequencedFakeToolParser([[]])
    agent_data = _make_agent_data()

    async def run_test():
        loop = _make_conditional_retry_loop(
            tokenizer,
            server_manager,
            tool_parser,
            lambda **kwargs: {"accuracy": 1.0, "score": 1.0},
        )
        return await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    assert FORCE_ACTION_REQUIRED_TAG_KEY not in server_manager.sampling_params_by_call[0].get("extra_args", {})
    assert agent_data.extra_fields["conditional_answer_retry_last_grade"]["accuracy"] == 1.0
    assert agent_data.extra_fields.get("conditional_answer_retry_count") is None


def test_conditional_answer_retry_is_disabled_for_validation_when_configured():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager([[30, 31, 1, 2, 20, 21, 22, 50]])
    tool_parser = _SequencedFakeToolParser([[]])
    agent_data = _make_agent_data()

    def unexpected_reward_fn(**kwargs):
        raise AssertionError("validation rollout must not invoke the GT retry validator")

    async def run_test():
        loop = _make_conditional_retry_loop(
            tokenizer,
            server_manager,
            tool_parser,
            unexpected_reward_fn,
        )
        loop.conditional_answer_retry_enable_validation = False
        return await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={
                "data_source": "unit",
                "reward_model": {"ground_truth": "ok"},
                "extra_info": {},
                "_debug_trajectory": {"validate": True},
            },
        )

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    assert agent_data.extra_fields.get("conditional_answer_retry_last_grade") is None
    assert agent_data.extra_fields.get("conditional_answer_retry_count") is None


def test_validation_without_retry_still_forces_answer_on_final_turn():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager([[31, 1, 2, 20, 21, 22, 50]])
    tool_parser = _SequencedFakeToolParser([[]])
    agent_data = _make_agent_data()
    agent_data.assistant_turns = 10

    async def run_test():
        loop = _make_conditional_retry_loop(
            tokenizer,
            server_manager,
            tool_parser,
            lambda **kwargs: {"accuracy": 0.0},
        )
        loop.force_answer_on_final_turn = False
        loop.force_answer_on_final_turn_validation = True
        loop.conditional_answer_retry_enable_validation = False
        return await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"_debug_trajectory": {"validate": True}},
        )

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    extra_args = server_manager.sampling_params_by_call[0]["extra_args"]
    assert extra_args[FORCE_ACTION_REQUIRED_TAG_KEY] == "<answer>"


def test_conditional_answer_retry_grades_with_response_display_prefix():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager([[31, 1, 20, 21, 22, 50]])
    tool_parser = _SequencedFakeToolParser([[]])
    agent_data = _make_agent_data()
    graded_solution = {}

    def reward_fn(**kwargs):
        graded_solution["solution_str"] = kwargs["solution_str"]
        return {"accuracy": 1.0 if kwargs["solution_str"].startswith("<think>reason</think><answer>") else 0.0}

    async def run_test():
        loop = _make_conditional_retry_loop(tokenizer, server_manager, tool_parser, reward_fn)
        loop.response_display_prefix = "<think>"
        return await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    assert graded_solution["solution_str"] == "<think>reason</think><answer>final"
    assert agent_data.extra_fields["conditional_answer_retry_last_grade"]["accuracy"] == 1.0
    assert agent_data.extra_fields.get("conditional_answer_retry_count") is None


def test_llm_judge_response_parser_prepends_open_brace():
    loop = object.__new__(ToolAgentLoop)
    loop.llm_judge_assistant_prefill = "{"

    parsed = loop._parse_llm_judge_response('"is_equivalent": true, "partial_credit": false}')

    assert parsed == {"is_equivalent": True, "partial_credit": False}


def test_conditional_answer_retry_uses_llm_judge_for_search_answer():
    tokenizer = _FakeTokenizer()
    tokenizer.id_to_text[80] = '"is_equivalent": true, "partial_credit": false}'
    server_manager = _SequencedFakeServerManager(
        [
            [30, 31, 1, 20, 21, 22, 50, 23],
            [80],
        ]
    )
    tool_parser = _SequencedFakeToolParser([[]])
    agent_data = _make_agent_data()
    calls = {"reward_fn": 0}

    def reward_fn(**kwargs):
        calls["reward_fn"] += 1
        assert kwargs["solution_str"] == "<think>reason</think><answer>final</answer>"
        assert kwargs["extra_info"]["assistant_turns"] == 1
        return {"score": 0.1, "accuracy": 0.0, "format": 1.0}

    async def run_test():
        loop = _make_conditional_retry_loop(tokenizer, server_manager, tool_parser, reward_fn)
        loop.llm_judge_enabled = True
        loop.llm_judge_prompt_template = "Question: [QUESTION]\nReference answer: [REFERENCE]\nModel answer: [PREDICTION]"
        loop._encode_llm_judge_prompt = lambda prompt: [999]
        return await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={
                "data_source": "unit",
                "reward_model": {"ground_truth": "a ship"},
                "extra_info": {"reward_type": "search", "question": "What object is visible?"},
            },
        )

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    assert calls["reward_fn"] == 1
    assert server_manager.prompt_ids_by_call[1] == [999]
    grade = agent_data.extra_fields["conditional_answer_retry_last_grade"]
    assert grade["llm_judge_used"] is True
    assert grade["accuracy"] == 1.0
    assert grade["score"] == 1.0
    assert agent_data.response_ids == [30, 31, 1, 20, 21, 22, 50, 23]


def test_final_llm_judge_score_uses_final_trajectory_without_mutating_tokens():
    tokenizer = _FakeTokenizer()
    tokenizer.id_to_text[80] = '"is_equivalent": false, "partial_credit": true}'
    server_manager = _SequencedFakeServerManager([[80]])
    tool_parser = _SequencedFakeToolParser([])
    agent_data = _make_agent_data()
    agent_data.prompt_ids = [100, 30, 31, 1, 20, 21, 22, 50, 23]
    agent_data.response_mask = [1] * 8

    def reward_fn(**kwargs):
        assert kwargs["solution_str"] == "<think>reason</think><answer>final</answer>"
        assert kwargs["extra_info"]["assistant_turns"] == agent_data.assistant_turns
        return {"score": 0.1, "accuracy": 0.0, "format": 1.0}

    async def run_test():
        loop = _make_conditional_retry_loop(tokenizer, server_manager, tool_parser, reward_fn)
        loop.llm_judge_enabled = True
        loop.llm_judge_prompt_template = "Question: [QUESTION]\nReference answer: [REFERENCE]\nModel answer: [PREDICTION]"
        loop._encode_llm_judge_prompt = lambda prompt: [999]
        return await loop._maybe_score_final_with_llm_judge(
            agent_data,
            {
                "data_source": "unit",
                "reward_model": {"ground_truth": "a ship"},
                "extra_info": {"reward_type": "search", "question": "What object is visible?"},
            },
        )

    grade = asyncio.run(run_test())

    assert grade["llm_judge_used"] is True
    assert grade["accuracy"] == 0.5
    assert grade["score"] == pytest.approx(0.55)
    assert agent_data.prompt_ids == [100, 30, 31, 1, 20, 21, 22, 50, 23]
    assert agent_data.response_mask == [1] * 8


def test_conditional_answer_retry_rolls_back_appends_suffix_and_forces_next_tool_call():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager(
        [
            [30, 31, 1, 2, 20, 21, 22, 50],
            [30, 31, 1, 2, 10, 40, 11],
        ]
    )
    tool_parser = _SequencedFakeToolParser(
        [
            [],
            [FunctionCall(name="image_zoom_in_tool", arguments="{}")],
        ]
    )
    agent_data = _make_agent_data()

    async def run_test():
        loop = _make_conditional_retry_loop(
            tokenizer,
            server_manager,
            tool_parser,
            lambda **kwargs: {"accuracy": 0.0, "score": 0.0},
        )
        first_state = await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )
        second_state = await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )
        return first_state, second_state

    first_state, second_state = asyncio.run(run_test())

    assert first_state == AgentState.GENERATING
    assert second_state == AgentState.PROCESSING_TOOLS
    assert agent_data.messages == [{"role": "user", "content": "question\n\nretry suffix"}]
    assert agent_data.extra_fields["conditional_answer_retry_count"] == 1
    assert agent_data.assistant_turns == 1
    assert agent_data.prompt_ids[:2] == [100, 101]
    assert len(agent_data.response_mask) == len(server_manager.sampling_params_by_call[1]["extra_args"][FORCE_ACTION_TAG_IDS_KEY]) + 5
    extra_args = server_manager.sampling_params_by_call[1]["extra_args"]
    assert extra_args[FORCE_ACTION_REQUIRED_TAG_KEY] == "<tool_call>"
    assert extra_args[FORCE_ACTION_TAG_IDS_KEY] == [2, 10]


def test_conditional_answer_retry_assistant_prefill_is_masked_and_in_prompt():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager(
        [
            [30, 31, 1, 2, 20, 21, 22, 50],
            [30, 31, 1, 2, 10, 40, 11],
        ]
    )
    tool_parser = _SequencedFakeToolParser(
        [
            [],
            [FunctionCall(name="image_zoom_in_tool", arguments="{}")],
        ]
    )
    agent_data = _make_agent_data()

    async def run_test():
        loop = _make_conditional_retry_loop(
            tokenizer,
            server_manager,
            tool_parser,
            lambda **kwargs: {"accuracy": 0.0, "score": 0.0},
        )
        loop.conditional_answer_retry_assistant_prefill = "retry prefill"
        first_state = await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )
        second_state = await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )
        return first_state, second_state

    first_state, second_state = asyncio.run(run_test())

    assert first_state == AgentState.GENERATING
    assert second_state == AgentState.PROCESSING_TOOLS
    assert server_manager.prompt_ids_by_call[1][-2:] == [70, 71]
    assert agent_data.response_mask == [0, 0] + [1] * 7
    assert agent_data.response_logprobs == [0.0, 0.0] + [0.0] * 7
    assert agent_data.prompt_ids[-9:] == [70, 71, 30, 31, 1, 2, 10, 40, 11]


def test_teacher_only_retry_prefill_is_not_stored_or_seen_by_later_student():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager(
        [
            [30, 31, 1, 2, 20, 21, 22, 50],
            [31, 1, 2, 10, 41, 11],
            [30, 31, 1, 2, 20, 21, 22, 23],
        ]
    )
    tool_parser = _SequencedFakeToolParser(
        [
            [],
            [
                FunctionCall(
                    name="image_zoom_in_tool",
                    arguments='{"label":"shelf","bbox_2d":[100,100,500,500],"img_idx":0}',
                )
            ],
            [],
        ]
    )
    agent_data = _make_agent_data()

    async def run_test():
        loop = _make_conditional_retry_loop(
            tokenizer,
            server_manager,
            tool_parser,
            lambda **kwargs: {"accuracy": 0.0, "score": 0.0},
        )
        loop.response_display_prefix = "<think>\n"
        loop.conditional_answer_retry_assistant_prefill = "retry prefill"
        loop.conditional_answer_retry_teacher_only_prefill = True
        loop.conditional_answer_retry_sft_enabled = True
        first_state = await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )
        teacher_state = await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )
        # Simulate the tool-response prompt tokens. Later student generation must
        # use the stored suffix, never the teacher-only prefill.
        agent_data.prompt_ids += [90]
        agent_data.response_mask += [0]
        agent_data.retry_sft_mask += [0]
        loop.conditional_answer_retry_enabled = False
        student_state = await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )
        return first_state, teacher_state, student_state

    first_state, teacher_state, student_state = asyncio.run(run_test())

    assert first_state == AgentState.GENERATING
    assert teacher_state == AgentState.PROCESSING_TOOLS
    assert student_state == AgentState.TERMINATED
    assert server_manager.prompt_ids_by_call[1][-2:] == [70, 71]
    assert 70 not in server_manager.prompt_ids_by_call[2]
    assert 71 not in server_manager.prompt_ids_by_call[2]
    assert 70 not in agent_data.prompt_ids
    assert 71 not in agent_data.prompt_ids
    teacher_len = 6
    assert agent_data.response_mask[:teacher_len] == [0] * teacher_len
    assert agent_data.response_mask[-8:] == [1] * 8
    candidate = agent_data.retry_sft_candidates[0]
    assert candidate["static_rejection_reasons"] == []
    assert candidate["response_start"] == 0
    assert candidate["response_end"] == teacher_len


def test_conditional_answer_retry_assistant_prefill_is_trainable_when_enabled():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager(
        [
            [30, 31, 1, 2, 20, 21, 22, 50],
            [30, 31, 1, 2, 10, 40, 11],
        ]
    )
    tool_parser = _SequencedFakeToolParser(
        [
            [],
            [FunctionCall(name="image_zoom_in_tool", arguments="{}")],
        ]
    )
    agent_data = _make_agent_data()

    async def run_test():
        loop = _make_conditional_retry_loop(
            tokenizer,
            server_manager,
            tool_parser,
            lambda **kwargs: {"accuracy": 0.0, "score": 0.0},
        )
        loop.conditional_answer_retry_assistant_prefill = "retry prefill"
        loop.conditional_answer_retry_assistant_prefill_trainable = True
        first_state = await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )
        second_state = await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )
        return first_state, second_state

    first_state, second_state = asyncio.run(run_test())

    assert first_state == AgentState.GENERATING
    assert second_state == AgentState.PROCESSING_TOOLS
    assert server_manager.prompt_ids_by_call[1][-2:] == [70, 71]
    assert agent_data.response_mask == [1] * 9
    assert agent_data.response_logprobs == [0.0] * 9
    assert len(agent_data.response_mask) == len(agent_data.response_logprobs)
    assert agent_data.prompt_ids[-9:] == [70, 71, 30, 31, 1, 2, 10, 40, 11]


def test_retry_and_final_answer_prefills_are_both_trainable_in_one_trajectory():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager(
        [
            [30, 31, 1, 2, 20, 21, 22, 50],
            [30, 31, 1, 2, 10, 40, 11],
            [31, 1, 2, 20, 21, 22, 50],
        ]
    )
    tool_parser = _SequencedFakeToolParser(
        [
            [],
            [FunctionCall(name="image_zoom_in_tool", arguments="{}")],
            [],
        ]
    )
    agent_data = _make_agent_data()

    async def run_test():
        loop = _make_conditional_retry_loop(
            tokenizer,
            server_manager,
            tool_parser,
            lambda **kwargs: {"accuracy": 0.0, "score": 0.0},
        )
        loop.conditional_answer_retry_assistant_prefill = "retry prefill"
        loop.conditional_answer_retry_assistant_prefill_trainable = True
        loop.force_final_answer_think_prefill = (
            "This is the final turn, so I need to answer from the evidence gathered so far."
        )
        loop.force_final_answer_think_prefill_trainable = True
        loop.force_answer_on_final_turn = True

        first_state = await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )
        second_state = await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )
        agent_data.assistant_turns = 10
        final_state = await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )
        return first_state, second_state, final_state

    first_state, second_state, final_state = asyncio.run(run_test())

    assert first_state == AgentState.GENERATING
    assert second_state == AgentState.PROCESSING_TOOLS
    assert final_state == AgentState.TERMINATED
    assert server_manager.prompt_ids_by_call[1][-2:] == [70, 71]
    assert server_manager.prompt_ids_by_call[2][-1:] == [72]
    assert agent_data.response_mask == [1] * 17
    assert agent_data.response_logprobs == [0.0] * 17
    assert len(agent_data.response_mask) == len(agent_data.response_logprobs)


def test_conditional_answer_retry_skips_when_caps_prevent_more_exploration():
    tokenizer = _FakeTokenizer()
    server_manager = _SequencedFakeServerManager([[30, 31, 1, 2, 20, 21, 22, 50]])
    tool_parser = _SequencedFakeToolParser([[]])
    agent_data = _make_agent_data()

    async def run_test():
        loop = _make_conditional_retry_loop(
            tokenizer,
            server_manager,
            tool_parser,
            lambda **kwargs: {"accuracy": 0.0, "score": 0.0},
        )
        loop.max_assistant_turns = 1
        return await loop._handle_generating_state(
            agent_data,
            {"temperature": 1.0},
            rollout_kwargs={"data_source": "unit", "reward_model": {"ground_truth": "ok"}, "extra_info": {}},
        )

    state = asyncio.run(run_test())

    assert state == AgentState.TERMINATED
    assert agent_data.messages == [{"role": "user", "content": "question"}]
    assert agent_data.assistant_turns == 1
    assert agent_data.extra_fields["conditional_answer_retry_skipped_reason"] == "max_assistant_turns"
    assert agent_data.extra_fields.get("conditional_answer_retry_count") is None


def test_vllm_server_helper_appends_forced_action_processor_when_force_mode_enabled():
    engine_kwargs = {"logits_processors": ["existing.module:Processor"]}

    ensure_forced_action_logits_processor(engine_kwargs)

    assert engine_kwargs["logits_processors"] == ["existing.module:Processor", FORCE_ACTION_LOGITS_PROCESSOR_FQCN]


def test_vllm_server_helper_does_not_duplicate_forced_action_processor():
    engine_kwargs = {"logits_processors": [FORCE_ACTION_LOGITS_PROCESSOR_FQCN]}

    ensure_forced_action_logits_processor(engine_kwargs)

    assert engine_kwargs["logits_processors"] == [FORCE_ACTION_LOGITS_PROCESSOR_FQCN]
