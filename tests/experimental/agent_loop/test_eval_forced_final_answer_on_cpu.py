import asyncio
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "vero_eval"))

from lmms_eval.models.chat import async_openai as async_openai_module
from lmms_eval.models.chat.async_openai import AsyncOpenAIChat
from lmms_eval.tasks.visual_probe import utils as visual_probe_utils


class _Tokenizer:
    _ids = {
        "\n<answer>": [2, 20, 21, 22],
        "\n<tool_call>": [2, 10],
        "</think>": [1],
        "</": [80],
        "think": [81],
        ">": [82],
        "<": [83],
        "/": [84],
        "/think": [85],
    }

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return list(self._ids[text])


class _Completions:
    def __init__(self, responses):
        self.responses = list(responses)
        self.requests = []

    async def create(self, **kwargs):
        self.requests.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


def _response(content, *, tool_calls=None, finish_reason="stop", usage=None):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=content, tool_calls=tool_calls),
            )
        ],
        usage=usage,
    )


def _critic_response(token, *, accept_logprob, retry_logprob, usage=None):
    selected_logprob = accept_logprob if token == "A" else retry_logprob
    top_logprobs = [
        SimpleNamespace(token="A", logprob=accept_logprob),
        SimpleNamespace(token="R", logprob=retry_logprob),
    ]
    response = _response(token, usage=usage)
    response.choices[0].logprobs = SimpleNamespace(
        content=[SimpleNamespace(token=token, logprob=selected_logprob, top_logprobs=top_logprobs)]
    )
    return response


def _model(*responses):
    model = object.__new__(AsyncOpenAIChat)
    model.client = SimpleNamespace(chat=SimpleNamespace(completions=_Completions(responses)))
    model.force_final_answer_think_prefill = (
        "This is the final turn, so I need to answer from the evidence gathered so far."
    )
    model.force_final_answer_prefix = ""
    model._forced_action_tokenizer = _Tokenizer()
    model._forced_final_stats = {
        "primary_attempts": 0,
        "primary_successes": 0,
        "fallback_attempts": 0,
        "fallback_successes": 0,
        "primary_input_tokens": 0,
        "primary_output_tokens": 0,
        "fallback_input_tokens": 0,
        "fallback_output_tokens": 0,
    }
    model._forced_final_failures = {}
    model.critic_guided_answer_retry = False
    model.critic_guided_answer_retry_threshold = 0.5
    model.critic_guided_answer_retry_prefill_profile = "persist"
    model._retry_critic_accept_token_id = 32
    model._retry_critic_retry_token_id = 49
    model._retry_critic_stats = {
        "decisions": 0,
        "accepts": 0,
        "retries": 0,
        "forced_accepts": 0,
        "failures": 0,
        "retry_probability_sum": 0.0,
        "retry_probability_count": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
    }
    return model


def _payload():
    return {
        "model": "qwen_vllm",
        "messages": [{"role": "user", "content": "question"}],
        "tools": [{"type": "function", "function": {"name": "zoom"}}],
        "tool_choice": "auto",
    }


def test_primary_forced_answer_preserves_tools_and_disables_tool_choice():
    model = _model(_response("reasoning\n</think>\n<answer>20</answer>"))

    response, prefill, attempts = asyncio.run(model._obtain_forced_final_answer(_payload(), 4))

    assert response.choices[0].message.content.endswith("<answer>20</answer>")
    assert len(attempts) == 1
    request = model.client.chat.completions.requests[0]
    assert request["tools"] == _payload()["tools"]
    assert request["tool_choice"] == "none"
    assert request["messages"][-1]["content"] == (
        "<think>\nThis is the final turn, so I need to answer from the evidence gathered so far."
    )
    assert request["extra_body"]["continue_final_message"] is True
    assert "extra_args" not in request["extra_body"]
    assert request["extra_body"]["vllm_xargs"]["verl_force_action_tag_after_think_required_tag"] == "<answer>"
    assert request["extra_body"]["vllm_xargs"]["verl_force_action_tag_after_think_tag_ids"] == [2, 20, 21, 22]
    assert model._forced_final_stats["primary_successes"] == 1


@pytest.mark.parametrize(
    "primary",
    [
        _response("reasoning\n</think>\n<tool_call>{}</tool_call>"),
        _response("reasoning\n</think>"),
        _response("reasoning\n</think>\n<answer></answer>"),
        _response("reasoning\n</think>\n<answer>20"),
        _response("", tool_calls=[SimpleNamespace(function=SimpleNamespace(name="zoom", arguments="{}"))]),
    ],
)
def test_malformed_primary_uses_clean_hard_fallback(primary):
    fallback = _response("\n<answer>20</answer>")
    model = _model(primary, fallback)

    response, prefill, attempts = asyncio.run(model._obtain_forced_final_answer(_payload(), 7))

    assert response is fallback
    assert len(attempts) == 2
    assert prefill.endswith("\n</think>")
    fallback_request = model.client.chat.completions.requests[1]
    assert fallback_request["messages"][:-1] == _payload()["messages"]
    assert fallback_request["messages"][-1]["content"].endswith("\n</think>")
    assert fallback_request["tools"] == _payload()["tools"]
    assert fallback_request["tool_choice"] == "none"
    assert fallback_request["extra_body"]["vllm_xargs"]["verl_force_action_tag_after_think_required_tag"] == "<answer>"
    assert model._forced_final_stats["fallback_successes"] == 1


def test_hard_fallback_failure_returns_per_sample_failure_marker():
    model = _model(
        _response("reasoning\n</think>\n<tool_call>{}</tool_call>"),
        _response("\n<answer></answer>"),
    )

    response, prefill, attempts = asyncio.run(model._obtain_forced_final_answer(_payload(), 9))

    assert response.choices[0].message.content == (
        "[LMMS_EVAL_FORCED_FINAL_ANSWER_FAILED: empty_answer]"
    )
    assert prefill == ""
    assert len(attempts) == 2
    assert model._forced_final_failures["fallback_empty_answer"] == 1


def test_forced_answer_validator_requires_exactly_one_nonempty_closed_block():
    assert AsyncOpenAIChat._forced_final_answer_failure(_response("<answer>white</answer>")) is None
    assert AsyncOpenAIChat._forced_final_answer_failure(_response("<answer> </answer>")) == "empty_answer"
    assert (
        AsyncOpenAIChat._forced_final_answer_failure(_response("<answer>a</answer><answer>b</answer>"))
        == "multiple_answer_tags"
    )


def test_response_token_counts_include_reasoning_details():
    response = _response(
        "<answer>white</answer>",
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=7,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=5),
        ),
    )

    assert AsyncOpenAIChat._response_token_counts(response) == (11, 7, 5)


def test_visual_probe_scores_generation_failure_marker_zero(monkeypatch):
    engine = SimpleNamespace(
        generate_json_batch=lambda prompts, use_tqdm: [
            '{"is_equivalent": true}' for _ in prompts
        ]
    )
    monkeypatch.setattr(visual_probe_utils, "get_judge_engine", lambda _: engine)
    results = [
        {
            "question": "question",
            "reference": "correct",
            "prediction": "correct",
            "raw_prediction": "[LMMS_EVAL_FORCED_FINAL_ANSWER_FAILED: missing_answer_close]",
            "resp_key": "failed",
        },
        {
            "question": "question",
            "reference": "correct",
            "prediction": "correct",
            "raw_prediction": "<answer>correct</answer>",
            "resp_key": "valid",
        },
    ]

    score = visual_probe_utils.visual_probe_aggregate_judge_results(results)

    assert score == 0.5
    assert visual_probe_utils.visual_probe_aggregate_judge_results.individual_scores["failed"]["score"] == 0.0
    assert visual_probe_utils.visual_probe_aggregate_judge_results.individual_scores["valid"]["score"] == 1.0


def test_openai_sdk_serializes_vllm_xargs_at_request_top_level():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "forced-answer-test",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen_vllm",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "stop",
                        "message": {"role": "assistant", "content": "\n<answer>ok</answer>"},
                    }
                ],
            },
        )

    client = OpenAI(
        api_key="test",
        base_url="http://test.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    model = _model()
    payload, _ = model._forced_final_answer_request(_payload(), hard_fallback=True)

    client.chat.completions.create(**payload)

    assert "extra_args" not in captured
    assert captured["vllm_xargs"]["verl_force_action_tag_after_think_required_tag"] == "<answer>"
    assert isinstance(
        captured["vllm_xargs"]["verl_force_action_tag_after_think_after_think_ids_options"],
        str,
    )


def test_openai_sdk_serializes_retry_allowed_tokens_and_continuation_flags():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured.update(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "retry-critic-test",
                "object": "chat.completion",
                "created": 0,
                "model": "qwen_vllm",
                "choices": [
                    {
                        "index": 0,
                        "finish_reason": "length",
                        "message": {"role": "assistant", "content": "A"},
                        "logprobs": {
                            "content": [
                                {
                                    "token": "A",
                                    "logprob": -0.1,
                                    "bytes": [65],
                                    "top_logprobs": [],
                                }
                            ]
                        },
                    }
                ],
            },
        )

    client = OpenAI(
        api_key="test",
        base_url="http://test.invalid/v1",
        http_client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    model = _enable_retry_critic(_model())
    payload = model._retry_critic_request(
        {"model": "qwen_vllm", "messages": [{"role": "user", "content": "q"}]},
        "<answer>x</answer>",
    )

    client.chat.completions.create(**payload)

    assert captured["allowed_token_ids"] == [32, 49]
    assert captured["continue_final_message"] is True
    assert captured["add_generation_prompt"] is False
    assert captured["top_k"] == -1
    assert captured["max_tokens"] == 1
    assert captured["top_logprobs"] == 2


def _tool_call(call_id, name="image_zoom_in_tool", arguments="{}"):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


class _FakeMCPClient:
    def __init__(self, *, fail_on_attempts=()):
        self.attempts = []
        self.fail_on_attempts = set(fail_on_attempts)

    async def get_function_list(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "image_zoom_in_tool",
                    "description": "zoom",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ]

    async def run_tool(self, name, arguments, env=None):
        self.attempts.append((name, arguments, env))
        if len(self.attempts) in self.fail_on_attempts:
            raise RuntimeError("synthetic tool failure")
        return SimpleNamespace(
            content=[
                {"type": "image_url", "image_url": {"url": "data:image/png;base64,AA=="}},
                {"type": "text", "text": "zoom complete"},
            ]
        )

    @staticmethod
    def convert_result_to_openai_format(content):
        return [content]


def _budget_model(tmp_path, *responses, max_tool_iterations=2, max_tool_images=0, fail_on_attempts=()):
    model = _model(*responses)
    model.model_version = "qwen_vllm"
    model.max_tool_iterations = max_tool_iterations
    model.max_tool_images = max_tool_images
    model.tool_image_as_user_turn = False
    model.force_tool_calls_until_final_turn = True
    model.force_answer_after_max_tool_iterations = False
    model.mcp_client = _FakeMCPClient(fail_on_attempts=fail_on_attempts)
    model._tool_functions = None
    model._task_tools_cache = {}
    model._task_system_prompt_cache = {}
    model.task_metadata = {"budget_task": {}}
    model.system_prompt = None
    model.task_dict = {"budget_task": {"test": [{}]}}
    model.max_pixels = 1000
    model.min_pixels = 1
    model.fps = None
    model.nframes = 1
    model.max_frames = None
    model.is_qwen3_vl = False
    model.work_dir = str(tmp_path)
    return model


def _budget_request():
    return SimpleNamespace(
        args=(
            None,
            lambda doc: [{"role": "user", "content": [{"type": "text", "text": "question"}]}],
            {},
            0,
            "budget_task",
            "test",
        )
    )


def test_no_tool_request_with_zero_tool_budget_preserves_plain_final_answer(tmp_path):
    model = _budget_model(tmp_path, _response("the blue sign"), max_tool_iterations=0, max_tool_images=0)
    model.mcp_client = None
    model.force_tool_calls_until_final_turn = False
    text, _, _, _ = asyncio.run(model.maybe_forward_with_tool(_budget_request(), 0))
    assert text == "the blue sign"
    requests = model.client.chat.completions.requests
    assert len(requests) == 1
    assert 'tools' not in requests[0] and 'tool_choice' not in requests[0]
    assert all(message['role'] != 'system' for message in requests[0]['messages'])


def _forced_tag(request):
    return request["extra_body"]["vllm_xargs"]["verl_force_action_tag_after_think_required_tag"]


def test_budget_forcing_runs_two_tool_turns_then_forces_answer(tmp_path):
    model = _budget_model(
        tmp_path,
        _response(
            "<think>first</think>",
            tool_calls=[_tool_call("call-1"), _tool_call("call-2")],
            finish_reason="tool_calls",
        ),
        _response(
            "<think>second</think>",
            tool_calls=[_tool_call("call-3")],
            finish_reason="tool_calls",
        ),
        _response("<think>done</think>\n<answer>correct</answer>"),
        max_tool_iterations=2,
        max_tool_images=0,
        fail_on_attempts={1},
    )

    text, idx, _, metadata = asyncio.run(model.maybe_forward_with_tool(_budget_request(), 5))

    assert idx == 5
    assert metadata is None
    assert text.endswith("<answer>correct</answer>")
    assert "[LMMS_EVAL_BUDGET_FORCING_FAILED:" not in text
    assert "[MAX_TOOL_ITERATIONS_REACHED]" not in text
    assert len(model.mcp_client.attempts) == 3
    requests = model.client.chat.completions.requests
    assert [_forced_tag(request) for request in requests] == ["<tool_call>", "<tool_call>", "<answer>"]
    assert all("tools" in request for request in requests)


@pytest.mark.parametrize(
    ("response", "reason"),
    [
        (_response("<think>x</think><answer>early</answer>"), "early_answer_tag"),
        (_response("<think>x</think>"), "finish_reason_stop"),
        (_response("no closed think", tool_calls=[_tool_call("call-x")], finish_reason="tool_calls"), "missing_think_close"),
        (_response("<think>x</think>", tool_calls=[], finish_reason="tool_calls"), "missing_tool_call"),
    ],
)
def test_forced_tool_turn_failure_reasons(response, reason):
    assert AsyncOpenAIChat._forced_tool_call_failure(response) == reason
    assert AsyncOpenAIChat._budget_forcing_failure_marker(reason) == (
        f"[LMMS_EVAL_BUDGET_FORCING_FAILED: {reason}]"
    )


def test_budget_forcing_failure_marker_scores_zero(monkeypatch):
    engine = SimpleNamespace(generate_json_batch=lambda prompts, use_tqdm: ['{"is_equivalent": true}'] * len(prompts))
    monkeypatch.setattr(visual_probe_utils, "get_judge_engine", lambda _: engine)
    results = [
        {
            "question": "question",
            "reference": "correct",
            "prediction": "correct",
            "raw_prediction": "<answer>early</answer>\n[LMMS_EVAL_BUDGET_FORCING_FAILED: finish_reason_stop]",
            "resp_key": "budget_failed",
        }
    ]

    assert visual_probe_utils.visual_probe_aggregate_judge_results(results) == 0.0
    assert visual_probe_utils.visual_probe_aggregate_judge_results.individual_scores["budget_failed"]["generation_failed"] is True


def test_budget_forcing_marks_initial_early_answer_invalid(tmp_path):
    model = _budget_model(
        tmp_path,
        _response("<think>done</think><answer>early</answer>"),
    )

    text, _, _, _ = asyncio.run(model.maybe_forward_with_tool(_budget_request(), 0))

    assert text.endswith("[LMMS_EVAL_BUDGET_FORCING_FAILED: early_answer_tag]")
    assert model.mcp_client.attempts == []
    assert len(model.client.chat.completions.requests) == 1


def test_budget_forcing_marks_intermediate_non_tool_turn_invalid(tmp_path):
    model = _budget_model(
        tmp_path,
        _response(
            "<think>first</think>",
            tool_calls=[_tool_call("call-1")],
            finish_reason="tool_calls",
        ),
        _response("<think>stopped</think>"),
    )

    text, _, _, _ = asyncio.run(model.maybe_forward_with_tool(_budget_request(), 0))

    assert text.endswith("[LMMS_EVAL_BUDGET_FORCING_FAILED: finish_reason_stop]")
    assert len(model.mcp_client.attempts) == 1
    assert [_forced_tag(request) for request in model.client.chat.completions.requests] == [
        "<tool_call>",
        "<tool_call>",
    ]


def test_budget_forcing_disabled_preserves_natural_initial_answer(tmp_path):
    model = _budget_model(
        tmp_path,
        _response("<think>done</think><answer>natural</answer>"),
    )
    model.force_tool_calls_until_final_turn = False

    text, _, _, _ = asyncio.run(model.maybe_forward_with_tool(_budget_request(), 0))

    assert text == "<think>done</think><answer>natural</answer>"
    assert "vllm_xargs" not in model.client.chat.completions.requests[0].get("extra_body", {})


def _enable_retry_critic(model, *, profile="persist", threshold=0.5):
    model.critic_guided_answer_retry = True
    model.critic_guided_answer_retry_prefill_profile = profile
    model.critic_guided_answer_retry_threshold = threshold
    model.force_tool_calls_until_final_turn = False
    return model


def test_retry_critic_request_serialization_preserves_system_and_uses_fixed_sidecar():
    model = _enable_retry_critic(_model())
    payload = {
        "model": "qwen_vllm",
        "messages": [
            {"role": "system", "content": "unchanged system"},
            {"role": "user", "content": "question"},
        ],
        "tools": [{"type": "function", "function": {"name": "zoom"}}],
        "tool_choice": "auto",
        "extra_body": {"top_k": 20},
    }

    request = model._retry_critic_request(payload, "<think>x</think><answer>candidate</answer>")

    assert payload["messages"] == [
        {"role": "system", "content": "unchanged system"},
        {"role": "user", "content": "question"},
    ]
    assert request["messages"][0] == payload["messages"][0]
    assert request["messages"][-3] == {
        "role": "assistant",
        "content": "<think>x</think><answer>candidate</answer>",
    }
    assert request["messages"][-2]["content"] == (
        'Assuming another tool round is available, should the preceding candidate be accepted or retried? '
        'In the <answer> block, output only "A" or "R".'
    )
    assert request["messages"][-1]["content"] == (
        "<think>\nI need to classify whether the preceding candidate should be accepted or retried.\n"
        "</think>\n<answer>\n"
    )
    assert request["max_tokens"] == 1
    assert request["temperature"] == 0.0
    assert request["logprobs"] is True
    assert request["top_logprobs"] == 2
    assert request["tool_choice"] == "none"
    assert request["extra_body"]["allowed_token_ids"] == [32, 49]
    assert request["extra_body"]["continue_final_message"] is True
    assert request["extra_body"]["add_generation_prompt"] is False
    assert request["extra_body"]["top_k"] == -1


def test_retry_critic_initialization_validates_single_token_labels(monkeypatch, tmp_path):
    tokenizer = SimpleNamespace(
        encode=lambda text, add_special_tokens=False: {"A": [32], "R": [49]}[text]
    )
    monkeypatch.setattr(async_openai_module, "_get_forced_action_tokenizer", lambda _: tokenizer)

    model = AsyncOpenAIChat(
        model_version="qwen_vllm",
        base_url="http://localhost:1/v1",
        api_key="test",
        num_cpus=1,
        work_dir=str(tmp_path),
        critic_guided_answer_retry=True,
    )

    assert model._retry_critic_accept_token_id == 32
    assert model._retry_critic_retry_token_id == 49


def test_retry_critic_initialization_rejects_non_distinct_labels(monkeypatch, tmp_path):
    tokenizer = SimpleNamespace(encode=lambda text, add_special_tokens=False: [7])
    monkeypatch.setattr(async_openai_module, "_get_forced_action_tokenizer", lambda _: tokenizer)

    with pytest.raises(ValueError, match="distinct single tokens"):
        AsyncOpenAIChat(
            model_version="qwen_vllm",
            base_url="http://localhost:1/v1",
            api_key="test",
            num_cpus=1,
            work_dir=str(tmp_path),
            critic_guided_answer_retry=True,
        )


def test_disabled_retry_critic_does_not_load_tokenizer(monkeypatch, tmp_path):
    def unexpected_tokenizer_load(_):
        raise AssertionError("disabled critic must not load its tokenizer")

    monkeypatch.setattr(async_openai_module, "_get_forced_action_tokenizer", unexpected_tokenizer_load)

    model = AsyncOpenAIChat(
        model_version="qwen_vllm",
        base_url="http://localhost:1/v1",
        api_key="test",
        num_cpus=1,
        work_dir=str(tmp_path),
        critic_guided_answer_retry=False,
    )

    assert model._forced_action_tokenizer is None


def test_retry_critic_probability_normalization_and_tie_acceptance():
    model = _enable_retry_critic(_model())
    token, selected, p_accept, p_retry = model._parse_retry_critic_response(
        _critic_response("R", accept_logprob=math.log(0.2), retry_logprob=math.log(0.8))
    )
    assert token == "R"
    assert selected == pytest.approx(math.log(0.8))
    assert p_accept == pytest.approx(0.2)
    assert p_retry == pytest.approx(0.8)

    response = _critic_response("A", accept_logprob=math.log(0.5), retry_logprob=math.log(0.5))
    # Parsing is deterministic at the boundary: selected A cannot round p(R) above 0.5.
    _, _, _, tied_p_retry = model._parse_retry_critic_response(response)
    assert tied_p_retry == 0.5
    assert not (tied_p_retry > model.critic_guided_answer_retry_threshold)


@pytest.mark.parametrize(
    ("profile", "prefill_visible"),
    [("persist", True), ("teacher_only", False)],
)
def test_retry_critic_feasible_retry_discards_candidate_and_matches_prefill_profile(
    tmp_path,
    profile,
    prefill_visible,
):
    model = _enable_retry_critic(
        _budget_model(
            tmp_path,
            _response("<think>guess</think><answer>wrong</answer>"),
            _critic_response("R", accept_logprob=math.log(0.1), retry_logprob=math.log(0.9)),
            _response(
                " I should inspect elsewhere.\n</think>",
                tool_calls=[_tool_call("retry-call")],
                finish_reason="tool_calls",
            ),
            _response("<think>verified</think><answer>correct</answer>"),
            _critic_response(
                "A",
                accept_logprob=math.log(0.9),
                retry_logprob=math.log(0.1),
                usage=SimpleNamespace(
                    prompt_tokens=13,
                    completion_tokens=1,
                    completion_tokens_details=SimpleNamespace(reasoning_tokens=0),
                ),
            ),
            max_tool_iterations=2,
            max_tool_images=10,
        ),
        profile=profile,
    )

    text, _, _, metadata = asyncio.run(model.maybe_forward_with_tool(_budget_request(), 3))

    assert "wrong" not in text
    assert "correct" in text
    assert "Assuming another tool round" not in text
    assert "<answer>\nR" not in text
    retry_prefill = "Let’s see. I still don’t have decisive evidence"
    assert (retry_prefill in text) is prefill_visible
    assert len(model.mcp_client.attempts) == 1
    decisions = metadata["retry_critic"]["decisions"]
    assert [record["controller_action"] for record in decisions] == ["retry", "accept"]
    assert decisions[0]["p_retry"] == pytest.approx(0.9)
    assert decisions[1]["retry_feasible"] is True
    assert metadata["retry_critic"]["summary"]["retries"] == 1
    assert metadata["retry_critic"]["summary"]["critic_token_usage"] == {
        "input_tokens": 13,
        "output_tokens": 1,
        "reasoning_tokens": 0,
        "total_tokens": 14,
    }
    requests = model.client.chat.completions.requests
    assert requests[1]["messages"][-2]["role"] == "user"
    assert requests[1]["messages"][-3]["content"].endswith("<answer>wrong</answer>")
    retry_request = requests[2]
    assert retry_request["messages"][-1]["content"].startswith("<think>\nLet’s see.")
    assert retry_request["extra_body"]["continue_final_message"] is True
    final_request_assistant_messages = [
        message for message in requests[3]["messages"] if message["role"] == "assistant"
    ]
    assert (retry_prefill in final_request_assistant_messages[-1]["content"]) is prefill_visible


def test_retry_critic_r_at_tool_limit_forces_accept(tmp_path):
    model = _enable_retry_critic(
        _budget_model(
            tmp_path,
            _response("<think>guess</think><answer>kept</answer>"),
            _critic_response("R", accept_logprob=math.log(0.1), retry_logprob=math.log(0.9)),
            max_tool_iterations=0,
            max_tool_images=10,
        )
    )

    text, _, _, metadata = asyncio.run(model.maybe_forward_with_tool(_budget_request(), 0))

    assert "<answer>kept</answer>" in text
    decision = metadata["retry_critic"]["decisions"][0]
    assert decision["controller_action"] == "forced_accept"
    assert decision["skip_or_error_reason"] == "max_tool_iterations"
    assert len(model.client.chat.completions.requests) == 2


def test_retry_critic_repeats_until_accept_or_existing_budget(tmp_path):
    retry = lambda: _critic_response(  # noqa: E731 - concise response fixture
        "R", accept_logprob=math.log(0.2), retry_logprob=math.log(0.8)
    )
    tool = lambda call_id: _response(  # noqa: E731 - concise response fixture
        " inspect another region.\n</think>",
        tool_calls=[_tool_call(call_id)],
        finish_reason="tool_calls",
    )
    model = _enable_retry_critic(
        _budget_model(
            tmp_path,
            _response("<think>one</think><answer>reject-one</answer>"),
            retry(),
            tool("retry-1"),
            _response("<think>two</think><answer>reject-two</answer>"),
            retry(),
            tool("retry-2"),
            _response("<think>three</think><answer>keep-three</answer>"),
            _critic_response("A", accept_logprob=math.log(0.8), retry_logprob=math.log(0.2)),
            max_tool_iterations=3,
            max_tool_images=10,
        )
    )

    text, _, _, metadata = asyncio.run(model.maybe_forward_with_tool(_budget_request(), 0))

    assert "reject-one" not in text
    assert "reject-two" not in text
    assert "keep-three" in text
    assert len(model.mcp_client.attempts) == 2
    assert [
        record["controller_action"] for record in metadata["retry_critic"]["decisions"]
    ] == ["retry", "retry", "accept"]


def test_retry_critic_r_at_image_limit_forces_accept(tmp_path):
    model = _enable_retry_critic(
        _budget_model(
            tmp_path,
            _response("<think>guess</think><answer>kept</answer>"),
            _critic_response("R", accept_logprob=math.log(0.1), retry_logprob=math.log(0.9)),
            max_tool_iterations=2,
            max_tool_images=0,
        )
    )

    text, _, _, metadata = asyncio.run(model.maybe_forward_with_tool(_budget_request(), 0))

    assert "<answer>kept</answer>" in text
    decision = metadata["retry_critic"]["decisions"][0]
    assert decision["controller_action"] == "forced_accept"
    assert decision["skip_or_error_reason"] == "max_tool_images"


def test_retry_critic_failure_accepts_candidate_without_sidecar_leakage(tmp_path):
    model = _enable_retry_critic(
        _budget_model(
            tmp_path,
            _response("<think>guess</think><answer>kept</answer>"),
            RuntimeError("critic unavailable"),
            max_tool_iterations=2,
            max_tool_images=10,
        )
    )

    text, _, _, metadata = asyncio.run(model.maybe_forward_with_tool(_budget_request(), 0))

    assert text == "<think>guess</think><answer>kept</answer>"
    decision = metadata["retry_critic"]["decisions"][0]
    assert decision["controller_action"] == "accept_fallback"
    assert "critic unavailable" in decision["skip_or_error_reason"]


def test_retry_critic_missing_logprobs_accepts_candidate(tmp_path):
    model = _enable_retry_critic(
        _budget_model(
            tmp_path,
            _response("<think>guess</think><answer>kept</answer>"),
            _response("A"),
            max_tool_iterations=2,
            max_tool_images=10,
        )
    )

    text, _, _, metadata = asyncio.run(model.maybe_forward_with_tool(_budget_request(), 0))

    assert text == "<think>guess</think><answer>kept</answer>"
    decision = metadata["retry_critic"]["decisions"][0]
    assert decision["controller_action"] == "accept_fallback"
    assert "did not include token logprobs" in decision["skip_or_error_reason"]


def test_retry_critic_malformed_forced_tool_uses_existing_failure_marker(tmp_path):
    model = _enable_retry_critic(
        _budget_model(
            tmp_path,
            _response("<think>guess</think><answer>discarded</answer>"),
            _critic_response("R", accept_logprob=math.log(0.1), retry_logprob=math.log(0.9)),
            _response("<think>failed</think>", finish_reason="stop"),
            max_tool_iterations=2,
            max_tool_images=10,
        )
    )

    text, _, _, metadata = asyncio.run(model.maybe_forward_with_tool(_budget_request(), 0))

    assert "discarded" not in text
    assert text.endswith("[LMMS_EVAL_BUDGET_FORCING_FAILED: finish_reason_stop]")
    assert metadata["retry_critic"]["summary"]["retries"] == 1


def _datasetbalanced_profile():
    from lmms_eval.models.model_utils.training_alignment import TrainingAlignment

    return TrainingAlignment(Path(__file__).resolve().parents[3] / "vero2_scripts/config/branchstrata20_fresh_lr1e5_critic_zoom_joint3_datasetbalanced.yaml")


def test_training_alignment_preserves_sampling_and_uses_training_prompt(tmp_path):
    profile = _datasetbalanced_profile()
    model = _budget_model(tmp_path, _response("<think>done</think><answer>correct</answer>"))
    model.training_alignment = profile
    profile.configure(model)
    # Avoid a critic side request for this prompt/sampling test.
    model.critic_guided_answer_retry = False

    async def count_tokens(payload):
        return 100

    model._training_prompt_tokens = count_tokens
    request = _budget_request()
    sampling = dict(temperature=0.37, top_p=0.8, top_k=20, repetition_penalty=1.1)
    request.args = (*request.args[:2], sampling, *request.args[3:])
    asyncio.run(model.maybe_forward_with_tool(request, 0))
    sent = model.client.chat.completions.requests[0]
    assert sent["messages"][0] == {"role": "system", "content": [{"type": "text", "text": profile.system_prompt}]}
    assert sent["tools"] == [profile.tool_schema]
    assert sent["temperature"] == 0.37
    assert sent["top_p"] == 0.8
    assert sent["extra_body"]["top_k"] == 20
    assert sent["extra_body"]["repetition_penalty"] == 1.1
    assert sent["max_tokens"] == 4096
    assert "_training_initial_tokens" not in sent
    assert sampling["temperature"] == 0.37


def test_training_alignment_critic_and_retry_text_are_verbatim():
    profile = _datasetbalanced_profile()
    model = _model()
    model.training_alignment = profile
    profile.configure(model)
    critic = model._retry_critic_request(_payload(), "<think>x</think><answer>candidate</answer>")
    assert critic["messages"][-2]["content"] == profile.multi["conditional_answer_retry_critic_question"]
    assert critic["messages"][-1]["content"] == "<think>\n" + profile.multi["conditional_answer_retry_critic_assistant_prefill"]
    retry, persisted = model._retry_tool_call_request(_payload())
    assert persisted == "<think>\n" + profile.multi["conditional_answer_retry_assistant_prefill"]
    assert retry["messages"][-1]["content"] == persisted
    assert _forced_tag(retry) == "<tool_call>"
    assert profile.image_indexing == "multi_index"
    assert model.max_tool_iterations == 10
    assert model.critic_guided_answer_retry_threshold == 0.5


def test_training_alignment_exports_exact_template_and_context(tmp_path):
    profile = _datasetbalanced_profile()
    template = tmp_path / "template.jinja"
    env = profile.shell_environment(template)
    assert template.read_text() == profile.template
    assert env["MAX_MODEL_LEN"] == 40960
    assert env["FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS"] is True
    assert env["FORCE_FINAL_ANSWER_PREFIX"] == ""
    assert env["FORCE_FINAL_ANSWER_THINK_PREFILL"] == profile.multi["force_final_answer_think_prefill"]
    assert not {"TEMPERATURE", "TOP_P", "TOP_K"} & env.keys()


def test_training_alignment_caps_remaining_tokens_and_does_not_charge_critic():
    profile = _datasetbalanced_profile()
    model = _model(_response("answer"), _response("A"))
    model.training_alignment = profile

    async def count_tokens(payload):
        return 100 + profile.data["max_response_length"] - 73

    model._training_prompt_tokens = count_tokens
    payload = {**_payload(), "_training_initial_tokens": 100, "max_tokens": 4096}
    asyncio.run(model._create_training_completion(payload))
    asyncio.run(model._create_training_completion({**payload, "max_tokens": 1}, critic=True))
    assert [r["max_tokens"] for r in model.client.chat.completions.requests] == [73, 1]
    assert all("_training_initial_tokens" not in r for r in model.client.chat.completions.requests)


def test_training_alignment_does_not_make_untrained_final_fallback():
    model = _model(_response("unfinished", finish_reason="length"))
    model.training_alignment = _datasetbalanced_profile()
    response, _, usage = asyncio.run(model._obtain_forced_final_answer(_payload(), 0))
    assert response.choices[0].finish_reason == "length"
    assert len(usage) == len(model.client.chat.completions.requests) == 1


def test_training_alignment_forces_answer_at_response_margin_before_tool_cap(tmp_path):
    model = _budget_model(
        tmp_path,
        _response("inspect</think>", tool_calls=[_tool_call("crop")], finish_reason="tool_calls"),
        _response("done</think><answer>correct</answer>"),
    )
    profile = _datasetbalanced_profile()
    model.training_alignment = profile
    profile.configure(model)
    model.critic_guided_answer_retry = False
    counts = iter([100, 100, 24000, 24010])

    async def count_tokens(payload):
        return next(counts)

    model._training_prompt_tokens = count_tokens
    asyncio.run(model.maybe_forward_with_tool(_budget_request(), 0))
    first, final = model.client.chat.completions.requests
    assert first["max_tokens"] == 4096
    assert _forced_tag(final) == "<answer>"
    assert final["max_tokens"] == 24576 - (24010 - 100)
    assert len(model.mcp_client.attempts) == 1


def test_training_alignment_prefers_recorded_checkpoint_prompt_over_changed_yaml(tmp_path):
    from lmms_eval.models.model_utils.training_alignment import TrainingAlignment

    profile = _datasetbalanced_profile()
    hf = tmp_path / "global_step_200/actor/huggingface"
    hf.mkdir(parents=True)
    (hf / "chat_template.jinja").write_text("checkpoint template")
    validation = tmp_path / "validation_data"
    validation.mkdir()
    schema = json.loads(json.dumps(profile.tool_schema))
    schema["function"]["parameters"]["properties"]["bbox_2d"] = {"type": "array", "description": "recorded schema"}
    row = {"debug": {"comprehensive_debug": {
        "raw_prompt": [{"role": "system", "content": [{"type": "text", "text": "recorded system\n"}]}],
        "debug_events": [
            {"event": "sample_start", "tool_prompt": [schema]},
            {"event": "tool_response", "tool_response_text": "Zoomed in on the image to the region [0, 0, 10, 10] with label test."},
        ],
    }}}
    (validation / "200.jsonl").write_text(json.dumps(row) + "\n")
    actual = TrainingAlignment(profile.path, str(hf))
    assert actual.template == "checkpoint template"
    assert actual.system_prompt == "recorded system\n"
    assert actual.tool_schema == schema
    assert actual.image_indexing == "training_faithful"
    assert actual.validation_evidence == str(validation / "200.jsonl")


def test_conservative_eval_threshold_survives_alignment_and_controls_retry(monkeypatch, tmp_path):
    monkeypatch.setenv("VERO_EVAL_CRITIC_THRESHOLD", "0.65")
    profile = _datasetbalanced_profile()
    model = _model(
        _critic_response("R", accept_logprob=math.log(0.4), retry_logprob=math.log(0.6)),
        _critic_response("R", accept_logprob=math.log(0.3), retry_logprob=math.log(0.7)),
    )
    profile.configure(model)
    assert profile.shell_environment(tmp_path / "template.jinja")["CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD"] == 0.65
    assert profile.multi["conditional_answer_retry_critic_decision_threshold"] == 0.5
    _, accept = asyncio.run(model._obtain_retry_critic_decision(_payload(), "candidate"))
    _, retry = asyncio.run(model._obtain_retry_critic_decision(_payload(), "candidate"))
    assert accept["decision"] == "A"
    assert retry["decision"] == "R"
    assert accept["threshold"] == retry["threshold"] == 0.65


@pytest.mark.parametrize(('task', 'p_retry', 'threshold', 'decision'), [
    ('visual_probe_easy_reasoning_zoom', 0.6, 0.65, 'A'),
    ('hstarbench_mixed_lite_reasoning_zoom', 0.6, 0.65, 'A'),
    ('pinpoint_bench_reasoning_zoom', 0.6, 0.5, 'R'),
    ('screenspotpro_point_in_box_lite_reasoning_zoom', 0.4, 0.35, 'R'),
    ('screenspot_point_in_box_reasoning_zoom', 0.4, 0.35, 'R'),
    ('unknown_task', 0.6, 0.65, 'A'),
])
def test_critic_uses_task_threshold_without_changing_shared_fallback(task, p_retry, threshold, decision):
    model = _model(_critic_response('R' if p_retry > .5 else 'A',
                                   accept_logprob=math.log(1-p_retry), retry_logprob=math.log(p_retry)))
    model.critic_guided_answer_retry_threshold = .65
    model.critic_domain_thresholds = dict(visual_probe=.65, hstar=.65, pinpoint=.5, screenspot=.35)
    _, record = asyncio.run(model._obtain_retry_critic_decision(_payload(), 'candidate', task=task))
    assert record['threshold'] == threshold
    assert record['decision'] == decision
    assert model.critic_guided_answer_retry_threshold == .65


def test_sample_metadata_records_effective_domain_threshold(tmp_path):
    from lmms_eval.models.model_utils.critic_thresholds import parse_domain_thresholds

    model = _budget_model(tmp_path,
                          _response('<think>done</think><answer>yes</answer>'),
                          _critic_response('A', accept_logprob=math.log(.9), retry_logprob=math.log(.1)))
    _enable_retry_critic(model, threshold=.65)
    model.force_tool_calls_until_final_turn = False
    model.critic_domain_thresholds = parse_domain_thresholds('{"screenspot":0.35}')
    task = 'screenspotpro_point_in_box_lite_reasoning_zoom'
    model.task_dict[task] = model.task_dict.pop('budget_task')
    request = _budget_request()
    request.args = (*request.args[:4], task, request.args[5])
    _, _, _, metadata = asyncio.run(model.maybe_forward_with_tool(request, 0))
    assert metadata['retry_critic']['threshold'] == .35
    assert metadata['retry_critic']['decisions'][0]['threshold'] == .35
    assert model.critic_guided_answer_retry_threshold == .65


@pytest.mark.parametrize("task,doc,expected", [
    ("screenspotpro_point_in_box_lite_reasoning_zoom", {"instruction": "add report in vivado"},
     '\nadd report in vivado In the your final answer, output only this JSON array: [{"point_2d": [x, y]}].'),
    ("hstarbench_mixed_lite_reasoning_zoom", {"task": "Find a red chair."},
     '\nFind a red chair. In the your final answer, output only this JSON array: [{"point_2d": [x, y]}].'),
    ("visual_probe_medium_reasoning_zoom", {"problem": "<image>\nWhat color is it?"}, '\nWhat color is it?'),
    ("pinpoint_bench_reasoning_zoom", {"problem": "What does the sign say?"}, '\nWhat does the sign say?'),
])
def test_combined_question_wrapper_preserves_media(task, doc, expected):
    profile = _datasetbalanced_profile()
    media = {"type": "image_url", "image_url": {"url": "image.png"}}
    messages = [{"role": "user", "content": [media, {"type": "text", "text": "eval-only wording"}]}]
    result = profile.align_user_prompt(messages, task, doc)
    assert result[0]["content"] == [media, {"type": "text", "text": expected}]
    assert messages[0]["content"][-1]["text"] == "eval-only wording"
    assert profile.align_user_prompt(messages, "unrelated_task", doc) is messages


def test_training_alignment_retry_uses_saved_run_config(tmp_path):
    import yaml
    from lmms_eval.models.model_utils.training_alignment import TrainingAlignment

    profile = _datasetbalanced_profile()
    saved = tmp_path / "debug_logs/config_resolved.yaml"
    saved.parent.mkdir()
    config = profile.config
    config["actor_rollout_ref"]["rollout"]["multi_turn"]["conditional_answer_retry_assistant_prefill"] = "Saved retry wording."
    saved.write_text(yaml.safe_dump(config))
    actual = TrainingAlignment(profile.path, tmp_path / "global_step_50/actor/huggingface")
    assert actual.multi["conditional_answer_retry_assistant_prefill"] == "Saved retry wording."
