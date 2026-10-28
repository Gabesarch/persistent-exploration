import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from omegaconf import OmegaConf
from PIL import Image

from verl.experimental.agent_loop.replay_judge_agent_loop import ReplayJudgeAgentLoop
from verl.workers.rollout.replica import TokenOutput


class CharacterTokenizer:
    eos_token = None

    def encode(self, text, add_special_tokens=False):
        return list(map(ord, text))

    def decode(self, ids, skip_special_tokens=True):
        return "".join(map(chr, ids))


def make_loop(*, prefix=True, response="Inspect evidence.</think><answer>A</answer>"):
    loop = ReplayJudgeAgentLoop.__new__(ReplayJudgeAgentLoop)
    loop.config = OmegaConf.create(
        {"trainer": {"trajectory_replay": {"enabled": True}, "comprehensive_debug_logging": {"enabled": False}}}
    )
    loop.rollout_config = SimpleNamespace(prompt_length=16384, response_length=24576)
    loop.tokenizer = CharacterTokenizer()
    loop.process_vision_info = AsyncMock(return_value={"images": [Image.new("RGB", (32, 32))]})
    loop.apply_chat_template = AsyncMock(
        return_value=loop.tokenizer.encode("assistant\n" + ("<think>\n" if prefix else ""))
    )
    loop.server_manager = SimpleNamespace(
        generate=AsyncMock(
            return_value=TokenOutput(
                token_ids=loop.tokenizer.encode(response), log_probs=[-0.1] * len(response), extra_fields={}
            )
        )
    )
    entry = {
        "question": "Question",
        "answer": "Candidate",
        "images": [Image.new("RGB", (32, 32))],
        "label": "A",
        "entry_id": "row",
        "turn": 1,
        "evidence_mode": "initial",
    }
    return loop, entry


@pytest.mark.parametrize("prefix", [True, False])
def test_current_policy_judge_prefill_once_and_sampled_tokens_use_only_pg_mask(prefix):
    loop, entry = make_loop(prefix=prefix)
    output = asyncio.run(
        loop.run(
            {"temperature": 1.0, "allowed_token_ids": [1], "extra_args": {"force_tag": "tool"}}, replay_candidate=entry
        )
    )
    params = loop.server_manager.generate.await_args.kwargs["sampling_params"]
    assert params["max_tokens"] == 5120 and params["temperature"] == 1.0
    assert "allowed_token_ids" not in params and "extra_args" not in params
    assert loop.tokenizer.decode(output.prompt_ids).count("<think>") == 1
    assert all(output.response_mask) and not any(output.retry_critic_mask) and not any(output.retry_sft_mask)
    assert len(output.response_ids) == len(output.response_logprobs) == sum(output.response_mask)
    assert output.reward_score == 1.0 and output.extra_fields["reward_extra_info"]["accuracy"] == 1.0
    assert output.extra_fields["is_replay_judge"] and output.extra_fields["gdpo_baseline_weight"] == 1.0
    assert output.extra_fields["crop_image_refs"] == []
    assert output.extra_fields["judge_reasoning_tokens"] > 0
    assert output.extra_fields["judge_prefill_active_tokens"] == 0
    trace = output.extra_fields["replay_judge_trace"]
    assert trace["policy_tokens"] == sum(output.response_mask)
    assert trace["retry_critic_tokens"] == sum(output.retry_critic_mask)
    assert trace["retry_sft_tokens"] == sum(output.retry_sft_mask)
    assert not output.branch_outputs
    messages = loop.apply_chat_template.await_args.args[0]
    assert "label" not in str(messages) and "ground_truth" not in str(messages)
    assert loop.apply_chat_template.await_args.kwargs["tools"] is None


def test_empty_think_completion_keeps_decision_accuracy_but_loses_format_reward():
    loop, entry = make_loop(response="</think><answer>A</answer>")
    output = asyncio.run(loop.run({"temperature": 1.0}, replay_candidate=entry))
    assert output.extra_fields["reward_extra_info"]["accuracy"] == 1
    assert output.extra_fields["reward_extra_info"]["raw_format"] == 0
    assert output.reward_score == 0.8


def test_judge_budget_does_not_inherit_actor_four_thousand_per_turn_cap():
    loop, entry = make_loop(response="x" * 5100)
    output = asyncio.run(loop.run({}, replay_candidate=entry))
    assert len(output.response_ids) == 5100
    assert output.reward_score < 0 and output.extra_fields["reward_extra_info"]["format"] < 0


def test_agent_rejects_prompt_overflow_before_policy_generation():
    loop, entry = make_loop()
    loop.rollout_config.prompt_length = 2
    with pytest.raises(ValueError, match="prompt_length"):
        asyncio.run(loop.run({}, replay_candidate=entry))
    loop.server_manager.generate.assert_not_awaited()
