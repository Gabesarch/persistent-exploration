"""Accuracy-only smoke contract, including the policy judge and raw token preservation."""

import asyncio
from pathlib import Path

import pytest
from PIL import Image

from verl.models.gemma4 import protocol as p
from verl.utils.reward_score.gemma4_native import compute_score_from_data_source as strict_score
from verl.utils.reward_score.gemma4_untagged_accuracy_smoke import (
    compute_score_from_data_source as smoke_score,
)
from verl.utils.reward_score.gemma4_untagged_accuracy_smoke import (
    parse_accuracy_trajectory,
)

ROOT = Path(__file__).resolve().parents[3]
CALL = '<|tool_call>call:image_zoom_in_tool{bbox_2d:[100,200,700,800],img_idx:0,label:<|"|>region<|"|>}<tool_call|>'


def final(answer):
    return p.THOUGHT + "Evidence, not an action: <answer>dog</answer>." + p.END_THOUGHT + answer


@pytest.mark.parametrize("tagged", [False, True])
def test_accuracy_only_and_no_sidecar_leakage(tagged):
    answer = "<answer>cat</answer>" if tagged else "cat"
    text = final(answer + "<turn|>") + p.USER + "Critic question<turn|>" + p.MODEL + "R<turn|>"
    kwargs = dict(
        data_source="fixture",
        solution_str=text,
        ground_truth="cat",
        extra_info={"reward_type": "search"},
        format_score=0.1,
        min_tool_calls_for_bonus=0,
        tool_call_bonus=0.2,
    )
    strict, relaxed = strict_score(**kwargs), smoke_score(**kwargs)
    assert parse_accuracy_trajectory(text)[:2] == (True, "cat")
    assert relaxed["accuracy"] == 1 and relaxed["format"] == int(tagged)
    for key in strict.keys() - {"accuracy", "score"}:
        assert relaxed[key] == strict[key]
    assert relaxed["score"] == pytest.approx(strict["score"] + (0 if tagged else 0.9))


@pytest.mark.parametrize(
    "action",
    [
        "cat",
        "<answer>cat<turn|>",
        "cat</answer><turn|>",
        "<answer></answer><turn|>",
        "<|tool_call>broken<turn|>",
        "<|tool_response>cat<turn|>",
        "<turn|>",
    ],
)
def test_malformed_truncated_and_feedback_not_answers(action):
    assert not parse_accuracy_trajectory(final(action))[0]


def test_crop_history_and_earlier_malformed_calls():
    prefix = final(CALL + p.TOOL_RESPONSE + "cat is feedback<tool_response|><turn|>" + p.USER + "crop<turn|>")
    text = prefix + p.MODEL + final("dog<turn|>")
    parsed = parse_accuracy_trajectory(text)
    assert parsed[:3] == (True, "dog", 1)
    assert parsed[3][0]["arguments"]["bbox_2d"] == [200, 100, 800, 700]
    assert not parse_accuracy_trajectory(text.replace("img_idx:0", "img_idx:-1"))[0]
    assert not parse_accuracy_trajectory(prefix)[0]
    assert not parse_accuracy_trajectory("cat<turn|>")[0]


@pytest.mark.parametrize("judge_accuracy", [1.0, 0.0, 0.5])
def test_live_controller_path_uses_policy_judge_and_keeps_raw_tokens(tmp_path, monkeypatch, judge_accuracy):
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    from verl.experimental.agent_loop.agent_loop import DictConfigWrap
    from verl.experimental.agent_loop.gemma4_tool_agent_loop import Gemma4ToolAgentLoop
    from verl.experimental.agent_loop.gemma4_untagged_accuracy_smoke import Gemma4UntaggedAccuracySmokeAgent
    from verl.utils.dataset.gemma4_dataset import Gemma4Dataset
    from verl.utils.tokenizer import hf_processor
    from verl.workers.rollout.replica import TokenOutput

    for key, value in {
        "VERL_ROOT": str(ROOT),
        "HF_HOME": "/scratch/gpfs/ZHUANGL/gs1693/hf_cache",
        "SAVE_PATH": str(tmp_path),
        "IMAGE_ROOT": "",
        "VERO2_CROP_CACHE_ROOT": str(tmp_path / "crops"),
    }.items():
        monkeypatch.setenv(key, value)
    stem = "branchstrata20_fresh_lr1e5_critic_gemma4_e4b_smoke2gpu"
    with initialize_config_dir(config_dir=str(ROOT / "vero2_scripts/config"), version_base=None):
        strict_cfg = compose(config_name=stem)
        cfg = compose(config_name=stem + "_untagged_accuracy")
    # Only reward extraction, agent registration, and artifact identity differ.
    before, after = (OmegaConf.to_container(c, resolve=False) for c in (strict_cfg, cfg))
    for c in (before, after):
        c["reward"]["custom_reward_function"]["path"] = "ignored"
        c["actor_rollout_ref"]["rollout"]["agent"]["default_agent_loop"] = "ignored"
        c["actor_rollout_ref"]["rollout"]["agent"]["agent_loop_config_path"] = "ignored"
        c["trainer"]["experiment_name"] = c["trainer"]["default_local_dir"] = "ignored"
    assert before == after
    processor = hf_processor("/scratch/gpfs/ZHUANGL/gs1693/hf_cache/google/gemma-4-E4B-it")
    sampled = final("cat<turn|>")
    sampled_ids = processor.tokenizer.encode(sampled, add_special_tokens=False)

    class Server:
        judge_calls = 0

        async def generate(self, **kwargs):
            if "-llm-judge-" in kwargs["request_id"]:
                self.judge_calls += 1
                prompt = processor.tokenizer.decode(kwargs["prompt_ids"], skip_special_tokens=False)
                assert "cat" in prompt and "<|think|>" not in prompt
                text = (
                    '"is_equivalent":'
                    + str(judge_accuracy == 1.0).lower()
                    + ',"partial_credit":'
                    + str(judge_accuracy == 0.5).lower()
                    + "}<turn|>"
                )
                ids = processor.tokenizer.encode(text, add_special_tokens=False)
            else:
                ids = list(sampled_ids)
            return TokenOutput(token_ids=ids, log_probs=[-0.1] * len(ids))

    image = tmp_path / "image.png"
    Image.new("RGB", (100, 100)).save(image)
    server = Server()

    async def run():
        loop = Gemma4UntaggedAccuracySmokeAgent(
            DictConfigWrap(cfg), server, processor.tokenizer, processor, Gemma4Dataset, DictConfigWrap(cfg.data)
        )
        assert not Gemma4ToolAgentLoop._parse_solver_output(loop, sampled)[0]
        assert not loop._has_final_answer(sampled)  # Retry/critic detection stays unchanged.
        return await asyncio.wait_for(
            loop.run(
                {"temperature": 1.0, "top_p": 1.0, "top_k": -1, "logprobs": True},
                raw_prompt=[
                    {
                        "role": "user",
                        "content": [{"type": "image", "image": str(image)}, {"type": "text", "text": "Which animal?"}],
                    }
                ],
                reward_model={"ground_truth": "cat"},
                data_source="fixture",
                extra_info={"reward_type": "search"},
                _debug_trajectory={"step": 1, "validate": False},
            ),
            timeout=30,
        )

    output = asyncio.run(run())
    assert server.judge_calls == 1
    assert output.response_ids == sampled_ids
    assert output.response_logprobs == [-0.1] * len(sampled_ids)
    assert sum(output.retry_critic_mask or []) == 0
    assert output.extra_fields["reward_extra_info"]["accuracy"] == judge_accuracy
    assert output.extra_fields["reward_extra_info"]["llm_judge_used"] is True
    assert output.extra_fields["reward_extra_info"]["format"] == 0
