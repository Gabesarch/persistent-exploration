"""On-policy, no-tool reasoning rollouts for sanitized cached candidates."""

import re
from pathlib import Path
from typing import Any
from uuid import uuid4

from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.debug.comprehensive_logging import append_jsonl
from verl.utils.profiler import simple_timer
from verl.utils.trajectory_replay import build_judge_messages, judge_rewards


def _reasoning_policy_tokens(tokenizer, response_ids: list[int], response_mask: list[int]) -> int:
    decoded = tokenizer.decode(response_ids, skip_special_tokens=True)
    closing = decoded.find("</think>")
    if closing < 0:
        return sum(response_mask)
    # Count sampled tokens ending before the closing tag, rather than counting
    # a retokenized reasoning string with potentially different BPE boundaries.
    low, high = 0, len(response_ids)
    while low < high:
        midpoint = (low + high + 1) // 2
        prefix = tokenizer.decode(response_ids[:midpoint], skip_special_tokens=True)
        if len(prefix) <= closing:
            low = midpoint
        else:
            high = midpoint - 1
    return sum(response_mask[:low])


@register("replay_judge_agent")
class ReplayJudgeAgentLoop(AgentLoopBase):
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        replay = self.config.trainer.get("trajectory_replay", {})
        if not bool(replay.get("enabled", False)):
            raise ValueError("replay_judge_agent requires trainer.trajectory_replay.enabled=true")
        candidate = kwargs["replay_candidate"]
        messages = build_judge_messages(
            candidate["question"],
            candidate["answer"],
            candidate["images"],
            system_prompt=replay.get("system_prompt_path") or replay.get("system_prompt"),
        )
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        if not images:
            raise ValueError("A replay judge prompt requires visual evidence")
        prompt_ids = list(await self.apply_chat_template(messages, tools=None, images=images, videos=None))
        rendered = self.tokenizer.decode(prompt_ids, skip_special_tokens=False)
        # The experiment's custom template already supplies this prefix. Native
        # templates also work, with the prefix kept in conditioning, never loss.
        if not re.search(r"<think>\s*$", rendered):
            prompt_ids.extend(self.tokenizer.encode("<think>\n", add_special_tokens=False))
        if len(prompt_ids) > int(self.rollout_config.prompt_length):
            raise ValueError("Replay judge prompt exceeds configured prompt_length; preflight it before sampling")
        normal_budget = int(replay.get("normal_budget", 4096))
        overlong_buffer = int(replay.get("overlong_buffer", 1024))
        max_tokens = normal_budget + overlong_buffer
        if max_tokens > int(self.rollout_config.response_length):
            raise ValueError("Replay judge response budget exceeds rollout response_length")
        params = {**sampling_params, "max_tokens": max_tokens}
        # A judge is an unconstrained reasoning completion, never a forced tool
        # action or the old constrained, teacher-forced one-token classifier.
        params.pop("extra_args", None)
        params.pop("allowed_token_ids", None)
        request_id = uuid4().hex
        metrics = {}
        with simple_timer("generate_sequences", metrics):
            generated = await self.server_manager.generate(
                request_id=request_id,
                prompt_ids=prompt_ids,
                sampling_params=params,
                image_data=images,
                video_data=None,
            )
        if len(generated.token_ids) > max_tokens:
            raise ValueError("Replay judge server exceeded its hard response budget")
        response_ids = list(generated.token_ids)
        decoded = self.tokenizer.decode(response_ids, skip_special_tokens=True)
        eos = getattr(self.tokenizer, "eos_token", None)
        if eos and decoded.endswith(eos):
            decoded = decoded[: -len(eos)]
        response_text = "<think>\n" + decoded
        rewards = judge_rewards(
            response_text,
            candidate["label"],
            len(response_ids),
            normal_budget=normal_budget,
            overlong_buffer=overlong_buffer,
            format_weight=float(replay.get("format_weight", 0.2)),
        )
        response_mask = [1] * len(response_ids)
        retry_critic_mask = [0] * len(response_ids)
        retry_sft_mask = [0] * len(response_ids)
        combined_policy_mask = [0] * len(prompt_ids) + response_mask
        prefill_tokens = len(self.tokenizer.encode("<think>\n", add_special_tokens=False))
        prefill_active_tokens = sum(combined_policy_mask[max(0, len(prompt_ids) - prefill_tokens) : len(prompt_ids)])
        reasoning_policy_tokens = _reasoning_policy_tokens(self.tokenizer, response_ids, response_mask)
        metrics["num_preempted"] = generated.num_preempted if generated.num_preempted is not None else -1
        trace = {
            "event": "replay_judge_rollout",
            "request_id": request_id,
            "step": (kwargs.get("_debug_trajectory") or {}).get("step"),
            "entry_id": candidate["entry_id"],
            "evidence_mode": candidate["evidence_mode"],
            "source_turn": candidate["turn"],
            "image_count": len(images),
            "prompt_tokens": len(prompt_ids),
            "generated_tokens": len(response_ids),
            "judge_generated_tokens": len(response_ids),
            "max_tokens": max_tokens,
            "thinking_prefill_in_prompt": True,
            "policy_tokens": sum(response_mask),
            "reasoning_policy_tokens": reasoning_policy_tokens,
            "prefill_active_tokens": prefill_active_tokens,
            "retry_critic_tokens": sum(retry_critic_mask),
            "retry_sft_tokens": sum(retry_sft_mask),
            "response_text": response_text,
            "response": response_text,
            "label": candidate["label"],
            "rewards": rewards,
        }
        debug = self.config.trainer.get("comprehensive_debug_logging", {})
        if debug and bool(debug.get("enabled", False)):
            step = trace["step"]
            interval = int(debug.get("log_every_n_steps", 1))
            if step is None or interval <= 1 or int(step) % interval == 0:
                directory = debug.get("dir") or str(Path(self.config.trainer.default_local_dir) / "debug_logs")
                append_jsonl(Path(directory) / "replay_judge_traces.jsonl", trace)
        return AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            retry_sft_mask=retry_sft_mask,
            retry_critic_mask=retry_critic_mask,
            response_logprobs=list(generated.log_probs) if generated.log_probs is not None else None,
            routed_experts=generated.routed_experts,
            multi_modal_data=multi_modal_data,
            reward_score=rewards["score"],
            num_turns=2,
            metrics=metrics,
            extra_fields={
                "is_replay_judge": True,
                "replay_judge_rewards": rewards,
                "reward_extra_info": rewards,
                "replay_candidate_entry_id": candidate["entry_id"],
                "replay_judge_trace": trace,
                "judge_reasoning_tokens": reasoning_policy_tokens,
                "judge_prefill_active_tokens": prefill_active_tokens,
                "request_id": request_id,
                "assistant_turns": 1,
                "is_conditional_answer_retry_branch": False,
                "gdpo_baseline_weight": 1.0,
                "conditional_answer_retry_count": 0,
                "crop_image_refs": [],
                "turn_scores": [],
                "tool_rewards": [],
            },
        )
