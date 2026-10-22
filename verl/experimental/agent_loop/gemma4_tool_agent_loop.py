"""Native Gemma protocol around the unchanged branch/retry/critic controller."""

import json

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.models.gemma4 import protocol as native
from verl.utils.forced_action_tags import (
    FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY,
    FORCE_ACTION_REQUIRED_TAG_KEY,
    FORCE_ACTION_TAG_IDS_KEY,
)


@register("gemma4_tool_agent")
class Gemma4ToolAgentLoop(ToolAgentLoop):
    skip_response_special_tokens = False

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # The legacy Pydantic tool schema drops array items/minItems/maxItems.
        # Preserve the exact evaluation schema only in this opt-in agent.
        from omegaconf import OmegaConf

        tool_config = OmegaConf.load(self.rollout_config.multi_turn.tool_config_path)
        self.tool_schemas = [OmegaConf.to_container(tool.tool_schema, resolve=True) for tool in tool_config.tools]
        if self.processor.__class__.__name__ != "Gemma4Processor":
            raise ValueError("gemma4_tool_agent requires Gemma4Processor")
        for label in ("A", "R"):
            if len(self.tokenizer.encode(label, add_special_tokens=False)) != 1:
                raise ValueError(f"Gemma critic label {label} is not one token")

    def _has_final_answer(self, text):
        return native.split_thought(text)["answer"].lstrip().startswith("<answer>")

    def _parse_solver_output(self, text):
        return native.parse_trajectory(text)

    async def apply_chat_template(self, messages, tools=None, images=None, videos=None, remove_system_prompt=False):
        if videos:
            raise ValueError("Gemma critic recipe supports images only")
        if remove_system_prompt:
            raise ValueError("Gemma requires contextual continuation rendering")
        return native.encode(self.processor, native.render(self.processor, messages, tools), images)

    async def _append_conditional_retry_suffix_to_previous_message(self, agent_data, suffix):
        # The Gemma recipe has no user/tool suffix. After rollback, its native
        # continuation is already complete, including the contextual tool return
        # and crop image turn. Re-rendering an unchanged standalone block would
        # lose the preceding assistant call (or rewrite sampled history).
        # Retry thought prefills and action forcing are applied separately by the
        # controller on the next generation; retain its masking/branch behavior.
        if not suffix:
            return
        await super()._append_conditional_retry_suffix_to_previous_message(agent_data, suffix)

    def _append_assistant_prefill(self, agent_data, prefill, *, trainable):
        return super()._append_assistant_prefill(
            agent_data, native.THOUGHT + prefill if prefill else "", trainable=trainable
        )

    def _with_forced_action_extra_args(self, sampling_params, required_tag):
        action = "<|tool_call>call:" if required_tag == "<tool_call>" else "<answer>" + self.force_final_answer_prefix
        extra = dict(sampling_params.get("extra_args") or {})
        extra.update(
            {
                FORCE_ACTION_REQUIRED_TAG_KEY: required_tag,
                FORCE_ACTION_TAG_IDS_KEY: self.tokenizer.encode(action, add_special_tokens=False),
                FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY: [
                    self.tokenizer.encode(native.END_THOUGHT, add_special_tokens=False)
                ],
            }
        )
        return {**sampling_params, "extra_args": extra}

    async def _retry_critic_prompt_prefix_ids(self):
        text = "\n" + native.USER + self.conditional_answer_retry_critic_question + "<turn|>\n" + native.MODEL
        text += self.conditional_answer_retry_critic_assistant_prefill
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _critic_closing_text(self):
        return "\n</answer><turn|>\n"

    def _encode_llm_judge_prompt(self, judge_prompt):
        text = native.render(
            self.processor,
            [
                {
                    "role": "system",
                    "content": "You are a strict reward judge. Respond with exactly one JSON object and no extra text.",
                },
                {"role": "user", "content": judge_prompt},
            ],
            thinking=False,
        )
        return self.tokenizer.encode(text + (self.llm_judge_assistant_prefill or ""), add_special_tokens=False)

    @staticmethod
    def _tool_call_as_reward_dict(tool_call):
        call = ToolAgentLoop._tool_call_as_reward_dict(tool_call)
        if call and call["name"] == "image_zoom_in_tool":
            call["arguments"]["bbox_2d"] = native.xyxy(call["arguments"]["bbox_2d"])
        return call

    async def _render_tool_response(self, agent_data, messages, **kwargs):
        calls = [
            {"name": c.name, "arguments": json.loads(c.arguments)}
            for c in agent_data.tool_calls[: self.max_parallel_calls]
        ]
        texts = []
        for message in messages:
            content = message["content"]
            texts.append(content if isinstance(content, str) else "".join(c.get("text", "") for c in content))
        images = kwargs.get("images") or []
        response_token = self.tokenizer.convert_tokens_to_ids(native.TOOL_RESPONSE)
        ids = native.tool_continuation(
            self.processor,
            calls,
            texts,
            images,
            stopped_with_response=bool(agent_data.prompt_ids and agent_data.prompt_ids[-1] == response_token),
        )
        # Store the model-facing transcript for diagnostics; the sampled sequence is untouched.
        messages[:] = [{"role": "tool", "name": c["name"], "content": t} for c, t in zip(calls, texts, strict=True)]
        if images:
            messages.append(
                {
                    "role": "user",
                    "content": [{"type": "image"} for _ in images]
                    + [{"type": "text", "text": "Above are the tool-returned image(s)."}],
                }
            )
        return ids

    async def _debug_log_sample_final(self, output, agent_data, kwargs, trajectory):
        await super()._debug_log_sample_final(output, agent_data, kwargs, trajectory)
        if not self._debug_dir():
            return
        from verl.utils.debug.comprehensive_logging import append_jsonl, path_join

        def record(sample):
            return {
                "prompt_ids": sample.prompt_ids,
                "response_ids": sample.response_ids,
                "response_mask": sample.response_mask,
                "retry_sft_mask": sample.retry_sft_mask,
                "retry_critic_mask": sample.retry_critic_mask,
                "response_logprobs": sample.response_logprobs,
                "extra_fields": sample.extra_fields,
            }

        append_jsonl(
            path_join(self._debug_dir(), "gemma4_raw_trajectories.jsonl"),
            {
                "request_id": agent_data.request_id,
                "trajectory": trajectory,
                "main": record(output),
                "branches": [record(b) for b in output.branch_outputs or []],
                "crop_image_refs": agent_data.crop_image_refs,
                "events": agent_data.debug_events,
            },
        )
