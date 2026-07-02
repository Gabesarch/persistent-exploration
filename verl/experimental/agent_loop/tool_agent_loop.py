# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import asyncio
import inspect
import json
import logging
import os
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.interactions.base import BaseInteraction
from verl.interactions.utils.interaction_registry import initialize_interactions_from_config
from verl.tools.schemas import ToolResponse
from verl.tools.utils.tool_registry import initialize_tools_from_config
from verl.trainer.ppo.reward import get_custom_reward_fn
from verl.utils.crop_cache import CROP_IMAGE_REFS_KEY, resolve_crop_cache_root
from verl.utils.debug.comprehensive_logging import (
    append_jsonl,
    get_debug_options,
    path_join,
    save_image,
    should_log_step,
    to_jsonable,
)
from verl.utils.profiler import simple_timer
from verl.utils.response_display import apply_response_display_prefix, get_response_display_prefix
from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
    _extract_reward_answer,
    _extract_reward_type,
    _parse_multiturn_tool_calling_output,
)
from verl.utils.rollout_trace import rollout_trace_op
from verl.utils.forced_action_tags import (
    FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY,
    FORCE_ACTION_REQUIRED_TAG_KEY,
    FORCE_ACTION_TAG_IDS_KEY,
)
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

TERMINAL_TOOL_EXTRA_FIELD_DEFAULTS: dict[str, Any] = {
    "terminated_by_tool_error": False,
    "terminal_tool_name": None,
    "terminal_tool_error": None,
    "terminal_reason": None,
    "forced_structure_error": False,
    "forced_structure_error_reason": None,
    "forced_structure_error_assistant_turn": None,
    "conditional_answer_retry_count": 0,
    "conditional_answer_retry_last_grade": None,
    "conditional_answer_retry_skipped_reason": None,
}

_LLM_JUDGE_SYSTEM_PROMPT = (
    "You are a strict reward judge. Respond with exactly one JSON object and no extra text."
)
_VISION_PLACEHOLDER_REPLACEMENTS = (
    "<|vision_start|><|image_pad|><|vision_end|>",
    "<|vision_start|><|video_pad|><|vision_end|>",
    "<image>",
    "<video>",
)


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"
    INTERACTING = "interacting"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
        original_prompt_image_sources: Optional[list[Any]] = None,
        crop_cache_root: Optional[str] = None,
        crop_cache_step: Optional[int | str] = None,
        interaction: Optional[BaseInteraction] = None,
        interaction_kwargs: Optional[dict[str, Any]] = None,
    ):
        self.messages = messages
        self.image_data = image_data
        self.video_data = video_data
        self.original_prompt_image_sources = original_prompt_image_sources or []
        self.crop_cache_root = crop_cache_root
        self.crop_cache_step = crop_cache_step
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs
        self.interaction = interaction
        self.interaction_kwargs = interaction_kwargs or {}

        # State variables
        self.prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}
        self.debug_events: list[dict[str, Any]] = []
        self.crop_image_refs: list[dict[str, Any]] = []
        self.force_next_tool_call = False
        self.last_user_tool_prompt_start: int | None = None
        self.last_user_tool_prompt_end: int | None = None
        self.last_user_tool_response_mask_start: int | None = None
        self.last_user_tool_response_mask_end: int | None = None
        self.last_user_tool_response_logprobs_start: int | None = None
        self.last_user_tool_response_logprobs_end: int | None = None
        self.last_user_tool_message_index: int | None = None
        self.last_user_tool_block_start_index: int | None = None
        self.last_user_tool_block_images: list[Any] | None = None
        self.last_user_tool_block_videos: list[Any] | None = None


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Initialize tools from config file
        multi_turn_config = self.rollout_config.multi_turn
        self.max_user_turns = multi_turn_config.max_user_turns
        self.max_assistant_turns = multi_turn_config.max_assistant_turns
        self.max_parallel_calls = multi_turn_config.max_parallel_calls
        self.max_tool_response_length = multi_turn_config.max_tool_response_length
        self.tool_response_truncate_side = multi_turn_config.tool_response_truncate_side
        self.force_tool_calls_until_final_turn = getattr(multi_turn_config, "force_tool_calls_until_final_turn", False)
        self.final_answer_token_margin = getattr(multi_turn_config, "final_answer_token_margin", 2048)
        self.force_final_answer_prefix = getattr(multi_turn_config, "force_final_answer_prefix", "")
        self.tool_response_turn_suffix_enabled = getattr(multi_turn_config, "tool_response_turn_suffix_enabled", False)
        self.tool_response_explore_turns = getattr(multi_turn_config, "tool_response_explore_turns", 10)
        self.tool_response_explore_suffix = getattr(multi_turn_config, "tool_response_explore_suffix", "")
        self.tool_response_final_suffix = getattr(multi_turn_config, "tool_response_final_suffix", "")
        self.conditional_answer_retry_enabled = getattr(
            multi_turn_config, "conditional_answer_retry_enabled", False
        )
        self.conditional_answer_retry_force_next_tool_call = getattr(
            multi_turn_config, "conditional_answer_retry_force_next_tool_call", True
        )
        self.conditional_answer_retry_suffix = getattr(
            multi_turn_config,
            "conditional_answer_retry_suffix",
            (
                "You have not gathered enough information yet. In your next turn, keep exploring rather than "
                "answering prematurely: use more tool calls to broaden the search, test alternative hypotheses, "
                "look for disconfirming evidence, and dig deeper into the most likely explanations. Finalize only "
                "when the evidence clearly supports a conclusion."
            ),
        )
        self.conditional_answer_retry_assistant_prefill = getattr(
            multi_turn_config, "conditional_answer_retry_assistant_prefill", ""
        )
        self.conditional_answer_retry_correct_accuracy_threshold = getattr(
            multi_turn_config, "conditional_answer_retry_correct_accuracy_threshold", 0.0
        )
        self.conditional_answer_retry_max_retries = getattr(
            multi_turn_config, "conditional_answer_retry_max_retries", None
        )
        self.llm_judge_enabled = bool(getattr(multi_turn_config, "llm_judge_enabled", False))
        self.conditional_answer_retry_reward_fn = (
            get_custom_reward_fn(self.config)
            if self.conditional_answer_retry_enabled or self.llm_judge_enabled
            else None
        )
        self.llm_judge_prompt_path = getattr(multi_turn_config, "llm_judge_prompt_path", None)
        self.llm_judge_prompt_template = self._load_llm_judge_prompt_template(self.llm_judge_prompt_path)
        self.llm_judge_reward_types = set(getattr(multi_turn_config, "llm_judge_reward_types", ["search"]) or [])
        self.llm_judge_max_tokens = int(getattr(multi_turn_config, "llm_judge_max_tokens", 48))
        self.llm_judge_temperature = float(getattr(multi_turn_config, "llm_judge_temperature", 0.0))
        self.llm_judge_top_p = float(getattr(multi_turn_config, "llm_judge_top_p", 1.0))
        self.llm_judge_partial_credit_score = float(
            getattr(multi_turn_config, "llm_judge_partial_credit_score", 0.5)
        )
        self.llm_judge_timeout_s = getattr(multi_turn_config, "llm_judge_timeout_s", None)
        self.llm_judge_fallback_to_string_reward = bool(
            getattr(multi_turn_config, "llm_judge_fallback_to_string_reward", True)
        )
        self.llm_judge_assistant_prefill = getattr(multi_turn_config, "llm_judge_assistant_prefill", "{")
        reward_config = getattr(self.config, "reward", {}) if self.config is not None else {}
        reward_kwargs = dict(reward_config.get("custom_reward_function", {}).get("reward_kwargs", {}))
        self.llm_judge_format_score = float(reward_kwargs.get("format_score", 0.5))
        self.llm_judge_tool_efficiency_score_weight = float(reward_kwargs.get("tool_efficiency_score_weight", 0.0))
        tool_config_path = multi_turn_config.tool_config_path
        tool_list = initialize_tools_from_config(tool_config_path) if tool_config_path else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format
        self.debug_options = get_debug_options(self.config)
        self.response_display_prefix = get_response_display_prefix(self.config)
        lazy_multi_modal_config = self.rollout_config.get("lazy_multi_modal_inputs", {})
        self.crop_cache_root = str(resolve_crop_cache_root(lazy_multi_modal_config.get("crop_cache_root")))

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

        # Initialize interactions from config file
        self.interaction_config_file = self.rollout_config.multi_turn.interaction_config_path
        if self.interaction_config_file:
            self.interaction_map: dict[str, BaseInteraction] = self._initialize_interactions(
                self.interaction_config_file
            )

    @staticmethod
    def _collect_original_prompt_image_sources(messages: list[dict[str, Any]]) -> list[Any]:
        sources = []
        for message in messages:
            content = message.get("content") if isinstance(message, dict) else None
            content_items = content if isinstance(content, list) else [content]
            for item in content_items:
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "image" or "image" in item or "image_url" in item:
                    sources.append(dict(item))
        return sources

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])
        debug_trajectory = kwargs.get("_debug_trajectory", {})
        original_prompt_image_sources = self._collect_original_prompt_image_sources(messages)

        # extract images and videos from messages
        multi_modal_data = await self.process_vision_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        # Initialize interaction if needed
        interaction = None
        interaction_kwargs = {}
        if self.interaction_config_file:
            interaction_kwargs = kwargs["extra_info"]["interaction_kwargs"]
            if "name" not in interaction_kwargs:
                raise ValueError("'name' key is required in interaction_kwargs")
            interaction_name = interaction_kwargs["name"]
            if interaction_name not in self.interaction_map:
                raise ValueError(
                    f"Interaction '{interaction_name}' not found in interaction_map. Available interactions: "
                    f"{list(self.interaction_map.keys())}"
                )
            interaction = self.interaction_map[interaction_name]
            await interaction.start_interaction(request_id, **interaction_kwargs)
        # Create AgentData instance to encapsulate all state
        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
            original_prompt_image_sources=original_prompt_image_sources,
            crop_cache_root=self.crop_cache_root,
            crop_cache_step=debug_trajectory.get("step"),
            interaction=interaction,
            interaction_kwargs=interaction_kwargs,
        )
        debug_enabled = self._debug_enabled_for_trajectory(debug_trajectory)
        agent_data.debug_logging_enabled = debug_enabled
        if debug_enabled:
            agent_data.extra_fields["debug_trajectory"] = debug_trajectory
            await self._debug_log_sample_start(agent_data, kwargs, debug_trajectory, sampling_params)

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params, rollout_kwargs=kwargs)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            elif state == AgentState.INTERACTING:
                state = await self._handle_interacting_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :]
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields={**TERMINAL_TOOL_EXTRA_FIELD_DEFAULTS, **agent_data.extra_fields},
        )
        output.extra_fields.update(
            {
                "request_id": agent_data.request_id,
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                CROP_IMAGE_REFS_KEY: agent_data.crop_image_refs,
            }
        )
        final_grade = await self._maybe_score_final_with_llm_judge(agent_data, kwargs)
        if final_grade is not None:
            output.reward_score = float(final_grade.get("score", 0.0) or 0.0)
            output.extra_fields["reward_extra_info"] = final_grade
        if debug_enabled:
            await self._debug_log_sample_final(output, agent_data, kwargs, debug_trajectory)
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        prompt_ids = await self.apply_chat_template(
            agent_data.messages,
            tools=self.tool_schemas,
            images=agent_data.image_data,
            videos=agent_data.video_data,
        )
        agent_data.prompt_ids = prompt_ids
        self._record_last_user_tool_prompt_block(
            agent_data=agent_data,
            prompt_start=0,
            prompt_end=len(prompt_ids),
            response_mask_start=0,
            response_mask_end=0,
            response_logprobs_start=0,
            response_logprobs_end=0,
            block_start_index=0,
            block_images=agent_data.image_data,
            block_videos=agent_data.video_data,
        )
        return AgentState.GENERATING

    def _must_answer_now(self, agent_data: AgentData) -> bool:
        """Whether the upcoming assistant turn must produce a final answer rather than a tool call.

        True on the final allowable assistant turn, when max user turns is reached, or when the
        remaining response budget is too small to safely fit another tool round plus a final answer.
        """
        return (
            (self.max_assistant_turns and agent_data.assistant_turns + 1 >= self.max_assistant_turns)
            or (self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
            or (len(agent_data.response_mask) + self.final_answer_token_margin >= self.response_length)
        )

    def _with_forced_action_extra_args(self, sampling_params: dict[str, Any], required_tag: str) -> dict[str, Any]:
        # The semantic tag passed to vLLM (and used for downstream validation) stays `required_tag`,
        # but the actually-forced token sequence may extend past it. On the forced final turn we can
        # additionally force a fixed answer prefix (e.g. "\n\\boxed{") right after the `<answer>` tag.
        forced_text = f"\n{required_tag}"
        if required_tag == "<answer>" and self.force_final_answer_prefix:
            forced_text += self.force_final_answer_prefix
        tag_ids = self.tokenizer.encode(forced_text, add_special_tokens=False)
        if not tag_ids:
            raise ValueError(f"required action tag encoded to no tokens: {required_tag}")

        extra_args = dict(sampling_params.get("extra_args") or {})
        extra_args.update(
            {
                FORCE_ACTION_REQUIRED_TAG_KEY: required_tag,
                FORCE_ACTION_TAG_IDS_KEY: tag_ids,
                FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY: [
                    self.tokenizer.encode("</think>", add_special_tokens=False),
                ],
            }
        )
        return {**sampling_params, "extra_args": extra_args}

    def _tool_response_turn_suffix(self, assistant_turn: int) -> str:
        if not self.tool_response_turn_suffix_enabled:
            return ""
        if assistant_turn < self.tool_response_explore_turns:
            return self.tool_response_explore_suffix or ""
        if assistant_turn == self.tool_response_explore_turns:
            return self.tool_response_final_suffix or ""
        return ""

    @staticmethod
    def _append_text_suffix(text: str, suffix: str) -> str:
        if not suffix:
            return text
        if not text:
            return suffix
        return f"{text}\n\n{suffix}"

    def _append_suffix_to_tool_message(self, message: dict[str, Any], suffix: str) -> dict[str, Any]:
        if not suffix:
            return message

        content = message.get("content")
        if isinstance(content, str):
            return {**message, "content": self._append_text_suffix(content, suffix)}

        if isinstance(content, list):
            new_content: list[Any] = []
            appended = False
            for item in content:
                if not appended and isinstance(item, dict) and item.get("type") == "text" and "text" in item:
                    new_content.append({**item, "text": self._append_text_suffix(str(item.get("text") or ""), suffix)})
                    appended = True
                else:
                    new_content.append(item)
            if not appended:
                new_content.append({"type": "text", "text": suffix})
            return {**message, "content": new_content}

        return {**message, "content": suffix}

    async def _handle_generating_state(
        self,
        agent_data: AgentData,
        sampling_params: dict[str, Any],
        ignore_termination: bool = False,
        rollout_kwargs: Optional[dict[str, Any]] = None,
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        add_messages: list[dict[str, Any]] = []
        forced_required_tag: Optional[str] = None
        assistant_prefill_ids: list[int] = []
        generation_snapshot = self._snapshot_generation_state(agent_data)

        if self.force_tool_calls_until_final_turn:
            forced_required_tag = "<answer>" if self._must_answer_now(agent_data) else "<tool_call>"
            sampling_params = self._with_forced_action_extra_args(sampling_params, forced_required_tag)
        elif getattr(self, "conditional_answer_retry_enabled", False) and agent_data.force_next_tool_call:
            forced_required_tag = "<tool_call>"
            sampling_params = self._with_forced_action_extra_args(sampling_params, forced_required_tag)
            assistant_prefill_ids = self._append_conditional_retry_assistant_prefill(agent_data)
            agent_data.force_next_tool_call = False

        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            if assistant_prefill_ids:
                agent_data.response_logprobs += [0.0] * len(assistant_prefill_ids)
            agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        if self._debug_enabled_for_agent(agent_data):
            assistant_message = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(
                    assistant_prefill_ids + agent_data.response_ids, skip_special_tokens=True
                ),
            )
            assistant_message = apply_response_display_prefix(assistant_message, self.response_display_prefix)
            event = {
                "event": "assistant_turn",
                "request_id": agent_data.request_id,
                "assistant_turn": agent_data.assistant_turns,
                "text": assistant_message,
                "num_tokens": len(agent_data.response_ids),
            }
            agent_data.debug_events.append(event)
            self._debug_append_event(event)

        # Extract tool calls before termination checks so forced final turns can be validated.
        tools = [tool.tool_schema for tool in self.tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        if forced_required_tag == "<tool_call>" and not agent_data.tool_calls:
            self._mark_forced_structure_error(agent_data, "missing_tool_call")
            return AgentState.TERMINATED
        if forced_required_tag == "<answer>":
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            if agent_data.tool_calls or "<answer>" not in assistant_message:
                reason = "unexpected_tool_call_on_final_turn" if agent_data.tool_calls else "missing_final_answer"
                self._mark_forced_structure_error(agent_data, reason)
                return AgentState.TERMINATED

        if (
            getattr(self, "conditional_answer_retry_enabled", False)
            and not agent_data.tool_calls
            and rollout_kwargs is not None
        ):
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            assistant_message = apply_response_display_prefix(assistant_message, self.response_display_prefix)
            if "<answer>" in assistant_message:
                retry_state = await self._maybe_retry_incorrect_answer(
                    agent_data=agent_data,
                    generation_snapshot=generation_snapshot,
                    assistant_message=assistant_message,
                    rollout_kwargs=rollout_kwargs,
                    ignore_termination=ignore_termination,
                )
                if retry_state is not None:
                    return retry_state

        # Check termination conditions
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        if self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns:
            return AgentState.TERMINATED
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return AgentState.TERMINATED

        # Handle interaction if needed
        if self.interaction_config_file:
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            add_messages.append({"role": "assistant", "content": assistant_message})
            agent_data.messages.extend(add_messages)

        # Determine next state
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        elif self.interaction_config_file:
            return AgentState.INTERACTING
        else:
            return AgentState.TERMINATED

    def _append_conditional_retry_assistant_prefill(self, agent_data: AgentData) -> list[int]:
        prefill = self.conditional_answer_retry_assistant_prefill or ""
        if not prefill:
            return []
        prefill_ids = self.tokenizer.encode(prefill, add_special_tokens=False)
        if not prefill_ids:
            return []
        if len(agent_data.response_mask) + len(prefill_ids) >= self.response_length:
            return []
        agent_data.prompt_ids += prefill_ids
        agent_data.response_mask += [0] * len(prefill_ids)
        return prefill_ids

    def _snapshot_generation_state(self, agent_data: AgentData) -> dict[str, Any]:
        return {
            "prompt_ids_len": len(agent_data.prompt_ids),
            "response_mask_len": len(agent_data.response_mask),
            "response_logprobs_len": len(agent_data.response_logprobs),
            "assistant_turns": agent_data.assistant_turns,
            "extra_fields": dict(agent_data.extra_fields),
            "debug_events_len": len(agent_data.debug_events),
        }

    def _rollback_generation_state(self, agent_data: AgentData, snapshot: dict[str, Any]) -> None:
        agent_data.prompt_ids = agent_data.prompt_ids[: snapshot["prompt_ids_len"]]
        agent_data.response_ids = []
        agent_data.response_mask = agent_data.response_mask[: snapshot["response_mask_len"]]
        agent_data.response_logprobs = agent_data.response_logprobs[: snapshot["response_logprobs_len"]]
        agent_data.assistant_turns = snapshot["assistant_turns"]
        agent_data.tool_calls = []
        agent_data.extra_fields = snapshot["extra_fields"]
        agent_data.debug_events = agent_data.debug_events[: snapshot["debug_events_len"]]

    async def _maybe_retry_incorrect_answer(
        self,
        agent_data: AgentData,
        generation_snapshot: dict[str, Any],
        assistant_message: str,
        rollout_kwargs: dict[str, Any],
        ignore_termination: bool,
    ) -> AgentState | None:
        grade = await self._grade_current_response(agent_data, rollout_kwargs)
        agent_data.extra_fields["conditional_answer_retry_last_grade"] = grade
        accuracy = self._grade_accuracy(grade)
        if accuracy > float(self.conditional_answer_retry_correct_accuracy_threshold):
            return None

        retry_count = int(agent_data.extra_fields.get("conditional_answer_retry_count") or 0)
        skipped_reason = self._incorrect_answer_retry_skipped_reason(
            agent_data=agent_data,
            snapshot=generation_snapshot,
            retry_count=retry_count,
        )
        if skipped_reason is not None:
            agent_data.extra_fields["conditional_answer_retry_skipped_reason"] = skipped_reason
            self._log_incorrect_answer_retry_skipped(agent_data, grade, retry_count, skipped_reason)
            return None

        self._rollback_generation_state(agent_data, generation_snapshot)
        retry_count += 1
        suffix = self.conditional_answer_retry_suffix or ""
        await self._append_conditional_retry_suffix_to_previous_message(agent_data, suffix)
        if self.conditional_answer_retry_force_next_tool_call:
            agent_data.force_next_tool_call = True
        agent_data.extra_fields["conditional_answer_retry_count"] = retry_count
        agent_data.extra_fields["conditional_answer_retry_last_grade"] = grade
        agent_data.extra_fields["conditional_answer_retry_skipped_reason"] = None
        self._log_incorrect_answer_retry(
            agent_data=agent_data,
            grade=grade,
            retry_count=retry_count,
            suffix=suffix,
            removed_assistant_turn=generation_snapshot["assistant_turns"] + 1,
            removed_assistant_text=assistant_message,
        )

        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        return AgentState.GENERATING

    async def _grade_current_response(self, agent_data: AgentData, rollout_kwargs: dict[str, Any]) -> dict[str, Any]:
        if self.conditional_answer_retry_reward_fn is None:
            return {"accuracy": 0.0, "error": "conditional answer retry requires a custom reward function"}

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :] if agent_data.response_mask else []
        solution_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )
        solution_str = apply_response_display_prefix(solution_str, self.response_display_prefix)
        return await self._score_solution_with_optional_llm_judge(
            agent_data=agent_data,
            solution_str=solution_str,
            rollout_kwargs=rollout_kwargs,
            use_llm_judge=bool(getattr(self, "llm_judge_enabled", False)),
        )

    async def _maybe_score_final_with_llm_judge(
        self, agent_data: AgentData, rollout_kwargs: dict[str, Any]
    ) -> dict[str, Any] | None:
        if not getattr(self, "llm_judge_enabled", False):
            return None
        if self.conditional_answer_retry_reward_fn is None:
            return None

        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :] if agent_data.response_mask else []
        solution_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(response_ids, skip_special_tokens=True)
        )
        solution_str = apply_response_display_prefix(solution_str, self.response_display_prefix)
        grade = await self._score_solution_with_optional_llm_judge(
            agent_data=agent_data,
            solution_str=solution_str,
            rollout_kwargs=rollout_kwargs,
            use_llm_judge=True,
        )
        if grade.get("llm_judge_used"):
            return grade
        if grade.get("llm_judge_error") and not getattr(self, "llm_judge_fallback_to_string_reward", True):
            return grade
        return None

    async def _score_solution_with_optional_llm_judge(
        self,
        agent_data: AgentData,
        solution_str: str,
        rollout_kwargs: dict[str, Any],
        use_llm_judge: bool,
    ) -> dict[str, Any]:
        reward_model = rollout_kwargs.get("reward_model") or {}
        ground_truth = reward_model.get("ground_truth") if isinstance(reward_model, dict) else None
        extra_info = rollout_kwargs.get("extra_info")
        base_result = self.conditional_answer_retry_reward_fn(
            data_source=rollout_kwargs.get("data_source", "unknown"),
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        if inspect.isawaitable(base_result):
            base_result = await base_result
        if isinstance(base_result, dict):
            grade = dict(base_result)
        else:
            grade = {"score": float(base_result), "accuracy": float(base_result)}

        if not use_llm_judge or not self._llm_judge_applies(extra_info):
            return grade

        judge_result = await self._run_llm_judge(
            agent_data=agent_data,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
        )
        if judge_result.get("llm_judge_used"):
            return self._merge_llm_judge_accuracy(grade, judge_result)
        if getattr(self, "llm_judge_fallback_to_string_reward", True):
            grade.update(judge_result)
            return grade

        grade.update(judge_result)
        grade["accuracy"] = 0.0
        grade["score"] = self._compose_score_from_grade(grade, 0.0)
        return grade

    def _llm_judge_applies(self, extra_info: Any) -> bool:
        reward_type = _extract_reward_type(extra_info)
        return bool(reward_type and reward_type in getattr(self, "llm_judge_reward_types", {"search"}))

    @staticmethod
    def _load_llm_judge_prompt_template(prompt_path: str | None) -> str | None:
        if not prompt_path:
            return None
        try:
            with open(os.path.expandvars(str(prompt_path)), encoding="utf-8") as handle:
                return handle.read()
        except OSError as exc:
            logger.warning("failed to read llm judge prompt template %s: %s", prompt_path, exc)
            return None

    async def _run_llm_judge(
        self,
        agent_data: AgentData,
        solution_str: str,
        ground_truth: Any,
        extra_info: Any,
    ) -> dict[str, Any]:
        if not getattr(self, "llm_judge_prompt_template", None):
            return {"llm_judge_used": False, "llm_judge_error": "missing_prompt_template"}

        is_valid, final_answer_text, _ = _parse_multiturn_tool_calling_output(solution_str)
        if not is_valid or not final_answer_text:
            return {"llm_judge_used": False, "llm_judge_error": "invalid_trajectory_format"}

        reward_type = _extract_reward_type(extra_info)
        final_answer = _extract_reward_answer(final_answer_text, reward_type)
        if final_answer is None:
            return {"llm_judge_used": False, "llm_judge_error": "missing_final_answer"}

        question = self._extract_llm_judge_question(agent_data, extra_info)
        if not question:
            return {"llm_judge_used": False, "llm_judge_error": "missing_question"}

        judge_prompt = self._format_llm_judge_prompt(
            question=question,
            reference=str(ground_truth).strip(),
            prediction=str(final_answer).strip(),
        )
        prompt_ids = self._encode_llm_judge_prompt(judge_prompt)
        sampling_params = {
            "max_tokens": int(getattr(self, "llm_judge_max_tokens", 48)),
            "temperature": float(getattr(self, "llm_judge_temperature", 0.0)),
            "top_p": float(getattr(self, "llm_judge_top_p", 1.0)),
            "top_k": -1,
            "logprobs": False,
        }

        try:
            judge_coro = self.server_manager.generate(
                request_id=f"{agent_data.request_id}-llm-judge-{uuid4().hex}",
                prompt_ids=prompt_ids,
                sampling_params=sampling_params,
                image_data=None,
                video_data=None,
            )
            timeout_s = getattr(self, "llm_judge_timeout_s", None)
            output = await asyncio.wait_for(judge_coro, timeout=float(timeout_s)) if timeout_s else await judge_coro
            raw_suffix = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(output.token_ids, skip_special_tokens=True)
            )
            parsed = self._parse_llm_judge_response(raw_suffix)
        except Exception as exc:  # noqa: BLE001 - judge failures should be recoverable.
            return {"llm_judge_used": False, "llm_judge_error": f"{type(exc).__name__}: {exc}"}

        is_equivalent = bool(parsed.get("is_equivalent"))
        partial_credit = bool(parsed.get("partial_credit"))
        if is_equivalent:
            accuracy = 1.0
        elif partial_credit:
            accuracy = float(getattr(self, "llm_judge_partial_credit_score", 0.5))
        else:
            accuracy = 0.0

        return {
            "llm_judge_used": True,
            "llm_judge_raw_response": self._llm_judge_prefilled_response(raw_suffix),
            "llm_judge_is_equivalent": is_equivalent,
            "llm_judge_partial_credit": partial_credit,
            "llm_judge_accuracy": accuracy,
            "llm_judge_error": None,
        }

    def _format_llm_judge_prompt(self, question: str, reference: str, prediction: str) -> str:
        return (
            str(self.llm_judge_prompt_template)
            .replace("[QUESTION]", question)
            .replace("[REFERENCE]", reference)
            .replace("[PREDICTION]", prediction)
        )

    def _encode_llm_judge_prompt(self, judge_prompt: str) -> list[int]:
        prefill = getattr(self, "llm_judge_assistant_prefill", "{") or ""
        text = (
            f"<|im_start|>system\n{_LLM_JUDGE_SYSTEM_PROMPT}<|im_end|>\n"
            f"<|im_start|>user\n{judge_prompt}<|im_end|>\n"
            f"<|im_start|>assistant\n{prefill}"
        )
        return self.tokenizer.encode(text, add_special_tokens=False)

    def _parse_llm_judge_response(self, raw_suffix: str) -> dict[str, Any]:
        raw_response = self._llm_judge_prefilled_response(raw_suffix)
        start = raw_response.find("{")
        end = raw_response.rfind("}")
        if start < 0 or end < start:
            raise ValueError(f"judge response did not contain a JSON object: {raw_response!r}")
        parsed = json.loads(raw_response[start : end + 1])
        if not isinstance(parsed, dict):
            raise ValueError("judge response JSON was not an object")
        return parsed

    def _llm_judge_prefilled_response(self, raw_suffix: str) -> str:
        prefill = getattr(self, "llm_judge_assistant_prefill", "{") or ""
        suffix = str(raw_suffix or "").strip()
        if prefill and suffix.startswith(prefill):
            return suffix
        return f"{prefill}{suffix}"

    @staticmethod
    def _extract_llm_judge_question(agent_data: AgentData, extra_info: Any) -> str | None:
        if isinstance(extra_info, dict):
            for key in ("question", "problem", "prompt", "query"):
                value = extra_info.get(key)
                if value:
                    return ToolAgentLoop._clean_llm_judge_question(value)

        for message in agent_data.messages:
            if message.get("role") == "user":
                text = ToolAgentLoop._message_content_to_text(message.get("content"))
                if text:
                    return ToolAgentLoop._clean_llm_judge_question(text)
        return None

    @staticmethod
    def _message_content_to_text(content: Any) -> str:
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict):
                    if "text" in item:
                        parts.append(str(item["text"]))
                    elif item.get("type") in {"image", "video"} or "image" in item or "image_url" in item:
                        continue
                elif isinstance(item, str):
                    parts.append(item)
            return "\n".join(parts)
        return str(content) if content is not None else ""

    @staticmethod
    def _clean_llm_judge_question(value: Any) -> str:
        text = str(value)
        for placeholder in _VISION_PLACEHOLDER_REPLACEMENTS:
            text = text.replace(placeholder, " ")
        return " ".join(text.split())

    def _merge_llm_judge_accuracy(self, grade: dict[str, Any], judge_result: dict[str, Any]) -> dict[str, Any]:
        merged = dict(grade)
        accuracy = float(judge_result.get("llm_judge_accuracy", 0.0) or 0.0)
        merged.update(judge_result)
        merged["accuracy"] = accuracy
        merged["score"] = self._compose_score_from_grade(merged, accuracy)
        return merged

    def _compose_score_from_grade(self, grade: dict[str, Any], accuracy: float) -> float:
        formatting = float(grade.get("format", 0.0) or 0.0)
        format_score = float(getattr(self, "llm_judge_format_score", 0.5))
        tool_efficiency_score_weight = float(getattr(self, "llm_judge_tool_efficiency_score_weight", 0.0))
        tool_efficiency = float(grade.get("tool_efficiency", 0.0) or 0.0)
        tool_call_bonus = float(grade.get("tool_call_bonus", 0.0) or 0.0)
        return (
            (1.0 - format_score - tool_efficiency_score_weight) * float(accuracy)
            + format_score * formatting
            + tool_efficiency_score_weight * tool_efficiency
            + tool_call_bonus
        )

    @staticmethod
    def _grade_accuracy(grade: Any) -> float:
        if isinstance(grade, dict):
            try:
                return float(grade.get("accuracy", 0.0) or 0.0)
            except (TypeError, ValueError):
                return 0.0
        try:
            return float(grade)
        except (TypeError, ValueError):
            return 0.0

    def _incorrect_answer_retry_skipped_reason(
        self,
        agent_data: AgentData,
        snapshot: dict[str, Any],
        retry_count: int,
    ) -> str | None:
        max_retries = self.conditional_answer_retry_max_retries
        if max_retries is not None and retry_count >= int(max_retries):
            return "max_retries"
        if self.max_assistant_turns and snapshot["assistant_turns"] + 2 > self.max_assistant_turns:
            return "max_assistant_turns"
        if self.max_user_turns and agent_data.user_turns >= self.max_user_turns:
            return "max_user_turns"
        if snapshot["response_mask_len"] + self.final_answer_token_margin >= self.response_length:
            return "response_length"
        if self._latest_user_or_tool_message_index(agent_data) is None:
            return "missing_previous_user_or_tool_message"
        return None

    def _latest_user_or_tool_message_index(self, agent_data: AgentData) -> int | None:
        for index in range(len(agent_data.messages) - 1, -1, -1):
            if agent_data.messages[index].get("role") in {"user", "tool"}:
                return index
        return None

    def _record_last_user_tool_prompt_block(
        self,
        agent_data: AgentData,
        prompt_start: int,
        prompt_end: int,
        response_mask_start: int,
        response_mask_end: int,
        response_logprobs_start: int,
        response_logprobs_end: int,
        block_start_index: int,
        block_images: Any,
        block_videos: Any,
    ) -> None:
        message_index = self._latest_user_or_tool_message_index(agent_data)
        if message_index is None:
            return
        agent_data.last_user_tool_prompt_start = prompt_start
        agent_data.last_user_tool_prompt_end = prompt_end
        agent_data.last_user_tool_response_mask_start = response_mask_start
        agent_data.last_user_tool_response_mask_end = response_mask_end
        agent_data.last_user_tool_response_logprobs_start = response_logprobs_start
        agent_data.last_user_tool_response_logprobs_end = response_logprobs_end
        agent_data.last_user_tool_message_index = message_index
        agent_data.last_user_tool_block_start_index = block_start_index
        agent_data.last_user_tool_block_images = block_images
        agent_data.last_user_tool_block_videos = block_videos

    async def _append_conditional_retry_suffix_to_previous_message(self, agent_data: AgentData, suffix: str) -> None:
        message_index = self._latest_user_or_tool_message_index(agent_data)
        if message_index is None:
            raise RuntimeError("conditional answer retry could not find a previous user/tool message")

        agent_data.messages[message_index] = self._append_suffix_to_tool_message(agent_data.messages[message_index], suffix)
        prompt_start = agent_data.last_user_tool_prompt_start
        prompt_end = agent_data.last_user_tool_prompt_end
        mask_start = agent_data.last_user_tool_response_mask_start
        mask_end = agent_data.last_user_tool_response_mask_end
        logprobs_start = agent_data.last_user_tool_response_logprobs_start
        logprobs_end = agent_data.last_user_tool_response_logprobs_end
        block_start = agent_data.last_user_tool_block_start_index

        if (
            prompt_start is None
            or prompt_end is None
            or mask_start is None
            or mask_end is None
            or logprobs_start is None
            or logprobs_end is None
            or block_start is None
            or message_index < block_start
        ):
            prompt_ids = await self.apply_chat_template(
                agent_data.messages,
                tools=self.tool_schemas,
                images=agent_data.image_data,
                videos=agent_data.video_data,
            )
            agent_data.prompt_ids = prompt_ids
            agent_data.response_mask = []
            agent_data.response_logprobs = []
            self._record_last_user_tool_prompt_block(
                agent_data=agent_data,
                prompt_start=0,
                prompt_end=len(prompt_ids),
                response_mask_start=0,
                response_mask_end=0,
                response_logprobs_start=0,
                response_logprobs_end=0,
                block_start_index=0,
                block_images=agent_data.image_data,
                block_videos=agent_data.video_data,
            )
            return

        if prompt_start == 0:
            replacement_ids = await self.apply_chat_template(
                agent_data.messages,
                tools=self.tool_schemas,
                images=agent_data.image_data,
                videos=agent_data.video_data,
            )
        else:
            block_messages = agent_data.messages[block_start : message_index + 1]
            replacement_ids = await self.apply_chat_template(
                block_messages,
                images=agent_data.last_user_tool_block_images,
                videos=agent_data.last_user_tool_block_videos,
                remove_system_prompt=True,
            )
        old_len = prompt_end - prompt_start
        delta = len(replacement_ids) - old_len
        agent_data.prompt_ids = agent_data.prompt_ids[:prompt_start] + replacement_ids + agent_data.prompt_ids[prompt_end:]
        if mask_end > mask_start:
            agent_data.response_mask = (
                agent_data.response_mask[:mask_start]
                + [0] * len(replacement_ids)
                + agent_data.response_mask[mask_end:]
            )
        if agent_data.response_logprobs and logprobs_end > logprobs_start:
            agent_data.response_logprobs = (
                agent_data.response_logprobs[:logprobs_start]
                + [0.0] * len(replacement_ids)
                + agent_data.response_logprobs[logprobs_end:]
            )
        agent_data.last_user_tool_prompt_end = prompt_end + delta
        agent_data.last_user_tool_response_mask_end = mask_start + len(replacement_ids)
        agent_data.last_user_tool_response_logprobs_end = logprobs_start + len(replacement_ids)

    def _log_incorrect_answer_retry(
        self,
        agent_data: AgentData,
        grade: dict[str, Any],
        retry_count: int,
        suffix: str,
        removed_assistant_turn: int,
        removed_assistant_text: str,
    ) -> None:
        if not self._debug_enabled_for_agent(agent_data):
            return
        event = {
            "event": "incorrect_answer_retry",
            "request_id": agent_data.request_id,
            "retry_count": retry_count,
            "removed_assistant_turn": removed_assistant_turn,
            "removed_assistant_text": removed_assistant_text,
            "grade": grade,
            "suffix": suffix,
            "force_next_tool_call": bool(self.conditional_answer_retry_force_next_tool_call),
        }
        agent_data.debug_events.append(event)
        self._debug_append_event(event)

    def _log_incorrect_answer_retry_skipped(
        self,
        agent_data: AgentData,
        grade: dict[str, Any],
        retry_count: int,
        reason: str,
    ) -> None:
        if not self._debug_enabled_for_agent(agent_data):
            return
        event = {
            "event": "incorrect_answer_retry_skipped",
            "request_id": agent_data.request_id,
            "retry_count": retry_count,
            "reason": reason,
            "grade": grade,
        }
        agent_data.debug_events.append(event)
        self._debug_append_event(event)

    def _mark_forced_structure_error(self, agent_data: AgentData, reason: str) -> None:
        agent_data.extra_fields.update(
            {
                "forced_structure_error": True,
                "forced_structure_error_reason": reason,
                "forced_structure_error_assistant_turn": agent_data.assistant_turns,
            }
        )
        if self._debug_enabled_for_agent(agent_data):
            event = {
                "event": "forced_structure_error",
                "request_id": agent_data.request_id,
                "assistant_turn": agent_data.assistant_turns,
                "reason": reason,
            }
            agent_data.debug_events.append(event)
            self._debug_append_event(event)

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute

        tasks = []
        tool_call_names = []
        active_tool_calls = agent_data.tool_calls[: self.max_parallel_calls]
        for tool_index, tool_call in enumerate(active_tool_calls):
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data, tool_index=tool_index))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        for tool_call, (tool_response, tool_reward, tool_info) in zip(active_tool_calls, responses, strict=False):
            if not tool_response.terminate:
                continue
            if self._debug_enabled_for_agent(agent_data):
                event = {
                    "event": "tool_response",
                    "request_id": agent_data.request_id,
                    "assistant_turn": agent_data.assistant_turns,
                    "tool_name": tool_call.name,
                    "tool_arguments": tool_call.arguments,
                    "tool_response_text": tool_response.text,
                    "tool_reward": tool_reward,
                    "tool_info": tool_info,
                    "terminal": True,
                }
                agent_data.debug_events.append(event)
                self._debug_append_event(event)

            terminal_text = tool_response.text or "Error processing tool call."
            agent_data.messages.append({"role": "tool", "content": terminal_text})
            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)
            agent_data.extra_fields.update(
                {
                    "terminated_by_tool_error": True,
                    "terminal_tool_name": tool_call.name,
                    "terminal_tool_error": terminal_text,
                    "terminal_reason": tool_response.terminal_reason,
                }
            )
            response_ids = await self.apply_chat_template(
                [{"role": "tool", "content": terminal_text}],
                images=None,
                videos=None,
                remove_system_prompt=True,
            )
            agent_data.prompt_ids += response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)
            agent_data.user_turns += 1
            return AgentState.TERMINATED

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        pending_crop_refs: list[dict[str, Any]] = []
        pending_tool_rewards: list[float] = []
        for tool_call, (tool_response, tool_reward, tool_info) in zip(active_tool_calls, responses, strict=False):
            response_suffix = self._tool_response_turn_suffix(agent_data.assistant_turns)
            if self._debug_enabled_for_agent(agent_data):
                event = {
                    "event": "tool_response",
                    "request_id": agent_data.request_id,
                    "assistant_turn": agent_data.assistant_turns,
                    "tool_name": tool_call.name,
                    "tool_arguments": tool_call.arguments,
                    "tool_response_text": self._append_text_suffix(tool_response.text or "", response_suffix),
                    "tool_reward": tool_reward,
                    "tool_info": tool_info,
                }
                agent_data.debug_events.append(event)
                self._debug_append_event(event)
            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.extend([{"type": "image"} for _ in tool_response.image])
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}

            message = self._append_suffix_to_tool_message(message, response_suffix)
            add_messages.append(message)

            crop_refs = list(tool_response.crop_image_refs or [])

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img_idx, img in enumerate(tool_response.image):
                        if img is not None:  # Add a check to ensure the image is not None
                            crop_ref = dict(crop_refs[img_idx]) if img_idx < len(crop_refs) else {}
                            if crop_ref:
                                crop_ref["image_index"] = (
                                    len(agent_data.image_data or []) + len(new_images_this_turn)
                                )
                                crop_ref["order"] = len(agent_data.crop_image_refs) + len(pending_crop_refs)
                                crop_ref["assistant_turn"] = agent_data.assistant_turns
                                crop_ref["tool_name"] = tool_call.name
                                pending_crop_refs.append(crop_ref)
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        if crop_refs:
                            crop_ref = dict(crop_refs[0])
                            crop_ref["image_index"] = len(agent_data.image_data or []) + len(new_images_this_turn)
                            crop_ref["order"] = len(agent_data.crop_image_refs) + len(pending_crop_refs)
                            crop_ref["assistant_turn"] = agent_data.assistant_turns
                            crop_ref["tool_name"] = tool_call.name
                            pending_crop_refs.append(crop_ref)
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                pending_tool_rewards.append(tool_reward)

        if self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            if self._debug_enabled_for_agent(agent_data):
                event = {
                    "event": "tool_response_dropped_over_limit",
                    "request_id": agent_data.request_id,
                    "assistant_turn": agent_data.assistant_turns,
                    "dropped_tool_messages": len(add_messages),
                    "dropped_images": len(new_images_this_turn),
                    "dropped_crop_refs": len(pending_crop_refs),
                    "current_response_tokens": len(agent_data.response_mask),
                    "tool_response_tokens": len(response_ids),
                    "max_response_tokens": self.response_length,
                }
                agent_data.debug_events.append(event)
                self._debug_append_event(event)
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        agent_data.messages.extend(add_messages)
        agent_data.crop_image_refs.extend(pending_crop_refs)
        agent_data.tool_rewards.extend(pending_tool_rewards)

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        prompt_start = len(agent_data.prompt_ids)
        mask_start = len(agent_data.response_mask)
        logprobs_start = len(agent_data.response_logprobs)
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        self._record_last_user_tool_prompt_block(
            agent_data=agent_data,
            prompt_start=prompt_start,
            prompt_end=len(agent_data.prompt_ids),
            response_mask_start=mask_start,
            response_mask_end=len(agent_data.response_mask),
            response_logprobs_start=logprobs_start,
            response_logprobs_end=len(agent_data.response_logprobs),
            block_start_index=len(agent_data.messages) - len(add_messages),
            block_images=images,
            block_videos=videos,
        )
        return AgentState.GENERATING

    async def _handle_interacting_state(self, agent_data: AgentData) -> AgentState:
        """Handle the interacting state: get user input from interaction."""
        (
            should_terminate_sequence,
            interaction_responses,
            reward,
            metrics,
        ) = await agent_data.interaction.generate_response(
            agent_data.request_id, agent_data.messages, **agent_data.interaction_kwargs
        )
        agent_data.user_turns += 1

        add_messages: list[dict[str, Any]] = [{"role": "user", "content": interaction_responses}]
        agent_data.messages.extend(add_messages)

        if reward is not None:
            agent_data.turn_scores.append(reward)

        # Update prompt with user responses (similar to _handle_processing_tools_state)
        response_ids = await self.apply_chat_template(
            add_messages,
            remove_system_prompt=True,
        )

        # Update prompt_ids and response_mask
        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)

        # double check prompt
        # Check termination condition
        if should_terminate_sequence:
            return AgentState.TERMINATED
        else:
            return AgentState.GENERATING

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData, tool_index: int = 0
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response."""
        tool, instance_id = None, None
        res = {}
        tool_name = tool_call.name
        try:
            # TODO: append malformed tool_call to the prompt: invalid function name or arguments
            tool_args = json.loads(tool_call.arguments)
            if self._debug_enabled_for_agent(agent_data):
                event = {
                    "event": "tool_call",
                    "request_id": agent_data.request_id,
                    "assistant_turn": agent_data.assistant_turns,
                    "tool_index": tool_index,
                    "tool_name": tool_name,
                    "tool_arguments": tool_args,
                }
                agent_data.debug_events.append(event)
                self._debug_append_event(event)
            tool = self.tools[tool_name]
            kwargs = tools_kwargs.get(tool_name, {})
            instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
            tool_execution_response, tool_reward, res = await tool.execute(
                instance_id, tool_args, agent_data=agent_data
            )
            if self._debug_enabled_for_agent(agent_data):
                res.setdefault("debug_context", agent_data.extra_fields.get("debug_trajectory", {}))
            if self._debug_enabled_for_agent(agent_data):
                self._debug_save_tool_images(
                    agent_data=agent_data,
                    tool_response=tool_execution_response,
                    tool_info=res,
                    tool_index=tool_index,
                )
        except Exception as e:
            logger.warning(f"Error when executing tool: {e}")
            res = {"success": False, "error": str(e)}
            if tool_name == "image_zoom_in_tool":
                res.update({"terminal": True, "terminal_reason": "image_zoom_in_tool_error"})
                return (
                    ToolResponse(
                        text=f"Error processing image zoom-in: {e}",
                        terminate=True,
                        terminal_reason="image_zoom_in_tool_error",
                    ),
                    -0.05,
                    res,
                )
            return (
                ToolResponse(
                    text=f"Error when executing tool: {e}",
                ),
                0.0,
                {},
            )
        finally:
            if tool and instance_id:
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            elif self.tool_response_truncate_side == "right":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value
        if tool_execution_response.crop_image_refs:
            tool_response_kwargs["crop_image_refs"] = tool_execution_response.crop_image_refs
        if tool_execution_response.terminate:
            tool_response_kwargs["terminate"] = tool_execution_response.terminate
            tool_response_kwargs["terminal_reason"] = tool_execution_response.terminal_reason

        return ToolResponse(**tool_response_kwargs), tool_reward, res

    def _debug_dir(self) -> str | None:
        return self.debug_options.get("dir")

    def _debug_enabled_for_trajectory(self, trajectory: dict[str, Any]) -> bool:
        return bool(self.debug_options.get("enabled") and should_log_step(self.config, trajectory.get("step")))

    @staticmethod
    def _debug_enabled_for_agent(agent_data: AgentData) -> bool:
        return bool(getattr(agent_data, "debug_logging_enabled", False))

    def _debug_max_text_chars(self) -> int | None:
        return self.debug_options.get("max_text_chars")

    def _debug_append_event(self, event: dict[str, Any]) -> None:
        debug_dir = self._debug_dir()
        if not debug_dir:
            return
        append_jsonl(path_join(debug_dir, "agent_events.jsonl"), event, max_text_chars=self._debug_max_text_chars())

    def _debug_base_record(self, agent_data: AgentData, kwargs: dict[str, Any], trajectory: dict[str, Any]) -> dict:
        reward_model = kwargs.get("reward_model") or {}
        extra_info = kwargs.get("extra_info") or {}
        return {
            "request_id": agent_data.request_id,
            "trajectory": trajectory,
            "uid": kwargs.get("uid"),
            "index": kwargs.get("index"),
            "data_source": kwargs.get("data_source"),
            "ground_truth": reward_model.get("ground_truth") if isinstance(reward_model, dict) else None,
            "reward_model": reward_model,
            "extra_info": extra_info,
            "image_references": self._debug_extract_image_references(kwargs["raw_prompt"], extra_info),
            "question": self._debug_extract_question(kwargs["raw_prompt"]),
        }

    async def _debug_log_sample_start(
        self,
        agent_data: AgentData,
        kwargs: dict[str, Any],
        trajectory: dict[str, Any],
        sampling_params: dict[str, Any],
    ) -> None:
        debug_dir = self._debug_dir()
        if not debug_dir:
            return
        source_image_paths = []
        if self.debug_options.get("save_source_images") and agent_data.image_data:
            images = agent_data.image_data if isinstance(agent_data.image_data, list) else [agent_data.image_data]
            step = trajectory.get("step", "unknown")
            for image_index, image in enumerate(images):
                path = path_join(
                    debug_dir,
                    "source_images",
                    f"step_{step}",
                    f"request_{agent_data.request_id}_image_{image_index}.png",
                )
                saved_path = save_image(image, path)
                if saved_path:
                    source_image_paths.append(saved_path)

        record = {
            "event": "sample_start",
            **self._debug_base_record(agent_data, kwargs, trajectory),
            "raw_prompt": kwargs["raw_prompt"],
            "system_prompt": self._debug_extract_system_prompt(kwargs["raw_prompt"]),
            "tool_prompt": self.tool_schemas,
            "tool_config_path": self.rollout_config.multi_turn.tool_config_path,
            "sampling_params": sampling_params,
            "source_image_paths": source_image_paths,
        }
        agent_data.debug_events.append(record)
        append_jsonl(path_join(debug_dir, "sample_traces.jsonl"), record, max_text_chars=self._debug_max_text_chars())
        self._debug_append_event(record)

    async def _debug_log_sample_final(
        self,
        output: AgentLoopOutput,
        agent_data: AgentData,
        kwargs: dict[str, Any],
        trajectory: dict[str, Any],
    ) -> None:
        debug_dir = self._debug_dir()
        if not debug_dir:
            return
        final_response = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(output.response_ids, skip_special_tokens=True)
        )
        final_response = apply_response_display_prefix(final_response, self.response_display_prefix)
        record = {
            "event": "sample_final",
            **self._debug_base_record(agent_data, kwargs, trajectory),
            "raw_prompt": kwargs["raw_prompt"],
            "messages": agent_data.messages,
            "final_response": final_response,
            "num_turns": output.num_turns,
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
            "debug_events": agent_data.debug_events,
            "metrics": output.metrics.model_dump(),
        }
        output.extra_fields["comprehensive_debug"] = to_jsonable(
            record, max_text_chars=self._debug_max_text_chars()
        )
        append_jsonl(path_join(debug_dir, "sample_traces.jsonl"), record, max_text_chars=self._debug_max_text_chars())
        self._debug_append_event({"event": "sample_final", **record})

    def _debug_save_tool_images(
        self,
        agent_data: AgentData,
        tool_response: ToolResponse,
        tool_info: dict[str, Any],
        tool_index: int,
    ) -> None:
        debug_dir = self._debug_dir()
        if not debug_dir or not self.debug_options.get("save_cropped_images"):
            return
        images = tool_response.image
        if images is None:
            return
        if not isinstance(images, list):
            images = [images]
        step = "unknown"
        context = tool_info.get("debug_context") if isinstance(tool_info, dict) else None
        if isinstance(context, dict):
            step = context.get("step", step)
        for image_index, image in enumerate(images):
            path = path_join(
                debug_dir,
                "crops",
                f"step_{step}",
                f"request_{agent_data.request_id}_turn_{agent_data.assistant_turns}_tool_{tool_index}_{image_index}.png",
            )
            saved_path = save_image(image, path)
            if saved_path:
                tool_info.setdefault("saved_images", []).append(saved_path)

    @staticmethod
    def _debug_extract_system_prompt(raw_prompt: list[dict[str, Any]]) -> Any:
        for message in raw_prompt:
            if message.get("role") == "system":
                return message.get("content")
        return None

    @staticmethod
    def _debug_extract_question(raw_prompt: list[dict[str, Any]]) -> str | None:
        user_texts = []
        for message in raw_prompt:
            if message.get("role") != "user":
                continue
            content = message.get("content")
            if isinstance(content, str):
                user_texts.append(content)
            elif isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and "text" in item:
                        user_texts.append(str(item["text"]))
        return "\n".join(user_texts) if user_texts else None

    def _debug_extract_image_references(self, raw_prompt: list[dict[str, Any]], extra_info: Any) -> list[Any]:
        references = []
        if isinstance(extra_info, dict):
            for key in (self.data_config.get("image_key", "images"), "images", "image"):
                if key in extra_info:
                    references.append(extra_info[key])
        for message in raw_prompt:
            content = message.get("content")
            if not isinstance(content, list):
                continue
            for item in content:
                if not isinstance(item, dict):
                    continue
                for key in ("image", "image_url", "path"):
                    if key in item:
                        references.append(item[key])
        return references

    def _initialize_interactions(self, interaction_config_file):
        """Initialize interactions from configuration.
        Returns:
            dict[str, BaseInteraction]: A dictionary mapping interaction names to interaction instances.
        """
        if interaction_config_file is None:
            return {}

        interaction_map = initialize_interactions_from_config(interaction_config_file)
        return interaction_map
