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
import math
import os
import random
import re
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
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
from verl.utils.forced_action_tags import build_forced_action_extra_args
from verl.utils.profiler import simple_timer
from verl.utils.response_display import apply_response_display_prefix, get_response_display_prefix
from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
    _contains_reserved_tag,
    _extract_reward_type,
    _parse_multiturn_tool_calling_output,
    _parse_tool_call_payload,
    extract_answer_for_accuracy,
    tool_repeat_overlap_call_diagnostics,
)
from verl.utils.rollout_trace import rollout_trace_op
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
        self.retry_sft_mask: list[int] = []
        self.retry_critic_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0
        self.effective_max_assistant_turns: int | None = None

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}
        self.branch_outputs: list[AgentLoopOutput] = []
        self.debug_events: list[dict[str, Any]] = []
        self.crop_image_refs: list[dict[str, Any]] = []
        self.force_next_tool_call = False
        self.retry_sft_candidates: list[dict[str, Any]] = []
        self.executed_tool_calls: list[dict[str, Any]] = []
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
        self.max_assistant_tokens_per_turn = getattr(
            multi_turn_config, "max_assistant_tokens_per_turn", None
        )
        if self.max_assistant_tokens_per_turn is not None:
            self.max_assistant_tokens_per_turn = int(self.max_assistant_tokens_per_turn)
            if self.max_assistant_tokens_per_turn <= 0:
                raise ValueError("multi_turn.max_assistant_tokens_per_turn must be positive or null")
        self.max_parallel_calls = multi_turn_config.max_parallel_calls
        self.max_tool_response_length = multi_turn_config.max_tool_response_length
        # Per-turn generation cap (ported from tmp/training-files-main): clamp each turn's max_tokens.
        self.max_assistant_response_length = getattr(multi_turn_config, "max_assistant_response_length", None)
        self.tool_response_truncate_side = multi_turn_config.tool_response_truncate_side
        self.force_tool_calls_until_final_turn = getattr(multi_turn_config, "force_tool_calls_until_final_turn", False)
        self.force_answer_on_final_turn = getattr(multi_turn_config, "force_answer_on_final_turn", False)
        self.force_answer_on_final_turn_validation = getattr(
            multi_turn_config, "force_answer_on_final_turn_validation", None
        )
        self.randomized_max_assistant_turns_enabled = bool(
            getattr(multi_turn_config, "randomized_max_assistant_turns_enabled", False)
        )
        self.randomized_max_assistant_turns_early_window = getattr(
            multi_turn_config, "randomized_max_assistant_turns_early_window", 0
        )
        self.randomized_max_assistant_turns_per_turn_probability = getattr(
            multi_turn_config, "randomized_max_assistant_turns_per_turn_probability", 0.0
        )
        self.randomized_max_assistant_turns_seed = getattr(
            multi_turn_config, "randomized_max_assistant_turns_seed", 42
        )
        self.randomized_max_assistant_turns_enable_validation = bool(
            getattr(multi_turn_config, "randomized_max_assistant_turns_enable_validation", False)
        )
        self._validate_randomized_max_assistant_turns_config()
        self.final_answer_token_margin = getattr(multi_turn_config, "final_answer_token_margin", 2048)
        self.force_final_answer_think_prefill = getattr(multi_turn_config, "force_final_answer_think_prefill", "")
        self.force_final_answer_think_prefill_trainable = bool(
            getattr(multi_turn_config, "force_final_answer_think_prefill_trainable", False)
        )
        self.force_final_answer_prefix = getattr(multi_turn_config, "force_final_answer_prefix", "")
        self.tool_response_turn_suffix_enabled = getattr(multi_turn_config, "tool_response_turn_suffix_enabled", False)
        self.tool_response_explore_turns = getattr(multi_turn_config, "tool_response_explore_turns", 10)
        self.tool_response_explore_suffix = getattr(multi_turn_config, "tool_response_explore_suffix", "")
        self.tool_response_final_suffix = getattr(multi_turn_config, "tool_response_final_suffix", "")
        self.conditional_answer_retry_enabled = getattr(
            multi_turn_config, "conditional_answer_retry_enabled", False
        )
        self.conditional_answer_retry_enable_validation = bool(
            getattr(multi_turn_config, "conditional_answer_retry_enable_validation", True)
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
        self.conditional_answer_retry_assistant_prefill_trainable = bool(
            getattr(multi_turn_config, "conditional_answer_retry_assistant_prefill_trainable", False)
        )
        self.conditional_answer_retry_teacher_only_prefill = bool(
            getattr(multi_turn_config, "conditional_answer_retry_teacher_only_prefill", False)
        )
        self.conditional_answer_retry_teacher_suffix_gspo_enabled = bool(
            getattr(multi_turn_config, "conditional_answer_retry_teacher_suffix_gspo_enabled", False)
        )
        self.conditional_answer_retry_sft_enabled = bool(
            getattr(multi_turn_config, "conditional_answer_retry_sft_enabled", False)
        )
        self.conditional_answer_retry_sft_target_selection = str(
            getattr(multi_turn_config, "conditional_answer_retry_sft_target_selection", "all")
        )
        if self.conditional_answer_retry_sft_target_selection not in {"all", "first", "last"}:
            raise ValueError(
                "conditional_answer_retry_sft_target_selection must be one of 'all', 'first', or 'last'"
            )
        if self.conditional_answer_retry_teacher_only_prefill:
            if not self.conditional_answer_retry_assistant_prefill:
                raise ValueError("teacher-only retry prefill requires a nonempty assistant prefill")
            if self.conditional_answer_retry_assistant_prefill_trainable:
                raise ValueError("teacher-only retry prefill is incompatible with trainable injected prefills")
        if self.conditional_answer_retry_sft_enabled and not self.conditional_answer_retry_teacher_only_prefill:
            raise ValueError("retry SFT requires conditional_answer_retry_teacher_only_prefill=true")
        if self.conditional_answer_retry_teacher_suffix_gspo_enabled:
            if not self.conditional_answer_retry_teacher_only_prefill:
                raise ValueError(
                    "retry teacher-suffix GSPO requires conditional_answer_retry_teacher_only_prefill=true"
                )
            if self.conditional_answer_retry_sft_enabled:
                raise ValueError("retry teacher-suffix GSPO is incompatible with retry SFT")
        self.conditional_answer_retry_correct_accuracy_threshold = getattr(
            multi_turn_config, "conditional_answer_retry_correct_accuracy_threshold", 0.0
        )
        self.conditional_answer_retry_max_retries = getattr(
            multi_turn_config, "conditional_answer_retry_max_retries", None
        )
        self.conditional_answer_retry_branch_enabled = getattr(
            multi_turn_config, "conditional_answer_retry_branch_enabled", False
        )
        self.conditional_answer_retry_branch_enable_validation = getattr(
            multi_turn_config, "conditional_answer_retry_branch_enable_validation", False
        )
        self.conditional_answer_retry_branch_mask_mode = getattr(
            multi_turn_config, "conditional_answer_retry_branch_mask_mode", "divergence_only"
        )
        self.conditional_answer_retry_branch_reward_components = dict(
            getattr(multi_turn_config, "conditional_answer_retry_branch_reward_components", {}) or {}
        )
        self.conditional_answer_retry_branch_use_graded_rewards = bool(
            getattr(multi_turn_config, "conditional_answer_retry_branch_use_graded_rewards", False)
        )
        self.conditional_answer_retry_branch_baseline_weight_key = getattr(
            multi_turn_config, "conditional_answer_retry_branch_baseline_weight_key", "gdpo_baseline_weight"
        )
        self.conditional_answer_retry_critic_enabled = bool(
            getattr(multi_turn_config, "conditional_answer_retry_critic_enabled", False)
        )
        self.conditional_answer_retry_critic_oracle_supervision_enabled = bool(
            getattr(multi_turn_config, "conditional_answer_retry_critic_oracle_supervision_enabled", True)
        )
        self.conditional_answer_retry_critic_enable_validation = bool(
            getattr(multi_turn_config, "conditional_answer_retry_critic_enable_validation", False)
        )
        self.conditional_answer_retry_critic_question = str(
            getattr(multi_turn_config, "conditional_answer_retry_critic_question", "")
        )
        self.conditional_answer_retry_critic_assistant_prefill = str(
            getattr(multi_turn_config, "conditional_answer_retry_critic_assistant_prefill", "")
        )
        self.conditional_answer_retry_critic_accept_token = str(
            getattr(multi_turn_config, "conditional_answer_retry_critic_accept_token", "A")
        )
        self.conditional_answer_retry_critic_retry_token = str(
            getattr(multi_turn_config, "conditional_answer_retry_critic_retry_token", "R")
        )
        self.conditional_answer_retry_critic_decision_threshold = float(
            getattr(multi_turn_config, "conditional_answer_retry_critic_decision_threshold", 0.5)
        )
        self._retry_critic_accept_token_id = None
        self._retry_critic_retry_token_id = None
        if self.conditional_answer_retry_critic_enabled:
            accept_ids = self.tokenizer.encode(
                self.conditional_answer_retry_critic_accept_token, add_special_tokens=False
            )
            retry_ids = self.tokenizer.encode(
                self.conditional_answer_retry_critic_retry_token, add_special_tokens=False
            )
            if len(accept_ids) != 1 or len(retry_ids) != 1 or accept_ids[0] == retry_ids[0]:
                raise ValueError(
                    "retry critic accept/retry labels must encode as distinct single tokens; "
                    f"got {accept_ids!r} and {retry_ids!r}"
                )
            if not self.conditional_answer_retry_critic_question.strip():
                raise ValueError("retry critic question must be nonempty")
            if not self.conditional_answer_retry_critic_assistant_prefill.strip():
                raise ValueError("retry critic assistant prefill must be nonempty")
            if not 0.0 <= self.conditional_answer_retry_critic_decision_threshold <= 1.0:
                raise ValueError("retry critic decision threshold must be in [0, 1]")
            self._retry_critic_accept_token_id = accept_ids[0]
            self._retry_critic_retry_token_id = retry_ids[0]
        self.llm_judge_enabled = bool(getattr(multi_turn_config, "llm_judge_enabled", False))
        self.conditional_answer_retry_reward_fn = (
            get_custom_reward_fn(self.config)
            if (
                self.conditional_answer_retry_enabled
                or self.llm_judge_enabled
                or (
                    self.conditional_answer_retry_critic_enabled
                    and self.conditional_answer_retry_critic_oracle_supervision_enabled
                )
            )
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
        self.allow_unboxed_answer_for_accuracy = bool(reward_kwargs.get("allow_unboxed_answer_for_accuracy", True))
        self.tool_repeat_overlap_iou_threshold = float(
            reward_kwargs.get("tool_repeat_overlap_iou_threshold", 0.9)
        )
        self.tool_repeat_overlap_small_crop_area_threshold = float(
            reward_kwargs.get("tool_repeat_overlap_small_crop_area_threshold", 0.0002)
        )
        self.tool_repeat_overlap_free_depth_refinements = int(
            reward_kwargs.get("tool_repeat_overlap_free_depth_refinements", 2)
        )
        self.tool_repeat_overlap_depth_containment_threshold = float(
            reward_kwargs.get("tool_repeat_overlap_depth_containment_threshold", 0.95)
        )
        if bool(reward_kwargs.get("assistant_turn_overlong_enabled", False)):
            reward_max_turns = reward_kwargs.get("assistant_turn_overlong_max_turns")
            if reward_max_turns != self.max_assistant_turns:
                raise ValueError(
                    "assistant_turn_overlong_max_turns must match multi_turn.max_assistant_turns when enabled, "
                    f"got {reward_max_turns!r} != {self.max_assistant_turns!r}."
                )
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
        agent_data.effective_max_assistant_turns = self._sample_effective_max_assistant_turns(debug_trajectory)
        if self.randomized_max_assistant_turns_enabled:
            agent_data.extra_fields.update(self._randomized_turn_cap_metadata(agent_data))
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
            multi_modal_data["images"] = list(agent_data.image_data)
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = list(agent_data.video_data)

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            retry_sft_mask=agent_data.retry_sft_mask[: self.response_length],
            retry_critic_mask=agent_data.retry_critic_mask[: self.response_length],
            multi_modal_data=multi_modal_data,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=agent_data.routed_experts,
            extra_fields={**TERMINAL_TOOL_EXTRA_FIELD_DEFAULTS, **agent_data.extra_fields},
        )
        if self.conditional_answer_retry_branch_enabled:
            output.extra_fields[self.conditional_answer_retry_branch_baseline_weight_key] = 1.0
            output.extra_fields["is_conditional_answer_retry_branch"] = False
        output.extra_fields.update(
            {
                "request_id": agent_data.request_id,
                "assistant_turns": agent_data.assistant_turns,
                "turn_scores": agent_data.turn_scores,
                "tool_rewards": agent_data.tool_rewards,
                **self._randomized_turn_cap_metadata(agent_data),
                CROP_IMAGE_REFS_KEY: agent_data.crop_image_refs,
                "__agent_loop_retry_sft_candidates": list(agent_data.retry_sft_candidates),
                "retry_sft_candidate_count": len(agent_data.retry_sft_candidates),
            }
        )
        final_grade = output.extra_fields.pop("__agent_loop_precritic_grade", None)
        agent_data.extra_fields.pop("__agent_loop_precritic_grade", None)
        if final_grade is None:
            final_grade = await self._maybe_score_final_with_llm_judge(agent_data, kwargs)
        if final_grade is not None:
            final_grade = {**final_grade, **self._retry_critic_validation_summary(agent_data)}
            output.reward_score = float(final_grade.get("score", 0.0) or 0.0)
            output.extra_fields["reward_extra_info"] = final_grade
        output.branch_outputs = agent_data.branch_outputs
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

    def _validate_randomized_max_assistant_turns_config(self) -> None:
        if not self.randomized_max_assistant_turns_enabled:
            return
        configured_cap = self.max_assistant_turns
        if isinstance(configured_cap, bool) or not isinstance(configured_cap, int) or configured_cap <= 0:
            raise ValueError(
                "multi_turn.max_assistant_turns must be a positive integer when randomized caps are enabled"
            )
        early_window = self.randomized_max_assistant_turns_early_window
        if isinstance(early_window, bool) or not isinstance(early_window, int):
            raise ValueError("randomized_max_assistant_turns_early_window must be an integer")
        if not 1 <= early_window < configured_cap:
            raise ValueError(
                "randomized_max_assistant_turns_early_window must be in "
                f"[1, {configured_cap - 1}], got {early_window!r}"
            )
        probability = self.randomized_max_assistant_turns_per_turn_probability
        if isinstance(probability, bool) or not isinstance(probability, int | float):
            raise ValueError("randomized_max_assistant_turns_per_turn_probability must be numeric")
        if not 0.0 <= float(probability) <= 1.0:
            raise ValueError(
                "randomized_max_assistant_turns_per_turn_probability must be in "
                f"[0, 1], got {probability!r}"
            )
        seed = self.randomized_max_assistant_turns_seed
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise ValueError("randomized_max_assistant_turns_seed must be an integer")

    def _sample_effective_max_assistant_turns(self, debug_trajectory: Optional[dict[str, Any]]) -> int | None:
        configured_cap = self.max_assistant_turns
        if not self.randomized_max_assistant_turns_enabled:
            return configured_cap
        debug_trajectory = debug_trajectory or {}
        if debug_trajectory.get("validate") and not self.randomized_max_assistant_turns_enable_validation:
            return configured_cap
        seed_material = json.dumps(
            [
                self.randomized_max_assistant_turns_seed,
                debug_trajectory.get("step"),
                debug_trajectory.get("sample_index"),
                debug_trajectory.get("rollout_n"),
            ],
            separators=(",", ":"),
            default=str,
        )
        rng = random.Random(seed_material)
        earliest_cap = configured_cap - self.randomized_max_assistant_turns_early_window
        probability = float(self.randomized_max_assistant_turns_per_turn_probability)
        for candidate_cap in range(earliest_cap, configured_cap):
            if rng.random() < probability:
                return candidate_cap
        return configured_cap

    def _effective_max_assistant_turns(self, agent_data: AgentData) -> int | None:
        effective_cap = getattr(agent_data, "effective_max_assistant_turns", None)
        return self.max_assistant_turns if effective_cap is None else effective_cap

    def _randomized_turn_cap_metadata(self, agent_data: AgentData) -> dict[str, Any]:
        if not getattr(self, "randomized_max_assistant_turns_enabled", False):
            return {}
        return {
            "configured_max_assistant_turns": self.max_assistant_turns,
            "effective_max_assistant_turns": self._effective_max_assistant_turns(agent_data),
        }

    def _must_answer_now(self, agent_data: AgentData) -> bool:
        """Whether the upcoming assistant turn must produce a final answer rather than a tool call.

        True on the final allowable assistant turn, when max user turns is reached, or when the
        remaining response budget is too small to safely fit another tool round plus a final answer.
        """
        return (
            (
                self._effective_max_assistant_turns(agent_data)
                and agent_data.assistant_turns + 1 >= self._effective_max_assistant_turns(agent_data)
            )
            or (self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
            or (len(agent_data.response_mask) + self.final_answer_token_margin >= self.response_length)
        )

    def _with_forced_action_extra_args(self, sampling_params: dict[str, Any], required_tag: str) -> dict[str, Any]:
        extra_args = dict(sampling_params.get("extra_args") or {})
        extra_args.update(
            build_forced_action_extra_args(
                self.tokenizer,
                required_tag,
                forced_suffix=self.force_final_answer_prefix if required_tag == "<answer>" else "",
            )
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

    def _conditional_answer_retry_active(self, rollout_kwargs: Optional[dict[str, Any]]) -> bool:
        if not getattr(self, "conditional_answer_retry_enabled", False):
            return False
        debug_trajectory = (rollout_kwargs or {}).get("_debug_trajectory", {}) or {}
        is_validation = bool(debug_trajectory.get("validate"))
        return not is_validation or getattr(self, "conditional_answer_retry_enable_validation", True)

    def _retry_critic_validation_active(self, rollout_kwargs: Optional[dict[str, Any]]) -> bool:
        debug_trajectory = (rollout_kwargs or {}).get("_debug_trajectory", {}) or {}
        return bool(
            getattr(self, "conditional_answer_retry_critic_enabled", False)
            and getattr(self, "conditional_answer_retry_critic_enable_validation", False)
            and debug_trajectory.get("validate")
        )

    def _force_answer_on_final_turn_active(self, rollout_kwargs: Optional[dict[str, Any]]) -> bool:
        debug_trajectory = (rollout_kwargs or {}).get("_debug_trajectory", {}) or {}
        if debug_trajectory.get("validate"):
            validation_override = getattr(self, "force_answer_on_final_turn_validation", None)
            if validation_override is not None:
                return bool(validation_override)
        return bool(getattr(self, "force_answer_on_final_turn", False))

    def _with_assistant_turn_token_cap(
        self,
        sampling_params: dict[str, Any],
        agent_data: AgentData,
    ) -> dict[str, Any]:
        turn_cap = getattr(self, "max_assistant_tokens_per_turn", None)
        if turn_cap is None:
            return sampling_params
        remaining_budget = self.response_length - len(agent_data.response_mask)
        if remaining_budget <= 0:
            raise RuntimeError("assistant generation requested after the response budget was exhausted")
        caps = [int(turn_cap), int(remaining_budget)]
        existing_cap = sampling_params.get("max_tokens")
        if existing_cap is not None:
            caps.append(int(existing_cap))
        return {**sampling_params, "max_tokens": min(caps)}

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
        teacher_only_prefill_ids: list[int] = []
        generation_snapshot = self._snapshot_generation_state(agent_data)

        if self.force_tool_calls_until_final_turn:
            forced_required_tag = "<answer>" if self._must_answer_now(agent_data) else "<tool_call>"
            sampling_params = self._with_forced_action_extra_args(sampling_params, forced_required_tag)
        elif self._force_answer_on_final_turn_active(rollout_kwargs) and self._must_answer_now(agent_data):
            forced_required_tag = "<answer>"
            sampling_params = self._with_forced_action_extra_args(sampling_params, forced_required_tag)
        elif agent_data.force_next_tool_call:
            forced_required_tag = "<tool_call>"
            sampling_params = self._with_forced_action_extra_args(sampling_params, forced_required_tag)
            if getattr(self, "conditional_answer_retry_teacher_only_prefill", False):
                teacher_only_prefill_ids = self.tokenizer.encode(
                    self.conditional_answer_retry_assistant_prefill,
                    add_special_tokens=False,
                )
                if not teacher_only_prefill_ids:
                    raise ValueError("teacher-only retry prefill encoded to an empty token sequence")
            else:
                assistant_prefill_ids = self._append_conditional_retry_assistant_prefill(agent_data)
            agent_data.force_next_tool_call = False
        if forced_required_tag == "<answer>":
            assistant_prefill_ids = self._append_force_final_answer_think_prefill(agent_data)
        sampling_params = self._with_assistant_turn_token_cap(sampling_params, agent_data)

        with simple_timer("generate_sequences", agent_data.metrics):
            generation_prompt_ids = agent_data.prompt_ids + teacher_only_prefill_ids
            # Per-turn response cap (ported from tmp/training-files-main): clamp this turn's max_tokens
            # to max_assistant_response_length so a single turn can't run out to the full response_length.
            # Copy first so the forced-action extra args set above are preserved.
            turn_sampling_params = sampling_params
            if self.max_assistant_response_length is not None:
                remaining_response_length = max(self.response_length - len(agent_data.response_mask), 0)
                turn_sampling_params = dict(sampling_params)
                turn_sampling_params["max_tokens"] = min(
                    int(self.max_assistant_response_length),
                    remaining_response_length,
                )
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=generation_prompt_ids,
                sampling_params=turn_sampling_params,
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
        response_start = len(agent_data.response_mask)
        agent_data.prompt_ids += agent_data.response_ids
        is_teacher_retry_turn = bool(teacher_only_prefill_ids)
        train_teacher_suffix_with_gspo = (
            is_teacher_retry_turn
            and bool(getattr(self, "conditional_answer_retry_teacher_suffix_gspo_enabled", False))
        )
        teacher_suffix_loss_mask = int(train_teacher_suffix_with_gspo or not is_teacher_retry_turn)
        agent_data.response_mask += [teacher_suffix_loss_mask] * len(agent_data.response_ids)
        agent_data.retry_sft_mask += [0] * len(agent_data.response_ids)
        agent_data.retry_critic_mask += [0] * len(agent_data.response_ids)
        if output.log_probs:
            if assistant_prefill_ids:
                agent_data.response_logprobs += [0.0] * len(assistant_prefill_ids)
            if is_teacher_retry_turn and not train_teacher_suffix_with_gspo:
                agent_data.response_logprobs += [0.0] * len(agent_data.response_ids)
            else:
                # Preserve the actual H+prefill behavior likelihood for diagnostics.
                # GSPO's old-policy anchor is recomputed separately under stored H.
                agent_data.response_logprobs += output.log_probs

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        if self._debug_enabled_for_agent(agent_data):
            assistant_message = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.decode(
                    assistant_prefill_ids + teacher_only_prefill_ids + agent_data.response_ids,
                    skip_special_tokens=True,
                ),
            )
            assistant_message = apply_response_display_prefix(assistant_message, self.response_display_prefix)
            event = {
                "event": "assistant_turn",
                "request_id": agent_data.request_id,
                "assistant_turn": agent_data.assistant_turns,
                "text": assistant_message,
                "num_tokens": len(agent_data.response_ids),
                "teacher_only_prefill": bool(teacher_only_prefill_ids),
            }
            if teacher_only_prefill_ids:
                event["teacher_prompt_text"] = self.tokenizer.decode(
                    generation_prompt_ids, skip_special_tokens=True
                )
                event["teacher_prompt_suffix"] = self.tokenizer.decode(
                    teacher_only_prefill_ids, skip_special_tokens=True
                )
                event["stored_student_turn"] = apply_response_display_prefix(
                    self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True),
                    self.response_display_prefix,
                )
            agent_data.debug_events.append(event)
            self._debug_append_event(event)

        # Extract tool calls before termination checks so forced final turns can be validated.
        tools = [tool.tool_schema for tool in self.tools.values()]
        _, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(agent_data.response_ids, tools)

        if is_teacher_retry_turn:
            self._record_retry_sft_candidate(
                agent_data=agent_data,
                response_start=response_start,
                response_end=len(agent_data.response_mask),
            )

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
            (
                self._conditional_answer_retry_active(rollout_kwargs)
                or self._retry_critic_validation_active(rollout_kwargs)
            )
            and not agent_data.tool_calls
            and rollout_kwargs is not None
        ):
            assistant_message = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
            )
            assistant_message = apply_response_display_prefix(assistant_message, self.response_display_prefix)
            if "<answer>" in assistant_message:
                if self._retry_critic_validation_active(rollout_kwargs):
                    retry_state = await self._maybe_retry_answer_with_critic(
                        agent_data=agent_data,
                        generation_snapshot=generation_snapshot,
                        assistant_message=assistant_message,
                        ignore_termination=ignore_termination,
                    )
                else:
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
        effective_max_assistant_turns = self._effective_max_assistant_turns(agent_data)
        if effective_max_assistant_turns and agent_data.assistant_turns >= effective_max_assistant_turns:
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
        return self._append_assistant_prefill(
            agent_data,
            prefill,
            trainable=self.conditional_answer_retry_assistant_prefill_trainable,
        )

    def _append_force_final_answer_think_prefill(self, agent_data: AgentData) -> list[int]:
        prefill = self.force_final_answer_think_prefill or ""
        return self._append_assistant_prefill(
            agent_data,
            prefill,
            trainable=self.force_final_answer_think_prefill_trainable,
        )

    def _append_assistant_prefill(
        self,
        agent_data: AgentData,
        prefill: str,
        *,
        trainable: bool,
    ) -> list[int]:
        if not prefill:
            return []
        prefill_ids = self.tokenizer.encode(prefill, add_special_tokens=False)
        if not prefill_ids:
            return []
        if len(agent_data.response_mask) + len(prefill_ids) >= self.response_length:
            return []
        agent_data.prompt_ids += prefill_ids
        # Trainable prefills are counterfactual, teacher-forced actions. Their rollout
        # log-probability placeholders remain zero for alignment, so configurations
        # enabling them must recompute old-policy likelihoods rather than use rollout IS.
        agent_data.response_mask += [int(trainable)] * len(prefill_ids)
        agent_data.retry_sft_mask += [0] * len(prefill_ids)
        agent_data.retry_critic_mask += [0] * len(prefill_ids)
        return prefill_ids

    @staticmethod
    def _tool_call_as_reward_dict(tool_call: FunctionCall) -> dict[str, Any] | None:
        arguments = tool_call.arguments
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                return None
        if not isinstance(arguments, dict):
            return None
        return {"name": tool_call.name, "arguments": arguments}

    def _record_retry_sft_candidate(
        self,
        *,
        agent_data: AgentData,
        response_start: int,
        response_end: int,
    ) -> None:
        suffix_text = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        standalone_text = apply_response_display_prefix(
            suffix_text,
            self.response_display_prefix or "<think>\n",
        )
        match = re.fullmatch(
            r"\s*<think>(?P<think>.*?)</think>\s*<tool_call>(?P<tool_call>.*?)</tool_call>\s*",
            standalone_text,
            flags=re.DOTALL,
        )
        think_nonempty = bool(match and match.group("think").strip())
        format_valid = False
        parsed_payload: dict[str, Any] | None = None
        if match is not None:
            payload = _parse_tool_call_payload(match.group("tool_call"))
            if payload is not None and not _contains_reserved_tag(match.group("think")):
                parsed_payload = payload
                format_valid = think_nonempty and len(agent_data.tool_calls) == 1

        parsed_tool_call = (
            self._tool_call_as_reward_dict(agent_data.tool_calls[0])
            if len(agent_data.tool_calls) == 1
            else None
        )
        tool_matches_text = parsed_payload is not None and parsed_payload == parsed_tool_call
        tool_history = list(agent_data.executed_tool_calls)
        if parsed_tool_call is not None:
            tool_history.append(parsed_tool_call)
        diagnostics = tool_repeat_overlap_call_diagnostics(
            tool_history,
            iou_threshold=self.tool_repeat_overlap_iou_threshold,
            small_crop_area_threshold=self.tool_repeat_overlap_small_crop_area_threshold,
            free_depth_refinements=self.tool_repeat_overlap_free_depth_refinements,
            depth_containment_threshold=self.tool_repeat_overlap_depth_containment_threshold,
        )
        current_diagnostics = diagnostics[-1] if parsed_tool_call is not None and diagnostics else {"valid": False}
        rejected_grade = agent_data.extra_fields.get("conditional_answer_retry_last_grade")
        rejected_accuracy = rejected_grade.get("accuracy") if isinstance(rejected_grade, dict) else None
        static_reasons = []
        if not format_valid or not tool_matches_text:
            static_reasons.append("invalid_format")
        if not think_nonempty:
            static_reasons.append("empty_think")
        if not bool(current_diagnostics.get("valid")):
            static_reasons.append("invalid_coordinate")
        if bool(current_diagnostics.get("is_repeat")):
            static_reasons.append("repeat")
        if bool(current_diagnostics.get("is_small_crop")):
            static_reasons.append("small_crop")
        if bool(current_diagnostics.get("is_excess_depth")):
            static_reasons.append("excess_depth")

        agent_data.retry_sft_candidates.append(
            {
                "assistant_turn": agent_data.assistant_turns,
                "response_start": response_start,
                "response_end": response_end,
                "rejected_accuracy": rejected_accuracy,
                "static_rejection_reasons": static_reasons,
                "tool_executed_successfully": False,
                "tool_call": parsed_tool_call,
                "crop_diagnostics": current_diagnostics,
            }
        )

    def _snapshot_generation_state(self, agent_data: AgentData) -> dict[str, Any]:
        return {
            "prompt_ids_len": len(agent_data.prompt_ids),
            "response_mask_len": len(agent_data.response_mask),
            "retry_sft_mask_len": len(agent_data.retry_sft_mask),
            "retry_critic_mask_len": len(agent_data.retry_critic_mask),
            "response_logprobs_len": len(agent_data.response_logprobs),
            "assistant_turns": agent_data.assistant_turns,
            "extra_fields": dict(agent_data.extra_fields),
            "debug_events_len": len(agent_data.debug_events),
        }

    def _rollback_generation_state(self, agent_data: AgentData, snapshot: dict[str, Any]) -> None:
        agent_data.prompt_ids = agent_data.prompt_ids[: snapshot["prompt_ids_len"]]
        agent_data.response_ids = []
        agent_data.response_mask = agent_data.response_mask[: snapshot["response_mask_len"]]
        agent_data.retry_sft_mask = agent_data.retry_sft_mask[: snapshot["retry_sft_mask_len"]]
        agent_data.retry_critic_mask = agent_data.retry_critic_mask[: snapshot["retry_critic_mask_len"]]
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
            appended = await self._append_retry_critic_supervision(agent_data, grade=grade, label="A")
            if appended:
                agent_data.extra_fields["__agent_loop_precritic_grade"] = grade
            return None

        retry_count = int(agent_data.extra_fields.get("conditional_answer_retry_count") or 0)
        skipped_reason = self._incorrect_answer_retry_skipped_reason(
            agent_data=agent_data,
            snapshot=generation_snapshot,
            retry_count=retry_count,
        )
        if skipped_reason is not None:
            agent_data.extra_fields["conditional_answer_retry_skipped_reason"] = skipped_reason
            appended = await self._append_retry_critic_supervision(agent_data, grade=grade, label="R")
            if appended:
                agent_data.extra_fields["__agent_loop_precritic_grade"] = grade
            self._log_incorrect_answer_retry_skipped(agent_data, grade, retry_count, skipped_reason)
            return None

        await self._append_retry_critic_supervision(agent_data, grade=grade, label="R")
        self._maybe_append_wrong_answer_branch(
            agent_data=agent_data,
            generation_snapshot=generation_snapshot,
            grade=grade,
            retry_count=retry_count + 1,
            removed_assistant_text=assistant_message,
            rollout_kwargs=rollout_kwargs,
        )
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

    def _retry_critic_grade_is_supervisable(self, grade: Any) -> bool:
        if not isinstance(grade, dict) or grade.get("error"):
            return False
        accuracy = grade.get("accuracy")
        return isinstance(accuracy, int | float) and not isinstance(accuracy, bool) and math.isfinite(float(accuracy))

    async def _retry_critic_prompt_prefix_ids(self) -> list[int]:
        question_ids = await self.apply_chat_template(
            [{"role": "user", "content": self.conditional_answer_retry_critic_question}],
            remove_system_prompt=True,
        )
        prefill_ids = self.tokenizer.encode(
            self.conditional_answer_retry_critic_assistant_prefill,
            add_special_tokens=False,
        )
        return list(question_ids) + list(prefill_ids)

    async def _append_retry_critic_supervision(
        self,
        agent_data: AgentData,
        *,
        grade: dict[str, Any],
        label: str,
    ) -> bool:
        if not (
            getattr(self, "conditional_answer_retry_critic_enabled", False)
            and getattr(self, "conditional_answer_retry_critic_oracle_supervision_enabled", False)
            and self._retry_critic_grade_is_supervisable(grade)
        ):
            return False
        token_id = (
            self._retry_critic_accept_token_id
            if label == "A"
            else self._retry_critic_retry_token_id
            if label == "R"
            else None
        )
        if token_id is None:
            raise ValueError(f"unsupported retry critic label {label!r}")
        prefix_ids = await self._retry_critic_prompt_prefix_ids()
        closing_ids = self.tokenizer.encode("\n</answer><|im_end|>\n", add_special_tokens=False)
        sidecar_ids = prefix_ids + [int(token_id)] + closing_ids
        if len(agent_data.response_mask) + len(sidecar_ids) > self.response_length:
            agent_data.extra_fields["retry_critic_supervision_skipped_reason"] = "response_length"
            return False

        agent_data.prompt_ids += sidecar_ids
        agent_data.response_mask += [0] * len(sidecar_ids)
        agent_data.retry_sft_mask += [0] * len(sidecar_ids)
        agent_data.retry_critic_mask += [0] * len(prefix_ids) + [1] + [0] * len(closing_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(sidecar_ids)
        agent_data.extra_fields["retry_critic_label"] = label
        agent_data.extra_fields["retry_critic_token_count"] = 1
        return True

    async def _maybe_retry_answer_with_critic(
        self,
        *,
        agent_data: AgentData,
        generation_snapshot: dict[str, Any],
        assistant_message: str,
        ignore_termination: bool,
    ) -> AgentState | None:
        retry_count = int(agent_data.extra_fields.get("conditional_answer_retry_count") or 0)
        candidate_turn = int(generation_snapshot["assistant_turns"]) + 1
        try:
            prefix_ids = await self._retry_critic_prompt_prefix_ids()
            critic_output: TokenOutput = await self.server_manager.generate(
                request_id=f"{agent_data.request_id}-retry-critic-{uuid4().hex}",
                prompt_ids=agent_data.prompt_ids + prefix_ids,
                sampling_params={
                    "max_tokens": 1,
                    "temperature": 0.0,
                    "top_p": 1.0,
                    "top_k": -1,
                    "logprobs": True,
                    "allowed_token_ids": [
                        int(self._retry_critic_accept_token_id),
                        int(self._retry_critic_retry_token_id),
                    ],
                },
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
            )
            if len(critic_output.token_ids) != 1 or not critic_output.log_probs:
                raise ValueError("critic generation did not return exactly one token and its log probability")
            selected_id = int(critic_output.token_ids[0])
            selected_logprob = float(critic_output.log_probs[0])
            if selected_id not in {self._retry_critic_accept_token_id, self._retry_critic_retry_token_id}:
                raise ValueError(f"critic returned disallowed token id {selected_id}")
            selected_probability = min(max(math.exp(selected_logprob), 0.0), 1.0)
            if selected_id == self._retry_critic_retry_token_id:
                p_retry = selected_probability
                model_token = "R"
            else:
                p_retry = 1.0 - selected_probability
                model_token = "A"
            p_accept = 1.0 - p_retry
            decision = "R" if p_retry > self.conditional_answer_retry_critic_decision_threshold else "A"
        except Exception as exc:  # noqa: BLE001 - validation critic failures must retain the candidate.
            self._record_retry_critic_validation_decision(
                agent_data,
                {
                    "candidate_turn": candidate_turn,
                    "model_token": None,
                    "decision": None,
                    "selected_logprob": None,
                    "p_accept": None,
                    "p_retry": None,
                    "retry_feasible": None,
                    "controller_action": "accept_fallback",
                    "retry_depth": retry_count,
                    "skip_or_error_reason": f"{type(exc).__name__}: {exc}",
                },
                failure=True,
            )
            return None

        skipped_reason = None
        if decision == "R":
            skipped_reason = self._incorrect_answer_retry_skipped_reason(
                agent_data=agent_data,
                snapshot=generation_snapshot,
                retry_count=retry_count,
            )
        retry_feasible = decision == "R" and skipped_reason is None
        action = "retry" if retry_feasible else "forced_accept" if decision == "R" else "accept"
        record = {
            "candidate_turn": candidate_turn,
            "model_token": model_token,
            "decision": decision,
            "selected_logprob": selected_logprob,
            "p_accept": p_accept,
            "p_retry": p_retry,
            "retry_feasible": retry_feasible,
            "controller_action": action,
            "retry_depth": retry_count + int(retry_feasible),
            "skip_or_error_reason": skipped_reason,
        }
        if not retry_feasible:
            self._record_retry_critic_validation_decision(
                agent_data,
                record,
                forced_accept=decision == "R",
            )
            return None

        self._rollback_generation_state(agent_data, generation_snapshot)
        retry_count += 1
        suffix = self.conditional_answer_retry_suffix or ""
        await self._append_conditional_retry_suffix_to_previous_message(agent_data, suffix)
        if self.conditional_answer_retry_force_next_tool_call:
            agent_data.force_next_tool_call = True
        agent_data.extra_fields["conditional_answer_retry_count"] = retry_count
        agent_data.extra_fields["conditional_answer_retry_skipped_reason"] = None
        self._record_retry_critic_validation_decision(agent_data, record, retried=True)
        if not ignore_termination and len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED
        return AgentState.GENERATING

    def _record_retry_critic_validation_decision(
        self,
        agent_data: AgentData,
        record: dict[str, Any],
        *,
        retried: bool = False,
        forced_accept: bool = False,
        failure: bool = False,
    ) -> None:
        records = list(agent_data.extra_fields.get("retry_critic_validation_decision_log") or [])
        records.append(record)
        agent_data.extra_fields["retry_critic_validation_decision_log"] = records
        agent_data.extra_fields["retry_critic_validation_decisions"] = len(records)
        agent_data.extra_fields["retry_critic_validation_accepts"] = sum(
            item.get("controller_action") == "accept" for item in records
        )
        agent_data.extra_fields["retry_critic_validation_retries"] = sum(
            item.get("controller_action") == "retry" for item in records
        )
        agent_data.extra_fields["retry_critic_validation_forced_accepts"] = sum(
            item.get("controller_action") == "forced_accept" for item in records
        )
        agent_data.extra_fields["retry_critic_validation_failures"] = sum(
            item.get("controller_action") == "accept_fallback" for item in records
        )
        probabilities = [item.get("p_retry") for item in records if isinstance(item.get("p_retry"), int | float)]
        agent_data.extra_fields["retry_critic_validation_mean_retry_probability"] = (
            sum(float(value) for value in probabilities) / len(probabilities) if probabilities else None
        )
        if self._debug_enabled_for_agent(agent_data):
            event = {"event": "retry_critic_validation_decision", "request_id": agent_data.request_id, **record}
            agent_data.debug_events.append(event)
            self._debug_append_event(event)

    @staticmethod
    def _retry_critic_validation_summary(agent_data: AgentData) -> dict[str, Any]:
        keys = (
            "retry_critic_validation_decisions",
            "retry_critic_validation_accepts",
            "retry_critic_validation_retries",
            "retry_critic_validation_forced_accepts",
            "retry_critic_validation_failures",
            "retry_critic_validation_mean_retry_probability",
        )
        return {key: agent_data.extra_fields[key] for key in keys if key in agent_data.extra_fields}

    def _maybe_append_wrong_answer_branch(
        self,
        agent_data: AgentData,
        generation_snapshot: dict[str, Any],
        grade: dict[str, Any],
        retry_count: int,
        removed_assistant_text: str,
        rollout_kwargs: dict[str, Any],
    ) -> None:
        if not self.conditional_answer_retry_branch_enabled:
            return
        debug_trajectory = rollout_kwargs.get("_debug_trajectory", {}) or {}
        if debug_trajectory.get("validate") and not self.conditional_answer_retry_branch_enable_validation:
            return
        if self.conditional_answer_retry_branch_mask_mode != "divergence_only":
            raise ValueError(
                "conditional_answer_retry_branch_mask_mode only supports 'divergence_only', "
                f"got {self.conditional_answer_retry_branch_mask_mode!r}"
            )

        divergence_start = int(generation_snapshot["response_mask_len"])
        response_ids = agent_data.prompt_ids[-len(agent_data.response_mask) :] if agent_data.response_mask else []
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - len(agent_data.response_mask)]
        branch_response_mask = [
            int(mask) if idx >= divergence_start else 0 for idx, mask in enumerate(agent_data.response_mask)
        ]
        branch_response_mask = branch_response_mask[: len(response_ids)]

        branch_logprobs = None
        if agent_data.response_logprobs:
            branch_logprobs = list(agent_data.response_logprobs[: len(response_ids)])

        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = list(agent_data.image_data)
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = list(agent_data.video_data)

        reward_extra_info = self._branch_reward_extra_info(grade)
        baseline_key = self.conditional_answer_retry_branch_baseline_weight_key
        branch_output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=branch_response_mask[: self.response_length],
            retry_sft_mask=[0] * min(len(response_ids), self.response_length),
            retry_critic_mask=agent_data.retry_critic_mask[: min(len(response_ids), self.response_length)],
            multi_modal_data=multi_modal_data,
            response_logprobs=branch_logprobs[: self.response_length] if branch_logprobs else None,
            reward_score=float(reward_extra_info.get("score", 0.0) or 0.0),
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=AgentLoopMetrics(),
            routed_experts=None,
            extra_fields={
                **TERMINAL_TOOL_EXTRA_FIELD_DEFAULTS,
                "request_id": agent_data.request_id,
                "assistant_turns": agent_data.assistant_turns,
                "is_conditional_answer_retry_branch": True,
                baseline_key: 0.0,
                "conditional_answer_retry_branch_mask_mode": self.conditional_answer_retry_branch_mask_mode,
                "conditional_answer_retry_branch_retry_count": retry_count,
                "conditional_answer_retry_branch_grade": grade,
                "conditional_answer_retry_branch_removed_assistant_text": removed_assistant_text,
                "conditional_answer_retry_count": retry_count,
                "conditional_answer_retry_last_grade": grade,
                "conditional_answer_retry_skipped_reason": None,
                "retry_critic_label": agent_data.extra_fields.get("retry_critic_label"),
                "retry_critic_token_count": int(sum(agent_data.retry_critic_mask)),
                "reward_extra_info": reward_extra_info,
                "turn_scores": list(agent_data.turn_scores),
                "tool_rewards": list(agent_data.tool_rewards),
                **self._randomized_turn_cap_metadata(agent_data),
                CROP_IMAGE_REFS_KEY: list(agent_data.crop_image_refs),
            },
        )
        agent_data.branch_outputs.append(branch_output)

    def _branch_reward_extra_info(self, grade: dict[str, Any]) -> dict[str, Any]:
        reward_extra_info = dict(grade) if isinstance(grade, dict) else {}
        if self.conditional_answer_retry_branch_use_graded_rewards:
            return reward_extra_info

        reward_extra_info["score"] = 0.0
        for key in ("accuracy", "format", "tool_repeat_overlap"):
            reward_extra_info[key] = 0.0
        for key, value in self.conditional_answer_retry_branch_reward_components.items():
            reward_extra_info[str(key)] = float(value)
        return reward_extra_info

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
        if getattr(self, "conditional_answer_retry_sft_enabled", False):
            # Retry-SFT eligibility is finalized during agent-loop postprocessing,
            # before trainer-side reward extraction. Return the ordinary base grade
            # for non-judge domains as well so final accuracy is available there.
            return grade
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
        raw_extra_info = rollout_kwargs.get("extra_info")
        extra_info = dict(raw_extra_info) if isinstance(raw_extra_info, dict) else {}
        extra_info["assistant_turns"] = agent_data.assistant_turns
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

        is_valid, final_answer_text, _, _ = _parse_multiturn_tool_calling_output(solution_str)
        if not is_valid or not final_answer_text:
            if "<answer>" in str(solution_str) and "</answer>" in str(solution_str):
                return {"llm_judge_used": False, "llm_judge_error": "missing_final_answer"}
            return {"llm_judge_used": False, "llm_judge_error": "invalid_trajectory_format"}

        reward_type = _extract_reward_type(extra_info)
        answer_info = extract_answer_for_accuracy(
            final_answer_text,
            reward_type,
            allow_unboxed_answer_for_accuracy=getattr(self, "allow_unboxed_answer_for_accuracy", True),
        )
        if answer_info.prediction is None:
            return {"llm_judge_used": False, "llm_judge_error": "missing_final_answer"}

        question = self._extract_llm_judge_question(agent_data, extra_info)
        if not question:
            return {"llm_judge_used": False, "llm_judge_error": "missing_question"}

        judge_prompt = self._format_llm_judge_prompt(
            question=question,
            reference=str(ground_truth).strip(),
            prediction=str(answer_info.prediction).strip(),
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
            "answer_has_boxed": answer_info.has_boxed,
            "answer_used_unboxed_fallback": answer_info.used_unboxed_fallback,
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
        effective_max_assistant_turns = self._effective_max_assistant_turns(agent_data)
        if effective_max_assistant_turns and snapshot["assistant_turns"] + 2 > effective_max_assistant_turns:
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
            agent_data.retry_sft_mask = []
            agent_data.retry_critic_mask = []
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
            agent_data.retry_sft_mask = (
                agent_data.retry_sft_mask[:mask_start]
                + [0] * len(replacement_ids)
                + agent_data.retry_sft_mask[mask_end:]
            )
            agent_data.retry_critic_mask = (
                agent_data.retry_critic_mask[:mask_start]
                + [0] * len(replacement_ids)
                + agent_data.retry_critic_mask[mask_end:]
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
            agent_data.retry_sft_mask += [0] * len(response_ids)
            agent_data.retry_critic_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)
            agent_data.user_turns += 1
            return AgentState.TERMINATED

        successful_calls = [
            self._tool_call_as_reward_dict(tool_call)
            for tool_call, (tool_response, _, _) in zip(active_tool_calls, responses, strict=False)
            if not tool_response.terminate
        ]
        successful_calls = [call for call in successful_calls if call is not None]
        agent_data.executed_tool_calls.extend(successful_calls)
        for candidate in reversed(agent_data.retry_sft_candidates):
            if candidate.get("assistant_turn") != agent_data.assistant_turns:
                continue
            candidate["tool_executed_successfully"] = (
                len(successful_calls) == 1 and successful_calls[0] == candidate.get("tool_call")
            )
            break

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
        agent_data.retry_sft_mask += [0] * len(response_ids)
        agent_data.retry_critic_mask += [0] * len(response_ids)
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
        agent_data.retry_sft_mask += [0] * len(response_ids)
        agent_data.retry_critic_mask += [0] * len(response_ids)
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
            "retry_critic_tokens": int(sum(output.retry_critic_mask or [])),
            "retry_critic_label": output.extra_fields.get("retry_critic_label"),
            "num_turns": output.num_turns,
            "turn_scores": agent_data.turn_scores,
            "tool_rewards": agent_data.tool_rewards,
            "conditional_answer_retry_branch_outputs": [
                {
                    "response_tokens": len(branch.response_ids),
                    "masked_tokens": int(sum(branch.response_mask)),
                    "retry_critic_tokens": int(sum(branch.retry_critic_mask or [])),
                    "retry_critic_label": branch.extra_fields.get("retry_critic_label"),
                    "is_conditional_answer_retry_branch": branch.extra_fields.get(
                        "is_conditional_answer_retry_branch"
                    ),
                    self.conditional_answer_retry_branch_baseline_weight_key: branch.extra_fields.get(
                        self.conditional_answer_retry_branch_baseline_weight_key
                    ),
                    "reward_extra_info": branch.extra_fields.get("reward_extra_info"),
                }
                for branch in (output.branch_outputs or [])
            ],
            "debug_events": agent_data.debug_events,
            "retry_critic_validation_decision_log": agent_data.extra_fields.get(
                "retry_critic_validation_decision_log", []
            ),
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
