# Copyright 2026 Bytedance Ltd. and/or its affiliates
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
from typing import Any

import torch

FORCE_ACTION_LOGITS_PROCESSOR_FQCN = "verl.utils.vllm_forced_action_logits_processor:ForceActionTagAfterThinkLogitsProcessor"
FORCE_ACTION_REQUIRED_TAG_KEY = "verl_force_action_tag_after_think_required_tag"
FORCE_ACTION_TAG_IDS_KEY = "verl_force_action_tag_after_think_tag_ids"
FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY = "verl_force_action_tag_after_think_after_think_ids_options"

SUPPORTED_FORCE_ACTION_TAGS = {"<tool_call>", "<answer>"}


class ForceActionTagAfterThinkRequestProcessor:
    """Request-level processor that forces the action tag after a think block."""

    def __init__(self, after_think_ids_options: list[list[int]], tag_ids: list[int]):
        self.after_think_ids_options = after_think_ids_options
        self.tag_ids = tag_ids

    def __call__(self, output_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        forced_token_id = self._forced_token_id(output_ids)
        if forced_token_id is None:
            return logits

        logits.fill_(float("-inf"))
        logits[forced_token_id] = 0.0
        return logits

    def _forced_token_id(self, token_ids: list[int]) -> int | None:
        for after_think_ids in self.after_think_ids_options:
            if self._endswith(token_ids, after_think_ids):
                return self.tag_ids[0]

            for tag_prefix_len in range(1, len(self.tag_ids)):
                prefix = after_think_ids + self.tag_ids[:tag_prefix_len]
                if self._endswith(token_ids, prefix):
                    return self.tag_ids[tag_prefix_len]

        return None

    @staticmethod
    def _endswith(tokens: list[int], suffix: list[int]) -> bool:
        return bool(suffix) and len(tokens) >= len(suffix) and tokens[-len(suffix) :] == suffix


def validate_forced_action_payload(required_tag: Any, tag_ids: Any, after_think_ids_options: Any) -> None:
    if required_tag not in SUPPORTED_FORCE_ACTION_TAGS:
        raise ValueError(f"unsupported forced action tag: {required_tag!r}")
    if not _is_int_list(tag_ids):
        raise ValueError("forced action tag ids must be a non-empty list of ints")
    if not isinstance(after_think_ids_options, list) or not after_think_ids_options:
        raise ValueError("forced action after_think ids options must be a non-empty list")
    for option in after_think_ids_options:
        if not _is_int_list(option):
            raise ValueError("each forced action after_think ids option must be a non-empty list of ints")


def _is_int_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(isinstance(item, int) for item in value)


def ensure_forced_action_logits_processor(engine_kwargs: dict[str, Any]) -> None:
    logits_processors = engine_kwargs.get("logits_processors")
    if logits_processors is None:
        engine_kwargs["logits_processors"] = [FORCE_ACTION_LOGITS_PROCESSOR_FQCN]
        return
    if isinstance(logits_processors, str):
        logits_processors = [logits_processors]
    else:
        logits_processors = list(logits_processors)
    if FORCE_ACTION_LOGITS_PROCESSOR_FQCN not in logits_processors:
        logits_processors.append(FORCE_ACTION_LOGITS_PROCESSOR_FQCN)
    engine_kwargs["logits_processors"] = logits_processors
