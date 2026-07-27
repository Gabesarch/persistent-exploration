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
import json
from typing import Any

import torch

FORCE_ACTION_LOGITS_PROCESSOR_FQCN = "verl.utils.vllm_forced_action_logits_processor:ForceActionTagAfterThinkLogitsProcessor"
FORCE_ACTION_REQUIRED_TAG_KEY = "verl_force_action_tag_after_think_required_tag"
FORCE_ACTION_TAG_IDS_KEY = "verl_force_action_tag_after_think_tag_ids"
FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY = "verl_force_action_tag_after_think_after_think_ids_options"
VLLM_OPENAI_EXTRA_ARGS_KEY = "vllm_xargs"

SUPPORTED_FORCE_ACTION_TAGS = {"<tool_call>", "<answer>"}


def build_forced_action_extra_args(
    tokenizer: Any,
    required_tag: str,
    *,
    forced_suffix: str = "",
) -> dict[str, Any]:
    """Build the per-request vLLM payload used to force an action after ``</think>``."""
    forced_text = f"\n{required_tag}{forced_suffix}"
    tag_ids = tokenizer.encode(forced_text, add_special_tokens=False)
    after_think_ids_options = _after_think_token_id_options(tokenizer)
    validate_forced_action_payload(required_tag, tag_ids, after_think_ids_options)
    return {
        FORCE_ACTION_REQUIRED_TAG_KEY: required_tag,
        FORCE_ACTION_TAG_IDS_KEY: tag_ids,
        FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY: after_think_ids_options,
    }


def build_forced_action_openai_extra_body(
    tokenizer: Any,
    required_tag: str,
    *,
    forced_suffix: str = "",
) -> dict[str, Any]:
    """Build the vLLM OpenAI API body carrying forced-action sampling args.

    vLLM's in-process interface calls this ``SamplingParams.extra_args``, but
    its OpenAI-compatible request schema exposes the same mapping as
    ``vllm_xargs``. Sending an ``extra_args`` JSON field is silently ignored by
    the OpenAI route in vLLM 0.23, leaving the request processor disabled.
    """
    extra_args = build_forced_action_extra_args(
        tokenizer,
        required_tag,
        forced_suffix=forced_suffix,
    )
    # ChatCompletionRequest.vllm_xargs only permits scalar values and flat
    # scalar lists. Preserve the multiple accepted marker encodings as a JSON
    # string and restore them in the server-side adapter.
    extra_args[FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY] = json.dumps(
        extra_args[FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY],
        separators=(",", ":"),
    )
    return {VLLM_OPENAI_EXTRA_ARGS_KEY: extra_args}


def _after_think_token_id_options(tokenizer: Any) -> list[list[int]]:
    # A model may emit the registered Qwen special token or ordinary vocabulary
    # pieces which decode to the same literal. Accept both representations.
    tokenizations = [
        tokenizer.encode("</think>", add_special_tokens=False),
        _encode_token_parts(tokenizer, ("</", "think", ">")),
        _encode_token_parts(tokenizer, ("<", "/", "think", ">")),
        _encode_token_parts(tokenizer, ("<", "/think", ">")),
    ]
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for token_ids in tokenizations:
        key = tuple(int(token_id) for token_id in token_ids)
        if key and key not in seen:
            seen.add(key)
            unique.append(list(key))
    return unique


def _encode_token_parts(tokenizer: Any, parts: tuple[str, ...]) -> list[int]:
    token_ids: list[int] = []
    for part in parts:
        token_ids.extend(tokenizer.encode(part, add_special_tokens=False))
    return token_ids


class ForceActionTagAfterThinkRequestProcessor:
    """Request-level processor that forces the action tag after a think block."""

    def __init__(self, after_think_ids_options: list[list[int]], tag_ids: list[int]):
        self.after_think_ids_options = after_think_ids_options
        self.tag_ids = tag_ids
        # A response can emit another literal </think> after the required
        # action tag has already been forced (this is especially common when
        # a hard final-answer retry starts directly inside <answer>).  The
        # contract is to force the transition once, not to inject a fresh
        # action tag after every later marker.
        self._forced_action_completed = False

    def __call__(
        self,
        prompt_ids: list[int],
        output_ids: list[int],
        logits: torch.Tensor,
    ) -> torch.Tensor:
        forced_token_id = self._forced_token_id(prompt_ids, output_ids)
        if forced_token_id is None:
            return logits

        logits.fill_(float("-inf"))
        logits[forced_token_id] = 0.0
        return logits

    def _forced_token_id(self, prompt_ids: list[int], output_ids: list[int]) -> int | None:
        if self._forced_action_completed:
            return None

        # Observe completion on the decoding step immediately after the final
        # forced token.  Keeping this one bit of request-local state avoids
        # rescanning or copying the full prompt/output sequence on every step.
        for after_think_ids in self.after_think_ids_options:
            if self._combined_endswith(prompt_ids, output_ids, after_think_ids + self.tag_ids):
                self._forced_action_completed = True
                return None

        for after_think_ids in self.after_think_ids_options:
            if self._combined_endswith(prompt_ids, output_ids, after_think_ids):
                return self.tag_ids[0]

            for tag_prefix_len in range(1, len(self.tag_ids)):
                prefix = after_think_ids + self.tag_ids[:tag_prefix_len]
                if self._combined_endswith(prompt_ids, output_ids, prefix):
                    return self.tag_ids[tag_prefix_len]

        return None

    @staticmethod
    def _combined_endswith(prompt_ids: list[int], output_ids: list[int], suffix: list[int]) -> bool:
        if not suffix or len(prompt_ids) + len(output_ids) < len(suffix):
            return False
        output_needed = min(len(output_ids), len(suffix))
        if output_needed and output_ids[-output_needed:] != suffix[-output_needed:]:
            return False
        prompt_needed = len(suffix) - output_needed
        return not prompt_needed or prompt_ids[-prompt_needed:] == suffix[:prompt_needed]


def validate_forced_action_payload(required_tag: Any, tag_ids: Any, after_think_ids_options: Any) -> None:
    after_think_ids_options = decode_after_think_ids_options(after_think_ids_options)
    if required_tag not in SUPPORTED_FORCE_ACTION_TAGS:
        raise ValueError(f"unsupported forced action tag: {required_tag!r}")
    if not _is_int_list(tag_ids):
        raise ValueError("forced action tag ids must be a non-empty list of ints")
    if not isinstance(after_think_ids_options, list) or not after_think_ids_options:
        raise ValueError("forced action after_think ids options must be a non-empty list")
    for option in after_think_ids_options:
        if not _is_int_list(option):
            raise ValueError("each forced action after_think ids option must be a non-empty list of ints")


def decode_after_think_ids_options(value: Any) -> list[list[int]]:
    """Restore marker-token alternatives serialized through vLLM's OpenAI API."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError("forced action after_think ids options must be valid JSON") from exc
    return value


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
