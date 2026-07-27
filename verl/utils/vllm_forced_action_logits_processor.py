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
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import AdapterLogitsProcessor

from verl.utils.forced_action_tags import (
    FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY,
    FORCE_ACTION_REQUIRED_TAG_KEY,
    FORCE_ACTION_TAG_IDS_KEY,
    ForceActionTagAfterThinkRequestProcessor,
    decode_after_think_ids_options,
    validate_forced_action_payload,
)


class ForceActionTagAfterThinkLogitsProcessor(AdapterLogitsProcessor):
    """vLLM v1 batch adapter for forced action tags after generated think blocks."""

    @classmethod
    def validate_params(cls, sampling_params: SamplingParams):
        extra_args = sampling_params.extra_args or {}
        present_keys = {
            key
            for key in (
                FORCE_ACTION_REQUIRED_TAG_KEY,
                FORCE_ACTION_TAG_IDS_KEY,
                FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY,
            )
            if key in extra_args
        }
        if not present_keys:
            return
        if present_keys != {
            FORCE_ACTION_REQUIRED_TAG_KEY,
            FORCE_ACTION_TAG_IDS_KEY,
            FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY,
        }:
            raise ValueError(f"incomplete forced action tag extra_args: {sorted(present_keys)}")

        validate_forced_action_payload(
            extra_args[FORCE_ACTION_REQUIRED_TAG_KEY],
            extra_args[FORCE_ACTION_TAG_IDS_KEY],
            extra_args[FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY],
        )

    def is_argmax_invariant(self) -> bool:
        return False

    def new_req_logits_processor(self, params: SamplingParams) -> ForceActionTagAfterThinkRequestProcessor | None:
        extra_args = params.extra_args or {}
        if FORCE_ACTION_REQUIRED_TAG_KEY not in extra_args:
            return None

        self.validate_params(params)
        return ForceActionTagAfterThinkRequestProcessor(
            after_think_ids_options=decode_after_think_ids_options(
                extra_args[FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY]
            ),
            tag_ids=extra_args[FORCE_ACTION_TAG_IDS_KEY],
        )
