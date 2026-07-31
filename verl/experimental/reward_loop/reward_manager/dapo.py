# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

import inspect

from verl import DataProto
from verl.experimental.reward_loop.reward_manager import register
from verl.experimental.reward_loop.reward_manager.base import RewardManagerBase
from verl.utils.debug.comprehensive_logging import append_jsonl, get_debug_options, path_join, should_log_step
from verl.utils.reward_score import default_compute_score


@register("dapo")
class DAPORewardManager(RewardManagerBase):
    """DAPO Reward Manager."""

    def __init__(self, config, tokenizer, compute_score, reward_router_address=None, reward_model_tokenizer=None):
        super().__init__(config, tokenizer, compute_score)
        self.compute_score = compute_score or default_compute_score
        self.is_async_reward_score = inspect.iscoroutinefunction(self.compute_score)

        # DAPO Reward Config
        reward_kwargs = config.reward.get("reward_kwargs", {})
        overlong_buffer_cfg = reward_kwargs.get("overlong_buffer_cfg", None)
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = reward_kwargs.get("max_resp_len", None)
        self.reward_response_prefix = reward_kwargs.get("reward_response_prefix", "")
        self.reward_router_address = reward_router_address
        self.reward_model_tokenizer = reward_model_tokenizer
        self.debug_options = get_debug_options(config)

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, (
                f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
            )
            assert self.max_resp_len >= self.overlong_buffer_cfg.len, (
                "max_resp_len must be larger than overlong_buffer.len"
            )
            assert not self.overlong_buffer_cfg.enable or self.overlong_buffer_cfg.len > 0, (
                "overlong_buffer.len must be positive when overlong penalty is enabled,"
                f"but got {self.overlong_buffer_cfg.len}."
                "To disable the overlong penalty, set overlong_buffer.enable = False"
            )

    async def run_single(self, data: DataProto) -> dict:
        assert len(data) == 1, "Only support single data item"
        data_item = data[0]
        response_ids = data_item.batch["responses"]
        response_length = response_ids.shape[-1]
        valid_response_length = data_item.batch["attention_mask"][-response_length:].sum()
        valid_response_ids = response_ids[:valid_response_length]

        data_source = data_item.non_tensor_batch["data_source"]
        ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
        raw_extra_info = data_item.non_tensor_batch.get("extra_info", {})
        extra_info = dict(raw_extra_info) if isinstance(raw_extra_info, dict) else {}
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields", None)
        if isinstance(tool_extra_fields, dict):
            extra_info.update(tool_extra_fields)

        response_str = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
        )
        if self.reward_response_prefix:
            response_str = f"{self.reward_response_prefix}{response_str}"
        extra_reward_kwargs = (
            {
                "reward_router_address": self.reward_router_address,
                "reward_model_tokenizer": self.reward_model_tokenizer,
            }
            if self.reward_router_address is not None
            else {}
        )
        if self.is_async_reward_score:
            result = await self.compute_score(
                data_source=data_source,
                solution_str=response_str,
                ground_truth=ground_truth,
                extra_info=extra_info,
                **extra_reward_kwargs,
            )
        else:
            result = await self.loop.run_in_executor(
                None,
                lambda: self.compute_score(
                    data_source=data_source,
                    solution_str=response_str,
                    ground_truth=ground_truth,
                    extra_info=extra_info,
                    **extra_reward_kwargs,
                ),
            )

        reward_extra_info = {}

        score: float
        if isinstance(result, dict):
            score = result["score"]
            for key, value in result.items():
                reward_extra_info[key] = value
        else:
            score = result
            reward_extra_info["acc"] = score

        reward = score

        if self.overlong_buffer_cfg is not None and self.overlong_buffer_cfg.enable:
            overlong_buffer_len = self.overlong_buffer_cfg.len
            expected_len = self.max_resp_len - overlong_buffer_len
            exceed_len = valid_response_length - expected_len
            overlong_penalty_factor = self.overlong_buffer_cfg.penalty_factor
            overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            reward += overlong_reward
            if self.overlong_buffer_cfg.log:
                reward_extra_info["overlong_reward"] = overlong_reward
                reward_extra_info["overlong"] = overlong_reward < 0

        self._maybe_log_reward_assignment(
            data_item=data_item,
            data_source=data_source,
            response_str=response_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            raw_result=result,
            reward_extra_info=reward_extra_info,
            final_reward=reward,
            valid_response_length=valid_response_length,
        )
        return {"reward_score": reward, "reward_extra_info": reward_extra_info}

    def _maybe_log_reward_assignment(
        self,
        data_item: DataProto,
        data_source,
        response_str,
        ground_truth,
        extra_info,
        raw_result,
        reward_extra_info,
        final_reward,
        valid_response_length,
    ) -> None:
        debug_dir = self.debug_options.get("dir")
        if not self.debug_options.get("enabled") or not debug_dir:
            return
        global_step = self._extract_global_step(data_item)
        if not should_log_step(self.config, global_step):
            return
        reward_type = extra_info.get("reward_type") if isinstance(extra_info, dict) else None
        tolerance = extra_info.get("tolerance") if isinstance(extra_info, dict) else None
        record = {
            "event": "reward_assignment",
            "global_step": global_step,
            "data_source": data_source,
            "uid": data_item.non_tensor_batch.get("uid"),
            "request_id": self._extract_request_id(data_item),
            "ground_truth": ground_truth,
            "reward_type": reward_type,
            "tolerance": tolerance,
            "extra_info": extra_info,
            "response": response_str,
            "valid_response_length": valid_response_length,
            "raw_result": raw_result,
            "reward_extra_info": reward_extra_info,
            "score": reward_extra_info.get("score"),
            "accuracy": reward_extra_info.get("accuracy"),
            "format": reward_extra_info.get("format"),
            "final_reward": final_reward,
        }
        append_jsonl(
            path_join(debug_dir, "reward_assignments.jsonl"),
            record,
            max_text_chars=self.debug_options.get("max_text_chars"),
        )

    @staticmethod
    def _extract_request_id(data_item) -> str | None:
        request_id = data_item.non_tensor_batch.get("request_id")
        if request_id is not None:
            return request_id
        tool_extra_fields = data_item.non_tensor_batch.get("tool_extra_fields")
        if isinstance(tool_extra_fields, dict):
            return tool_extra_fields.get("request_id")
        return None

    @staticmethod
    def _extract_global_step(data_item) -> int | None:
        for source in (getattr(data_item, "meta_info", None), getattr(data_item, "non_tensor_batch", None)):
            if isinstance(source, dict):
                value = source.get("global_steps")
                if value is not None:
                    return value
        return None
