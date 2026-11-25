# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
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
"""
Single Process Actor
"""

import logging
import os
import time
from typing import Any

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.models.gemma4.profiling import actor_stage, profile_actor, stage
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.crop_cache import CROP_IMAGE_REFS_KEY
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.import_utils import deprecated
from verl.utils.lazy_multi_modal import (
    lazy_multi_modal_assert_metadata_match,
    lazy_multi_modal_inputs_enabled,
    rehydrate_multi_modal_inputs,
    should_rehydrate_multi_modal_inputs,
)
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig
from verl.workers.utils.auxiliary_mass_bounds import advantage_mass_sums, detached_cap_multiplier, mass_ratio
from verl.workers.utils.gradient_diagnostics import GradientProbe, gradient_step_selected, preserved_rng
from verl.workers.utils.loss_diagnostics import COMPONENTS, reduce_loss_diagnostics, weighted_loss_components
from verl.workers.utils.losses import (
    combine_retry_critic_loss,
    combine_retry_sft_loss,
    masked_sft_nll,
    parent_balanced_retry_critic_loss,
    parent_balanced_retry_critic_strata_loss,
)

__all__ = ["DataParallelPPOActor"]

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


@deprecated("legacy worker implementation is deprecated and will be removed in v0.8.0")
class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(
        self,
        config: ActorConfig,
        actor_module: nn.Module,
        actor_optimizer: torch.optim.Optimizer = None,
        tokenizer: Any = None,
        processor: Any = None,
        dataset_cls: Any = None,
        data_config: Any = None,
        lazy_multi_modal_inputs_config: Any = None,
    ):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        self.tokenizer = tokenizer
        self.processor = processor
        self.dataset_cls = dataset_cls
        self.data_config = data_config
        self.lazy_multi_modal_inputs_config = lazy_multi_modal_inputs_config
        self.lazy_multi_modal_inputs_enabled = lazy_multi_modal_inputs_enabled(lazy_multi_modal_inputs_config)
        self.lazy_multi_modal_assert_metadata_match = lazy_multi_modal_assert_metadata_match(
            lazy_multi_modal_inputs_config
        )
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        self.use_dynamic_bsz = self.config.get("use_dynamic_bsz", False)

        self.use_prefix_grouper = self.config.get("use_prefix_grouper", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_prefix_grouper={self.use_prefix_grouper}")

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

        # Sum of squared probabilities computation (for optimal_token_baseline)
        # Only initialize if calculate_sum_pi_squared config is enabled
        if self.config.get("calculate_sum_pi_squared", False):
            self.calculate_sum_pi_squared_from_logits = (
                torch.compile(verl_F.calculate_sum_pi_squared_from_logits, dynamic=True)
                if self.config.get("use_torch_compile", True)
                else verl_F.calculate_sum_pi_squared_from_logits
            )
            assert not (self.use_fused_kernels or self.use_prefix_grouper), (
                "calculate_sum_pi_squared is not supported with "
                f"{self.use_fused_kernels=} or {self.use_prefix_grouper=} for now."
            )

    @actor_stage
    def _extract_multi_modal_inputs(self, micro_batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if "multi_modal_inputs" not in micro_batch:
            return {}

        from verl.utils.model import extract_multi_modal_inputs

        multi_modal_inputs_batch = micro_batch["multi_modal_inputs"]
        if not self.lazy_multi_modal_inputs_enabled:
            return extract_multi_modal_inputs(multi_modal_inputs_batch)

        if not should_rehydrate_multi_modal_inputs(multi_modal_inputs_batch):
            return extract_multi_modal_inputs(multi_modal_inputs_batch)

        if "raw_prompt" not in micro_batch:
            raise ValueError("lazy multimodal inputs require raw_prompt in the actor/ref microbatch")

        rebuilt_multi_modal_inputs = rehydrate_multi_modal_inputs(
            raw_prompts=micro_batch["raw_prompt"],
            input_ids=micro_batch["input_ids"],
            tokenizer=self.tokenizer,
            processor=self.processor,
            dataset_cls=self.dataset_cls,
            data_config=self.data_config,
            original_multi_modal_inputs=multi_modal_inputs_batch,
            crop_image_refs=micro_batch.get(CROP_IMAGE_REFS_KEY),
            assert_metadata_match=self.lazy_multi_modal_assert_metadata_match,
        )
        return extract_multi_modal_inputs(rebuilt_multi_modal_inputs)

    @actor_stage
    def _forward_micro_batch(
        self, micro_batch: dict[str, torch.Tensor], temperature: float, calculate_entropy: bool = False
    ) -> dict[str, torch.Tensor]:
        """
        Returns:
            dict[str, torch.Tensor]:
                log_probs: (bs, response_len)
                if calculate_entropy is True:
                    entropys: (bs, response_len)
                if calculate_sum_pi_squared is False:
                    sum_pi_squared: (bs, response_len)
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)
        sum_pi_squared_checkpointing = self.config.get("sum_pi_squared_checkpointing", False)
        # PrefixGrouper path for shared-prefix optimization
        if self.use_prefix_grouper:
            can_use_pg = (
                not self.use_remove_padding
                and not self.use_ulysses_sp
                and not self.use_fused_kernels
                and not self.use_dynamic_bsz
            )
            if can_use_pg and "response_mask" in micro_batch and "uid" in micro_batch:
                from verl.trainer.ppo.prefix_grouper_utils import forward_micro_batch_with_prefix_grouper

                return forward_micro_batch_with_prefix_grouper(
                    micro_batch=micro_batch,
                    model=self.actor_module,
                    temperature=temperature,
                    calculate_entropy=calculate_entropy,
                    device_name=self.device_name,
                    param_dtype=self.param_dtype,
                    use_chunking_entropy=self.config.get("entropy_from_logits_with_chunking", False),
                )

        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = self._extract_multi_modal_inputs(micro_batch)

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                    entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                else:
                    logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                    if temperature != 1.0:
                        logits_rmpad = logits_rmpad / temperature

                    # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                    inplace_backward = True
                    if calculate_entropy:
                        inplace_backward = False
                    log_probs = logprobs_from_logits(
                        logits=logits_rmpad,
                        labels=input_ids_rmpad_rolled,
                        inplace_backward=inplace_backward,
                    )

                    # compute entropy
                    if calculate_entropy:
                        # ((total_nnz / sp) + pad)
                        entropy_rmpad = (
                            self.compute_entropy_from_logits(logits_rmpad)
                            if not self.config.entropy_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.compute_entropy_from_logits, logits_rmpad)
                        )

                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = (
                            self.calculate_sum_pi_squared_from_logits(logits_rmpad)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(
                                self.calculate_sum_pi_squared_from_logits, logits_rmpad
                            )
                        )

                # gather log_prob if sp > 1
                if self.use_ulysses_sp:
                    # gather and unpad for the ulysses sp
                    log_probs = gather_outputs_and_unpad(
                        log_probs,
                        gather_dim=0,
                        unpad_dim=0,
                        padding_size=pad_size,
                    )
                    if calculate_entropy:
                        entropy_rmpad = gather_outputs_and_unpad(
                            entropy_rmpad,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                    if calculate_sum_pi_squared:
                        sum_pi_squared_rmpad = gather_outputs_and_unpad(
                            sum_pi_squared_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size
                        )

                if is_mask_all_zero:
                    log_probs = log_probs[:0]
                    if calculate_entropy:
                        entropy_rmpad = entropy_rmpad[:0]

                # pad back to (bsz, seqlen)
                if calculate_entropy:
                    full_entropy = pad_input(
                        hidden_states=entropy_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                if calculate_sum_pi_squared:
                    full_sum_pi_squared = pad_input(
                        hidden_states=sum_pi_squared_rmpad.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                full_log_probs = pad_input(
                    hidden_states=log_probs.unsqueeze(-1),
                    indices=indices,
                    batch=batch_size,
                    seqlen=seqlen,
                )

                # only return response part:
                if calculate_entropy:
                    entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                if calculate_sum_pi_squared:
                    # (bsz, response_length)
                    sum_pi_squared = full_sum_pi_squared.squeeze(-1)[:, -response_length - 1 : -1]
                log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)

            else:  # not using rmpad and no ulysses sp
                if self.config.fsdp_config.get("gemma4_policy_chunk_size", 0):
                    if self.use_fused_kernels or self.use_ulysses_sp or calculate_sum_pi_squared:
                        raise ValueError("Gemma chunked policy requires unfused SP=1 without sum_pi_squared")
                    if self.config.fsdp_config.get("gemma4_trim_padding", False):
                        from verl.models.gemma4.trimming import validate_loss_masks

                        validate_loss_masks(micro_batch)
                    output = self.actor_module(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        position_ids=position_ids,
                        **multi_modal_inputs,
                        use_cache=False,
                        gemma4_response_labels=micro_batch["responses"],
                        gemma4_temperature=temperature,
                        gemma4_calculate_entropy=calculate_entropy,
                    )
                    result = {"log_probs": output["log_probs"]}
                    if calculate_entropy:
                        result["entropys"] = output["entropys"]
                    return result
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    if temperature != 1.0:
                        logits = logits / temperature
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)
                    # Compute sum_pi_squared if requested (for optimal_token_baseline)
                    if calculate_sum_pi_squared:
                        sum_pi_squared = (
                            self.calculate_sum_pi_squared_from_logits(logits)
                            if not sum_pi_squared_checkpointing
                            else torch.utils.checkpoint.checkpoint(self.calculate_sum_pi_squared_from_logits, logits)
                        )

            outputs = {"log_probs": log_probs}
            if calculate_entropy:
                outputs["entropys"] = entropy
            if calculate_sum_pi_squared:
                outputs["sum_pi_squared"] = sum_pi_squared
            return outputs

    @actor_stage
    def _optimizer_step(self):
        gemma_probe = None
        if self.config.get("fsdp_config", {}).get("gemma4_smoke_diagnostics", False):
            from verl.models.gemma4.training_diagnostics import before_optimizer_step

            gemma_probe = before_optimizer_step(self.actor_module)
        with stage("gradient_clipping"):
            assert self.config.grad_clip is not None
            if self.scaler is not None:
                self.scaler.unscale_(self.actor_optimizer)
            if isinstance(self.actor_module, FSDP):
                grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
            elif isinstance(self.actor_module, FSDPModule):
                grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
            else:
                grad_norm = torch.nn.utils.clip_grad_norm_(
                    self.actor_module.parameters(), max_norm=self.config.grad_clip
                )

            if isinstance(grad_norm, DTensor):
                grad_norm = grad_norm.full_tensor()

        gradient_probe = getattr(self, "_active_gradient_probe", None)
        if gradient_probe is not None:
            probe_start = time.perf_counter()
            gradient_probe.after_clip(grad_norm.item(), self.config.grad_clip)
            gradient_probe.metrics["postclip_measurement_seconds"] = time.perf_counter() - probe_start

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                with stage("optimizer_step"):
                    self.actor_optimizer.step()

        # Clear cached weight scales for QAT (weights changed)
        if getattr(self.actor_module, "_qat_fuse_enabled", False):
            from verl.utils.qat import invalidate_all_scales

            invalidate_all_scales(self.actor_module)

        if gemma_probe is not None:
            from verl.models.gemma4.training_diagnostics import after_optimizer_step

            self._gemma4_smoke_metrics = after_optimizer_step(gemma_probe)
        return grad_norm

    def _gradient_replay_loss(
        self, log_prob, inputs, *, on_policy, bounded, critic_multiplier,
        has_rl, has_critic, dp_size, critic_parent_mass, dataset_balance, stratum_parent_mass,
    ):
        """Local loss construction; no distributed operations or optimizer state."""
        old = (log_prob.detach() if on_policy and not self.config.get("use_rollout_log_probs", False)
               else inputs["old_log_probs"])
        gspo = {}
        get_policy_loss_fn("gspo")(
            old_log_prob=old, log_prob=log_prob, advantages=inputs["advantages"],
            response_mask=inputs["response_mask"], config=self.config,
            rollout_is_weights=inputs.get("rollout_is_weights"),
            branch_policy_loss_weight=inputs["branch_policy_loss_weight"],
            branch_policy_loss_is_branch=inputs["branch_policy_loss_is_branch"], loss_components=gspo,
        )
        ce = log_prob.sum() * 0.0
        if has_critic:
            if dataset_balance:
                ce, *_ = parent_balanced_retry_critic_strata_loss(
                    log_prob, inputs["retry_critic_mask"], inputs["retry_critic_stratum_id"],
                    inputs["retry_critic_parent_weight"],
                    global_stratum_parent_mass=stratum_parent_mass, dp_size=dp_size,
                )
            else:
                ce, *_ = parent_balanced_retry_critic_loss(
                    log_prob, inputs["retry_critic_mask"], inputs["retry_critic_is_retry"],
                    inputs["retry_critic_parent_weight"],
                    global_accept_parent_mass=critic_parent_mass[0],
                    global_retry_parent_mass=critic_parent_mass[1], dp_size=dp_size,
                )
        loss_scale = (inputs["response_mask"].shape[0] / self.config.ppo_mini_batch_size
                      if self.config.use_dynamic_bsz or bounded else 1 / self.gradient_accumulation)
        return weighted_loss_components(
            gspo, ce, loss_scale=loss_scale,
            critic_fraction=float(self.config.policy_loss.retry_critic.loss_fraction),
            has_rl=has_rl, has_critic=has_critic, bounded=bounded, critic_multiplier=critic_multiplier,
        )

    def _replay_gradient_components(
        self, micro_batches, *, temperature, pad_token_id, on_policy, bounded,
        critic_multiplier, has_rl, has_critic, dp_size, critic_parent_mass,
        dataset_balance, stratum_parent_mass,
    ):
        """Five isolated backwards; no optimizer/scaler updates or retained graphs."""
        settings = self.config.policy_loss.loss_diagnostics
        probe = GradientProbe(self.actor_module, get_device_id(), settings.gradient_reconstruction_rtol)
        scale = self.scaler.get_scale() if self.scaler is not None else 1.0
        started = time.perf_counter()
        for name in COMPONENTS:
            self.actor_optimizer.zero_grad(set_to_none=True)
            local_loss = torch.zeros((), dtype=torch.float64, device=get_device_id())
            ok = True
            with preserved_rng(get_device_id()):
                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    output = self._forward_micro_batch(
                        inputs, temperature=temperature, calculate_entropy=self.config.calculate_entropy,
                    )
                    log_prob = output["log_probs"]
                    loss, ok = probe.local_call(
                        lambda log_prob=log_prob, inputs=inputs, name=name: self._gradient_replay_loss(
                            log_prob, inputs, on_policy=on_policy, bounded=bounded,
                            critic_multiplier=critic_multiplier, has_rl=has_rl, has_critic=has_critic,
                            dp_size=dp_size, critic_parent_mass=critic_parent_mass,
                            dataset_balance=dataset_balance, stratum_parent_mass=stratum_parent_mass,
                        )[name], f"loss_{name}",
                    )
                    if not ok:
                        # Complete a zero backward to finish FSDP/checkpoint hooks
                        # for this forward before doing the ordinary real pass.
                        (log_prob.sum() * 0.0).backward()
                        del output, log_prob, loss
                        break
                    local_loss += loss.detach().double()
                    (self.scaler.scale(loss) if self.scaler is not None else loss).backward()
                    del output, log_prob, loss
            if not ok:
                break
            if torch.distributed.is_initialized():
                torch.distributed.all_reduce(local_loss)
            _, ok = probe.local_call(
                lambda local_loss=local_loss: probe.capture(local_loss.item() / dp_size, scale=scale),
                f"capture_{name}",
            )
            if not ok:
                # All ranks stop replay together; the ordinary backward below
                # starts from clean gradients and the restored RNG state.
                break
        self.actor_optimizer.zero_grad(set_to_none=True)
        probe.metrics["replay_seconds"] = time.perf_counter() - started
        return probe

    @torch.no_grad()
    def _measure_current_retry_critic_ce(
        self, micro_batches, temperature, pad_token_id, dp_size,
        global_critic_parent_mass, dataset_balance=False, global_stratum_parent_mass=None,
    ):
        """Current-policy CE with the same global class/parent reduction as backward.

        A no-grad prepass avoids retaining all microbatch graphs. Restore RNG so
        the ordinary forward/backward pass sees the same stochastic forwards.
        The actual accumulated CE is independently audited before the update.
        """
        total = torch.zeros((), dtype=torch.float64, device=get_device_id())
        devices = [torch.cuda.current_device()] if self.device_name == "cuda" else []
        with torch.random.fork_rng(devices=devices):
            for micro_batch in micro_batches:
                micro_batch = micro_batch.to(get_device_id())
                inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                outputs = self._forward_micro_batch(inputs, temperature=temperature, calculate_entropy=False)
                if dataset_balance:
                    ce, _, _ = parent_balanced_retry_critic_strata_loss(
                        outputs["log_probs"], inputs["retry_critic_mask"],
                        inputs["retry_critic_stratum_id"], inputs["retry_critic_parent_weight"],
                        global_stratum_parent_mass=global_stratum_parent_mass, dp_size=dp_size,
                    )
                else:
                    ce, *_ = parent_balanced_retry_critic_loss(
                        outputs["log_probs"], inputs["retry_critic_mask"],
                        inputs["retry_critic_is_retry"], inputs["retry_critic_parent_weight"],
                        global_accept_parent_mass=global_critic_parent_mass[0],
                        global_retry_parent_mass=global_critic_parent_mass[1], dp_size=dp_size,
                    )
                total += ce.double()
                del outputs, ce
        if torch.distributed.is_initialized():
            torch.distributed.all_reduce(total)
        return float(total.item() / dp_size)

    @GPUMemoryLogger(role="dp actor", logger=logger)
    @profile_actor
    def compute_log_prob(self, data: DataProto, calculate_entropy: bool = False) -> dict[str, torch.Tensor]:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            dict[str, torch.Tensor]: a dict containing keys
                - ``log_probs``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``entropys``: tensor of shape [batch_size, response_length]. torch.float32.
                - ``sum_pi_squared``: tensor of shape [batch_size, response_length]. torch.float32.
        """
        calculate_sum_pi_squared = self.config.get("calculate_sum_pi_squared", False)

        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        pad_token_id = data.meta_info.get("pad_token_id", 0)
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []
        if self.lazy_multi_modal_inputs_enabled and has_multi_modal_inputs:
            if "raw_prompt" not in data.non_tensor_batch:
                raise ValueError("lazy multimodal inputs require raw_prompt in the rollout batch")
            non_tensor_select_keys.append("raw_prompt")
            if CROP_IMAGE_REFS_KEY in data.non_tensor_batch:
                non_tensor_select_keys.append(CROP_IMAGE_REFS_KEY)
        if self.use_prefix_grouper:
            select_keys += [k for k in ["prompts", "response_mask"] if k in data.batch]
            if "uid" in data.non_tensor_batch:
                non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(
                data, max_token_len=max_token_len, dp_group=torch.distributed.group.WORLD
            )
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        sum_pi_squared_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())
            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
            with torch.no_grad():
                outputs = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )
            log_probs_lst.append(outputs["log_probs"])
            if calculate_entropy:
                entropy_lst.append(outputs["entropys"])
            if calculate_sum_pi_squared:
                sum_pi_squared_lst.append(outputs["sum_pi_squared"])

        log_probs = torch.concat(log_probs_lst, dim=0)
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)
        if calculate_sum_pi_squared:
            sum_pi_squared = torch.concat(sum_pi_squared_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)
            if calculate_sum_pi_squared:
                sum_pi_squared = restore_dynamic_batch(sum_pi_squared, batch_idx_list)

        outputs = {"log_probs": log_probs}
        if calculate_entropy:
            outputs["entropys"] = entropys
        if calculate_sum_pi_squared:
            outputs["sum_pi_squared"] = sum_pi_squared
        return outputs

    @GPUMemoryLogger(role="dp actor", logger=logger)
    @profile_actor
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        pad_token_id = data.meta_info.get("pad_token_id", 0)

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.use_prefix_grouper and "prompts" in data.batch.keys():
            select_keys.append("prompts")
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")
        branch_strata_enabled = bool(self.config.policy_loss.branch_strata.enabled)
        mass_bounds = self.config.policy_loss.get("auxiliary_mass_bounds", {})
        mass_bounds_enabled = bool(mass_bounds.get("enabled", False))
        diagnostics = self.config.policy_loss.get("loss_diagnostics", {})
        diagnostics_enabled = bool(diagnostics.get("enabled", False))
        gradient_step = gradient_step_selected(
            diagnostics, data.meta_info.get("global_steps"),
            not getattr(self, "_gradient_diagnostics_seen_update", False),
        )
        gradient_step = gradient_step and not getattr(self, "_gradient_diagnostics_disabled", False)
        if gradient_step and (self.ulysses_sequence_parallel_size != 1 or isinstance(self.actor_module, FSDP)):
            raise ValueError("Gradient diagnostics support FSDP2 with SP=1, or an unsharded actor.")
        if diagnostics_enabled:
            if (not branch_strata_enabled or self.config.policy_loss.loss_mode != "gspo"
                    or self.config.policy_loss.retry_sft.enabled
                    or self.config.entropy_coeff != 0 or self.config.use_kl_loss):
                raise ValueError("Five-component diagnostics require branch GSPO without SFT/entropy/KL.")
        if mass_bounds_enabled:
            if not branch_strata_enabled or self.config.policy_loss.loss_mode != "gspo":
                raise ValueError("Auxiliary mass bounds require branch-stratified GSPO.")
            if not self.config.policy_loss.retry_critic.enabled or self.config.policy_loss.retry_sft.enabled:
                raise ValueError("Auxiliary mass bounds require retry critic and no retry-SFT.")
            if self.config.entropy_coeff != 0 or self.config.use_kl_loss or self.ulysses_sequence_parallel_size != 1:
                raise ValueError("Auxiliary mass bounds currently require zero entropy/KL losses and SP=1.")
        if branch_strata_enabled:
            for key in ("branch_policy_loss_weight", "branch_policy_loss_is_branch"):
                if key not in data.batch.keys():
                    raise ValueError(f"Enabled branch-stratified policy loss requires tensor batch key {key!r}.")
                select_keys.append(key)
        retry_sft_enabled = bool(self.config.policy_loss.retry_sft.enabled)
        if retry_sft_enabled:
            if "retry_sft_mask" not in data.batch.keys():
                raise ValueError("Enabled retry SFT requires tensor batch key 'retry_sft_mask'.")
            select_keys.append("retry_sft_mask")
        retry_critic_enabled = bool(self.config.policy_loss.retry_critic.enabled)
        retry_critic_dataset_balance_config = self.config.policy_loss.retry_critic.get("dataset_balance", {})
        retry_critic_dataset_balance = bool(
            retry_critic_enabled
            and retry_critic_dataset_balance_config
            and retry_critic_dataset_balance_config.get("enabled", False)
        )
        retry_critic_stratum_names = tuple(data.meta_info.get("retry_critic_stratum_names", ()))
        if retry_critic_enabled:
            for key in ("retry_critic_mask", "retry_critic_is_retry", "retry_critic_parent_weight"):
                if key not in data.batch.keys():
                    raise ValueError(f"Enabled retry critic requires tensor batch key {key!r}.")
                select_keys.append(key)
            if retry_critic_dataset_balance:
                if "retry_critic_stratum_id" not in data.batch.keys():
                    raise ValueError(
                        "Dataset-balanced retry critic requires tensor batch key 'retry_critic_stratum_id'."
                    )
                if not retry_critic_stratum_names:
                    raise ValueError(
                        "Dataset-balanced retry critic requires nonempty retry_critic_stratum_names metadata."
                    )
                select_keys.append("retry_critic_stratum_id")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = []
        if has_multi_modal_inputs:
            non_tensor_select_keys.append("multi_modal_inputs")
        if self.lazy_multi_modal_inputs_enabled and has_multi_modal_inputs:
            if "raw_prompt" not in data.non_tensor_batch:
                raise ValueError("lazy multimodal inputs require raw_prompt in the rollout batch")
            non_tensor_select_keys.append("raw_prompt")
            if CROP_IMAGE_REFS_KEY in data.non_tensor_batch:
                non_tensor_select_keys.append(CROP_IMAGE_REFS_KEY)
        if self.use_prefix_grouper and "uid" in data.non_tensor_batch.keys():
            non_tensor_select_keys.append("uid")

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
            "actor/retry_sft_loss": 0.0,
            "actor/retry_sft_weighted_loss": 0.0,
            "actor/retry_critic_loss": 0.0,
            "actor/retry_critic_accept_nll": 0.0,
            "actor/retry_critic_retry_nll": 0.0,
            "actor/retry_critic_weighted_loss": 0.0,
            "actor/combined_policy_loss": 0.0,
        }
        bound_ratios = {"branch": [], "critic": []}
        if mass_bounds_enabled:
            for term in ("canonical_gspo", "branch_gspo", "critic_ce"):
                metrics[f"actor/mass_bounds/{term}_contribution_sum"] = 0.0
        measured_minibatches = 0
        for ppo_epoch in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if diagnostics_enabled:
                    diagnostic_totals = torch.zeros(7, dtype=torch.float64, device=get_device_id())
                    scalar_diagnostic_errors = []
                if mass_bounds_enabled:
                    # Keep the original weights for the next PPO epoch.
                    mini_batch = mini_batch.select(batch_keys=list(mini_batch.batch.keys()))
                dp_size = torch.distributed.get_world_size() if torch.distributed.is_initialized() else 1
                global_retry_sft_tokens = 0
                global_rl_tokens = 0
                if retry_sft_enabled or retry_critic_enabled:
                    token_counts = torch.tensor(
                        [
                            int(mini_batch.batch["retry_sft_mask"].sum().item()) if retry_sft_enabled else 0,
                            int(mini_batch.batch["response_mask"].sum().item()),
                        ],
                        dtype=torch.long,
                        device=get_device_id(),
                    )
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(token_counts, op=torch.distributed.ReduceOp.SUM)
                    global_retry_sft_tokens = int(token_counts[0].item())
                    global_rl_tokens = int(token_counts[1].item())
                global_critic_examples = [0, 0]
                global_critic_parent_mass = [0.0, 0.0]
                global_critic_stratum_examples: list[int] = []
                global_critic_stratum_parent_mass: list[float] = []
                if retry_critic_enabled:
                    critic_rows = mini_batch.batch["retry_critic_mask"].to(dtype=torch.bool).any(dim=-1)
                    critic_is_retry = mini_batch.batch["retry_critic_is_retry"].to(dtype=torch.bool)
                    critic_parent_weight = mini_batch.batch["retry_critic_parent_weight"].to(dtype=torch.float32)
                    if retry_critic_dataset_balance:
                        num_strata = len(retry_critic_stratum_names)
                        critic_stratum_id = mini_batch.batch["retry_critic_stratum_id"].to(dtype=torch.long)
                        invalid_rows = critic_rows & ((critic_stratum_id < 0) | (critic_stratum_id >= num_strata))
                        if bool(invalid_rows.any().item()):
                            raise ValueError("Active retry critic rows contain invalid dataset stratum ids.")
                        active_stratum_id = critic_stratum_id[critic_rows]
                        stratum_examples = torch.bincount(active_stratum_id, minlength=num_strata).to(
                            dtype=torch.float64
                        )
                        stratum_parent_mass = torch.bincount(
                            active_stratum_id,
                            weights=critic_parent_weight[critic_rows].to(dtype=torch.float64),
                            minlength=num_strata,
                        )
                        critic_stats = torch.cat((stratum_examples, stratum_parent_mass)).to(device=get_device_id())
                    else:
                        critic_stats = torch.tensor(
                            [
                                int((critic_rows & ~critic_is_retry).sum().item()),
                                int((critic_rows & critic_is_retry).sum().item()),
                                float((critic_parent_weight * (critic_rows & ~critic_is_retry)).sum().item()),
                                float((critic_parent_weight * (critic_rows & critic_is_retry)).sum().item()),
                            ],
                            dtype=torch.float64,
                            device=get_device_id(),
                        )
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(critic_stats, op=torch.distributed.ReduceOp.SUM)
                    if retry_critic_dataset_balance:
                        num_strata = len(retry_critic_stratum_names)
                        global_critic_stratum_examples = [int(value) for value in critic_stats[:num_strata].tolist()]
                        global_critic_stratum_parent_mass = [
                            float(value) for value in critic_stats[num_strata:].tolist()
                        ]
                        global_critic_examples = [
                            sum(global_critic_stratum_examples[0::2]),
                            sum(global_critic_stratum_examples[1::2]),
                        ]
                        global_critic_parent_mass = [
                            sum(global_critic_stratum_parent_mass[0::2]),
                            sum(global_critic_stratum_parent_mass[1::2]),
                        ]
                    else:
                        global_critic_examples = [int(critic_stats[0].item()), int(critic_stats[1].item())]
                        global_critic_parent_mass = [float(critic_stats[2].item()), float(critic_stats[3].item())]
                branch_multiplier = critic_multiplier = 1.0
                if mass_bounds_enabled:
                    epsilon = float(mass_bounds.epsilon)
                    configured_critic_fraction = float(self.config.policy_loss.retry_critic.loss_fraction)
                    policy_fraction = 1.0 - configured_critic_fraction
                    global_normalizer = self.config.ppo_mini_batch_size * dp_size
                    mass_args = (
                        mini_batch.batch["advantages"], mini_batch.batch["response_mask"],
                        mini_batch.batch["branch_policy_loss_weight"],
                        mini_batch.batch["branch_policy_loss_is_branch"],
                        mini_batch.batch.get("rollout_is_weights"),
                    )
                    mass = advantage_mass_sums(*mass_args).to(get_device_id())
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(mass)
                    mass *= policy_fraction / global_normalizer
                    canonical_mass = float((mass[0] + mass[1]).item())
                    branch_mass_before = float((mass[2] + mass[3]).item())
                    branch_multiplier = detached_cap_multiplier(
                        canonical_mass, branch_mass_before, mass_bounds.max_branch_to_canonical, epsilon,
                    )
                    branch_flags = mini_batch.batch["branch_policy_loss_is_branch"].bool()
                    original_weights = mini_batch.batch["branch_policy_loss_weight"]
                    mini_batch.batch["branch_policy_loss_weight"] = torch.where(
                        branch_flags, original_weights * branch_multiplier, original_weights,
                    )
                    mass_after = advantage_mass_sums(
                        mini_batch.batch["advantages"], mini_batch.batch["response_mask"],
                        mini_batch.batch["branch_policy_loss_weight"], branch_flags,
                        mini_batch.batch.get("rollout_is_weights"),
                    ).to(get_device_id())
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(mass_after)
                    mass_after *= policy_fraction / global_normalizer
                    branch_mass_after = float((mass_after[2] + mass_after[3]).item())
                    branch_limit = mass_bounds.max_branch_to_canonical * canonical_mass
                    if branch_mass_after > branch_limit + mass_bounds.verification_atol:
                        raise RuntimeError("Branch absolute advantage mass exceeded its configured cap.")

                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(
                        mini_batch, max_token_len=max_token_len, dp_group=torch.distributed.group.WORLD
                    )
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                if mass_bounds_enabled:
                    has_global_critic = any(global_critic_examples)
                    critic_ce_before = self._measure_current_retry_critic_ce(
                        micro_batches, temperature, pad_token_id, dp_size, global_critic_parent_mass,
                        dataset_balance=retry_critic_dataset_balance,
                        global_stratum_parent_mass=global_critic_stratum_parent_mass
                        if retry_critic_dataset_balance else None,
                    ) if has_global_critic else 0.0
                    critic_mass_before = configured_critic_fraction * critic_ce_before
                    critic_multiplier = detached_cap_multiplier(
                        canonical_mass, critic_mass_before, mass_bounds.max_critic_to_canonical, epsilon,
                    )
                    actual_bound_terms = torch.zeros(3, dtype=torch.float64, device=get_device_id())

                probe = None
                if gradient_step and (diagnostics.gradient_max_minibatches == 0
                                      or measured_minibatches < diagnostics.gradient_max_minibatches):
                    probe = self._replay_gradient_components(
                        micro_batches, temperature=temperature, pad_token_id=pad_token_id,
                        on_policy=on_policy, bounded=mass_bounds_enabled, critic_multiplier=critic_multiplier,
                        has_rl=global_rl_tokens > 0, has_critic=any(global_critic_examples), dp_size=dp_size,
                        critic_parent_mass=global_critic_parent_mass, dataset_balance=retry_critic_dataset_balance,
                        stratum_parent_mass=global_critic_stratum_parent_mass,
                    )
                    actual_backward_start = time.perf_counter()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())
                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch, "pad_token_id": pad_token_id}
                    response_mask = model_inputs["response_mask"]
                    retry_sft_mask = model_inputs.get("retry_sft_mask", None)
                    if retry_sft_mask is not None and torch.any(
                        response_mask.to(dtype=torch.bool) & retry_sft_mask.to(dtype=torch.bool)
                    ):
                        raise ValueError("response_mask and retry_sft_mask must be disjoint")
                    retry_critic_mask = model_inputs.get("retry_critic_mask", None)
                    if retry_critic_mask is not None:
                        critic_bool = retry_critic_mask.to(dtype=torch.bool)
                        if torch.any(response_mask.to(dtype=torch.bool) & critic_bool):
                            raise ValueError("response_mask and retry_critic_mask must be disjoint")
                        if retry_sft_mask is not None and torch.any(retry_sft_mask.to(dtype=torch.bool) & critic_bool):
                            raise ValueError("retry_sft_mask and retry_critic_mask must be disjoint")
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz or mass_bounds_enabled:
                        # The budget uses this same nominal row denominator,
                        # including a partial final fixed-size microbatch.
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    outputs = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )
                    log_prob = outputs["log_probs"]
                    entropy = outputs["entropys"] if calculate_entropy else None

                    with stage("loss_construction"):
                        # for fully_async_policy
                        if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                            old_log_prob = model_inputs["old_log_probs"]
                        else:
                            if on_policy:
                                old_log_prob = log_prob.detach()
                            else:
                                old_log_prob = model_inputs["old_log_probs"]

                        loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                        # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                        # Extract pre-computed rollout correction weights if present
                        # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                        rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                        # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                        # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                        policy_loss_fn = get_policy_loss_fn(loss_mode)

                        branch_policy_loss_weight = model_inputs.get("branch_policy_loss_weight", None)
                        branch_policy_loss_is_branch = model_inputs.get("branch_policy_loss_is_branch", None)

                        # Compute policy loss (any function is expected to return 2 values)
                        policy_loss_kwargs = {}
                        gspo_components = {} if diagnostics_enabled else None
                        if diagnostics_enabled:
                            policy_loss_kwargs["loss_components"] = gspo_components
                        if branch_strata_enabled:
                            policy_loss_kwargs.update(
                                branch_policy_loss_weight=branch_policy_loss_weight,
                                branch_policy_loss_is_branch=branch_policy_loss_is_branch,
                            )
                        pg_loss, pg_metrics = policy_loss_fn(
                            old_log_prob=old_log_prob,
                            log_prob=log_prob,
                            advantages=advantages,
                            response_mask=response_mask,
                            loss_agg_mode=loss_agg_mode,
                            config=self.config,
                            rollout_is_weights=rollout_is_weights,
                            **policy_loss_kwargs,
                        )
                        if mass_bounds_enabled:
                            actual_bound_terms[0] += (
                                pg_metrics.pop("actor/pg_loss_canonical_weighted") * loss_scale_factor * policy_fraction
                            )
                            actual_bound_terms[1] += (
                                pg_metrics.pop("actor/pg_loss_branch_weighted") * loss_scale_factor * policy_fraction
                            )
                        micro_batch_metrics.update(pg_metrics)

                        # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                        rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                        if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                            # Compute metrics using CURRENT policy π_θ vs π_rollout
                            # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                            from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                            rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                                log_prob=log_prob,
                                rollout_log_prob=rollout_log_prob,
                                response_mask=response_mask,
                            )
                            micro_batch_metrics.update(rollout_corr_metrics)

                        policy_loss = pg_loss
                        if calculate_entropy and entropy is not None:
                            entropy_agg = agg_loss(
                                loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode
                            )
                            micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                            if entropy_coeff != 0:
                                policy_loss -= entropy_agg * entropy_coeff

                        if self.config.use_kl_loss:
                            ref_log_prob = model_inputs["ref_log_prob"]
                            # compute kl loss
                            kld = kl_penalty(
                                logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                            )
                            kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                            policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                            metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                            micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                        rl_loss_for_backward = policy_loss * loss_scale_factor
                        retry_sft_loss = log_prob.sum() * 0.0
                        if retry_sft_enabled and global_retry_sft_tokens > 0:
                            retry_sft_loss = masked_sft_nll(
                                log_prob,
                                retry_sft_mask.to(dtype=torch.bool),
                                batch_num_tokens=global_retry_sft_tokens,
                                dp_size=dp_size,
                            )
                            retry_sft_fraction = float(self.config.policy_loss.retry_sft.loss_fraction)
                            loss = combine_retry_sft_loss(
                                rl_loss_for_backward,
                                retry_sft_loss,
                                loss_fraction=retry_sft_fraction,
                                has_rl=global_rl_tokens > 0,
                                has_retry_sft=True,
                            )
                        else:
                            retry_sft_fraction = 0.0
                            loss = combine_retry_sft_loss(
                                rl_loss_for_backward,
                                retry_sft_loss,
                                loss_fraction=float(self.config.policy_loss.retry_sft.loss_fraction)
                                if retry_sft_enabled
                                else 0.0,
                                has_rl=global_rl_tokens > 0 if retry_sft_enabled else True,
                                has_retry_sft=False,
                            )
                        retry_critic_loss = log_prob.sum() * 0.0
                        retry_critic_accept_nll = log_prob.sum() * 0.0
                        retry_critic_retry_nll = log_prob.sum() * 0.0
                        retry_critic_fraction = 0.0
                        has_critic = any(global_critic_examples)
                        if retry_critic_enabled and has_critic:
                            if retry_critic_dataset_balance:
                                retry_critic_loss, stratum_nll, present_strata = (
                                    parent_balanced_retry_critic_strata_loss(
                                        log_prob,
                                        retry_critic_mask,
                                        model_inputs["retry_critic_stratum_id"],
                                        model_inputs["retry_critic_parent_weight"],
                                        global_stratum_parent_mass=global_critic_stratum_parent_mass,
                                        dp_size=dp_size,
                                    )
                                )
                                accept_present = present_strata[0::2]
                                retry_present = present_strata[1::2]
                                if bool(accept_present.any().item()):
                                    retry_critic_accept_nll = stratum_nll[0::2][accept_present].mean()
                                if bool(retry_present.any().item()):
                                    retry_critic_retry_nll = stratum_nll[1::2][retry_present].mean()
                                for stratum_index, stratum_name in enumerate(retry_critic_stratum_names):
                                    if bool(present_strata[stratum_index].item()):
                                        metrics[f"actor/retry_critic_strata/{stratum_name}_nll"] = (
                                            metrics.get(f"actor/retry_critic_strata/{stratum_name}_nll", 0.0)
                                            + stratum_nll[stratum_index].detach().item()
                                        )
                            else:
                                (
                                    retry_critic_loss,
                                    retry_critic_accept_nll,
                                    retry_critic_retry_nll,
                                    _,
                                    _,
                                ) = parent_balanced_retry_critic_loss(
                                    log_prob,
                                    retry_critic_mask,
                                    model_inputs["retry_critic_is_retry"],
                                    model_inputs["retry_critic_parent_weight"],
                                    global_accept_parent_mass=global_critic_parent_mass[0],
                                    global_retry_parent_mass=global_critic_parent_mass[1],
                                    dp_size=dp_size,
                                )
                            retry_critic_fraction = float(self.config.policy_loss.retry_critic.loss_fraction)
                        if mass_bounds_enabled:
                            # Preserve canonical scaling even in a minibatch with
                            # no critic labels; never renormalize canonical upward.
                            loss = (policy_fraction * loss
                                    + configured_critic_fraction * critic_multiplier * retry_critic_loss)
                            actual_bound_terms[2] += retry_critic_loss.detach().double()
                        else:
                            loss = combine_retry_critic_loss(
                                loss,
                                retry_critic_loss,
                                loss_fraction=float(self.config.policy_loss.retry_critic.loss_fraction)
                                if retry_critic_enabled
                                else 0.0,
                                has_existing=(global_rl_tokens > 0 or global_retry_sft_tokens > 0),
                                has_critic=retry_critic_enabled and has_critic,
                            )
                        if diagnostics_enabled:
                            diagnostic_totals[5] += loss.detach().double()
                            try:
                                weighted_components = weighted_loss_components(
                                    gspo_components, retry_critic_loss, loss_scale=loss_scale_factor,
                                    critic_fraction=float(self.config.policy_loss.retry_critic.loss_fraction),
                                    has_rl=global_rl_tokens > 0, has_critic=has_critic,
                                    bounded=mass_bounds_enabled, critic_multiplier=critic_multiplier,
                                )
                                diagnostic_totals[:5] += torch.stack(
                                    [weighted_components[name].detach().double() for name in COMPONENTS]
                                )
                            except Exception as error:
                                # Optional scalar accounting must never replace
                                # or veto the already constructed real objective.
                                diagnostic_totals[6] += 1
                                scalar_diagnostic_errors.append(f"{type(error).__name__}: {error}")
                                logger.warning("Optional loss partition failed; training continues: %s", error)
                    with stage("backward"):
                        if self.scaler is not None:
                            self.scaler.scale(loss).backward()
                        else:
                            loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    metrics["actor/retry_sft_loss"] += retry_sft_loss.detach().item()
                    metrics["actor/retry_sft_weighted_loss"] += retry_sft_loss.detach().item() * retry_sft_fraction
                    metrics["actor/retry_critic_loss"] += retry_critic_loss.detach().item()
                    metrics["actor/retry_critic_accept_nll"] += retry_critic_accept_nll.detach().item()
                    metrics["actor/retry_critic_retry_nll"] += retry_critic_retry_nll.detach().item()
                    metrics["actor/retry_critic_weighted_loss"] += (
                        retry_critic_loss.detach().item() * retry_critic_fraction * critic_multiplier
                    )
                    metrics["actor/combined_policy_loss"] += loss.detach().item()
                    append_to_dict(metrics, micro_batch_metrics)

                if mass_bounds_enabled:
                    if torch.distributed.is_initialized():
                        torch.distributed.all_reduce(actual_bound_terms)
                    actual_bound_terms /= dp_size
                    if not torch.isfinite(actual_bound_terms).all():
                        self.actor_optimizer.zero_grad()
                        raise RuntimeError("Nonfinite bounded loss contribution; no optimizer update was applied.")
                    actual_critic_before = float(actual_bound_terms[2].item()) * configured_critic_fraction
                    actual_critic_after = actual_critic_before * critic_multiplier
                    critic_limit = mass_bounds.max_critic_to_canonical * canonical_mass
                    if actual_critic_after > critic_limit + mass_bounds.verification_atol:
                        self.actor_optimizer.zero_grad()
                        raise RuntimeError("Actual critic CE exceeded its cap; no optimizer update was applied.")
                    prefix = "actor/mass_bounds"
                    bound_metrics = {
                        f"{prefix}/enabled": 1.0,
                        f"{prefix}/canonical_abs_mass": canonical_mass,
                        f"{prefix}/canonical_positive_mass": float(mass[0].item()),
                        f"{prefix}/canonical_negative_mass": float(mass[1].item()),
                        f"{prefix}/canonical_zero": float(canonical_mass == 0),
                        f"{prefix}/branch_abs_mass_before": branch_mass_before,
                        f"{prefix}/branch_abs_mass_after": branch_mass_after,
                        f"{prefix}/branch_multiplier": branch_multiplier,
                        f"{prefix}/branch_cap_active": float(branch_multiplier < 1),
                        f"{prefix}/branch_limit": float(mass_bounds.max_branch_to_canonical),
                        f"{prefix}/branch_to_canonical_before": mass_ratio(branch_mass_before, canonical_mass, epsilon),
                        f"{prefix}/branch_to_canonical_after": mass_ratio(branch_mass_after, canonical_mass, epsilon),
                        f"{prefix}/critic_ce_prepass": critic_ce_before,
                        f"{prefix}/critic_ce_mass_prepass": critic_mass_before,
                        f"{prefix}/critic_ce_mass_before": actual_critic_before,
                        f"{prefix}/critic_ce_mass_after": actual_critic_after,
                        f"{prefix}/critic_multiplier": critic_multiplier,
                        f"{prefix}/critic_cap_active": float(critic_multiplier < 1),
                        f"{prefix}/critic_limit": float(mass_bounds.max_critic_to_canonical),
                        f"{prefix}/critic_effective_coefficient": configured_critic_fraction * critic_multiplier,
                        f"{prefix}/critic_to_canonical_before": mass_ratio(
                            actual_critic_before, canonical_mass, epsilon
                        ),
                        f"{prefix}/critic_to_canonical_after": mass_ratio(actual_critic_after, canonical_mass, epsilon),
                        f"{prefix}/critic_prepass_absolute_error": abs(actual_critic_before - critic_mass_before),
                        f"{prefix}/canonical_gspo_contribution": float(actual_bound_terms[0].item()),
                        f"{prefix}/branch_gspo_contribution": float(actual_bound_terms[1].item()),
                        f"{prefix}/critic_ce_contribution": actual_critic_after,
                    }
                    total_mass = canonical_mass + branch_mass_after + actual_critic_after
                    for term, value in (("canonical", canonical_mass), ("branch", branch_mass_after),
                                        ("critic", actual_critic_after)):
                        bound_metrics[f"{prefix}/{term}_mass_share_after"] = mass_ratio(value, total_mass, epsilon)
                    append_to_dict(metrics, bound_metrics)
                    for key, value in (("canonical_gspo", actual_bound_terms[0].item()),
                                       ("branch_gspo", actual_bound_terms[1].item()),
                                       ("critic_ce", actual_critic_after)):
                        metrics[f"{prefix}/{key}_contribution_sum"] += float(value)
                    bound_ratios["branch"].append(bound_metrics[f"{prefix}/branch_to_canonical_after"])
                    bound_ratios["critic"].append(bound_metrics[f"{prefix}/critic_to_canonical_after"])

                if diagnostics_enabled:
                    loss_diagnostics = reduce_loss_diagnostics(diagnostic_totals)
                    for name, value in loss_diagnostics.items():
                        key = f"actor/loss_components/{name}_sum"
                        if name == "reconstruction_abs_error":
                            key = "actor/loss_components/reconstruction_abs_error_max"
                            metrics[key] = max(metrics.get(key, 0.0), value)
                        elif name in ("finite", "computation_valid", "reconstruction_valid"):
                            key = f"actor/loss_components/{name}_min"
                            metrics[key] = min(metrics.get(key, 1.0), value)
                        else:
                            metrics[key] = metrics.get(key, 0.0) + value

                if probe is not None:
                    actual_seconds = time.perf_counter() - actual_backward_start
                    replay_seconds = probe.metrics["replay_seconds"]
                    probe.compare_losses(loss_diagnostics)
                    probe_start = time.perf_counter()
                    probe.measure_before_clip(self.scaler.get_scale() if self.scaler is not None else 1.0)
                    for term, multiplier in (("branch", branch_multiplier), ("critic_ce", critic_multiplier)):
                        norm = probe.metrics.get(f"{term}_norm_preclip")
                        probe.metrics[f"{term}_uncapped_norm_defined"] = float(multiplier > 0 and norm is not None)
                        probe.metrics[f"{term}_norm_before_auxiliary_multiplier"] = (
                            norm / multiplier if multiplier > 0 and norm is not None else 0.0
                        )
                    probe.metrics["gram_seconds"] = time.perf_counter() - probe_start
                    probe.metrics["replay_seconds"] = replay_seconds
                    probe.metrics["actual_forward_backward_seconds"] = actual_seconds
                    self._active_gradient_probe = probe
                try:
                    grad_norm = self._optimizer_step()
                finally:
                    self._active_gradient_probe = None
                if probe is not None:
                    measured_minibatches += 1
                    probe.metrics["extra_seconds"] = sum(
                        probe.metrics.get(k, 0.0)
                        for k in ("replay_seconds", "gram_seconds", "postclip_measurement_seconds")
                    )
                    probe.write_record(diagnostics.gradient_output_dir, {
                        "global_step": data.meta_info["global_steps"], "ppo_epoch": ppo_epoch,
                        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "process_id": os.getpid(),
                        "recorded_at_unix": time.time(),
                        "minibatch_index": batch_idx, "local_rows": len(mini_batch), "world_size": dp_size,
                        "branch_multiplier": branch_multiplier, "critic_multiplier": critic_multiplier,
                        "weighted_losses": loss_diagnostics,
                        "replayed_weighted_losses": dict(zip(COMPONENTS, probe.losses, strict=False)),
                        "scalar_diagnostic_errors": scalar_diagnostic_errors,
                    })
                    if probe.metrics.get("technical_error", 0) and diagnostics.gradient_disable_on_error:
                        self._gradient_diagnostics_disabled = True
                        gradient_step = False
                    metrics["actor/gradient_components/invalid_minibatches"] = (
                        metrics.get("actor/gradient_components/invalid_minibatches", 0)
                        + int(not probe.metrics.get("valid", 0))
                    )
                    append_to_dict(metrics, {f"actor/gradient_components/{k}": v for k, v in probe.metrics.items()})
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                mini_batch_metrics.update(getattr(self, "_gemma4_smoke_metrics", {}))
                if retry_critic_enabled:
                    mini_batch_metrics.update(
                        {
                            "actor/retry_critic_accept_examples": global_critic_examples[0],
                            "actor/retry_critic_retry_examples": global_critic_examples[1],
                            "actor/retry_critic_accept_parents": global_critic_parent_mass[0],
                            "actor/retry_critic_retry_parents": global_critic_parent_mass[1],
                            "actor/retry_critic_missing_accept": float(global_critic_examples[0] == 0),
                            "actor/retry_critic_missing_retry": float(global_critic_examples[1] == 0),
                        }
                    )
                    if retry_critic_dataset_balance:
                        mini_batch_metrics["actor/retry_critic_configured_strata"] = len(retry_critic_stratum_names)
                        mini_batch_metrics["actor/retry_critic_present_strata"] = sum(
                            count > 0 for count in global_critic_stratum_examples
                        )
                        for stratum_index, stratum_name in enumerate(retry_critic_stratum_names):
                            prefix = f"actor/retry_critic_strata/{stratum_name}"
                            mini_batch_metrics[f"{prefix}_examples"] = global_critic_stratum_examples[stratum_index]
                            mini_batch_metrics[f"{prefix}_parents"] = global_critic_stratum_parent_mass[stratum_index]
                            mini_batch_metrics[f"{prefix}_missing"] = float(
                                global_critic_stratum_examples[stratum_index] == 0
                            )
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        if diagnostics_enabled and diagnostics.get("gradient_enabled", False):
            self._gradient_diagnostics_seen_update = True
            metrics["actor/gradient_components/measured_minibatches"] = measured_minibatches
            metrics["actor/gradient_components/every_n_steps"] = diagnostics.gradient_every_n_steps
            metrics["actor/gradient_components/disabled_after_error"] = float(
                getattr(self, "_gradient_diagnostics_disabled", False)
            )
        if mass_bounds_enabled:
            for term, ratios in bound_ratios.items():
                metrics[f"actor/mass_bounds/{term}_to_canonical_max"] = max(ratios, default=0.0)
        return metrics
