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


import torch
from tensordict import TensorDict

from verl.trainer.ppo.core_algos import agg_loss, compute_value_loss, get_policy_loss_fn, kl_penalty
from verl.trainer.ppo.diffusion_algos import kl_penalty_image
from verl.utils import tensordict_utils as tu
from verl.utils.dataset.dataset_utils import DatasetPadMode
from verl.utils.metric import AggregationType, Metric
from verl.utils.torch_functional import masked_mean, masked_sum
from verl.workers.config import ActorConfig, CriticConfig
from verl.workers.utils.padding import no_padding_2_padding


def masked_sft_nll(
    log_prob: torch.Tensor,
    loss_mask: torch.Tensor,
    *,
    batch_num_tokens: int | torch.Tensor,
    dp_size: int = 1,
) -> torch.Tensor:
    """Token-normalized supervised NLL shared by SFT and hybrid PPO actors."""
    if isinstance(batch_num_tokens, torch.Tensor):
        denominator = batch_num_tokens.to(device=log_prob.device, dtype=log_prob.dtype)
    else:
        denominator = log_prob.new_tensor(batch_num_tokens)
    if denominator.item() <= 0:
        return log_prob.sum() * 0.0
    return -masked_sum(log_prob, loss_mask) / denominator * dp_size


def combine_retry_sft_loss(
    rl_loss: torch.Tensor,
    retry_sft_loss: torch.Tensor,
    *,
    loss_fraction: float,
    has_rl: bool,
    has_retry_sft: bool,
) -> torch.Tensor:
    """Combine independently normalized RL and retry-SFT strata."""
    if not has_retry_sft:
        return rl_loss
    if not has_rl:
        return retry_sft_loss
    fraction = float(loss_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"retry SFT loss_fraction must satisfy 0 <= value < 1, got {fraction!r}")
    return (1.0 - fraction) * rl_loss + fraction * retry_sft_loss


def parent_balanced_retry_critic_loss(
    log_prob: torch.Tensor,
    critic_mask: torch.Tensor,
    is_retry: torch.Tensor,
    parent_weight: torch.Tensor,
    *,
    global_accept_parent_mass: int | float | torch.Tensor,
    global_retry_parent_mass: int | float | torch.Tensor,
    dp_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, bool, bool]:
    """Return class-balanced A/R NLL after averaging attempts per parent.

    ``parent_weight`` is one over the number of retained attempts for the
    row's (parent, class). Its globally summed mass is therefore the number of
    represented parents in that class.
    """

    is_retry = is_retry.to(device=log_prob.device, dtype=torch.bool)
    stratum_id = is_retry.to(dtype=torch.long)
    critic_loss, stratum_nll, present = parent_balanced_retry_critic_strata_loss(
        log_prob,
        critic_mask,
        stratum_id,
        parent_weight,
        global_stratum_parent_mass=[global_accept_parent_mass, global_retry_parent_mass],
        dp_size=dp_size,
    )
    accept_nll, retry_nll = stratum_nll.unbind()
    has_accept, has_retry = (bool(value.item()) for value in present)
    return critic_loss, accept_nll, retry_nll, has_accept, has_retry


def parent_balanced_retry_critic_strata_loss(
    log_prob: torch.Tensor,
    critic_mask: torch.Tensor,
    stratum_id: torch.Tensor,
    parent_weight: torch.Tensor,
    *,
    global_stratum_parent_mass: list[int | float] | torch.Tensor,
    dp_size: int = 1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the equally weighted mean NLL over globally present critic strata.

    Attempts are averaged per parent before this reducer. ``stratum_id`` is a
    zero-based row assignment, with negative values reserved for inactive rows.
    Missing strata contribute no mass; the target weights are renormalized over
    the strata represented in the current distributed PPO minibatch.
    """

    critic_mask = critic_mask.to(device=log_prob.device, dtype=torch.bool)
    stratum_id = stratum_id.to(device=log_prob.device, dtype=torch.long)
    parent_weight = parent_weight.to(device=log_prob.device, dtype=log_prob.dtype)
    if isinstance(global_stratum_parent_mass, torch.Tensor):
        denominators = global_stratum_parent_mass.to(device=log_prob.device, dtype=log_prob.dtype).flatten()
    else:
        denominators = torch.stack(
            [
                torch.as_tensor(value, device=log_prob.device, dtype=log_prob.dtype)
                for value in global_stratum_parent_mass
            ]
        ).flatten()
    if denominators.numel() == 0:
        raise ValueError("global_stratum_parent_mass must contain at least one stratum.")

    active_rows = critic_mask.any(dim=-1)
    invalid_active = active_rows & ((stratum_id < 0) | (stratum_id >= denominators.numel()))
    if bool(invalid_active.any().item()):
        invalid_ids = sorted(set(stratum_id[invalid_active].detach().cpu().tolist()))
        raise ValueError(
            f"Active retry critic rows have stratum ids outside [0, {denominators.numel()}): {invalid_ids}."
        )

    row_nll = -(log_prob * critic_mask.to(dtype=log_prob.dtype)).sum(dim=-1)
    safe_stratum_id = stratum_id.clamp(min=0, max=denominators.numel() - 1)
    weighted_row_nll = row_nll * parent_weight * active_rows.to(dtype=log_prob.dtype)
    numerators = torch.zeros_like(denominators).scatter_add(0, safe_stratum_id, weighted_row_nll)
    present = denominators > 0
    safe_denominators = torch.where(present, denominators, torch.ones_like(denominators))
    stratum_nll = numerators / safe_denominators * dp_size
    stratum_nll = torch.where(present, stratum_nll, torch.zeros_like(stratum_nll))
    if bool(present.any().item()):
        critic_loss = stratum_nll[present].mean()
    else:
        critic_loss = log_prob.sum() * 0.0
    return critic_loss, stratum_nll, present


def combine_retry_critic_loss(
    existing_loss: torch.Tensor,
    critic_loss: torch.Tensor,
    *,
    loss_fraction: float,
    has_existing: bool,
    has_critic: bool,
) -> torch.Tensor:
    """Mix the shared-weight critic CE with the already-composed actor loss."""

    if not has_critic:
        return existing_loss
    if not has_existing:
        return critic_loss
    fraction = float(loss_fraction)
    if not 0.0 <= fraction < 1.0:
        raise ValueError(f"retry critic loss_fraction must satisfy 0 <= value < 1, got {fraction!r}")
    return (1.0 - fraction) * existing_loss + fraction * critic_loss


def sft_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    pad_mode = tu.get_non_tensor_data(data=data, key="pad_mode", default=DatasetPadMode.NO_PADDING)
    dp_size = data["dp_size"]
    batch_num_tokens = data["batch_num_tokens"]

    log_prob = model_output["log_probs"]

    if pad_mode == DatasetPadMode.NO_PADDING:
        # log_prob and loss mask are nested tensors of shape [bsz, j1]
        # for each sample, loss mask shape is [1, prompt_length + response_length]
        loss_mask = data["loss_mask"]

        log_prob_flatten = log_prob.values()
        loss_mask_flatten = loss_mask.values()

        # left-shift the loss mask by one token to align with log_prob
        loss_mask_flatten = torch.roll(loss_mask_flatten, shifts=-1, dims=0)

        # NOTE: loss is averaged over all tokens in the batch across all data parallel groups,
        # For FSDP backend, the loss is directly used for backward; while for Megatron backend,
        # the loss should be scaled by `num_microbatches` for pp schedule.
        loss = masked_sft_nll(
            log_prob_flatten,
            loss_mask_flatten,
            batch_num_tokens=batch_num_tokens,
            dp_size=dp_size,
        )
    else:
        response_mask = data["response_mask"].to(bool)
        loss = masked_sft_nll(
            log_prob,
            response_mask,
            batch_num_tokens=batch_num_tokens,
            dp_size=dp_size,
        )

    return loss, {}


def ppo_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Computes ppo loss from model output (log_prob, entropy, values, etc. ) and old_log_probs from data."""
    log_prob = no_padding_2_padding(model_output["log_probs"], data)
    entropy = model_output.get("entropy", None)
    if entropy is not None:
        entropy = no_padding_2_padding(entropy, data)

    # global batch info for loss aggregation
    config.global_batch_info["dp_size"] = data["dp_size"]
    config.global_batch_info["batch_num_tokens"] = data["batch_num_tokens"]
    config.global_batch_info["global_batch_size"] = data["global_batch_size"]
    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    # assumes that if any of the global batch info is set, the policy_loss_fn will
    # normalize using dp_size/global_bsz/global_token; in this case, metric aggregation should be SUM
    # to reflect the mean loss over the global batch
    if (
        data["dp_size"] > 1
        or data["batch_num_tokens"] is not None
        or data["global_batch_size"] is not None
        or config.loss_scale_factor is not None
    ):
        metric_aggregation = AggregationType.SUM
    else:
        metric_aggregation = AggregationType.MEAN

    metrics = {}

    # select fields and convert to padded tensor
    fields = ["response_mask", "old_log_probs", "advantages"]
    if "rollout_is_weights" in data:
        fields.append("rollout_is_weights")
    if "ref_log_prob" in data:
        fields.append("ref_log_prob")
    data = data.select(*fields).to_padded_tensor()

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]
    rollout_is_weights = data.get("rollout_is_weights", None)

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "vanilla")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=rollout_is_weights,
    )

    # AggregationType.MEAN for pg metrics: assumes policy_loss_fn normalizes by local_bsz/local_tokens
    # Ex: in compute_policy_loss_vanilla, pg_metrics are pg_clipfrac, ppo_kl, pg_clipfrac_lower
    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=metric_aggregation)
    policy_loss = pg_loss

    # add entropy loss
    if entropy is not None:
        entropy_loss = agg_loss(
            loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode, **config.global_batch_info
        )
        entropy_coeff = config.entropy_coeff
        policy_loss -= entropy_coeff * entropy_loss
        metrics["actor/entropy_loss"] = Metric(value=entropy_loss, aggregation=metric_aggregation)

    # add kl loss
    if config.use_kl_loss:
        ref_log_prob = data["ref_log_prob"]
        # compute kl loss
        kld = kl_penalty(logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=config.kl_loss_type)
        kl_loss = agg_loss(
            loss_mat=kld, loss_mask=response_mask, loss_agg_mode=config.loss_agg_mode, **config.global_batch_info
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=metric_aggregation)
        metrics["kl_coef"] = config.kl_loss_coef

    return policy_loss, metrics


def value_loss(config: CriticConfig, model_output, data: TensorDict, dp_group=None):
    """value loss

    Args:
        config: CriticConfig
        model_output: model output from the model
        data: the input to the model
        dp_group: data paralle group

    Returns:
        value loss
    """
    vpreds = no_padding_2_padding(model_output["values"], data)  # (bsz, response_length)

    # select fields and convert to padded tensor
    data = data.select("values", "returns", "response_mask").to_padded_tensor()
    values = data["values"]
    returns = data["returns"]
    response_mask = data["response_mask"].to(bool)

    vf_loss, vf_clipfrac = compute_value_loss(
        vpreds=vpreds,
        values=values,
        returns=returns,
        response_mask=response_mask,
        cliprange_value=config.cliprange_value,
        loss_agg_mode=config.loss_agg_mode,
    )

    metrics = {}

    metrics.update(
        {
            "critic/vf_loss": vf_loss.detach().item(),
            "critic/vf_clipfrac": vf_clipfrac.detach().item(),
            "critic/vpred_mean": masked_mean(vpreds, response_mask).detach().item(),
        }
    )

    return vf_loss, metrics


def diffusion_loss(config: ActorConfig, model_output, data: TensorDict, dp_group=None):
    """Compute loss for diffusion model"""
    log_prob = model_output["log_probs"]

    config.global_batch_info["loss_scale_factor"] = config.loss_scale_factor

    metrics = {}

    response_mask = data["response_mask"].to(bool)
    # compute policy loss
    old_log_prob = data["old_log_probs"]
    advantages = data["advantages"]

    loss_agg_mode = config.loss_agg_mode

    loss_mode = config.policy_loss.get("loss_mode", "flow_grpo")

    policy_loss_fn = get_policy_loss_fn(loss_mode)
    pg_loss, pg_metrics = policy_loss_fn(
        old_log_prob=old_log_prob,
        log_prob=log_prob,
        advantages=advantages,
        response_mask=response_mask,
        loss_agg_mode=loss_agg_mode,
        config=config,
        rollout_is_weights=None,
    )

    pg_metrics = Metric.from_dict(pg_metrics, aggregation=AggregationType.MEAN)

    metrics.update(pg_metrics)
    metrics["actor/pg_loss"] = Metric(value=pg_loss, aggregation=AggregationType.MEAN)
    policy_loss = pg_loss

    if config.use_kl_loss:
        ref_prev_sample_mean = data["ref_prev_sample_mean"]
        prev_sample_mean = model_output["prev_sample_mean"]
        std_dev_t = model_output["std_dev_t"]
        kl_loss = kl_penalty_image(
            prev_sample_mean=prev_sample_mean, ref_prev_sample_mean=ref_prev_sample_mean, std_dev_t=std_dev_t
        )

        policy_loss += kl_loss * config.kl_loss_coef
        metrics["kl_loss"] = Metric(value=kl_loss, aggregation=AggregationType.MEAN)
        metrics["kl_coef"] = config.kl_loss_coef

    gradient_accumulation_steps = tu.get_non_tensor_data(data, "gradient_accumulation_steps", default=None)
    policy_loss = policy_loss / gradient_accumulation_steps

    return policy_loss, metrics
