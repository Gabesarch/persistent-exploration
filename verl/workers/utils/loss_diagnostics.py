"""Diagnostics of the five terms in the shared GSPO/retry-critic objective."""

import logging
import math

import torch

from verl.workers.utils.losses import combine_retry_critic_loss

COMPONENTS = ("canonical_positive", "canonical_negative", "branch_positive", "branch_negative", "critic_ce")
logger = logging.getLogger(__name__)


def weighted_loss_components(gspo, critic_ce, *, loss_scale, critic_fraction, has_rl, has_critic,
                             bounded=False, critic_multiplier=1.0):
    """Use the same mixture/fallback as training, including critic-only batches."""
    zero = critic_ce * 0.0
    if bounded:
        result = {name: gspo[name] * loss_scale * (1.0 - critic_fraction) for name in COMPONENTS[:4]}
        result["critic_ce"] = critic_ce * critic_fraction * critic_multiplier
    else:
        result = {
            name: combine_retry_critic_loss(gspo[name] * loss_scale, zero, loss_fraction=critic_fraction,
                                           has_existing=has_rl, has_critic=has_critic)
            for name in COMPONENTS[:4]
        }
        result["critic_ce"] = combine_retry_critic_loss(
            zero, critic_ce, loss_fraction=critic_fraction, has_existing=has_rl, has_critic=has_critic,
        )
    return result


def reduce_loss_diagnostics(totals):
    """DP-mean local backward contributions; totals include actual combined loss."""
    if torch.distributed.is_initialized():
        torch.distributed.all_reduce(totals)
        totals /= torch.distributed.get_world_size()
    values = totals.tolist()
    finite = all(math.isfinite(value) for value in values)
    # Do not send NaN/Inf diagnostic values to the training logger. The actual
    # objective/optimizer still use their original tensors and safety checks.
    values = [value if math.isfinite(value) else 0.0 for value in values]
    result = {name: values[i] for i, name in enumerate(COMPONENTS)}
    result["canonical"] = values[0] + values[1]
    result["branch"] = values[2] + values[3]
    result["positive_gspo"] = values[0] + values[2]
    result["negative_gspo"] = values[1] + values[3]
    result["combined"] = values[5]
    result["reconstruction_abs_error"] = abs(sum(values[:5]) - values[5])
    result["finite"] = float(finite)
    result["computation_valid"] = float(len(values) == 6 or values[6] == 0)
    result["reconstruction_valid"] = float(
        finite and result["computation_valid"]
        and result["reconstruction_abs_error"] <= 1e-6 + 1e-5 * sum(abs(v) for v in values[:5])
    )
    if not result["reconstruction_valid"]:
        logger.warning("Optional five-component loss diagnostic failed its audit: %s", result)
    return result
