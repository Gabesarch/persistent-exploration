"""Opt-in positive-advantage scaling after the estimator's final normalization."""

import math

import torch


@torch.no_grad()
def amplify_positive_advantages(
    advantages: torch.Tensor,
    response_mask: torch.Tensor,
    factor: float,
    diagnostics: dict[str, float] | None = None,
) -> torch.Tensor:
    """Scale active A > 0 only, without mutating returns or normalizing again.

    This is a sign-based policy weight, not R3L's group-maximum override. Zero
    advantages, including tied all-failure groups, never become positive.
    """
    factor = float(factor)
    if not math.isfinite(factor) or factor < 1.0:
        raise ValueError("algorithm.positive_advantage_amplification must be finite and at least 1.")
    if factor == 1.0:
        return advantages
    if advantages.shape != response_mask.shape or not advantages.is_floating_point():
        raise ValueError("Positive amplification requires floating advantages with the same shape as response_mask.")
    active = response_mask.bool()
    if not torch.isfinite(advantages[active]).all():
        raise ValueError("Positive amplification received nonfinite active advantages.")
    positive = active & (advantages > 0)
    # Out of place: outcome estimators can alias advantages and returns. The
    # latter remain the original estimator targets, not policy-loss weights.
    amplified = torch.where(positive, advantages * factor, advantages)
    if not torch.isfinite(amplified[active]).all():
        raise ValueError("Positive amplification overflowed active advantages.")
    if diagnostics is not None:
        diagnostics.update({
            "advantage_amplification/factor": factor,
            "advantage_amplification/positive_active_tokens": float(positive.sum().item()),
            "advantage_amplification/positive_token_sum_before": float(advantages[positive].double().sum().item()),
            "advantage_amplification/positive_token_sum_after": float(amplified[positive].double().sum().item()),
        })
    return amplified
