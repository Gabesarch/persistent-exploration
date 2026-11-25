"""Detached, sequence-normalized budgets for opt-in auxiliary actor losses."""

import math

import torch


@torch.no_grad()
def advantage_mass_sums(advantages, response_mask, row_weights, is_branch, rollout_is_weights=None):
    """Return local [canonical+, canonical-, branch+, branch-] numerators.

    Sum across DP ranks and divide by the global configured optimizer-minibatch
    size, matching GSPO's microbatch/gradient-accumulation reduction. Padding
    already has zero row weight. Positive and negative mass never cancel.
    """
    mask = response_mask.to(dtype=torch.bool)
    values = torch.where(mask, advantages.detach().double(), 0.0)
    weights = row_weights.detach().to(device=values.device, dtype=torch.float64)
    branches = is_branch.to(device=values.device, dtype=torch.bool)
    if weights.ndim != 1 or weights.shape[0] != values.shape[0] or branches.shape != weights.shape:
        raise ValueError("Mass bounds require one row weight and branch flag per response.")
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Mass bounds require finite, nonnegative policy row weights.")
    if rollout_is_weights is not None:
        correction = torch.where(mask, rollout_is_weights.detach().to(values), 0.0)
        if not torch.isfinite(correction).all() or (correction < 0).any():
            raise ValueError("Mass bounds require finite, nonnegative rollout correction weights.")
        values = values * correction
    if not torch.isfinite(values).all():
        raise ValueError("Mass bounds received nonfinite active advantages.")
    lengths = mask.sum(-1).clamp_min(1)
    positive = values.clamp_min(0).sum(-1) / lengths * weights
    negative = (-values).clamp_min(0).sum(-1) / lengths * weights
    return torch.stack(
        (positive[~branches].sum(), negative[~branches].sum(), positive[branches].sum(), negative[branches].sum())
    )


def detached_cap_multiplier(reference, mass, max_ratio, epsilon=1e-12):
    """Downscale only; return a Python float with no autograd connection."""
    reference, mass, max_ratio, epsilon = map(float, (reference, mass, max_ratio, epsilon))
    if not all(math.isfinite(x) for x in (reference, mass, max_ratio, epsilon)):
        raise ValueError("Auxiliary mass budgets must be finite.")
    if min(reference, mass, max_ratio) < 0 or epsilon <= 0:
        raise ValueError("Auxiliary masses/ratios must be nonnegative and epsilon positive.")
    if mass == 0:
        return 1.0
    return min(1.0, max_ratio * reference / (mass + epsilon))


def mass_ratio(mass, reference, epsilon=1e-12):
    """Finite reporting ratio; canonical_zero distinguishes an empty reference."""
    return float(mass) / max(float(reference), float(epsilon))
