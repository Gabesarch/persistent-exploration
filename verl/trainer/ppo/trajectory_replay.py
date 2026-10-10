"""Controller-side mixing for opt-in, on-policy trajectory-judge replay."""

from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np
import torch

from verl.trainer.ppo import core_algos


def object_array(values):
    result = np.empty(len(values), dtype=object)
    result[:] = values
    return result


def apportion(total, weights):
    """Largest remainder allocation, with stable ties in input order."""
    weights = np.asarray(weights, dtype=float)
    if total < 0 or not np.isfinite(weights).all() or (weights < 0).any() or weights.sum() <= 0:
        raise ValueError("Allocation requires a nonnegative total and positive finite weights")
    expected = total * weights / weights.sum()
    counts = np.floor(expected).astype(int)
    for index in sorted(range(len(weights)), key=lambda i: (-(expected[i] - counts[i]), i))[: total - counts.sum()]:
        counts[index] += 1
    return counts


def replay_prompt_target(batch_size, fraction):
    """Round to balanced A/R pairs, retaining at least one fresh task."""
    if not 0 <= fraction < 1:
        raise ValueError("Replay prompt_fraction must satisfy 0 <= fraction < 1")
    return min(2 * int(math.floor(batch_size * fraction / 2 + 0.5)), 2 * ((batch_size - 1) // 2))


def choose_task_rows(batch, task_count, domain_weights, rng):
    """Keep the configured task-domain mix when replacing input prompt rows."""
    if not isinstance(domain_weights, Mapping) or not domain_weights or "ability" not in batch.non_tensor_batch:
        return np.sort(rng.choice(len(batch), task_count, replace=False))
    ability = np.asarray(batch.non_tensor_batch["ability"], dtype=object)
    domains = [domain for domain, weight in domain_weights.items() if float(weight) > 0]
    groups = [np.flatnonzero(ability == domain) for domain in domains]
    counts = apportion(task_count, [float(domain_weights[domain]) for domain in domains])
    # Respect the rows actually supplied by the existing balanced dataloader.
    excess = sum(max(0, count - len(group)) for count, group in zip(counts, groups, strict=True))
    counts = np.minimum(counts, [len(group) for group in groups])
    while excess:
        available = [i for i, group in enumerate(groups) if counts[i] < len(group)]
        if not available:
            raise ValueError("Fresh batch cannot supply configured task domains")
        for i in available:
            counts[i] += 1
            excess -= 1
            if not excess:
                break
    return np.sort(
        np.concatenate([rng.choice(group, count, replace=False) for group, count in zip(groups, counts, strict=True)])
    )


def fitting_replay_samples(cache, target, step, fits):
    """Backfill ineligible prompts while preserving balanced, distinct sources."""
    skipped = 0

    def eligible(candidate):
        nonlocal skipped
        valid = fits(candidate)
        skipped += int(not valid)
        return valid

    return cache.sample(target, step, eligible=eligible), skipped


def assign_prompt_group_schedules(count, config, step):
    single = float(config.get("schedule_single_fraction", 0.10))
    double = float(config.get("schedule_double_fraction", 0.05))
    if single < 0 or double < 0 or single + double > 1:
        raise ValueError("Prompt-group single/double fractions must be nonnegative and sum to at most one")
    first = int(config.get("schedule_first_turn", 2))
    last = int(config.get("schedule_last_turn", 9))
    gap = int(config.get("schedule_min_gap", 2))
    if first < 2 or last < first or gap < 2:
        raise ValueError("Prompt-group turns must start at >=2 and use a gap >=2")
    pairs = [(a, b) for a in range(first, last + 1) for b in range(a + gap, last + 1)]
    if double and not pairs:
        raise ValueError("Prompt-group turn range cannot supply two separated deadlines")
    rng = np.random.default_rng(np.random.SeedSequence([int(config.get("schedule_seed", 42)), int(step)]))
    counts = apportion(count, [1 - single - double, single, double])
    categories = np.repeat(np.arange(3), counts)
    rng.shuffle(categories)
    schedules = []
    for row, category in enumerate(categories):
        turns = []
        if category == 1:
            turns = [int(rng.integers(first, last + 1))]
        elif category == 2:
            turns = list(pairs[int(rng.integers(len(pairs)))])
        schedules.append(
            {
                "force_first_tool_call": bool(category),
                "answer_turns": turns,
                "group_id": f"step{step}:task{row}",
                "category": ["normal", "single", "double"][category],
            }
        )
    return object_array(schedules)


def judge_row_mask(batch):
    return np.array(
        [
            value is True or isinstance(value, np.bool_) and bool(value)
            for value in batch.non_tensor_batch.get("is_replay_judge", np.zeros(len(batch), dtype=bool))
        ]
    )


def attach_mixed_loss_weights(batch, strata, judge_share, metrics, allow_inactive_canonical=False):
    """Keep task branch weighting independent of judge rows and branch expansion."""
    judge = judge_row_mask(batch)
    padding = np.asarray(
        batch.non_tensor_batch.get("__branch_padding_row", np.zeros(len(batch), dtype=bool)), dtype=bool
    )
    active = batch.batch["response_mask"].bool().any(dim=-1).cpu().numpy()
    task_indices = np.flatnonzero(~judge & ~padding)
    judge_indices = np.flatnonzero(judge & ~padding & active)
    if not len(judge_indices):
        judge_share = 0.0
    weights = torch.zeros(len(batch), dtype=torch.float32, device=batch.batch["response_mask"].device)
    flags = np.asarray(
        batch.non_tensor_batch[strata.get("branch_indicator_key", "is_conditional_answer_retry_branch")], dtype=object
    )
    if len(task_indices):
        task_weights, diagnostics = core_algos.compute_branch_strata_loss_weights(
            response_mask=batch.batch["response_mask"][task_indices],
            branch_flags=flags[task_indices],
            parent_ids=batch.non_tensor_batch[strata.get("parent_id_key", "request_id")][task_indices],
            padding_mask=np.zeros(len(task_indices), dtype=bool),
            branch_fraction=float(strata.get("branch_fraction", 0.2)),
            cap_by_parent_prevalence=bool(strata.get("cap_by_parent_prevalence", True)),
            parent_balance=bool(strata.get("parent_balance", True)),
            allow_inactive_canonical=allow_inactive_canonical,
        )
        weights[task_indices] = task_weights * (len(batch) * (1 - judge_share) / task_weights.sum().clamp_min(1))
        metrics.update({f"branch_strata/{key}": value for key, value in diagnostics.items()})
    elif len(judge_indices):
        judge_share = 1.0
    if len(judge_indices):
        weights[judge_indices] = len(batch) * judge_share / len(judge_indices)
    batch.batch["branch_policy_loss_weight"] = weights
    batch.batch["branch_policy_loss_is_branch"] = torch.tensor(
        [bool(flag) if isinstance(flag, bool | np.bool_) else False for flag in flags],
        device=weights.device,
    ) & torch.as_tensor(~judge & ~padding, device=weights.device)
    metrics["replay/loss_share"] = float(weights[judge_indices].sum().item() / len(batch))
    metrics["replay/task_loss_share"] = float(weights[task_indices].sum().item() / len(batch))


def compute_mixed_advantages(batch, algorithm, compute_advantage, rollout_n):
    from omegaconf import OmegaConf

    judge = judge_row_mask(batch)
    padding = np.asarray(
        batch.non_tensor_batch.get("__branch_padding_row", np.zeros(len(batch), dtype=bool)), dtype=bool
    )
    advantages = torch.zeros_like(batch.batch["token_level_rewards"])
    returns = torch.zeros_like(advantages)
    for mask, is_judge in ((~judge & ~padding, False), (judge & ~padding, True)):
        indices = np.flatnonzero(mask)
        if not len(indices):
            continue
        config = OmegaConf.create(OmegaConf.to_container(algorithm, resolve=True))
        if is_judge:
            config.gdpo_reward_keys = ["accuracy", "format"]
            config.gdpo_reward_weights = [0.8, 0.2]
            config.gdpo_reward_eligibility = {}
        part = compute_advantage(
            batch.select_idxs(indices),
            adv_estimator=config.adv_estimator,
            gamma=config.get("gamma", 1.0),
            lam=config.get("lam", 1.0),
            num_repeat=rollout_n,
            norm_adv_by_std_in_grpo=config.get("norm_adv_by_std_in_grpo", True),
            config=config,
        )
        advantages[indices] = part.batch["advantages"]
        returns[indices] = part.batch["returns"]
        batch.meta_info[f"gdpo_{'judge' if is_judge else 'task'}_diagnostics"] = part.meta_info.get(
            "gdpo_stability_metrics", {}
        )
    batch.batch["advantages"], batch.batch["returns"] = advantages, returns
    return batch
