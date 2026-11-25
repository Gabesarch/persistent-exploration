import torch

from verl.workers.utils.losses import (
    combine_retry_critic_loss,
    parent_balanced_retry_critic_loss,
    parent_balanced_retry_critic_strata_loss,
)


def _critic_loss(accept_nlls, retry_nlls, accept_weights=None, retry_weights=None, dp_size=1):
    nlls = list(accept_nlls) + list(retry_nlls)
    log_prob = -torch.tensor(nlls, dtype=torch.float32).unsqueeze(1)
    mask = torch.ones_like(log_prob, dtype=torch.bool)
    is_retry = torch.tensor([False] * len(accept_nlls) + [True] * len(retry_nlls))
    weights = torch.tensor(
        list(accept_weights or [1.0] * len(accept_nlls))
        + list(retry_weights or [1.0] * len(retry_nlls)),
        dtype=torch.float32,
    )
    return parent_balanced_retry_critic_loss(
        log_prob,
        mask,
        is_retry,
        weights,
        global_accept_parent_mass=sum(accept_weights or [1.0] * len(accept_nlls)),
        global_retry_parent_mass=sum(retry_weights or [1.0] * len(retry_nlls)),
        dp_size=dp_size,
    )


def test_retry_critic_is_class_balanced_under_1_to_100_and_100_to_1_imbalance():
    one_to_hundred = _critic_loss([2.0], [6.0] * 100)
    hundred_to_one = _critic_loss([2.0] * 100, [6.0])
    assert torch.allclose(one_to_hundred[0], torch.tensor(4.0))
    assert torch.allclose(hundred_to_one[0], torch.tensor(4.0))
    assert torch.allclose(one_to_hundred[1], torch.tensor(2.0))
    assert torch.allclose(one_to_hundred[2], torch.tensor(6.0))


def test_retry_critic_averages_attempts_within_parent_before_classes():
    # The first parent's two A attempts have weights 1/2 each; the second has one attempt.
    result = _critic_loss([2.0, 6.0, 10.0], [4.0], accept_weights=[0.5, 0.5, 1.0])
    assert torch.allclose(result[1], torch.tensor(7.0))  # ((2+6)/2 + 10) / 2 parents
    assert torch.allclose(result[0], torch.tensor(5.5))


def test_retry_critic_single_class_gets_full_mass_and_reports_missing_class():
    critic, accept, retry, has_accept, has_retry = _critic_loss([3.0, 5.0], [])
    assert torch.allclose(critic, torch.tensor(4.0))
    assert torch.allclose(accept, torch.tensor(4.0))
    assert retry.item() == 0.0
    assert has_accept is True
    assert has_retry is False


def test_retry_critic_distributed_denominator_scaling_matches_global_average():
    # One rank contributes one of two globally represented A parents. Multiplying
    # by dp_size compensates DDP's gradient averaging.
    log_prob = torch.tensor([[-2.0]])
    result = parent_balanced_retry_critic_loss(
        log_prob,
        torch.tensor([[True]]),
        torch.tensor([False]),
        torch.tensor([1.0]),
        global_accept_parent_mass=2.0,
        global_retry_parent_mass=0.0,
        dp_size=2,
    )
    assert torch.allclose(result[0], torch.tensor(2.0))


def test_retry_critic_mixes_at_three_percent_and_leaves_existing_loss_when_empty():
    existing = torch.tensor(2.0)
    critic = torch.tensor(6.0)
    assert torch.allclose(
        combine_retry_critic_loss(
            existing, critic, loss_fraction=0.03, has_existing=True, has_critic=True
        ),
        torch.tensor(2.12),
    )
    assert torch.equal(
        combine_retry_critic_loss(
            existing, critic, loss_fraction=0.03, has_existing=True, has_critic=False
        ),
        existing,
    )


def _strata_loss(nlls, stratum_ids, parent_weights, parent_mass, dp_size=1):
    log_prob = -torch.tensor(nlls, dtype=torch.float32).unsqueeze(1)
    mask = torch.ones_like(log_prob, dtype=torch.bool)
    return parent_balanced_retry_critic_strata_loss(
        log_prob,
        mask,
        torch.tensor(stratum_ids),
        torch.tensor(parent_weights, dtype=torch.float32),
        global_stratum_parent_mass=parent_mass,
        dp_size=dp_size,
    )


def test_retry_critic_balances_six_dataset_label_strata_despite_skewed_row_counts():
    # Each stratum has a constant NLL equal to 1..6. Repeated rows are attempts
    # from the same parent and therefore split one unit of parent mass.
    result = _strata_loss(
        nlls=[1, 2, 2, 2, 3, 3, 4, 5, 5, 5, 5, 6, 6],
        stratum_ids=[0, 1, 1, 1, 2, 2, 3, 4, 4, 4, 4, 5, 5],
        parent_weights=[1, 1 / 3, 1 / 3, 1 / 3, 1, 1, 1, 0.25, 0.25, 0.25, 0.25, 1, 1],
        parent_mass=[1, 1, 2, 1, 1, 2],
    )
    assert torch.allclose(result[0], torch.tensor(3.5))
    assert torch.allclose(result[1], torch.arange(1, 7, dtype=torch.float32))
    assert result[2].tolist() == [True] * 6


def test_retry_critic_supports_four_datasets_as_eight_equal_strata():
    result = _strata_loss(
        nlls=list(range(1, 9)),
        stratum_ids=list(range(8)),
        parent_weights=[1] * 8,
        parent_mass=[1] * 8,
    )
    assert torch.allclose(result[0], torch.tensor(4.5))
    assert result[1].numel() == 8


def test_two_dataset_critic_has_equal_gradient_mass_despite_7030_source_skew():
    # Visual Probe A/R and HSTAR A/R: source and label imbalance must not
    # change the four strata's loss weights.
    counts = [70, 7, 30, 1]
    ids = torch.repeat_interleave(torch.arange(4), torch.tensor(counts))
    log_prob = torch.full((sum(counts), 1), -1.0, requires_grad=True)
    loss, _, present = parent_balanced_retry_critic_strata_loss(
        log_prob,
        torch.ones_like(log_prob, dtype=torch.bool),
        ids,
        torch.ones(sum(counts)),
        global_stratum_parent_mass=counts,
    )
    loss.backward()
    assert present.tolist() == [True] * 4
    for stratum in range(4):
        assert torch.allclose(-log_prob.grad[ids == stratum].sum(), torch.tensor(0.25))


def test_retry_critic_renormalizes_over_present_dataset_label_strata():
    result = _strata_loss(
        nlls=list(range(1, 8)),
        stratum_ids=list(range(7)),
        parent_weights=[1] * 7,
        parent_mass=[1] * 7 + [0],
    )
    assert torch.allclose(result[0], torch.tensor(4.0))
    assert result[1][-1].item() == 0.0
    assert result[2].tolist() == [True] * 7 + [False]


def test_retry_critic_strata_distributed_scaling_matches_global_average():
    rank_zero = _strata_loss([2.0], [0], [1.0], [2.0, 1.0], dp_size=2)[0]
    rank_one = _strata_loss([4.0, 6.0], [0, 1], [1.0, 1.0], [2.0, 1.0], dp_size=2)[0]
    assert torch.allclose((rank_zero + rank_one) / 2, torch.tensor(4.5))
