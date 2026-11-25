"""Exercise budgets and the actual actor update with a tiny CPU policy."""

import math
from pathlib import Path

import pytest
import torch

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
from verl.workers.config import (
    AuxiliaryMassBoundsConfig,
    BranchLossStrataConfig,
    FSDPActorConfig,
    PolicyLossConfig,
    RetryCriticLossConfig,
)
from verl.workers.utils.auxiliary_mass_bounds import advantage_mass_sums, detached_cap_multiplier, mass_ratio


def test_agreed_weighted_example_and_no_amplification():
    canonical, branch, critic = 0.198, 0.297, 0.016
    beta_b = detached_cap_multiplier(canonical, branch, 0.25)
    beta_c = detached_cap_multiplier(canonical, critic, 0.05)
    assert beta_b == pytest.approx(1 / 6)
    assert beta_c == pytest.approx(0.61875)
    assert branch * beta_b == pytest.approx(0.0495)
    assert critic * beta_c == pytest.approx(0.0099)
    assert detached_cap_multiplier(canonical, 0.001, 0.25) == 1
    assert detached_cap_multiplier(0, 1, 0.25) == 0
    assert detached_cap_multiplier(0, 0, 0.25) == 1
    assert detached_cap_multiplier(1, 1, 0) == 0
    assert mass_ratio(0, 0) == 0


def test_sequence_normalization_masking_signs_padding_and_importance_weights():
    advantage = torch.tensor(
        [[2.0, -2.0, math.nan], [4.0, 100.0, 100.0], [-6.0, -6.0, 100.0], [9.0, 9.0, 9.0]], requires_grad=True
    )
    mask = torch.tensor([[1, 1, 0], [1, 0, 0], [1, 1, 0], [0, 0, 0]])
    weights = torch.tensor([2.0, 0.5, 1.0, 0.0])
    branch = torch.tensor([False, False, True, True])
    importance = torch.tensor([[0.5, 2.0, math.nan], [2.0, 1.0, 1.0], [0.5, 1.5, 1.0], [1.0, 1.0, 1.0]])
    mass = advantage_mass_sums(advantage, mask, weights, branch, importance)
    assert mass.tolist() == pytest.approx([5.0, 4.0, 0.0, 6.0])
    assert not mass.requires_grad
    # Uneven DP/microbatch partitions must yield the same global numerator.
    partitioned = sum(
        advantage_mass_sums(advantage[s], mask[s], weights[s], branch[s], importance[s])
        for s in (slice(0, 1), slice(1, 3), slice(3, 4))
    )
    torch.testing.assert_close(partitioned, mass)


@pytest.mark.parametrize("args", [(math.nan, 1, 0.25), (1, math.inf, 0.25), (-1, 1, 0.25), (1, 1, -1)])
def test_invalid_masses_are_rejected(args):
    with pytest.raises(ValueError):
        detached_cap_multiplier(*args)


@pytest.mark.parametrize(
    "kwargs", [{"epsilon": 0}, {"max_branch_to_canonical": -1}, {"max_critic_to_canonical": math.nan}]
)
def test_invalid_config_is_rejected(kwargs):
    with pytest.raises(ValueError):
        AuxiliaryMassBoundsConfig(**kwargs)


def _config(enabled=True, micro=1, dynamic=False, epochs=1):
    return FSDPActorConfig(
        strategy="fsdp2",
        rollout_n=1,
        ppo_mini_batch_size=4,
        ppo_micro_batch_size_per_gpu=micro,
        use_dynamic_bsz=dynamic,
        ppo_epochs=epochs,
        loss_agg_mode="seq-mean-token-mean",
        use_torch_compile=False,
        policy_loss=PolicyLossConfig(
            loss_mode="gspo",
            branch_strata=BranchLossStrataConfig(enabled=True),
            retry_critic=RetryCriticLossConfig(enabled=True, loss_fraction=0.01),
            auxiliary_mass_bounds=AuxiliaryMassBoundsConfig(enabled=enabled),
        ),
    )


def _data():
    return DataProto.from_dict(
        tensors={
            "input_ids": torch.arange(4).unsqueeze(1),
            "position_ids": torch.zeros(4, 2, dtype=torch.long),
            "attention_mask": torch.ones(4, 2, dtype=torch.long),
            "responses": torch.zeros(4, 2, dtype=torch.long),
            "response_mask": torch.tensor([[1, 0]] * 4),
            "old_log_probs": torch.full((4, 2), -10.0),
            "advantages": torch.tensor([[1.0, 0.0], [-1.0, 0.0], [4.0, 0.0], [-4.0, 0.0]]),
            "branch_policy_loss_weight": torch.ones(4),
            "branch_policy_loss_is_branch": torch.tensor([False, False, True, True]),
            "retry_critic_mask": torch.tensor([[0, 1]] * 4),
            "retry_critic_is_retry": torch.tensor([False, True, False, True]),
            "retry_critic_parent_weight": torch.ones(4),
        },
        meta_info={"temperature": 1.0},
    )


def _actor(monkeypatch, config, stochastic=False, inaccurate_prepass=False):
    from verl.workers.actor import dp_actor

    monkeypatch.setattr(dp_actor, "get_device_id", lambda: "cpu")
    monkeypatch.setattr(dp_actor, "prepare_dynamic_batch", lambda data, **kwargs: ([data[:1], data[1:]], None))
    actor = object.__new__(dp_actor.DataParallelPPOActor)
    actor.config = config
    actor.actor_module = torch.nn.Embedding(4, 2)
    with torch.no_grad():
        actor.actor_module.weight.fill_(-10.0)
    actor.actor_optimizer = torch.optim.SGD(actor.actor_module.parameters(), lr=0.0)
    actor.device_name = "cpu"
    actor.use_prefix_grouper = False
    actor.lazy_multi_modal_inputs_enabled = False
    actor.ulysses_sequence_parallel_size = 1
    actor.scaler = None
    actor.forward_calls = []
    actor.recorded_gradients = []

    def forward(inputs, **kwargs):
        actor.forward_calls.append(torch.is_grad_enabled())
        log_prob = actor.actor_module(inputs["input_ids"][:, 0])
        if stochastic:
            log_prob = log_prob + torch.rand(()) * 0.1
        return {"log_probs": log_prob}

    def step():
        grad = actor.actor_module.weight.grad.detach().clone()
        actor.recorded_gradients.append(grad)
        actor.actor_optimizer.step()
        return grad.norm()

    actor._forward_micro_batch = forward
    actor._optimizer_step = step
    if inaccurate_prepass:
        actor._measure_current_retry_critic_ce = lambda *args, **kwargs: 0.1
    return actor


@pytest.mark.parametrize("micro,dynamic", [(1, False), (2, False), (1, True)])
def test_actor_caps_apply_to_real_backward_and_weighted_metrics_reconcile(monkeypatch, micro, dynamic):
    actor = _actor(monkeypatch, _config(micro=micro, dynamic=dynamic))
    metrics = actor.update_policy(_data())
    p = "actor/mass_bounds/"
    assert metrics[p + "canonical_abs_mass"] == pytest.approx([0.495])
    assert metrics[p + "branch_multiplier"] == pytest.approx([0.0625])
    assert metrics[p + "critic_multiplier"] == pytest.approx([0.2475])
    assert metrics[p + "branch_to_canonical_max"] == pytest.approx(0.25)
    assert metrics[p + "critic_to_canonical_max"] == pytest.approx(0.05)
    grad = actor.recorded_gradients[0]
    torch.testing.assert_close(grad[:, 0], torch.tensor([-0.2475, 0.2475, -0.061875, 0.061875]))
    torch.testing.assert_close(grad[:, 1], torch.full((4,), -0.00061875))
    assert sum(metrics[p + t + "_contribution_sum"] for t in ("canonical_gspo", "branch_gspo", "critic_ce")) == (
        pytest.approx(metrics["actor/combined_policy_loss"], abs=1e-6)
    )
    assert sum(metrics[p + t + "_mass_share_after"][0] for t in ("canonical", "branch", "critic")) == pytest.approx(1)


def test_multiple_ppo_epochs_do_not_compound_branch_weights(monkeypatch):
    actor = _actor(monkeypatch, _config(epochs=2))
    data = _data()
    metrics = actor.update_policy(data)
    assert metrics["actor/mass_bounds/branch_multiplier"] == pytest.approx([0.0625, 0.0625])
    torch.testing.assert_close(data.batch["branch_policy_loss_weight"], torch.ones(4))
    torch.testing.assert_close(actor.recorded_gradients[0], actor.recorded_gradients[1])


def test_partial_fixed_microbatch_uses_the_same_global_row_denominator(monkeypatch):
    actor = _actor(monkeypatch, _config(micro=2))
    metrics = actor.update_policy(_data()[:3])
    assert metrics["actor/mass_bounds/canonical_abs_mass"] == pytest.approx([0.495])
    grad = actor.recorded_gradients[0]
    torch.testing.assert_close(grad[:, 0], torch.tensor([-0.2475, 0.2475, -0.12375, 0.0]))
    # Two A rows share half the class-balanced CE; the single R row gets half.
    torch.testing.assert_close(grad[:, 1], torch.tensor([-0.00061875, -0.0012375, -0.00061875, 0.0]))


def test_ce_prepass_restores_rng_and_actual_audit_matches(monkeypatch):
    actor = _actor(monkeypatch, _config(), stochastic=True)
    metrics = actor.update_policy(_data())
    assert metrics["actor/mass_bounds/critic_prepass_absolute_error"] == pytest.approx([0], abs=1e-8)
    assert actor.forward_calls == [False] * 4 + [True] * 4


def test_actual_ce_violation_aborts_before_optimizer_step(monkeypatch):
    actor = _actor(monkeypatch, _config(), inaccurate_prepass=True)
    with pytest.raises(RuntimeError, match="Actual critic CE exceeded"):
        actor.update_policy(_data())
    assert actor.recorded_gradients == []
    assert actor.actor_module.weight.grad is None


def test_disabled_setting_uses_legacy_objective_without_extra_forward(monkeypatch):
    actor = _actor(monkeypatch, _config(enabled=False))
    metrics = actor.update_policy(_data())
    assert not any("mass_bounds" in key for key in metrics)
    assert actor.forward_calls == [True] * 4
    torch.testing.assert_close(actor.recorded_gradients[0][:, 0], torch.tensor([-0.2475, 0.2475, -0.99, 0.99]))
    torch.testing.assert_close(actor.recorded_gradients[0][:, 1], torch.full((4,), -0.0025))


def test_zero_canonical_mass_suppresses_both_auxiliaries(monkeypatch):
    actor = _actor(monkeypatch, _config())
    data = _data()
    data.batch["advantages"][:2] = 0
    metrics = actor.update_policy(data)
    assert metrics["actor/mass_bounds/canonical_zero"] == [1]
    assert metrics["actor/mass_bounds/branch_multiplier"] == [0]
    assert metrics["actor/mass_bounds/critic_multiplier"] == [0]
    assert metrics["actor/combined_policy_loss"] == 0
    torch.testing.assert_close(actor.recorded_gradients[0], torch.zeros(4, 2))


def test_absent_critic_skips_prepass_and_preserves_canonical_coefficient(monkeypatch):
    actor = _actor(monkeypatch, _config())
    data = _data()
    data.batch["retry_critic_mask"].zero_()
    metrics = actor.update_policy(data)
    assert metrics["actor/mass_bounds/critic_multiplier"] == [1]
    assert metrics["actor/mass_bounds/critic_ce_mass_after"] == [0]
    assert actor.forward_calls == [True] * 4
    torch.testing.assert_close(actor.recorded_gradients[0][:2, 0], torch.tensor([-0.2475, 0.2475]))


def test_gspo_component_metrics_sum_even_with_clipping_and_importance_weights():
    data = _data().batch
    log_prob = data["old_log_probs"] + torch.tensor([[0.4, 0.0], [-0.4, 0.0], [0.2, 0.0], [-0.2, 0.0]])
    loss, metrics = compute_policy_loss_gspo(
        data["old_log_probs"],
        log_prob,
        data["advantages"],
        data["response_mask"],
        config=_config(),
        branch_policy_loss_weight=torch.tensor([0.8, 0.8, 0.1, 0.1]),
        branch_policy_loss_is_branch=data["branch_policy_loss_is_branch"],
        rollout_is_weights=torch.tensor([[2.0], [0.5], [1.0], [3.0]]),
    )
    assert loss.item() == pytest.approx(
        metrics["actor/pg_loss_canonical_weighted"] + metrics["actor/pg_loss_branch_weighted"], abs=1e-7
    )


def test_symmetric_dp_reduction_preserves_global_budget_and_gradients(monkeypatch):
    actor = _actor(monkeypatch, _config())
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 2)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda tensor, **kwargs: tensor.mul_(2))
    metrics = actor.update_policy(_data())
    assert metrics["actor/mass_bounds/canonical_abs_mass"] == pytest.approx([0.495])
    assert metrics["actor/mass_bounds/branch_to_canonical_max"] == pytest.approx(0.25)
    assert metrics["actor/mass_bounds/critic_to_canonical_max"] == pytest.approx(0.05)
    torch.testing.assert_close(actor.recorded_gradients[0][:, 1], torch.full((4,), -0.00061875))
