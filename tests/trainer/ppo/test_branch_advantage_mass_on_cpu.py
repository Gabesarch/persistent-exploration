"""Check masked advantage diagnostics against the actual GSPO loss."""

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
from verl.trainer.ppo.metric_utils import compute_branch_advantage_mass_metrics
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.workers.config.actor import ActorConfig, BranchLossStrataConfig, PolicyLossConfig


def make_batch():
    # Unequal suffix lengths, mixed signs, an empty canonical anchor and padding.
    return DataProto.from_dict(tensors={
        "advantages": torch.tensor([
            [10000., 2., -4., 10000.],
            [-10000., -10000., 3., -10000.],
            [10000., 10000., -5., 7.],
            [10000., 10000., 10000., 10000.],
            [10000., 10000., 10000., 10000.],
        ], requires_grad=True),
        "response_mask": torch.tensor([[0, 1, 1, 0], [0, 0, 1, 0], [0, 0, 1, 1], [0, 0, 0, 0], [0, 0, 0, 0]]),
        "branch_policy_loss_weight": torch.tensor([2., 2., 1., 0., 0.]),
        "branch_policy_loss_is_branch": torch.tensor([False, False, True, False, False]),
    })


@pytest.mark.parametrize("importance_sampling", [False, True])
def test_mass_matches_gspo_at_unit_policy_ratio_and_leaves_inputs_unchanged(importance_sampling):
    batch = make_batch()
    if importance_sampling:
        batch.batch["rollout_is_weights"] = torch.tensor([
            [9., 0.5, 2., 9.], [9., 9., 2., 9.], [9., 9., 1., 0.5], [9.] * 4, [9.] * 4,
        ])
    original = batch.batch.clone()
    metrics = compute_branch_advantage_mass_metrics(batch)
    prefix = "actor/advantage_mass/"
    if not importance_sampling:
        # Canonical: (2*[1,2] + 2*[3,0])/5; branch: [3.5,2.5]/5.
        assert metrics[prefix + "canonical_positive"] == pytest.approx(1.6)
        assert metrics[prefix + "canonical_negative"] == pytest.approx(0.8)
        assert metrics[prefix + "branch_positive"] == pytest.approx(0.7)
        assert metrics[prefix + "branch_negative"] == pytest.approx(0.5)
    assert metrics[prefix + "positive"] == pytest.approx(
        metrics[prefix + "canonical_positive"] + metrics[prefix + "branch_positive"]
    )
    assert metrics[prefix + "negative"] == pytest.approx(
        metrics[prefix + "canonical_negative"] + metrics[prefix + "branch_negative"]
    )
    config = ActorConfig(
        strategy="fsdp", rollout_n=1, use_dynamic_bsz=True, loss_agg_mode="seq-mean-token-mean",
        policy_loss=PolicyLossConfig(loss_mode="gspo", branch_strata=BranchLossStrataConfig(enabled=True)),
    )
    log_prob = torch.zeros_like(batch.batch["advantages"], requires_grad=True)
    loss, _ = compute_policy_loss_gspo(
        old_log_prob=log_prob.detach(), log_prob=log_prob, config=config,
        **{key: batch.batch[key] for key in (
            "advantages", "response_mask", "branch_policy_loss_weight", "branch_policy_loss_is_branch",
        )},
        rollout_is_weights=batch.batch.get("rollout_is_weights"),
    )
    assert loss.item() == pytest.approx(metrics[prefix + "negative"] - metrics[prefix + "positive"])
    loss.backward()
    assert torch.count_nonzero(log_prob.grad[batch.batch["response_mask"] == 0]) == 0
    assert all(torch.equal(batch.batch[key], original[key]) for key in original.keys())
    assert metrics == pytest.approx(compute_branch_advantage_mass_metrics(batch[[2, 4, 0, 3, 1]]))


def test_fully_masked_batch_has_zero_finite_mass():
    batch = make_batch()
    batch.batch["response_mask"].zero_()
    batch.batch["advantages"] = torch.full_like(batch.batch["advantages"], float("nan"))
    assert all(value == 0.0 for value in compute_branch_advantage_mass_metrics(batch).values())


@pytest.mark.parametrize("enabled", [False, True])
def test_trainer_allows_empty_canonical_anchor_only_when_masking_is_enabled(enabled):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = OmegaConf.create({"actor_rollout_ref": {
        "actor": {"policy_loss": {"branch_strata": {"enabled": True}}},
        "rollout": {"multi_turn": {"conditional_answer_retry_mask_canonical_prefix": enabled}},
    }})
    batch = make_batch()
    batch.non_tensor_batch.update({
        "request_id": np.array(["a", "b", "c", "c", "padding"], dtype=object),
        "is_conditional_answer_retry_branch": np.array([False, False, True, False, False], dtype=object),
        "__branch_padding_row": np.array([False, False, False, False, True]),
    })
    if not enabled:
        with pytest.raises(ValueError, match="canonical row 3 has no active response tokens"):
            trainer._attach_branch_strata_loss_weights(batch, {})
        return
    trainer._attach_branch_strata_loss_weights(batch, {})
    assert batch.batch["branch_policy_loss_weight"][3:].tolist() == [0., 0.]
    assert batch.batch["branch_policy_loss_weight"].sum().item() == pytest.approx(5.)
    metrics = compute_branch_advantage_mass_metrics(batch)
    assert metrics["actor/advantage_mass/branch_negative"] > 0
    assert metrics["actor/advantage_mass/canonical_positive"] > 0


def test_mass_logging_requires_branch_strata():
    assert BranchLossStrataConfig().log_advantage_mass is False
    with pytest.raises(ValueError, match="requires branch strata"):
        BranchLossStrataConfig(log_advantage_mass=True)
