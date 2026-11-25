"""Post-normalization amplification and isolation of the copied Gemma variant."""

from pathlib import Path

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from verl import DataProto
from verl.trainer.config import AlgoConfig
from verl.trainer.ppo.advantage_amplification import amplify_positive_advantages
from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
from verl.trainer.ppo.ray_trainer import compute_advantage
from verl.workers.config import ActorConfig, BranchLossStrataConfig, PolicyLossConfig


def _algorithm(**overrides):
    return OmegaConf.create({
        "adv_estimator": "gdpo", "gdpo_reward_keys": ["accuracy"], "gdpo_reward_weights": [1.0],
        "gdpo_baseline_weight_key": "gdpo_baseline_weight", "gdpo_stability_enabled": True,
        "gdpo_component_std_floor": 0.05, "gdpo_component_adv_clip": 5.0,
        "gdpo_final_std_floor": 0.05, "gdpo_final_adv_clip": 0.25,
        **overrides,
    })


def _data(all_wrong=False):
    mask = torch.tensor([[1, 1, 0]] * 4 + [[1, 0, 1], [0, 0, 0]])
    return DataProto.from_dict(
        tensors={
            "prompts": torch.ones(6, 2, dtype=torch.long), "responses": torch.zeros(6, 3, dtype=torch.long),
            "attention_mask": torch.cat([torch.ones(6, 2, dtype=torch.long), mask], dim=1),
            "response_mask": mask, "token_level_rewards": torch.zeros(6, 3),
        },
        non_tensors={
            "uid": np.array(["a"] * 6, dtype=object),
            "accuracy": np.array([0.] * 6 if all_wrong else [0., 1., 0., 1., 1., 0.]),
            "gdpo_baseline_weight": np.array([1., 1., 1., 1., 0., 0.]),
            "is_conditional_answer_retry_branch": np.array([False] * 4 + [True, False]),
        },
    )


def test_only_active_positive_advantages_change_and_input_is_not_mutated():
    original = torch.tensor([[2., -3., 0., 4.], [-2., 1., 0., -4.]], dtype=torch.float64)
    before = original.clone()
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]])
    result = amplify_positive_advantages(original, mask, 3)
    torch.testing.assert_close(result, torch.tensor([[6., -3., 0., 4.], [-2., 3., 0., -4.]], dtype=torch.float64))
    torch.testing.assert_close(original, before, rtol=0, atol=0)
    assert result.dtype == original.dtype
    assert amplify_positive_advantages(original, mask, 1) is original


@pytest.mark.parametrize("factor", [0., -1., 0.5, float("nan"), float("inf")])
def test_invalid_factors_are_rejected(factor):
    with pytest.raises(ValueError, match="finite and at least 1"):
        amplify_positive_advantages(torch.ones(2, 3), torch.ones(2, 3), factor)


def test_gdpo_amplification_is_after_final_whitening_and_clipping_and_preserves_returns():
    base = compute_advantage(_data(), "gdpo", config=_algorithm())
    amplified = compute_advantage(_data(), "gdpo", config=_algorithm(positive_advantage_amplification=3.0))
    original = base.batch["advantages"]
    expected = torch.where(original > 0, original * 3, original)
    torch.testing.assert_close(amplified.batch["advantages"], expected, rtol=0, atol=0)
    # The estimator clips at 0.25 BEFORE amplification; no subsequent clipping
    # or re-centering is allowed to erase the requested 3x positive weight.
    assert original.max().item() == 0.25
    assert amplified.batch["advantages"].max().item() == 0.75
    torch.testing.assert_close(amplified.batch["returns"], base.batch["returns"], rtol=0, atol=0)
    assert amplified.batch["advantages"].data_ptr() != amplified.batch["returns"].data_ptr()
    for key in base.batch.keys():
        if key != "advantages":
            torch.testing.assert_close(amplified.batch[key], base.batch[key], rtol=0, atol=0)
    assert amplified.meta_info["gdpo_stability_metrics"] == base.meta_info["gdpo_stability_metrics"]
    metrics = amplified.meta_info["positive_advantage_amplification_metrics"]
    assert metrics["advantage_amplification/factor"] == 3
    assert metrics["advantage_amplification/positive_token_sum_after"] == (
        3 * metrics["advantage_amplification/positive_token_sum_before"]
    )


def test_all_wrong_tied_group_and_inactive_critic_padding_rows_stay_zero():
    batch = compute_advantage(_data(all_wrong=True), "gdpo", config=_algorithm(positive_advantage_amplification=3.0))
    assert batch.batch["advantages"].count_nonzero() == 0
    assert batch.batch["returns"].count_nonzero() == 0


def test_missing_setting_and_factor_one_never_enter_amplification(monkeypatch):
    import verl.trainer.ppo.advantage_amplification as implementation

    def fail(*args, **kwargs):
        raise AssertionError("The opt-in transform was called for an existing run")

    monkeypatch.setattr(implementation, "amplify_positive_advantages", fail)
    base = compute_advantage(_data(), "gdpo", config=_algorithm())
    explicit_one = compute_advantage(_data(), "gdpo", config=_algorithm(positive_advantage_amplification=1.0))
    for key in base.batch.keys():
        torch.testing.assert_close(base.batch[key], explicit_one.batch[key], rtol=0, atol=0)
    assert base.meta_info == explicit_one.meta_info
    assert "positive_advantage_amplification_metrics" not in base.meta_info
    assert AlgoConfig().positive_advantage_amplification == 1.0


def test_actual_clipped_gspo_gradients_amplify_positive_terms_only():
    cfg = ActorConfig(
        strategy="fsdp", rollout_n=1, use_dynamic_bsz=True, clip_ratio_low=0.2, clip_ratio_high=0.2,
        loss_agg_mode="seq-mean-token-mean",
        policy_loss=PolicyLossConfig(loss_mode="gspo", branch_strata=BranchLossStrataConfig(enabled=True)),
    )
    advantages = torch.tensor([[1., 1.], [-2., -2.], [3., 3.], [-4., -4.]], dtype=torch.float64)
    mask = torch.tensor([[1., 1.], [1., 0.], [1., 1.], [1., 1.]])
    ratios = torch.tensor([1., 1., 1.5, 1.], dtype=torch.float64)
    results, gradients = [], []
    for factor in (1., 3.):
        log_prob = ratios.log().unsqueeze(-1).expand(-1, 2).clone().requires_grad_()
        parts = {}
        loss, _ = compute_policy_loss_gspo(
            torch.zeros_like(log_prob), log_prob, amplify_positive_advantages(advantages, mask, factor), mask,
            config=cfg, branch_policy_loss_weight=torch.tensor([1., 1., 0.5, 0.5]),
            branch_policy_loss_is_branch=torch.tensor([False, False, True, True]), loss_components=parts,
        )
        gradients.append(torch.autograd.grad(loss, log_prob)[0])
        results.append(parts)
    torch.testing.assert_close(gradients[1], torch.where(advantages > 0, gradients[0] * 3, gradients[0]))
    assert gradients[0][0].count_nonzero() > 0
    assert gradients[1][2].count_nonzero() == 0  # positive PPO-clipped branch remains clipped
    for name in results[0]:
        torch.testing.assert_close(results[1][name], results[0][name] * (3 if name.endswith("positive") else 1))
