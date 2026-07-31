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

import random
import unittest

import numpy as np
import pytest
import torch

import verl.trainer.ppo.core_algos
from verl.trainer.ppo.core_algos import (
    _compute_gdpo_component_advantage,
    compute_branch_strata_loss_weights,
    compute_gae_advantage_return,
    compute_gdpo_outcome_advantage,
    compute_grpo_outcome_advantage,
    compute_grpo_vectorized_outcome_advantage,
    compute_policy_loss_gspo,
    compute_rloo_outcome_advantage,
    compute_rloo_vectorized_outcome_advantage,
    get_adv_estimator_fn,
    register_adv_est,
)
from verl.workers.config.actor import ActorConfig, BranchLossStrataConfig, PolicyLossConfig


def mock_test_fn():
    pass


def _branch_strata_actor_config(enabled: bool = True) -> ActorConfig:
    return ActorConfig(
        strategy="fsdp",
        rollout_n=1,
        use_dynamic_bsz=True,
        loss_agg_mode="seq-mean-token-mean",
        policy_loss=PolicyLossConfig(
            loss_mode="gspo",
            branch_strata=BranchLossStrataConfig(enabled=enabled),
        ),
    )


class TestRegisterAdvEst(unittest.TestCase):
    def setUp(self):
        """Clear the registry before each test"""
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY = {
            "gae": lambda x: x * 2,
            "vtrace": lambda x: x + 1,
        }
        self.ADV_ESTIMATOR_REGISTRY = verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY

    def tearDown(self) -> None:
        verl.trainer.ppo.core_algos.ADV_ESTIMATOR_REGISTRY.clear()
        return super().tearDown()

    def test_register_new_function(self):
        """Test registering a new function with a string name"""

        @register_adv_est("test_estimator")
        def test_fn():
            pass

        self.assertIn("test_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_estimator"], test_fn)

    def test_register_with_enum(self):
        """Test registering with an enum value (assuming AdvantageEstimator exists)"""
        from enum import Enum

        class AdvantageEstimator(Enum):
            TEST = "test_enum_estimator"

        @register_adv_est(AdvantageEstimator.TEST)
        def test_fn():
            pass

        self.assertIn("test_enum_estimator", self.ADV_ESTIMATOR_REGISTRY)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["test_enum_estimator"], test_fn)

    def test_duplicate_registration_same_function(self):
        """Test that registering the same function twice doesn't raise an error"""
        register_adv_est("duplicate_test")(mock_test_fn)
        register_adv_est("duplicate_test")(mock_test_fn)

        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["duplicate_test"], mock_test_fn)

    def test_duplicate_registration_different_function(self):
        """Test that registering different functions with same name raises ValueError"""

        @register_adv_est("conflict_test")
        def test_fn1():
            pass

        with self.assertRaises(ValueError):

            @register_adv_est("conflict_test")
            def test_fn2():
                pass

    def test_decorator_preserves_function(self):
        """Test that the decorator returns the original function"""

        def test_fn():
            return "original"

        decorated = register_adv_est("preserve_test")(test_fn)
        self.assertEqual(decorated(), "original")

    def test_multiple_registrations(self):
        """Test registering multiple different functions"""
        init_adv_count = len(self.ADV_ESTIMATOR_REGISTRY)

        @register_adv_est("estimator1")
        def fn1():
            pass

        @register_adv_est("estimator2")
        def fn2():
            pass

        self.assertEqual(len(self.ADV_ESTIMATOR_REGISTRY), 2 + init_adv_count)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator1"], fn1)
        self.assertEqual(self.ADV_ESTIMATOR_REGISTRY["estimator2"], fn2)

    def test_get_adv_estimator_fn_valid_names(self):
        """Test that valid names return the correct function from registry."""
        # Test GAE
        gae_fn = get_adv_estimator_fn("gae")
        assert gae_fn(5) == 10  # 5 * 2 = 10

        # Test Vtrace
        vtrace_fn = get_adv_estimator_fn("vtrace")
        assert vtrace_fn(5) == 6  # 5 + 1 = 6

    def test_get_adv_estimator_fn_invalid_name(self):
        """Test that invalid names raise ValueError."""
        with pytest.raises(ValueError) as excinfo:
            get_adv_estimator_fn("invalid_name")
        assert "Unknown advantage estimator simply: invalid_name" in str(excinfo.value)

    def test_get_adv_estimator_fn_case_sensitive(self):
        """Test that name lookup is case-sensitive."""
        with pytest.raises(ValueError):
            get_adv_estimator_fn("GAE")  # Different case


def test_multi_turn_compute_gae_advantage_return():
    """Test multi-turn GAE skip observation tokens."""
    gamma = random.uniform(0.0, 1.0)
    lam = random.uniform(0.0, 1.0)

    rewards = torch.tensor([[0.0, 0.0, 0.1, 0.1, 0.1, 0.0, 0.0, 0.1, 1.0, 0.0, 0.0]], dtype=torch.float)

    values1 = torch.tensor(
        [
            [
                random.uniform(-100.0, 100.0),
                random.random(),
                4.0,
                5.0,
                6.0,
                random.uniform(-100.0, 0),
                random.random(),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    values2 = torch.tensor(
        [
            [
                random.random(),
                random.uniform(-100.0, 100.0),
                4.0,
                5.0,
                6.0,
                random.random(),
                random.uniform(0.0, 100.0),
                7.0,
                9.0,
                0.0,
                0.0,
            ]
        ],
        dtype=torch.float,
    )

    response_mask = torch.tensor([[0, 0, 1, 1, 1, 0, 0, 1, 1, 0, 0]], dtype=torch.float)

    adv1, ret1 = compute_gae_advantage_return(rewards, values1, response_mask, gamma, lam)
    adv2, ret2 = compute_gae_advantage_return(rewards, values2, response_mask, gamma, lam)

    ret1 *= response_mask
    ret2 *= response_mask
    assert torch.equal(adv1, adv2), f"{adv1=}, {adv2=}"
    assert torch.equal(ret1, ret2), f"{ret1=}, {ret2=}"
    print(f" [CORRECT] \n\n{adv1=}, \n\n{ret1=}")


def _make_group_index(batch_size: int, num_groups: int) -> np.ndarray:
    """Create a numpy index array ensuring each group has at least 2 samples."""
    assert num_groups * 2 <= batch_size, "batch_size must allow >=2 samples per group"
    counts: list[int] = [2] * num_groups
    remaining = batch_size - 2 * num_groups
    for _ in range(remaining):
        counts[random.randrange(num_groups)] += 1
    index = []
    for gid, c in enumerate(counts):
        index.extend([gid] * c)
    random.shuffle(index)
    return np.asarray(index, dtype=np.int64)


def _rand_mask(batch_size: int, seq_len: int) -> torch.Tensor:
    mask = torch.randint(0, 2, (batch_size, seq_len), dtype=torch.int64).float()
    rows_without_one = (mask.sum(dim=-1) == 0).nonzero(as_tuple=True)[0]
    if len(rows_without_one) > 0:
        mask[rows_without_one, -1] = 1.0
    return mask


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_rloo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    index = _make_group_index(batch_size, num_groups)
    response_mask = _rand_mask(batch_size, seq_len)
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask
    adv1, ret1 = compute_rloo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    adv2, ret2 = compute_rloo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )
    # Print concise diagnostics for visibility during test runs
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[RLOO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize(
    "batch_size,seq_len,num_groups,seed",
    [
        (64, 128, 5, 0),
        (128, 256, 8, 1),
        (512, 512, 10, 2),
    ],
)
def test_grpo_and_vectorized_equivalence(batch_size: int, seq_len: int, num_groups: int, seed: int):
    # Set seeds for reproducibility
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # Generate group indices (numpy array of shape [batch_size])
    index = _make_group_index(batch_size, num_groups)

    # Generate binary response mask (at least one valid token per row)
    response_mask = _rand_mask(batch_size, seq_len)

    # Generate token-level rewards and apply mask
    base_rewards = torch.randn(batch_size, seq_len, dtype=torch.float32)
    token_level_rewards = base_rewards * response_mask

    # Compute GRPO outcome advantage (original implementation)
    adv1, ret1 = compute_grpo_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Compute GRPO outcome advantage (vectorized implementation)
    adv2, ret2 = compute_grpo_vectorized_outcome_advantage(
        token_level_rewards=token_level_rewards,
        response_mask=response_mask,
        index=index,
    )

    # Diagnostic info for visibility (same style as RLOO test)
    adv_max_diff = (adv1 - adv2).abs().max().item()
    ret_max_diff = (ret1 - ret2).abs().max().item()
    total_mask_tokens = int(response_mask.sum().item())
    print(
        f"[GRPO] seed={seed} groups={num_groups} shape={adv1.shape} "
        f"mask_tokens={total_mask_tokens} adv_max_diff={adv_max_diff:.3e} ret_max_diff={ret_max_diff:.3e}"
    )

    # Assert shape and numerical equivalence
    assert adv1.shape == adv2.shape == (batch_size, seq_len)
    assert ret1.shape == ret2.shape == (batch_size, seq_len)
    assert torch.allclose(adv1, adv2, rtol=1e-5, atol=1e-6)
    assert torch.allclose(ret1, ret2, rtol=1e-5, atol=1e-6)


def _make_gdpo_test_batch(num_rows: int, response_len: int = 3) -> dict[str, torch.Tensor]:
    prompt_len = 2
    return {
        "prompts": torch.ones((num_rows, prompt_len), dtype=torch.int64),
        "attention_mask": torch.ones((num_rows, prompt_len + response_len), dtype=torch.int64),
    }


def _make_scalar_reward_component(values: list[float], response_mask: torch.Tensor) -> torch.Tensor:
    rewards = torch.zeros_like(response_mask, dtype=torch.float32)
    rewards[:, -1] = torch.tensor(values, dtype=torch.float32)
    return rewards


def _stable_gdpo_config(**overrides) -> dict:
    config = {
        "gdpo_reward_keys": ["accuracy"],
        "gdpo_reward_weights": [1.0],
        "gdpo_baseline_weight_key": "gdpo_baseline_weight",
        "gdpo_stability_enabled": True,
        "gdpo_component_std_floor": 0.05,
        "gdpo_component_adv_clip": 5.0,
        "gdpo_final_std_floor": 0.05,
        "gdpo_final_adv_clip": 10.0,
    }
    config.update(overrides)
    return config


def _response_advantages(advantages: torch.Tensor, response_mask: torch.Tensor) -> torch.Tensor:
    lengths = response_mask.sum(dim=-1).clamp_min(1)
    return (advantages * response_mask).sum(dim=-1) / lengths


def test_gdpo_without_eligibility_matches_existing_component_sum_behavior():
    response_mask = torch.ones((4, 3), dtype=torch.float32)
    index = np.array(["prompt-a", "prompt-a", "prompt-b", "prompt-b"], dtype=object)
    batch = _make_gdpo_test_batch(num_rows=4)
    non_tensor_batch = {
        "accuracy": np.array([1.0, 0.0, 0.0, 1.0], dtype=np.float32),
        "format": np.array([1.0, 1.0, 0.0, 0.0], dtype=np.float32),
    }
    config = {
        "gdpo_reward_keys": ["accuracy", "format"],
        "gdpo_reward_weights": [0.9, 0.1],
    }

    adv, returns = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=index,
        config=config,
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )

    acc_adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=_make_scalar_reward_component([1.0, 0.0, 0.0, 1.0], response_mask),
        response_mask=response_mask,
        index=index,
    )
    fmt_adv, _ = compute_grpo_outcome_advantage(
        token_level_rewards=_make_scalar_reward_component([1.0, 1.0, 0.0, 0.0], response_mask),
        response_mask=response_mask,
        index=index,
    )
    expected = verl.trainer.ppo.core_algos.verl_F.masked_whiten(0.9 * acc_adv + 0.1 * fmt_adv, response_mask)
    expected = expected * response_mask

    assert adv.shape == returns.shape == response_mask.shape
    assert torch.allclose(adv, expected, rtol=1e-5, atol=1e-6)
    assert torch.allclose(returns, expected, rtol=1e-5, atol=1e-6)


def test_gdpo_component_eligibility_normalizes_tool_efficiency_only_over_correct_rows():
    response_mask = torch.ones((4, 3), dtype=torch.float32)
    index = np.array(["prompt-a"] * 4, dtype=object)
    tool_rewards = _make_scalar_reward_component([1.0, 0.5, 1.0 / 3.0, 0.25], response_mask)
    correct_mask = torch.tensor([True, False, True, False])

    component_adv = _compute_gdpo_component_advantage(
        token_level_rewards=tool_rewards,
        response_mask=response_mask,
        index=index,
        eligible_mask=correct_mask,
        min_group_size=2,
    )
    scalar_adv = component_adv[:, -1]

    assert scalar_adv[0] > scalar_adv[2]
    assert torch.equal(scalar_adv[[1, 3]], torch.zeros(2))
    assert torch.all(component_adv[:, :-1] == component_adv[:, -1:])


def test_gdpo_component_eligibility_zeroes_tool_channel_when_too_few_correct_rows():
    response_mask = torch.ones((3, 3), dtype=torch.float32)
    index = np.array(["prompt-a"] * 3, dtype=object)
    tool_rewards = _make_scalar_reward_component([1.0, 0.5, 1.0 / 3.0], response_mask)
    correct_mask = torch.tensor([False, True, False])

    component_adv = _compute_gdpo_component_advantage(
        token_level_rewards=tool_rewards,
        response_mask=response_mask,
        index=index,
        eligible_mask=correct_mask,
        min_group_size=2,
    )

    assert torch.equal(component_adv, torch.zeros_like(component_adv))


def test_gdpo_component_eligibility_is_group_local():
    response_mask = torch.ones((6, 3), dtype=torch.float32)
    index = np.array(["prompt-a", "prompt-a", "prompt-a", "prompt-b", "prompt-b", "prompt-b"], dtype=object)
    tool_rewards = _make_scalar_reward_component([1.0, 0.5, 0.25, 1.0, 0.5, 0.25], response_mask)
    correct_mask = torch.tensor([True, True, False, True, False, False])

    component_adv = _compute_gdpo_component_advantage(
        token_level_rewards=tool_rewards,
        response_mask=response_mask,
        index=index,
        eligible_mask=correct_mask,
        min_group_size=2,
    )
    scalar_adv = component_adv[:, -1]

    assert scalar_adv[0] > scalar_adv[1]
    assert scalar_adv[2] == 0
    assert torch.equal(scalar_adv[3:], torch.zeros(3))


def test_gdpo_conditional_tool_efficiency_uses_eligibility_config():
    response_mask = torch.ones((4, 3), dtype=torch.float32)
    index = np.array(["prompt-a"] * 4, dtype=object)
    batch = _make_gdpo_test_batch(num_rows=4)
    non_tensor_batch = {
        "tool_efficiency": np.array([1.0, 0.5, 1.0 / 3.0, 0.25], dtype=np.float32),
        "accuracy": np.array([1.0, 0.0, 1.0, 0.0], dtype=np.float32),
    }
    config = {
        "gdpo_reward_keys": ["tool_efficiency"],
        "gdpo_reward_weights": [1.0],
        "gdpo_reward_eligibility": {
            "tool_efficiency": {
                "key": "accuracy",
                "threshold": 1.0,
                "min_group_size": 2,
            },
        },
    }

    adv, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=index,
        config=config,
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )

    scalar_adv = adv[:, -1]
    assert scalar_adv[0] > scalar_adv[2]
    assert torch.allclose(scalar_adv[1], scalar_adv[3])


def test_gdpo_baseline_weight_excludes_branch_from_component_and_final_stats():
    canonical_rewards = [0.8, 0.7, 0.9, 0.6]
    branch_reward = 0.0
    response_mask = torch.ones((5, 3), dtype=torch.float32)
    index = np.array(["prompt-a"] * 5, dtype=object)
    batch = _make_gdpo_test_batch(num_rows=5)
    non_tensor_batch = {
        "accuracy": np.array(canonical_rewards + [branch_reward], dtype=np.float32),
        "gdpo_baseline_weight": np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
    }
    config = {
        "gdpo_reward_keys": ["accuracy"],
        "gdpo_reward_weights": [1.0],
        "gdpo_baseline_weight_key": "gdpo_baseline_weight",
    }

    component_adv = _compute_gdpo_component_advantage(
        token_level_rewards=_make_scalar_reward_component(canonical_rewards + [branch_reward], response_mask),
        response_mask=response_mask,
        index=index,
        baseline_mask=torch.tensor([True, True, True, True, False]),
    )
    canonical = torch.tensor(canonical_rewards, dtype=torch.float32)
    expected_branch = (branch_reward - canonical.mean()) / (canonical.std() + 1e-6)

    assert torch.allclose(component_adv[4, -1], expected_branch, rtol=1e-5, atol=1e-6)

    adv_with_branch, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=index,
        config=config,
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )
    adv_without_branch, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros((4, 3), dtype=torch.float32),
        response_mask=torch.ones((4, 3), dtype=torch.float32),
        index=np.array(["prompt-a"] * 4, dtype=object),
        config={"gdpo_reward_keys": ["accuracy"], "gdpo_reward_weights": [1.0]},
        non_tensor_batch={"accuracy": np.array(canonical_rewards, dtype=np.float32)},
        batch=_make_gdpo_test_batch(num_rows=4),
    )

    assert torch.allclose(adv_with_branch[:4], adv_without_branch, rtol=1e-5, atol=1e-6)


def test_gdpo_baseline_weight_padding_rows_do_not_change_canonical_advantages():
    canonical_rewards = [0.8, 0.7, 0.9, 0.6]
    branch_reward = 0.0
    response_mask = torch.ones((8, 3), dtype=torch.float32)
    response_mask[5:] = 0
    index = np.array(["prompt-a"] * 8, dtype=object)
    batch = _make_gdpo_test_batch(num_rows=8)
    non_tensor_batch = {
        "accuracy": np.array(canonical_rewards + [branch_reward, 0.0, 0.0, 0.0], dtype=np.float32),
        "gdpo_baseline_weight": np.array([1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
    }
    config = {
        "gdpo_reward_keys": ["accuracy"],
        "gdpo_reward_weights": [1.0],
        "gdpo_baseline_weight_key": "gdpo_baseline_weight",
    }

    adv_with_padding, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=index,
        config=config,
        non_tensor_batch=non_tensor_batch,
        batch=batch,
    )
    adv_without_padding, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros((5, 3), dtype=torch.float32),
        response_mask=torch.ones((5, 3), dtype=torch.float32),
        index=np.array(["prompt-a"] * 5, dtype=object),
        config=config,
        non_tensor_batch={
            "accuracy": np.array(canonical_rewards + [branch_reward], dtype=np.float32),
            "gdpo_baseline_weight": np.array([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        },
        batch=_make_gdpo_test_batch(num_rows=5),
    )

    assert torch.allclose(adv_with_padding[:5], adv_without_padding, rtol=1e-5, atol=1e-6)
    assert torch.all(adv_with_padding[5:] == 0)


def test_stable_gdpo_constant_canonical_rewards_preserve_bounded_branch_signal():
    response_mask = torch.ones((5, 3), dtype=torch.float32)
    rewards = [1.0, 1.0, 1.0, 1.0, 0.0]
    diagnostics = {}
    adv, returns = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=np.array(["prompt-a"] * 5, dtype=object),
        config=_stable_gdpo_config(),
        non_tensor_batch={
            "accuracy": np.asarray(rewards, dtype=np.float32),
            "gdpo_baseline_weight": np.asarray([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        },
        batch=_make_gdpo_test_batch(num_rows=5),
        diagnostics=diagnostics,
    )

    expected = torch.tensor([0.0, 0.0, 0.0, 0.0, -1.0])
    assert torch.equal(_response_advantages(adv, response_mask), expected)
    assert torch.equal(returns, adv)
    assert diagnostics["gdpo/stability/accuracy/low_variance_group_rate"] == 1.0
    assert diagnostics["gdpo/stability/final_low_variance_fallback"] == 1.0
    assert diagnostics["gdpo/stability/final_max_abs"] == 1.0


def test_stable_gdpo_constant_canonical_and_branch_rewards_produce_zero():
    response_mask = torch.ones((5, 2), dtype=torch.float32)
    adv, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=np.array(["prompt-a"] * 5, dtype=object),
        config=_stable_gdpo_config(),
        non_tensor_batch={
            "accuracy": np.ones(5, dtype=np.float32),
            "gdpo_baseline_weight": np.asarray([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        },
        batch=_make_gdpo_test_batch(num_rows=5, response_len=2),
    )
    assert torch.equal(adv, torch.zeros_like(adv))


def test_stable_gdpo_nearly_constant_component_uses_raw_centered_difference():
    response_mask = torch.ones((5, 2), dtype=torch.float32)
    rewards = [0.99, 1.0, 1.0, 1.0, 0.0]
    adv, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=np.array(["prompt-a"] * 5, dtype=object),
        config=_stable_gdpo_config(),
        non_tensor_batch={
            "accuracy": np.asarray(rewards, dtype=np.float32),
            "gdpo_baseline_weight": np.asarray([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        },
        batch=_make_gdpo_test_batch(num_rows=5, response_len=2),
    )

    scalars = _response_advantages(adv, response_mask)
    assert torch.allclose(scalars[:4], torch.tensor([-0.0075, 0.0025, 0.0025, 0.0025]), atol=1e-6)
    assert torch.allclose(scalars[4], torch.tensor(-0.9975), atol=1e-6)


def test_stable_gdpo_healthy_variance_keeps_gdpo_standardization():
    response_mask = torch.ones((5, 2), dtype=torch.float32)
    rewards = [1.0, 0.0, 1.0, 0.0, 0.0]
    adv, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=np.array(["prompt-a"] * 5, dtype=object),
        config=_stable_gdpo_config(),
        non_tensor_batch={
            "accuracy": np.asarray(rewards, dtype=np.float32),
            "gdpo_baseline_weight": np.asarray([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        },
        batch=_make_gdpo_test_batch(num_rows=5, response_len=2),
    )

    expected = torch.tensor([0.8660254, -0.8660254, 0.8660254, -0.8660254, -0.8660254])
    assert torch.allclose(_response_advantages(adv, response_mask), expected, atol=1e-5)


def test_stable_gdpo_without_component_std_uses_raw_centered_differences():
    response_mask = torch.ones((5, 2), dtype=torch.float32)
    rewards = [1.0, 0.0, 1.0, 0.0, 0.0]
    adv, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=np.array(["prompt-a"] * 5, dtype=object),
        norm_adv_by_std_in_grpo=False,
        config=_stable_gdpo_config(gdpo_final_std_floor=2.0),
        non_tensor_batch={
            "accuracy": np.asarray(rewards, dtype=np.float32),
            "gdpo_baseline_weight": np.asarray([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32),
        },
        batch=_make_gdpo_test_batch(num_rows=5, response_len=2),
    )
    expected = torch.tensor([0.5, -0.5, 0.5, -0.5, -0.5])
    assert torch.allclose(_response_advantages(adv, response_mask), expected, atol=1e-6)


def test_stable_gdpo_component_and_final_clips_bound_excluded_branch():
    response_mask = torch.ones((5, 2), dtype=torch.float32)
    baseline_weights = np.asarray([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    index = np.array(["prompt-a"] * 5, dtype=object)
    batch = _make_gdpo_test_batch(num_rows=5, response_len=2)

    component_diagnostics = {}
    component_adv, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=index,
        config=_stable_gdpo_config(),
        non_tensor_batch={
            "accuracy": np.asarray([1.0, 1.0, 1.0, 1.0, -100.0], dtype=np.float32),
            "gdpo_baseline_weight": baseline_weights,
        },
        batch=batch,
        diagnostics=component_diagnostics,
    )
    assert _response_advantages(component_adv, response_mask)[-1] == -5.0
    assert component_diagnostics["gdpo/stability/component_clipped_row_rate"] == 0.2

    final_diagnostics = {}
    final_adv, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=index,
        config=_stable_gdpo_config(gdpo_component_adv_clip=None),
        non_tensor_batch={
            "accuracy": np.asarray([1.0, 0.0, 1.0, 0.0, -100.0], dtype=np.float32),
            "gdpo_baseline_weight": baseline_weights,
        },
        batch=batch,
        diagnostics=final_diagnostics,
    )
    assert _response_advantages(final_adv, response_mask)[-1] == -10.0
    assert final_diagnostics["gdpo/stability/final_clipped_row_rate"] == 0.2


def test_stable_gdpo_response_statistics_ignore_lengths_padding_and_branch_multiplicity():
    rewards = np.asarray([1.0, 0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    weights = np.asarray([1.0, 1.0, 1.0, 1.0, 0.0], dtype=np.float32)
    index = np.array(["prompt-a"] * 5, dtype=object)
    short_mask = torch.ones((5, 4), dtype=torch.float32)
    varied_mask = torch.tensor(
        [[1, 0, 0, 0], [1, 1, 0, 0], [1, 1, 1, 0], [1, 1, 1, 1], [1, 0, 0, 0]], dtype=torch.float32
    )

    def compute(mask, reward_values, baseline_weights, groups):
        attention_mask = torch.cat((torch.ones((mask.size(0), 2), dtype=torch.int64), mask.long()), dim=-1)
        adv, _ = compute_gdpo_outcome_advantage(
            token_level_rewards=torch.zeros_like(mask),
            response_mask=mask,
            index=groups,
            config=_stable_gdpo_config(),
            non_tensor_batch={"accuracy": reward_values, "gdpo_baseline_weight": baseline_weights},
            batch={"prompts": torch.ones((mask.size(0), 2), dtype=torch.int64), "attention_mask": attention_mask},
        )
        return _response_advantages(adv, mask)

    short = compute(short_mask, rewards, weights, index)
    varied = compute(varied_mask, rewards, weights, index)
    assert torch.allclose(short, varied, atol=1e-6)

    expanded_mask = torch.cat((varied_mask, varied_mask[4:].repeat(3, 1), torch.zeros((2, 4))), dim=0)
    expanded_rewards = np.concatenate((rewards, np.zeros(5, dtype=np.float32)))
    expanded_weights = np.concatenate((weights, np.zeros(5, dtype=np.float32)))
    expanded_groups = np.array(["prompt-a"] * 10, dtype=object)
    expanded = compute(expanded_mask, expanded_rewards, expanded_weights, expanded_groups)
    assert torch.allclose(expanded[:5], varied, atol=1e-6)
    assert torch.all(expanded[-2:] == 0)


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"gdpo_component_std_floor": -0.1}, "gdpo_component_std_floor"),
        ({"gdpo_final_std_floor": float("nan")}, "gdpo_final_std_floor"),
        ({"gdpo_component_adv_clip": 0.0}, "gdpo_component_adv_clip"),
        ({"gdpo_final_adv_clip": float("inf")}, "gdpo_final_adv_clip"),
    ],
)
def test_stable_gdpo_rejects_invalid_stability_config(override, message):
    response_mask = torch.ones((2, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match=message):
        compute_gdpo_outcome_advantage(
            token_level_rewards=torch.zeros_like(response_mask),
            response_mask=response_mask,
            index=np.array(["prompt-a"] * 2, dtype=object),
            config=_stable_gdpo_config(**override),
            non_tensor_batch={
                "accuracy": np.asarray([1.0, 0.0], dtype=np.float32),
                "gdpo_baseline_weight": np.asarray([1.0, 1.0], dtype=np.float32),
            },
            batch=_make_gdpo_test_batch(num_rows=2, response_len=2),
        )


def test_gdpo_uses_assistant_turn_overlong_as_an_independent_component():
    response_mask = torch.ones((2, 2), dtype=torch.float32)
    config = _stable_gdpo_config(
        gdpo_reward_keys=["accuracy", "assistant_turn_overlong"],
        gdpo_reward_weights=[0.8, 0.2],
    )
    advantages, _ = compute_gdpo_outcome_advantage(
        token_level_rewards=torch.zeros_like(response_mask),
        response_mask=response_mask,
        index=np.array(["prompt-a", "prompt-a"], dtype=object),
        config=config,
        non_tensor_batch={
            "accuracy": np.asarray([1.0, 1.0], dtype=np.float32),
            "assistant_turn_overlong": np.asarray([0.0, -1.0], dtype=np.float32),
            "gdpo_baseline_weight": np.asarray([1.0, 1.0], dtype=np.float32),
        },
        batch=_make_gdpo_test_batch(num_rows=2, response_len=2),
    )

    response_advantages = _response_advantages(advantages, response_mask)
    assert response_advantages[0] > response_advantages[1]


def test_stable_gdpo_randomized_branch_heavy_batches_are_finite_and_bounded():
    rng = np.random.default_rng(1234)
    for _ in range(20):
        canonical_count = 8
        branch_count = int(rng.integers(1, 17))
        padding_count = int(rng.integers(0, 4))
        num_rows = canonical_count + branch_count + padding_count
        response_mask = torch.zeros((num_rows, 7), dtype=torch.float32)
        for row in range(canonical_count + branch_count):
            response_mask[row, : int(rng.integers(1, 8))] = 1
        accuracy = np.concatenate(
            (rng.integers(0, 2, canonical_count), rng.integers(0, 2, branch_count), np.zeros(padding_count))
        ).astype(np.float32)
        fmt_base = float(rng.choice([0.0, 0.99, 1.0]))
        format_rewards = np.concatenate(
            (
                np.clip(fmt_base + rng.normal(0, 0.002, canonical_count), 0, 1),
                rng.uniform(0, 1, branch_count),
                np.zeros(padding_count),
            )
        ).astype(np.float32)
        baseline_weights = np.concatenate((np.ones(canonical_count), np.zeros(branch_count + padding_count))).astype(
            np.float32
        )
        attention_mask = torch.cat((torch.ones((num_rows, 2), dtype=torch.int64), response_mask.long()), dim=-1)
        config = _stable_gdpo_config(gdpo_reward_keys=["accuracy", "format"], gdpo_reward_weights=[0.9, 0.1])
        adv, _ = compute_gdpo_outcome_advantage(
            token_level_rewards=torch.zeros_like(response_mask),
            response_mask=response_mask,
            index=np.array(["prompt-a"] * num_rows, dtype=object),
            config=config,
            non_tensor_batch={
                "accuracy": accuracy,
                "format": format_rewards,
                "gdpo_baseline_weight": baseline_weights,
            },
            batch={"prompts": torch.ones((num_rows, 2), dtype=torch.int64), "attention_mask": attention_mask},
        )
        assert torch.isfinite(adv).all()
        assert float(adv.abs().max()) <= 10.0
        assert torch.all(adv[response_mask == 0] == 0)


def test_multiturn_tool_reward_returns_tool_efficiency_without_changing_score():
    from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import compute_score

    response = (
        "<think>zoom</think>"
        '<tool_call>\n{"name": "image_zoom_in", "arguments": {"bbox": [0, 0, 1, 1]}}\n</tool_call>'
        "<tool_response>ok</tool_response>"
        "<think>answer</think>"
        "<answer>\\boxed{red}</answer>"
    )

    result = compute_score(
        response,
        "red",
        format_score=0.1,
        min_tool_calls_for_bonus=1,
        tool_call_bonus=0.1,
        extra_info={"reward_type": "short_string"},
    )

    assert result["tool_call_count"] == 1
    assert result["tool_efficiency"] == 0.5
    assert result["tool_call_bonus"] == 0.1
    assert result["score"] == pytest.approx(1.1)


def test_multiturn_tool_efficiency_score_weight_changes_scalar_score():
    from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import compute_score

    response = (
        "<think>first zoom</think>"
        '<tool_call>\n{"name": "image_zoom_in", "arguments": {"bbox": [0, 0, 1, 1]}}\n</tool_call>'
        "<tool_response>ok</tool_response>"
        "<think>second zoom</think>"
        '<tool_call>\n{"name": "image_zoom_in", "arguments": {"bbox": [1, 1, 2, 2]}}\n</tool_call>'
        "<tool_response>ok</tool_response>"
        "<think>answer</think>"
        "<answer>\\boxed{red}</answer>"
    )

    result = compute_score(
        response,
        "red",
        format_score=0.1,
        min_tool_calls_for_bonus=None,
        tool_call_bonus=0.0,
        tool_efficiency_reward_mode="concave_length_penalty",
        tool_efficiency_free_calls=1,
        tool_efficiency_concave_k=10.0,
        tool_efficiency_concave_q=0.95,
        tool_efficiency_score_weight=0.1,
        extra_info={"reward_type": "short_string"},
    )

    assert result["tool_call_bonus"] == 0.0
    assert result["score"] == pytest.approx(0.8 + 0.1 + 0.1 * result["tool_efficiency"])


def test_tool_efficiency_default_mode_matches_inverse_count():
    from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
        tool_efficiency_reward,
    )

    assert tool_efficiency_reward(0) == pytest.approx(1.0)
    assert tool_efficiency_reward(1) == pytest.approx(0.5)
    assert tool_efficiency_reward(9) == pytest.approx(0.1)


def test_tool_efficiency_inverse_count_respects_free_calls():
    from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
        tool_efficiency_reward,
    )

    assert tool_efficiency_reward(0, mode="inverse_count", free_calls=1) == pytest.approx(1.0)
    assert tool_efficiency_reward(1, mode="inverse_count", free_calls=1) == pytest.approx(1.0)
    assert tool_efficiency_reward(2, mode="inverse_count", free_calls=1) == pytest.approx(0.5)


def test_tool_efficiency_concave_penalty_respects_free_calls_and_is_monotonic():
    from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
        tool_efficiency_reward,
    )

    rewards = [
        tool_efficiency_reward(
            tool_calls,
            mode="concave_length_penalty",
            free_calls=1,
            concave_k=10.0,
            concave_q=0.95,
        )
        for tool_calls in range(11)
    ]

    assert rewards[0] == pytest.approx(0.0)
    assert rewards[1] == pytest.approx(0.0)
    assert all(rewards[i] >= rewards[i + 1] for i in range(len(rewards) - 1))

    early_drop = rewards[1] - rewards[2]
    late_drop = rewards[9] - rewards[10]
    assert late_drop < early_drop


def test_tool_efficiency_concave_penalty_log_limit_for_q_one():
    from verl.utils.reward_score.math_verify_reward_type_multiturn_tool_calling_optional_think import (
        tool_efficiency_reward,
    )

    reward = tool_efficiency_reward(
        3,
        mode="concave_length_penalty",
        free_calls=1,
        concave_k=10.0,
        concave_q=1.0,
    )

    assert reward == pytest.approx(-torch.log1p(torch.tensor(20.0)).item() / 10.0)


def test_branch_strata_weights_parent_balance_and_fraction():
    response_mask = torch.ones((6, 2), dtype=torch.float32)
    branch_flags = np.asarray([False, False, False, True, True, True], dtype=object)
    parent_ids = np.asarray(["a", "b", "c", "a", "a", "b"], dtype=object)
    weights, metrics = compute_branch_strata_loss_weights(
        response_mask=response_mask,
        branch_flags=branch_flags,
        parent_ids=parent_ids,
        padding_mask=np.zeros(6, dtype=bool),
        branch_fraction=0.2,
    )

    assert weights.sum().item() == pytest.approx(6.0)
    assert weights[:3].sum().item() / 6.0 == pytest.approx(0.8)
    assert weights[3:].sum().item() / 6.0 == pytest.approx(0.2)
    assert weights[3].item() + weights[4].item() == pytest.approx(weights[5].item())
    assert metrics["effective_fraction"] == pytest.approx(0.2)
    assert metrics["branch_parents"] == 2


def test_branch_strata_weights_cap_rare_branch_and_ignore_padding():
    response_mask = torch.ones((6, 2), dtype=torch.float32)
    response_mask[-1] = 0
    weights, metrics = compute_branch_strata_loss_weights(
        response_mask=response_mask,
        branch_flags=np.asarray([False, False, False, False, True, None], dtype=object),
        parent_ids=np.asarray(["a", "b", "c", "d", "a", None], dtype=object),
        padding_mask=np.asarray([False, False, False, False, False, True]),
        branch_fraction=0.4,
    )

    assert metrics["effective_fraction"] == pytest.approx(0.25)
    assert weights[-1].item() == 0.0
    assert weights.sum().item() == pytest.approx(6.0)
    assert weights[4].item() <= 6.0 / 4.0


def test_branch_strata_weights_no_branches_recover_canonical_mean():
    response_mask = torch.ones((4, 2), dtype=torch.float32)
    weights, metrics = compute_branch_strata_loss_weights(
        response_mask=response_mask,
        branch_flags=np.asarray([False] * 4, dtype=object),
        parent_ids=np.asarray(["a", "b", "c", "d"], dtype=object),
        padding_mask=np.zeros(4, dtype=bool),
        branch_fraction=0.2,
    )
    assert torch.equal(weights, torch.ones(4))
    assert metrics["effective_fraction"] == 0.0


def test_branch_strata_zero_fraction_removes_branch_weight():
    weights, metrics = compute_branch_strata_loss_weights(
        response_mask=torch.ones((3, 2), dtype=torch.float32),
        branch_flags=np.asarray([False, False, True], dtype=object),
        parent_ids=np.asarray(["a", "b", "a"], dtype=object),
        padding_mask=np.zeros(3, dtype=bool),
        branch_fraction=0.0,
    )

    assert torch.equal(weights, torch.tensor([1.5, 1.5, 0.0]))
    assert metrics["effective_fraction"] == 0.0
    assert metrics["branch_weight_share"] == 0.0


@pytest.mark.parametrize("malformed_flag", [1, "true", None])
def test_branch_strata_weights_reject_malformed_active_flag(malformed_flag):
    with pytest.raises(ValueError, match="non-boolean branch flag"):
        compute_branch_strata_loss_weights(
            response_mask=torch.ones((2, 2), dtype=torch.float32),
            branch_flags=np.asarray([False, malformed_flag], dtype=object),
            parent_ids=np.asarray(["a", "a"], dtype=object),
            padding_mask=np.zeros(2, dtype=bool),
            branch_fraction=0.2,
        )


def test_branch_strata_weights_reject_branch_without_divergence_tokens():
    with pytest.raises(ValueError, match="positive divergence mask"):
        compute_branch_strata_loss_weights(
            response_mask=torch.tensor([[1, 1], [0, 0]], dtype=torch.float32),
            branch_flags=np.asarray([False, True], dtype=object),
            parent_ids=np.asarray(["a", "a"], dtype=object),
            padding_mask=np.zeros(2, dtype=bool),
            branch_fraction=0.2,
        )


def test_branch_strata_allows_sft_only_canonical_anchor_when_enabled():
    weights, metrics = compute_branch_strata_loss_weights(
        response_mask=torch.tensor([[0, 0], [1, 1], [1, 0]], dtype=torch.float32),
        branch_flags=np.asarray([False, True, False], dtype=object),
        parent_ids=np.asarray(["sft-parent", "sft-parent", "active-parent"], dtype=object),
        padding_mask=np.zeros(3, dtype=bool),
        branch_fraction=0.2,
        allow_inactive_canonical=True,
    )

    assert weights[0].item() == 0.0
    assert weights[1].item() > 0.0
    assert weights[2].item() > 0.0
    assert metrics["inactive_canonical_rows"] == 1.0


@pytest.mark.parametrize("parent_ids", [["a", "b", "missing"], ["a", "a", "a"]])
def test_branch_strata_weights_reject_missing_or_duplicate_parent(parent_ids):
    response_mask = torch.ones((3, 2), dtype=torch.float32)
    with pytest.raises(ValueError, match="exactly one canonical"):
        compute_branch_strata_loss_weights(
            response_mask=response_mask,
            branch_flags=np.asarray([False, False, True], dtype=object),
            parent_ids=np.asarray(parent_ids, dtype=object),
            padding_mask=np.zeros(3, dtype=bool),
            branch_fraction=0.2,
        )


def test_gspo_branch_strata_weighted_loss_and_suffix_mask():
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=torch.float32)
    scalar_advantages = torch.tensor([1.0, 3.0, 5.0, 7.0]).unsqueeze(-1)
    advantages = scalar_advantages.expand(-1, 3).clone()
    advantages[2:, :2] = 1_000_000.0
    weights = torch.tensor([1.6, 1.6, 0.4, 0.4])
    is_branch = torch.tensor([False, False, True, True])
    zeros = torch.zeros_like(advantages)

    loss, metrics = compute_policy_loss_gspo(
        old_log_prob=zeros,
        log_prob=zeros,
        advantages=advantages,
        response_mask=response_mask,
        config=_branch_strata_actor_config(),
        branch_policy_loss_weight=weights,
        branch_policy_loss_is_branch=is_branch,
    )

    assert loss.item() == pytest.approx(-2.8)
    assert metrics["actor/pg_loss_canonical_unweighted"] == pytest.approx(-2.0)
    assert metrics["actor/pg_loss_branch_unweighted"] == pytest.approx(-6.0)


def test_gspo_trainable_prefill_mask_controls_gradient():
    old_log_prob = torch.zeros((1, 4), dtype=torch.float32)
    advantages = torch.ones((1, 4), dtype=torch.float32)
    config = _branch_strata_actor_config(enabled=False)

    masked_log_prob = torch.zeros((1, 4), dtype=torch.float32, requires_grad=True)
    masked_prefill = torch.tensor([[0, 0, 1, 1]], dtype=torch.float32)
    masked_loss, _ = compute_policy_loss_gspo(
        old_log_prob=old_log_prob,
        log_prob=masked_log_prob,
        advantages=advantages,
        response_mask=masked_prefill,
        config=config,
    )
    masked_loss.backward()

    trainable_log_prob = torch.zeros((1, 4), dtype=torch.float32, requires_grad=True)
    trainable_prefill = torch.ones((1, 4), dtype=torch.float32)
    trainable_loss, _ = compute_policy_loss_gspo(
        old_log_prob=old_log_prob,
        log_prob=trainable_log_prob,
        advantages=advantages,
        response_mask=trainable_prefill,
        config=config,
    )
    trainable_loss.backward()

    assert torch.equal(masked_log_prob.grad[0, :2], torch.zeros(2))
    assert torch.all(masked_log_prob.grad[0, 2:].abs() > 0)
    assert torch.all(trainable_log_prob.grad[0, :2].abs() > 0)


def test_gspo_branch_strata_dynamic_microbatch_accumulation_is_invariant():
    response_mask = torch.tensor([[1, 1, 1], [1, 1, 1], [0, 0, 1], [0, 0, 1]], dtype=torch.float32)
    advantages = torch.tensor([1.0, 3.0, 5.0, 7.0]).view(-1, 1).expand(-1, 3)
    weights = torch.tensor([1.6, 1.6, 0.4, 0.4])
    is_branch = torch.tensor([False, False, True, True])
    zeros = torch.zeros_like(advantages)
    config = _branch_strata_actor_config()

    full, _ = compute_policy_loss_gspo(
        old_log_prob=zeros,
        log_prob=zeros,
        advantages=advantages,
        response_mask=response_mask,
        config=config,
        branch_policy_loss_weight=weights,
        branch_policy_loss_is_branch=is_branch,
    )
    accumulated = full.new_zeros(())
    for rows in (slice(0, 3), slice(3, 4)):
        micro, _ = compute_policy_loss_gspo(
            old_log_prob=zeros[rows],
            log_prob=zeros[rows],
            advantages=advantages[rows],
            response_mask=response_mask[rows],
            config=config,
            branch_policy_loss_weight=weights[rows],
            branch_policy_loss_is_branch=is_branch[rows],
        )
        accumulated += micro * (response_mask[rows].shape[0] / response_mask.shape[0])
    assert accumulated.item() == pytest.approx(full.item())


def test_compute_policy_loss_flow_grpo() -> None:
    """Test flow-GRPO policy loss computation."""

    # prepare input
    batch_size = 8
    steps = 10
    rollout_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    current_log_probs = torch.randn((batch_size, steps), dtype=torch.float32)
    advantages = torch.randn((batch_size, steps), dtype=torch.float32)
    response_mask = torch.ones((batch_size, steps), dtype=torch.int32)
    import os

    from hydra import compose, initialize_config_dir

    from verl.trainer.ppo.diffusion_algos import compute_policy_loss_flow_grpo
    from verl.utils.config import omega_conf_to_dataclass
    from verl.workers.config.actor import FSDPActorConfig

    with initialize_config_dir(config_dir=os.path.abspath("verl/trainer/config/actor")):
        cfg = compose(
            config_name="dp_actor",
            overrides=[
                "strategy=fsdp",
                "clip_ratio=0.0001",
                "clip_ratio_high=5.0",
                "ppo_micro_batch_size_per_gpu=8",
            ],
        )
    actor_config: FSDPActorConfig = omega_conf_to_dataclass(cfg)

    for step in range(steps):
        pg_loss, pg_metrics = compute_policy_loss_flow_grpo(
            old_log_prob=rollout_log_probs[:, step],
            log_prob=current_log_probs[:, step],
            advantages=advantages[:, step],
            response_mask=response_mask[:, step],
            loss_agg_mode="token-mean",
            config=actor_config,
        )

        assert pg_loss.shape == ()
        assert isinstance(pg_loss.item(), float)
        assert "actor/ppo_kl" in pg_metrics.keys()


if __name__ == "__main__":
    unittest.main()
