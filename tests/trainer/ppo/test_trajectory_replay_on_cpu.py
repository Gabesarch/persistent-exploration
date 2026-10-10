"""Exercise mixed replay preparation, branch weighting, advantages and gradients."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from verl import DataProto
from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
from verl.trainer.ppo.ray_trainer import RayPPOTrainer, compute_advantage
from verl.trainer.ppo.trajectory_replay import (
    assign_prompt_group_schedules,
    attach_mixed_loss_weights,
    choose_task_rows,
    compute_mixed_advantages,
    object_array,
    replay_prompt_target,
)
from verl.utils.trajectory_replay import ReplayCache, persist_candidate
from verl.workers.config.actor import ActorConfig, BranchLossStrataConfig, PolicyLossConfig


def trainer_config(enabled=True):
    return OmegaConf.create(
        {
            "trainer": {"trajectory_replay": {"enabled": enabled, "prompt_fraction": 0.2, "seed": 42}},
            "actor_rollout_ref": {
                "actor": {"policy_loss": {"branch_strata": {"enabled": True}, "retry_critic": {"enabled": False}}},
                "rollout": {
                    "agent": {"default_agent_loop": "tool_agent"},
                    "multi_turn": {"prompt_group_schedule_enabled": enabled},
                },
            },
        }
    )


def prompt_batch(size=128):
    return DataProto.from_dict(
        tensors={"dummy_tensor": torch.zeros(size, 1)},
        non_tensors={
            "ability": object_array(["chart"] * 43 + ["counting"] * 43 + ["search"] * 42),
            "raw_prompt": object_array([[{"role": "user", "content": f"question {i}"}] for i in range(size)]),
            "reward_model": object_array([{"ground_truth": str(i)} for i in range(size)]),
            "data_source": object_array(["task"] * size),
            "extra_info": object_array([{} for _ in range(size)]),
            "index": np.arange(size),
        },
    )


def fill_cache(cache, count=13):
    candidates = []
    for label in ("A", "R"):
        for i in range(count):
            candidates.append(
                persist_candidate(
                    cache.root,
                    question=f"What is {label}{i}?",
                    answer=str(i),
                    initial_images=[Image.new("RGB", (32, 32), (i, 0, 0))],
                    tool_images=[],
                    candidate_turn=1,
                    tool_call_count=0,
                    accuracy=float(label == "A"),
                    source_step=1,
                    source_ids={"request_id": f"{label}{i}"},
                )
            )
    cache.commit(candidates, 1)


def make_trainer(tmp_path, cache_count=13):
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = trainer_config()
    trainer.global_steps = 2
    trainer.train_dataset = SimpleNamespace(per_batch_domain_weights={"chart": 1, "counting": 1, "search": 1})
    trainer.trajectory_replay_cache = ReplayCache(tmp_path / "trajectory_replay")
    fill_cache(trainer.trajectory_replay_cache, cache_count)
    trainer._replay_prompt_fits = lambda entry: True
    trainer._replay_restore_step = -1
    return trainer


def test_preparation_keeps_fixed_batch_balanced_domains_and_shared_groups(tmp_path):
    trainer = make_trainer(tmp_path)
    metrics = {}
    batch = trainer._prepare_trajectory_replay_batch(prompt_batch(), metrics)
    assert len(batch) == 128 and metrics["replay/prompt_count"] == 26
    assert metrics["replay/prompt_fraction"] == pytest.approx(26 / 128)
    assert metrics["replay/loss_share"] == pytest.approx(0.2)
    for domain in ("chart", "counting", "search"):
        assert list(batch.non_tensor_batch["ability"]).count(domain) == 34
    assert len(set(entry["entry_id"] for entry in batch.non_tensor_batch["replay_candidate"][102:])) == 26
    for messages, entry in zip(
        batch.non_tensor_batch["raw_prompt"][102:], batch.non_tensor_batch["replay_candidate"][102:], strict=True
    ):
        assert entry["answer"] in messages[1]["content"][-1]["text"]
        assert entry["label"] not in messages[0]["content"].split("A means")[0]
    repeated = batch.repeat(8, interleave=True)
    for i in range(102):
        group = repeated.non_tensor_batch["prompt_group_schedule"][8 * i : 8 * i + 8]
        assert all(item == group[0] for item in group)
    assert all(item is None for item in repeated.non_tensor_batch["prompt_group_schedule"][816:])
    # Generation-only fields pop out, but judge identity and labels stay aligned.
    batch.non_tensor_batch["uid"] = object_array([str(i) for i in range(len(batch))])
    generated = trainer._get_gen_batch(batch)
    assert "is_replay_judge" in batch.non_tensor_batch and "is_replay_judge" in generated.non_tensor_batch
    assert sum(batch.non_tensor_batch["is_replay_judge"]) == 26


def test_short_cache_uses_balanced_pairs_and_scales_judge_mass(tmp_path):
    trainer = make_trainer(tmp_path, cache_count=2)
    metrics = {}
    batch = trainer._prepare_trajectory_replay_batch(prompt_batch(), metrics)
    assert metrics["replay/prompt_count"] == 4 and len(batch) == 128
    assert metrics["replay/accept_prompts"] == metrics["replay/reject_prompts"] == 2
    assert batch.meta_info["replay_loss_share"] == pytest.approx(0.2 * 4 / 26)


def test_disabled_features_leave_batch_untouched():
    trainer = RayPPOTrainer.__new__(RayPPOTrainer)
    trainer.config = trainer_config(False)
    batch = prompt_batch()
    keys = set(batch.non_tensor_batch)
    assert trainer._prepare_trajectory_replay_batch(batch, {}) is batch
    assert set(batch.non_tensor_batch) == keys


def test_prompt_allocation_and_deadlines_are_deterministic_and_separated():
    schedules = assign_prompt_group_schedules(102, {}, 7)
    assert list(schedules) == list(assign_prompt_group_schedules(102, {}, 7))
    assert [sum(s["category"] == c for s in schedules) for c in ("normal", "single", "double")] == [87, 10, 5]
    assert list(schedules) != list(assign_prompt_group_schedules(102, {}, 8))
    for schedule in schedules:
        assert schedule["force_first_tool_call"] == bool(schedule["answer_turns"])
        assert all(2 <= turn <= 9 for turn in schedule["answer_turns"])
        if len(schedule["answer_turns"]) == 2:
            assert schedule["answer_turns"][1] - schedule["answer_turns"][0] >= 2
    assert replay_prompt_target(128, 0.2) == 26
    assert replay_prompt_target(20, 0.2) == 4
    assert (
        len(choose_task_rows(prompt_batch(), 102, {"chart": 1, "counting": 1, "search": 1}, np.random.default_rng(2)))
        == 102
    )


def mixed_batch():
    # Two task rollouts, one rejected branch, four judges, one padding copy.
    mask = torch.ones(8, 4)
    mask[-1] = 0
    attention = torch.cat([torch.ones(8, 2), mask], dim=-1).long()
    return DataProto.from_dict(
        tensors={
            "response_mask": mask,
            "prompts": torch.zeros(8, 2, dtype=torch.long),
            "attention_mask": attention,
            "token_level_rewards": torch.zeros(8, 4),
        },
        non_tensors={
            "uid": object_array(["task", "task", "task", "judgeA", "judgeA", "judgeR", "judgeR", "judgeR"]),
            "request_id": object_array(["p0", "p1", "p0", "j0", "j1", "j2", "j3", "j3"]),
            "is_conditional_answer_retry_branch": np.array([False, False, True, False, False, False, False, False]),
            "is_replay_judge": np.array([False, False, False, True, True, True, True, True]),
            "__branch_padding_row": np.array([False] * 7 + [True]),
            "gdpo_baseline_weight": np.array([1, 1, 0, 1, 1, 1, 1, 0], dtype=float),
            "accuracy": np.array([1, 0, 0, 1, 0, 1, 1, 1], dtype=float),
            "format": np.array([1, 1, 1, 1, 1, -4, 1, 1], dtype=float),
            "tool_efficiency": object_array([0.8, 0.2, 0.5, None, None, None, None, None]),
            "tool_repeat_overlap": object_array([0, 1, 0.5, None, None, None, None, None]),
        },
    )


def algorithm():
    return OmegaConf.create(
        {
            "adv_estimator": "gdpo",
            "gamma": 1.0,
            "lam": 1.0,
            "gdpo_reward_keys": ["accuracy", "format", "tool_efficiency", "tool_repeat_overlap"],
            "gdpo_reward_weights": [0.8, 0.2, 0.15, 0.2],
            "gdpo_baseline_weight_key": "gdpo_baseline_weight",
            "gdpo_stability_enabled": True,
            "gdpo_component_std_floor": 0.05,
            "gdpo_component_adv_clip": 5.0,
            "gdpo_final_std_floor": 0.05,
            "gdpo_final_adv_clip": 10.0,
        }
    )


def test_advantages_separate_judge_components_and_penalty_reaches_policy_gradients():
    batch = mixed_batch()
    compute_mixed_advantages(batch, algorithm(), compute_advantage, 2)
    assert torch.isfinite(batch.batch["advantages"]).all()
    assert batch.batch["advantages"][-1].eq(0).all()
    # Equal decision accuracy, but the overlong judge receives lower advantage.
    assert batch.batch["advantages"][5].mean() < batch.batch["advantages"][6].mean()
    attach_mixed_loss_weights(batch, {}, 0.2, {})
    assert batch.batch["branch_policy_loss_weight"][3:7].sum().item() / len(batch) == pytest.approx(0.2)
    assert batch.batch["branch_policy_loss_weight"][-1] == 0
    actor = ActorConfig(
        strategy="fsdp",
        rollout_n=2,
        use_dynamic_bsz=True,
        loss_agg_mode="seq-mean-token-mean",
        policy_loss=PolicyLossConfig(loss_mode="gspo", branch_strata=BranchLossStrataConfig(enabled=True)),
    )
    log_probs = torch.zeros(8, 4, requires_grad=True)
    loss, _ = compute_policy_loss_gspo(
        old_log_prob=log_probs.detach(),
        log_prob=log_probs,
        advantages=batch.batch["advantages"],
        response_mask=batch.batch["response_mask"],
        config=actor,
        branch_policy_loss_weight=batch.batch["branch_policy_loss_weight"],
        branch_policy_loss_is_branch=batch.batch["branch_policy_loss_is_branch"],
    )
    loss.backward()
    assert log_probs.grad[3:7, :2].abs().sum() > 0  # generated reasoning tokens receive policy gradient
    assert log_probs.grad[-1].eq(0).all()


@pytest.mark.parametrize("branch_count", [0, 1, 3])
def test_judge_mass_is_independent_of_task_branch_expansion(branch_count):
    canonical = 4
    count = canonical + branch_count + 2 + 1
    flags = [False] * canonical + [True] * branch_count + [False] * 3
    mask = torch.ones(count, 3)
    mask[-1] = 0
    batch = DataProto.from_dict(
        tensors={"response_mask": mask},
        non_tensors={
            "request_id": object_array(
                [f"p{i}" for i in range(canonical)] + ["p0"] * branch_count + ["j0", "j1", "j1"]
            ),
            "is_conditional_answer_retry_branch": np.array(flags),
            "is_replay_judge": np.array([False] * (canonical + branch_count) + [True] * 3),
            "__branch_padding_row": np.array([False] * (count - 1) + [True]),
        },
    )
    metrics = {}
    attach_mixed_loss_weights(batch, {}, 0.2, metrics)
    assert metrics["replay/loss_share"] == pytest.approx(0.2)
    assert metrics["replay/task_loss_share"] == pytest.approx(0.8)
    weights = batch.batch["branch_policy_loss_weight"]
    if branch_count:
        assert weights[canonical : canonical + branch_count].sum().item() / count == pytest.approx(0.8 * 0.2)


def test_fresh_actor_discards_live_cache_without_reusing_future_state(tmp_path):
    trainer = make_trainer(tmp_path)
    trainer.trajectory_replay_cache.commit([], 4)
    trainer._reset_trajectory_replay_cache()
    assert trainer.trajectory_replay_cache.step == 0 and trainer.trajectory_replay_cache.entries == []
    trainer.trajectory_replay_cache.reload()
    assert trainer.trajectory_replay_cache.entries == []
