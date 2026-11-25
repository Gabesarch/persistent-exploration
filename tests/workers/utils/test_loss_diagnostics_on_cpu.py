"""Independent loss/gradient partition and actor integration regressions."""

from dataclasses import replace

import pytest
import torch
from test_auxiliary_mass_bounds_on_cpu import _actor, _config, _data

from verl.trainer.ppo.core_algos import compute_policy_loss_gspo
from verl.workers.config import LossDiagnosticsConfig
from verl.workers.utils.gradient_diagnostics import GradientProbe, gradient_step_selected
from verl.workers.utils.loss_diagnostics import COMPONENTS, weighted_loss_components


def test_actual_clipped_terms_with_masks_weights_and_active_clipping():
    cfg = replace(_config(enabled=False), clip_ratio_low=0.2, clip_ratio_high=0.2)
    ratios = torch.tensor([1.5, 0.5, 0.9, 1.1, 1.0], dtype=torch.float64)
    logp = ratios.log().unsqueeze(-1).expand(-1, 2).clone().requires_grad_()
    advantages = torch.tensor([[2., 2.], [-3., -3.], [4., 4.], [-5., -5.], [0., 0.]])
    mask = torch.tensor([[1., 1.], [1., 0.], [1., 1.], [1., 0.], [0., 0.]])
    row_weights = torch.tensor([1., 2., 0.25, 0.5, 0.])
    is_weights = torch.tensor([[1., 2.], [0.5, 1.], [2., 2.], [1., 1.], [1., 1.]])
    parts = {}
    loss, _ = compute_policy_loss_gspo(
        torch.zeros_like(logp), logp, advantages, mask, config=cfg,
        rollout_is_weights=is_weights, branch_policy_loss_weight=row_weights,
        branch_policy_loss_is_branch=torch.tensor([False, False, True, True, True]), loss_components=parts,
    )
    # Explicit hand-calculated clipped terms. The first two rows are clipped
    # and have NONZERO scalar loss but zero policy gradient.
    expected = [-2 * 1.2 * 1.5 / 5, 3 * 0.8 * 0.5 * 2 / 5,
                -4 * 0.9 * 2 * 0.25 / 5, 5 * 1.1 * 0.5 / 5]
    assert [parts[k].item() for k in COMPONENTS[:4]] == pytest.approx(expected)
    torch.testing.assert_close(sum(parts.values()), loss)
    gradients = [torch.autograd.grad(parts[k], logp, retain_graph=True)[0] for k in COMPONENTS[:4]]
    assert gradients[0].count_nonzero() == gradients[1].count_nonzero() == 0
    torch.testing.assert_close(sum(gradients), torch.autograd.grad(loss, logp)[0])


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("has_rl,has_critic", [(True, True), (True, False), (False, True), (False, False)])
def test_mixture_coefficients_and_missing_strata(bounded, has_rl, has_critic):
    parts = {name: torch.tensor((i + 1.) if has_rl else 0.) for i, name in enumerate(COMPONENTS[:4])}
    critic = torch.tensor(7. if has_critic else 0.)
    weighted = weighted_loss_components(parts, critic, loss_scale=0.5, critic_fraction=0.1,
                                        has_rl=has_rl, has_critic=has_critic, bounded=bounded,
                                        critic_multiplier=0.2)
    if bounded:
        expected = 0.9 * 0.5 * sum(parts.values()) + 0.1 * 0.2 * critic
    elif has_rl and has_critic:
        expected = 0.9 * 0.5 * sum(parts.values()) + 0.1 * critic
    else:
        expected = 0.5 * sum(parts.values()) + critic
    torch.testing.assert_close(sum(weighted.values()), expected)


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("micro,dynamic", [(1, False), (2, False), (1, True)])
def test_actor_loss_logging_does_not_change_gradients(monkeypatch, bounded, micro, dynamic):
    cfg = _config(enabled=bounded, micro=micro, dynamic=dynamic, epochs=2)
    baseline = _actor(monkeypatch, cfg)
    baseline_metrics = baseline.update_policy(_data())
    cfg = replace(cfg, policy_loss=replace(cfg.policy_loss, loss_diagnostics=LossDiagnosticsConfig(enabled=True)))
    diagnostic = _actor(monkeypatch, cfg)
    metrics = diagnostic.update_policy(_data())
    for a, b in zip(baseline.recorded_gradients, diagnostic.recorded_gradients, strict=True):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for key in baseline_metrics:
        assert metrics[key] == baseline_metrics[key]
    prefix = "actor/loss_components/"
    assert sum(metrics[prefix + k + "_sum"] for k in COMPONENTS) == pytest.approx(
        metrics["actor/combined_policy_loss"], abs=1e-6,
    )
    assert metrics[prefix + "canonical_positive_sum"] < 0
    assert metrics[prefix + "canonical_negative_sum"] > 0
    assert metrics[prefix + "reconstruction_abs_error_max"] < 1e-6


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("missing", ["critic", "rl", "both"])
def test_actor_missing_strata_reconcile(monkeypatch, bounded, missing):
    cfg = _config(enabled=bounded)
    cfg = replace(cfg, policy_loss=replace(cfg.policy_loss, loss_diagnostics=LossDiagnosticsConfig(enabled=True)))
    data = _data()
    if missing in ("critic", "both"):
        data.batch["retry_critic_mask"].zero_()
    if missing in ("rl", "both"):
        data.batch["response_mask"].zero_()
    actor = _actor(monkeypatch, cfg)
    metrics = actor.update_policy(data)
    assert sum(metrics[f"actor/loss_components/{k}_sum"] for k in COMPONENTS) == pytest.approx(
        metrics["actor/combined_policy_loss"], abs=1e-6,
    )


def test_replay_recipe_without_critic_ce(monkeypatch):
    cfg = _config(enabled=False)
    cfg = replace(cfg, policy_loss=replace(
        cfg.policy_loss, retry_critic=replace(cfg.policy_loss.retry_critic, enabled=False),
        loss_diagnostics=LossDiagnosticsConfig(enabled=True),
    ))
    actor = _actor(monkeypatch, cfg)
    metrics = actor.update_policy(_data())
    assert metrics["actor/loss_components/critic_ce_sum"] == 0
    assert sum(metrics[f"actor/loss_components/{k}_sum"] for k in COMPONENTS) == pytest.approx(
        metrics["actor/combined_policy_loss"], abs=1e-6,
    )


def test_sampling_schedule_and_disable():
    cfg = LossDiagnosticsConfig(enabled=True, gradient_enabled=True, gradient_every_n_steps=7)
    assert gradient_step_selected(cfg, 19, True)
    assert not gradient_step_selected(cfg, 19, False)
    assert gradient_step_selected(cfg, 21, False)
    assert not gradient_step_selected(replace(cfg, gradient_first_update=False), 19, True)
    assert not gradient_step_selected(replace(cfg, gradient_enabled=False), None, True)
    assert not gradient_step_selected(replace(cfg, enabled=False), None, True)
    with pytest.raises(ValueError, match="global_steps"):
        gradient_step_selected(cfg, None, False)


def test_known_vectors_gram_cancellation_and_real_clipping():
    module = torch.nn.Linear(3, 1, bias=False, dtype=torch.float64)
    grads = torch.tensor([[1., 0., 0.], [-0.5, 0., 0.], [0., 2., 0.], [0., -1., 0.],
                          [-0.5, 0., 3.]], dtype=torch.float64)
    probe = GradientProbe(module, "cpu")
    for row in grads:
        module.weight.grad = row.reshape(1, -1).clone()
        probe.capture(0)
    module.weight.grad = grads.sum(0).reshape(1, -1)
    metrics = probe.measure_before_clip()
    expected = torch.cat([grads, grads.sum(0, keepdim=True)])
    torch.testing.assert_close(probe.gram, expected @ expected.T)
    assert metrics["canonical_norm_preclip"] == pytest.approx(0.5)
    assert metrics["branch_norm_preclip"] == pytest.approx(1.0)
    assert metrics["canonical_positive__canonical_negative_cosine"] == pytest.approx(-1)
    assert metrics["canonical__branch_cosine"] == pytest.approx(0)
    assert metrics["reconstruction_abs_error"] == pytest.approx(0)
    total = torch.nn.utils.clip_grad_norm_(module.parameters(), 0.25)
    probe.after_clip(total, 0.25)
    assert metrics["combined_norm_postclip_measured"] == pytest.approx(0.25, rel=1e-6)
    assert metrics["clip_post_norm_relative_error"] < 1e-12
    assert sum(metrics[f"{k}_projection_on_combined"] for k in COMPONENTS) == pytest.approx(1)


def test_zero_gradients_have_undefined_cosines_not_false_orthogonality():
    module = torch.nn.Linear(2, 1)
    probe = GradientProbe(module, "cpu")
    for _ in COMPONENTS:
        probe.capture(0)
    metrics = probe.measure_before_clip()
    assert metrics["canonical_positive__canonical_negative_cosine_defined"] == 0
    assert metrics["projection_on_combined_defined"] == 0
    assert metrics["combined_norm_preclip"] == 0


@pytest.mark.parametrize("bounded", [False, True])
@pytest.mark.parametrize("micro,dynamic", [(1, False), (2, False), (1, True)])
def test_gradient_replay_preserves_rng_and_real_actor_update(monkeypatch, tmp_path, bounded, micro, dynamic):
    import json
    from types import MethodType

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    cfg = _config(enabled=bounded, micro=micro, dynamic=dynamic, epochs=2)
    data = _data()
    step = 19 if bounded else 21
    data.meta_info["global_steps"] = step
    actors, outputs, rngs = [], [], []
    for enabled in (False, True):
        diagnostics = LossDiagnosticsConfig(enabled=True, gradient_enabled=enabled,
                                           gradient_output_dir=str(tmp_path), gradient_max_minibatches=0)
        config = replace(cfg, policy_loss=replace(cfg.policy_loss, loss_diagnostics=diagnostics))
        actor = _actor(monkeypatch, config, stochastic=True)
        actor._optimizer_step = MethodType(DataParallelPPOActor._optimizer_step, actor)
        # Make optimizer-state and parameter equivalence nontrivial.
        actor.actor_optimizer = torch.optim.AdamW(actor.actor_module.parameters(), lr=1e-3)
        torch.manual_seed(883)
        outputs.append(actor.update_policy(data))
        rngs.append(torch.get_rng_state())
        actors.append(actor)
    torch.testing.assert_close(rngs[0], rngs[1], rtol=0, atol=0)
    torch.testing.assert_close(actors[0].actor_module.weight, actors[1].actor_module.weight, rtol=0, atol=0)
    for key in outputs[0]:
        assert outputs[1][key] == outputs[0][key]
    for key, value in actors[0].actor_optimizer.state_dict()["state"][0].items():
        torch.testing.assert_close(value, actors[1].actor_optimizer.state_dict()["state"][0][key], rtol=0, atol=0)
    records = [json.loads(line) for line in (tmp_path / f"step_{step:06d}.jsonl").read_text().splitlines()]
    assert len(records) == 2
    assert outputs[1]["actor/gradient_components/measured_minibatches"] == 2
    for record in records:
        assert record["metrics"]["reconstruction_relative_to_component_norm_sum"] < 1e-6
        assert record["metrics"]["clip_post_norm_relative_error"] < 1e-6

@pytest.mark.parametrize("case", ["no_critic", "critic_disabled", "accept_only", "dataset_balanced"])
def test_gradient_replay_critic_variants(monkeypatch, case):
    from types import MethodType

    from verl.workers.actor.dp_actor import DataParallelPPOActor
    from verl.workers.config import RetryCriticDatasetBalanceConfig

    cfg = _config(enabled=case != "critic_disabled")
    data = _data()
    data.meta_info["global_steps"] = 19
    if case == "no_critic":
        data.batch["retry_critic_mask"].zero_()
    elif case == "critic_disabled":
        cfg = replace(cfg, policy_loss=replace(
            cfg.policy_loss, retry_critic=replace(cfg.policy_loss.retry_critic, enabled=False),
        ))
    elif case == "accept_only":
        data.batch["retry_critic_is_retry"].zero_()
    else:
        data.batch["retry_critic_stratum_id"] = torch.arange(4)
        data.meta_info["retry_critic_stratum_names"] = ("a_accept", "a_retry", "b_accept", "b_retry")
        cfg = replace(cfg, policy_loss=replace(cfg.policy_loss, retry_critic=replace(
            cfg.policy_loss.retry_critic, dataset_balance=RetryCriticDatasetBalanceConfig(enabled=True),
        )))
    cfg = replace(cfg, policy_loss=replace(cfg.policy_loss, loss_diagnostics=LossDiagnosticsConfig(
        enabled=True, gradient_enabled=True,
    )))
    actor = _actor(monkeypatch, cfg)
    actor._optimizer_step = MethodType(DataParallelPPOActor._optimizer_step, actor)
    result = actor.update_policy(data)
    prefix = "actor/gradient_components/"
    assert result[prefix + "reconstruction_relative_to_component_norm_sum"][0] < 1e-6
    assert result[prefix + "measured_minibatches"] == 1
    if case in ("no_critic", "critic_disabled"):
        assert result[prefix + "critic_ce_norm_preclip"] == [0.0]


def test_gradient_first_update_then_global_interval_and_first_minibatch_only(monkeypatch):
    from types import MethodType

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    cfg = _config(epochs=2)
    cfg = replace(cfg, policy_loss=replace(cfg.policy_loss, loss_diagnostics=LossDiagnosticsConfig(
        enabled=True, gradient_enabled=True, gradient_every_n_steps=10, gradient_max_minibatches=1,
    )))
    actor = _actor(monkeypatch, cfg)
    actor._optimizer_step = MethodType(DataParallelPPOActor._optimizer_step, actor)
    data = _data()
    for step, expected in [(19, 1), (19, 0), (20, 1), (21, 0)]:
        data.meta_info["global_steps"] = step
        result = actor.update_policy(data)
        assert result["actor/gradient_components/measured_minibatches"] == expected
        if expected:
            assert len(result["actor/gradient_components/combined_norm_preclip"]) == 1


@pytest.mark.parametrize("failure", [
    "reconstruction", "replay_loss", "loss_construction", "capture", "gram", "summary", "postclip",
    "clipping_audit", "record", "scalar_partition", "scalar_construction",
])
def test_diagnostic_failures_preserve_real_adam_update_and_rng(monkeypatch, tmp_path, failure):
    """Inject failures at each diagnostic boundary; the real update is identical."""
    from types import MethodType

    from verl.workers.actor import dp_actor

    cfg = _config(epochs=2)
    data = _data()
    data.meta_info["global_steps"] = 20
    actors, states, results = [], [], []
    for enabled in (False, True):
        output_dir = tmp_path / "records"
        if enabled and failure == "record":
            output_dir.write_text("This file deliberately prevents creating a diagnostic directory")
        diag = LossDiagnosticsConfig(enabled=True, gradient_enabled=enabled, gradient_max_minibatches=0,
                                     gradient_output_dir=str(output_dir))
        actor = _actor(monkeypatch, replace(cfg, policy_loss=replace(cfg.policy_loss, loss_diagnostics=diag)),
                       stochastic=True)
        actor._optimizer_step = MethodType(dp_actor.DataParallelPPOActor._optimizer_step, actor)
        actor.actor_optimizer = torch.optim.AdamW(actor.actor_module.parameters(), lr=1e-3)
        if enabled:
            def fail(*args, **kwargs):
                raise RuntimeError("Injected optional diagnostic failure")

            if failure in ("reconstruction", "replay_loss"):
                original = GradientProbe.capture

                def corrupt(probe, loss_total, scale=1.0, original=original):
                    original(probe, loss_total + (1 if failure == "replay_loss" else 0), scale)
                    if failure == "reconstruction":
                        for shard in probe.shards[-1]:
                            if shard is not None:
                                shard.mul_(1.25)

                monkeypatch.setattr(GradientProbe, "capture", corrupt)
            elif failure == "clipping_audit":
                original = GradientProbe._local_postclip_sq
                monkeypatch.setattr(
                    GradientProbe, "_local_postclip_sq", lambda probe, original=original: original(probe) * 4,
                )
            elif failure == "scalar_partition":
                original = dp_actor.reduce_loss_diagnostics

                def corrupt_partition(totals, original=original):
                    totals[0] += 1
                    return original(totals)

                monkeypatch.setattr(dp_actor, "reduce_loss_diagnostics", corrupt_partition)
            elif failure == "scalar_construction":
                monkeypatch.setattr(dp_actor, "weighted_loss_components", fail)
            elif failure == "loss_construction":
                actor._gradient_replay_loss = fail
            elif failure != "record":
                method = {"capture": "capture", "gram": "_local_gram", "summary": "_summarize_gram",
                          "postclip": "_local_postclip_sq"}[failure]
                monkeypatch.setattr(GradientProbe, method, fail)
        torch.manual_seed(41)
        results.append(actor.update_policy(data))
        states.append(torch.get_rng_state())
        actors.append(actor)
    torch.testing.assert_close(states[0], states[1], rtol=0, atol=0)
    torch.testing.assert_close(actors[0].actor_module.weight, actors[1].actor_module.weight, rtol=0, atol=0)
    for key, value in actors[0].actor_optimizer.state_dict()["state"][0].items():
        torch.testing.assert_close(value, actors[1].actor_optimizer.state_dict()["state"][0][key], rtol=0, atol=0)
    assert results[0]["actor/combined_policy_loss"] == results[1]["actor/combined_policy_loss"]
    prefix = "actor/gradient_components/"
    assert results[1][prefix + "invalid_minibatches"] >= 1
    assert all(v == 0 for v in results[1][prefix + "valid"])
    if failure in ("loss_construction", "capture", "gram", "summary", "postclip", "record", "scalar_construction"):
        assert results[1][prefix + "disabled_after_error"] == 1
        assert results[1][prefix + "measured_minibatches"] == 1


@pytest.mark.parametrize("dtype", [torch.float64, torch.bfloat16])
def test_real_low_precision_backward_nonadditivity_is_nonfatal(dtype):
    """Identical losses/forwards can have nonadditive finite-precision gradients."""
    from torch import nn

    torch.manual_seed(9)
    module = nn.Sequential(nn.Linear(16, 32), nn.GELU(), nn.Linear(32, 32), nn.GELU(), nn.Linear(32, 5)).to(dtype)
    inputs = torch.randn(24, 16).to(dtype)
    probe = GradientProbe(module, "cpu", reconstruction_rtol=1e-6)
    for k in range(6):
        module.zero_grad(set_to_none=True)
        output = module(inputs).to(torch.float64 if dtype == torch.float64 else torch.float32)
        losses = [output[i::5].square().sum() * (1 if i % 2 else -1) / 24 for i in range(5)]
        loss = sum(losses) if k == 5 else losses[k]
        loss.backward()
        if k < 5:
            probe.capture(loss.item())
    actual_before = [p.grad.clone() for p in module.parameters()]
    probe.measure_before_clip()
    for expected, parameter in zip(actual_before, module.parameters(), strict=True):
        torch.testing.assert_close(expected, parameter.grad, rtol=0, atol=0)
    error = probe.metrics["reconstruction_relative_to_component_norm_sum"]
    if dtype == torch.bfloat16:
        assert error > 1e-5
        assert probe.metrics["reconstruction_valid"] == probe.metrics["valid"] == 0
        assert probe.errors[0]["phase"] == "reconstruction"
    else:
        assert error < 1e-7
        assert probe.metrics["valid"] == 1
    assert not probe.shards


def test_nonfinite_scalar_diagnostic_is_flagged_without_raising():
    from verl.workers.utils.loss_diagnostics import reduce_loss_diagnostics

    values = reduce_loss_diagnostics(torch.tensor([float("nan"), 1., 2., 3., 4., 10.]))
    assert values["finite"] == values["reconstruction_valid"] == 0
    assert all(torch.isfinite(torch.tensor(value)) for value in values.values())


def test_bfloat16_accumulation_can_exceed_two_percent_with_exact_loss_partition():
    """Low-precision accumulation alone can exceed the old fatal 2% threshold."""
    torch.manual_seed(7)
    coefficients = torch.randn(1024, 5, 32) * 0.01
    for dtype in (torch.float64, torch.bfloat16):
        parameter = torch.nn.Parameter(torch.zeros(32, dtype=dtype))
        gradients = []
        for component in range(6):
            parameter.grad = None
            for microbatch in coefficients:
                losses = [(parameter.to(row.dtype) * row).sum() for row in microbatch]
                loss = sum(losses) if component == 5 else losses[component]
                loss.backward()
            gradients.append(parameter.grad.double().clone())
        residual = (sum(gradients[:5]) - gradients[5]).norm() / sum(g.norm() for g in gradients[:5])
        assert residual > 0.02 if dtype == torch.bfloat16 else residual < 1e-12
