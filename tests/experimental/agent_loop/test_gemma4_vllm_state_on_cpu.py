"""Exercise real vLLM constant norm tensors and its buffer save/restore contract."""

from types import SimpleNamespace

import pytest
import torch

from verl.models.gemma4.vllm_state import preserve_constant_norm_weights


def make_model():
    from vllm.config import DeviceConfig, VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import RMSNorm

    with set_current_vllm_config(VllmConfig(device_config=DeviceConfig(device="cpu"))):
        model = torch.nn.Module()
        model.config = SimpleNamespace(model_type="gemma4")
        model.constant = RMSNorm(8, has_weight=False)
        model.learned = RMSNorm(8, has_weight=True)
    return model


def test_constant_weights_survive_buffer_restore_without_checkpoint_changes():
    model = make_model()
    original = model.constant.weight
    keys = set(model.state_dict())
    assert "constant.weight" not in dict(model.named_buffers())
    assert preserve_constant_norm_weights(model) == ["constant"]
    assert preserve_constant_norm_weights(model) == ["constant"]
    assert model.constant.weight is original
    assert set(model.state_dict()) == keys
    assert "learned.weight" in dict(model.named_parameters())
    saved = {name: buffer.cpu().clone() for name, buffer in model.named_buffers()}
    original.zero_()  # Model the loss of GPU storage during level-2 sleep.
    for name, buffer in model.named_buffers():
        buffer.copy_(saved[name])
    assert torch.equal(original, torch.ones(8))


def test_other_models_are_unchanged_and_existing_corruption_is_rejected():
    from verl.workers.config.rollout import RolloutConfig

    assert RolloutConfig().gemma4_preserve_constant_norm_weights is False
    model = make_model()
    model.config.model_type = "qwen2_5_vl"
    assert preserve_constant_norm_weights(model) == []
    assert "constant.weight" not in dict(model.named_buffers())
    model.config.model_type = "gemma4"
    model.constant.weight.zero_()
    with pytest.raises(ValueError, match="already corrupted"):
        preserve_constant_norm_weights(model)
