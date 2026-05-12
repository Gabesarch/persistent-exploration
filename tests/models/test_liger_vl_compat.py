# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

"""Test that Liger kernel integration doesn't break VL models.

Regression test for https://github.com/verl-project/verl/issues/2609
and https://github.com/verl-project/verl/issues/1720.

The bug: _apply_liger_kernel_to_instance with default kwargs sets
fused_linear_cross_entropy=True, which replaces the model forward.
After verl's apply_monkey_patch patches the base model forward,
Liger's forward crashes with 'BaseModelOutputWithPast has no attribute rope_deltas'.
"""

from types import SimpleNamespace

import pytest
import torch
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLVisionModel

from verl.models.transformers.monkey_patch import apply_monkey_patch
from verl.models.transformers.qwen3_vl import fast_pos_embed_interpolate as verl_fast_pos_embed_interpolate


def create_tiny_qwen3_vl():
    """Create a minimal Qwen3-VL model (random weights, 1 layer) for testing."""
    config = Qwen3VLConfig(
        text_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=1000,
            rope_scaling=dict(
                type="mrope",
                mrope_section=[4, 4, 4],
            ),
        ),
        vision_config=dict(
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=1,
            num_attention_heads=4,
            patch_size=16,
            temporal_patch_size=2,
            in_channels=3,
            spatial_merge_size=2,
        ),
    )
    model = Qwen3VLForConditionalGeneration(config)
    return model.to(dtype=torch.bfloat16, device="cuda")


class _FakePosEmbed:
    def __init__(self):
        self.weight = torch.empty(1, dtype=torch.float32, device="cpu")

    def __call__(self, idx_tensor):
        assert idx_tensor.device.type == "cuda"
        return torch.ones(*idx_tensor.shape, 8, dtype=self.weight.dtype, device=idx_tensor.device)


class _FakeVisionModel:
    def __init__(self):
        self.config = SimpleNamespace(spatial_merge_size=2)
        self.num_grid_per_side = 4
        self.pos_embed = _FakePosEmbed()


class _FakeQwen3VLModel:
    config = SimpleNamespace(
        model_type="qwen3_vl",
        text_config=SimpleNamespace(num_attention_heads=4, num_key_value_heads=2),
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_qwen3_vl_fast_pos_embed_interpolate_uses_grid_device():
    """Regression test for FSDP2 cpu_offload moving pos_embed weight to CPU."""
    grid_thw = torch.tensor([[1, 4, 4]], dtype=torch.long, device="cuda")

    output = verl_fast_pos_embed_interpolate(_FakeVisionModel(), grid_thw)

    assert output.device.type == "cuda"
    assert output.shape == (16, 8)


def test_apply_monkey_patch_installs_qwen3_vl_pos_embed_patch():
    apply_monkey_patch(_FakeQwen3VLModel(), use_remove_padding=False, use_fused_kernels=False)

    assert Qwen3VLVisionModel.fast_pos_embed_interpolate is verl_fast_pos_embed_interpolate


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
def test_liger_vl_forward_with_monkey_patch():
    """Liger with fused_linear_cross_entropy=False + apply_monkey_patch works on VL models."""
    pytest.importorskip("liger_kernel")
    from liger_kernel.transformers.monkey_patch import _apply_liger_kernel_to_instance

    model = create_tiny_qwen3_vl()

    _apply_liger_kernel_to_instance(
        model=model,
        fused_linear_cross_entropy=False,
        swiglu=True,
    )
    apply_monkey_patch(model, use_remove_padding=False, use_fused_kernels=False)

    input_ids = torch.randint(0, 500, (1, 32), device="cuda")
    attention_mask = torch.ones_like(input_ids)
    output = model(input_ids=input_ids, attention_mask=attention_mask)

    assert output.logits is not None
    assert output.logits.shape[-1] == 1000  # vocab_size

    del model
    torch.cuda.empty_cache()
