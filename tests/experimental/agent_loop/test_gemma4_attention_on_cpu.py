"""Large-head dispatch must preserve outputs/gradients and instance isolation."""

from types import SimpleNamespace

import pytest
import torch

from verl.models.gemma4.attention import enable_large_head_sdpa, large_head_sdpa


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("masked", [False, True])
def test_large_head_sdpa_math_oracle(device, masked):
    from torch.nn.attention import SDPBackend, sdpa_kernel

    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("Requires GPU smoke")
    torch.manual_seed(91)
    dtype = torch.float32
    q = torch.randn(1, 8, 33, 512, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, 2, 33, 512, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn_like(k, requires_grad=True)
    qr, kr, vr = [x.detach().clone().requires_grad_() for x in (q, k, v)]
    mask = torch.ones(33, 33, device=device, dtype=torch.bool).tril()[None, None] if masked else None
    module = SimpleNamespace(num_key_value_groups=4, is_causal=True)
    # Excluding MATH verifies that the fix actually avoids the silent fallback.
    backend = SDPBackend.EFFICIENT_ATTENTION if device == "cuda" else SDPBackend.MATH
    with sdpa_kernel(backend):
        actual, _ = large_head_sdpa(module, q, k, v, mask, scaling=0.1)
        probe = torch.randn_like(actual)
        (actual * probe).sum().backward()
    with sdpa_kernel(SDPBackend.MATH):
        expected = torch.nn.functional.scaled_dot_product_attention(
            qr, kr, vr, attn_mask=mask, is_causal=not masked, enable_gqa=True, scale=0.1
        ).transpose(1, 2)
        (expected * probe).sum().backward()
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    for actual_grad, expected_grad in zip((q.grad, k.grad, v.grad), (qr.grad, kr.grad, vr.grad), strict=True):
        torch.testing.assert_close(actual_grad, expected_grad, atol=3e-5, rtol=3e-5)


def test_dispatch_is_instance_local():
    from test_gemma4_chunked_policy_on_cpu import tiny_model

    original, patched = tiny_model(), tiny_model()
    enable_large_head_sdpa(patched)
    assert original.config.text_config._attn_implementation == "sdpa"
    assert patched.config.text_config._attn_implementation == "sdpa"
    assert all(layer.self_attn.config._attn_implementation == "sdpa" for layer in original.model.language_model.layers)
    assert all(
        layer.self_attn.config._attn_implementation == "vero2_gemma4_large_head_sdpa"
        for layer in patched.model.language_model.layers
    )
