"""Instance-local large-head SDPA dispatch, based on Transformers PR #46960.

The pinned Transformers overlay enables GQA broadcast with head_dim=512 when
padding is absent. PyTorch 2.11 then falls back to quadratic math attention.
Registering this interface is explicit; importing this module changes nothing.
"""

import copy
from types import SimpleNamespace


def large_head_sdpa(module, query, key, value, attention_mask, **kwargs):
    from transformers.integrations.sdpa_attention import repeat_kv, sdpa_attention_forward

    if key.shape[-1] == value.shape[-1] and key.shape[-1] <= 256:
        return sdpa_attention_forward(module, query, key, value, attention_mask, **kwargs)
    # Preserve the upstream MHA path explicitly. The proxy prevents the frozen
    # helper from re-enabling GQA or repeating these already expanded tensors.
    groups = query.shape[1] // key.shape[1]
    if query.shape[1] % key.shape[1] or key.shape[1] != value.shape[1]:
        raise ValueError("Gemma SDPA requires matching KV heads dividing query heads")
    key, value = repeat_kv(key, groups), repeat_kv(value, groups)
    proxy = SimpleNamespace(is_causal=getattr(module, "is_causal", True))
    return sdpa_attention_forward(proxy, query, key, value, attention_mask, **kwargs)


def enable_large_head_sdpa(model):
    """Change text-attention dispatch only; native mask builders remain SDPA."""
    from transformers.modeling_utils import ALL_ATTENTION_FUNCTIONS
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextAttention

    name = "vero2_gemma4_large_head_sdpa"
    modules = [m for m in model.modules() if isinstance(m, Gemma4TextAttention)]
    if not modules or any(m.config._attn_implementation not in {"sdpa", name} for m in modules):
        raise ValueError("Gemma large-head dispatch requires Gemma4 text layers using SDPA")
    ALL_ATTENTION_FUNCTIONS.register(name, large_head_sdpa)
    for module in modules:
        # TextModel/Gemma4Model still need the original 'sdpa' mask builder.
        module.config = copy.copy(module.config)
        module.config._attn_implementation = name
    return len(modules)
