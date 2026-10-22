"""Explicit Gemma-only preservation of vLLM inference constants across sleep."""

import torch


def preserve_constant_norm_weights(model):
    """Make unlearned RMSNorm weights visible to vLLM's sleep buffer snapshot.

    vLLM 0.23 keeps has_weight=False RMSNorm weights as plain tensor attributes.
    They still feed its CUDA norm kernel, but level-2 sleep discards their CUDA
    storage. They have no HF checkpoint entries to restore them. Registering the
    existing tensor as a nonpersistent buffer preserves object identity and lets
    GPUWorker.sleep/wake_up save and restore it, without adding checkpoint keys.
    Call once after construction, before the first sleep; never on import.
    """
    if getattr(getattr(model, "config", None), "model_type", None) != "gemma4":
        return []
    names = []
    for name, module in model.named_modules():
        if type(module).__name__ != "RMSNorm" or module.has_weight:
            continue
        weight = module.weight
        if not torch.all(weight == 1):
            raise ValueError(f"Gemma constant norm weight already corrupted before preservation: {name}")
        if "weight" not in module._buffers:
            del module.weight
            module.register_buffer("weight", weight, persistent=False)
        names.append(name)
    return names
