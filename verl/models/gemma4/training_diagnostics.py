"""Opt-in smoke measurements of active Gemma text/vision gradients and updates."""

import torch
from torch.distributed.tensor import DTensor


def local(tensor):
    return tensor.to_local() if isinstance(tensor, DTensor) else tensor


def before_optimizer_step(model):
    selected = {}
    for name, param in model.named_parameters():
        group = "text" if "language_model" in name else "vision" if "vision_tower" in name else None
        if group is None or group in selected or not name.endswith(("q_proj.weight", "q_proj.linear.weight")):
            continue
        if param.grad is None:
            raise RuntimeError(f"Missing active Gemma gradient: {name}")
        grad = local(param.grad).detach()
        if not torch.isfinite(grad).all():
            raise RuntimeError(f"Nonfinite Gemma gradient: {name}")
        selected[group] = (
            param,
            local(param).detach().float().clone(),
            float(grad.float().norm()),
            grad.dtype,
            param.dtype,
        )
    if set(selected) != {"text", "vision"}:
        raise RuntimeError(f"Missing Gemma text/vision gradient probes: {set(selected)}")
    return selected


def after_optimizer_step(selected):
    metrics = {}
    for group, (param, before, norm, grad_dtype, param_dtype) in selected.items():
        prefix = f"actor/gemma4_{group}"
        metrics[prefix + "_grad_norm"] = norm
        metrics[prefix + "_update_max_abs"] = float((local(param).detach().float() - before).abs().max())
        metrics[prefix + "_param_bf16"] = float(param_dtype == torch.bfloat16)
        metrics[prefix + "_grad_fp32"] = float(grad_dtype == torch.float32)
        metrics[prefix + "_grad_bf16"] = float(grad_dtype == torch.bfloat16)
    return metrics
