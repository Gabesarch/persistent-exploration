"""Opt-in FSDP support for Gemma shared KV states and replicated buffers."""

import torch
import torch.distributed as dist


def group_text_wrap_targets(model, modules):
    """Group text decoders so FSDP's input pytree hook cannot copy their KV dict.

    Torch 2.11 reconstructs kwargs when registering backward hooks, even with
    cast_forward_inputs=False. One FSDP group runs that hook only on its first
    decoder; subsequent KV producers and consumers retain the original dict.
    This uses the public fully_shard(list_of_modules) API, retaining gradient
    hooks and native gradient checkpointing. Vision layers stay separate.
    """
    text_models = [module for module in model.modules() if type(module).__name__ == "Gemma4TextModel"]
    if len(text_models) != 1:
        raise ValueError("Gemma shared-KV wrapping requires exactly one Gemma4TextModel")
    layers = list(text_models[0].layers)
    if not layers or any(layer not in modules for layer in layers):
        raise ValueError("Gemma shared-KV wrapping requires all text decoder layers in wrap_policy")
    attention = layers[0].self_attn
    if attention.store_full_length_kv or attention.is_kv_shared_layer:
        raise ValueError("The first grouped Gemma decoder must not produce or consume shared KV states")
    layer_ids = {id(layer) for layer in layers}
    targets = []
    for module in modules:
        if module is layers[0]:
            targets.append(layers)
        elif id(module) not in layer_ids:
            targets.append(module)
    return targets


@torch.no_grad()
def broadcast_buffers_by_name(model):
    # Gemma4TextRotaryEmbedding registers buffers by iterating a Python set.
    # Different process hash seeds therefore produce different registration order.
    buffers = sorted(model.named_buffers(), key=lambda item: item[0])
    manifest = [(name, tuple(buf.shape), str(buf.dtype)) for name, buf in buffers]
    manifests = [None] * dist.get_world_size()
    dist.all_gather_object(manifests, manifest)
    if any(other != manifest for other in manifests):
        raise ValueError("Gemma FSDP buffer names, shapes, or dtypes differ across ranks")
    for _, buf in buffers:
        dist.broadcast(buf, src=0)
    # Complete collectives before the caller moves CPU-offloaded buffer storage.
    for device in {buf.device for _, buf in buffers if buf.is_cuda}:
        torch.cuda.synchronize(device)
