"""Validate replicated Gemma buffers before the existing tensor weight transfer."""

import hashlib

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

BUFFER_NAMES = {"layer_scalar", "std_bias", "std_scale", "input_min", "input_max", "output_min", "output_max"}


def verify_replicated_buffers(state):
    digest = hashlib.sha256()
    count = 0
    for name, tensor in sorted(state.items()):
        if name.rsplit(".", 1)[-1] not in BUFFER_NAMES:
            continue
        if isinstance(tensor, DTensor):
            if not all(placement.is_replicate() for placement in tensor.placements):
                raise ValueError(f"Gemma buffer unexpectedly sharded: {name}")
            tensor = tensor.to_local()
        digest.update(f"{name}:{tensor.dtype}:{tuple(tensor.shape)}".encode())
        digest.update(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes())
        count += 1
    if not count:
        raise ValueError("Gemma weight transfer has no replicated buffers")
    fingerprint = digest.hexdigest()
    if dist.is_initialized() and dist.get_world_size() > 1:
        fingerprints = [None] * dist.get_world_size()
        dist.all_gather_object(fingerprints, fingerprint)
        if len(set(fingerprints)) != 1:
            raise ValueError("Gemma replicated buffers differ across weight-transfer ranks")
    # The existing transfer sends each state entry once; no concatenation is needed.
    return {"buffers": count, "sha256": fingerprint}
