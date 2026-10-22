"""Instance-local backport of Transformers PR #46176 (Gemma4 vision lookup).

https://github.com/huggingface/transformers/pull/46176
Importing this module does not patch Transformers or any model instance.
"""

from types import MethodType

import torch
import torch.nn.functional as F


def position_embeddings_lookup(self, pixel_position_ids, padding_positions):
    # Preserve negative padding handling and out-of-range errors from upstream.
    clamped_positions = pixel_position_ids.clamp(min=0)
    x_emb = F.embedding(clamped_positions[..., 0], self.position_embedding_table[0])
    y_emb = F.embedding(clamped_positions[..., 1], self.position_embedding_table[1])
    if x_emb.is_cuda and torch.is_autocast_enabled("cuda"):
        # The original matmul runs in the autocast dtype, then CUDA sum
        # promotes to FP32. Cast BEFORE adding to retain that rounding order.
        dtype = torch.get_autocast_dtype("cuda")
        x_emb = x_emb.to(dtype).float()
        y_emb = y_emb.to(dtype).float()
    position_embeddings = x_emb + y_emb
    return torch.where(padding_positions.unsqueeze(-1), 0.0, position_embeddings)


def enable_position_embedding_lookup(model):
    """Patch existing Gemma embedders before FSDP; retain parameters/state keys."""
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder

    embedders = [module for module in model.modules() if isinstance(module, Gemma4VisionPatchEmbedder)]
    if not embedders:
        raise ValueError("Gemma4 position lookup requested but no Gemma4VisionPatchEmbedder found")
    for module in embedders:
        module._position_embeddings = MethodType(position_embeddings_lookup, module)
    return len(embedders)
