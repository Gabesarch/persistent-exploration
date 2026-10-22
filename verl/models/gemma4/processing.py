"""Gemma image tensors aligned to the actual padded rollout IDs."""

from .protocol import MAX_SOFT_TOKENS


def is_gemma4(processor):
    return processor is not None and processor.__class__.__name__ == "Gemma4Processor"


def multimodal_inputs(processor, input_ids, images):
    ids = input_ids.reshape(1, -1)
    result = {}
    mask = (ids == processor.image_token_id).long()
    if images:
        # Probe the processor's actual expansion (aspect ratio dependent).
        probe = processor(
            text=["<|image|>" * len(images)],
            images=images,
            return_tensors="pt",
            add_special_tokens=False,
            max_soft_tokens=MAX_SOFT_TOKENS,
        )
        result = {
            key: value
            for key, value in probe.items()
            if key not in {"input_ids", "attention_mask", "mm_token_type_ids", "num_soft_tokens_per_image"}
        }
        expected = (probe["input_ids"] == processor.image_token_id).sum().item()
        if mask.sum().item() != expected:
            raise ValueError(f"Gemma image token mismatch: rollout={mask.sum().item()}, processor={expected}")
    elif mask.any():
        raise ValueError("Gemma image tokens have no corresponding images")
    result["mm_token_type_ids"] = mask
    return result


def collapse_image_tokens(ids, processor):
    """Undo only HF's image expansion for vLLM's own expansion, never model tokens."""
    boi = processor.config.boi_token_id
    eoi = processor.config.eoi_token_id
    image = processor.image_token_id
    result = []
    i = 0
    while i < len(ids):
        if ids[i] != boi:
            result.append(ids[i])
            i += 1
            continue
        j = i + 1
        while j < len(ids) and ids[j] == image:
            j += 1
        if j == i + 1 or j >= len(ids) or ids[j] != eoi:
            raise ValueError("Malformed Gemma expanded image span")
        result.append(image)
        i = j + 1
    return result
