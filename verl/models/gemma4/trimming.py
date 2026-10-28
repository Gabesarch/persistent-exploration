"""Single-trajectory padding removal; leaves externally visible layouts intact."""

import torch


def trim_inputs(inputs, labels, *, alignment=1):
    """Return compact inputs, aligned response labels, and original response width.

    Only left-padded prompts followed by right-padded responses are accepted.
    Position IDs are selected, never renumbered. Image patch positions are not
    sequence positions and must not be sliced.

    Remove whole alignment blocks at both ends. Keeping small boundary margins
    preserves the attention kernel's tiling relative to the valid tokens. This
    matters for the pinned BF16 SDPA kernels: shifting tile boundaries changes
    rounding, which the pretrained E4B layers can amplify substantially.
    """
    if not isinstance(alignment, int) or alignment < 1:
        raise ValueError("Gemma trim alignment must be a positive integer")
    ids, mask = inputs["input_ids"], inputs["attention_mask"]
    if ids.ndim != 2 or ids.shape[0] != 1 or mask.shape != ids.shape:
        raise ValueError("Gemma trimming requires one trajectory and a matching 2D attention mask")
    if labels.ndim != 2 or labels.shape[0] != 1 or not 0 < labels.shape[1] < ids.shape[1]:
        raise ValueError("Gemma trimming requires a nonempty prompt and response allocation")
    if not torch.all((mask == 0) | (mask == 1)):
        raise ValueError("Gemma trimming requires a binary attention mask")
    width = labels.shape[1]
    boundary = ids.shape[1] - width
    if not torch.equal(labels, ids[:, boundary:]):
        raise ValueError("Gemma response labels must match the input suffix")
    prompt_count = int(mask[:, :boundary].sum().item())
    response_count = int(mask[:, boundary:].sum().item())
    start, end = boundary - prompt_count, boundary + response_count
    expected = torch.zeros_like(mask)
    expected[:, start:end] = 1
    if prompt_count == 0 or not torch.equal(mask, expected):
        raise ValueError("Gemma trimming accepts only left prompt/right response padding, without internal gaps")
    start = start // alignment * alignment
    end = min(ids.shape[1], (end + alignment - 1) // alignment * alignment)
    result = dict(inputs)
    if result.get("position_ids") is None:
        result["position_ids"] = torch.arange(ids.shape[1], device=ids.device).unsqueeze(0)
    for key in ("input_ids", "attention_mask", "position_ids", "mm_token_type_ids"):
        value = result.get(key)
        if value is not None:
            if value.shape != ids.shape:
                raise ValueError(f"Gemma trimming requires {key} aligned to input_ids")
            result[key] = value[:, start:end]
    return result, labels[:, : end - boundary], width


def validate_loss_masks(batch):
    """Dropping padding is valid only if no objective supervises those positions."""
    response_attention = batch["attention_mask"][:, -batch["responses"].shape[1] :].bool()
    for key in ("response_mask", "retry_sft_mask", "retry_critic_mask"):
        if key in batch and torch.any(batch[key].bool() & ~response_attention):
            raise ValueError(f"Gemma trimming cannot discard supervised padding in {key}")
