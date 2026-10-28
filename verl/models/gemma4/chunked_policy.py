"""Opt-in response-only Gemma LM-head statistics with bounded vocabulary memory.

Projection, softcapping, temperature, and normalization are recomputed per token
chunk in backward. No full sequence-by-vocabulary tensor is retained. Model
parameters, token sequences, and loss masks are unchanged.
"""

from types import MethodType

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .profiling import stage


def _chunk_statistics(hidden, weight, labels, temperature, softcap, calculate_entropy):
    with stage("head/projection"):
        logits = F.linear(hidden, weight)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    if temperature != 1.0:
        logits = logits / temperature
    # FP32 normalization is important with BF16 logits and Gemma's large vocabulary.
    logits = logits.float()
    with stage("head/normalization"):
        log_probs = -F.cross_entropy(logits, labels, reduction="none")
    if not calculate_entropy:
        return log_probs
    probabilities = torch.softmax(logits, dim=-1)
    entropy = torch.logsumexp(logits, dim=-1) - torch.sum(probabilities * logits, dim=-1)
    return log_probs, entropy


def response_statistics(hidden, weight, labels, *, chunk_size, temperature=1.0, softcap=None, calculate_entropy=False):
    if chunk_size <= 0 or temperature <= 0:
        raise ValueError("Gemma policy chunk size and temperature must be positive")
    if hidden.shape[:-1] != labels.shape or not labels.numel():
        raise ValueError("Gemma response labels must match nonempty hidden-state positions")
    shape = labels.shape
    hidden = hidden.reshape(-1, hidden.shape[-1])
    labels = labels.reshape(-1)
    log_probs, entropies = [], []
    for start in range(0, len(labels), chunk_size):
        args = (
            hidden[start : start + chunk_size],
            weight,
            labels[start : start + chunk_size],
            temperature,
            softcap,
            calculate_entropy,
        )
        if torch.is_grad_enabled() and (hidden.requires_grad or weight.requires_grad):
            result = checkpoint(_chunk_statistics, *args, use_reentrant=False)
        else:
            result = _chunk_statistics(*args)
        if calculate_entropy:
            log_prob, entropy = result
            entropies.append(entropy)
        else:
            log_prob = result
        log_probs.append(log_prob)
    output = {"log_probs": torch.cat(log_probs).reshape(shape)}
    if calculate_entropy:
        output["entropys"] = torch.cat(entropies).reshape(shape)
    return output


def _policy_forward(
    self, *args, gemma4_response_labels=None, gemma4_temperature=1.0, gemma4_calculate_entropy=False, **kwargs
):
    if gemma4_response_labels is None:
        return self._gemma4_original_forward(*args, **kwargs)
    if args or kwargs.get("labels") is not None or kwargs.get("use_cache", False):
        raise ValueError("Gemma chunked policy requires keyword inputs, no LM labels, and use_cache=False")
    original_width = gemma4_response_labels.shape[1]
    if self._gemma4_trim_padding:
        from .trimming import trim_inputs

        with stage("trim_inputs"):
            kwargs, gemma4_response_labels, original_width = trim_inputs(
                kwargs, gemma4_response_labels, alignment=self._gemma4_trim_alignment
            )
    with stage("model_forward"):
        output = self.model(**kwargs, return_dict=True)
    response_length = gemma4_response_labels.shape[-1]
    if response_length == 0:
        # Keep a graph-connected zero so an empty response still participates in
        # FSDP backward collectives on every rank, with no supervised positions.
        zero = (output.last_hidden_state[:, :1, :].float().sum(-1) * 0).expand(1, original_width)
        return {"log_probs": zero, **({"entropys": zero} if gemma4_calculate_entropy else {})}
    hidden = output.last_hidden_state[:, -response_length - 1 : -1, :]
    statistics = response_statistics(
        hidden,
        self.lm_head.weight,
        gemma4_response_labels,
        chunk_size=self._gemma4_policy_chunk_size,
        temperature=gemma4_temperature,
        softcap=self.config.get_text_config().final_logit_softcapping,
        calculate_entropy=gemma4_calculate_entropy,
    )
    if self._gemma4_trim_padding:
        # Alignment may retain masked response tokens. They remain unsupervised
        # and, like removed padding, have zero externally visible statistics.
        valid = kwargs["attention_mask"][:, -response_length:].bool()
        statistics = {key: value.masked_fill(~valid, 0) for key, value in statistics.items()}
    # FSDP2 must traverse the output pytree to install its backward hooks.
    if original_width == response_length:
        return statistics
    return {key: F.pad(value, (0, original_width - response_length)) for key, value in statistics.items()}


def enable_chunked_policy(model, chunk_size, *, trim_padding=False, trim_alignment=1):
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    if not isinstance(model, Gemma4ForConditionalGeneration) or chunk_size <= 0:
        raise ValueError("Chunked Gemma policy requires Gemma4ForConditionalGeneration and positive chunk size")
    if not isinstance(trim_alignment, int) or trim_alignment < 1:
        raise ValueError("Gemma trim alignment must be a positive integer")
    if not hasattr(model, "_gemma4_original_forward"):
        model._gemma4_original_forward = model.forward
        model.forward = MethodType(_policy_forward, model)
    model._gemma4_policy_chunk_size = int(chunk_size)
    model._gemma4_trim_padding = bool(trim_padding)
    model._gemma4_trim_alignment = trim_alignment
