"""Smoke-only, same-trajectory padded/trimmed comparison before optimization."""

import json
import os
import time
from pathlib import Path

import torch

from .training_diagnostics import local


def benchmark_padding(actor, data):
    if any(p.grad is not None for p in actor.actor_module.parameters()):
        raise ValueError("Padding benchmark must run before gradient accumulation")
    actor.actor_module.train()
    sample = data[:1].to(torch.cuda.current_device())
    inputs = {**sample.batch, **sample.non_tensor_batch, "pad_token_id": data.meta_info.get("pad_token_id", 0)}
    valid = inputs["attention_mask"][:, -inputs["responses"].shape[1] :].bool()
    weights = inputs["response_mask"].float().clone()
    for key in ("retry_sft_mask", "retry_critic_mask"):
        if key in inputs:
            weights += inputs[key].float() * 0.01
    weights /= weights.sum().clamp_min(1)
    selected = {}
    for name, param in actor.actor_module.named_parameters():
        group = "text" if "language_model" in name else "vision" if "vision_tower" in name else None
        if group is not None and group not in selected and name.endswith(("q_proj.weight", "q_proj.linear.weight")):
            selected[group] = (name, param)
    if set(selected) != {"text", "vision"}:
        raise ValueError("Padding benchmark requires both text and vision gradient probes")
    results, outputs, gradients = {}, {}, {}
    original = actor.actor_module._gemma4_trim_padding
    try:
        # Preserve RNG for the actual update. Both modes see the same RNG state.
        rng = torch.random.get_rng_state()
        cuda_rng = torch.cuda.get_rng_state()
        for mode in ("padded", "trimmed"):
            with torch.random.fork_rng(devices=[torch.cuda.current_device()]):
                torch.random.set_rng_state(rng)
                torch.cuda.set_rng_state(cuda_rng)
                actor.actor_module._gemma4_trim_padding = mode == "trimmed"
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                started = time.perf_counter()
                output = actor._forward_micro_batch(inputs, temperature=data.meta_info["temperature"])["log_probs"]
                torch.cuda.synchronize()
                forward = time.perf_counter() - started
                loss = (output * weights).sum()
                started = time.perf_counter()
                loss.backward()
                torch.cuda.synchronize()
                backward = time.perf_counter() - started
                results[mode] = dict(
                    forward_seconds=forward,
                    backward_seconds=backward,
                    loss=float(loss.detach()),
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                )
                outputs[mode] = output.detach()[valid].float().cpu()
                gradients[mode] = {
                    name: local(param.grad).detach().float().cpu().clone()
                    for name, param in selected.values()
                    if param.grad is not None
                }
                actor.actor_optimizer.zero_grad(set_to_none=True)
                del output, loss
    finally:
        actor.actor_module._gemma4_trim_padding = original
        actor.actor_optimizer.zero_grad(set_to_none=True)
    expected_names = {name for name, _ in selected.values()}
    if gradients["padded"].keys() != expected_names or gradients["trimmed"].keys() != expected_names:
        raise RuntimeError("Missing text or vision gradient in padding benchmark")
    errors = []
    try:
        torch.testing.assert_close(outputs["trimmed"], outputs["padded"], atol=0.05, rtol=0.002)
    except AssertionError as error:
        errors.append(str(error))
    comparisons = {}
    for name, reference in gradients["padded"].items():
        actual = gradients["trimmed"][name]
        assert torch.isfinite(actual).all() and torch.isfinite(reference).all()
        relative = float((actual - reference).norm() / reference.norm().clamp_min(1e-12))
        if relative >= 0.05:
            errors.append(f"{name}: gradient relative L2 error {relative} >= 0.05")
        comparisons[name] = dict(relative_l2_error=relative, max_abs_error=float((actual - reference).abs().max()))
    rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
    report = dict(
        parity_passed=not errors,
        parity_errors=errors,
        timings=results,
        gradient_probes=comparisons,
        valid_logprob_max_abs_error=float((outputs["trimmed"] - outputs["padded"]).abs().max()) if valid.any() else 0,
        allocated_tokens=inputs["input_ids"].numel(),
        actual_tokens=int(inputs["attention_mask"].sum()),
        actual_response_tokens=int(valid.sum()),
        note="One paired probe per rank, padded first; includes FSDP communication and image reconstruction. "
        "No optimizer step; probe gradients cleared and RNG restored. Not a warmed throughput benchmark.",
    )
    path = Path(os.environ["GEMMA4_ARTIFACT_DIR"]) / f"padding_benchmark_rank{rank}.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"[Gemma4 padding benchmark] {path}: {results}", flush=True)

    if errors:
        raise AssertionError(f"Padding benchmark parity failed; see {path}: {errors}")
