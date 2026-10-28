"""Gemma chunked projection: dense reference, gradients, tied weights, FSDP2."""

import copy
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from verl.models.gemma4.chunked_policy import enable_chunked_policy, response_statistics


def dense_statistics(hidden, weight, labels, temperature, softcap):
    logits = F.linear(hidden, weight)
    if softcap is not None:
        logits = torch.tanh(logits / softcap) * softcap
    logits = (logits / temperature).float()
    lp = F.log_softmax(logits, -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    entropy = torch.logsumexp(logits, -1) - (F.softmax(logits, -1) * logits).sum(-1)
    return lp, entropy


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("entropy", [False, True])
def test_statistics_forward_backward(device, dtype, entropy):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA parity runs in GPU smoke")
    torch.manual_seed(3)
    hidden = torch.randn(2, 7, 16, device=device, dtype=dtype, requires_grad=True)
    weight = torch.randn(37, 16, device=device, dtype=dtype, requires_grad=True)
    ref_h, ref_w = (v.detach().clone().requires_grad_() for v in (hidden, weight))
    labels = torch.randint(0, 37, (2, 7), device=device)
    probe = torch.randn(2, 7, device=device)
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=dtype == torch.bfloat16):
        expected_lp, expected_entropy = dense_statistics(ref_h, ref_w, labels, 0.7, 30.0)
        actual = response_statistics(
            hidden, weight, labels, chunk_size=3, temperature=0.7, softcap=30.0, calculate_entropy=entropy
        )
    tolerance = dict(atol=0.015, rtol=0.02) if dtype == torch.bfloat16 else dict(atol=3e-6, rtol=3e-5)
    torch.testing.assert_close(actual["log_probs"], expected_lp, **tolerance)
    loss, ref_loss = actual["log_probs"] * probe, expected_lp * probe
    if entropy:
        torch.testing.assert_close(actual["entropys"], expected_entropy, **tolerance)
        loss = loss + 0.1 * actual["entropys"]
        ref_loss = ref_loss + 0.1 * expected_entropy
    else:
        assert "entropys" not in actual
    loss.sum().backward()
    ref_loss.sum().backward()
    torch.testing.assert_close(hidden.grad, ref_h.grad, **tolerance)
    torch.testing.assert_close(weight.grad, ref_w.grad, **tolerance)


def tiny_model():
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    config = AutoConfig.from_pretrained("/scratch/gpfs/ZHUANGL/gs1693/hf_cache/google/gemma-4-E4B-it")
    config.audio_config = config.vision_config = None
    for key, value in dict(
        hidden_size=32,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        global_head_dim=8,
        vocab_size=64,
        vocab_size_per_layer_input=64,
        hidden_size_per_layer_input=8,
    ).items():
        setattr(config.text_config, key, value)
    config._attn_implementation = config.text_config._attn_implementation = "sdpa"
    return Gemma4ForConditionalGeneration(config).to(torch.bfloat16)


def test_dense_model_chunked_head_fp32_gradients():
    """Independent native dense reference for projection/temperature/loss math.

    FSDP tests use identical heads to avoid conflating sharding with reduced-
    precision reduction order; this test retains the independent dense oracle.
    """
    torch.manual_seed(42)
    reference = tiny_model().float()
    model = copy.deepcopy(reference)
    enable_chunked_policy(model, 2)
    ids = torch.tensor([[0, 2, 4, 6, 8, 9, 10, 0]])
    mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]])
    labels = ids[:, -4:]
    actual = model(
        input_ids=ids, attention_mask=mask, use_cache=False, gemma4_response_labels=labels, gemma4_temperature=0.7
    )["log_probs"]
    logits = reference(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, -5:-1, :] / 0.7
    expected = F.log_softmax(logits.float(), -1).gather(-1, labels.unsqueeze(-1)).squeeze(-1)
    torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
    weights = torch.tensor([[0.2, -0.3, 0.01, 0.0]])
    (actual * weights).sum().backward()
    (expected * weights).sum().backward()
    for name, param in model.named_parameters():
        ref_grad = dict(reference.named_parameters())[name].grad
        delta = (param.grad - ref_grad).norm()
        assert delta <= 3e-4 * ref_grad.norm() + 1e-7, (name, float(delta), float(ref_grad.norm()))


def fsdp_worker(rank, directory, trim=False):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy

    from verl.utils.fsdp_utils import apply_fsdp2

    torch.set_num_threads(1)
    torch.manual_seed(42)
    reference = tiny_model()
    patched = copy.deepcopy(reference)
    enable_chunked_policy(patched, 2, trim_padding=trim)
    # Compare identical numerical programs to isolate sharding/checkpointing.
    # Dense-vs-chunked and padded-vs-trimmed math have separate oracle tests.
    enable_chunked_policy(reference, 2, trim_padding=trim)
    if trim:
        from verl.models.gemma4.profiling import install_module_ranges

        install_module_ranges(patched)
    patched.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(directory / "rendezvous"),
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("fsdp",))
        apply_fsdp2(
            patched,
            dict(
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False
                ),
            ),
            {
                "wrap_policy": {"transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer"]},
                "gemma4_group_text_layers": True,
            },
        )
        ids = torch.tensor([[0, 2, 4, 6, 8, 9, 10, 0]])
        mask = torch.tensor([[0, 1, 1, 1, 1, 1, 1, 0]])
        labels = ids[:, -4:]
        optim = torch.optim.SGD(patched.parameters(), lr=0.01)
        ref_optim = torch.optim.SGD(reference.parameters(), lr=0.01)
        for _ in range(2):
            optim.zero_grad()
            ref_optim.zero_grad()
            actual = patched(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                gemma4_response_labels=labels,
                gemma4_temperature=0.7,
                gemma4_calculate_entropy=False,
            )
            expected = reference(
                input_ids=ids,
                attention_mask=mask,
                use_cache=False,
                gemma4_response_labels=labels,
                gemma4_temperature=0.7,
            )["log_probs"]
            torch.testing.assert_close(actual["log_probs"][:, :3], expected[:, :3], atol=0.02, rtol=0.01)
            # Includes response padding with zero loss mask, and a single-token sidecar-like label.
            weights = torch.tensor([[0.2, -0.3, 0.01, 0.0]])
            (actual["log_probs"] * weights).sum().backward()
            (expected * weights).sum().backward()
            for name, param in patched.named_parameters():
                grad = param.grad.full_tensor()
                assert torch.isfinite(grad).all(), name
                ref_grad = dict(reference.named_parameters())[name].grad
                torch.testing.assert_close(grad, ref_grad, atol=0, rtol=0, msg=name)
            optim.step()
            ref_optim.step()
        assert patched.lm_head.weight is patched.model.language_model.embed_tokens.weight
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("trim", [False, True])
def test_fsdp_tied_head_checkpointed_backward(tmp_path, trim):
    processes, logs = [], []
    try:
        for rank in range(2):
            log = (tmp_path / f"rank{rank}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-B", __file__, "--worker", str(rank), str(tmp_path), str(int(trim))],
                    env={**os.environ, "OMP_NUM_THREADS": "1"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        for rank, proc in enumerate(processes):
            assert proc.wait(timeout=180) == 0, (tmp_path / f"rank{rank}.log").read_text()
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.kill()
                proc.wait()
        for log in logs:
            log.close()


def test_patch_keeps_native_forward_and_defaults():
    from verl.workers.config.engine import FSDPEngineConfig

    assert FSDPEngineConfig().gemma4_policy_chunk_size == 0
    assert not FSDPEngineConfig().gemma4_skip_log_prob_entropy
    model = tiny_model().eval()
    ids = torch.tensor([[2, 3, 4]])
    with torch.no_grad():
        expected = model(input_ids=ids, use_cache=False).logits
        enable_chunked_policy(model, 2)
        actual = model(input_ids=ids, use_cache=False).logits
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    with pytest.raises(ValueError):
        enable_chunked_policy(torch.nn.Linear(2, 2), 2)


def test_padded_actor_routes_to_chunked_head_without_entropy():
    from omegaconf import OmegaConf

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    model = tiny_model().eval()
    enable_chunked_policy(model, 2)
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.config = OmegaConf.create({"fsdp_config": {"gemma4_policy_chunk_size": 2}})
    actor.use_prefix_grouper = actor.use_remove_padding = actor.use_fused_kernels = actor.use_ulysses_sp = False
    actor.device_name, actor.param_dtype = "cpu", torch.bfloat16
    ids = torch.tensor([[0, 2, 3, 4, 5, 6, 0]])
    batch = {
        "input_ids": ids,
        "responses": ids[:, -3:],
        "attention_mask": (ids != 0).long(),
        "position_ids": torch.arange(7).unsqueeze(0),
    }
    with torch.no_grad():
        expected = model(
            input_ids=ids, attention_mask=batch["attention_mask"], position_ids=batch["position_ids"], use_cache=False
        ).logits[:, -4:-1, :]
        expected = F.log_softmax(expected.float(), -1).gather(-1, batch["responses"].unsqueeze(-1)).squeeze(-1)
        actual = actor._forward_micro_batch(batch, temperature=1.0, calculate_entropy=False)
    assert set(actual) == {"log_probs"}
    torch.testing.assert_close(actual["log_probs"], expected, atol=0.01, rtol=0.01)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Full response/vocab memory test requires GPU")
def test_production_response_and_vocab_memory():
    # Exact production response/vocabulary dimensions; small hidden width keeps
    # this targeted statistics test fast. Actual E4B training follows in the job.
    torch.manual_seed(11)
    hidden = torch.randn(1, 24576, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    weight = torch.randn(262144, 32, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    labels = torch.randint(0, 262144, (1, 24576), device="cuda")
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        result = response_statistics(hidden, weight, labels, chunk_size=128, calculate_entropy=False)
        loss = result["log_probs"].mean()
    loss.backward()
    torch.cuda.synchronize()
    peak = torch.cuda.max_memory_allocated() - baseline
    assert torch.isfinite(loss) and torch.isfinite(hidden.grad).all() and torch.isfinite(weight.grad).all()
    assert peak < 4 * 2**30, peak
    report = {
        "response_tokens": 24576,
        "vocab_size": 262144,
        "chunk_size": 128,
        "hidden_width": 32,
        "entropy_computed": False,
        "forward_backward_finite": True,
        "peak_extra_bytes": peak,
    }
    print(json.dumps(report))
    if directory := os.environ.get("GEMMA4_ARTIFACT_DIR"):
        (Path(directory) / "chunked_policy_memory.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    fsdp_worker(int(sys.argv[2]), Path(sys.argv[3]), bool(int(sys.argv[4])))
