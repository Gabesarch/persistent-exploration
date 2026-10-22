"""Opt-in Gemma vision lookup: numerical, gradient, FSDP, and GPU memory checks."""

import copy
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from verl.models.gemma4.position_embeddings import enable_position_embedding_lookup


def embedder(dtype=torch.float32, device="cpu", *, full_size=False):
    from transformers.models.gemma4.configuration_gemma4 import Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionPatchEmbedder

    config = Gemma4VisionConfig(
        hidden_size=768 if full_size else 16, position_embedding_size=10240 if full_size else 32, patch_size=16
    )
    model = Gemma4VisionPatchEmbedder(config).to(device=device, dtype=dtype)
    with torch.no_grad():
        model.position_embedding_table.normal_(std=0.1)
    return model


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("dtype,autocast", [(torch.float32, False), (torch.bfloat16, False), (torch.bfloat16, True)])
def test_forward_and_gradient_equivalence(device, dtype, autocast):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU parity runs inside the submitted smoke")
    torch.manual_seed(42)
    original = embedder(dtype, device)
    patched = copy.deepcopy(original)
    enable_position_embedding_lookup(patched)
    positions = torch.tensor([[[0, 31], [7, 3], [7, 3], [-1, -1], [31, 0]]], device=device)
    padding = (positions == -1).all(-1)
    pixels = torch.randn(1, 5, 768, device=device, dtype=dtype)
    # Binary-exact probe isolates accumulation differences in repeated lookups.
    probe = torch.randint(-2, 3, (1, 5, 16), device=device).float() / 4
    with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=autocast):
        expected = original(pixels, positions, padding)
        actual = patched(pixels, positions, padding)
    assert actual.dtype == expected.dtype
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    (actual.float() * probe).sum().backward()
    (expected.float() * probe).sum().backward()
    for name, param in patched.named_parameters():
        torch.testing.assert_close(param.grad, dict(original.named_parameters())[name].grad, atol=0, rtol=0)
    pos = patched._position_embeddings(positions, padding)
    assert torch.count_nonzero(pos[padding]) == 0


def test_patch_is_instance_local_idempotent_and_preserves_state(monkeypatch):
    from verl.workers.config.engine import FSDPEngineConfig

    original, patched = embedder(), embedder()
    native_function = type(original)._position_embeddings
    parameters = dict(patched.named_parameters())
    state = copy.deepcopy(patched.state_dict())
    assert FSDPEngineConfig().gemma4_position_embedding_lookup is False
    assert enable_position_embedding_lookup(patched) == enable_position_embedding_lookup(patched) == 1
    assert type(original)._position_embeddings is native_function
    assert original._position_embeddings.__func__ is native_function
    assert patched.state_dict().keys() == state.keys()
    for name, value in patched.state_dict().items():
        torch.testing.assert_close(value, state[name], atol=0, rtol=0)
    assert all(param is parameters[name] for name, param in patched.named_parameters())

    def forbidden(*args, **kwargs):
        raise AssertionError("position lookup materialized a one-hot tensor")

    monkeypatch.setattr(torch.nn.functional, "one_hot", forbidden)
    patched._position_embeddings(torch.tensor([[[2, 3]]]), torch.tensor([[False]]))
    with pytest.raises((IndexError, RuntimeError)):
        patched._position_embeddings(torch.tensor([[[32, 0]]]), torch.tensor([[False]]))
    with pytest.raises(ValueError, match="no Gemma4VisionPatchEmbedder"):
        enable_position_embedding_lookup(torch.nn.Linear(2, 2))


def fsdp_worker(rank, directory):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    torch.set_num_threads(1)
    torch.manual_seed(42)
    reference = embedder(torch.bfloat16)
    patched = copy.deepcopy(reference)
    enable_position_embedding_lookup(patched)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(directory / "rendezvous"),
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        mesh = init_device_mesh("cpu", (2,), mesh_dim_names=("fsdp",))
        fully_shard(
            patched,
            mesh=mesh,
            mp_policy=MixedPrecisionPolicy(
                param_dtype=torch.bfloat16, reduce_dtype=torch.float32, cast_forward_inputs=False
            ),
        )
        positions = torch.tensor([[[0, 31], [7, 3], [7, 3], [-1, -1], [31, 0]]])
        padding = (positions == -1).all(-1)
        pixels = torch.randn(1, 5, 768, dtype=torch.bfloat16)
        probe = torch.randint(-2, 3, (1, 5, 16)).float() / 4
        ref_optim = torch.optim.SGD(reference.parameters(), lr=0.1)
        optim = torch.optim.SGD(patched.parameters(), lr=0.1)
        before = patched.position_embedding_table.full_tensor().detach().clone()
        for _ in range(2):
            ref_optim.zero_grad()
            optim.zero_grad()
            expected = reference(pixels, positions, padding)
            actual = patched(pixels, positions, padding)
            torch.testing.assert_close(actual, expected, atol=0, rtol=0)
            (actual.float() * probe).sum().backward()
            (expected.float() * probe).sum().backward()
            for name, param in patched.named_parameters():
                torch.testing.assert_close(
                    param.grad.full_tensor(), dict(reference.named_parameters())[name].grad, atol=0, rtol=0
                )
            ref_optim.step()
            optim.step()
        after = patched.position_embedding_table.full_tensor().detach()
        assert not torch.equal(before, after)
        torch.testing.assert_close(after, reference.position_embedding_table, atol=0, rtol=0)
    finally:
        dist.destroy_process_group()


def test_two_rank_fsdp_forward_backward_and_update(tmp_path):
    processes, logs = [], []
    try:
        for rank in range(2):
            log = (tmp_path / f"rank{rank}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-B", __file__, "--worker", str(rank), str(tmp_path)],
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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="GPU memory measurement runs in smoke allocation")
def test_gpu_memory_at_failed_batch_shape():
    # The failed production allocation was 29 images x 10080 patches x 2 x 10240 int64.
    # Run before loading the training model so both implementations can be measured.
    model = embedder(torch.bfloat16, "cuda", full_size=True)
    positions = torch.randint(0, 100, (29, 10080, 2), device="cuda")
    padding = torch.zeros((29, 10080), device="cuda", dtype=torch.bool)

    def measure():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        with torch.no_grad():
            result = model._position_embeddings(positions, padding)
        torch.cuda.synchronize()
        return result, torch.cuda.max_memory_allocated() - baseline

    expected, old_peak = measure()
    enable_position_embedding_lookup(model)
    actual, new_peak = measure()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert new_peak < old_peak / 10
    report = {
        "images": 29,
        "patches_per_image": 10080,
        "old_peak_extra_bytes": old_peak,
        "lookup_peak_extra_bytes": new_peak,
        "forward_exact": True,
    }
    print(json.dumps(report))
    if directory := os.environ.get("GEMMA4_ARTIFACT_DIR"):
        (Path(directory) / "position_embedding_memory.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    assert sys.argv[1] == "--worker"
    fsdp_worker(int(sys.argv[2]), Path(sys.argv[3]))
