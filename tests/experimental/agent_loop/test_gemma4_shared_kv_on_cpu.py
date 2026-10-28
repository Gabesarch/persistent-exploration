"""Real Torch FSDP2 + Gemma shared-KV forward/backward regression (CPU/Gloo)."""

import copy
import datetime
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import torch

MODEL = os.environ.get("GEMMA4_MODEL", "/scratch/gpfs/ZHUANGL/gs1693/hf_cache/google/gemma-4-E4B-it")


def worker(rank, directory, grouped):
    import torch.distributed as dist
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import MixedPrecisionPolicy
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextModel

    from verl.utils.fsdp_utils import apply_fsdp2

    torch.set_num_threads(1)
    config = AutoConfig.from_pretrained(MODEL).text_config
    # Retain E4B's actual 42-layer topology and source layers 22/23, shrink widths.
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
        setattr(config, key, value)
    config._attn_implementation = "sdpa"
    torch.manual_seed(42)
    reference = Gemma4TextModel(config).to(torch.bfloat16).train()
    model = copy.deepcopy(reference)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    world_size = 2 if grouped else 1
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(directory / "rendezvous"),
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=90),
    )
    try:
        mesh = init_device_mesh("cpu", (world_size,), mesh_dim_names=("fsdp",))
        apply_fsdp2(
            model,
            dict(
                mesh=mesh,
                reshard_after_forward=True,
                mp_policy=MixedPrecisionPolicy(
                    param_dtype=torch.bfloat16,
                    reduce_dtype=torch.float32,
                    cast_forward_inputs=False,
                ),
            ),
            {
                "wrap_policy": {"transformer_layer_cls_to_wrap": ["Gemma4TextDecoderLayer"]},
                "gemma4_group_text_layers": grouped,
            },
        )
        ids = torch.tensor([[2, 4, 6, 8, 9, 10]])
        if not grouped:
            with pytest.raises(KeyError, match="22"):
                model(input_ids=ids, use_cache=False)
            return
        probe = torch.randn(1, 6, 32, generator=torch.Generator().manual_seed(123))
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ref_optimizer = torch.optim.SGD(reference.parameters(), lr=0.1)
        ref_params = dict(reference.named_parameters())
        for name, param in model.named_parameters():
            torch.testing.assert_close(param.full_tensor(), ref_params[name], atol=0, rtol=0, msg=name)
        before = model.layers[22].self_attn.k_proj.weight.full_tensor().detach().clone()
        losses = []
        for step in range(2):
            optimizer.zero_grad(set_to_none=True)
            ref_optimizer.zero_grad(set_to_none=True)
            expected = reference(input_ids=ids, use_cache=False).last_hidden_state
            actual = model(input_ids=ids, use_cache=False).last_hidden_state
            torch.testing.assert_close(actual, expected, atol=0, rtol=0, msg=f"forward step {step}")
            loss = (actual.float() * probe).mean()
            (expected.float() * probe).mean().backward()
            loss.backward()
            losses.append(loss.item())
            for name, param in model.named_parameters():
                assert param.grad is not None, name
                grad = param.grad.full_tensor()
                assert grad.dtype == torch.bfloat16, name
                assert torch.isfinite(grad).all(), name
                torch.testing.assert_close(grad, ref_params[name].grad, atol=0, rtol=0, msg=name)
            for source in (22, 23):
                for projection in ("k_proj", "v_proj"):
                    grad = getattr(model.layers[source].self_attn, projection).weight.grad.full_tensor()
                    assert grad.count_nonzero() > 0, (source, projection)
            optimizer.step()
            ref_optimizer.step()
        assert not torch.equal(before, model.layers[22].self_attn.k_proj.weight.full_tensor())
        (directory / f"rank{rank}.json").write_text(json.dumps({"steps": 2, "losses": losses}))
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("grouped", [False, True], ids=["reproduce_key22", "two_rank_gradient_parity"])
def test_real_gemma_shared_kv_fsdp(tmp_path, grouped):
    processes, logs = [], []
    try:
        for rank in range(2 if grouped else 1):
            log = (tmp_path / f"rank{rank}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-B", __file__, "--worker", str(rank), str(tmp_path), str(int(grouped))],
                    env={**os.environ, "OMP_NUM_THREADS": "1"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        for rank, process in enumerate(processes):
            assert process.wait(timeout=180) == 0, (tmp_path / f"rank{rank}.log").read_text()
        if grouped:
            results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
            assert results[0] == results[1]
            assert results[0]["steps"] == 2
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def test_grouping_is_opt_in_and_rejects_other_models():
    from verl.models.gemma4.fsdp import group_text_wrap_targets
    from verl.workers.config.engine import FSDPEngineConfig

    assert FSDPEngineConfig().gemma4_group_text_layers is False
    with pytest.raises(ValueError, match="exactly one Gemma4TextModel"):
        group_text_wrap_targets(torch.nn.Linear(2, 2), [])


if __name__ == "__main__":
    assert sys.argv[1] == "--worker"
    worker(int(sys.argv[2]), Path(sys.argv[3]), bool(int(sys.argv[4])))
