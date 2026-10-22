"""Two-process regression for Gemma rotary-buffer registration order."""

import json
import os
import subprocess
import sys
from pathlib import Path

import torch

MODEL = os.environ.get("GEMMA4_MODEL", "/scratch/gpfs/ZHUANGL/gs1693/hf_cache/google/gemma-4-E4B-it")


def worker(rank, directory):
    import datetime

    import torch.distributed as dist
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4TextRotaryEmbedding

    from verl.models.gemma4.fsdp import broadcast_buffers_by_name

    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method="file://" + str(directory / "rendezvous"),
        rank=rank,
        world_size=2,
        timeout=datetime.timedelta(seconds=45),
    )
    try:
        model = Gemma4TextRotaryEmbedding(AutoConfig.from_pretrained(MODEL).text_config)
        order = [(name, list(buf.shape)) for name, buf in model.named_buffers()]
        expected = {name: buf.clone() for name, buf in model.named_buffers()}
        if rank == 1:
            for buf in model.buffers():
                buf.add_(100)
        broadcast_buffers_by_name(model)
        assert all(torch.equal(buf, expected[name]) for name, buf in model.named_buffers())
        # A real schema mismatch must raise on both ranks before tensor broadcasts.
        if rank == 1:
            model.full_attention_inv_freq = torch.zeros(3)
        try:
            broadcast_buffers_by_name(model)
        except ValueError as exc:
            assert "differ across ranks" in str(exc)
        else:
            raise AssertionError("Mismatched buffer shape was not rejected")
        (directory / f"rank{rank}.json").write_text(json.dumps({"order": order, "passed": True}))
    finally:
        dist.destroy_process_group()


def test_real_gemma_buffers_sync_across_different_hash_seeds(tmp_path):
    processes = []
    logs = []
    try:
        for rank in range(2):
            log = (tmp_path / f"rank{rank}.log").open("w")
            logs.append(log)
            processes.append(
                subprocess.Popen(
                    [sys.executable, "-B", __file__, "--worker", str(rank), str(tmp_path)],
                    env={**os.environ, "PYTHONHASHSEED": str(rank + 1), "OMP_NUM_THREADS": "1"},
                    stdout=log,
                    stderr=subprocess.STDOUT,
                )
            )
        for rank, process in enumerate(processes):
            assert process.wait(timeout=120) == 0, (tmp_path / f"rank{rank}.log").read_text()
        results = [json.loads((tmp_path / f"rank{rank}.json").read_text()) for rank in range(2)]
        assert results[0]["order"] != results[1]["order"]
        assert sorted(results[0]["order"]) == sorted(results[1]["order"])
        assert all(result["passed"] for result in results)
    finally:
        for process in processes:
            if process.poll() is None:
                process.kill()
                process.wait()
        for log in logs:
            log.close()


def test_shared_fsdp_loader_keeps_original_buffer_order_by_default(monkeypatch):
    import torch.distributed.checkpoint.state_dict as state_dict

    from verl.utils import fsdp_utils
    from verl.workers.config.engine import FSDPEngineConfig

    model = torch.nn.Module()
    model.register_buffer("z", torch.ones(3))
    model.register_buffer("a", torch.ones(1))
    calls = []
    monkeypatch.setattr(fsdp_utils, "get_device_id", lambda: torch.device("cpu"))
    monkeypatch.setattr(fsdp_utils.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(fsdp_utils.dist, "broadcast", lambda buf, src: calls.append(buf.numel()))
    monkeypatch.setattr(state_dict, "set_model_state_dict", lambda *args, **kwargs: None)
    fsdp_utils.fsdp2_load_full_state_dict(model, model.state_dict())
    assert calls == [3, 1]
    assert FSDPEngineConfig().sync_buffers_by_name is False


if __name__ == "__main__":
    assert sys.argv[1] == "--worker"
    worker(int(sys.argv[2]), Path(sys.argv[3]))
