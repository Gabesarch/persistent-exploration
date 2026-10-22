#!/usr/bin/env python3
"""Smoke checkpoint: merge all FSDP ranks, verify update, strict reload, GPU forward."""

import argparse
import json
from pathlib import Path


def load_smoke_state(actor_dir, model_path):
    import torch
    from torch.distributed.tensor import DTensor

    from verl.model_merger.base_model_merger import ModelMergerConfig
    from verl.model_merger.fsdp_model_merger import FSDPModelMerger

    merger = FSDPModelMerger(
        ModelMergerConfig(
            operation="merge", backend="fsdp", local_dir=str(actor_dir), hf_model_config_path=str(model_path)
        )
    )
    world_size = merger._get_world_size()
    paths = [actor_dir / f"model_world_size_{world_size}_rank_{rank}.pt" for rank in range(world_size)]
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing second-step checkpoint shard: {path}")
    state = torch.load(paths[0], map_location="cpu", weights_only=False)
    if world_size == 1:
        return {k: (v.to_local() if isinstance(v, DTensor) else v) for k, v in state.items()}, paths
    # Frozen audio tensors or replicated buffers can sort before the first shard.
    # Select an actual DTensor instead of inferring the mesh from the first key.
    pivot = next((value for value in state.values() if isinstance(value, DTensor)), None)
    if pivot is None:
        raise ValueError("Multi-rank Gemma FSDP2 checkpoint has no DTensor mesh")
    mesh_shape = tuple(pivot.device_mesh.mesh.shape)
    mesh_names = pivot.device_mesh.mesh_dim_names
    if mesh_shape != (world_size,) or mesh_names != ("fsdp",):
        raise ValueError(f"Unsupported smoke checkpoint mesh: {mesh_shape}, {mesh_names}")
    del state, pivot
    return merger._load_and_merge_state_dicts(world_size, world_size, mesh_shape, mesh_names), paths


def main():
    import torch
    from accelerate import init_empty_weights
    from PIL import Image
    from safetensors import safe_open
    from transformers import AutoConfig, AutoModelForImageTextToText

    from verl.models.gemma4 import protocol as native
    from verl.utils.tokenizer import hf_processor

    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    actor_dir = args.run / "global_step_2/actor"
    state, paths = load_smoke_state(actor_dir, args.model)
    # Quantization observer bounds are intentionally +/-inf; trained weights must be finite.
    bounded = {"input_min", "input_max", "output_min", "output_max"}
    bad = [
        k
        for k, v in state.items()
        if k.rsplit(".", 1)[-1] not in bounded and v.is_floating_point() and not torch.isfinite(v).all()
    ]
    if bad:
        raise AssertionError(f"Nonfinite checkpoint tensors: {bad}")
    index_path = args.model / "model.safetensors.index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text())["weight_map"]
    else:
        with safe_open(str(args.model / "model.safetensors"), framework="pt", device="cpu") as f:
            index = {key: "model.safetensors" for key in f.keys()}
    candidates = [k for k in state if k.endswith("q_proj.weight") and "language_model" in k]
    changes = {}
    for key in candidates[:3]:
        with safe_open(str(args.model / index[key]), framework="pt", device="cpu") as f:
            base = f.get_tensor(key)
        changes[key] = float((state[key].float() - base.float()).abs().max())
    if not changes or not any(changes.values()):
        raise AssertionError(f"No verified parameter update: {changes}")
    config = AutoConfig.from_pretrained(args.model)
    with init_empty_weights(include_buffers=False):
        model = AutoModelForImageTextToText.from_config(config, attn_implementation="sdpa", dtype=torch.bfloat16)
    incompatible = model.load_state_dict(state, strict=True, assign=True)
    model.tie_weights()
    del state
    if not torch.cuda.is_available():
        raise AssertionError("Checkpoint reload forward requires the smoke GPU")
    model.eval().to("cuda")
    processor = hf_processor(args.model)
    image = Image.new("RGB", (512, 512), color="white")
    text = native.render(
        processor, [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Describe the image."}]}]
    )
    inputs = processor(
        text=[text], images=[image], return_tensors="pt", add_special_tokens=False, max_soft_tokens=1120
    ).to("cuda")
    with torch.inference_mode():
        output = model(**inputs, use_cache=False, logits_to_keep=1)
    assert torch.isfinite(output.logits).all()
    result = {
        "checkpoint": str(actor_dir),
        "checkpoint_shards": [str(path) for path in paths],
        "parameter_max_abs_change_from_base": changes,
        "finite_checkpoint": True,
        "strict_reload": str(incompatible),
        "finite_reload_forward": True,
        "logits_dtype": str(output.logits.dtype),
        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
