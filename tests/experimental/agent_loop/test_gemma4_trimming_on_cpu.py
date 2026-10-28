"""Padding trim equivalence, multimodal alignment, and opt-in instrumentation."""

import copy
import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_gemma4_chunked_policy_on_cpu import tiny_model

from verl.models.gemma4.chunked_policy import enable_chunked_policy
from verl.models.gemma4.profiling import install_module_ranges, profile_actor
from verl.models.gemma4.trimming import trim_inputs, validate_loss_masks


def make_inputs(left=3, right=5, count=4):
    prompt = [2, 5, 8]
    response = [9, 10, 11, 12][:count]
    ids = torch.tensor([[0] * left + prompt + response + [0] * right])
    mask = torch.tensor([[0] * left + [1] * (len(prompt) + count) + [0] * right])
    return dict(
        input_ids=ids, attention_mask=mask, position_ids=(mask.cumsum(-1) - 1).clamp_min(0), use_cache=False
    ), ids[:, len(prompt) + left :]


@pytest.mark.parametrize("left,right,count", [(3, 5, 4), (0, 0, 4), (2, 0, 4), (0, 3, 1), (3, 2, 0)])
@pytest.mark.parametrize("entropy", [False, True])
def test_trim_forward_gradients(left, right, count, entropy):
    torch.manual_seed(71)
    reference = tiny_model().float().train()
    model = copy.deepcopy(reference)
    enable_chunked_policy(reference, 3)
    enable_chunked_policy(model, 3, trim_padding=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    inputs, labels = make_inputs(left, right, count)
    expected = reference(**inputs, gemma4_response_labels=labels, gemma4_calculate_entropy=entropy)
    actual = model(**inputs, gemma4_response_labels=labels, gemma4_calculate_entropy=entropy)
    weights = torch.zeros_like(labels, dtype=torch.float32)
    # Disjoint policy, injected-prefill, and one-token critic-like supervision.
    weights[:, :count] = torch.tensor([0.2, -0.3, 0.01, 0.1])[:count]
    loss, ref_loss = 0, 0
    for key in expected:
        torch.testing.assert_close(actual[key][:, :count], expected[key][:, :count], atol=2e-5, rtol=2e-5)
        assert actual[key].shape == labels.shape
        assert actual[key][:, count:].count_nonzero() == 0
        loss = loss + (actual[key] * weights).sum()
        ref_loss = ref_loss + (expected[key] * weights).sum()
    loss.backward()
    ref_loss.backward()
    for name, param in model.named_parameters():
        ref_grad = dict(reference.named_parameters())[name].grad
        if param.grad is None:
            assert ref_grad is None or ref_grad.count_nonzero() == 0, name
        elif ref_grad is not None:
            # Sequence-length changes alter SDPA accumulation; compare the whole
            # gradient as well as forward values, avoiding near-zero element ratios.
            delta = (param.grad - ref_grad).norm()
            assert delta <= 3e-4 * ref_grad.norm() + 1e-7, (name, float(delta), float(ref_grad.norm()))


def test_alignment_and_rejections():
    inputs, labels = make_inputs()
    inputs["mm_token_type_ids"] = torch.arange(inputs["input_ids"].numel()).unsqueeze(0)
    inputs["image_position_ids"] = torch.tensor([[[0, 0], [1, 0]], [[0, 0], [0, 1]]])
    original = {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in inputs.items()}
    trimmed, short_labels, width = trim_inputs(inputs, labels)
    assert width == 9 and short_labels.shape == (1, 4)
    assert trimmed["image_position_ids"] is inputs["image_position_ids"]
    for key in ("input_ids", "attention_mask", "position_ids", "mm_token_type_ids"):
        assert torch.equal(trimmed[key], inputs[key][:, 3:10])
        assert torch.equal(inputs[key], original[key])
    for key, value in [("attention_mask", torch.ones(1, 2)), ("position_ids", torch.ones(2, 15))]:
        with pytest.raises(ValueError):
            trim_inputs({**inputs, key: value}, labels)
    hole = inputs["attention_mask"].clone()
    hole[:, 5] = 0
    with pytest.raises(ValueError, match="internal gaps"):
        trim_inputs({**inputs, "attention_mask": hole}, labels)
    with pytest.raises(ValueError, match="one trajectory"):
        trim_inputs({**inputs, "input_ids": inputs["input_ids"].repeat(2, 1)}, labels)
    with pytest.raises(ValueError, match="suffix"):
        trim_inputs(inputs, labels + 1)
    for key in ("response_mask", "retry_sft_mask", "retry_critic_mask"):
        with pytest.raises(ValueError, match="supervised padding"):
            validate_loss_masks({**inputs, "responses": labels, key: torch.ones_like(labels)})


@pytest.mark.parametrize("alignment", [1, 4, 512])
def test_sliding_window_boundary_and_actor_masks(alignment):
    """Exercise a window crossing without changing position IDs or loss layout."""
    from omegaconf import OmegaConf

    from verl.workers.actor.dp_actor import DataParallelPPOActor

    torch.manual_seed(23)
    reference = tiny_model().float().train()
    # Keep all shared-KV layers but shrink the sliding window for a short fixture.
    reference.config.text_config.sliding_window = 4
    for layer in reference.model.language_model.layers:
        if layer.self_attn.sliding_window is not None:
            layer.self_attn.sliding_window = 4
    model = copy.deepcopy(reference)
    enable_chunked_policy(reference, 2)
    enable_chunked_policy(model, 2, trim_padding=True, trim_alignment=alignment)
    actor = object.__new__(DataParallelPPOActor)
    actor.actor_module = model
    actor.config = OmegaConf.create({"fsdp_config": {"gemma4_policy_chunk_size": 2, "gemma4_trim_padding": True}})
    actor.use_prefix_grouper = actor.use_remove_padding = actor.use_fused_kernels = actor.use_ulysses_sp = False
    actor.device_name, actor.param_dtype = "cpu", torch.bfloat16
    inputs, labels = make_inputs(left=7, right=2)
    inputs.pop("use_cache")
    batch = {
        **inputs,
        "responses": labels,
        "response_mask": torch.tensor([[1, 1, 0, 0, 0, 0]]),
        "retry_sft_mask": torch.tensor([[0, 0, 1, 0, 0, 0]]),
        "retry_critic_mask": torch.tensor([[0, 0, 0, 1, 0, 0]]),
    }
    with torch.no_grad(), torch.autocast("cpu", dtype=torch.bfloat16):
        expected = reference(**inputs, use_cache=False, gemma4_response_labels=labels)["log_probs"]
        actual = actor._forward_micro_batch(batch, temperature=1.0)["log_probs"]
    torch.testing.assert_close(actual[:, :4], expected[:, :4], atol=0.03, rtol=0.01)
    assert actual.shape == labels.shape and actual[:, 4:].count_nonzero() == 0
    batch["retry_critic_mask"][0, -1] = 1
    with pytest.raises(ValueError, match="supervised padding"):
        actor._forward_micro_batch(batch, temperature=1.0)


def test_real_processor_and_lazy_crop_alignment():
    from PIL import Image

    from verl.models.gemma4.processing import multimodal_inputs
    from verl.utils.lazy_multi_modal import _compute_multi_modal_inputs_for_sample
    from verl.utils.tokenizer import hf_processor

    processor = hf_processor("/scratch/gpfs/ZHUANGL/gs1693/hf_cache/google/gemma-4-E4B-it")
    images = [Image.new("RGB", (2000, 1000)), Image.new("RGB", (1200, 600))]
    encoded = processor(
        text=["<|image|> original <|image|> crop answer"],
        images=images,
        return_tensors="pt",
        add_special_tokens=False,
        max_soft_tokens=1120,
    )
    ids = encoded["input_ids"]
    padded = torch.nn.functional.pad(ids, (7, 13), value=processor.tokenizer.pad_token_id)
    mask = torch.nn.functional.pad(torch.ones_like(ids), (7, 13))
    mm = _compute_multi_modal_inputs_for_sample(
        tokenizer=processor.tokenizer, processor=processor, input_ids=padded[0], multi_modal_data={"images": images}
    )
    kwargs = {**mm, "input_ids": padded, "attention_mask": mask, "position_ids": mask.cumsum(-1) - 1}
    trimmed, labels, width = trim_inputs(kwargs, padded[:, -23:])
    assert labels.shape[1] == 10 and width == 23
    expected = multimodal_inputs(processor, ids, images)
    assert torch.equal(trimmed["input_ids"], ids)
    for key in ("mm_token_type_ids", "pixel_values", "image_position_ids"):
        assert torch.equal(trimmed[key], expected[key]), key


def vision_fixture(device, dtype):
    from transformers import AutoConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4ForConditionalGeneration

    config = tiny_model().config
    config.vision_config = AutoConfig.from_pretrained(
        "/scratch/gpfs/ZHUANGL/gs1693/hf_cache/google/gemma-4-E4B-it"
    ).vision_config
    for key, value in dict(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        head_dim=8,
        global_head_dim=8,
        standardize=False,
    ).items():
        setattr(config.vision_config, key, value)
    config.vision_config._attn_implementation = "sdpa"
    config.image_token_id = 63
    torch.manual_seed(6)
    reference = Gemma4ForConditionalGeneration(config).to(device=device, dtype=dtype).train()
    model = copy.deepcopy(reference)
    enable_chunked_policy(reference, 3)
    enable_chunked_policy(model, 3, trim_padding=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    # Each 3x3 patch image becomes one soft token. Second image is a tool crop
    # inside the response; keep its feature while excluding it from policy loss.
    ids = torch.tensor([[0, 0, 2, 63, 4, 9, 63, 10, 11, 0, 0]], device=device)
    mask = (ids != 0).long()
    labels = ids[:, 5:]
    k = config.vision_config.patch_size
    pixels = torch.randn(2, 9, k * k * 3, device=device, dtype=dtype)
    positions = torch.tensor([[[x, y] for y in range(3) for x in range(3)]] * 2, device=device)
    inputs = dict(
        input_ids=ids,
        attention_mask=mask,
        position_ids=(mask.cumsum(-1) - 1).clamp_min(0),
        pixel_values=pixels,
        image_position_ids=positions,
        mm_token_type_ids=(ids == 63).long(),
        use_cache=False,
    )
    return reference, model, inputs, labels


@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_real_vision_original_and_crop_gradients(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("GPU parity in smoke")
    from torch.nn.attention import SDPBackend, sdpa_kernel

    reference, model, inputs, labels = vision_fixture(device, torch.float32)
    # FP32 math attention is the independent numerical oracle. BF16 agreement
    # across different shapes is not an appropriate per-parameter oracle.
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        with sdpa_kernel(SDPBackend.MATH):
            expected = reference(**inputs, gemma4_response_labels=labels)["log_probs"]
            actual = model(**inputs, gemma4_response_labels=labels)["log_probs"]
            torch.testing.assert_close(actual[:, :4], expected[:, :4], atol=3e-5, rtol=3e-4)
            weights = torch.tensor([[0.2, 0, -0.3, 0.01, 0, 0]], device=device)
            (actual * weights).sum().backward()
            (expected * weights).sum().backward()
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    for name, param in model.named_parameters():
        ref = dict(reference.named_parameters())[name]
        if param.grad is None:
            assert ref.grad is None, name
        else:
            delta = (param.grad - ref.grad).norm()
            assert delta <= 3e-4 * ref.grad.norm() + 1e-7, (name, float(delta), float(ref.grad.norm()))
    assert model.model.vision_tower.patch_embedder.input_proj.weight.grad.abs().sum() > 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="BF16 CUDA shape/rounding validation")
def test_bf16_trimming_against_independent_compact_input():
    padded, model, inputs, labels = vision_fixture("cuda", torch.bfloat16)
    compact = copy.deepcopy(padded)
    # Independent, explicit fixture layout: retain original image plus crop and
    # all four real response tokens. Never call trim_inputs for the reference.
    compact_inputs = dict(inputs)
    for key in ("input_ids", "attention_mask", "position_ids", "mm_token_type_ids"):
        compact_inputs[key] = inputs[key][:, 2:9]
    expected = compact(**compact_inputs, gemma4_response_labels=labels[:, :4])["log_probs"]
    actual = model(**inputs, gemma4_response_labels=labels)["log_probs"]
    padded_lp = padded(**inputs, gemma4_response_labels=labels)["log_probs"]
    torch.testing.assert_close(actual[:, :4], expected, atol=0, rtol=0)
    torch.testing.assert_close(actual[:, :4], padded_lp[:, :4], atol=0.04, rtol=0.01)
    weights = torch.tensor([[0.2, 0, -0.3, 0.01, 0, 0]], device="cuda")
    (actual * weights).sum().backward()
    (expected * weights[:, :4]).sum().backward()
    (padded_lp * weights).sum().backward()
    rows, delta_squared, reference_squared = [], 0.0, 0.0
    for name, param in model.named_parameters():
        ref = dict(compact.named_parameters())[name]
        base = dict(padded.named_parameters())[name]
        if param.grad is None:
            assert ref.grad is None and base.grad is None, name
            continue
        # Exact BF16 gradient equivalence to the independent compact layout,
        # including gradient checkpointing; no relaxed implementation tolerance.
        torch.testing.assert_close(param.grad, ref.grad, atol=0, rtol=0, msg=name)
        assert torch.isfinite(param.grad).all() and torch.isfinite(base.grad).all()
        delta = float((param.grad.float() - base.grad.float()).norm())
        norm = float(base.grad.float().norm())
        delta_squared += delta**2
        reference_squared += norm**2
        rows.append(dict(name=name, relative_l2=delta / max(norm, 1e-12), absolute_l2=delta))
    global_relative = (delta_squared / max(reference_squared, 1e-30)) ** 0.5
    report = dict(
        device=torch.cuda.get_device_name(),
        independent_compact_gradients_exact=True,
        padded_vs_trimmed_global_relative_l2=global_relative,
        padded_vs_trimmed_parameters=sorted(rows, key=lambda r: r["relative_l2"], reverse=True),
        note="Per-parameter BF16 differences across shapes are diagnostic; mathematical parity "
        "is gated separately in FP32 and implementation parity is exact against compact BF16.",
    )
    if directory := os.environ.get("GEMMA4_ARTIFACT_DIR"):
        (Path(directory) / "trimming_precision.json").write_text(json.dumps(report, indent=2) + "\n")
    assert global_relative < 0.05, report


def test_profiler_is_opt_in_and_exports(tmp_path, monkeypatch):
    from verl.workers.config.engine import FSDPEngineConfig

    cfg = FSDPEngineConfig()
    assert not cfg.gemma4_trim_padding and not cfg.gemma4_profile_actor and not cfg.gemma4_benchmark_padding
    monkeypatch.setenv("GEMMA4_ARTIFACT_DIR", str(tmp_path))
    model = tiny_model().float()
    enable_chunked_policy(model, 2, trim_padding=True)
    install_module_ranges(model)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    inputs, labels = make_inputs()
    actor = SimpleNamespace(config=SimpleNamespace(fsdp_config={"gemma4_profile_actor": False}))

    @profile_actor
    def probe(self):
        output = model(**inputs, gemma4_response_labels=labels)["log_probs"]
        output.sum().backward()
        return output.detach()

    expected = probe(actor)
    assert not list(tmp_path.iterdir())
    model.zero_grad()
    actor.config.fsdp_config["gemma4_profile_actor"] = True
    actual = probe(actor)
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    report = json.loads((tmp_path / "actor_profiles/probe_rank0_call1.json").read_text())
    names = {row["name"] for row in report["events"]}
    assert "gemma4/model_forward" in names and "gemma4/head/projection" in names
    assert list(tmp_path.glob("actor_profiles/*.trace.json.gz"))
    assert list(tmp_path.glob("actor_profiles/*.pstats"))


@pytest.mark.parametrize("alignment", [1, 32, 128, 512])
@pytest.mark.parametrize("response_count", [0, 1775])
def test_aligned_trim_preserves_valid_ids_positions_and_loss_layout(alignment, response_count):
    ids = torch.arange(4096).reshape(1, -1)
    mask = torch.zeros_like(ids)
    mask[:, 1701 : 2048 + response_count] = 1
    inputs = dict(input_ids=ids, attention_mask=mask, position_ids=ids + 9, mm_token_type_ids=(ids == 1800).long())
    labels = ids[:, 2048:]
    trimmed, short_labels, width = trim_inputs(inputs, labels, alignment=alignment)
    start = 1701 // alignment * alignment
    end = min(4096, (2048 + response_count + alignment - 1) // alignment * alignment)
    assert trimmed["input_ids"].shape[1] == end - start
    assert start % alignment == 0
    assert (4096 - end) % alignment == 0
    assert width == 2048
    assert torch.equal(short_labels, labels[:, : end - 2048])
    for key in inputs:
        assert torch.equal(trimmed[key], inputs[key][:, start:end])
    assert torch.equal(trimmed["input_ids"][trimmed["attention_mask"].bool()], ids[mask.bool()])
    for invalid in [0, -1, 1.5]:
        with pytest.raises(ValueError, match="alignment"):
            trim_inputs(inputs, labels, alignment=invalid)
