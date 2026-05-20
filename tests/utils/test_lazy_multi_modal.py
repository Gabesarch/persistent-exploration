import asyncio

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from PIL import Image

from verl.utils.config import validate_config
from verl.utils.crop_cache import cleanup_crop_cache_paths
from verl.utils.lazy_multi_modal import (
    _process_vision_info_sync,
    rehydrate_multi_modal_inputs,
    should_rehydrate_multi_modal_inputs,
    strip_heavy_multi_modal_inputs,
)


class _FakeBatchFeature(dict):
    def convert_to_tensors(self, tensor_type):
        assert tensor_type == "pt"
        return self


class _FakeTokenizer:
    def decode(self, input_ids, skip_special_tokens=True):
        assert skip_special_tokens
        return "decoded prompt"


class _FakeImageProcessor:
    patch_size = 14


class _FakeProcessor:
    image_processor = _FakeImageProcessor()

    def __call__(self, **kwargs):
        assert kwargs["images"][0] == "image"
        image_count = len(kwargs["images"])
        image_grid_thw = torch.tensor([[1, 2 + idx, 3 + idx] for idx in range(image_count)])
        return _FakeBatchFeature(
            {
                "input_ids": torch.tensor([[1, 2]]),
                "attention_mask": torch.tensor([[1, 1]]),
                "pixel_values": torch.ones(image_count, 3),
                "image_grid_thw": image_grid_thw,
                "mm_token_type_ids": torch.tensor([[1, 1]]),
            }
        )


class _FakeDataset:
    @classmethod
    async def process_vision_info(cls, messages, image_patch_size, config):
        assert image_patch_size == 14
        assert config["max_pixels"] == 128
        assert messages[0]["role"] == "user"
        return ["image"], None


def test_strip_heavy_multi_modal_inputs_keeps_metadata():
    multi_modal_inputs = {
        "pixel_values": torch.ones(1, 3),
        "pixel_values_videos": torch.ones(1, 3),
        "image_grid_thw": torch.tensor([[1, 2, 3]]),
        "images_seqlens": torch.tensor([6]),
    }

    stripped = strip_heavy_multi_modal_inputs(multi_modal_inputs)

    assert "pixel_values" not in stripped
    assert "pixel_values_videos" not in stripped
    assert torch.equal(stripped["image_grid_thw"], torch.tensor([[1, 2, 3]]))
    assert torch.equal(stripped["images_seqlens"], torch.tensor([6]))
    assert should_rehydrate_multi_modal_inputs([stripped])


def test_rehydrate_multi_modal_inputs_matches_metadata():
    original_inputs = [{"image_grid_thw": torch.tensor([[1, 2, 3]]), "images_seqlens": torch.tensor([6])}]
    raw_prompts = np.array([[{"role": "user", "content": "look"}]], dtype=object)

    rebuilt = rehydrate_multi_modal_inputs(
        raw_prompts=raw_prompts,
        input_ids=torch.tensor([[1, 2, 3]]),
        tokenizer=_FakeTokenizer(),
        processor=_FakeProcessor(),
        dataset_cls=_FakeDataset,
        data_config={"max_pixels": 128},
        original_multi_modal_inputs=original_inputs,
        assert_metadata_match=True,
    )

    assert "pixel_values" in rebuilt[0]
    assert "mm_token_type_ids" not in rebuilt[0]
    assert torch.equal(rebuilt[0]["image_grid_thw"], torch.tensor([[1, 2, 3]]))
    assert torch.equal(rebuilt[0]["images_seqlens"], torch.tensor([6]))


def test_rehydrate_multi_modal_inputs_appends_crop_paths_in_order(tmp_path):
    crop_path = tmp_path / "crop.png"
    Image.new("RGB", (12, 10), color="red").save(crop_path)
    original_inputs = [
        {
            "image_grid_thw": torch.tensor([[1, 2, 3], [1, 3, 4]]),
            "images_seqlens": torch.tensor([6, 12]),
        }
    ]

    rebuilt = rehydrate_multi_modal_inputs(
        raw_prompts=[[{"role": "user", "content": "look"}]],
        input_ids=torch.tensor([[1, 2, 3]]),
        tokenizer=_FakeTokenizer(),
        processor=_FakeProcessor(),
        dataset_cls=_FakeDataset,
        data_config={"max_pixels": 128},
        original_multi_modal_inputs=original_inputs,
        crop_image_refs=[[{"path": str(crop_path), "image_index": 1}]],
        assert_metadata_match=True,
    )

    assert rebuilt[0]["pixel_values"].shape[0] == 2
    assert torch.equal(rebuilt[0]["image_grid_thw"], torch.tensor([[1, 2, 3], [1, 3, 4]]))
    assert torch.equal(rebuilt[0]["images_seqlens"], torch.tensor([6, 12]))


def test_rehydrate_multi_modal_inputs_raises_on_metadata_mismatch():
    original_inputs = [{"image_grid_thw": torch.tensor([[1, 9, 9]]), "images_seqlens": torch.tensor([81])}]

    with pytest.raises(ValueError, match="metadata mismatch"):
        rehydrate_multi_modal_inputs(
            raw_prompts=[[{"role": "user", "content": "look"}]],
            input_ids=torch.tensor([[1, 2, 3]]),
            tokenizer=_FakeTokenizer(),
            processor=_FakeProcessor(),
            dataset_cls=_FakeDataset,
            data_config={"max_pixels": 128},
            original_multi_modal_inputs=original_inputs,
            assert_metadata_match=True,
        )


def test_process_vision_info_sync_inside_running_event_loop():
    async def _run():
        output = _process_vision_info_sync(
            processor=_FakeProcessor(),
            dataset_cls=_FakeDataset,
            data_config={"max_pixels": 128},
            messages=[{"role": "user", "content": "look"}],
        )
        assert output["images"] == ["image"]
        assert "videos" not in output

    asyncio.run(_run())


def test_cleanup_crop_cache_paths_removes_files_and_tolerates_missing(tmp_path):
    crop_path = tmp_path / "crop.png"
    crop_path.write_bytes(b"crop")

    cleanup_crop_cache_paths([str(crop_path), str(tmp_path / "missing.png")])

    assert not crop_path.exists()


def _minimal_lazy_config(**overrides):
    config = {
        "actor_rollout_ref": {
            "actor": {"strategy": "fsdp2", "use_dynamic_bsz": True},
            "rollout": {"lazy_multi_modal_inputs": {"enabled": True, "assert_metadata_match": True}},
        },
        "trainer": {"use_legacy_worker_impl": "auto", "n_gpus_per_node": 1, "nnodes": 1},
    }
    config.update(overrides)
    return OmegaConf.create(config)


def test_lazy_multi_modal_inputs_rejects_critic():
    with pytest.raises(ValueError, match="actor/ref only"):
        validate_config(_minimal_lazy_config(), use_reference_policy=True, use_critic=True)


def test_lazy_multi_modal_inputs_rejects_new_worker_impl():
    config = _minimal_lazy_config(trainer={"use_legacy_worker_impl": "disable", "n_gpus_per_node": 1, "nnodes": 1})
    with pytest.raises(ValueError, match="legacy worker"):
        validate_config(config, use_reference_policy=True, use_critic=False)
