# Copyright 2026 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import concurrent.futures
from collections.abc import Mapping, Sequence
from numbers import Integral
from typing import Any

import numpy as np
import torch

from verl.utils.crop_cache import load_crop_images, normalize_crop_image_refs

HEAVY_MULTI_MODAL_INPUT_KEYS = {
    "pixel_values",
    "pixel_values_videos",
    "image_embeds",
    "video_embeds",
}

METADATA_COMPARE_KEYS = (
    "image_grid_thw",
    "video_grid_thw",
    "images_seqlens",
    "image_position_ids",
    "mm_token_type_ids",
)


def lazy_multi_modal_inputs_enabled(config: Any) -> bool:
    lazy_config = _get_lazy_config(config)
    return bool(_config_get(lazy_config, "enabled", False))


def lazy_multi_modal_assert_metadata_match(config: Any) -> bool:
    lazy_config = _get_lazy_config(config)
    return bool(_config_get(lazy_config, "assert_metadata_match", True))


def strip_heavy_multi_modal_inputs(multi_modal_inputs: dict[str, Any] | None) -> dict[str, Any] | None:
    """Drop large processor tensors while preserving metadata needed for later checks."""
    if multi_modal_inputs is None:
        return None
    return {key: value for key, value in multi_modal_inputs.items() if key not in HEAVY_MULTI_MODAL_INPUT_KEYS}


def should_rehydrate_multi_modal_inputs(batch_multi_modal_inputs: Sequence[Any]) -> bool:
    """Return True when metadata is present but heavy image/video tensors were stripped."""
    for sample_inputs in batch_multi_modal_inputs:
        if sample_inputs is None:
            continue
        if hasattr(sample_inputs, "data"):
            sample_inputs = sample_inputs.data
        if not isinstance(sample_inputs, Mapping):
            continue
        has_image_metadata = (
            "image_grid_thw" in sample_inputs or "image_position_ids" in sample_inputs
        ) and "pixel_values" not in sample_inputs
        has_video_metadata = "video_grid_thw" in sample_inputs and "pixel_values_videos" not in sample_inputs
        if has_image_metadata or has_video_metadata:
            return True
    return False


def rehydrate_multi_modal_inputs(
    *,
    raw_prompts: Sequence[Any],
    input_ids: torch.Tensor,
    tokenizer: Any,
    processor: Any,
    dataset_cls: Any,
    data_config: Any,
    original_multi_modal_inputs: Sequence[Any] | None = None,
    crop_image_refs: Sequence[Any] | None = None,
    assert_metadata_match: bool = True,
) -> list[dict[str, torch.Tensor]]:
    if processor is None:
        raise ValueError("lazy multimodal inputs require a processor, but processor is None")
    if tokenizer is None:
        raise ValueError("lazy multimodal inputs require a tokenizer, but tokenizer is None")
    if dataset_cls is None:
        raise ValueError("lazy multimodal inputs require a dataset class for process_vision_info")

    rebuilt_inputs = []
    input_ids_cpu = input_ids.detach().cpu()
    for idx, raw_prompt in enumerate(raw_prompts):
        messages = _normalize_raw_prompt(raw_prompt)
        multi_modal_data = _process_vision_info_sync(processor, dataset_cls, data_config, messages)
        original_inputs = _normalize_original_inputs(original_multi_modal_inputs, idx)
        _append_crop_images(
            multi_modal_data,
            _get_sample_crop_image_refs(crop_image_refs, idx),
            original_inputs=original_inputs,
        )
        sample_inputs = _compute_multi_modal_inputs_for_sample(
            tokenizer=tokenizer,
            processor=processor,
            input_ids=input_ids_cpu[idx],
            multi_modal_data=multi_modal_data,
        )
        if assert_metadata_match and original_inputs is not None:
            _assert_metadata_matches(original_inputs, sample_inputs, idx)
        rebuilt_inputs.append(sample_inputs)
    return rebuilt_inputs


def _compute_multi_modal_inputs_for_sample(
    *,
    tokenizer: Any,
    processor: Any,
    input_ids: torch.Tensor,
    multi_modal_data: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if processor.__class__.__name__ == "Gemma4Processor":
        from verl.models.gemma4.processing import multimodal_inputs
        return multimodal_inputs(processor, input_ids, multi_modal_data.get("images"))
    images = multi_modal_data.get("images")
    videos = multi_modal_data.get("videos")
    if videos is not None:
        videos, video_metadatas = zip(*videos, strict=False)
        videos, video_metadatas = list(videos), list(video_metadatas)
    else:
        video_metadatas = None

    current_text = tokenizer.decode(input_ids, skip_special_tokens=True)
    multi_modal_inputs = processor(
        text=[current_text],
        images=images,
        videos=videos,
        video_metadata=video_metadatas,
        return_tensors="pt",
        do_sample_frames=False,
    )
    multi_modal_inputs.pop("input_ids", None)
    multi_modal_inputs.pop("attention_mask", None)
    if hasattr(multi_modal_inputs, "convert_to_tensors"):
        multi_modal_inputs = dict(multi_modal_inputs.convert_to_tensors("pt"))
    else:
        multi_modal_inputs = dict(multi_modal_inputs)
    multi_modal_inputs.pop("mm_token_type_ids", None)

    image_grid_thw = multi_modal_inputs.get("image_grid_thw")
    if image_grid_thw is not None:
        images_seqlens = torch.repeat_interleave(image_grid_thw[:, 1] * image_grid_thw[:, 2], image_grid_thw[:, 0])
        multi_modal_inputs["images_seqlens"] = images_seqlens
    return multi_modal_inputs


def _process_vision_info_sync(
    processor: Any, dataset_cls: Any, data_config: Any, messages: list[dict]
) -> dict[str, Any]:
    image_patch_size = processor.image_processor.patch_size
    images, videos = _run_coro_sync(
        dataset_cls.process_vision_info(messages, image_patch_size=image_patch_size, config=data_config)
    )
    multi_modal_data = {}
    if images is not None:
        multi_modal_data["images"] = images
    if videos is not None:
        multi_modal_data["videos"] = videos
    return multi_modal_data


def _append_crop_images(
    multi_modal_data: dict[str, Any],
    crop_image_refs: Any,
    *,
    original_inputs: Mapping[str, Any] | None = None,
) -> None:
    refs = _filter_crop_refs_for_original_metadata(
        crop_image_refs,
        original_inputs=original_inputs,
        base_image_count=len(multi_modal_data.get("images") or []),
    )
    if not refs:
        return
    images = list(multi_modal_data.get("images") or [])
    images.extend(load_crop_images(refs))
    multi_modal_data["images"] = images


def _filter_crop_refs_for_original_metadata(
    crop_image_refs: Any,
    *,
    original_inputs: Mapping[str, Any] | None,
    base_image_count: int,
) -> list[dict[str, Any]]:
    refs = normalize_crop_image_refs(crop_image_refs)
    if not refs or original_inputs is None:
        return refs

    image_grid_thw = original_inputs.get("image_grid_thw")
    if image_grid_thw is None:
        return refs

    original_image_count = int(_as_cpu_tensor(image_grid_thw).shape[0])
    expected_crop_count = max(0, original_image_count - int(base_image_count))
    if len(refs) <= expected_crop_count:
        return refs

    indexed_refs = [
        ref
        for ref in refs
        if isinstance(ref.get("image_index"), Integral)
        and base_image_count <= int(ref["image_index"]) < original_image_count
    ]
    if len(indexed_refs) == expected_crop_count:
        return indexed_refs

    return refs[:expected_crop_count]


def _get_sample_crop_image_refs(crop_image_refs: Sequence[Any] | None, idx: int) -> Any:
    if crop_image_refs is None:
        return None
    if isinstance(crop_image_refs, np.ndarray):
        return crop_image_refs[idx]
    return crop_image_refs[idx]


def _normalize_original_inputs(original_multi_modal_inputs: Sequence[Any] | None, idx: int) -> Mapping[str, Any] | None:
    if original_multi_modal_inputs is None:
        return None
    original_inputs = original_multi_modal_inputs[idx]
    if hasattr(original_inputs, "data"):
        original_inputs = original_inputs.data
    return original_inputs or {}


def _run_coro_sync(coro):
    """Run a coroutine from sync code, even if the current thread already has a running event loop."""
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result()


def _assert_metadata_matches(original_inputs: Any, rebuilt_inputs: dict[str, torch.Tensor], sample_idx: int) -> None:
    original_inputs = _normalize_original_inputs([original_inputs], 0)
    for key in METADATA_COMPARE_KEYS:
        original_value = original_inputs.get(key)
        rebuilt_value = rebuilt_inputs.get(key)
        if original_value is None and rebuilt_value is None:
            continue
        if original_value is None or rebuilt_value is None:
            raise ValueError(
                f"lazy multimodal metadata mismatch for sample {sample_idx}: key {key!r} "
                f"present in only one of rollout/rebuilt inputs"
            )
        if not torch.equal(_as_cpu_tensor(original_value), _as_cpu_tensor(rebuilt_value)):
            original_tensor = _as_cpu_tensor(original_value)
            rebuilt_tensor = _as_cpu_tensor(rebuilt_value)
            raise ValueError(
                f"lazy multimodal metadata mismatch for sample {sample_idx}: key {key!r} differs "
                "between rollout and rebuilt inputs "
                f"(rollout_shape={tuple(original_tensor.shape)}, rebuilt_shape={tuple(rebuilt_tensor.shape)}, "
                f"rollout_preview={_tensor_preview(original_tensor)}, rebuilt_preview={_tensor_preview(rebuilt_tensor)})"
            )


def _tensor_preview(value: torch.Tensor, limit: int = 12) -> list[Any]:
    flat = value.detach().cpu().reshape(-1)
    return flat[:limit].tolist()


def _as_cpu_tensor(value: Any) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    return torch.as_tensor(value)


def _normalize_raw_prompt(raw_prompt: Any) -> list[dict]:
    if isinstance(raw_prompt, np.ndarray):
        raw_prompt = raw_prompt.tolist()
    if isinstance(raw_prompt, tuple):
        raw_prompt = list(raw_prompt)
    return list(raw_prompt)


def _get_lazy_config(config: Any) -> Any:
    if config is None:
        return None
    if _config_get(config, "enabled", None) is not None:
        return config
    lazy_config = _config_get(config, "lazy_multi_modal_inputs", None)
    if lazy_config is not None:
        return lazy_config
    rollout_config = _config_get(config, "rollout", None)
    if rollout_config is not None:
        return _config_get(rollout_config, "lazy_multi_modal_inputs", None)
    actor_rollout_ref = _config_get(config, "actor_rollout_ref", None)
    if actor_rollout_ref is not None:
        rollout_config = _config_get(actor_rollout_ref, "rollout", None)
        if rollout_config is not None:
            return _config_get(rollout_config, "lazy_multi_modal_inputs", None)
    return None


def _config_get(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)
