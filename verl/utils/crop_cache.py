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

import os
import re
import shutil
from pathlib import Path
from typing import Any
from uuid import uuid4

import numpy as np
from PIL import Image

CROP_IMAGE_REFS_KEY = "crop_image_refs"


def resolve_crop_cache_root(config_root: str | os.PathLike | None = None) -> Path:
    """Resolve the root used for tool-generated crop PNGs."""
    root = config_root or os.getenv("VERO2_CROP_CACHE_ROOT")
    if root:
        return Path(os.path.expandvars(os.path.expanduser(str(root))))

    user = os.getenv("USER") or "unknown"
    run_id = os.getenv("SLURM_JOB_ID") or f"pid_{os.getpid()}"
    return Path("/tmp") / user / "vero2_crop_cache" / run_id


def crop_cache_cleanup_enabled(config: Any) -> bool:
    lazy_config = _get_lazy_config(config)
    return bool(_config_get(lazy_config, "cleanup_crop_cache", True))


def save_crop_image(
    image: Image.Image,
    *,
    cache_root: str | os.PathLike | None = None,
    request_id: str | None = None,
    step: int | str | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    root = resolve_crop_cache_root(cache_root)
    step_name = _safe_component(f"step_{step}" if step is not None else "step_unknown")
    request_name = _safe_component(f"request_{request_id}" if request_id else f"request_{uuid4().hex}")
    cache_dir = root / step_name / request_name
    cache_dir.mkdir(parents=True, exist_ok=True)

    path = cache_dir / f"crop_{uuid4().hex}.png"
    image.save(path, format="PNG")

    crop_ref: dict[str, Any] = {
        "path": str(path),
        "format": "png",
        "width": image.width,
        "height": image.height,
    }
    if request_id is not None:
        crop_ref["request_id"] = request_id
    if step is not None:
        crop_ref["step"] = step
    if metadata:
        crop_ref.update(metadata)
    return crop_ref


def load_crop_images(crop_image_refs: Any) -> list[Image.Image]:
    images = []
    for crop_ref in normalize_crop_image_refs(crop_image_refs):
        path = crop_ref.get("path")
        if not path:
            raise ValueError(f"crop image ref is missing 'path': {crop_ref!r}")
        with Image.open(path) as image:
            images.append(image.convert("RGB").copy())
    return images


def normalize_crop_image_refs(crop_image_refs: Any) -> list[dict[str, Any]]:
    if crop_image_refs is None:
        return []
    if hasattr(crop_image_refs, "data"):
        crop_image_refs = crop_image_refs.data
    if isinstance(crop_image_refs, np.ndarray):
        crop_image_refs = crop_image_refs.tolist()
    if isinstance(crop_image_refs, dict):
        crop_image_refs = [crop_image_refs]
    elif isinstance(crop_image_refs, tuple):
        crop_image_refs = list(crop_image_refs)
    elif not isinstance(crop_image_refs, list):
        return []

    refs = [dict(ref) for ref in crop_image_refs if isinstance(ref, dict) and ref.get("path")]
    return sorted(refs, key=lambda ref: ref.get("image_index", ref.get("order", 0)))


def crop_cache_paths_from_batch(batch: Any) -> list[str]:
    non_tensor_batch = getattr(batch, "non_tensor_batch", None)
    if not non_tensor_batch or CROP_IMAGE_REFS_KEY not in non_tensor_batch:
        return []
    return crop_cache_paths_from_refs(non_tensor_batch[CROP_IMAGE_REFS_KEY])


def crop_cache_paths_from_refs(crop_image_refs: Any) -> list[str]:
    paths = []
    if crop_image_refs is None:
        return paths
    if isinstance(crop_image_refs, np.ndarray):
        crop_image_refs = crop_image_refs.tolist()
    if isinstance(crop_image_refs, dict):
        crop_image_refs = [crop_image_refs]
    if isinstance(crop_image_refs, list | tuple):
        for item in crop_image_refs:
            if isinstance(item, dict):
                path = item.get("path")
                if path:
                    paths.append(str(path))
            elif isinstance(item, list | tuple | np.ndarray):
                paths.extend(crop_cache_paths_from_refs(item))
    return paths


def cleanup_crop_cache_for_batch(batch: Any) -> None:
    cleanup_crop_cache_paths(crop_cache_paths_from_batch(batch))


def cleanup_crop_cache_paths(paths: list[str]) -> None:
    for path_str in set(paths):
        path = Path(path_str)
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            continue
        _remove_empty_parents(path.parent)


def cleanup_crop_cache_root(root: str | os.PathLike | None = None) -> None:
    root_path = resolve_crop_cache_root(root)
    shutil.rmtree(root_path, ignore_errors=True)


def _remove_empty_parents(path: Path) -> None:
    for parent in [path, *path.parents[:2]]:
        try:
            parent.rmdir()
        except OSError:
            break


def _safe_component(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.=-]+", "_", value)[:128] or "unknown"


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
