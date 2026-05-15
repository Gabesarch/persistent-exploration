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

from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

try:
    import torch
except Exception:  # pragma: no cover - torch is present in trainer environments
    torch = None

_WRITE_LOCK = threading.Lock()


def _select(config: Any, key: str, default: Any = None) -> Any:
    try:
        if isinstance(config, DictConfig):
            return OmegaConf.select(config, key, default=default)
        current = config
        for part in key.split("."):
            if isinstance(current, Mapping):
                current = current.get(part, default)
            else:
                current = getattr(current, part, default)
            if current is default:
                return default
        return current
    except Exception:
        return default


def is_enabled(config: Any) -> bool:
    return bool(_select(config, "trainer.comprehensive_debug_logging.enabled", False))


def get_debug_dir(config: Any) -> str | None:
    debug_dir = _select(config, "trainer.comprehensive_debug_logging.dir", None)
    if debug_dir:
        return os.path.expanduser(str(debug_dir))
    checkpoint_dir = _select(config, "trainer.default_local_dir", None)
    if checkpoint_dir:
        return os.path.join(os.path.expanduser(str(checkpoint_dir)), "debug_logs")
    return None


def get_debug_options(config: Any) -> dict[str, Any]:
    return {
        "enabled": is_enabled(config),
        "dir": get_debug_dir(config),
        "save_cropped_images": bool(_select(config, "trainer.comprehensive_debug_logging.save_cropped_images", True)),
        "save_source_images": bool(_select(config, "trainer.comprehensive_debug_logging.save_source_images", False)),
        "max_text_chars": _select(config, "trainer.comprehensive_debug_logging.max_text_chars", None),
    }


def ensure_debug_dir(config: Any) -> str | None:
    if not is_enabled(config):
        return None
    debug_dir = get_debug_dir(config)
    if not debug_dir:
        return None
    try:
        os.makedirs(debug_dir, exist_ok=True)
        return debug_dir
    except Exception:
        return None


def _truncate_text(value: str, max_text_chars: int | None) -> str:
    if max_text_chars is None:
        return value
    try:
        max_len = int(max_text_chars)
    except (TypeError, ValueError):
        return value
    if max_len <= 0 or len(value) <= max_len:
        return value
    half = max(1, max_len // 2)
    return value[:half] + "...(truncated)..." + value[-half:]


def to_jsonable(value: Any, max_text_chars: int | None = None) -> Any:
    try:
        if isinstance(value, (DictConfig, ListConfig)):
            value = OmegaConf.to_container(value, resolve=True)
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            return _truncate_text(value, max_text_chars)
        if isinstance(value, os.PathLike):
            return os.fspath(value)
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            if value.dtype == object:
                return [to_jsonable(item, max_text_chars=max_text_chars) for item in value.tolist()]
            return value.tolist()
        if torch is not None and isinstance(value, torch.Tensor):
            detached = value.detach().cpu()
            if detached.numel() <= 128:
                return detached.tolist()
            return {
                "type": "torch.Tensor",
                "shape": list(detached.shape),
                "dtype": str(detached.dtype),
                "numel": int(detached.numel()),
            }
        if isinstance(value, Image.Image):
            return {"type": "PIL.Image.Image", "mode": value.mode, "size": list(value.size)}
        if isinstance(value, Mapping):
            return {str(key): to_jsonable(val, max_text_chars=max_text_chars) for key, val in value.items()}
        if isinstance(value, Sequence) and not isinstance(value, bytes | bytearray):
            return [to_jsonable(item, max_text_chars=max_text_chars) for item in value]
        if isinstance(value, bytes | bytearray):
            return {"type": type(value).__name__, "length": len(value)}
        return _truncate_text(str(value), max_text_chars)
    except Exception:
        return repr(value)


def process_metadata() -> dict[str, Any]:
    env_keys = [
        "USER",
        "PWD",
        "PYTHONPATH",
        "VERL_ROOT",
        "SAVE_PATH",
        "IMAGE_ROOT",
        "HF_HOME",
        "WANDB_MODE",
        "SLURM_JOB_ID",
        "SLURM_ARRAY_JOB_ID",
        "SLURM_ARRAY_TASK_ID",
        "SLURM_JOB_NAME",
        "SLURM_JOB_USER",
        "SLURM_GPUS_ON_NODE",
        "SLURM_CPUS_PER_TASK",
        "CUDA_VISIBLE_DEVICES",
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
    ]
    return {
        "time_unix": time.time(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "argv": list(sys.argv),
        "cwd": os.getcwd(),
        "env": {key: os.environ.get(key) for key in env_keys if os.environ.get(key) is not None},
    }


def append_jsonl(path: str | os.PathLike[str], record: Mapping[str, Any], max_text_chars: int | None = None) -> None:
    try:
        path = os.fspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        payload = to_jsonable(dict(record), max_text_chars=max_text_chars)
        line = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        with _WRITE_LOCK:
            with open(path, "a", encoding="utf-8") as handle:
                handle.write(line + "\n")
    except Exception:
        return


def write_json(path: str | os.PathLike[str], payload: Any, max_text_chars: int | None = None) -> None:
    try:
        path = os.fspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(to_jsonable(payload, max_text_chars=max_text_chars), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except Exception:
        return


def write_config_yaml(path: str | os.PathLike[str], config: Any) -> None:
    try:
        path = os.fspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(OmegaConf.to_yaml(config, resolve=True))
    except Exception:
        return


def save_image(image: Any, path: str | os.PathLike[str]) -> str | None:
    try:
        if not isinstance(image, Image.Image):
            return None
        path = os.fspath(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        image.save(path)
        return path
    except Exception:
        return None


def exception_payload(exc: BaseException) -> dict[str, str]:
    return {
        "type": exc.__class__.__name__,
        "message": str(exc),
        "traceback": traceback.format_exc(),
    }


def path_join(*parts: Any) -> str:
    return os.fspath(Path(os.path.join(*(str(part) for part in parts))))
