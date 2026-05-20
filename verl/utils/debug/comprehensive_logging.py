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
import subprocess
import sys
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, ListConfig, OmegaConf
from PIL import Image

try:
    import torch
except Exception:  # pragma: no cover - torch is present in trainer environments
    torch = None

try:
    import psutil
except Exception:  # pragma: no cover - psutil is present in trainer environments
    psutil = None

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


def _debug_select(config: Any, suffix: str, default: Any = None) -> Any:
    full_key = f"trainer.comprehensive_debug_logging.{suffix}" if suffix else "trainer.comprehensive_debug_logging"
    value = _select(config, full_key, default)
    if value is not default:
        return value
    worker_key = f"comprehensive_debug_logging.{suffix}" if suffix else "comprehensive_debug_logging"
    return _select(config, worker_key, default)


def is_enabled(config: Any) -> bool:
    return bool(_debug_select(config, "enabled", False))


def get_debug_dir(config: Any) -> str | None:
    debug_dir = _debug_select(config, "dir", None)
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
        "save_cropped_images": bool(_debug_select(config, "save_cropped_images", True)),
        "save_source_images": bool(_debug_select(config, "save_source_images", False)),
        "max_text_chars": _debug_select(config, "max_text_chars", None),
        "gpu_memory_logging": get_gpu_memory_logging_options(config),
    }


def get_gpu_memory_logging_options(config: Any) -> dict[str, Any]:
    return {
        "enabled": bool(is_enabled(config) and _debug_select(config, "gpu_memory_logging.enabled", False)),
        "sync_cuda": bool(_debug_select(config, "gpu_memory_logging.sync_cuda", False)),
        "reset_peak_stats": bool(_debug_select(config, "gpu_memory_logging.reset_peak_stats", True)),
        "log_driver_stages": bool(_debug_select(config, "gpu_memory_logging.log_driver_stages", True)),
        "log_worker_stages": bool(_debug_select(config, "gpu_memory_logging.log_worker_stages", True)),
        "log_nvidia_smi": bool(_debug_select(config, "gpu_memory_logging.log_nvidia_smi", True)),
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
        if isinstance(value, DictConfig | ListConfig):
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


def _env_metadata() -> dict[str, Any]:
    return {
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "world_size": os.environ.get("WORLD_SIZE"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
    }


def _cpu_memory_snapshot() -> dict[str, Any]:
    try:
        if psutil is None:
            return {}
        proc = psutil.Process(os.getpid())
        return {
            "rss_gb": proc.memory_info().rss / (1024**3),
            "system_used_gb": psutil.virtual_memory().used / (1024**3),
        }
    except Exception as exc:
        return {"error": f"{exc.__class__.__name__}: {exc}"}


def _torch_gpu_snapshot(sync_cuda: bool = False) -> dict[str, Any]:
    if torch is None:
        return {"available": False, "error": "torch unavailable"}
    try:
        if not torch.cuda.is_available():
            return {"available": False}
        if sync_cuda:
            torch.cuda.synchronize()
        device_id = torch.cuda.current_device()
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
        return {
            "available": True,
            "device": device_id,
            "device_name": torch.cuda.get_device_name(device_id),
            "allocated_gb": torch.cuda.memory_allocated(device_id) / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved(device_id) / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated(device_id) / (1024**3),
            "max_reserved_gb": torch.cuda.max_memory_reserved(device_id) / (1024**3),
            "mem_get_info": {
                "free_gb": free_bytes / (1024**3),
                "used_gb": (total_bytes - free_bytes) / (1024**3),
                "total_gb": total_bytes / (1024**3),
            },
        }
    except Exception as exc:
        return {"available": False, "error": f"{exc.__class__.__name__}: {exc}"}


def _parse_nvidia_smi_csv(text: str, keys: list[str]) -> list[dict[str, Any]]:
    rows = []
    for line in text.splitlines():
        if not line.strip():
            continue
        values = [part.strip() for part in line.split(",")]
        rows.append({key: values[index] if index < len(values) else None for index, key in enumerate(keys)})
    return rows


def _nvidia_smi_snapshot() -> dict[str, Any]:
    try:
        gpu_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,memory.used,memory.free,memory.total,utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        proc_result = subprocess.run(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            check=False,
            capture_output=True,
            text=True,
            timeout=2,
        )
        snapshot = {
            "gpus": _parse_nvidia_smi_csv(
                gpu_result.stdout,
                ["index", "uuid", "name", "memory_used_mb", "memory_free_mb", "memory_total_mb", "utilization_gpu_pct"],
            ),
            "gpu_query_returncode": gpu_result.returncode,
        }
        if gpu_result.stderr.strip():
            snapshot["gpu_query_stderr"] = gpu_result.stderr.strip()
        snapshot["processes"] = _parse_nvidia_smi_csv(
            proc_result.stdout, ["gpu_uuid", "pid", "process_name", "used_memory_mb"]
        )
        snapshot["process_query_returncode"] = proc_result.returncode
        if proc_result.stderr.strip():
            snapshot["process_query_stderr"] = proc_result.stderr.strip()
        return snapshot
    except Exception as exc:
        return {"error": f"{exc.__class__.__name__}: {exc}"}


def summarize_numeric_sequence(values: Any) -> dict[str, Any]:
    try:
        if torch is not None and isinstance(values, torch.Tensor):
            values = values.detach().cpu().reshape(-1).tolist()
        elif isinstance(values, np.ndarray):
            values = values.reshape(-1).tolist()
        elif not isinstance(values, Sequence) or isinstance(values, str | bytes | bytearray):
            return {"value": to_jsonable(values)}
        cleaned = [float(value) for value in values if value is not None]
        if not cleaned:
            return {"count": 0}
        return {
            "count": len(cleaned),
            "min": min(cleaned),
            "max": max(cleaned),
            "mean": sum(cleaned) / len(cleaned),
            "sum": sum(cleaned),
        }
    except Exception:
        return {"value": to_jsonable(values)}


def gpu_memory_context_from_batch(batch: Any) -> dict[str, Any]:
    context: dict[str, Any] = {}
    try:
        context["batch_size"] = len(batch)
    except Exception:
        pass
    try:
        td = getattr(batch, "batch", None)
        if td is not None:
            for key in ["prompts", "responses", "input_ids", "attention_mask", "response_mask"]:
                if key in td:
                    value = td[key]
                    context[f"{key}_shape"] = list(value.shape)
                    if (
                        key in {"attention_mask", "response_mask"}
                        and torch is not None
                        and isinstance(value, torch.Tensor)
                    ):
                        context[f"{key}_tokens"] = summarize_numeric_sequence(value.sum(dim=-1))
    except Exception:
        pass
    try:
        meta = getattr(batch, "meta_info", None) or {}
        for key in ["global_steps", "temperature", "validate", "global_token_num", "images_seqlens"]:
            if key in meta:
                if key in {"global_token_num", "images_seqlens"}:
                    context[key] = summarize_numeric_sequence(meta[key])
                else:
                    context[key] = to_jsonable(meta[key])
    except Exception:
        pass
    return context


def log_gpu_memory_event(
    config: Any,
    stage: str,
    phase: str,
    *,
    global_step: int | None = None,
    epoch: int | None = None,
    role: str | None = None,
    context: Mapping[str, Any] | None = None,
    exception: BaseException | None = None,
    worker: bool = False,
) -> None:
    try:
        options = get_gpu_memory_logging_options(config)
        if not options["enabled"]:
            return
        if worker and not options["log_worker_stages"]:
            return
        if not worker and not options["log_driver_stages"]:
            return
        debug_dir = ensure_debug_dir(config)
        if not debug_dir:
            return

        record: dict[str, Any] = {
            "event": "gpu_memory",
            "stage": stage,
            "phase": phase,
            "global_step": global_step,
            "epoch": epoch,
            "role": role,
            "time_unix": time.time(),
            **_env_metadata(),
            "cpu_memory": _cpu_memory_snapshot(),
            "torch_cuda": _torch_gpu_snapshot(sync_cuda=options["sync_cuda"]),
        }
        if context:
            record["context"] = context
        if exception is not None:
            record["exception"] = exception_payload(exception)
        if options["log_nvidia_smi"]:
            record["nvidia_smi"] = _nvidia_smi_snapshot()

        append_jsonl(os.path.join(debug_dir, "gpu_memory_events.jsonl"), record)

        if phase == "after" and options["reset_peak_stats"] and torch is not None:
            try:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
            except Exception:
                pass
    except Exception:
        return


@contextmanager
def gpu_memory_stage(
    config: Any,
    stage: str,
    *,
    global_step: int | None = None,
    epoch: int | None = None,
    role: str | None = None,
    context: Mapping[str, Any] | None = None,
    worker: bool = False,
):
    log_gpu_memory_event(
        config,
        stage,
        "before",
        global_step=global_step,
        epoch=epoch,
        role=role,
        context=context,
        worker=worker,
    )
    try:
        yield
    except Exception as exc:
        log_gpu_memory_event(
            config,
            stage,
            "exception",
            global_step=global_step,
            epoch=epoch,
            role=role,
            context=context,
            exception=exc,
            worker=worker,
        )
        raise
    else:
        log_gpu_memory_event(
            config,
            stage,
            "after",
            global_step=global_step,
            epoch=epoch,
            role=role,
            context=context,
            worker=worker,
        )


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
