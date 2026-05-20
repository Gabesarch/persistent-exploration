import json

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from verl.utils.debug.comprehensive_logging import (
    append_jsonl,
    ensure_debug_dir,
    get_debug_options,
    get_gpu_memory_logging_options,
    gpu_memory_stage,
    log_gpu_memory_event,
    save_image,
    to_jsonable,
    write_config_yaml,
    write_json,
)


def test_comprehensive_debug_logging_serializes_common_non_json_types(tmp_path):
    image = Image.new("RGB", (8, 6), color="white")
    payload = {
        "array": np.array([1, 2, 3]),
        "scalar": np.float32(0.5),
        "image": image,
        "text": "abcdef",
    }

    serialized = to_jsonable(payload, max_text_chars=4)

    assert serialized["array"] == [1, 2, 3]
    assert serialized["scalar"] == np.float32(0.5).item()
    assert serialized["image"] == {"type": "PIL.Image.Image", "mode": "RGB", "size": [8, 6]}
    assert serialized["text"] == "ab...(truncated)...ef"


def test_comprehensive_debug_logging_writes_jsonl_json_yaml_and_images(tmp_path):
    config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path / "ckpt"),
                "comprehensive_debug_logging": {
                    "enabled": True,
                    "dir": str(tmp_path / "ckpt" / "debug_logs"),
                    "save_cropped_images": True,
                    "save_source_images": False,
                    "max_text_chars": None,
                },
            }
        }
    )

    debug_dir = ensure_debug_dir(config)
    options = get_debug_options(config)

    assert debug_dir == str(tmp_path / "ckpt" / "debug_logs")
    assert options["enabled"] is True

    jsonl_path = tmp_path / "ckpt" / "debug_logs" / "events.jsonl"
    append_jsonl(jsonl_path, {"event": "ok", "value": np.int64(3)})
    assert json.loads(jsonl_path.read_text()) == {"event": "ok", "value": 3}

    json_path = tmp_path / "ckpt" / "debug_logs" / "metadata.json"
    write_json(json_path, {"value": np.array([4])})
    assert json.loads(json_path.read_text()) == {"value": [4]}

    yaml_path = tmp_path / "ckpt" / "debug_logs" / "config.yaml"
    write_config_yaml(yaml_path, config)
    assert "comprehensive_debug_logging" in yaml_path.read_text()

    image_path = tmp_path / "ckpt" / "debug_logs" / "image.png"
    assert save_image(Image.new("RGB", (4, 4), color="black"), image_path) == str(image_path)
    assert image_path.exists()


def test_gpu_memory_logging_options_and_jsonl_event_are_cpu_safe(tmp_path):
    config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path / "ckpt"),
                "comprehensive_debug_logging": {
                    "enabled": True,
                    "dir": str(tmp_path / "ckpt" / "debug_logs"),
                    "gpu_memory_logging": {
                        "enabled": True,
                        "sync_cuda": False,
                        "reset_peak_stats": True,
                        "log_driver_stages": True,
                        "log_worker_stages": False,
                        "log_nvidia_smi": False,
                    },
                },
            }
        }
    )

    options = get_gpu_memory_logging_options(config)

    assert options["enabled"] is True
    assert options["log_worker_stages"] is False

    log_gpu_memory_event(config, "unit_stage", "before", global_step=7, context={"batch_size": 2})
    log_path = tmp_path / "ckpt" / "debug_logs" / "gpu_memory_events.jsonl"
    record = json.loads(log_path.read_text())
    assert record["event"] == "gpu_memory"
    assert record["stage"] == "unit_stage"
    assert record["phase"] == "before"
    assert record["global_step"] == 7
    assert record["context"]["batch_size"] == 2
    assert "torch_cuda" in record
    assert "cpu_memory" in record
    assert "nvidia_smi" not in record


def test_gpu_memory_stage_logs_exception_and_reraises(tmp_path):
    config = OmegaConf.create(
        {
            "trainer": {
                "default_local_dir": str(tmp_path / "ckpt"),
                "comprehensive_debug_logging": {
                    "enabled": True,
                    "dir": str(tmp_path / "ckpt" / "debug_logs"),
                    "gpu_memory_logging": {
                        "enabled": True,
                        "log_nvidia_smi": False,
                    },
                },
            }
        }
    )

    try:
        with gpu_memory_stage(config, "failing_stage", global_step=3):
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    else:
        raise AssertionError("gpu_memory_stage should re-raise exceptions")

    log_path = tmp_path / "ckpt" / "debug_logs" / "gpu_memory_events.jsonl"
    records = [json.loads(line) for line in log_path.read_text().splitlines()]
    assert [record["phase"] for record in records] == ["before", "exception"]
    assert records[-1]["stage"] == "failing_stage"
    assert records[-1]["exception"]["type"] == "RuntimeError"
    assert records[-1]["exception"]["message"] == "boom"
