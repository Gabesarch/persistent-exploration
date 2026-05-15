import json

import numpy as np
from omegaconf import OmegaConf
from PIL import Image

from verl.utils.debug.comprehensive_logging import (
    append_jsonl,
    ensure_debug_dir,
    get_debug_options,
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
