import asyncio
import json
import sys
import types
from collections import Counter

import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import pytest
from torch.utils.data import SequentialSampler

from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn


def _write_jsonl(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def _text_from_content(content):
    if isinstance(content, str):
        return content
    return "".join(item.get("text", "") for item in content if isinstance(item, dict))


class _DummyTokenizer:
    name_or_path = "dummy-overlong-cache-tokenizer"
    chat_template = "dummy-overlong-cache-template"
    init_kwargs = {"name_or_path": name_or_path}

    def __init__(self, fail_on_tokenize=False):
        self.fail_on_tokenize = fail_on_tokenize
        self.calls = 0

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True, **kwargs):
        self.calls += 1
        if self.fail_on_tokenize:
            raise AssertionError("cache was not used")

        text_parts = []
        for message in messages:
            content = message.get("content", "")
            if isinstance(content, str):
                text_parts.append(content)
            elif isinstance(content, list):
                text_parts.extend(item.get("text", "") for item in content if isinstance(item, dict))
        text = "\n".join(text_parts)
        return list(range(200 if "DROP_ME" in text else 5))


def _overlong_cache_dirs(cache_root):
    cache_dir = cache_root / "overlong_filter_cache"
    if not cache_dir.exists():
        return []
    return sorted(path for path in cache_dir.iterdir() if path.is_dir())


def test_gaze_jsonl_string_image_paths_and_system_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets_cache"))
    image_path = tmp_path / "chart.png"
    Image.new("RGB", (32, 24), color="white").save(image_path)

    system_prompt = tmp_path / "system.txt"
    system_prompt.write_text("Use grounded points before answering.", encoding="utf-8")

    rows = [
        {
            "data_source": "qwen3vl_filtering",
            "prompt": [{"role": "user", "content": "<image>\nHow many bars are shown?"}],
            "images": [str(image_path)],
            "ability": "chart_ocr",
            "reward_model": {"style": "rule", "ground_truth": "6"},
            "extra_info": {"index": 1, "reward_type": "numeric"},
        },
        {
            "data_source": "qwen3vl_filtering",
            "prompt": "<image>\nWhat word is highlighted?",
            "images": [str(image_path)],
            "ability": "chart_ocr",
            "reward_model": {"style": "rule", "ground_truth": "alpha"},
            "extra_info": {"index": 2, "reward_type": "string_match"},
        },
    ]
    data_path = tmp_path / "gaze.jsonl"
    _write_jsonl(data_path, rows)

    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "image_key": "images",
            "max_prompt_length": 1024,
            "filter_overlong_prompts": False,
            "system_prompt": str(system_prompt),
            "max_pixels": 123456,
            "normalize_bbox_to_1000": True,
            "cache_dir": str(tmp_path / "cache"),
        }
    )

    dataset = RLHFDataset(data_files=str(data_path), tokenizer=None, config=config, processor=None)
    first = dataset[0]

    assert first["reward_model"]["ground_truth"] == "6"
    assert first["extra_info"]["images"] == [str(image_path)]
    assert first["extra_info"]["image_sizes"] == [(32, 24)]
    assert first["extra_info"]["normalize_bbox_to_1000"] is True

    messages = first["raw_prompt"]
    assert messages[0]["role"] == "system"
    assert "grounded points" in _text_from_content(messages[0]["content"])

    user_content = messages[1]["content"]
    assert user_content[0] == {"type": "image", "image": str(image_path), "max_pixels": 123456}
    assert user_content[1]["type"] == "text"
    assert "How many bars" in user_content[1]["text"]

    batch = collate_fn([dataset[0], dataset[1]])
    assert batch["raw_prompt"].shape == (2,)
    assert batch["reward_model"].shape == (2,)


def test_image_content_block_applies_max_pixels_to_pil_images():
    dataset = object.__new__(RLHFDataset)
    dataset.max_pixels = 789

    block = dataset._image_to_content_block(Image.new("RGB", (32, 24), color="white"))

    assert block["type"] == "image"
    assert block["max_pixels"] == 789
    assert isinstance(block["image"], Image.Image)


def test_process_vision_info_applies_default_max_pixels(monkeypatch):
    captured = {}

    def fake_process_vision_info(messages, image_patch_size, return_video_metadata):
        captured["messages"] = messages
        captured["image_patch_size"] = image_patch_size
        captured["return_video_metadata"] = return_video_metadata
        return ["processed-image"], None

    fake_qwen_vl_utils = types.SimpleNamespace(process_vision_info=fake_process_vision_info)
    monkeypatch.setitem(sys.modules, "qwen_vl_utils", fake_qwen_vl_utils)

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "first.png"},
                {"type": "image", "image": "second.png", "max_pixels": 111},
                {"type": "text", "text": "What is shown?"},
            ],
        }
    ]
    config = OmegaConf.create({"max_pixels": 222})

    images, videos = asyncio.run(RLHFDataset.process_vision_info(messages, image_patch_size=14, config=config))

    assert images == ["processed-image"]
    assert videos is None
    assert captured["image_patch_size"] == 14
    assert captured["return_video_metadata"] is True
    image_blocks = captured["messages"][0]["content"][:2]
    assert image_blocks[0]["max_pixels"] == 222
    assert image_blocks[1]["max_pixels"] == 111
    assert "max_pixels" not in messages[0]["content"][0]


def test_gaze_jsonl_optional_fixation_post_prompt(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets_cache"))
    data_path = tmp_path / "gaze.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "data_source": "qwen3vl_filtering",
                "prompt": "Answer from the chart.",
                "fixations": [[1.2, 3.8], [5, 7]],
                "reward_model": {"style": "rule", "ground_truth": "ok"},
                "extra_info": {"index": 1, "reward_type": "string_match"},
            }
        ],
    )

    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 1024,
            "filter_overlong_prompts": False,
            "add_fixation_post_prompt": True,
            "cache_dir": str(tmp_path / "cache"),
        }
    )

    dataset = RLHFDataset(data_files=str(data_path), tokenizer=None, config=config, processor=None)
    user_message = dataset[0]["raw_prompt"][0]

    assert "Answer from the chart." in user_message["content"]
    assert "Here are some image regions" in user_message["content"]
    assert "1. [1, 4]" in user_message["content"]
    assert "2. [5, 7]" in user_message["content"]


def test_per_batch_domain_weights_builds_balanced_sampler(tmp_path, monkeypatch):
    from verl.trainer.main_ppo import create_rl_sampler

    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets_cache"))
    data_path = tmp_path / "domains.jsonl"
    rows = []
    for ability in ("chart", "counting", "search"):
        for idx in range(45):
            rows.append(
                {
                    "data_source": "qwen3vl_filtering",
                    "prompt": f"{ability} question {idx}",
                    "ability": ability,
                    "reward_model": {"style": "rule", "ground_truth": "ok"},
                    "extra_info": {"index": len(rows), "reward_type": "string_match"},
                }
            )
    _write_jsonl(data_path, rows)

    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 1024,
            "filter_overlong_prompts": False,
            "per_batch_domain_weights": {"chart": 1, "counting": 1, "search": 1},
            "train_batch_size": 128,
            "shuffle": False,
            "seed": 123,
            "sampler": {"class_path": None, "class_name": None},
            "cache_dir": str(tmp_path / "cache"),
        }
    )

    dataset = RLHFDataset(data_files=str(data_path), tokenizer=None, config=config, processor=None)
    sampler = create_rl_sampler(config, dataset)
    first_batch_indices = list(iter(sampler))[:128]
    counts = Counter(dataset.dataframe[int(idx)]["ability"] for idx in first_batch_indices)

    assert len(first_batch_indices) == 128
    assert sorted(counts.values()) == [42, 43, 43]
    assert set(counts) == {"chart", "counting", "search"}


def test_per_batch_domain_weights_requires_ability_column(tmp_path, monkeypatch):
    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets_cache"))
    data_path = tmp_path / "missing_ability.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "data_source": "qwen3vl_filtering",
                "prompt": "question",
                "reward_model": {"style": "rule", "ground_truth": "ok"},
                "extra_info": {"index": 1, "reward_type": "string_match"},
            }
        ],
    )

    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 1024,
            "filter_overlong_prompts": False,
            "per_batch_domain_weights": {"chart": 1, "counting": 1, "search": 1},
            "cache_dir": str(tmp_path / "cache"),
        }
    )

    with pytest.raises(KeyError, match="ability"):
        RLHFDataset(data_files=str(data_path), tokenizer=None, config=config, processor=None)


def test_default_sampler_unchanged_without_domain_weights(tmp_path, monkeypatch):
    from verl.trainer.main_ppo import create_rl_sampler

    monkeypatch.setenv("HF_DATASETS_CACHE", str(tmp_path / "hf_datasets_cache"))
    data_path = tmp_path / "plain.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "data_source": "qwen3vl_filtering",
                "prompt": f"question {idx}",
                "ability": "chart",
                "reward_model": {"style": "rule", "ground_truth": "ok"},
                "extra_info": {"index": idx, "reward_type": "string_match"},
            }
            for idx in range(4)
        ],
    )
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 1024,
            "filter_overlong_prompts": False,
            "train_batch_size": 4,
            "shuffle": False,
            "sampler": {"class_path": None, "class_name": None},
            "cache_dir": str(tmp_path / "cache"),
        }
    )

    dataset = RLHFDataset(data_files=str(data_path), tokenizer=None, config=config, processor=None)
    sampler = create_rl_sampler(config, dataset)

    assert dataset.domain2indices is None
    assert isinstance(sampler, SequentialSampler)
    assert list(iter(sampler)) == [0, 1, 2, 3]


def test_overlong_filter_cache_reuses_kept_indices(tmp_path, monkeypatch):
    hf_cache = tmp_path / "hf_datasets_cache"
    monkeypatch.setenv("HF_DATASETS_CACHE", str(hf_cache))
    data_path = tmp_path / "filter_cache.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "data_source": "qwen3vl_filtering",
                "prompt": "keep first",
                "reward_model": {"style": "rule", "ground_truth": "ok"},
                "extra_info": {"index": 0},
            },
            {
                "data_source": "qwen3vl_filtering",
                "prompt": "DROP_ME",
                "reward_model": {"style": "rule", "ground_truth": "ok"},
                "extra_info": {"index": 1},
            },
            {
                "data_source": "qwen3vl_filtering",
                "prompt": "keep second",
                "reward_model": {"style": "rule", "ground_truth": "ok"},
                "extra_info": {"index": 2},
            },
        ],
    )
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 50,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": None,
            "cache_dir": str(tmp_path / "cache"),
        }
    )

    first_tokenizer = _DummyTokenizer()
    first = RLHFDataset(data_files=str(data_path), tokenizer=first_tokenizer, config=config, processor=None)
    assert [item["index"] for item in first.dataframe["extra_info"]] == [0, 2]
    assert first_tokenizer.calls == 3

    cache_dirs = _overlong_cache_dirs(hf_cache)
    assert len(cache_dirs) == 1
    assert (cache_dirs[0] / "meta.json").exists()
    assert np.load(cache_dirs[0] / "kept_indices.npy", allow_pickle=False).tolist() == [0, 2]

    second_tokenizer = _DummyTokenizer(fail_on_tokenize=True)
    second = RLHFDataset(data_files=str(data_path), tokenizer=second_tokenizer, config=config, processor=None)
    assert [item["index"] for item in second.dataframe["extra_info"]] == [0, 2]
    assert second_tokenizer.calls == 0


def test_overlong_filter_cache_key_changes_with_prompt_length(tmp_path, monkeypatch):
    hf_cache = tmp_path / "hf_datasets_cache"
    monkeypatch.setenv("HF_DATASETS_CACHE", str(hf_cache))
    data_path = tmp_path / "filter_cache_key.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "data_source": "qwen3vl_filtering",
                "prompt": "keep",
                "reward_model": {"style": "rule", "ground_truth": "ok"},
                "extra_info": {"index": 0},
            }
        ],
    )
    base_config = {
        "prompt_key": "prompt",
        "filter_overlong_prompts": True,
        "filter_overlong_prompts_workers": None,
        "cache_dir": str(tmp_path / "cache"),
    }

    RLHFDataset(
        data_files=str(data_path),
        tokenizer=_DummyTokenizer(),
        config=OmegaConf.create({**base_config, "max_prompt_length": 50}),
        processor=None,
    )
    RLHFDataset(
        data_files=str(data_path),
        tokenizer=_DummyTokenizer(),
        config=OmegaConf.create({**base_config, "max_prompt_length": 51}),
        processor=None,
    )

    assert len(_overlong_cache_dirs(hf_cache)) == 2


def test_invalid_overlong_filter_cache_is_ignored(tmp_path, monkeypatch):
    hf_cache = tmp_path / "hf_datasets_cache"
    monkeypatch.setenv("HF_DATASETS_CACHE", str(hf_cache))
    data_path = tmp_path / "filter_cache_invalid.jsonl"
    _write_jsonl(
        data_path,
        [
            {
                "data_source": "qwen3vl_filtering",
                "prompt": "keep",
                "reward_model": {"style": "rule", "ground_truth": "ok"},
                "extra_info": {"index": 0},
            },
            {
                "data_source": "qwen3vl_filtering",
                "prompt": "DROP_ME",
                "reward_model": {"style": "rule", "ground_truth": "ok"},
                "extra_info": {"index": 1},
            },
        ],
    )
    config = OmegaConf.create(
        {
            "prompt_key": "prompt",
            "max_prompt_length": 50,
            "filter_overlong_prompts": True,
            "filter_overlong_prompts_workers": None,
            "cache_dir": str(tmp_path / "cache"),
        }
    )

    first = RLHFDataset(data_files=str(data_path), tokenizer=_DummyTokenizer(), config=config, processor=None)
    assert [item["index"] for item in first.dataframe["extra_info"]] == [0]

    cache_dirs = _overlong_cache_dirs(hf_cache)
    assert len(cache_dirs) == 1
    (cache_dirs[0] / "meta.json").write_text('{"format_version": 999}', encoding="utf-8")

    second = RLHFDataset(data_files=str(data_path), tokenizer=_DummyTokenizer(), config=config, processor=None)
    assert [item["index"] for item in second.dataframe["extra_info"]] == [0]
