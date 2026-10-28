import json
import sys
from pathlib import Path

import pytest
from PIL import Image

# The repository root is not installed as an lmms_eval package in CPU-only tests.
# ruff: noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "vero_eval"))

from lmms_eval import utils as lmms_utils
from lmms_eval.tasks import TaskManager
from lmms_eval.tasks.pinpoint_bench import utils as pinpoint_utils
from lmms_eval.tasks.visual_probe import utils as visual_probe_utils

ROOT_TASKS = [
    "pinpoint_bench",
    "pinpoint_bench_gpt5nano_zs",
    "pinpoint_bench_mimo_zs",
    "pinpoint_bench_qwen25_zs",
    "pinpoint_bench_qwen3_draw",
    "pinpoint_bench_qwen3_thinking_zs",
    "pinpoint_bench_qwen3_zoom",
    "pinpoint_bench_qwen3_zoom_strict",
    "pinpoint_bench_qwen3_zs",
    "pinpoint_bench_reasoning",
    "pinpoint_bench_reasoning_draw",
    "pinpoint_bench_reasoning_samplingq3",
    "pinpoint_bench_reasoning_zoom",
    "pinpoint_bench_reasoning_zoom_greedy",
    "pinpoint_bench_reasoning_zoom_multiidx",
    "pinpoint_bench_reasoning_zoom_strict",
]
NESTED_TASKS = [
    "pinpoint_bench_drawing_instruct",
    "pinpoint_bench_zoomdrawing_instruct",
    "pinpoint_bench_drawing_thinking",
    "pinpoint_bench_zoomdrawing_thinking",
    "pinpoint_bench_drawing_vero",
    "pinpoint_bench_zoomdrawing_vero",
]
ALL_TASKS = ROOT_TASKS + NESTED_TASKS

CONFIG_PAIRS = [
    ("pinpoint_bench.yaml", "visual_probe_hard.yaml"),
    *[
        (f"{task}.yaml", f"visual_probe_hard{task.removeprefix('pinpoint_bench')}.yaml")
        for task in ROOT_TASKS[1:]
    ],
    ("_instruct/pinpoint_bench_drawing_instruct.yaml", "_instruct/visual_probe_hard_drawing_instruct.yaml"),
    ("_instruct/pinpoint_bench_zoomdrawing_instruct.yaml", "_instruct/visual_probe_hard_zoomdrawing_instruct.yaml"),
    ("_thinking/pinpoint_bench_drawing_thinking.yaml", "_thinking/visual_probe_hard_drawing_thinking.yaml"),
    ("_thinking/pinpoint_bench_zoomdrawing_thinking.yaml", "_thinking/visual_probe_hard_zoomdrawing_thinking.yaml"),
    ("_vero/pinpoint_bench_drawing_vero.yaml", "_vero/visual_probe_hard_drawing_vero.yaml"),
    ("_vero/pinpoint_bench_zoomdrawing_vero.yaml", "_vero/visual_probe_hard_zoomdrawing_vero.yaml"),
]


def _row(doc_id="0"):
    return {
        "images": ["images/pinpoint_bench_0.jpg"],
        "doc_id": doc_id,
        "problem": "<image>\nWhat color is the flag?",
        "solution": "red and white",
        "data_source": "pinpoint-final",
    }


def test_pinpoint_builder_creates_validation_split(monkeypatch, tmp_path):
    metadata_path = tmp_path / "val.json"
    metadata_path.write_text(json.dumps([_row("0"), _row("1")]), encoding="utf-8")
    monkeypatch.setattr(pinpoint_utils, "hf_hub_download", lambda **_: str(metadata_path))

    dataset = pinpoint_utils.pinpoint_bench_builder()

    assert list(dataset) == ["validation"]
    assert len(dataset["validation"]) == 2
    assert dataset["validation"][0]["solution"] == "red and white"


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        ([], "non-empty JSON list"),
        ([{"doc_id": "0"}], "missing required fields"),
        ([_row("0"), _row("0")], "duplicate doc_id"),
    ],
)
def test_pinpoint_metadata_validation(rows, message, tmp_path):
    metadata_path = tmp_path / "val.json"
    metadata_path.write_text(json.dumps(rows), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        pinpoint_utils._load_metadata_rows(metadata_path)


def test_pinpoint_image_adapter_accepts_pil_local_and_hub_paths(monkeypatch, tmp_path):
    image_path = tmp_path / "sample.png"
    Image.new("L", (3, 2), color=128).save(image_path)

    local = pinpoint_utils.pinpoint_bench_doc_to_visual({"images": [str(image_path)]})
    in_memory = pinpoint_utils.pinpoint_bench_doc_to_visual({"images": [Image.new("L", (2, 2))]})

    calls = []

    def _download(**kwargs):
        calls.append(kwargs)
        return str(image_path)

    monkeypatch.setattr(pinpoint_utils, "hf_hub_download", _download)
    remote = pinpoint_utils.pinpoint_bench_doc_to_visual({"images": ["images/sample.png"]})

    assert [local[0].mode, in_memory[0].mode, remote[0].mode] == ["RGB", "RGB", "RGB"]
    assert calls == [
        {
            "repo_id": "godx7/Pinpoint-Bench",
            "filename": "images/sample.png",
            "repo_type": "dataset",
        }
    ]


def test_pinpoint_reuses_visual_probe_prompt_and_judging_contract():
    assert pinpoint_utils.pinpoint_bench_doc_to_text is visual_probe_utils.visual_probe_doc_to_text
    assert pinpoint_utils.pinpoint_bench_doc_to_target is visual_probe_utils.visual_probe_doc_to_target
    assert pinpoint_utils.pinpoint_bench_process_results is visual_probe_utils.visual_probe_process_results
    assert (
        pinpoint_utils.pinpoint_bench_aggregate_judge_results
        is visual_probe_utils.visual_probe_aggregate_judge_results
    )
    assert (
        pinpoint_utils.pinpoint_bench_aggregate_judge_results_strict
        is visual_probe_utils.visual_probe_aggregate_judge_results_strict
    )

    doc = _row("sample-id")
    assert pinpoint_utils.pinpoint_bench_doc_to_text(doc, {"pre_prompt": "\n", "post_prompt": ""}) == (
        "\nWhat color is the flag?"
    )
    assert pinpoint_utils.pinpoint_bench_doc_to_target(doc) == "red and white"

    payload = pinpoint_utils.pinpoint_bench_process_results(doc, ["<answer>red and white</answer>"])[
        "judge_accuracy"
    ]
    assert payload["question"] == "What color is the flag?"
    assert payload["reference"] == "red and white"
    assert payload["prediction"] == "red and white"
    assert payload["resp_key"] == "sample-id"


@pytest.mark.parametrize(("pinpoint_relative", "visual_relative"), CONFIG_PAIRS)
def test_pinpoint_yaml_matches_visual_probe_hard(pinpoint_relative, visual_relative):
    task_root = REPO_ROOT / "vero_eval/lmms_eval/tasks"
    pinpoint = lmms_utils.load_yaml_config(
        task_root / "pinpoint_bench" / pinpoint_relative,
        mode="simple",
    )
    visual_probe = lmms_utils.load_yaml_config(
        task_root / "visual_probe" / visual_relative,
        mode="simple",
    )

    expected_task = Path(pinpoint_relative).stem
    assert pinpoint["task"] == expected_task
    assert pinpoint["dataset_path"] == "godx7/Pinpoint-Bench"
    assert pinpoint["test_split"] == "validation"
    assert pinpoint["tag"] == "visual_search"
    assert pinpoint["output_type"] == visual_probe["output_type"] == "generate_until"
    assert pinpoint["generation_kwargs"] == visual_probe["generation_kwargs"]
    assert pinpoint["lmms_eval_specific_kwargs"] == visual_probe["lmms_eval_specific_kwargs"]
    assert pinpoint["metadata"] == visual_probe["metadata"]
    assert [metric["metric"] for metric in pinpoint["metric_list"]] == ["judge_accuracy"]
    assert [metric["higher_is_better"] for metric in pinpoint["metric_list"]] == [True]


def test_task_manager_discovers_all_pinpoint_tasks():
    task_manager = TaskManager()

    assert len(ALL_TASKS) == len(set(ALL_TASKS)) == 22
    assert set(ALL_TASKS).issubset(task_manager.all_subtasks)
