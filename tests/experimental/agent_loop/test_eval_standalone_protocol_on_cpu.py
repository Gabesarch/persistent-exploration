import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from jinja2.sandbox import ImmutableSandboxedEnvironment

ROOT = Path(__file__).resolve().parents[3]
EVAL_ROOT = ROOT / "vero_eval"
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(EVAL_ROOT))

from lmms_eval.models.chat.async_openai import AsyncOpenAIChat


def _prompt_model(fragment):
    model = object.__new__(AsyncOpenAIChat)
    model.system_prompt = ""
    model._task_system_prompt_cache = {}
    model.task_metadata = {
        "visual_probe_easy_reasoning_zoom": {
            "system_prompt_files": [str(fragment)],
            "system_prompt_preserve_trailing_newline": True,
        }
    }
    return model


def test_run_prompt_replaces_json_task_fragment_verbatim(monkeypatch, tmp_path):
    fragment = tmp_path / "json_prompt.txt"
    fragment.write_text("Use JSON tool calls.\n")
    prompt = EVAL_ROOT / "prompts/tools/system_vero_reasoning_qwen3_xml.txt"
    monkeypatch.setenv("RUN_SYSTEM_PROMPT_MODE", "run_env")
    monkeypatch.setenv("TRAINING_SYSTEM_PROMPT", str(prompt))
    model = _prompt_model(fragment)
    actual = model._resolve_task_system_prompt("visual_probe_easy_reasoning_zoom")
    assert actual.encode("utf-8") == prompt.read_bytes()
    assert "Use JSON tool calls." not in actual
    assert model._resolve_task_system_prompt("visual_probe_easy_reasoning_zoom") == actual


def test_default_prompt_keeps_task_metadata(monkeypatch, tmp_path):
    fragment = tmp_path / "json_prompt.txt"
    fragment.write_text("Use JSON tool calls.\n")
    monkeypatch.delenv("RUN_SYSTEM_PROMPT_MODE", raising=False)
    monkeypatch.setenv("TRAINING_SYSTEM_PROMPT", "/unused/run/prompt")
    assert _prompt_model(fragment)._resolve_task_system_prompt("visual_probe_easy_reasoning_zoom") == fragment.read_text()


def test_xml_template_preserves_tool_history_and_images():
    template = ImmutableSandboxedEnvironment().from_string((EVAL_ROOT / "prompts/vero_qwen3_xml.jinja").read_text())
    arguments = {"bbox_2d": [100, 200, 400, 500], "label": "chart title", "img_idx": 0}
    messages = [
        {"role": "system", "content": "XML system prompt\n"},
        {"role": "user", "content": [{"type": "image"}, {"type": "text", "text": "Question"}]},
        {"role": "assistant", "content": "<think>Inspect.</think>", "tool_calls": [
            {"function": {"name": "image_zoom_in_tool", "arguments": arguments}}
        ]},
        {"role": "tool", "content": [{"type": "text", "text": "Crop"}, {"type": "image"}]},
    ]
    actual = template.render(messages=messages, tools=[{"type": "function", "function": {"name": "image_zoom_in_tool"}}], add_generation_prompt=True)
    assert "<tool_call>\n<function=image_zoom_in_tool>" in actual
    assert "<parameter=bbox_2d>\n[100, 200, 400, 500]\n</parameter>" in actual
    assert '<parameter=img_idx>\n0\n</parameter>' in actual
    assert '"name": "image_zoom_in_tool", "arguments"' not in actual
    assert "put only the json object" not in actual
    assert actual.count("<|vision_start|><|image_pad|><|vision_end|>") == 2
    assert actual.endswith("<|im_start|>assistant\n<think>\n")


def _dry_run(tmp_path, key, **overrides):
    model = tmp_path / "model"
    model.mkdir(exist_ok=True)
    (model / "config.json").write_text("{}")
    (model / "model.safetensors").write_bytes(b"fixture")
    judge = tmp_path / "judge"
    judge.mkdir(exist_ok=True)
    (judge / "chat_template_no_think.jinja").write_text("fixture")
    env = {
        **os.environ, "EVAL_NAME": key, "TRAIN_DIR": str(model),
        "VLLM_JUDGE_MODEL_PATH": str(judge), "DRY_RUN": "1",
        "SKIP_IF_QUEUED": "false", "SKIP_IF_EXISTS": "false", "PORT_OFFSET": "200",
        **overrides,
    }
    for knob in ("RUN_SYSTEM_PROMPT_MODE", "CHAT_TEMPLATE", "TOOL_CALL_PARSER", "ONLY_EVAL_GLOBAL_STEPS", "ONLY_EVAL_EVERY_X", "MAX_STEPS_EVAL", "SYSTEM_PROMPT"):
        if knob not in overrides:
            env.pop(knob, None)
    return subprocess.run(
        ["bash", "examples/submit_zoom_eval_checkpoints_qwen35_27b_judge.sh"],
        cwd=EVAL_ROOT, env=env, capture_output=True, text=True, timeout=30,
    )


@pytest.mark.parametrize("key,step,parser,mode", [
    ("qwen35_vero2_step200", 200, "qwen3_coder", "run_env"),
    ("minio3_vero2_step100", 100, "hermes", "task_metadata"),
])
def test_standalone_dry_run_uses_steps_and_protocol(tmp_path, key, step, parser, mode):
    result = _dry_run(tmp_path, key)
    assert result.returncode == 0, result.stderr
    assert f"[global_step_{step}] checkpoint={tmp_path}/model" in result.stdout
    assert f"zoom_eval_{key}_visual_probe/global_step_{step}" in result.stdout
    assert f"TOOL_CALL_PARSER={parser}" in result.stdout
    assert f"RUN_SYSTEM_PROMPT_MODE={mode}" in result.stdout
    assert "MM_MAX_PIXELS=10000000" in result.stdout
    assert "JUDGE_BACKEND=engine" in result.stdout
    if parser == "qwen3_coder":
        assert f"CHAT_TEMPLATE={EVAL_ROOT}/prompts/vero_qwen3_xml.jinja" in result.stdout


@pytest.mark.parametrize("filters", [
    {"MAX_STEPS_EVAL": "100"}, {"ONLY_EVAL_EVERY_X": "30"}, {"ONLY_EVAL_GLOBAL_STEPS": "100"},
])
def test_standalone_step_filters_still_apply(tmp_path, filters):
    result = _dry_run(tmp_path, "qwen35_vero2_step200", **filters)
    assert result.returncode == 0, result.stderr
    assert "No checkpoints to submit after applying filters" in result.stdout
    assert "sbatch --" not in result.stdout


def test_standalone_incomplete_shard_download_is_skipped(tmp_path):
    _dry_run(tmp_path, "minio3_vero2_step100")
    (tmp_path / "model/model.safetensors.index.json").write_text(json.dumps({
        "weight_map": {"layer": "missing-shard.safetensors"}
    }))
    result = _dry_run(tmp_path, "minio3_vero2_step100")
    assert result.returncode == 0, result.stderr
    assert "until the checkpoint download is ready" in result.stdout
    assert "sbatch --" not in result.stdout
