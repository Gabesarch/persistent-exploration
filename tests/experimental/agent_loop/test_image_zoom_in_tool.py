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
from types import SimpleNamespace

import pytest
from PIL import Image, ImageChops

import verl.tools.image_zoom_in_tool as image_zoom_in_tool
from verl.experimental.agent_loop.tool_agent_loop import AgentData, AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.tools.image_zoom_in_tool import ImageZoomInTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


def _tool_schema() -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": "image_zoom_in_tool",
                "description": (
                    "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an "
                    "optional object label"
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items": {"type": "number"},
                            "minItems": 4,
                            "maxItems": 4,
                            "description": (
                                "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is "
                                "the top-left corner and (x2, y2) is the bottom-right corner"
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": "The name or label of the object in the specified bounding box",
                        },
                        "img_idx": {
                            "type": "number",
                            "description": "The index of the zoomed-in image (starting from 0)",
                        },
                    },
                    "required": ["bbox_2d", "label", "img_idx"],
                },
            },
        }
    )


@pytest.fixture
def zoom_tool(monkeypatch):
    return _make_zoom_tool(monkeypatch)


def _make_zoom_tool(monkeypatch, **config_overrides):
    def fail_if_pool_is_created(*args, **kwargs):
        raise AssertionError("ImageZoomInTool should not create a Ray execution pool")

    monkeypatch.setattr(image_zoom_in_tool, "init_visual_execution_pool", fail_if_pool_is_created)
    config = {
        "type": "native",
        "coordinate_system": "normalized_1000",
        "enable_global_rate_limit": False,
    }
    config.update(config_overrides)
    tool = ImageZoomInTool(
        config=config,
        tool_schema=_tool_schema(),
    )
    assert tool.execution_pool is None
    return tool


def _assert_terminal_zoom_error(response, reward, metrics, expected_error: str):
    assert reward == -0.05
    assert metrics["success"] is False
    assert metrics["terminal"] is True
    assert metrics["terminal_reason"] == "image_zoom_in_tool_error"
    assert metrics["error"] == expected_error
    assert response.image is None
    assert response.terminate is True
    assert response.terminal_reason == "image_zoom_in_tool_error"
    assert response.text == f"Error processing image zoom-in: {expected_error}"


def test_image_zoom_in_tool_selects_image_by_img_idx(zoom_tool):
    async def run_test():
        instance_id, _ = await zoom_tool.create()
        return await zoom_tool.execute(
            instance_id,
            {"bbox_2d": [250, 250, 750, 750], "label": "target", "img_idx": 1},
            agent_data=agent_data,
        )

    small_image = Image.new("RGB", (200, 100), color="white")
    large_image = Image.new("RGB", (2000, 1000), color="white")
    agent_data = SimpleNamespace(image_data=[small_image, large_image])

    response, reward, metrics = asyncio.run(run_test())

    assert reward == 0.0
    assert metrics["success"] is True
    assert metrics["source_image_size"] == [2000, 1000]
    assert metrics["pixel_bbox"] == [500.0, 250.0, 1500.0, 750.0]
    assert metrics["resized_bbox"] == [500.0, 250.0, 1500.0, 750.0]
    assert metrics["cropped_image_size"] == [992, 512]
    assert response.image[0].size == (992, 512)
    assert response.text == "Zoomed in on the image to the region [250, 250, 750, 750] with label target."


def test_image_zoom_in_tool_scales_normalized_bbox_and_applies_qwen_smart_resize(zoom_tool):
    async def run_test():
        instance_id, _ = await zoom_tool.create()
        return await zoom_tool.execute(
            instance_id,
            {"bbox_2d": [250, 250, 750, 750], "label": "center", "img_idx": 0},
            agent_data=agent_data,
        )

    image = Image.new("RGB", (200, 100), color="white")
    agent_data = SimpleNamespace(image_data=[image])

    response, reward, metrics = asyncio.run(run_test())

    assert reward == 0.0
    assert metrics["success"] is True
    assert response.image[0].size == (736, 384)
    assert response.image[0].size[0] % 32 == 0
    assert response.image[0].size[1] % 32 == 0
    assert response.image[0].size[0] * response.image[0].size[1] >= 256 * 32 * 32


def test_image_zoom_in_tool_saves_crop_png_ref(monkeypatch, tmp_path):
    async def run_test():
        instance_id, _ = await tool.create()
        return await tool.execute(
            instance_id,
            {"bbox_2d": [250, 250, 750, 750], "label": "center", "img_idx": 0},
            agent_data=agent_data,
        )

    image = Image.new("RGB", (200, 100), color="white")
    agent_data = SimpleNamespace(image_data=[image], request_id="request-1", crop_cache_step=3)
    tool = _make_zoom_tool(monkeypatch, crop_cache_root=str(tmp_path))

    response, reward, metrics = asyncio.run(run_test())

    assert reward == 0.0
    crop_ref = response.crop_image_refs[0]
    assert crop_ref == metrics["crop_image_ref"]
    assert crop_ref["path"].endswith(".png")
    assert crop_ref["request_id"] == "request-1"
    assert crop_ref["step"] == 3
    with Image.open(crop_ref["path"]) as saved_image:
        assert saved_image.format == "PNG"
        saved_image = saved_image.convert("RGB")
        assert saved_image.size == response.image[0].size
        assert ImageChops.difference(saved_image, response.image[0]).getbbox() is None


def test_image_zoom_in_tool_uses_original_resolution_source_when_enabled(monkeypatch, tmp_path):
    async def run_test():
        instance_id, _ = await tool.create()
        return await tool.execute(
            instance_id,
            {"bbox_2d": [250, 250, 750, 750], "label": "center", "img_idx": 0},
            agent_data=agent_data,
        )

    original_path = tmp_path / "original.png"
    Image.new("RGB", (2000, 1000), color="white").save(original_path)
    processed_image = Image.new("RGB", (200, 100), color="white")
    agent_data = SimpleNamespace(
        image_data=[processed_image],
        original_prompt_image_sources=[{"type": "image", "image": str(original_path), "max_pixels": 1000}],
    )
    tool = _make_zoom_tool(monkeypatch, use_original_resolution_source=True)

    response, reward, metrics = asyncio.run(run_test())

    assert reward == 0.0
    assert metrics["success"] is True
    assert metrics["used_original_resolution_source"] is True
    assert metrics["source_image_size"] == [2000, 1000]
    assert metrics["original_source_size"] == [2000, 1000]
    assert metrics["pixel_bbox"] == [500.0, 250.0, 1500.0, 750.0]
    assert response.image[0].size == (992, 512)


def test_image_zoom_in_tool_default_config_uses_processed_source_even_when_original_exists(monkeypatch, tmp_path):
    async def run_test():
        instance_id, _ = await tool.create()
        return await tool.execute(
            instance_id,
            {"bbox_2d": [250, 250, 750, 750], "label": "center", "img_idx": 0},
            agent_data=agent_data,
        )

    original_path = tmp_path / "original.png"
    Image.new("RGB", (2000, 1000), color="white").save(original_path)
    processed_image = Image.new("RGB", (200, 100), color="white")
    agent_data = SimpleNamespace(
        image_data=[processed_image],
        original_prompt_image_sources=[{"type": "image", "image": str(original_path)}],
    )
    tool = _make_zoom_tool(monkeypatch)

    response, reward, metrics = asyncio.run(run_test())

    assert reward == 0.0
    assert metrics["success"] is True
    assert metrics["used_original_resolution_source"] is False
    assert metrics["source_image_size"] == [200, 100]
    assert response.image[0].size == (736, 384)


def test_image_zoom_in_tool_falls_back_to_processed_source_when_original_unavailable(monkeypatch):
    async def run_test():
        instance_id, _ = await tool.create()
        return await tool.execute(
            instance_id,
            {"bbox_2d": [250, 250, 750, 750], "label": "center", "img_idx": 0},
            agent_data=agent_data,
        )

    processed_image = Image.new("RGB", (200, 100), color="white")
    agent_data = SimpleNamespace(
        image_data=[processed_image],
        original_prompt_image_sources=[{"type": "image", "image": "/missing/original.png"}],
    )
    tool = _make_zoom_tool(
        monkeypatch,
        use_original_resolution_source=True,
        original_source_unavailable_policy="fallback",
    )

    response, reward, metrics = asyncio.run(run_test())

    assert reward == 0.0
    assert metrics["success"] is True
    assert metrics["used_original_resolution_source"] is False
    assert metrics["original_source_fallback"] is True
    assert "original_source_error" in metrics
    assert metrics["source_image_size"] == [200, 100]
    assert metrics["processed_source_size"] == [200, 100]
    assert response.image[0].size == (736, 384)


def test_image_zoom_in_tool_errors_when_original_unavailable_and_policy_is_error(monkeypatch):
    async def run_test():
        instance_id, _ = await tool.create()
        return await tool.execute(
            instance_id,
            {"bbox_2d": [250, 250, 750, 750], "label": "center", "img_idx": 0},
            agent_data=agent_data,
        )

    agent_data = SimpleNamespace(
        image_data=[Image.new("RGB", (200, 100), color="white")],
        original_prompt_image_sources=[{"type": "image", "image": "/missing/original.png"}],
    )
    tool = _make_zoom_tool(
        monkeypatch,
        use_original_resolution_source=True,
        original_source_unavailable_policy="error",
    )

    response, reward, metrics = asyncio.run(run_test())

    assert reward == -0.05
    assert metrics["success"] is False
    assert metrics["terminal"] is True
    assert metrics["used_original_resolution_source"] is False
    assert metrics["error"].startswith("Original image source unavailable:")
    assert response.image is None
    assert response.terminate is True
    assert response.terminal_reason == "image_zoom_in_tool_error"
    assert response.text.startswith("Error processing image zoom-in: Original image source unavailable:")


def test_image_zoom_in_tool_uses_current_image_data_for_later_crop_indices(monkeypatch, tmp_path):
    async def run_test():
        instance_id, _ = await tool.create()
        return await tool.execute(
            instance_id,
            {"bbox_2d": [250, 250, 750, 750], "label": "later", "img_idx": 1},
            agent_data=agent_data,
        )

    original_path = tmp_path / "original.png"
    Image.new("RGB", (2000, 1000), color="white").save(original_path)
    processed_image = Image.new("RGB", (200, 100), color="white")
    prior_crop = Image.new("RGB", (400, 300), color="white")
    agent_data = SimpleNamespace(
        image_data=[processed_image, prior_crop],
        original_prompt_image_sources=[{"type": "image", "image": str(original_path)}],
    )
    tool = _make_zoom_tool(monkeypatch, use_original_resolution_source=True)

    response, reward, metrics = asyncio.run(run_test())

    assert reward == 0.0
    assert metrics["success"] is True
    assert metrics["used_original_resolution_source"] is False
    assert metrics["source_image_size"] == [400, 300]
    assert response.image[0].size == (608, 448)


def test_tool_agent_loop_collects_original_prompt_image_sources_before_processing():
    image = Image.new("RGB", (64, 32), color="white")
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": "/tmp/first.png", "max_pixels": 123},
                {"type": "text", "text": "question"},
                {"type": "image", "image": image, "min_pixels": 456},
                {"type": "video", "video": "/tmp/video.mp4"},
            ],
        },
        {"role": "assistant", "content": "ok"},
    ]

    sources = ToolAgentLoop._collect_original_prompt_image_sources(messages)

    assert sources == [
        {"type": "image", "image": "/tmp/first.png", "max_pixels": 123},
        {"type": "image", "image": image, "min_pixels": 456},
    ]
    assert sources[0] is not messages[0]["content"][0]


def test_image_zoom_in_tool_expands_small_bbox_with_qwen_bbox_logic(zoom_tool):
    resized_bbox = zoom_tool._maybe_resize_bbox([100, 100, 101, 101], image_width=1000, image_height=1000)

    assert resized_bbox == [84, 84, 117, 117]


def test_image_zoom_in_tool_uses_create_kwargs_image_when_agent_data_missing(zoom_tool):
    async def run_test():
        instance_id, _ = await zoom_tool.create(create_kwargs={"image": create_image})
        return await zoom_tool.execute(
            instance_id,
            {"bbox_2d": [250, 250, 750, 750], "label": "fallback", "img_idx": 0},
        )

    create_image = Image.new("RGB", (2000, 1000), color="white")

    response, reward, metrics = asyncio.run(run_test())

    assert reward == 0.0
    assert metrics["success"] is True
    assert response.image[0].size == (992, 512)


def test_image_zoom_in_tool_invalid_normalized_bbox_returns_error(zoom_tool):
    async def run_test():
        instance_id, _ = await zoom_tool.create()
        return await zoom_tool.execute(
            instance_id,
            {"bbox_2d": [750, 750, 250, 250], "label": "bad", "img_idx": 0},
            agent_data=agent_data,
        )

    image = Image.new("RGB", (200, 100), color="white")
    agent_data = SimpleNamespace(image_data=[image])

    response, reward, metrics = asyncio.run(run_test())

    _assert_terminal_zoom_error(
        response,
        reward,
        metrics,
        "bbox_2d must define a positive-area box with left < right and top < bottom.",
    )


def test_image_zoom_in_tool_non_finite_bbox_returns_error(zoom_tool):
    async def run_test():
        instance_id, _ = await zoom_tool.create()
        return await zoom_tool.execute(
            instance_id,
            {"bbox_2d": [250, float("nan"), 750, 750], "label": "bad", "img_idx": 0},
            agent_data=agent_data,
        )

    image = Image.new("RGB", (200, 100), color="white")
    agent_data = SimpleNamespace(image_data=[image])

    response, reward, metrics = asyncio.run(run_test())

    _assert_terminal_zoom_error(response, reward, metrics, "bbox_2d must contain 4 finite numeric values.")


def test_image_zoom_in_tool_out_of_bounds_bbox_returns_error_after_clamping(zoom_tool):
    async def run_test():
        instance_id, _ = await zoom_tool.create()
        return await zoom_tool.execute(
            instance_id,
            {"bbox_2d": [1100, 100, 1200, 200], "label": "bad", "img_idx": 0},
            agent_data=agent_data,
        )

    image = Image.new("RGB", (200, 100), color="white")
    agent_data = SimpleNamespace(image_data=[image])

    response, reward, metrics = asyncio.run(run_test())

    _assert_terminal_zoom_error(
        response,
        reward,
        metrics,
        "bbox_2d is outside image bounds or has no positive area after clamping.",
    )


def test_image_zoom_in_tool_missing_img_idx_returns_error(zoom_tool):
    async def run_test():
        instance_id, _ = await zoom_tool.create()
        return await zoom_tool.execute(
            instance_id,
            {"bbox_2d": [250, 250, 750, 750], "label": "target"},
            agent_data=agent_data,
        )

    image = Image.new("RGB", (200, 100), color="white")
    agent_data = SimpleNamespace(image_data=[image])

    response, reward, metrics = asyncio.run(run_test())

    _assert_terminal_zoom_error(response, reward, metrics, "img_idx parameter is missing.")


def test_image_zoom_in_tool_out_of_range_img_idx_returns_error(zoom_tool):
    async def run_test():
        instance_id, _ = await zoom_tool.create()
        return await zoom_tool.execute(
            instance_id,
            {"bbox_2d": [250, 250, 750, 750], "label": "target", "img_idx": 1},
            agent_data=agent_data,
        )

    image = Image.new("RGB", (200, 100), color="white")
    agent_data = SimpleNamespace(image_data=[image])

    response, reward, metrics = asyncio.run(run_test())

    _assert_terminal_zoom_error(response, reward, metrics, "img_idx=1 is out of range for 1 images.")


def test_image_zoom_in_tool_missing_label_returns_terminal_error(zoom_tool):
    async def run_test():
        instance_id, _ = await zoom_tool.create()
        return await zoom_tool.execute(
            instance_id,
            {"bbox_2d": [250, 250, 750, 750], "img_idx": 0},
            agent_data=agent_data,
        )

    image = Image.new("RGB", (200, 100), color="white")
    agent_data = SimpleNamespace(image_data=[image])

    response, reward, metrics = asyncio.run(run_test())

    _assert_terminal_zoom_error(response, reward, metrics, "label parameter is missing.")


def test_image_zoom_in_tool_extreme_aspect_ratio_returns_terminal_error(monkeypatch):
    async def run_test():
        instance_id, _ = await tool.create()
        return await tool.execute(
            instance_id,
            {"bbox_2d": [0, 0, 7000, 20], "label": "line", "img_idx": 0},
            agent_data=agent_data,
        )

    image = Image.new("RGB", (7000, 1000), color="white")
    agent_data = SimpleNamespace(image_data=[image])
    tool = _make_zoom_tool(monkeypatch, coordinate_system="pixel")

    response, reward, metrics = asyncio.run(run_test())

    assert reward == -0.05
    assert metrics["success"] is False
    assert metrics["terminal"] is True
    assert metrics["max_cropped_aspect_ratio"] == 199.0
    assert metrics["cropped_aspect_ratio"] > 199.0
    assert metrics["error"].startswith("cropped image aspect ratio ")
    assert metrics["error"].endswith(" exceeds max_cropped_aspect_ratio=199.0")
    assert response.image is None
    assert response.terminate is True
    assert response.terminal_reason == "image_zoom_in_tool_error"


def test_tool_agent_loop_terminal_tool_response_stops_trajectory():
    agent_data = AgentData(
        messages=[],
        image_data=None,
        video_data=None,
        metrics={},
        request_id="request-1",
        tools_kwargs={},
    )
    agent_data.prompt_ids = [101, 102]
    agent_data.response_mask = [1, 1]
    agent_data.tool_calls = [FunctionCall(name="image_zoom_in_tool", arguments='{"bbox_2d": []}')]

    class DummyLoop:
        max_parallel_calls = 1
        tool_parser_name = "hermes"

        @staticmethod
        def _debug_enabled_for_agent(agent_data):
            return False

        async def _call_tool(self, tool_call, tools_kwargs, agent_data, tool_index=0):
            return (
                ToolResponse(
                    text="Error processing image zoom-in: bbox_2d parameter is missing or not a list of 4 numbers.",
                    terminate=True,
                    terminal_reason="image_zoom_in_tool_error",
                ),
                -0.05,
                {"success": False, "terminal": True},
            )

        async def apply_chat_template(self, messages, images=None, videos=None, remove_system_prompt=False):
            assert messages == [
                {
                    "role": "tool",
                    "content": (
                        "Error processing image zoom-in: "
                        "bbox_2d parameter is missing or not a list of 4 numbers."
                    ),
                }
            ]
            assert images is None
            assert videos is None
            assert remove_system_prompt is True
            return [201, 202, 203]

    state = asyncio.run(ToolAgentLoop._handle_processing_tools_state(DummyLoop(), agent_data))

    assert state == AgentState.TERMINATED
    assert agent_data.messages == [
        {
            "role": "tool",
            "content": "Error processing image zoom-in: bbox_2d parameter is missing or not a list of 4 numbers.",
        }
    ]
    assert agent_data.prompt_ids == [101, 102, 201, 202, 203]
    assert agent_data.response_mask == [1, 1, 0, 0, 0]
    assert agent_data.tool_rewards == [-0.05]
    assert agent_data.user_turns == 1
    assert agent_data.extra_fields["terminated_by_tool_error"] is True
    assert agent_data.extra_fields["terminal_tool_name"] == "image_zoom_in_tool"
    assert agent_data.extra_fields["terminal_reason"] == "image_zoom_in_tool_error"
