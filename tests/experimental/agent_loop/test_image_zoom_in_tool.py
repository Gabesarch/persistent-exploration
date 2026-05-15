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
from PIL import Image

import verl.tools.image_zoom_in_tool as image_zoom_in_tool
from verl.tools.image_zoom_in_tool import ImageZoomInTool
from verl.tools.schemas import OpenAIFunctionToolSchema


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
    monkeypatch.setattr(image_zoom_in_tool, "init_visual_execution_pool", lambda *args, **kwargs: None)
    return ImageZoomInTool(
        config={
            "type": "native",
            "coordinate_system": "normalized_1000",
            "enable_global_rate_limit": False,
        },
        tool_schema=_tool_schema(),
    )


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

    assert reward < 0
    assert metrics["success"] is False
    assert response.image is None
    assert response.text.startswith("Error processing image zoom-in:")


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

    assert reward < 0
    assert metrics["success"] is False
    assert response.image is None
    assert response.text == "Error processing image zoom-in: img_idx parameter is missing."


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

    assert reward < 0
    assert metrics["success"] is False
    assert response.image is None
    assert response.text == "Error processing image zoom-in: img_idx=1 is out of range for 1 images."
