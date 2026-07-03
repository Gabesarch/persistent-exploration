# Copyright 2024 Bytedance Ltd. and/or its affiliates
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
"""Drawing tools for training VLMs to annotate images.

These tools mirror the MCP tools in vero_eval/lmms_eval/mcp/vero2_mcp.py exactly,
so training and evaluation use identical tool interfaces.

All coordinates use 0-1000 scale:
  - (0, 0) = top-left corner
  - (1000, 1000) = bottom-right corner
  - (500, 500) = center of image

Image referencing:
  - Tools use img_idx (0-based) to reference images in the conversation
  - img_idx=0 is the original input image
  - Tool outputs are appended to agent_data.image_data for chaining
"""

import logging
import os
from typing import Any, List, Optional, Union
from uuid import uuid4

import numpy as np
from PIL import Image, ImageDraw

from verl.utils.crop_cache import save_crop_image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

COORD_SCALE = 1000  # Coordinate system uses 0-1000
_VISUALIZER_CLS = None


def _to_pixel(value: float, size: int) -> float:
    """Convert 0-1000 coordinate to pixel coordinate."""
    return (value / COORD_SCALE) * size


def _get_visualizer_cls():
    global _VISUALIZER_CLS
    if _VISUALIZER_CLS is None:
        from vero_eval.lmms_eval.mcp.visualizer import Visualizer as _V

        _VISUALIZER_CLS = _V
    return _VISUALIZER_CLS


def Visualizer(*args, **kwargs):
    try:
        return _get_visualizer_cls()(*args, **kwargs)
    except ImportError as exc:
        raise RuntimeError(
            "This drawing tool needs the optional 'visualizer' deps (cv2 + matplotlib), "
            "which are not installed. Use the eval environment with those extras available. "
            f"({exc})"
        )


class DrawingToolBase(BaseTool):
    """Base class for drawing tools with shared functionality."""

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self._instance_dict = {}
        self.crop_cache_root = config.get("crop_cache_root")

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        if instance_id is None:
            instance_id = str(uuid4())
        self._instance_dict[instance_id] = {}
        return instance_id, ToolResponse()

    def _get_source_image(self, img_idx: int, agent_data: Any = None) -> Image.Image:
        """Get image from agent_data.image_data by index."""
        if agent_data is None or agent_data.image_data is None:
            raise ValueError("Missing source image: agent_data.image_data is required.")

        image_data = agent_data.image_data
        if not isinstance(image_data, list):
            image_data = [image_data]

        if img_idx < 0 or img_idx >= len(image_data):
            raise IndexError(f"img_idx={img_idx} out of range. Available: 0-{len(image_data)-1}")

        img = image_data[img_idx]
        if isinstance(img, Image.Image):
            return img.copy()
        return Image.fromarray(np.array(img)).copy()

    def _error_response(self, error: str, debug_info: dict) -> tuple[ToolResponse, float, dict]:
        debug_info["success"] = False
        debug_info["error"] = error
        debug_info["terminal"] = True
        debug_info["terminal_reason"] = f"{self.name}_error"
        return (
            ToolResponse(
                text=f"Error in {self.name}: {error}",
                terminate=True,
                terminal_reason=f"{self.name}_error",
            ),
            -0.05,
            debug_info,
        )

    def _save_and_return(
        self, image: Image.Image, response_text: str, debug_info: dict, agent_data: Any = None
    ) -> tuple[ToolResponse, float, dict]:
        crop_ref = save_crop_image(
            image,
            cache_root=getattr(agent_data, "crop_cache_root", None) or self.crop_cache_root,
            request_id=getattr(agent_data, "request_id", None),
            step=getattr(agent_data, "crop_cache_step", None),
            metadata={"tool": self.name, **debug_info.get("parameters", {})},
        )
        debug_info["crop_image_ref"] = crop_ref
        debug_info["success"] = True
        return (
            ToolResponse(image=[image], crop_image_refs=[crop_ref], text=response_text),
            0.0,
            debug_info,
        )

    def _next_img_idx(self, agent_data: Any = None) -> int:
        image_data = getattr(agent_data, "image_data", None)
        if image_data is None:
            return 0
        if not isinstance(image_data, list):
            image_data = [image_data]
        return len(image_data)

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]


class DrawPointsTool(DrawingToolBase):
    """Draw one or more point markers on an image.

    Coordinates use 0-1000 scale where (0,0)=top-left, (1000,1000)=bottom-right.
    For a single point, pass [x, y]. For multiple points, pass [[x1, y1], [x2, y2], ...].
    """

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        img_idx = parameters.get("img_idx", 0)
        points = parameters.get("points")
        color = parameters.get("color", "lime")
        radius = parameters.get("radius", 15)

        debug_info = {"tool": "draw_points", "parameters": parameters}

        if not points:
            return self._error_response("points parameter is required", debug_info)

        try:
            agent_data = kwargs.get("agent_data")
            image = self._get_source_image(img_idx, agent_data)
            w, h = image.size
            debug_info["image_size"] = [w, h]

            # Normalize to list of points
            if points and isinstance(points[0], (int, float)):
                points = [points]  # Single point: [x, y] -> [[x, y]]

            vis = Visualizer(np.array(image.convert("RGB")))
            pr = _to_pixel(radius, w)

            for point in points:
                x, y = point
                px, py = _to_pixel(x, w), _to_pixel(y, h)
                vis.draw_circle((px, py), color=color, radius=pr)

            response_text = f"Registered image as img_idx {self._next_img_idx(agent_data)}."
            image = Image.fromarray(vis.get_output().get_image())
            return self._save_and_return(image, response_text, debug_info, agent_data)

        except Exception as e:
            logger.error(f"Error in draw_points: {e}")
            return self._error_response(str(e), debug_info)


class DrawLinesTool(DrawingToolBase):
    """Draw one or more lines on an image.

    Coordinates use 0-1000 scale. For a single line, pass [x1, y1, x2, y2].
    For multiple lines, pass [[x1, y1, x2, y2], [x1, y1, x2, y2], ...].
    """

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        img_idx = parameters.get("img_idx", 0)
        lines = parameters.get("lines")
        color = parameters.get("color", "red")
        linewidth = parameters.get("linewidth", 3)

        debug_info = {"tool": "draw_lines", "parameters": parameters}

        if not lines:
            return self._error_response("lines parameter is required", debug_info)

        try:
            agent_data = kwargs.get("agent_data")
            image = self._get_source_image(img_idx, agent_data)
            w, h = image.size
            debug_info["image_size"] = [w, h]

            # Normalize to list of lines
            if lines and isinstance(lines[0], (int, float)):
                lines = [lines]  # Single line: [x1, y1, x2, y2] -> [[x1, y1, x2, y2]]

            vis = Visualizer(np.array(image.convert("RGB")))

            for line in lines:
                x1, y1, x2, y2 = line
                px1, py1 = _to_pixel(x1, w), _to_pixel(y1, h)
                px2, py2 = _to_pixel(x2, w), _to_pixel(y2, h)
                vis.draw_line([px1, px2], [py1, py2], color=color, linewidth=linewidth)

            response_text = f"Registered image as img_idx {self._next_img_idx(agent_data)}."
            image = Image.fromarray(vis.get_output().get_image())
            return self._save_and_return(image, response_text, debug_info, agent_data)

        except Exception as e:
            logger.error(f"Error in draw_lines: {e}")
            return self._error_response(str(e), debug_info)


class DrawBoxesTool(DrawingToolBase):
    """Draw one or more bounding boxes on an image.

    Coordinates use 0-1000 scale where (0,0)=top-left, (1000,1000)=bottom-right.
    For a single box, pass [x1, y1, x2, y2]. For multiple boxes, pass [[x1, y1, x2, y2], ...].
    """

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        img_idx = parameters.get("img_idx", 0)
        boxes = parameters.get("boxes")
        color = parameters.get("color", "lime")
        alpha = parameters.get("alpha", 0.8)
        linewidth = parameters.get("linewidth", 3)

        debug_info = {"tool": "draw_boxes", "parameters": parameters}

        if not boxes:
            return self._error_response("boxes parameter is required", debug_info)

        try:
            agent_data = kwargs.get("agent_data")
            image = self._get_source_image(img_idx, agent_data)
            w, h = image.size
            debug_info["image_size"] = [w, h]

            # Normalize to list of boxes
            if boxes and isinstance(boxes[0], (int, float)):
                boxes = [boxes]  # Single box: [x1, y1, x2, y2] -> [[x1, y1, x2, y2]]

            try:
                vis = Visualizer(np.array(image.convert("RGB")))
                for box in boxes:
                    x1, y1, x2, y2 = box
                    px1, py1 = _to_pixel(x1, w), _to_pixel(y1, h)
                    px2, py2 = _to_pixel(x2, w), _to_pixel(y2, h)
                    vis.draw_box([px1, py1, px2, py2], edge_color=color, alpha=alpha, linewidth=linewidth)
                image = Image.fromarray(vis.get_output().get_image())
            except RuntimeError:
                draw = ImageDraw.Draw(image)
                for box in boxes:
                    x1, y1, x2, y2 = box
                    px1, py1 = int(_to_pixel(x1, w)), int(_to_pixel(y1, h))
                    px2, py2 = int(_to_pixel(x2, w)), int(_to_pixel(y2, h))
                    draw.rectangle([px1, py1, px2, py2], outline=color, width=max(1, int(linewidth)))

            response_text = f"Registered image as img_idx {self._next_img_idx(agent_data)}."
            return self._save_and_return(image, response_text, debug_info, agent_data)

        except Exception as e:
            logger.error(f"Error in draw_boxes: {e}")
            return self._error_response(str(e), debug_info)


class DrawTextsTool(DrawingToolBase):
    """Add one or more text annotations to an image.

    Coordinates use 0-1000 scale. For a single text, pass {"x": x, "y": y, "text": "label"}.
    For multiple, pass a list of such dicts.
    """

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        img_idx = parameters.get("img_idx", 0)
        texts = parameters.get("texts")
        color = parameters.get("color", "red")
        font_size = parameters.get("font_size", 18)

        debug_info = {"tool": "draw_texts", "parameters": parameters}

        if not texts:
            return self._error_response("texts parameter is required", debug_info)

        try:
            agent_data = kwargs.get("agent_data")
            image = self._get_source_image(img_idx, agent_data)
            w, h = image.size
            debug_info["image_size"] = [w, h]

            # Normalize to list of text objects
            if isinstance(texts, dict):
                texts = [texts]

            vis = Visualizer(np.array(image.convert("RGB")))

            for text_obj in texts:
                x = text_obj["x"]
                y = text_obj["y"]
                text = text_obj["text"]
                px, py = _to_pixel(x, w), _to_pixel(y, h)
                vis.draw_text(text, (px, py), color=color, font_size=font_size)

            response_text = f"Registered image as img_idx {self._next_img_idx(agent_data)}."
            image = Image.fromarray(vis.get_output().get_image())
            return self._save_and_return(image, response_text, debug_info, agent_data)

        except Exception as e:
            logger.error(f"Error in draw_texts: {e}")
            return self._error_response(str(e), debug_info)


class DrawPolygonsTool(DrawingToolBase):
    """Draw one or more polygons on an image.

    Coordinates use 0-1000 scale. For a single polygon, pass [[x1,y1], [x2,y2], [x3,y3], ...].
    For multiple polygons, pass [[[x1,y1], ...], [[x1,y1], ...], ...].
    """

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        img_idx = parameters.get("img_idx", 0)
        polygons = parameters.get("polygons")
        color = parameters.get("color", "red")
        alpha = parameters.get("alpha", 0.5)

        debug_info = {"tool": "draw_polygons", "parameters": parameters}

        if not polygons:
            return self._error_response("polygons parameter is required", debug_info)

        try:
            agent_data = kwargs.get("agent_data")
            image = self._get_source_image(img_idx, agent_data)
            w, h = image.size
            debug_info["image_size"] = [w, h]

            # Normalize to list of polygons
            # Check if it's a single polygon (list of [x, y] pairs) or multiple polygons
            if polygons and isinstance(polygons[0], list) and len(polygons[0]) == 2 and isinstance(polygons[0][0], (int, float)):
                polygons = [polygons]  # Single polygon

            vis = Visualizer(np.array(image.convert("RGB")))

            for polygon in polygons:
                pixel_points = [[_to_pixel(p[0], w), _to_pixel(p[1], h)] for p in polygon]
                vis.draw_polygon(np.array(pixel_points), color=color, alpha=alpha)

            response_text = f"Registered image as img_idx {self._next_img_idx(agent_data)}."
            image = Image.fromarray(vis.get_output().get_image())
            return self._save_and_return(image, response_text, debug_info, agent_data)

        except Exception as e:
            logger.error(f"Error in draw_polygons: {e}")
            return self._error_response(str(e), debug_info)
