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

"""Training-side drawing tool.

Annotate an image (points / lines / boxes / texts / polygons) and hand the
annotated image back to the model, so it can *look at its own annotation* before
answering. This is the RL-training twin of the eval MCP drawing tools in
``vero_eval/lmms_eval/mcp/vero2_mcp.py`` (``draw_points`` / ``draw_lines`` /
``draw_boxes`` / ``draw_texts`` / ``draw_polygons``): identical 0-1000 normalized
coordinate system, identical PIL rendering, and the same "the annotated image
becomes a new addressable image" semantics.

Design mirrors ``verl/tools/image_zoom_in_tool.py`` on purpose:

* one class backs all five drawing operations; the operation is taken from the
  tool-schema name (``draw_points`` -> ``points`` etc.), so the YAML registers
  the same class five times with five schemas (see
  ``vero2_scripts/config/tool_config/image_draw_tool.yaml``);
* the source image is read from ``agent_data.image_data[img_idx]`` -- exactly the
  images the model currently sees in its context -- so ``img_idx`` addresses the
  original prompt image (0) or any earlier tool-produced image, just like zoom;
* ``execute`` returns ``ToolResponse(image=[annotated], crop_image_refs=[ref])``.
  The agent loop appends the annotated image to ``agent_data.image_data`` and the
  ``crop_image_refs`` entry (a PNG saved to the crop cache) is what lets the
  lazy-multimodal path rebuild the annotated image during the actor forward pass.
  This is why the drawing run works with ``lazy_multi_modal_inputs.enabled: true``
  unchanged -- the saved PNG *is* the rendered annotation, reloaded verbatim.
"""

import logging
import math
import os
from typing import Any, Optional
from uuid import uuid4

from PIL import Image, ImageColor, ImageDraw, ImageFont
from qwen_vl_utils import fetch_image

from verl.utils.crop_cache import save_crop_image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

COORD_SCALE = 1000.0  # coordinates are on a 0-1000 scale (0,0)=top-left, (1000,1000)=bottom-right

# Map a tool-schema name to the drawing operation it performs. Keeps a single
# class backing every drawing tool; the YAML picks the op by naming the schema.
_NAME_TO_OP = {
    "draw_points": "points",
    "draw_lines": "lines",
    "draw_boxes": "boxes",
    "draw_texts": "texts",
    "draw_polygons": "polygons",
}

_FONT_CANDIDATES = (
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans.ttf",
)


# --------------------------------------------------------------------------- #
# Parameter parsing helpers (ported verbatim in behavior from vero2_mcp.py so
# training and eval accept exactly the same tool arguments).
# --------------------------------------------------------------------------- #
def _parse_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number, got {value!r}.")
    return value


def _parse_positive_number(value: Any, name: str, default: float) -> float:
    if value is None:
        return default
    value = _parse_number(value, name)
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}.")
    return value


def _parse_alpha(value: Any, default: float) -> float:
    if value is None:
        return default
    return max(0.0, min(1.0, _parse_number(value, "alpha")))


def _parse_color(value: Any, default: str) -> tuple[int, int, int]:
    if not isinstance(value, str) or not value:
        value = default
    try:
        return ImageColor.getrgb(value)
    except ValueError as exc:
        raise ValueError(f"Unsupported color {value!r}.") from exc


def _to_pixel_clamped(value: Any, size: int, name: str) -> float:
    coord = max(0.0, min(COORD_SCALE, _parse_number(value, name)))
    return coord / COORD_SCALE * float(size)


def _is_numeric_vector(value: Any, length: int) -> bool:
    if not isinstance(value, (list, tuple)) or len(value) != length:
        return False
    try:
        for idx, item in enumerate(value):
            _parse_number(item, f"value[{idx}]")
    except ValueError:
        return False
    return True


def _normalize_vectors(value: Any, name: str, length: int) -> list[list[float]]:
    """Accept one length-N vector or a list of them, always return a list of vectors."""
    if value is None:
        raise ValueError(f"{name} parameter is required.")
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError(f"{name} must be a non-empty list.")
    if _is_numeric_vector(value, length):
        return [list(value)]
    if all(_is_numeric_vector(item, length) for item in value):
        return [list(item) for item in value]
    raise ValueError(f"{name} must be either one length-{length} numeric list or a list of those lists.")


def _normalize_texts(value: Any) -> list[dict]:
    if value is None:
        raise ValueError("texts parameter is required.")
    if isinstance(value, dict):
        value = [value]
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError("texts must be a non-empty object or list of objects.")
    for idx, item in enumerate(value):
        if not isinstance(item, dict) or not all(key in item for key in ("x", "y", "text")):
            raise ValueError(f"texts[{idx}] must be an object with x, y, and text.")
    return list(value)


def _normalize_polygons(value: Any) -> list[list[list[float]]]:
    if value is None:
        raise ValueError("polygons parameter is required.")
    if not isinstance(value, (list, tuple)) or len(value) == 0:
        raise ValueError("polygons must be a non-empty list.")
    if all(_is_numeric_vector(point, 2) for point in value):
        polygons = [list(value)]
    elif all(
        isinstance(poly, (list, tuple)) and all(_is_numeric_vector(point, 2) for point in poly) for poly in value
    ):
        polygons = [list(poly) for poly in value]
    else:
        raise ValueError("polygons must be one polygon [[x,y], ...] or a list of polygons.")
    for idx, polygon in enumerate(polygons):
        if len(polygon) < 3:
            raise ValueError(f"polygons[{idx}] must contain at least 3 points.")
    return polygons


def _point_to_pixels(point: Any, width: int, height: int, prefix: str) -> tuple[float, float]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError(f"{prefix} must be [x, y].")
    return (
        _to_pixel_clamped(point[0], width, f"{prefix}[0]"),
        _to_pixel_clamped(point[1], height, f"{prefix}[1]"),
    )


def _box_to_pixels(box: Any, width: int, height: int, prefix: str) -> tuple[float, float, float, float]:
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        raise ValueError(f"{prefix} must be [x1, y1, x2, y2].")
    return (
        _to_pixel_clamped(box[0], width, f"{prefix}[0]"),
        _to_pixel_clamped(box[1], height, f"{prefix}[1]"),
        _to_pixel_clamped(box[2], width, f"{prefix}[2]"),
        _to_pixel_clamped(box[3], height, f"{prefix}[3]"),
    )


def _load_font(font_size: int):
    for font_path in _FONT_CANDIDATES:
        if os.path.exists(font_path):
            return ImageFont.truetype(font_path, font_size)
    return ImageFont.load_default()


class ImageDrawTool(BaseTool):
    """Annotate an image and return the annotated version as a new image.

    A single class backs the five drawing operations (points / lines / boxes /
    texts / polygons); the operation is resolved from the tool-schema name (or an
    explicit ``draw_op`` config key). Coordinates use the 0-1000 normalized scale.
    """

    # Cap the returned image so an annotated full-resolution photo does not blow
    # up the model context. 1 MP matches the zoom tool's output budget
    # (cropped_max_pixels) and the eval DRAW_MAX_PIXELS.
    DEFAULT_MAX_PIXELS = 1048576

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        Example schema (one of five; see the YAML for the full set):
            {
                "type": "function",
                "function": {
                    "name": "draw_boxes",
                    "description": "Draw one or more bounding boxes on an image ...",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "img_idx": {"type": "number", ...},
                            "boxes": {"type": "array", ...},
                            "color": {"type": "string", ...},
                            "alpha": {"type": "number", ...},
                            "linewidth": {"type": "number", ...},
                        },
                        "required": ["img_idx", "boxes"],
                    },
                },
            }
        """
        super().__init__(config, tool_schema)
        self._instance_dict: dict[str, dict] = {}

        self.coordinate_system = config.get("coordinate_system", "normalized_1000")
        if self.coordinate_system not in ("normalized_1000", "pixel", "pixels"):
            raise ValueError(
                "coordinate_system must be 'normalized_1000' or 'pixel', "
                f"got {self.coordinate_system!r}."
            )
        self.draw_max_pixels = int(
            config.get("draw_max_pixels", config.get("cropped_max_pixels", self.DEFAULT_MAX_PIXELS))
        )
        self.crop_cache_root = config.get("crop_cache_root")

        # Resolve the drawing operation: explicit config wins, else derive from
        # the schema name (draw_points -> points, ...).
        draw_op = config.get("draw_op") or _NAME_TO_OP.get(self.name)
        if draw_op not in ("points", "lines", "boxes", "texts", "polygons"):
            raise ValueError(
                f"Could not resolve a drawing operation for tool {self.name!r}. Set config.draw_op to one of "
                "points/lines/boxes/texts/polygons, or name the tool draw_points/draw_lines/draw_boxes/"
                "draw_texts/draw_polygons."
            )
        self.draw_op = draw_op
        logger.info(f"Initialized ImageDrawTool(name={self.name}, op={self.draw_op}) with config: {config}")

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """Create a per-call instance.

        Mirrors ImageZoomInTool.create: an optional ``image`` (via ``create_kwargs``)
        is stored as a fallback source, but in training ``execute`` reads the source
        image from ``agent_data.image_data`` (the images the model currently sees).
        """
        if instance_id is None:
            instance_id = str(uuid4())
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)
        image = kwargs.get("image")
        img = fetch_image({"image": image}) if image is not None else None
        self._instance_dict[instance_id] = {"image": img}
        return instance_id, ToolResponse()

    # ------------------------------------------------------------------ #
    # source-image resolution (mirrors ImageZoomInTool)
    # ------------------------------------------------------------------ #
    def _parse_img_idx(self, parameters: dict[str, Any]) -> int:
        if "img_idx" not in parameters:
            raise ValueError("img_idx parameter is missing.")
        img_idx = parameters["img_idx"]
        if isinstance(img_idx, bool) or not isinstance(img_idx, (int, float)) or int(img_idx) != img_idx:
            raise ValueError(f"img_idx must be an integer image index, got {img_idx!r}.")
        img_idx = int(img_idx)
        if img_idx < 0:
            raise ValueError(f"img_idx must be non-negative, got {img_idx}.")
        return img_idx

    def _get_source_image(self, instance_id: str, img_idx: int, agent_data: Any = None) -> Image.Image:
        if agent_data is not None and getattr(agent_data, "image_data", None) is not None:
            image_data = agent_data.image_data
            if not isinstance(image_data, list):
                image_data = [image_data]
            if img_idx >= len(image_data):
                raise IndexError(f"img_idx={img_idx} is out of range for {len(image_data)} images.")
            return image_data[img_idx]

        instance_data = self._instance_dict.get(instance_id, {})
        image = instance_data.get("image")
        if image is not None:
            if img_idx != 0:
                raise IndexError(f"img_idx={img_idx} is out of range for 1 fallback image.")
            return image

        raise ValueError("Missing source image: provide agent_data.image_data or create_kwargs.image.")

    def _bound_image(self, image: Image.Image) -> Image.Image:
        """Downscale to <= draw_max_pixels so the returned annotation stays affordable."""
        image = image.convert("RGB")
        pixels = image.width * image.height
        if pixels > self.draw_max_pixels:
            scale = (self.draw_max_pixels / pixels) ** 0.5
            image = image.resize(
                (max(1, round(image.width * scale)), max(1, round(image.height * scale))),
                Image.Resampling.LANCZOS,
            )
        return image

    # ------------------------------------------------------------------ #
    # rendering (ported from vero2_mcp.py draw_* functions)
    # ------------------------------------------------------------------ #
    def _render(self, image: Image.Image, parameters: dict[str, Any]) -> Image.Image:
        image = image.convert("RGB")
        renderer = getattr(self, f"_draw_{self.draw_op}")
        return renderer(image, parameters)

    def _draw_points(self, image: Image.Image, parameters: dict[str, Any]) -> Image.Image:
        w, h = image.size
        parsed_points = _normalize_vectors(parameters.get("points"), "points", 2)
        rgb = _parse_color(parameters.get("color"), "lime")
        pr = max(1, int(round(_parse_positive_number(parameters.get("radius"), "radius", 15.0) / COORD_SCALE * w)))
        draw = ImageDraw.Draw(image)
        for i, point in enumerate(parsed_points):
            px, py = _point_to_pixels(point, w, h, f"points[{i}]")
            draw.ellipse([px - pr, py - pr, px + pr, py + pr], fill=rgb, outline=rgb)
        return image

    def _draw_lines(self, image: Image.Image, parameters: dict[str, Any]) -> Image.Image:
        w, h = image.size
        parsed_lines = _normalize_vectors(parameters.get("lines"), "lines", 4)
        rgb = _parse_color(parameters.get("color"), "red")
        line_width = max(1, int(round(_parse_positive_number(parameters.get("linewidth"), "linewidth", 3.0))))
        draw = ImageDraw.Draw(image)
        for i, line in enumerate(parsed_lines):
            px1, py1, px2, py2 = _box_to_pixels(line, w, h, f"lines[{i}]")
            draw.line([px1, py1, px2, py2], fill=rgb, width=line_width)
        return image

    def _draw_boxes(self, image: Image.Image, parameters: dict[str, Any]) -> Image.Image:
        w, h = image.size
        parsed_boxes = _normalize_vectors(parameters.get("boxes"), "boxes", 4)
        rgb = _parse_color(parameters.get("color"), "lime")
        alpha_value = _parse_alpha(parameters.get("alpha"), 0.8)
        rgba = (*rgb, int(round(alpha_value * 255)))
        line_width = max(1, int(round(_parse_positive_number(parameters.get("linewidth"), "linewidth", 3.0))))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for i, box in enumerate(parsed_boxes):
            px1, py1, px2, py2 = _box_to_pixels(box, w, h, f"boxes[{i}]")
            left, right = sorted((px1, px2))
            top, bottom = sorted((py1, py2))
            if right > left and bottom > top:
                draw.rectangle([left, top, right, bottom], outline=rgba, width=line_width)
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    def _draw_texts(self, image: Image.Image, parameters: dict[str, Any]) -> Image.Image:
        w, h = image.size
        parsed_texts = _normalize_texts(parameters.get("texts"))
        rgb = _parse_color(parameters.get("color"), "red")
        size = max(1, int(round(_parse_positive_number(parameters.get("font_size"), "font_size", 18.0))))
        font = _load_font(size)
        draw = ImageDraw.Draw(image)
        for i, text_obj in enumerate(parsed_texts):
            px = _to_pixel_clamped(text_obj["x"], w, f"texts[{i}].x")
            py = _to_pixel_clamped(text_obj["y"], h, f"texts[{i}].y")
            draw.text((px, py), str(text_obj["text"]), fill=rgb, font=font)
        return image

    def _draw_polygons(self, image: Image.Image, parameters: dict[str, Any]) -> Image.Image:
        w, h = image.size
        parsed_polygons = _normalize_polygons(parameters.get("polygons"))
        rgb = _parse_color(parameters.get("color"), "red")
        alpha_value = _parse_alpha(parameters.get("alpha"), 0.5)
        rgba = (*rgb, int(round(alpha_value * 255)))
        overlay = Image.new("RGBA", image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        for poly_idx, polygon in enumerate(parsed_polygons):
            points = [
                _point_to_pixels(point, w, h, f"polygons[{poly_idx}][{point_idx}]")
                for point_idx, point in enumerate(polygon)
            ]
            draw.polygon(points, fill=rgba, outline=(*rgb, 255))
        return Image.alpha_composite(image.convert("RGBA"), overlay).convert("RGB")

    def _error_response(self, error: str, debug_info: dict[str, Any]) -> tuple[ToolResponse, float, dict]:
        """Non-terminal error: let the model correct its arguments on a later turn.

        Unlike zoom (which terminates the trajectory on a bad bbox), a malformed
        drawing call just returns a helpful message and keeps the trajectory alive.
        """
        debug_info["success"] = False
        debug_info["error"] = error
        return (
            ToolResponse(text=f"Error processing drawing tool ({self.name}): {error}"),
            0.0,
            debug_info,
        )

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        debug_info: dict[str, Any] = {
            "tool": self.name,
            "draw_op": self.draw_op,
            "parameters": parameters,
            "coordinate_system": self.coordinate_system,
        }
        try:
            img_idx = self._parse_img_idx(parameters)
            debug_info["img_idx"] = img_idx
            agent_data = kwargs.get("agent_data")
            source_image = self._get_source_image(instance_id, img_idx, agent_data)
            debug_info["source_image_size"] = [source_image.width, source_image.height]

            # Render on a COPY so the source image kept in agent_data.image_data is
            # never mutated (a later img_idx reference must still see the original).
            annotated = self._render(source_image.copy(), parameters)
            annotated = self._bound_image(annotated)
            debug_info["annotated_image_size"] = [annotated.width, annotated.height]

            crop_ref = save_crop_image(
                annotated,
                cache_root=getattr(agent_data, "crop_cache_root", None) or self.crop_cache_root,
                request_id=getattr(agent_data, "request_id", None),
                step=getattr(agent_data, "crop_cache_step", None),
                metadata={
                    "tool": self.name,
                    "draw_op": self.draw_op,
                    "img_idx": img_idx,
                },
            )
            debug_info["crop_image_ref"] = crop_ref
            debug_info["success"] = True
        except Exception as e:  # noqa: BLE001 - surface a correctable error to the model
            logger.warning(f"Error processing drawing tool {self.name}: {e}")
            return self._error_response(str(e), debug_info)

        response_text = (
            f"Annotated image {img_idx} with {self.draw_op}. The annotated image is shown above; "
            "inspect it to verify your reasoning before answering."
        )
        return (
            ToolResponse(
                image=[annotated],
                crop_image_refs=[crop_ref],
                text=response_text,
            ),
            0.0,
            debug_info,
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
