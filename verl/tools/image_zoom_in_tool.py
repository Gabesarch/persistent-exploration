# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
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

import logging
import math
import os
import threading
from contextlib import ExitStack
from enum import Enum
from io import BytesIO
from math import ceil, floor
from typing import Any, Callable, Optional, TypeVar
from urllib.parse import unquote, urlparse
from uuid import uuid4

import ray
import ray.actor
from PIL import Image
from qwen_vl_utils import fetch_image

from verl.utils.crop_cache import save_crop_image

from .base_tool import BaseTool
from .schemas import OpenAIFunctionToolSchema, ToolResponse

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

T = TypeVar("T")


class OriginalSourceUnavailableError(ValueError):
    def __init__(self, message: str, source_info: dict[str, Any]):
        super().__init__(message)
        self.source_info = source_info


# Adapted from verl/tools/sandbox_fusion_tools.py
class PoolMode(Enum):
    """Execution pool mode enumeration."""

    ThreadMode = 1
    ProcessMode = 2


@ray.remote(concurrency_groups={"acquire": 1, "release": 10})
class TokenBucketWorker:
    """Ray actor for rate limiting using token bucket algorithm."""

    def __init__(self, rate_limit: int):
        self.rate_limit = rate_limit
        self.current_count = 0  # For observability
        self._semaphore = threading.Semaphore(rate_limit)

    @ray.method(concurrency_group="acquire")
    def acquire(self):
        """Acquire a token from the bucket."""
        self._semaphore.acquire()
        self.current_count += 1

    @ray.method(concurrency_group="release")
    def release(self):
        """Release a token back to the bucket."""
        self._semaphore.release()
        self.current_count -= 1

    def get_current_count(self):
        """Get current number of acquired tokens."""
        return self.current_count


class VisualExecutionWorker:
    """Worker for executing visual processing operations with optional rate limiting."""

    def __init__(self, enable_global_rate_limit=True, rate_limit=10):
        self.rate_limit_worker = self._init_rate_limit(rate_limit) if enable_global_rate_limit else None

    def _init_rate_limit(self, rate_limit):
        """Initialize singleton rate limiter."""
        return TokenBucketWorker.options(name="rate-limiter", get_if_exists=True).remote(rate_limit)

    def ping(self):
        """Health check method."""
        return True

    def execute(self, fn: Callable[..., T], *fn_args, **fn_kwargs) -> T:
        """Execute function with optional rate limiting."""
        if self.rate_limit_worker:
            with ExitStack() as stack:
                stack.callback(self.rate_limit_worker.release.remote)
                ray.get(self.rate_limit_worker.acquire.remote())
                try:
                    return fn(*fn_args, **fn_kwargs)
                except Exception as e:
                    # TODO we should make this available to the tool caller
                    logger.warning(f"Error when executing visual processing: {e}")
        else:
            return fn(*fn_args, **fn_kwargs)


def init_visual_execution_pool(
    num_workers: int, enable_global_rate_limit=True, rate_limit=10, mode: PoolMode = PoolMode.ThreadMode
):
    """Initialize visual execution pool."""
    if mode == PoolMode.ThreadMode:
        return (
            ray.remote(VisualExecutionWorker)
            .options(max_concurrency=num_workers)
            .remote(enable_global_rate_limit=enable_global_rate_limit, rate_limit=rate_limit)
        )
    else:
        raise NotImplementedError("Process mode is not implemented yet")


class ImageZoomInTool(BaseTool):
    """A tool for zooming in on an image by cropping it based on a bounding box.

    This tool provides a zoom-in functionality by cropping a region from an image,
    with rate limiting and concurrent execution support through Ray.

    Methods:
        get_openai_tool_schema: Return the tool schema in OpenAI format
        create: Create a tool instance for a trajectory
        execute: Execute the zoom-in operation
        calc_reward: Calculate the reward with respect to tool state
        release: Release the tool instance
    """

    MIN_DIMENSION = 32
    SMART_RESIZE_FACTOR = 32
    SMART_RESIZE_MIN_PIXELS = 256 * 32 * 32
    SMART_RESIZE_MAX_PIXELS = 12845056
    IMAGE_SIZE_CONTROL_KEYS = {
        "max_pixels",
        "min_pixels",
        "total_pixels",
        "resized_height",
        "resized_width",
        "image_factor",
    }

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        """
        _tool_schema = OpenAIFunctionToolSchema.model_validate({
            "type": "function",
            "function": {
                "name": "image_zoom_in_tool",
                "description": (
                    "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox) and an "
                    "optional object label."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "bbox_2d": {
                            "type": "array",
                            "items":{"type":"number"},
                            "minItems":4,
                            "maxItems":4,
                            "description": (
                                "The bounding box of the region to zoom in, as [x1, y1, x2, y2], where (x1, y1) is "
                                "the top-left corner and (x2, y2) is the bottom-right corner."
                            ),
                        },
                        "label": {
                            "type": "string",
                            "description": "The name or label of the object in the specified bounding box.",
                        },
                        "img_idx": {
                            "type": "number",
                            "description": "The index of the zoomed-in image (starting from 0).",
                        },
                    },
                    "required": ["bbox_2d", "label", "img_idx"],
                },
            }
        })
        """
        super().__init__(config, tool_schema)
        self._instance_dict = {}

        # Worker and rate limiting configuration
        self.num_workers = config.get("num_workers", 20)
        self.rate_limit = config.get("rate_limit", 50)
        self.timeout = config.get("timeout", 30)
        self.coordinate_system = config.get("coordinate_system", "pixel")
        self.cropped_min_pixels = int(config.get("cropped_min_pixels", self.SMART_RESIZE_MIN_PIXELS))
        self.cropped_max_pixels = int(
            config.get("cropped_max_pixels", config.get("max_pixels", self.SMART_RESIZE_MAX_PIXELS))
        )
        self.use_original_resolution_source = bool(config.get("use_original_resolution_source", False))
        self.original_source_unavailable_policy = config.get("original_source_unavailable_policy", "fallback")
        self.crop_cache_root = config.get("crop_cache_root")
        if self.original_source_unavailable_policy not in ("fallback", "error"):
            raise ValueError(
                "original_source_unavailable_policy must be either 'fallback' or 'error', "
                f"got {self.original_source_unavailable_policy!r}."
            )

        self.enable_global_rate_limit = config.get("enable_global_rate_limit", True)
        self.execution_pool = None
        logger.info(f"Initialized ImageZoomInTool with config: {config}")

    def round_by_factor(self, number: int, factor: int) -> int:
        return round(number / factor) * factor

    def ceil_by_factor(self, number: int, factor: int) -> int:
        return math.ceil(number / factor) * factor

    def floor_by_factor(self, number: int, factor: int) -> int:
        return math.floor(number / factor) * factor

    def smart_resize(
        self,
        height: int,
        width: int,
        factor: int = SMART_RESIZE_FACTOR,
        min_pixels: int = SMART_RESIZE_MIN_PIXELS,
        max_pixels: int = SMART_RESIZE_MAX_PIXELS,
    ) -> tuple[int, int]:
        h_bar = max(factor, self.round_by_factor(height, factor))
        w_bar = max(factor, self.round_by_factor(width, factor))
        if h_bar * w_bar > max_pixels:
            beta = math.sqrt((height * width) / max_pixels)
            h_bar = self.floor_by_factor(height / beta, factor)
            w_bar = self.floor_by_factor(width / beta, factor)
        elif h_bar * w_bar < min_pixels:
            beta = math.sqrt(min_pixels / (height * width))
            h_bar = self.ceil_by_factor(height * beta, factor)
            w_bar = self.ceil_by_factor(width * beta, factor)
        return h_bar, w_bar

    def _maybe_resize_bbox(self, bbox_2d: list[float], image_width: int, image_height: int) -> list[float]:
        """Clamp and resize bbox using Qwen-Agent Qwen3-VL behavior."""
        left, top, right, bottom = bbox_2d

        left = max(0, left)
        top = max(0, top)
        right = min(image_width, right)
        bottom = min(image_height, bottom)

        height = bottom - top
        width = right - left
        if height < self.MIN_DIMENSION or width < self.MIN_DIMENSION:
            center_x = (left + right) / 2.0
            center_y = (top + bottom) / 2.0
            ratio = self.MIN_DIMENSION / min(height, width)
            new_half_height = ceil(height * ratio * 0.5)
            new_half_width = ceil(width * ratio * 0.5)
            new_left = floor(center_x - new_half_width)
            new_right = ceil(center_x + new_half_width)
            new_top = floor(center_y - new_half_height)
            new_bottom = ceil(center_y + new_half_height)

            new_left = max(0, new_left)
            new_top = max(0, new_top)
            new_right = min(image_width, new_right)
            new_bottom = min(image_height, new_bottom)

            new_height = new_bottom - new_top
            new_width = new_right - new_left

            if new_height > self.MIN_DIMENSION and new_width > self.MIN_DIMENSION:
                return [new_left, new_top, new_right, new_bottom]

        return [left, top, right, bottom]

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return self.tool_schema

    async def create(self, instance_id: Optional[str] = None, **kwargs) -> tuple[str, ToolResponse]:
        """
        Creates a new instance for image zoom-in tool.

        This method initializes a new session for an image, which can then be used
        for operations like zooming. It fetches the image from various sources
        and stores it internally.

        Args:
            instance_id: An optional unique identifier for the instance. If not
                provided, a new UUID will be generated.
            **kwargs: Should contain 'image' key with image data, or 'create_kwargs'
                containing {'image': image_data}. Image can be one of the following:
                - A PIL.Image.Image object.
                - A string containing an HTTP or HTTPS URL.
                - A string containing a local file path.
                - A string containing a file URI (e.g., "file:///path/to/image.jpg").
                - A string containing a base64-encoded image in the format of "data:image/jpeg;base64,..."

        Returns:
            Tuple of (instance_id, ToolResponse)
        """
        if instance_id is None:
            instance_id = str(uuid4())

        # Handle create_kwargs parameter if passed
        create_kwargs = kwargs.get("create_kwargs", {})
        if create_kwargs:
            kwargs.update(create_kwargs)

        # Get image from kwargs. If omitted, execute() can fall back to the
        # original prompt image in agent_data.
        image = kwargs.get("image")
        img = fetch_image({"image": image}) if image is not None else None
        self._instance_dict[instance_id] = {
            "image": img,
            "response": "",
            "reward": 0.0,
        }
        return instance_id, ToolResponse()

    def _get_processed_source_image(self, instance_id: str, img_idx: int, agent_data: Any = None) -> Any:
        instance_data = self._instance_dict[instance_id]

        if agent_data is not None and agent_data.image_data is not None:
            image_data = agent_data.image_data
            if not isinstance(image_data, list):
                image_data = [image_data]
            if img_idx >= len(image_data):
                raise IndexError(f"img_idx={img_idx} is out of range for {len(image_data)} images.")
            return image_data[img_idx]

        image = instance_data.get("image")
        if image is not None:
            if img_idx != 0:
                raise IndexError(f"img_idx={img_idx} is out of range for 1 fallback image.")
            return image

        raise ValueError("Missing source image: provide agent_data.image_data or create_kwargs.image.")

    def _get_source_image(self, instance_id: str, img_idx: int, agent_data: Any = None) -> Any:
        return self._get_processed_source_image(instance_id, img_idx, agent_data)

    def _processed_image_count(self, agent_data: Any = None) -> Optional[int]:
        if agent_data is None or agent_data.image_data is None:
            return None
        image_data = agent_data.image_data
        return len(image_data) if isinstance(image_data, list) else 1

    def _strip_image_size_control_keys(self, source: dict[str, Any]) -> dict[str, Any]:
        return {key: value for key, value in source.items() if key not in self.IMAGE_SIZE_CONTROL_KEYS}

    def _load_original_source_image(self, source: Any) -> Image.Image:
        if isinstance(source, Image.Image):
            return source.convert("RGB").copy()

        if isinstance(source, dict):
            source = self._strip_image_size_control_keys(source)
            if "bytes" in source:
                return Image.open(BytesIO(source["bytes"])).convert("RGB")
            if "image" in source:
                return self._load_original_source_image(source["image"])
            if "image_url" in source:
                image_url = source["image_url"]
                if isinstance(image_url, dict) and "url" in image_url:
                    image_url = image_url["url"]
                return self._load_original_source_image(image_url)
            return fetch_image(source).convert("RGB")

        if isinstance(source, str):
            parsed = urlparse(source)
            if parsed.scheme == "file":
                path = unquote(parsed.path)
                with Image.open(path) as image:
                    return image.convert("RGB")
            if parsed.scheme in ("", None) and os.path.exists(source):
                with Image.open(source) as image:
                    return image.convert("RGB")
            return fetch_image({"image": source}).convert("RGB")

        raise TypeError(f"Unsupported original image source type: {type(source).__name__}")

    def _select_source_image(
        self, instance_id: str, img_idx: int, agent_data: Any = None
    ) -> tuple[Image.Image, dict[str, Any]]:
        source_info = {
            "used_original_resolution_source": False,
            "original_source_unavailable_policy": self.original_source_unavailable_policy,
        }

        original_sources = (
            getattr(agent_data, "original_prompt_image_sources", None) if agent_data is not None else None
        )
        if self.use_original_resolution_source and original_sources is not None and img_idx < len(original_sources):
            try:
                image = self._load_original_source_image(original_sources[img_idx])
                source_info["used_original_resolution_source"] = True
                source_info["original_source_size"] = [image.width, image.height]
                return image, source_info
            except Exception as exc:
                source_info["original_source_error"] = str(exc)
                if self.original_source_unavailable_policy == "error":
                    raise OriginalSourceUnavailableError(
                        f"Original image source unavailable: {exc}", source_info
                    ) from exc
                source_info["original_source_fallback"] = True

        image = self._get_processed_source_image(instance_id, img_idx, agent_data)
        source_info["processed_source_size"] = [image.width, image.height]
        source_info["initial_prompt_image_count"] = len(original_sources) if original_sources is not None else 0
        processed_count = self._processed_image_count(agent_data)
        if processed_count is not None:
            source_info["processed_image_count"] = processed_count
        return image, source_info

    def _parse_img_idx(self, parameters: dict[str, Any]) -> int:
        if "img_idx" not in parameters:
            raise ValueError("img_idx parameter is missing.")
        img_idx = parameters["img_idx"]
        if isinstance(img_idx, bool) or not isinstance(img_idx, int | float) or int(img_idx) != img_idx:
            raise ValueError(f"img_idx must be an integer image index, got {img_idx!r}.")
        img_idx = int(img_idx)
        if img_idx < 0:
            raise ValueError(f"img_idx must be non-negative, got {img_idx}.")
        return img_idx

    def _convert_bbox_to_pixels(self, bbox_2d: list[float], image_width: int, image_height: int) -> list[float]:
        if self.coordinate_system in ("pixel", "pixels", None):
            return bbox_2d
        if self.coordinate_system == "normalized_1000":
            left, top, right, bottom = [float(value) for value in bbox_2d]
            return [
                left / 1000.0 * image_width,
                top / 1000.0 * image_height,
                right / 1000.0 * image_width,
                bottom / 1000.0 * image_height,
            ]
        raise ValueError(f"Unsupported coordinate_system: {self.coordinate_system}")

    def _validate_and_clamp_bbox(self, bbox_2d: list[float], image_width: int, image_height: int) -> list[float]:
        try:
            left, top, right, bottom = [float(value) for value in bbox_2d]
        except (TypeError, ValueError) as exc:
            raise ValueError("bbox_2d must contain 4 numeric values.") from exc

        if any(isinstance(value, bool) for value in bbox_2d) or not all(
            math.isfinite(value) for value in (left, top, right, bottom)
        ):
            raise ValueError("bbox_2d must contain 4 finite numeric values.")

        if right <= left or bottom <= top:
            raise ValueError("bbox_2d must define a positive-area box with left < right and top < bottom.")

        left = max(0.0, left)
        top = max(0.0, top)
        right = min(float(image_width), right)
        bottom = min(float(image_height), bottom)

        if right <= left or bottom <= top:
            raise ValueError("bbox_2d is outside image bounds or has no positive area after clamping.")

        return [left, top, right, bottom]

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        bbox_2d = parameters.get("bbox_2d")
        label = parameters.get("label")
        debug_info = {
            "tool": "image_zoom_in_tool",
            "parameters": parameters,
            "bbox_2d": bbox_2d,
            "label": label,
            "coordinate_system": self.coordinate_system,
        }

        if not bbox_2d or len(bbox_2d) != 4:
            debug_info["success"] = False
            debug_info["error"] = "bbox_2d parameter is missing or not a list of 4 numbers."
            return (
                ToolResponse(text="Error: bbox_2d parameter is missing or not a list of 4 numbers."),
                -0.05,
                debug_info,
            )
        if "label" not in parameters:
            debug_info["success"] = False
            debug_info["error"] = "label parameter is missing."
            return ToolResponse(text="Error: label parameter is missing."), -0.05, debug_info

        try:
            img_idx = self._parse_img_idx(parameters)
            debug_info["img_idx"] = img_idx
            image, source_info = self._select_source_image(instance_id, img_idx, kwargs.get("agent_data"))
            debug_info.update(source_info)
            image_width, image_height = image.size
            debug_info["source_image_size"] = [image_width, image_height]
            pixel_bbox = self._convert_bbox_to_pixels(bbox_2d, image_width=image_width, image_height=image_height)
            debug_info["pixel_bbox"] = pixel_bbox
            clamped_bbox = self._validate_and_clamp_bbox(
                pixel_bbox, image_width=image_width, image_height=image_height
            )
            debug_info["clamped_bbox"] = clamped_bbox
            resized_bbox = self._maybe_resize_bbox(clamped_bbox, image_width=image_width, image_height=image_height)
            debug_info["resized_bbox"] = resized_bbox

            cropped_image = image.crop(resized_bbox)
            left, top, right, bottom = resized_bbox
            new_width, new_height = self.smart_resize(
                right - left,
                bottom - top,
                factor=self.SMART_RESIZE_FACTOR,
                min_pixels=self.cropped_min_pixels,
                max_pixels=self.cropped_max_pixels,
            )
            cropped_image = cropped_image.resize((new_width, new_height), resample=Image.BICUBIC)
            debug_info["cropped_image_size"] = [new_width, new_height]
            crop_ref = save_crop_image(
                cropped_image,
                cache_root=getattr(kwargs.get("agent_data"), "crop_cache_root", None) or self.crop_cache_root,
                request_id=getattr(kwargs.get("agent_data"), "request_id", None),
                step=getattr(kwargs.get("agent_data"), "crop_cache_step", None),
                metadata={
                    "tool": "image_zoom_in_tool",
                    "img_idx": img_idx,
                    "bbox_2d": list(bbox_2d),
                    "label": label,
                },
            )
            debug_info["crop_image_ref"] = crop_ref
            debug_info["success"] = True
            logger.info(f"Cropped image size: {cropped_image.size}")
        except Exception as e:
            logger.error(f"Error processing image zoom-in: {e}")
            if isinstance(e, OriginalSourceUnavailableError):
                debug_info.update(e.source_info)
            debug_info["success"] = False
            debug_info["error"] = str(e)
            return ToolResponse(text=f"Error processing image zoom-in: {e}"), -0.05, debug_info

        response_text = f"Zoomed in on the image to the region {bbox_2d}."
        response_text = f"Zoomed in on the image to the region {bbox_2d} with label {label}."

        return (
            ToolResponse(
                image=[cropped_image],
                crop_image_refs=[crop_ref],
                text=response_text,
            ),
            0.0,
            debug_info,
        )

    async def release(self, instance_id: str, **kwargs) -> None:
        if instance_id in self._instance_dict:
            del self._instance_dict[instance_id]
