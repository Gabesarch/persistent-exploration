# MCP server for SimpleVQA evaluation with Visual Sketchpad drawing tools
# Usage: python -m lmms_eval.mcp.simplevqa_tools
#
# All coordinates use 0-1000 scale:
#   - (0, 0) = top-left corner
#   - (1000, 1000) = bottom-right corner
#   - (500, 500) = center of image
#
import base64
import io
import json
import logging
import os
import tempfile
from typing import List, Union

# Offline mode for HuggingFace models
os.environ.setdefault("HF_HOME", "/scratch/gpfs/ZHUANGL/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/scratch/gpfs/ZHUANGL/hf_cache/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent, TextContent
from PIL import Image, ImageDraw

# The rich drawing tools below render via the heavy Visualizer (cv2 + matplotlib).
# Import it LAZILY through a same-named factory so this MCP server starts in a
# minimal/offline env. This is the single canonical tool server: it supersedes the
# old offline_image_tools.py (PIL-only) and vision_tools.py — the PIL-only tools
# image_zoom_in_tool / draw_box keep working even without cv2/matplotlib.
_VISUALIZER_CLS = None


def _get_visualizer_cls():
    global _VISUALIZER_CLS
    if _VISUALIZER_CLS is None:
        try:
            from .visualizer import Visualizer as _V
        except ImportError:
            from visualizer import Visualizer as _V
        _VISUALIZER_CLS = _V
    return _VISUALIZER_CLS


def Visualizer(*args, **kwargs):
    """Lazy factory for the cv2/matplotlib-backed Visualizer used by the rich
    drawing tools. Raises a clear RuntimeError (caught by the agentic tool loop)
    when the optional deps are missing, so the PIL-only tools stay usable offline."""
    try:
        return _get_visualizer_cls()(*args, **kwargs)
    except ImportError as exc:
        raise RuntimeError(
            "This drawing tool needs the optional 'visualizer' deps (cv2 + matplotlib), "
            "which are not installed. Use image_zoom_in_tool / draw_box (PIL-only) "
            f"instead, or install the extras. ({exc})"
        )


logger = logging.getLogger(__name__)

app = FastMCP("SimpleVQA Tools")

COORD_SCALE = 1000  # Coordinate system uses 0-1000


# Cap the size of every image a tool returns. Full-resolution PNGs of natural
# photos (e.g. a 2000x1398 visual_probe image -> ~5 MB base64) otherwise overflow
# both the MCP stdio transport (intermittent "TaskGroup" failures under concurrency)
# and the model's 32768-token context. 1 MP matches the zoom tool's output budget.
DRAW_MAX_PIXELS = 1048576
DRAW_JPEG_QUALITY = 85


_IMAGE_REGISTRY_ENV = "LMMS_EVAL_IMAGE_REGISTRY"


def _resolve_image_path(img_idx: int) -> str:
    """Resolve an image INDEX to a real file path via the registry the eval
    backend passes in the LMMS_EVAL_IMAGE_REGISTRY env var (one snapshot per
    tool call). Mirrors training, where a tool addresses images by img_idx into
    agent_data.image_data[img_idx] (verl/tools/image_zoom_in_tool.py)."""
    raw = os.environ.get(_IMAGE_REGISTRY_ENV)
    if not raw:
        raise ValueError("No image registry available (LMMS_EVAL_IMAGE_REGISTRY unset); pass img_idx for an image the eval registered.")
    registry = json.loads(raw)
    path = registry.get(str(int(img_idx)))
    if path is None:
        avail = sorted(int(k) for k in registry)
        raise ValueError(f"img_idx {img_idx} is not available. Available indices: {avail}.")
    return path


def _bound_image(image: Image.Image) -> Image.Image:
    """Downscale to <= DRAW_MAX_PIXELS so tool outputs stay small for the MCP
    stdio transport and the model context. 1 MP matches the zoom output budget."""
    image = image.convert("RGB")
    pixels = image.width * image.height
    if pixels > DRAW_MAX_PIXELS:
        scale = (DRAW_MAX_PIXELS / pixels) ** 0.5
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    return image


def _encode_jpeg_content(image: Image.Image) -> ImageContent:
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=DRAW_JPEG_QUALITY)
    buf.seek(0)
    return ImageContent(type="image", data=base64.b64encode(buf.getvalue()).decode("utf-8"), mimeType="image/jpeg")


def _save_and_emit(image: Image.Image):
    """Bound `image`, persist it to a temp file so a later tool call can address
    it by a fresh img_idx, and return [sentinel-text, bounded-JPEG-image]. The
    eval backend turns the 'Saved drawn image to:' sentinel into a new img_idx
    (mirrors training appending tool-output crops to agent_data.image_data)."""
    bounded = _bound_image(image)
    fd, path = tempfile.mkstemp(suffix=".jpg", prefix="vero2_tool_out_")
    os.close(fd)
    bounded.save(path, format="JPEG", quality=DRAW_JPEG_QUALITY)
    return [TextContent(type="text", text=f"Saved drawn image to: {path}"), _encode_jpeg_content(bounded)]


def _image_to_content(image: np.ndarray):
    """Numpy RGB array -> [sentinel, bounded JPEG] (re-addressable by img_idx)."""
    return _save_and_emit(Image.fromarray(image))


def _to_pixel(value: int, size: int) -> float:
    """Convert 0-1000 coordinate to pixel coordinate."""
    return (value / COORD_SCALE) * size


def _pil_to_content(image: Image.Image):
    """PIL image -> [sentinel, bounded JPEG] (re-addressable by img_idx)."""
    return _save_and_emit(image)


def _to_px_clamped(value: float, size: int) -> int:
    """0-1000 coordinate -> pixel, clamped to [0, size]."""
    return max(0, min(size, int((value / COORD_SCALE) * size)))


# ========== Offline-safe PIL-only tools (always available) ==========


# ---- Zooming ----
# Eval-side twin of the RL training zoom tool (verl/tools/image_zoom_in_tool.py +
# vero2_scripts/config/tool_config/image_zoom_in_tool.yaml): same name, same
# bbox_2d/label/img_idx arguments, same normalized_1000 coordinates and the same
# output pixel budget, so tool behavior matches what tool-trained checkpoints saw.
# Images are addressed by img_idx (a 0-based image index), like training: the eval
# backend maps img_idx -> the actual file via the LMMS_EVAL_IMAGE_REGISTRY env var.
ZOOM_MIN_PIXELS = 518400  # = training cropped_min_pixels
ZOOM_MAX_PIXELS = 1048576  # = training cropped_max_pixels
ZOOM_MIN_SIDE_PX = 28


def _expand_to_min_side(lo: int, hi: int, size: int, min_side: int = ZOOM_MIN_SIDE_PX):
    """Widen [lo, hi] around its center to at least min_side px, staying in [0, size]."""
    if hi - lo >= min_side:
        return lo, hi
    center = (lo + hi) // 2
    lo, hi = center - min_side // 2, center + (min_side + 1) // 2
    if lo < 0:
        lo, hi = 0, min(size, min_side)
    if hi > size:
        lo, hi = max(0, size - min_side), size
    return lo, hi


@app.tool(
    name="image_zoom_in_tool",
    description=(
        "Zoom in on a specific region of an image by cropping it based on a bounding box (bbox_2d) "
        "and an object label. img_idx is the 0-based index of the image to zoom into. Coordinates "
        "use the 0-1000 scale where (0,0)=top-left and (1000,1000)=bottom-right. The crop is returned "
        "magnified so small details become readable. "
        'Example: image_zoom_in_tool(img_idx=0, bbox_2d=[480, 90, 640, 210], label="street sign").'
    ),
)
def image_zoom_in_tool(img_idx: int, bbox_2d: List[int], label: str) -> list:
    """Crop bbox_2d=[x1,y1,x2,y2] (0-1000 scale) out of image `img_idx` and resize the
    crop into the training-time pixel budget. PIL-only (works offline)."""
    image = Image.open(_resolve_image_path(img_idx)).convert("RGB")
    w, h = image.size
    coords = [int(v) for v in list(bbox_2d)[:4]]
    if len(coords) != 4:
        raise ValueError(f"bbox_2d must be [x1, y1, x2, y2] on the 0-1000 scale, got {bbox_2d}")
    x1, y1, x2, y2 = coords
    px1, py1 = _to_px_clamped(min(x1, x2), w), _to_px_clamped(min(y1, y2), h)
    px2, py2 = _to_px_clamped(max(x1, x2), w), _to_px_clamped(max(y1, y2), h)
    px1, px2 = _expand_to_min_side(px1, px2, w)
    py1, py2 = _expand_to_min_side(py1, py2, h)
    crop = image.crop((px1, py1, max(px1 + 1, px2), max(py1 + 1, py2)))
    pixels = crop.width * crop.height
    if pixels < ZOOM_MIN_PIXELS:
        scale = (ZOOM_MIN_PIXELS / pixels) ** 0.5
    elif pixels > ZOOM_MAX_PIXELS:
        scale = (ZOOM_MAX_PIXELS / pixels) ** 0.5
    else:
        scale = 1.0
    if scale != 1.0:
        crop = crop.resize((max(1, round(crop.width * scale)), max(1, round(crop.height * scale))), Image.Resampling.LANCZOS)
    return _pil_to_content(crop)

# ========== MCP Drawing Tools (adopted from origin/main) ==========
# The 5 tools below (draw_boxes / draw_points / draw_lines / draw_texts /
# draw_polygons) are origin/main's redesigned drawing toolset, taken verbatim.
# Only their image-output path differs: they call _save_output_image, which we
# route through _save_and_emit (see helpers above) so every drawn image is
# downscaled to <= DRAW_MAX_PIXELS and JPEG-encoded. main's original
# _save_output_image emitted a full-resolution PNG, which overflowed the MCP
# stdio transport and the model context (this is what had zeroed the draw runs).


def _save_output_image(image: Image.Image):
    """Drawing-tool output path -> bounded JPEG via _save_and_emit. Returns
    ['Saved drawn image to: <path>' sentinel, bounded JPEG image]; the eval
    backend turns the sentinel into a fresh img_idx so a later tool call can
    address the drawn image (mirrors training appending crops to image_data)."""
    return _save_and_emit(image)


@app.tool(name="draw_points", description="Draw one or more point markers on an image. Coordinates use 0-1000 scale where (0,0)=top-left, (1000,1000)=bottom-right. For a single point, pass [x, y]. For multiple points, pass [[x1, y1], [x2, y2], ...]. Example: draw_points(img_idx=0, points=[[500, 500], [600, 600]]) draws 2 points.")
def draw_points(
    img_idx: int,
    points: Union[List[int], List[List[int]]],
    color: str = "lime",
    radius: int = 15,
) -> List[Union[TextContent, ImageContent]]:
    """
    Draw one or more point markers on an image.

    :param img_idx: Image index supplied by the caller.
    :param points: Either [x, y] for a single point, or [[x1, y1], [x2, y2], ...] for multiple points. Coordinates use 0-1000 scale.
    :param color: Color name for markers.
    :param radius: Radius of the circle markers (0-1000, relative to image width).
    :return: Image with point markers.
    """
    image_path = _resolve_image_path(img_idx)
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    vis = Visualizer(img)

    # Normalize to list of points
    if points and isinstance(points[0], int):
        points = [points]  # Single point: [x, y] -> [[x, y]]

    pr = _to_pixel(radius, w)
    for i, point in enumerate(points):
        x, y = point
        px, py = _to_pixel(x, w), _to_pixel(y, h)
        vis.draw_circle((px, py), color=color, radius=pr)

    return _save_output_image(Image.fromarray(vis.get_output().get_image()))


@app.tool(name="draw_lines", description="Draw one or more lines on an image. Coordinates use 0-1000 scale. For a single line, pass [x1, y1, x2, y2]. For multiple lines, pass [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]. Example: draw_lines(img_idx=0, lines=[[200, 300, 800, 700]]) draws one diagonal line.")
def draw_lines(
    img_idx: int,
    lines: Union[List[int], List[List[int]]],
    color: str = "red",
    linewidth: float = 3,
) -> List[Union[TextContent, ImageContent]]:
    """
    Draw one or more lines on an image.

    :param img_idx: Image index supplied by the caller.
    :param lines: Either [x1, y1, x2, y2] for a single line, or [[x1, y1, x2, y2], ...] for multiple lines. Coordinates use 0-1000 scale.
    :param color: Color name.
    :param linewidth: Width of the lines in pixels.
    :return: Image with lines drawn.
    """
    image_path = _resolve_image_path(img_idx)
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    vis = Visualizer(img)

    # Normalize to list of lines
    if lines and isinstance(lines[0], int):
        lines = [lines]  # Single line: [x1, y1, x2, y2] -> [[x1, y1, x2, y2]]

    for line in lines:
        x1, y1, x2, y2 = line
        px1, py1 = _to_pixel(x1, w), _to_pixel(y1, h)
        px2, py2 = _to_pixel(x2, w), _to_pixel(y2, h)
        vis.draw_line([px1, px2], [py1, py2], color=color, linewidth=linewidth)

    return _save_output_image(Image.fromarray(vis.get_output().get_image()))


@app.tool(name="draw_boxes", description="Draw one or more bounding boxes on an image. Coordinates use 0-1000 scale where (0,0)=top-left, (1000,1000)=bottom-right. For a single box, pass [x1, y1, x2, y2]. For multiple boxes, pass [[x1, y1, x2, y2], ...]. Example: draw_boxes(img_idx=0, boxes=[[100, 100, 200, 200], [300, 300, 400, 400]]) draws 2 boxes.")
def draw_boxes(
    img_idx: int,
    boxes: Union[List[int], List[List[int]]],
    color: str = "lime",
    alpha: float = 0.8,
    linewidth: int = 3,
) -> List[Union[TextContent, ImageContent]]:
    """
    Draw one or more bounding boxes on an image.

    :param img_idx: Image index supplied by the caller.
    :param boxes: Either [x1, y1, x2, y2] for a single box, or [[x1, y1, x2, y2], ...] for multiple boxes. Coordinates use 0-1000 scale.
    :param color: Color name for all boxes (e.g., 'lime', 'red', 'blue').
    :param alpha: Opacity of the box outlines.
    :param linewidth: Width of box outlines in pixels.
    :return: Image with boxes drawn.
    """
    image_path = _resolve_image_path(img_idx)
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]

    # Normalize to list of boxes
    if boxes and isinstance(boxes[0], int):
        boxes = [boxes]  # Single box: [x1, y1, x2, y2] -> [[x1, y1, x2, y2]]

    try:
        vis = Visualizer(img)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            px1, py1 = _to_pixel(x1, w), _to_pixel(y1, h)
            px2, py2 = _to_pixel(x2, w), _to_pixel(y2, h)
            vis.draw_box([px1, py1, px2, py2], edge_color=color, alpha=alpha, linewidth=linewidth)

        return _save_output_image(Image.fromarray(vis.get_output().get_image()))
    except RuntimeError:
        # Offline fallback: plain PIL rectangle (no cv2/matplotlib).
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            px1, py1 = _to_pixel(x1, w), _to_pixel(y1, h)
            px2, py2 = _to_pixel(x2, w), _to_pixel(y2, h)
            draw.rectangle([px1, py1, px2, py2], outline=color, width=max(1, int(linewidth)))
        return _save_output_image(pil_img)


@app.tool(name="draw_texts", description="Add one or more text annotations to an image. Coordinates use 0-1000 scale. For a single text, pass {\"x\": x, \"y\": y, \"text\": \"label\"}. For multiple, pass a list. Example: draw_texts(img_idx=0, texts=[{\"x\": 500, \"y\": 100, \"text\": \"Title\"}]) adds text at top center.")
def draw_texts(
    img_idx: int,
    texts: Union[dict, List[dict]],
    color: str = "red",
    font_size: int = 18,
) -> List[Union[TextContent, ImageContent]]:
    """
    Add one or more text annotations to an image.

    :param img_idx: Image index supplied by the caller.
    :param texts: Either {"x": x, "y": y, "text": "label"} for a single text, or a list of such dicts for multiple texts. Coordinates use 0-1000 scale.
    :param color: Text color name.
    :param font_size: Size of the font in pixels.
    :return: Image with text drawn.
    """
    image_path = _resolve_image_path(img_idx)
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    vis = Visualizer(img)

    # Normalize to list of text objects
    if isinstance(texts, dict):
        texts = [texts]

    for text_obj in texts:
        x = text_obj["x"]
        y = text_obj["y"]
        text = text_obj["text"]
        px, py = _to_pixel(x, w), _to_pixel(y, h)
        vis.draw_text(text, (px, py), color=color, font_size=font_size)

    return _save_output_image(Image.fromarray(vis.get_output().get_image()))


@app.tool(name="draw_polygons", description="Draw one or more polygons on an image. Coordinates use 0-1000 scale. For a single polygon, pass [[x1,y1], [x2,y2], [x3,y3], ...]. For multiple polygons, pass [[[x1,y1], ...], [[x1,y1], ...], ...]. Example: draw_polygons(img_idx=0, polygons=[[[200,200], [800,200], [500,800]]]) draws a triangle.")
def draw_polygons(
    img_idx: int,
    polygons: Union[List[List[int]], List[List[List[int]]]],
    color: str = "red",
    alpha: float = 0.5,
) -> List[Union[TextContent, ImageContent]]:
    """
    Draw one or more polygons on an image.

    :param img_idx: Image index supplied by the caller.
    :param polygons: Either [[x1,y1], [x2,y2], ...] for a single polygon, or [[[x1,y1], ...], [[x1,y1], ...]] for multiple polygons. Coordinates use 0-1000 scale.
    :param color: Color name.
    :param alpha: Opacity of the fill.
    :return: Image with polygons drawn.
    """
    image_path = _resolve_image_path(img_idx)
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    vis = Visualizer(img)

    # Normalize to list of polygons
    # Check if it's a single polygon (list of [x, y] pairs) or multiple polygons
    if polygons and isinstance(polygons[0], list) and len(polygons[0]) == 2 and isinstance(polygons[0][0], int):
        polygons = [polygons]  # Single polygon: [[x1,y1], [x2,y2], ...] -> [[[x1,y1], [x2,y2], ...]]

    for polygon in polygons:
        pixel_points = [[_to_pixel(p[0], w), _to_pixel(p[1], h)] for p in polygon]
        vis.draw_polygon(np.array(pixel_points), color=color, alpha=alpha)

    return _save_output_image(Image.fromarray(vis.get_output().get_image()))


if __name__ == "__main__":
    app.run()
