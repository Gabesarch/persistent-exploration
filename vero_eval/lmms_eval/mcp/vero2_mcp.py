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
import logging
import os
from typing import List, Optional, Union

# Offline mode for HuggingFace models
os.environ.setdefault("HF_HOME", "/scratch/gpfs/ZHUANGL/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/scratch/gpfs/ZHUANGL/hf_cache/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from PIL import Image, ImageDraw

# The rich drawing tools below render via the heavy Visualizer (cv2 + matplotlib).
# Import it LAZILY through a same-named factory so this MCP server starts in a
# minimal/offline env. This is the single canonical tool server: it supersedes the
# old offline_image_tools.py (PIL-only) and vision_tools.py — the PIL-only tools
# image_crop / image_resize / draw_box keep working even without cv2/matplotlib.
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
            "which are not installed. Use image_crop / image_resize / draw_box (PIL-only) "
            f"instead, or install the extras. ({exc})"
        )


logger = logging.getLogger(__name__)

app = FastMCP("SimpleVQA Tools")

COORD_SCALE = 1000  # Coordinate system uses 0-1000


def _image_to_content(image: np.ndarray) -> ImageContent:
    """Convert numpy RGB array to ImageContent."""
    pil_image = Image.fromarray(image)
    image_bytes = io.BytesIO()
    pil_image.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    data = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return ImageContent(type="image", data=data, mimeType="image/png")


def _to_pixel(value: int, size: int) -> float:
    """Convert 0-1000 coordinate to pixel coordinate."""
    return (value / COORD_SCALE) * size


def _pil_to_content(image: Image.Image) -> ImageContent:
    """Encode a PIL image as an MCP ImageContent (base64 PNG)."""
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return ImageContent(type="image", data=base64.b64encode(buf.getvalue()).decode("utf-8"), mimeType="image/png")


def _to_px_clamped(value: float, size: int) -> int:
    """0-1000 coordinate -> pixel, clamped to [0, size]."""
    return max(0, min(size, int((value / COORD_SCALE) * size)))


# ========== MCP Drawing Tools ==========


@app.tool(name="draw_points", description="Draw one or more point markers on an image. Coordinates use 0-1000 scale where (0,0)=top-left, (1000,1000)=bottom-right. For a single point, pass [x, y]. For multiple points, pass [[x1, y1], [x2, y2], ...]. Example: draw_points(img, [[500, 500], [600, 600]]) draws 2 points.")
def draw_points(
    image_path: str,
    points: Union[List[int], List[List[int]]],
    color: str = "lime",
    radius: int = 15,
    show_numbers: bool = True,
    show_total: bool = True,
) -> ImageContent:
    """
    Draw one or more point markers on an image.

    :param image_path: Path to the input image.
    :param points: Either [x, y] for a single point, or [[x1, y1], [x2, y2], ...] for multiple points. Coordinates use 0-1000 scale.
    :param color: Color name for markers.
    :param radius: Radius of the circle markers (0-1000, relative to image width).
    :param show_numbers: If True, display number labels next to each point.
    :param show_total: If True, display total count at top of image.
    :return: Image with point markers.
    """
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
        if show_numbers:
            vis.draw_text(str(i + 1), (px, py - pr - 5), color=color, font_size=12)

    if show_total and len(points) > 1:
        vis.draw_text(f"Count: {len(points)}", (10, 10), color="white", font_size=18)

    return _image_to_content(vis.get_output().get_image())


@app.tool(name="draw_lines", description="Draw one or more lines on an image. Coordinates use 0-1000 scale. For a single line, pass [x1, y1, x2, y2]. For multiple lines, pass [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]. Example: draw_lines(img, [[200, 300, 800, 700]]) draws one diagonal line.")
def draw_lines(
    image_path: str,
    lines: Union[List[int], List[List[int]]],
    color: str = "red",
    linewidth: float = 3,
) -> ImageContent:
    """
    Draw one or more lines on an image.

    :param image_path: Path to the input image.
    :param lines: Either [x1, y1, x2, y2] for a single line, or [[x1, y1, x2, y2], ...] for multiple lines. Coordinates use 0-1000 scale.
    :param color: Color name.
    :param linewidth: Width of the lines in pixels.
    :return: Image with lines drawn.
    """
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

    return _image_to_content(vis.get_output().get_image())


@app.tool(name="draw_boxes", description="Draw one or more bounding boxes on an image. Coordinates use 0-1000 scale where (0,0)=top-left, (1000,1000)=bottom-right. For a single box, pass [x1, y1, x2, y2]. For multiple boxes, pass [[x1, y1, x2, y2], ...]. Example: draw_boxes(img, [[100, 100, 200, 200], [300, 300, 400, 400]]) draws 2 boxes.")
def draw_boxes(
    image_path: str,
    boxes: Union[List[int], List[List[int]]],
    color: str = "lime",
    alpha: float = 0.8,
    linewidth: int = 3,
    show_numbers: bool = True,
    show_total: bool = True,
) -> ImageContent:
    """
    Draw one or more bounding boxes on an image.

    :param image_path: Path to the input image.
    :param boxes: Either [x1, y1, x2, y2] for a single box, or [[x1, y1, x2, y2], ...] for multiple boxes. Coordinates use 0-1000 scale.
    :param color: Color name for all boxes (e.g., 'lime', 'red', 'blue').
    :param alpha: Opacity of the box outlines.
    :param linewidth: Width of box outlines in pixels.
    :param show_numbers: If True, display number labels on each box.
    :param show_total: If True, display total count at top of image.
    :return: Image with boxes drawn.
    """
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
            if show_numbers:
                vis.draw_text(str(i + 1), (px1 + 2, py1 + 2), color=color, font_size=14)

        if show_total and len(boxes) > 1:
            vis.draw_text(f"Count: {len(boxes)}", (10, 10), color="white", font_size=18)

        return _image_to_content(vis.get_output().get_image())
    except RuntimeError:
        # Offline fallback: plain PIL rectangle (no cv2/matplotlib).
        pil_img = Image.fromarray(img)
        draw = ImageDraw.Draw(pil_img)
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = box
            px1, py1 = _to_pixel(x1, w), _to_pixel(y1, h)
            px2, py2 = _to_pixel(x2, w), _to_pixel(y2, h)
            draw.rectangle([px1, py1, px2, py2], outline=color, width=max(1, int(linewidth)))
        return _pil_to_content(pil_img)


@app.tool(name="draw_texts", description="Add one or more text annotations to an image. Coordinates use 0-1000 scale. For a single text, pass {\"x\": x, \"y\": y, \"text\": \"label\"}. For multiple, pass a list. Example: draw_texts(img, [{\"x\": 500, \"y\": 100, \"text\": \"Title\"}]) adds text at top center.")
def draw_texts(
    image_path: str,
    texts: Union[dict, List[dict]],
    color: str = "red",
    font_size: int = 18,
) -> ImageContent:
    """
    Add one or more text annotations to an image.

    :param image_path: Path to the input image.
    :param texts: Either {"x": x, "y": y, "text": "label"} for a single text, or a list of such dicts for multiple texts. Coordinates use 0-1000 scale.
    :param color: Text color name.
    :param font_size: Size of the font in pixels.
    :return: Image with text drawn.
    """
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

    return _image_to_content(vis.get_output().get_image())


@app.tool(name="draw_polygons", description="Draw one or more polygons on an image. Coordinates use 0-1000 scale. For a single polygon, pass [[x1,y1], [x2,y2], [x3,y3], ...]. For multiple polygons, pass [[[x1,y1], ...], [[x1,y1], ...], ...]. Example: draw_polygons(img, [[[200,200], [800,200], [500,800]]]) draws a triangle.")
def draw_polygons(
    image_path: str,
    polygons: Union[List[List[int]], List[List[List[int]]]],
    color: str = "red",
    alpha: float = 0.5,
) -> ImageContent:
    """
    Draw one or more polygons on an image.

    :param image_path: Path to the input image.
    :param polygons: Either [[x1,y1], [x2,y2], ...] for a single polygon, or [[[x1,y1], ...], [[x1,y1], ...]] for multiple polygons. Coordinates use 0-1000 scale.
    :param color: Color name.
    :param alpha: Opacity of the fill.
    :return: Image with polygons drawn.
    """
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

    return _image_to_content(vis.get_output().get_image())


if __name__ == "__main__":
    app.run()
