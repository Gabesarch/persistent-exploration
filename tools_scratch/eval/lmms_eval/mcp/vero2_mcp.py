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
from typing import List

# Offline mode for HuggingFace models
os.environ.setdefault("HF_HOME", "/scratch/gpfs/ZHUANGL/hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", "/scratch/gpfs/ZHUANGL/hf_cache/hub")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

import numpy as np
from mcp.server.fastmcp import FastMCP
from mcp.types import ImageContent
from PIL import Image

try:
    from .visualizer import Visualizer
except ImportError:
    from visualizer import Visualizer

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


# ========== MCP Drawing Tools ==========


# ========== Simple PIL-based tools (no distortion) ==========

@app.tool(name="draw_line_simple", description="Draw a line on an image using PIL (no distortion). Coordinates use 0-1000 scale.")
def draw_line_simple(
    image_path: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "red",
    linewidth: int = 3,
) -> ImageContent:
    """Draw a line using PIL ImageDraw (no matplotlib distortion)."""
    from PIL import ImageDraw
    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    px1, py1 = int(_to_pixel(x1, w)), int(_to_pixel(y1, h))
    px2, py2 = int(_to_pixel(x2, w)), int(_to_pixel(y2, h))
    draw = ImageDraw.Draw(img)
    draw.line([(px1, py1), (px2, py2)], fill=color, width=linewidth)
    return _image_to_content(np.array(img))


@app.tool(name="draw_box", description="Draw a bounding box on an image. Coordinates use 0-1000 scale where (0,0)=top-left, (1000,1000)=bottom-right. Example: draw_box(img, 100, 200, 500, 600, linewidth=5) draws a thick box.")
def draw_box(
    image_path: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "red",
    alpha: float = 0.8,
    linewidth: int = 3,
) -> ImageContent:
    """
    Draw a bounding box on an image.

    :param image_path: Path to the input image.
    :param x1: Left x coordinate (0-1000).
    :param y1: Top y coordinate (0-1000).
    :param x2: Right x coordinate (0-1000).
    :param y2: Bottom y coordinate (0-1000).
    :param color: Color name (e.g., 'red', 'blue', 'green').
    :param alpha: Opacity of the box outline.
    :param linewidth: Width of the box outline in pixels.
    :return: Image with box drawn.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    px1, py1 = _to_pixel(x1, w), _to_pixel(y1, h)
    px2, py2 = _to_pixel(x2, w), _to_pixel(y2, h)
    vis = Visualizer(img)
    vis.draw_box([px1, py1, px2, py2], edge_color=color, alpha=alpha, linewidth=linewidth)
    return _image_to_content(vis.get_output().get_image())


@app.tool(name="draw_numbered_boxes", description="Draw multiple numbered bounding boxes on an image. Coordinates use 0-1000 scale. Each box gets a unique number label. Example: draw_numbered_boxes(img, [[100,100,200,200], [300,300,400,400]], linewidth=4) draws 2 numbered boxes.")
def draw_numbered_boxes(
    image_path: str,
    boxes: List[List[int]],
    color: str = "lime",
    alpha: float = 0.8,
    linewidth: int = 3,
    show_total: bool = True,
) -> ImageContent:
    """
    Draw multiple numbered bounding boxes on an image.

    :param image_path: Path to the input image.
    :param boxes: List of boxes, each as [x1, y1, x2, y2] in 0-1000 scale.
    :param color: Color name for all boxes (e.g., 'lime', 'red', 'blue').
    :param alpha: Opacity of the box outlines.
    :param linewidth: Width of box outlines in pixels.
    :param show_total: If True, display total number at top of image.
    :return: Image with numbered boxes.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    vis = Visualizer(img)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        px1, py1 = _to_pixel(x1, w), _to_pixel(y1, h)
        px2, py2 = _to_pixel(x2, w), _to_pixel(y2, h)
        vis.draw_box([px1, py1, px2, py2], edge_color=color, alpha=alpha, linewidth=linewidth)
        vis.draw_text(str(i + 1), (px1 + 2, py1 + 2), color=color, font_size=14)

    if show_total:
        vis.draw_text(f"Count: {len(boxes)}", (10, 10), color="white", font_size=18)

    return _image_to_content(vis.get_output().get_image())


@app.tool(name="draw_numbered_points", description="Draw numbered point markers on an image. Coordinates use 0-1000 scale. Example: draw_numbered_points(img, [[500,500], [600,600]]) marks 2 numbered points.")
def draw_numbered_points(
    image_path: str,
    points: List[List[int]],
    color: str = "lime",
    radius: int = 15,
    show_total: bool = True,
) -> ImageContent:
    """
    Draw numbered point markers on an image.

    :param image_path: Path to the input image.
    :param points: List of points, each as [x, y] in 0-1000 scale.
    :param color: Color name for markers.
    :param radius: Radius of the circle markers (0-1000, relative to image width).
    :param show_total: If True, display total number at top of image.
    :return: Image with numbered point markers.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    vis = Visualizer(img)

    pr = _to_pixel(radius, w)
    for i, point in enumerate(points):
        x, y = point
        px, py = _to_pixel(x, w), _to_pixel(y, h)
        vis.draw_circle((px, py), color=color, radius=pr)
        vis.draw_text(str(i + 1), (px, py - pr - 5), color=color, font_size=12)

    if show_total:
        vis.draw_text(f"Count: {len(points)}", (10, 10), color="white", font_size=18)

    return _image_to_content(vis.get_output().get_image())


@app.tool(name="draw_grid", description="Draw a grid overlay on an image. Specify rows and columns. Example: draw_grid(img, 3, 3) draws a 3x3 grid.")
def draw_grid(
    image_path: str,
    rows: int = 3,
    cols: int = 3,
    color: str = "yellow",
    linewidth: float = 2,
    show_labels: bool = True,
) -> ImageContent:
    """
    Draw a grid overlay on an image.

    :param image_path: Path to the input image.
    :param rows: Number of grid rows.
    :param cols: Number of grid columns.
    :param color: Color of grid lines.
    :param linewidth: Width of grid lines.
    :param show_labels: If True, label each cell (A1, A2, B1, etc.).
    :return: Image with grid overlay.
    """
    from PIL import ImageDraw

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    cell_w = w / cols
    cell_h = h / rows

    for i in range(1, cols):
        x = int(i * cell_w)
        draw.line([(x, 0), (x, h)], fill=color, width=int(linewidth))

    for i in range(1, rows):
        y = int(i * cell_h)
        draw.line([(0, y), (w, y)], fill=color, width=int(linewidth))

    if show_labels:
        img_array = np.array(img)
        vis = Visualizer(img_array)
        for r in range(rows):
            for c in range(cols):
                label = f"{chr(65 + r)}{c + 1}"
                cx = int((c + 0.5) * cell_w)
                cy = int((r + 0.1) * cell_h) + 10
                vis.draw_text(label, (cx, cy), color=color, font_size=12, horizontal_alignment="center")
        return _image_to_content(vis.get_output().get_image())

    return _image_to_content(np.array(img))


@app.tool(name="draw_circle", description="Draw a circle on an image. Coordinates use 0-1000 scale. Radius is also 0-1000 (relative to width). Example: draw_circle(img, 500, 500, 50) draws a circle at center.")
def draw_circle(
    image_path: str,
    x: int,
    y: int,
    radius: int = 30,
    color: str = "red",
) -> ImageContent:
    """
    Draw a circle on an image.

    :param image_path: Path to the input image.
    :param x: X coordinate of the center (0-1000).
    :param y: Y coordinate of the center (0-1000).
    :param radius: Radius of the circle (0-1000, relative to image width).
    :param color: Color name.
    :return: Image with circle drawn.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    px, py = _to_pixel(x, w), _to_pixel(y, h)
    pr = _to_pixel(radius, w)
    vis = Visualizer(img)
    vis.draw_circle((px, py), color=color, radius=pr)
    return _image_to_content(vis.get_output().get_image())


@app.tool(name="draw_line", description="Draw a line on an image. Coordinates use 0-1000 scale. Example: draw_line(img, 200, 300, 800, 700) draws a diagonal line.")
def draw_line(
    image_path: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "red",
    linewidth: float = 3,
) -> ImageContent:
    """
    Draw a line on an image.

    :param image_path: Path to the input image.
    :param x1: Starting x coordinate (0-1000).
    :param y1: Starting y coordinate (0-1000).
    :param x2: Ending x coordinate (0-1000).
    :param y2: Ending y coordinate (0-1000).
    :param color: Color name.
    :param linewidth: Width of the line in pixels.
    :return: Image with line drawn.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    px1, py1 = _to_pixel(x1, w), _to_pixel(y1, h)
    px2, py2 = _to_pixel(x2, w), _to_pixel(y2, h)
    vis = Visualizer(img)
    vis.draw_line([px1, px2], [py1, py2], color=color, linewidth=linewidth)
    return _image_to_content(vis.get_output().get_image())


@app.tool(name="draw_arrow", description="Draw an arrow from one point to another. Coordinates use 0-1000 scale. Example: draw_arrow(img, 200, 300, 800, 700) draws an arrow pointing toward (800, 700).")
def draw_arrow(
    image_path: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "red",
    linewidth: int = 3,
    head_size: int = 20,
) -> ImageContent:
    """
    Draw an arrow from (x1, y1) to (x2, y2) with an arrowhead.

    :param image_path: Path to the input image.
    :param x1: Starting x coordinate (0-1000).
    :param y1: Starting y coordinate (0-1000).
    :param x2: Ending x coordinate (0-1000) - arrow points here.
    :param y2: Ending y coordinate (0-1000) - arrow points here.
    :param color: Color name.
    :param linewidth: Width of the arrow line in pixels.
    :param head_size: Size of the arrowhead in pixels.
    :return: Image with arrow drawn.
    """
    import math
    from PIL import ImageDraw

    img = Image.open(image_path).convert("RGB")
    w, h = img.size
    draw = ImageDraw.Draw(img)

    px1, py1 = int(_to_pixel(x1, w)), int(_to_pixel(y1, h))
    px2, py2 = int(_to_pixel(x2, w)), int(_to_pixel(y2, h))

    # Draw main line
    draw.line([(px1, py1), (px2, py2)], fill=color, width=linewidth)

    # Calculate arrowhead
    angle = math.atan2(py2 - py1, px2 - px1)
    head_angle = math.pi / 6  # 30 degrees

    # Arrowhead points
    ax1 = px2 - head_size * math.cos(angle - head_angle)
    ay1 = py2 - head_size * math.sin(angle - head_angle)
    ax2 = px2 - head_size * math.cos(angle + head_angle)
    ay2 = py2 - head_size * math.sin(angle + head_angle)

    # Draw arrowhead
    draw.polygon([(px2, py2), (ax1, ay1), (ax2, ay2)], fill=color)

    return _image_to_content(np.array(img))


@app.tool(name="draw_text", description="Add text annotation to an image. Coordinates use 0-1000 scale. Example: draw_text(img, 500, 100, 'Title') draws text at top center.")
def draw_text(
    image_path: str,
    x: int,
    y: int,
    text: str,
    color: str = "red",
    font_size: int = 18,
) -> ImageContent:
    """
    Add text to an image at the specified position.

    :param image_path: Path to the input image.
    :param x: X coordinate for the text (0-1000).
    :param y: Y coordinate for the text (0-1000).
    :param text: The text to draw.
    :param color: Text color name.
    :param font_size: Size of the font in pixels.
    :return: Image with text drawn.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    px, py = _to_pixel(x, w), _to_pixel(y, h)
    vis = Visualizer(img)
    vis.draw_text(text, (px, py), color=color, font_size=font_size)
    return _image_to_content(vis.get_output().get_image())


@app.tool(name="draw_polygon", description="Draw a polygon on an image. Coordinates use 0-1000 scale. Example: draw_polygon(img, [[200,200], [800,200], [500,800]]) draws a triangle.")
def draw_polygon(
    image_path: str,
    points: List[List[int]],
    color: str = "red",
    alpha: float = 0.5,
) -> ImageContent:
    """
    Draw a polygon on an image.

    :param image_path: Path to the input image.
    :param points: List of [x, y] coordinates (0-1000) forming the polygon vertices.
    :param color: Color name.
    :param alpha: Opacity of the fill.
    :return: Image with polygon drawn.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    h, w = img.shape[:2]
    pixel_points = [[_to_pixel(p[0], w), _to_pixel(p[1], h)] for p in points]
    vis = Visualizer(img)
    vis.draw_polygon(np.array(pixel_points), color=color, alpha=alpha)
    return _image_to_content(vis.get_output().get_image())


@app.tool(name="draw_mask", description="Draw a binary mask overlay on an image. Use this to highlight segmented regions.")
def draw_mask(
    image_path: str,
    mask_path: str,
    color: str = "blue",
    alpha: float = 0.5,
    text: str = None,
) -> ImageContent:
    """
    Draw a binary mask on an image.

    :param image_path: Path to the input image.
    :param mask_path: Path to the binary mask image (white=foreground).
    :param color: Color name for the mask overlay.
    :param alpha: Opacity of the mask.
    :param text: Optional text label to draw on the mask.
    :return: Image with mask drawn.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    vis = Visualizer(img)
    vis.draw_binary_mask(mask.astype(np.uint8), color=color, alpha=alpha, text=text)
    return _image_to_content(vis.get_output().get_image())


@app.tool(name="draw_numbered_mask", description="Draw a binary mask with a SoM-style numbered label at its center. Use this to mark and reference specific regions.")
def draw_numbered_mask(
    image_path: str,
    mask_path: str,
    number: int,
    color: str = None,
    alpha: float = 0.3,
    label_mode: str = "1",
    anno_mode: List[str] = ["Mask", "Mark"],
) -> ImageContent:
    """
    Draw a binary mask with a numbered label (Set-of-Mark style).

    :param image_path: Path to the input image.
    :param mask_path: Path to the binary mask image.
    :param number: The number to display on the mask.
    :param color: Color name (random if None).
    :param alpha: Opacity of the mask.
    :param label_mode: '1' for numbers, 'a' for letters.
    :param anno_mode: List containing 'Mask', 'Box', and/or 'Mark'.
    :return: Image with numbered mask drawn.
    """
    img = np.array(Image.open(image_path).convert("RGB"))
    mask = np.array(Image.open(mask_path).convert("L")) > 127
    vis = Visualizer(img)
    vis.draw_binary_mask_with_number(
        mask.astype(np.uint8),
        color=color,
        text=str(number),
        label_mode=label_mode,
        alpha=alpha,
        anno_mode=anno_mode,
    )
    return _image_to_content(vis.get_output().get_image())


@app.tool(name="image_crop", description="Crop a region from an image to zoom in. Coordinates use 0-1000 scale. Example: image_crop(img, 250, 250, 750, 750) crops the center 50%.")
def image_crop(
    image_path: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
) -> ImageContent:
    """
    Crop a region from an image.

    :param image_path: Path to the input image.
    :param x1: Left x coordinate (0-1000).
    :param y1: Top y coordinate (0-1000).
    :param x2: Right x coordinate (0-1000).
    :param y2: Bottom y coordinate (0-1000).
    :return: Cropped image.
    """
    image = Image.open(image_path)
    w, h = image.size
    px1, py1 = int((x1 / COORD_SCALE) * w), int((y1 / COORD_SCALE) * h)
    px2, py2 = int((x2 / COORD_SCALE) * w), int((y2 / COORD_SCALE) * h)
    cropped = image.crop([px1, py1, px2, py2])
    image_bytes = io.BytesIO()
    cropped.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    data = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return ImageContent(type="image", data=data, mimeType="image/png")


@app.tool(name="image_resize", description="Resize an image by a scale factor. Use this to see more details.")
def image_resize(image_path: str, scale: float = 2.0) -> ImageContent:
    """
    Resize an image by a scale factor.

    :param image_path: Path to the input image.
    :param scale: Scale factor (e.g., 2.0 = double the size).
    :return: Resized image.
    """
    image = Image.open(image_path)
    new_width = int(image.width * scale)
    new_height = int(image.height * scale)
    resized = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
    image_bytes = io.BytesIO()
    resized.save(image_bytes, format="PNG")
    image_bytes.seek(0)
    data = base64.b64encode(image_bytes.getvalue()).decode("utf-8")
    return ImageContent(type="image", data=data, mimeType="image/png")


# ========== Vision Foundation Model Tools ==========

# Lazy-loaded models
_depth_model = None
_sam_model = None
_sam_predictor = None
_grounding_dino_model = None
_grounding_dino_processor = None


def _get_depth_model():
    """Lazy load Depth Anything model."""
    global _depth_model
    if _depth_model is None:
        from transformers import pipeline, AutoModelForDepthEstimation, AutoImageProcessor
        logger.info("Loading Depth Anything model...")
        local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        # Use cuda:1 to avoid vLLM on cuda:0
        device = os.environ.get("DEPTH_DEVICE", "cuda:1")
        model_name = "depth-anything/Depth-Anything-V2-Small-hf"

        model = AutoModelForDepthEstimation.from_pretrained(
            model_name, local_files_only=local_only
        ).to(device)
        processor = AutoImageProcessor.from_pretrained(
            model_name, local_files_only=local_only
        )
        _depth_model = pipeline(
            task="depth-estimation",
            model=model,
            image_processor=processor,
            device=device,
        )
        logger.info(f"Depth Anything model loaded on {device}.")
    return _depth_model


def _get_sam_predictor():
    """Lazy load SAM model."""
    global _sam_model, _sam_predictor
    if _sam_predictor is None:
        from sam2.sam2_image_predictor import SAM2ImagePredictor
        logger.info("Loading SAM2 model...")
        # Use local_files_only if HF_HUB_OFFLINE is set
        local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        # Use cuda:1 to avoid vLLM on cuda:0
        device = os.environ.get("SAM_DEVICE", "cuda:1")
        _sam_predictor = SAM2ImagePredictor.from_pretrained(
            "facebook/sam2-hiera-large",
            local_files_only=local_only,
            device=device,
        )
        logger.info(f"SAM2 model loaded on {device}.")
    return _sam_predictor


def _get_grounding_dino():
    """Lazy load GroundingDINO model for text-prompted object detection."""
    global _grounding_dino_model, _grounding_dino_processor
    if _grounding_dino_model is None:
        from transformers import AutoProcessor, GroundingDinoForObjectDetection
        logger.info("Loading GroundingDINO model...")
        local_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
        # Support cuda:0, cuda:1, cpu, etc. Default to cuda:1 to avoid vLLM GPU
        device = os.environ.get("GROUNDING_DINO_DEVICE", "cuda:1")
        model_name = "IDEA-Research/grounding-dino-base"
        _grounding_dino_processor = AutoProcessor.from_pretrained(
            model_name,
            local_files_only=local_only,
        )
        _grounding_dino_model = GroundingDinoForObjectDetection.from_pretrained(
            model_name,
            local_files_only=local_only,
        ).to(device)
        _grounding_dino_model.eval()
        logger.info(f"GroundingDINO model loaded on {device}.")
    return _grounding_dino_model, _grounding_dino_processor


@app.tool(name="detect_objects", description="Detect objects matching a text description using GroundingDINO. Returns image with bounding boxes around detected objects and count. Use this to find specific objects like 'yellow flower' or 'person with red shirt'.")
def detect_objects(
    image_path: str,
    prompt: str,
    threshold: float = 0.3,
) -> ImageContent:
    """
    Detect objects matching a text description.

    :param image_path: Path to the input image.
    :param prompt: Text description of objects to detect (e.g., 'yellow flower', 'cat', 'person').
    :param threshold: Confidence threshold for detection (0-1).
    :return: Image with bounding boxes and detection count.
    """
    import torch

    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    model, processor = _get_grounding_dino()
    device = next(model.parameters()).device

    inputs = processor(images=image, text=prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_grounded_object_detection(
        outputs,
        inputs["input_ids"],
        target_sizes=[(h, w)],
    )[0]

    boxes = results["boxes"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    # Use text_labels for string labels (labels will return int ids in future versions)
    labels = results.get("text_labels", results.get("labels", []))

    # Filter by threshold
    mask = scores >= threshold
    boxes = boxes[mask]
    scores = scores[mask]
    labels = [l for l, m in zip(labels, mask) if m]

    vis = Visualizer(img_array)
    for i, (box, score, label) in enumerate(zip(boxes, scores, labels)):
        x1, y1, x2, y2 = box
        color = "green" if score > 0.5 else "yellow"
        vis.draw_box([x1, y1, x2, y2], edge_color=color, alpha=0.8)
        vis.draw_text(f"{i+1}: {label} ({score:.2f})", (x1, y1 - 5), color=color, font_size=14)

    result_img = vis.get_output().get_image()

    count = len(boxes)
    vis2 = Visualizer(result_img)
    vis2.draw_text(f"Found {count} '{prompt}'", (10, 10), color="white", font_size=18)

    return _image_to_content(vis2.get_output().get_image())


@app.tool(name="estimate_depth", description="Estimate depth map from an image using Depth Anything. Returns depth visualization where lower value/darker = closer, higher value/brighter = farther.")
def estimate_depth(
    image_path: str,
    colormap: str = "inferno",
) -> ImageContent:
    """
    Estimate depth map from an image.

    :param image_path: Path to the input image.
    :param colormap: Matplotlib colormap for visualization ('inferno', 'viridis', 'plasma', 'gray').
    :return: Depth map visualization.
    """
    import matplotlib.cm as cm

    image = Image.open(image_path).convert("RGB")
    depth_model = _get_depth_model()

    # Run depth estimation
    result = depth_model(image)
    depth = np.array(result["depth"])

    # Normalize and invert to match convention (lower value = closer)
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_inverted = 1.0 - depth_normalized  # Now lower = closer

    cmap = cm.get_cmap(colormap)
    depth_colored = (cmap(depth_inverted)[:, :, :3] * 255).astype(np.uint8)

    return _image_to_content(depth_colored)


@app.tool(name="estimate_depth_at_points", description="Estimate depth and mark two points A and B on the depth map. Lower depth value = closer to camera. Coordinates use 0-1000 scale.")
def estimate_depth_at_points(
    image_path: str,
    point_a_x: int,
    point_a_y: int,
    point_b_x: int,
    point_b_y: int,
    colormap: str = "inferno",
) -> ImageContent:
    """
    Estimate depth and mark points A and B on the depth map.

    :param image_path: Path to the input image.
    :param point_a_x: X coordinate of point A (0-1000 scale).
    :param point_a_y: Y coordinate of point A (0-1000 scale).
    :param point_b_x: X coordinate of point B (0-1000 scale).
    :param point_b_y: Y coordinate of point B (0-1000 scale).
    :param colormap: Matplotlib colormap for visualization.
    :return: Depth map with marked points and their depth values. Lower value = closer to camera.
    """
    import matplotlib.cm as cm

    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    depth_model = _get_depth_model()

    result = depth_model(image)
    depth = np.array(result["depth"])

    if depth.shape[:2] != (h, w):
        depth_pil = Image.fromarray(depth)
        depth_pil = depth_pil.resize((w, h), Image.Resampling.BILINEAR)
        depth = np.array(depth_pil)

    # Normalize and invert to match convention (lower value = closer)
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    depth_inverted = 1.0 - depth_normalized  # Now lower = closer

    cmap = cm.get_cmap(colormap)
    depth_colored = (cmap(depth_inverted)[:, :, :3] * 255).astype(np.uint8)

    vis = Visualizer(depth_colored)

    # Collect depth values for summary
    depth_values = {}
    points = [(point_a_x, point_a_y, "A"), (point_b_x, point_b_y, "B")]
    for px_norm, py_norm, label in points:
        px = int(_to_pixel(px_norm, w))
        py = int(_to_pixel(py_norm, h))
        depth_val = depth_inverted[min(py, h - 1), min(px, w - 1)]
        depth_values[label] = depth_val
        vis.draw_circle((px, py), color="white", radius=15)
        vis.draw_circle((px, py), color="black", radius=12)
        vis.draw_text(f"{label}", (px, py - 20), color="white", font_size=20)

    # Add summary at bottom showing both values
    a_val, b_val = depth_values["A"], depth_values["B"]
    vis.draw_text(f"A: {a_val:.2f}  B: {b_val:.2f}", (w // 2, h - 20), color="yellow", font_size=18, horizontal_alignment="center")

    return _image_to_content(vis.get_output().get_image())


@app.tool(name="estimate_depth_overlay", description="Overlay depth estimation on the original image. Useful to see depth context with the scene.")
def estimate_depth_overlay(
    image_path: str,
    alpha: float = 0.5,
    colormap: str = "inferno",
) -> ImageContent:
    """
    Overlay depth map on original image.

    :param image_path: Path to the input image.
    :param alpha: Opacity of the depth overlay (0-1).
    :param colormap: Matplotlib colormap for depth visualization.
    :return: Image with depth overlay.
    """
    import matplotlib.cm as cm

    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    depth_model = _get_depth_model()

    # Run depth estimation
    result = depth_model(image)
    depth = np.array(result["depth"])

    # Resize depth to match image if needed
    if depth.shape[:2] != img_array.shape[:2]:
        depth_pil = Image.fromarray(depth)
        depth_pil = depth_pil.resize((img_array.shape[1], img_array.shape[0]), Image.Resampling.BILINEAR)
        depth = np.array(depth_pil)

    # Normalize and colorize depth
    depth_normalized = (depth - depth.min()) / (depth.max() - depth.min() + 1e-8)
    cmap = cm.get_cmap(colormap)
    depth_colored = (cmap(depth_normalized)[:, :, :3] * 255).astype(np.uint8)

    # Blend with original
    blended = (img_array * (1 - alpha) + depth_colored * alpha).astype(np.uint8)

    return _image_to_content(blended)


@app.tool(name="segment_point", description="Segment an object at a specific point using SAM. Coordinates use 0-1000 scale. Returns the image with the segmented region highlighted.")
def segment_point(
    image_path: str,
    x: int,
    y: int,
    color: str = "blue",
    alpha: float = 0.5,
) -> ImageContent:
    """
    Segment the object at a specific point.

    :param image_path: Path to the input image.
    :param x: X coordinate of the point (0-1000).
    :param y: Y coordinate of the point (0-1000).
    :param color: Color for the segmentation mask.
    :param alpha: Opacity of the mask overlay.
    :return: Image with segmentation mask.
    """
    import matplotlib.colors as mcolors

    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Convert coordinates
    px = int(_to_pixel(x, w))
    py = int(_to_pixel(y, h))

    # Get SAM predictor
    predictor = _get_sam_predictor()
    predictor.set_image(img_array)

    # Predict mask
    point_coords = np.array([[px, py]])
    point_labels = np.array([1])  # 1 = foreground
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=True,
    )

    # Use highest scoring mask
    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    # Create colored overlay
    rgb_color = np.array(mcolors.to_rgb(color)) * 255
    overlay = img_array.copy().astype(np.float32)
    overlay[mask] = overlay[mask] * (1 - alpha) + rgb_color * alpha
    overlay = overlay.astype(np.uint8)

    # Draw the point
    vis = Visualizer(overlay)
    vis.draw_circle((px, py), color="red", radius=10)

    return _image_to_content(vis.get_output().get_image())


@app.tool(name="segment_box", description="Segment objects within a bounding box using SAM. Coordinates use 0-1000 scale.")
def segment_box(
    image_path: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    color: str = "blue",
    alpha: float = 0.5,
) -> ImageContent:
    """
    Segment objects within a bounding box.

    :param image_path: Path to the input image.
    :param x1: Left x coordinate (0-1000).
    :param y1: Top y coordinate (0-1000).
    :param x2: Right x coordinate (0-1000).
    :param y2: Bottom y coordinate (0-1000).
    :param color: Color for the segmentation mask.
    :param alpha: Opacity of the mask overlay.
    :return: Image with segmentation mask.
    """
    import matplotlib.colors as mcolors

    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)
    h, w = img_array.shape[:2]

    # Convert coordinates
    px1, py1 = int(_to_pixel(x1, w)), int(_to_pixel(y1, h))
    px2, py2 = int(_to_pixel(x2, w)), int(_to_pixel(y2, h))

    # Get SAM predictor
    predictor = _get_sam_predictor()
    predictor.set_image(img_array)

    # Predict mask with box prompt
    box = np.array([px1, py1, px2, py2])
    masks, scores, _ = predictor.predict(
        box=box,
        multimask_output=True,
    )

    # Use highest scoring mask
    best_idx = np.argmax(scores)
    mask = masks[best_idx]

    # Create colored overlay
    rgb_color = np.array(mcolors.to_rgb(color)) * 255
    overlay = img_array.copy().astype(np.float32)
    overlay[mask] = overlay[mask] * (1 - alpha) + rgb_color * alpha
    overlay = overlay.astype(np.uint8)

    # Draw the box
    vis = Visualizer(overlay)
    vis.draw_box([px1, py1, px2, py2], edge_color="red", alpha=0.8)

    return _image_to_content(vis.get_output().get_image())


@app.tool(name="segment_all", description="Segment all objects in an image using SAM automatic mask generation. Returns image with all segments colored. Use this to see all distinct objects/regions in the image.")
def segment_all(
    image_path: str,
    min_area: int = 100,
    max_masks: int = 30,
    alpha: float = 0.5,
) -> ImageContent:
    """
    Automatically segment all objects in an image.

    :param image_path: Path to the input image.
    :param min_area: Minimum mask area in pixels to include.
    :param max_masks: Maximum number of masks to show.
    :param alpha: Opacity of the mask overlay (0-1).
    :return: Image with all segments visualized with distinct colors.
    """
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
    import matplotlib.colors as mcolors

    image = Image.open(image_path).convert("RGB")
    img_array = np.array(image)

    # Get SAM model and create mask generator
    predictor = _get_sam_predictor()
    mask_generator = SAM2AutomaticMaskGenerator(predictor.model)

    # Generate masks
    masks = mask_generator.generate(img_array)

    # Filter and sort by area
    masks = [m for m in masks if m["area"] >= min_area]
    masks = sorted(masks, key=lambda x: x["area"], reverse=True)[:max_masks]

    # Visualize with distinct colors (no labels)
    vis = Visualizer(img_array)
    colors = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())
    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]
        color = colors[i % len(colors)]
        vis.draw_binary_mask(
            mask.astype(np.uint8),
            color=color,
            alpha=alpha,
        )

    return _image_to_content(vis.get_output().get_image())


if __name__ == "__main__":
    app.run()
