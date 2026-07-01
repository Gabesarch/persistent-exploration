"""ERP → iPhone-style horizontal cylindrical panorama strip for zoom-in tool."""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
from PIL import Image

# Match ERP horizontal density (5760/360 = 16 px/deg); vertical ~110° cylindrical band at ~16 px/deg
DEFAULT_STRIP_WIDTH = int(os.environ.get("HSTARBENCH_STRIP_WIDTH", os.environ.get("THINKIN360_STRIP_WIDTH", "5760")))
DEFAULT_STRIP_HEIGHT = int(os.environ.get("HSTARBENCH_STRIP_HEIGHT", os.environ.get("THINKIN360_STRIP_HEIGHT", "1792")))
# Vertical extent on cylinder (degrees from horizon); iPhone panos are ~50–60° tall
DEFAULT_VERTICAL_HALF_FOV_DEG = float(os.environ.get("HSTARBENCH_STRIP_VFOV_DEG", os.environ.get("THINKIN360_STRIP_VFOV_DEG", "55")))
# hstar annotation yaw is opposite the panorama-strip horizontal axis (empirically +180°).
DEFAULT_STRIP_YAW_OFFSET_DEG = float(os.environ.get("HSTARBENCH_STRIP_YAW_OFFSET_DEG", os.environ.get("THINKIN360_STRIP_YAW_OFFSET_DEG", "180")))


def annotation_yaw_to_strip_yaw(yaw_deg: float) -> float:
    return (float(yaw_deg) + DEFAULT_STRIP_YAW_OFFSET_DEG) % 360.0


def strip_yaw_to_annotation_yaw(yaw_deg: float) -> float:
    return (float(yaw_deg) - DEFAULT_STRIP_YAW_OFFSET_DEG) % 360.0


def erp_bgr_to_strip_bgr(
    erp_bgr: np.ndarray,
    out_width: int = DEFAULT_STRIP_WIDTH,
    out_height: int = DEFAULT_STRIP_HEIGHT,
    vertical_half_fov_deg: float = DEFAULT_VERTICAL_HALF_FOV_DEG,
) -> np.ndarray:
    """Unwrap equirectangular 360° ERP to a single wide cylindrical strip (iPhone-like).

    Horizontal axis: yaw 0° → 360° (left to right, seamless wrap).
    Vertical axis: pitch band around horizon (cylindrical, less pole stretch than ERP).
    """
    if erp_bgr is None or erp_bgr.size == 0:
        raise ValueError("empty ERP image")
    h_erp, w_erp = erp_bgr.shape[:2]

    xs = np.arange(out_width, dtype=np.float32)
    ys = np.arange(out_height, dtype=np.float32)
    x_grid, y_grid = np.meshgrid(xs, ys)

    # Longitude 0 .. 2π (full 360°)
    lon = (x_grid / float(out_width)) * (2.0 * np.pi)
    # Cylindrical height → latitude
    v_max = np.tan(np.deg2rad(vertical_half_fov_deg))
    v_norm = (0.5 - y_grid / float(out_height)) * 2.0
    lat = np.arctan(v_norm * v_max)

    u = (lon / (2.0 * np.pi)) * w_erp
    v = (0.5 - lat / np.pi) * h_erp
    u = u.astype(np.float32)
    v = v.astype(np.float32)

    return cv2.remap(
        erp_bgr,
        u,
        v,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def erp_to_strip_pil(
    erp: Union[Image.Image, np.ndarray, str, Path],
    out_width: int = DEFAULT_STRIP_WIDTH,
    out_height: int = DEFAULT_STRIP_HEIGHT,
    start_yaw_deg: Optional[float] = None,
    vertical_half_fov_deg: float = DEFAULT_VERTICAL_HALF_FOV_DEG,
) -> Image.Image:
    """Load ERP, convert to RGB strip; optionally mark initial heading."""
    if isinstance(erp, (str, Path)):
        from image_io import load_bgr

        bgr = load_bgr(str(erp))
        if bgr is None:
            raise FileNotFoundError(f"cannot read ERP: {erp}")
    elif isinstance(erp, Image.Image):
        bgr = cv2.cvtColor(np.array(erp.convert("RGB")), cv2.COLOR_RGB2BGR)
    else:
        arr = np.asarray(erp)
        if arr.ndim == 2:
            bgr = cv2.cvtColor(arr, cv2.COLOR_GRAY2BGR)
        elif arr.shape[2] == 4:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGBA2BGR)
        else:
            bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

    strip_bgr = erp_bgr_to_strip_bgr(
        bgr,
        out_width=out_width,
        out_height=out_height,
        vertical_half_fov_deg=vertical_half_fov_deg,
    )
    if start_yaw_deg is not None:
        x = int((float(start_yaw_deg) % 360.0) / 360.0 * out_width)
        cv2.line(strip_bgr, (x, 0), (x, out_height - 1), (0, 255, 0), 2)
        cv2.putText(
            strip_bgr,
            "start",
            (max(4, x - 20), 24),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )
    rgb = cv2.cvtColor(strip_bgr, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def save_strip(
    erp_path: Union[str, Path],
    out_path: Union[str, Path],
    start_yaw_deg: Optional[float] = None,
    **kwargs,
) -> Path:
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    strip = erp_to_strip_pil(erp_path, start_yaw_deg=start_yaw_deg, **kwargs)
    strip.save(out_path, quality=92)
    return out_path


def pitch_to_strip_cy(
    pitch_deg: float,
    vertical_half_fov_deg: float = DEFAULT_VERTICAL_HALF_FOV_DEG,
) -> float:
    """Inverse of bbox_center_to_yaw_pitch: pitch degrees → strip y in 0–1000 coords."""
    v_max = np.tan(np.deg2rad(vertical_half_fov_deg))
    v_norm = np.tan(np.deg2rad(float(pitch_deg))) / v_max
    return float(500.0 - v_norm * 500.0)


def yaw_to_strip_x(yaw_deg: float, *, annotation_space: bool = True) -> float:
    y = annotation_yaw_to_strip_yaw(yaw_deg) if annotation_space else float(yaw_deg) % 360.0
    return max(0.0, min(1000.0, y / 360.0 * 1000.0))


def _clamp_strip_bbox(x1: float, y1: float, x2: float, y2: float) -> Tuple[float, float, float, float]:
    bx1, bx2 = sorted((x1, x2))
    by1, by2 = sorted((y1, y2))
    return (
        max(0.0, min(1000.0, bx1)),
        max(0.0, min(1000.0, by1)),
        max(0.0, min(1000.0, bx2)),
        max(0.0, min(1000.0, by2)),
    )


def target_cone_to_strip_bboxes(
    target_yaw: Sequence[float],
    target_pitch: Sequence[float],
    vertical_half_fov_deg: float = DEFAULT_VERTICAL_HALF_FOV_DEG,
) -> List[Tuple[float, float, float, float]]:
    """Map target yaw/pitch acceptance cone to one or two strip bboxes (0–1000 xyxy).

    When yaw wraps across 0° (yaw_min > yaw_max), returns two horizontal segments.
    """
    y0, y1 = float(target_yaw[0]), float(target_yaw[1])
    p0, p1 = float(target_pitch[0]), float(target_pitch[1])
    cy_hi = pitch_to_strip_cy(p1, vertical_half_fov_deg)
    cy_lo = pitch_to_strip_cy(p0, vertical_half_fov_deg)
    y_top, y_bot = min(cy_hi, cy_lo), max(cy_hi, cy_lo)

    # Map annotation yaw cone to strip x; handle wrap in annotation space first.
    s0 = annotation_yaw_to_strip_yaw(y0)
    s1 = annotation_yaw_to_strip_yaw(y1)
    if s0 <= s1:
        return [_clamp_strip_bbox(yaw_to_strip_x(s0, annotation_space=False), y_top, yaw_to_strip_x(s1, annotation_space=False), y_bot)]

    return [
        _clamp_strip_bbox(yaw_to_strip_x(s0, annotation_space=False), y_top, 1000.0, y_bot),
        _clamp_strip_bbox(0.0, y_top, yaw_to_strip_x(s1, annotation_space=False), y_bot),
    ]


def bbox_center_to_yaw_pitch(
    bbox_2d_0_1000: Tuple[float, float, float, float],
    vertical_half_fov_deg: float = DEFAULT_VERTICAL_HALF_FOV_DEG,
) -> Tuple[float, float]:
    """Map zoom bbox center on strip (0–1000 coords) to approximate yaw/pitch degrees."""
    x1, y1, x2, y2 = bbox_2d_0_1000
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    strip_yaw = (cx / 1000.0) * 360.0
    yaw = strip_yaw_to_annotation_yaw(strip_yaw)
    v_max = np.tan(np.deg2rad(vertical_half_fov_deg))
    v_norm = (500.0 - cy) / 500.0
    pitch = float(np.rad2deg(np.arctan(v_norm * v_max)))
    return yaw, pitch
