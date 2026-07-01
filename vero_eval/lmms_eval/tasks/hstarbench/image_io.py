"""Robust image loading for hstarbench (cv2 + PIL fallback, in-memory ERP cache)."""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Dict, Iterator, Optional, Union

import cv2
import numpy as np
from PIL import Image

# Keep a handle to the real cv2.imread; patched_hstar_cv2_imread replaces cv2.imread on the
# shared cv2 module object, so load_bgr must not call the patched entry point.
_ORIGINAL_CV2_IMREAD = cv2.imread

_BGR_CACHE: Dict[str, np.ndarray] = {}


def load_rgb(path: Union[str, Path]) -> Optional[Image.Image]:
    p = Path(path)
    if not p.is_absolute():
        return None
    if not p.exists():
        return None
    try:
        with Image.open(p) as im:
            return im.convert("RGB")
    except Exception:
        return None


def load_bgr(path: Union[str, Path], use_cache: bool = True) -> Optional[np.ndarray]:
    key = str(Path(path).resolve())
    if use_cache and key in _BGR_CACHE:
        return _BGR_CACHE[key]

    bgr = _ORIGINAL_CV2_IMREAD(key, cv2.IMREAD_COLOR)
    if bgr is None:
        rgb = load_rgb(key)
        if rgb is None:
            return None
        bgr = cv2.cvtColor(np.asarray(rgb), cv2.COLOR_RGB2BGR)

    if use_cache:
        _BGR_CACHE[key] = bgr
    return bgr


def _patched_imread(path: str, flags: int = cv2.IMREAD_COLOR) -> Optional[np.ndarray]:
    if flags not in (cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED, cv2.IMREAD_ANYCOLOR):
        return _ORIGINAL_CV2_IMREAD(path, flags)
    return load_bgr(path)


@contextlib.contextmanager
def patched_hstar_cv2_imread() -> Iterator[None]:
    """Patch cv2.imread inside vagen HstarEnv so ERP reads use PIL fallback + cache."""
    import vagen.env.hstar.env as hstar_env

    def _wrapped(path, flags=cv2.IMREAD_COLOR):
        if flags in (cv2.IMREAD_COLOR, cv2.IMREAD_UNCHANGED, cv2.IMREAD_ANYCOLOR):
            out = load_bgr(path)
            if out is not None:
                return out
        return _ORIGINAL_CV2_IMREAD(path, flags)

    previous_imread = hstar_env.cv2.imread
    hstar_env.cv2.imread = _wrapped
    try:
        yield
    finally:
        hstar_env.cv2.imread = previous_imread
