"""Torch-free distillation adapter for the SketchVLM ball-drop task.

Standard adapter interface used by the generic harness (collect_traces.py):
doc_to_text / doc_to_visual / doc_id / score. ball-drop's own utils.py is already
dependency-light, so this just re-exports it under the uniform names and adds a
score() that wraps parse_bucket.
"""
from __future__ import annotations

import importlib.util
import os
from typing import Any, Dict

_UTILS = os.path.join(os.path.dirname(__file__), "utils.py")
_spec = importlib.util.spec_from_file_location("ball_drop_utils_for_adapter", _UTILS)
_u = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_u)

DATASET_PATH = "loganbolton/sketchvlm-physics-ball-drop"
DATASET_SPLIT = "train"


def doc_to_visual(doc):
    return _u.ball_drop_doc_to_visual(doc)


def doc_to_text(doc, lmms_eval_specific_kwargs=None):
    return _u.ball_drop_doc_to_text(doc, lmms_eval_specific_kwargs)


def doc_id(doc, index=0):
    return _u.ball_drop_doc_id(doc, index)


def score(doc: Dict[str, Any], final_text: str) -> Dict[str, Any]:
    pred = _u.parse_bucket(final_text)
    gt = int(doc["final_bucket_gt"])
    return {"pred": pred, "gt": gt, "correct": bool(pred is not None and pred == gt)}
