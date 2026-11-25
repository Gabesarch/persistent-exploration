"""In-loop validation reward for hstarbench panorama-strip bbox answers."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional


def _score_final_answer_bbox(doc: Dict[str, Any], response_text: str):
    repo_root = Path(__file__).resolve().parents[3]
    task_dir = repo_root / "persistent-exploration-eval/lmms_eval/tasks/hstarbench"
    if str(task_dir) not in sys.path:
        sys.path.insert(0, str(task_dir))
    from scoring import score_final_answer_bbox  # noqa: WPS433

    return score_final_answer_bbox(doc, response_text)


def _parse_ground_truth(ground_truth: str, extra_info: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    doc: Dict[str, Any] = {}
    if isinstance(extra_info, dict):
        doc.update(extra_info)
    if ground_truth:
        try:
            parsed = json.loads(ground_truth)
            if isinstance(parsed, dict):
                doc.update(parsed)
        except json.JSONDecodeError:
            pass
    return doc


def hstar_strip_reward(
    predict_str: str,
    ground_truth: str,
    *,
    extra_info: Optional[Dict[str, Any]] = None,
    iou_weight: float = 0.0,
) -> float:
    """Primary metric = cone_success; optional IoU blend for diagnostics."""
    doc = _parse_ground_truth(ground_truth, extra_info)
    if not doc.get("target_yaw"):
        doc["target_yaw"] = (extra_info or {}).get("target_yaw")
    if not doc.get("target_pitch"):
        doc["target_pitch"] = (extra_info or {}).get("target_pitch")
    outcome = _score_final_answer_bbox(doc, predict_str)
    cone = 1.0 if outcome.get("cone_success") else 0.0
    if iou_weight <= 0:
        return cone
    iou = 1.0 if outcome.get("iou_success") else 0.0
    return (1.0 - iou_weight) * cone + iou_weight * iou


def compute_score(
    predict_str: str,
    ground_truth: str,
    use_boxed: bool = False,  # noqa: ARG001
    extra_info: Optional[Dict[str, Any]] = None,
) -> float:
    return hstar_strip_reward(predict_str, ground_truth, extra_info=extra_info)
