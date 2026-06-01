import logging
import re
from typing import Any, Dict, List, Optional

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer

from math_verify import parse as math_verify_parse
from math_verify.parser import StringExtractionConfig

eval_logger = logging.getLogger("lmms-eval")

_CHOICE_STRINGS = tuple([str(i) for i in range(1, 10)])


def balldrop_doc_to_visual(doc: Dict[str, Any]) -> List:
    img = doc.get("start_image")
    if img is None:
        return []
    return [img.convert("RGB")]


def balldrop_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = (
        "This is a physics simulation. A ball will be dropped and will fall due to gravity, "
        "potentially bouncing off obstacles. There are numbered buckets at the bottom. "
        "Which bucket will the ball land in?"
    )
    prompt = f"{pre_prompt}{question}{post_prompt}"
    return prompt.strip()


def _extract_choice_token(text: str) -> str:
    """
    Extract an option token from model output that references choices as numbers (1, 2, ...).
    Returns empty string if no clean option token is found.
    """
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = re.sub(r"^(answer|option|choice|bucket)\s*[:\-]?\s*", "", text, flags=re.IGNORECASE)

    if math_verify_parse and StringExtractionConfig:
        try:
            parsed = math_verify_parse(text, extraction_config=[StringExtractionConfig(strings=_CHOICE_STRINGS)])
        except Exception:
            parsed = None
        if parsed:
            parsed_choice = str(parsed[0]).strip()
            if parsed_choice and parsed_choice in _CHOICE_STRINGS:
                return parsed_choice

    number_match = re.search(r"\b([1-9])\b", text)
    if number_match:
        return number_match.group(1)

    return ""


def _strip_answer_prefix(text: str) -> str:
    if not isinstance(text, str):
        return text
    matches = list(re.finditer(r"\banswer\b", text, flags=re.IGNORECASE))
    if not matches:
        return text.strip()
    tail = text[matches[-1].end():].lstrip(" :-\t\r\n")
    return tail if tail else text.strip()


def balldrop_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict]:
    key_name = "balldrop_acc"
    grounded_output_raw = doc.get("final_bucket_gt", "")
    grounded_output_text = str(grounded_output_raw)
    response = results[0] if results else ""

    response = extract_final_answer(response)
    response = _strip_answer_prefix(response)

    pred_choice = _extract_choice_token(response)
    gt_choice = _extract_choice_token(grounded_output_text)

    is_correct = pred_choice != "" and gt_choice != "" and pred_choice == gt_choice

    submission = {
        "id": doc.get("run_id", ""),
        "gt_content": gt_choice or grounded_output_text,
        "pred_parsed": pred_choice,
        "pred": response,
        "is_correct": is_correct,
    }
    return {key_name: submission}


def balldrop_cot_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict]:
    key_name = "balldrop_acc"
    grounded_output_raw = doc.get("final_bucket_gt", "")
    grounded_output_text = str(grounded_output_raw)
    response = results[0] if results else ""

    response = extract_final_answer(response)
    response = _strip_answer_prefix(response)

    # For CoT, extract the last number mentioned
    matches = re.findall(r"\b([1-9])\b", response if isinstance(response, str) else "")
    pred_choice = matches[-1] if matches else ""
    gt_choice = _extract_choice_token(grounded_output_text)

    is_correct = pred_choice != "" and gt_choice != "" and pred_choice == gt_choice

    submission = {
        "id": doc.get("run_id", ""),
        "gt_content": gt_choice or grounded_output_text,
        "pred_parsed": pred_choice,
        "pred": response,
        "is_correct": is_correct,
    }
    return {key_name: submission}


def balldrop_aggregate_results(results: List[Dict]) -> float:
    total_samples = len(results)
    if total_samples == 0:
        return 0.0
    total_correct = sum(1 for sample in results if sample["is_correct"])
    return total_correct / total_samples
