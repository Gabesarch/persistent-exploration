import json
import logging
import os
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer
from lmms_eval.tasks._task_utils.response_truncation import truncate_response_tail_tiktoken
from lmms_eval.tasks._task_utils.vllm_judge import extract_json_candidate, get_judge_engine

eval_logger = logging.getLogger("lmms-eval")

MODEL_HINT = os.getenv("JUDGE_MODEL_PATH", "")
_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "judge_prompt.txt"


def vrt_doc_to_visual(doc: Dict[str, Any]) -> List:
    img = doc.get("image")
    if img is None:
        return []
    return [img.convert("RGB")]


def vrt_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    question = doc.get("question", "")
    prompt = f"{pre_prompt}{question}{post_prompt}"
    return prompt.strip()


def _get_ground_truth(doc: Dict[str, Any]) -> str:
    """Extract ground truth answer from answer objects' captions."""
    # First check if ground_truth_answer is pre-computed (from conversion)
    gt = doc.get("ground_truth_answer", "")
    if gt:
        return gt

    # Otherwise, compute from objects_info and answer_objects
    answer_objects_raw = doc.get("answer_objects", "[]")
    if isinstance(answer_objects_raw, str):
        try:
            answer_objects = json.loads(answer_objects_raw)
        except json.JSONDecodeError:
            answer_objects = []
    else:
        answer_objects = answer_objects_raw or []

    objects_info_raw = doc.get("objects_info", "{}")
    if isinstance(objects_info_raw, str):
        try:
            objects_info = json.loads(objects_info_raw)
        except json.JSONDecodeError:
            objects_info = {}
    else:
        objects_info = objects_info_raw or {}

    captions = []
    for obj_id in answer_objects:
        obj = objects_info.get(str(obj_id), {})
        caption = obj.get("caption", "")
        if caption:
            captions.append(f"Object {obj_id}: {caption}")

    return "\n".join(captions)


def vrt_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict]:
    """Process results and prepare payload for LLM judge."""
    raw_prediction = results[0] if results else ""

    prediction = extract_final_answer(raw_prediction, parse_boxed=False, strip_latex_wrappers=True)
    prediction = truncate_response_tail_tiktoken(prediction)

    question = doc.get("question", "")
    ground_truth = _get_ground_truth(doc)
    resp_key = doc.get("key", "")

    payload = {
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "resp_key": resp_key,
    }
    return {"vrt_acc": payload}


def vrt_cot_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, Dict]:
    """Process CoT results and prepare payload for LLM judge."""
    raw_prediction = results[0] if results else ""

    prediction = extract_final_answer(raw_prediction, parse_boxed=False, strip_latex_wrappers=True)
    prediction = truncate_response_tail_tiktoken(prediction)

    question = doc.get("question", "")
    ground_truth = _get_ground_truth(doc)
    resp_key = doc.get("key", "")

    payload = {
        "question": question,
        "ground_truth": ground_truth,
        "prediction": prediction,
        "raw_prediction": raw_prediction,
        "resp_key": resp_key,
    }
    return {"vrt_acc": payload}


@lru_cache(maxsize=1)
def _load_judge_prompt() -> str:
    with _PROMPT_PATH.open("r", encoding="utf-8") as handle:
        return handle.read()


def _format_judge_prompt(question: str, ground_truth: str, prediction: str) -> str:
    template = _load_judge_prompt()
    prompt = template.replace("{question}", question or "")
    prompt = prompt.replace("{ground_truth}", ground_truth or "")
    prompt = prompt.replace("{prediction}", prediction or "")
    return prompt


def _parse_judge_output(raw: Any) -> bool:
    if not isinstance(raw, str):
        raw = "" if raw is None else str(raw)
    raw_text = raw.strip()

    parsed = extract_json_candidate(raw_text)
    if isinstance(parsed, dict):
        for key in ("is_equivalent", "equivalent", "result", "answer", "decision"):
            if key not in parsed:
                continue
            value = parsed[key]
            if isinstance(value, bool):
                return bool(value)
            if isinstance(value, str):
                lowered = value.strip().lower()
                if lowered.startswith(("y", "t")):
                    return True
                if lowered.startswith(("n", "f")):
                    return False
        raw_text = str(parsed)

    lowered = raw_text.lower()
    if lowered in {"yes", '"yes"', "'yes'", "true"}:
        return True
    if lowered in {"no", '"no"', "'no'", "false"}:
        return False
    if lowered.startswith(("yes", "true")):
        return True
    if lowered.startswith(("no", "false")):
        return False
    if "yes" in lowered and "no" not in lowered:
        return True
    if "no" in lowered and "yes" not in lowered:
        return False
    return False


def vrt_aggregate_results(results: List[Dict[str, Any]]) -> float:
    """Aggregate results using LLM judge."""
    if not results:
        vrt_aggregate_results.details = []
        vrt_aggregate_results.individual_scores = {}
        return 0.0

    prompts: List[str] = []
    metadata: List[Dict[str, Any]] = []
    for item in results:
        prompts.append(
            _format_judge_prompt(
                item.get("question", ""),
                item.get("ground_truth", ""),
                item.get("prediction", ""),
            )
        )
        metadata.append(item)

    engine = get_judge_engine(MODEL_HINT)
    judge_outputs = engine.generate_json_batch(prompts, use_tqdm=True)

    correct = 0
    details: Dict[str, Dict[str, Any]] = {}
    for idx, (meta, raw) in enumerate(zip(metadata, judge_outputs)):
        is_equivalent = _parse_judge_output(raw)
        if is_equivalent:
            correct += 1

        resp_key = str(meta.get("resp_key") or f"sample_{idx}")
        details[resp_key] = {
            "question": meta.get("question"),
            "ground_truth": meta.get("ground_truth"),
            "prediction": meta.get("prediction"),
            "raw_prediction": meta.get("raw_prediction"),
            "judge_output": raw,
            "score": 1.0 if is_equivalent else 0.0,
        }

    vrt_aggregate_results.details = list(details.values())
    vrt_aggregate_results.individual_scores = details
    total = len(judge_outputs)
    return correct / total if total else 0.0
