import base64
import io
import string
from collections import defaultdict
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger as eval_logger
from PIL import Image

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer, extract_multiple_choice_letter

try:
    from math_verify import parse as math_verify_parse
    from math_verify.parser import StringExtractionConfig
except Exception:  # pragma: no cover - optional dependency
    math_verify_parse = None
    StringExtractionConfig = None

_hrbench_yaml = Path(__file__).parent / "hrbench.yaml"
if _hrbench_yaml.exists():
    with open(_hrbench_yaml, "r") as f:
        raw_data = f.readlines()
        safe_data = [line for line in raw_data if "!function" not in line]
        config = yaml.safe_load("".join(safe_data))


def decode_base64_to_image(base64_string, target_size=-1):
    image_data = base64.b64decode(base64_string)
    image = Image.open(io.BytesIO(image_data)).convert("RGB")
    if target_size > 0:
        image.thumbnail((target_size, target_size))
    return image


def hrbench_doc_to_visual(doc):
    if isinstance(doc.get("image"), Image.Image):
        return [doc["image"].convert("RGB")]
    return [decode_base64_to_image(doc["image"])]


def hrbench_doc_to_options(doc):
    return {
        cand: doc[cand]
        for cand in string.ascii_uppercase
        if cand in doc and not pd.isna(doc[cand])
    }


def hrbench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    question = doc["question"].strip()
    options = hrbench_doc_to_options(doc)
    options_prompt = "".join(f"{key}. {item}\n" for key, item in options.items())
    answer_instruction = "Answer with the option's letter from the given choices directly."
    prompt = f"{question}\n{options_prompt}{answer_instruction}"

    if lmms_eval_specific_kwargs:
        if lmms_eval_specific_kwargs.get("strip_answer_instruction"):
            prompt = f"{question}\n{options_prompt}Answer the option letter directly."
        pre = lmms_eval_specific_kwargs.get("pre_prompt", "")
        post = lmms_eval_specific_kwargs.get("post_prompt", "")
        if pre:
            prompt = f"{pre}{prompt}"
        if post:
            prompt = f"{prompt}{post}"
    return prompt


def _final_answer_segment(response: str) -> str:
    """Keep only text after the last tool turn so bbox JSON cannot leak A-D letters."""
    for marker in ("</tool_response>", "</tool_call>"):
        if marker in response:
            return response.rsplit(marker, 1)[1]
    return response


def extract_answer_letter(response):
    if not isinstance(response, str):
        response = "" if response is None else str(response)
    response = extract_final_answer(_final_answer_segment(response))

    if math_verify_parse and StringExtractionConfig and len(response.strip()) <= 32:
        try:
            parsed = math_verify_parse(
                response.strip().upper(),
                extraction_config=[StringExtractionConfig(strings=("A", "B", "C", "D"))],
            )
        except Exception:
            parsed = None
        if parsed:
            parsed_choice = str(parsed[0]).strip().upper()
            if parsed_choice in ("A", "B", "C", "D"):
                return parsed_choice

    return extract_multiple_choice_letter(response)


def hrbench_process_results(doc, results):
    pred = results[0] if results else ""
    pred_letter = extract_answer_letter(pred) or ""
    gt_letter = str(doc["answer"]).strip().upper()
    score = 1.0 if pred_letter == gt_letter else 0.0

    category = doc.get("category", "unknown")
    result = {
        "index": doc.get("index"),
        "category": category,
        "score": score,
        "prediction": pred_letter,
        "ground_truth": gt_letter,
    }
    if score == 0:
        eval_logger.debug(
            f"Q: {doc['question'][:80]}... pred={pred_letter} gt={gt_letter} raw={str(pred)[:80]!r}"
        )
    return {"hrbench_overall_acc": result}


def hrbench_aggregate_results(results):
    if not results:
        return 0.0
    category_scores = defaultdict(list)
    all_scores = []
    for r in results:
        category_scores[r["category"]].append(r["score"])
        all_scores.append(r["score"])
    for category, scores in category_scores.items():
        if scores:
            eval_logger.info(f"{category}: {sum(scores) / len(scores):.4f} (n={len(scores)})")
    overall = sum(all_scores) / len(all_scores)
    eval_logger.info(f"Overall: {overall:.4f} (n={len(all_scores)})")
    return overall
