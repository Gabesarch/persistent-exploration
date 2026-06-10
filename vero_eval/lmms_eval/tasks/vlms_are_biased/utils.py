import re
from typing import Any, Dict, Optional

from PIL import Image as PILImage
from PIL import ImageFile

from lmms_eval.tasks._task_utils.answer_extraction import extract_final_answer

ImageFile.LOAD_TRUNCATED_IMAGES = True


def vlms_are_biased_doc_to_visual(doc: Dict[str, Any]):
    image = doc.get("image")
    if image is None:
        return []
    try:
        if isinstance(image, PILImage.Image):
            return [image.convert("RGB")]
        if hasattr(image, "convert"):
            return [image.convert("RGB")]
    except Exception:
        return []
    return []


def vlms_are_biased_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    prompt = str(doc.get("prompt") or "").strip()
    return f"{pre_prompt}{prompt}{post_prompt}"


def vlms_are_biased_doc_to_target(doc: Dict[str, Any]) -> str:
    return str(doc.get("ground_truth") or "").strip()


def _extract_braced_answer(text: str) -> Optional[str]:
    matches = re.findall(r"\{([^{}]+)\}", text)
    if matches:
        return matches[-1].strip()
    return None


def _normalize_answer(value: Any) -> str:
    text = "" if value is None else str(value)
    text = extract_final_answer(text, parse_boxed=False, strip_latex_wrappers=True)
    braced = _extract_braced_answer(text)
    if braced is not None:
        text = braced

    text = text.strip()
    text = re.sub(r"^(?:final\s+answer|answer|the\s+answer\s+is)\s*[:：-]\s*", "", text, flags=re.IGNORECASE)
    text = text.strip().strip("`'\".。,:;!?\n\r\t ")
    text = re.sub(r"\s+", " ", text).lower()

    if text in {"yes", "y", "true"}:
        return "yes"
    if text in {"no", "n", "false"}:
        return "no"

    if re.fullmatch(r"[-+]?\d+(?:\.0+)?", text):
        try:
            return str(int(float(text)))
        except Exception:
            return text

    return text


def vlms_are_biased_process_results(doc: Dict[str, Any], results):
    pred = results[0] if isinstance(results, (list, tuple)) and results else results
    prediction = _normalize_answer(pred)
    target = _normalize_answer(vlms_are_biased_doc_to_target(doc))
    return {"exact_match": 1.0 if prediction == target else 0.0}
