"""Torch-free distillation adapter for V* (vstar-bench).

The eval-side utils.py imports the heavy lmms_eval package (filters, math_verify,
torch via the package __init__), so the lightweight distillation harness cannot
import it. This module re-implements ONLY what the harness needs — doc_to_text,
doc_to_visual, a stable doc_id, a robust A-D answer extractor, and a score() —
with no heavy dependencies, while staying faithful to utils.vstar_doc_to_text /
utils.extract_answer_letter so teacher traces match the eval.

The generic harness (collect_traces.py) discovers this module + these function
names via the task yaml's metadata.distill.adapter block.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

# Dataset access (generic harness reads dataset_path/split from the yaml; these are
# fallbacks / documentation).
DATASET_PATH = "lmms-lab/vstar-bench"
DATASET_SPLIT = "test"


def doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    return [doc["image"].convert("RGB")]


def doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Port of utils.vstar_doc_to_text: format the question + (A)-(D) options, with
    optional stripping of the 'answer directly' instruction and pre/post prompts."""
    question = doc["text"]
    k = lmms_eval_specific_kwargs or {}

    if k.get("strip_answer_instruction"):
        instr = "Answer with the option's letter from the given choices directly."
        question = question.replace(f"\n{instr}", "").replace(instr, "").strip()

    options_match = re.findall(r"\([A-D]\)\s*([^()]+?)(?=\s*\([A-D]\)|$)", question)
    if options_match:
        question_text = question.split("(A)")[0].strip()
        letters = ["A", "B", "C", "D"]
        opts = [f"{letters[i]}. {o.strip()}" for i, o in enumerate(options_match) if i < len(letters)]
        question = f"{question_text}\n" + "\n".join(opts)

    if k.get("pre_prompt"):
        question = f"{k['pre_prompt']}{question}"
    if k.get("post_prompt"):
        question = f"{question}{k['post_prompt']}"
    return question


def doc_id(doc: Dict[str, Any], index: int = 0) -> str:
    cat = doc.get("category", "q")
    qid = doc.get("question_id")
    return f"{cat}_{qid}" if qid is not None else f"{cat}_{int(index):04d}"


def _strip_thinking(text: str) -> str:
    if not isinstance(text, str):
        return ""
    low = text.lower()
    i = low.rfind("</think>")
    return text[i + len("</think>"):] if i != -1 else text


_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*\(?\s*([A-Da-d])\s*\)?\s*\}")


def extract_answer_letter(response: str) -> Optional[str]:
    """Robust A-D extractor: prefers \\boxed{X} / <answer>X</answer>, then common
    'Answer: X' / '(X)' / standalone-letter patterns. Faithful to (and a superset
    of) utils.extract_answer_letter, minus the math_verify dependency."""
    if not isinstance(response, str) or not response.strip():
        return None
    # boxed anywhere wins
    m = _BOXED_RE.findall(response)
    if m:
        return m[-1].upper()
    # else prefer the last <answer> block, else text after </think>
    blocks = _ANSWER_RE.findall(response)
    answer = blocks[-1] if blocks else _strip_thinking(response)
    a = answer.strip().upper()
    patterns = [
        r"\\BOXED\s*\{\s*\(?\s*([A-D])\s*\)?\s*\}",
        r"(?:THE\s+)?(?:ANSWER|CHOICE|OPTION)(?:\s+IS)?[\s:]+\(?([A-D])\)?",
        r"\(([A-D])\)",
        r"^\s*([A-D])\s*[\.)\]]",
        r"(?:^|\s)([A-D])(?:\s|$)",
    ]
    for p in patterns:
        mm = re.search(p, a)
        if mm:
            return mm.group(1).upper()
    letters = re.findall(r"[A-D]", a)
    if len(letters) == 1:
        return letters[0]
    if a and a[0] in "ABCD":
        return a[0]
    return None


def score(doc: Dict[str, Any], final_text: str) -> Dict[str, Any]:
    """Return {pred, gt, correct} for the FINAL answer text."""
    pred = extract_answer_letter(final_text)
    gt = str(doc.get("label", "")).strip().upper()
    return {"pred": pred, "gt": gt, "correct": bool(pred is not None and pred == gt),
            "category": doc.get("category", "unknown")}
