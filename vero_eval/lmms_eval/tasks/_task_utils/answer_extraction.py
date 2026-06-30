from __future__ import annotations

import re
from typing import Any


def _extract_last_boxed_content(text: str) -> str | None:
    r"""Return content of the last balanced \boxed{...}, if present."""
    last_content: str | None = None
    start_idx = 0
    while True:
        idx = text.find("\\boxed{", start_idx)
        if idx == -1:
            break
        start = idx + len("\\boxed{")
        depth = 1
        pos = start
        while pos < len(text) and depth > 0:
            if text[pos] == "{":
                depth += 1
            elif text[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            last_content = text[start : pos - 1]
            start_idx = pos
        else:
            break
    return last_content


def _strip_boxed(text: str) -> str:
    r"""Remove \boxed{...} wrappers, keeping inner content. Handles nested braces."""
    result = text
    while True:
        idx = result.find("\\boxed{")
        if idx == -1:
            break
        start = idx + len("\\boxed{")
        depth = 1
        pos = start
        while pos < len(result) and depth > 0:
            if result[pos] == "{":
                depth += 1
            elif result[pos] == "}":
                depth -= 1
            pos += 1
        if depth == 0:
            inner = result[start : pos - 1]
            result = result[:idx] + inner + result[pos:]
        else:
            break  # Unbalanced braces — stop
    return result


def _strip_tex_text_wrapper(text: str) -> str:
    r"""If text is exactly \text{...}, unwrap it once (or repeatedly)."""
    result = text.strip()
    while result.startswith("\\text{") and result.endswith("}"):
        start = len("\\text{")
        depth = 1
        pos = start
        while pos < len(result) and depth > 0:
            if result[pos] == "{":
                depth += 1
            elif result[pos] == "}":
                depth -= 1
            pos += 1
        if depth != 0 or pos != len(result):
            break
        result = result[start : pos - 1].strip()
    return result


def extract_final_answer(
    raw_response: Any,
    parse_boxed: bool = True,
    strip_latex_wrappers: bool = False,
) -> str:
    """Extract the final answer, removing thinking reasoning.

    - If <answer>...</answer> exists, prefer its content.
    - Otherwise, drop content up to </think> if present.
    - Trim common answer prefixes.
    - *parse_boxed*: extract content of the last \\boxed{...} (for math tasks).
    - *strip_latex_wrappers*: remove \\boxed{...} and \\text{...} wrappers
      inline, keeping surrounding text intact (for instruction-following tasks
      where these are just noise from reasoning models).
    """
    if not isinstance(raw_response, str):
        return "" if raw_response is None else str(raw_response)

    text = raw_response.strip()
    if not text:
        return ""

    # Multi-turn tool transcripts (async_openai): the final answer always follows
    # the last tool response — everything before it is tool-loop history. A
    # terminal answer-protocol call (e.g. screenspot's computer_use <tool_call>)
    # comes after the last </tool_response>, so it survives this cut.
    if "</tool_response>" in text:
        text = text.rsplit("</tool_response>", 1)[-1].strip()
    if "[MAX_TOOL_ITERATIONS_REACHED]" in text:
        text = text.replace("[MAX_TOOL_ITERATIONS_REACHED]", "").strip()

    if "</think>" in text and "<think>" not in text:
        text = f"<think>\n{text}"
    if "<answer>" in text and "</answer>" in text:
        text = text.split("<answer>", 1)[1].split("</answer>", 1)[0].strip()
    if "</think>" in text:
        text = text.split("</think>", 1)[-1].strip()

    # if "Answer:" in text:
    #     text = text.rsplit("Answer:", 1)[-1].strip()
    # elif "answer:" in text:
    #     text = text.rsplit("answer:", 1)[-1].strip()

    if parse_boxed:
        boxed_content = _extract_last_boxed_content(text)
        if boxed_content is not None:
            text = boxed_content.strip()
        else:
            stripped = _strip_boxed(text)
            if stripped != text:
                text = stripped.strip()

        text = _strip_tex_text_wrapper(text)
    elif strip_latex_wrappers:
        text = _strip_boxed(text)
        text = _strip_tex_text_wrapper(text)

    return text.strip()


# Explicit conclusion phrases — prefer the *last* match (model may revise).
_MC_EXPLICIT_PATTERNS = (
    r"(?:THE\s+)?(?:CORRECT\s+)?(?:ANSWER|CHOICE|OPTION)\s+IS\s*:?\s*\n?\s*([A-D])\b",
    r"THEREFORE,?\s*(?:THE\s+)?(?:CORRECT\s+)?(?:ANSWER|CHOICE)\s+IS\s*:?\s*([A-D])\b",
)


def extract_multiple_choice_letter(text: str) -> str | None:
    """Extract a single multiple-choice letter (A–D) from a model response.

    Handles short direct answers (``A. rubber``, ``C``) and long post-tool
    analyses that start with prose such as "Based on the visual analysis...".
    The legacy ``^([A-D])`` prefix rule is intentionally omitted — it falsely
    matched the leading B in "Based".
    """
    if not isinstance(text, str):
        text = "" if text is None else str(text)
    original = text.strip()
    if not original:
        return None

    upper = original.upper()

    for pattern in _MC_EXPLICIT_PATTERNS:
        matches = list(re.finditer(pattern, upper))
        if matches:
            return matches[-1].group(1)

    # Last "A. option" line (common final-answer format).
    option_lines = list(re.finditer(r"(?:^|\n)\s*([A-D])\.\s+\S", original, re.IGNORECASE))
    if option_lines:
        return option_lines[-1].group(1).upper()

    # Short direct answers (typical no-tool outputs).
    if len(original) <= 32:
        short = re.match(r"^([A-D])\s*[\.\):]?\s*(?:\S.*)?$", original, re.IGNORECASE)
        if short:
            return short.group(1).upper()
        if re.fullmatch(r"[A-D]", original, re.IGNORECASE):
            return original.upper()

    # (A) — last parenthesized letter.
    paren = list(re.finditer(r"\(([A-D])\)", upper))
    if paren:
        return paren[-1].group(1)

    # Standalone letter bounded by whitespace/punctuation — last occurrence.
    standalone = list(re.finditer(r"(?:^|[\s\n])([A-D])(?:[\s\n\.]|$)", upper))
    if standalone:
        return standalone[-1].group(1)

    letters = re.findall(r"[A-D]", upper)
    if len(letters) == 1:
        return letters[0]

    return None
