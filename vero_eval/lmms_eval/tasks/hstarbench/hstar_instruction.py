"""Instruction normalization shared by HStar training-data and eval prompts."""

from __future__ import annotations

import re


def naturalize_hstar_instruction(value: str) -> str:
    """Rewrite an HStar task using the natural instructions seen during training."""
    instruction = value.replace("？", "?").replace("，", ",")
    instruction = re.sub(r"\s+", " ", instruction).strip()
    instruction = re.sub(r"\s+([,.;:!?])", r"\1", instruction)
    if not instruction:
        raise ValueError("empty instruction")

    context = re.split(
        r"\s*,?\s*(?=where(?:\b|')|which direction\b)",
        instruction,
        maxsplit=1,
        flags=re.I,
    )[0]
    context_lower = context.lower()
    if "scooter across the street" in context_lower:
        return "Which direction should you take to safely reach the scooter across the street?"
    if re.fullmatch(r"if you queue up to buy (?:the )?tickets", context, flags=re.I):
        return "Where should you queue to buy the tickets?"
    if re.fullmatch(r"you have lost something that is important", context, flags=re.I):
        return "Which direction should you take to recover something important you lost?"
    if re.fullmatch(r"you have something to print", context, flags=re.I):
        return "Which direction should you take to print something?"

    context = re.sub(r"^you\s+wan\s+to\b", "you want to", context, flags=re.I)
    goal_patterns = (
        (r"^(?:if\s+)?you\s+want\s+to\s+(.+)$", "action"),
        (r"^you\s+need\s+to\s+(.+)$", "action"),
        (r"^you\s+want\s+((?:go|exit|leave|take|buy|enter|throw|order|cross|ride|find|move)\b.+)$", "action"),
        (r"^you\s+want\s+(.+)$", "object"),
        (r"^you\s+need\s+(.+)$", "object"),
        (r"^you\s+are\s+going\s+to\s+(.+)$", "destination"),
    )
    for pattern, kind in goal_patterns:
        match = re.fullmatch(pattern, context, flags=re.I)
        if not match:
            continue
        goal = match.group(1).strip(" ,.?!")
        goal = re.sub(
            r"^go to (restroom|toilet|bathroom|subway platform|opposite platform|check-in counter)\b",
            r"go to the \1",
            goal,
            flags=re.I,
        )
        goal = re.sub(r"^take (elevator|escalator|subway)\b", r"take the \1", goal, flags=re.I)
        if kind == "destination":
            return f"Which direction leads to {goal}?"
        if kind == "object":
            return f"Which direction should you take to get {goal}?"
        return f"Which direction should you take to {goal}?"

    find_match = re.fullmatch(r"find\s+(.+)", instruction, flags=re.I)
    if find_match:
        target = find_match.group(1).strip(" .")
        first_word = target.split(maxsplit=1)[0].lower()
        if first_word not in {"a", "an", "and", "the", "your"}:
            instruction = f"Find the {target}"
    instruction = re.sub(r"^look for area\b", "Look for the area", instruction, flags=re.I)
    instruction = instruction[0].upper() + instruction[1:]
    if instruction[-1] not in ".?!":
        instruction += "."
    return instruction
