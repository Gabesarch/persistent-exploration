import logging
import os
import re
from typing import Any, Dict, List, Optional

eval_logger = logging.getLogger("lmms-eval")

# Original prompt, mirroring the official SketchVLM ball-drop evaluation
# (https://github.com/Brandon-Collins7/sketchvlm -> gpt_baseline_ball_drop.py DEFAULT_PROMPT).
# Kept for reference / reproducibility of the first baseline run (~22% acc).
BALL_DROP_PROMPT_ORIGINAL = (
    "You are given the start frame of a physics simulation. A ball is dropped from the "
    "top of the screen and falls due to gravity. The ball can roll off the lines or the "
    "walls in the image. The bouncing of the ball is relatively minor and realistic for "
    "normal gravity. Nothing in the image will move besides the ball. Predict which bucket "
    "will eventually catch the ball. There are 4 different buckets called bucket 1, bucket 2, "
    "bucket 3, and bucket 4. Please respond with what bucket the ball will fall into. Your "
    "final answer must be formatted as \"$\\boxed{bucket number}$\". For example, if the ball "
    "will fall into bucket 2, respond with \"$\\boxed{2}$\"."
)

# v2 — Guided prompt: targets the observed failure modes of the terse baseline — the model
# ignored the obstacle lines, mis-located the ball, and defaulted to "falls straight
# down -> bucket 1" without simulating any deflection. This version names the lines as
# solid ramps, states the rolling/deflection mechanic, and forces step-by-step tracing.
# Result: ~38% acc, but it OVER-corrected — it invented deflections on straight-down cases
# (non-deflected acc fell 0.36->0.19) and flipped its bias to over-predicting bucket 4.
BALL_DROP_PROMPT_GUIDED = (
    "You are given the start frame of a physics simulation. A blue ball is released from "
    "its current position near the top and falls straight down under gravity.\n"
    "The thin black diagonal line segments in the image are SOLID ramps/obstacles: the "
    "falling ball cannot pass through a line. When the ball lands on a line, it rolls and "
    "slides along that line toward the line's LOWER end, then falls off that end and keeps "
    "dropping. It may also bounce slightly and roll off the left/right side walls. Only the "
    "ball moves; every line and wall is fixed.\n"
    "The bottom of the image is split into 4 equally sized buckets, labeled 1, 2, 3, 4 from "
    "left to right. Bucket 1 is the leftmost quarter and bucket 4 is the rightmost quarter; "
    "the center of the image is the boundary between bucket 2 and bucket 3.\n"
    "Reason step by step before answering:\n"
    "1. Note the ball's horizontal position at the top (which bucket column is it above?).\n"
    "2. Find EVERY black line segment and note its slope (which end is lower).\n"
    "3. Trace the ball's path: it falls straight down until it meets a line, then slides to "
    "that line's lower end, then falls again — repeat for each line or wall it reaches.\n"
    "4. Determine the ball's final horizontal position and which of the 4 equal buckets it "
    "lands in.\n"
    "Your final answer must be formatted as \"$\\boxed{bucket number}$\". For example, if the "
    "ball will fall into bucket 2, respond with \"$\\boxed{2}$\"."
)

# v3 — Guarded prompt: v2 plus an explicit "do not assume a deflection" guard. v2 over-
# corrected by inventing rolls on balls that actually fall straight down; this version
# tells the model to first test whether a line truly lies in the ball's vertical path,
# fall straight into the bucket below when none does, and only then trace a roll.
BALL_DROP_PROMPT = (
    "You are given the start frame of a physics simulation. A blue ball is released from "
    "its current position near the top and falls straight down under gravity.\n"
    "The thin black diagonal line segments in the image are SOLID ramps/obstacles: the "
    "falling ball cannot pass through a line. When the ball lands on a line, it rolls and "
    "slides along that line toward the line's LOWER end, then falls off that end and keeps "
    "dropping. It may also bounce slightly and roll off the left/right side walls. Only the "
    "ball moves; every line and wall is fixed.\n"
    "The bottom of the image is split into 4 equally sized buckets, labeled 1, 2, 3, 4 from "
    "left to right. Bucket 1 is the leftmost quarter and bucket 4 is the rightmost quarter; "
    "the center of the image is the boundary between bucket 2 and bucket 3.\n"
    "IMPORTANT: a line only affects the ball if it actually lies in the ball's vertical fall "
    "path. A line that is off to the side, or that the ball would never reach, does nothing. "
    "Do NOT assume the ball is deflected — many balls fall straight down with no deflection "
    "at all, landing in the bucket directly below where they start.\n"
    "Reason step by step before answering:\n"
    "1. Note the ball's exact horizontal position at the top and which bucket column (1-4) is "
    "directly below it.\n"
    "2. List EVERY black line segment with its horizontal span and height.\n"
    "3. Check the straight-down path: would the ball, falling vertically from its current x, "
    "actually land on any line? If NO line is in that path, the ball falls straight into the "
    "bucket directly below its start position — answer that bucket.\n"
    "4. If a line IS in the path, the ball lands on the highest such line, rolls to that "
    "line's LOWER end, then falls straight down again from that end. Repeat the check in "
    "step 3 from the new position until the ball reaches the bottom.\n"
    "5. Determine the ball's final horizontal position and which of the 4 equal buckets it "
    "lands in.\n"
    "Your final answer must be formatted as \"$\\boxed{bucket number}$\". For example, if the "
    "ball will fall into bucket 2, respond with \"$\\boxed{2}$\"."
)

# v4 — minimal, BALANCED physics for capable reasoners. Fixes v1's flaw (it dismissed the
# lines as "minor obstacles" and under-deflected) without v2's deflection bias: states
# deflect-if-path-hits-a-line and straight-if-path-clear symmetrically, with no rigid
# numbered procedure, so a strong thinking model applies its own calibrated reasoning.
# This is the default for the tool-calling variants: the toolset system prompt drives the
# inspect-then-answer behavior, while this body supplies just enough physics.
BALL_DROP_PROMPT_V4 = (
    "You are given the start frame of a physics simulation. A blue ball falls straight down "
    "from its current position under gravity. The thin black line segments are solid: if the "
    "ball's downward path runs into a line, the ball rolls along that line and slides off its "
    "LOWER end, then keeps falling; but if the path directly below the ball is clear, it "
    "simply falls straight down into the bucket beneath it. The ball may also glance off the "
    "left/right walls, and bouncing is slight. At the bottom are 4 equally sized buckets "
    "numbered 1, 2, 3, 4 from left to right. Work out which bucket finally catches the ball, "
    "then give your final answer as \"$\\boxed{N}$\" (for example \"$\\boxed{2}$\")."
)

# v5 — like v4 but makes the single decisive TEST explicit (trace straight down from the
# ball; only a line that crosses that vertical path matters), still as one instruction
# rather than a step-by-step script.
BALL_DROP_PROMPT_V5 = (
    "You are given the start frame of a physics simulation. A blue ball falls straight down "
    "under gravity from where it is shown. The thin black lines are solid surfaces. Whether "
    "the ball gets deflected depends only on its horizontal position: follow a vertical line "
    "straight down from the ball — if that path crosses a black line, the ball rolls down "
    "that line and off its LOWER end before continuing to fall; if the vertical path is clear "
    "all the way down, the ball lands in the bucket directly below with no deflection. "
    "Bouncing is slight and it may roll off the side walls. The bottom has 4 equally sized "
    "buckets numbered 1, 2, 3, 4 from left to right. Decide which bucket catches the ball and "
    "answer \"$\\boxed{N}$\" (e.g. \"$\\boxed{2}$\")."
)

# Select the user-content prompt by version. Default (v3) keeps the qwen3vl_zs task
# unchanged; the reasoning/tool variants pin their own version via the yaml.
_PROMPT_BY_VERSION = {
    "v1": BALL_DROP_PROMPT_ORIGINAL,
    "v2": BALL_DROP_PROMPT_GUIDED,
    "v3": BALL_DROP_PROMPT,
    "v4": BALL_DROP_PROMPT_V4,
    "v5": BALL_DROP_PROMPT_V5,
}

# Strict boxed bucket, e.g. \boxed{2} or \boxed{ 3 } (with or without surrounding $).
_BOXED_RE = re.compile(r"\\boxed\s*\{\s*([1-4])\s*\}")
# "bucket 2", "bucket #3", "Bucket: 4"
_BUCKET_RE = re.compile(r"bucket\s*#?\s*:?\s*([1-4])", re.IGNORECASE)
# Last-resort: a lone bucket digit.
_DIGIT_RE = re.compile(r"\b([1-4])\b")
# Reasoning models (e.g. Vero) wrap the final answer in <answer>...</answer>.
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def ball_drop_doc_to_visual(doc: Dict[str, Any]) -> List[Any]:
    image = doc["start_image"]
    return [image.convert("RGB")]


def ball_drop_doc_to_text(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}
    pre_prompt = lmms_eval_specific_kwargs.get("pre_prompt", "")
    post_prompt = lmms_eval_specific_kwargs.get("post_prompt", "")
    # Env var takes precedence over the yaml kwarg, so the same task config can be
    # reused with a different prompt version (e.g. BALL_DROP_PROMPT_VERSION=v2).
    version = os.environ.get("BALL_DROP_PROMPT_VERSION") or lmms_eval_specific_kwargs.get("prompt_version", "v3")
    body = _PROMPT_BY_VERSION.get(version, BALL_DROP_PROMPT)
    prompt = f"{pre_prompt}{body}"
    if post_prompt:
        prompt = f"{prompt}\n{post_prompt}"
    return prompt.strip()


def _strip_thinking(text: str) -> str:
    """Drop chain-of-thought up to the final </think> if the model emitted one."""
    if not isinstance(text, str):
        return ""
    lowered = text.lower()
    idx = lowered.rfind("</think>")
    if idx != -1:
        return text[idx + len("</think>") :]
    return text


def parse_bucket(text: str) -> Optional[int]:
    """Extract the predicted bucket (1-4) from a model response.

    Preference order: the last ``\\boxed{N}`` -> the last "bucket N" mention ->
    the last lone digit 1-4. Returns ``None`` when nothing parseable is found.
    """
    if not isinstance(text, str) or not text.strip():
        return None
    # Prefer the content of the last <answer>...</answer> block (reasoning models);
    # otherwise drop everything up to the final </think>.
    answer_blocks = _ANSWER_RE.findall(text)
    answer = answer_blocks[-1] if answer_blocks else _strip_thinking(text)

    for pattern in (_BOXED_RE, _BUCKET_RE):
        matches = pattern.findall(answer)
        if matches:
            return int(matches[-1])

    # Fall back to scanning the whole (un-stripped) response for a boxed answer,
    # then for any lone bucket digit.
    boxed_all = _BOXED_RE.findall(text)
    if boxed_all:
        return int(boxed_all[-1])
    digits = _DIGIT_RE.findall(answer)
    if digits:
        return int(digits[-1])
    return None


def ball_drop_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    response = results[0] if results else ""
    gt = int(doc["final_bucket_gt"])
    pred = parse_bucket(response)
    correct = 1.0 if (pred is not None and pred == gt) else 0.0
    return {"ball_drop_acc": correct}


# ---- Distillation helpers (used by vero_eval/distill/collect_ball_drop_traces.py) ----
# Not referenced by lmms_eval task machinery; kept here so the distillation harness and
# the eval task share ONE definition of the prompt, the ground truth, and the parser.

def ball_drop_doc_id(doc: Dict[str, Any], index: int = 0) -> str:
    """Stable id for naming a doc's saved image / trace. Prefers the dataset's
    ``run_id`` (e.g. 'run_001_1'); falls back to a zero-padded index."""
    rid = doc.get("run_id")
    if rid:
        return str(rid)
    return f"idx_{int(index):05d}"
