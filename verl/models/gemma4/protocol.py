"""Opt-in Gemma 4 native protocol. Never reserializes generated training tokens."""

import math
import re

THOUGHT = "<|channel>thought\n"
END_THOUGHT = "<channel|>"
MODEL = "<|turn>model\n"
USER = "<|turn>user\n"
TOOL_RESPONSE = "<|tool_response>"
MAX_SOFT_TOKENS = 1120


def split_thought(text):
    from vllm.reasoning.gemma4_utils import parse_thinking_output

    return parse_thinking_output(text)


def xyxy(box):
    if not isinstance(box, list | tuple) or len(box) != 4:
        raise ValueError("bbox_2d must have four coordinates")
    if any(isinstance(x, bool) or not isinstance(x, int | float) or not math.isfinite(x) for x in box):
        raise ValueError("bbox_2d must contain finite numbers")
    y1, x1, y2, x2 = box
    if not 0 <= x1 < x2 <= 1000 or not 0 <= y1 < y2 <= 1000:
        raise ValueError("bbox_2d must be a positive area box on the 0..1000 grid")
    return [x1, y1, x2, y2]


def parse_calls(action):
    """Use the installed server parser, with strict complete-call/schema validation."""
    from vllm.tool_parsers.gemma4_tool_parser import _parse_gemma4_args

    calls = []
    # The one-call training contract deliberately disallows nested braces in strings.
    pattern = r"<\|tool_call>call:(\w+)\{(.*?)\}<tool_call\|>"
    matches = list(re.finditer(pattern, action, re.S))
    remaining = re.sub(pattern, "", action, flags=re.S).replace(TOOL_RESPONSE, "").strip()
    if remaining or len(matches) != 1:
        return []
    for match in matches:
        args = _parse_gemma4_args(match[2])
        if match[1] != "image_zoom_in_tool" or set(args) != {"bbox_2d", "label", "img_idx"}:
            return []
        xyxy(args["bbox_2d"])
        if not isinstance(args["label"], str) or type(args["img_idx"]) is not int or args["img_idx"] < 0:
            return []
        calls.append({"name": match[1], "arguments": args})
    return calls


def parse_trajectory(text):
    """Parse solver turns only, stopping at the final answer before critic sidecars.

    Native tool feedback and user images are skipped as environment turns; action
    strings mentioned inside a thought never become actions. Returns the shared
    reward parser's tuple, with canonical x/y geometry.
    """
    calls = []
    valid = True
    # Response starts inside model; subsequent model headers follow tool/user turns.
    for idx, turn in enumerate(text.split(MODEL)):
        if idx:
            turn = turn.lstrip()
        # Sidecars only follow a completed final; the return below excludes them.
        thought = re.match(r"\s*<\|channel>thought\n(.*?)<channel\|>", turn, re.S)
        if thought is None or not thought[1].strip():
            return False, None, len(calls), calls
        action = turn[thought.end() :]
        if action.lstrip().startswith("<answer>"):
            final = re.fullmatch(
                r"\s*<answer>(.*?)</answer>\s*(?:<turn\|>|<eos>)?\s*(?:<\|turn>user\n.*)?", action, re.S
            )
            answer = final[1].strip() if final else None
            if not answer or any(
                t in answer for t in ("<answer>", "<|tool_call>", "<|channel>", "<|turn>", "</answer>")
            ):
                return False, None, len(calls), calls
            return valid, answer, len(calls), calls
        # Feedback belongs to the preceding call, never to the model action.
        action = action.split(TOOL_RESPONSE, 1)[0].strip()
        try:
            native = parse_calls(action)
        except (ValueError, TypeError):
            native = []
        if not native:
            return False, None, len(calls), calls
        for call in native:
            calls.append({**call, "arguments": {**call["arguments"], "bbox_2d": xyxy(call["arguments"]["bbox_2d"])}})
    return False, None, len(calls), calls


def render(processor, messages, tools=None, *, thinking=True):
    return processor.apply_chat_template(
        messages,
        tools=tools,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=thinking,
        preserve_thinking=True,
    )


def encode(processor, text, images=None):
    return processor(
        text=[text],
        images=images or None,
        return_tensors="pt",
        add_special_tokens=False,
        max_soft_tokens=MAX_SOFT_TOKENS,
    )["input_ids"][0].tolist()


def tool_continuation(processor, calls, texts, images, *, stopped_with_response):
    """Render a contextual assistant/tool pair and retain only the tool suffix.

    The synthetic assistant is used solely to get the template's suffix. Raw
    sampled prefix IDs are neither compared to nor replaced by its rendering.
    """
    assistant = {
        "role": "assistant",
        "content": "",
        "reasoning": "context",
        "tool_calls": [{"type": "function", "function": call} for call in calls],
    }
    messages = [{"role": "user", "content": "context"}, assistant]
    messages += [
        {"role": "tool", "name": call["name"], "content": text} for call, text in zip(calls, texts, strict=True)
    ]
    if images:
        messages.append(
            {
                "role": "user",
                "content": [{"type": "image"} for _ in images]
                + [{"type": "text", "text": "Above are the tool-returned image(s)."}],
            }
        )
    full = render(processor, messages)
    suffix = full.split(TOOL_RESPONSE, 1)[1]
    if not stopped_with_response:
        suffix = TOOL_RESPONSE + suffix
    return encode(processor, suffix, images)
