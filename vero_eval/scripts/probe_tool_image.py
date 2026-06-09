#!/usr/bin/env python
"""Probe: does Qwen3-VL on the local vLLM serve actually SEE an image that is
returned inside a role:'tool' message? Compares three delivery paths against a
control. Prints a verdict used to set the tool-image feedback default in the
agentic eval (see memory multiturn-tool-eval-design).
"""
import base64
import io
import os
import sys

from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

BASE_URL = os.environ.get("BASE_URL", "http://localhost:9001/v1")
API_KEY = os.environ.get("API_KEY", "qwen")
MODEL = os.environ.get("MODEL_VERSION", "qwen_vllm")
SECRET = "317"  # unguessable number the model can only know by reading the image

client = OpenAI(base_url=BASE_URL, api_key=API_KEY, timeout=120)


def make_image() -> str:
    img = Image.new("RGB", (480, 320), (20, 60, 160))
    d = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", 180)
    except Exception:
        font = ImageFont.load_default()
    d.text((110, 60), SECRET, fill=(255, 255, 255), font=font)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


DATA_URL = f"data:image/png;base64,{make_image()}"
QUESTION = "What number is written in the image? Reply with ONLY the digits, nothing else."
TOOLS = [{
    "type": "function",
    "function": {
        "name": "reveal_image",
        "description": "Reveal a hidden image.",
        "parameters": {"type": "object", "properties": {}},
    },
}]


def ask(messages, tools=None):
    kwargs = dict(model=MODEL, messages=messages, max_tokens=64, temperature=0)
    if tools:
        kwargs["tools"] = tools
        kwargs["tool_choice"] = "none"  # we want a text answer on the final turn
    r = client.chat.completions.create(**kwargs)
    return (r.choices[0].message.content or "").strip()


def verdict(name, answer):
    ok = SECRET in (answer or "")
    print(f"[{name:14s}] saw_image={ok!s:5s}  answer={answer!r}")
    return ok


def control():
    msgs = [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": DATA_URL}},
        {"type": "text", "text": QUESTION},
    ]}]
    return verdict("A_control", ask(msgs))


def tool_role():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "Call reveal_image, then read the number in the returned image."}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "reveal_image", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "reveal_image", "content": [
            {"type": "image_url", "image_url": {"url": DATA_URL}},
            {"type": "text", "text": "Here is the revealed image."},
        ]},
        {"role": "user", "content": [{"type": "text", "text": QUESTION}]},
    ]
    try:
        return verdict("B_tool_role", ask(msgs, tools=TOOLS))
    except Exception as e:
        print(f"[B_tool_role   ] ERROR: {str(e)[:300]}")
        return False


def user_followup():
    msgs = [
        {"role": "user", "content": [{"type": "text", "text": "Call reveal_image, then read the number in the returned image."}]},
        {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call_1", "type": "function", "function": {"name": "reveal_image", "arguments": "{}"}},
        ]},
        {"role": "tool", "tool_call_id": "call_1", "name": "reveal_image",
         "content": [{"type": "text", "text": "Image revealed; shown in the next message."}]},
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": DATA_URL}},
            {"type": "text", "text": QUESTION},
        ]},
    ]
    try:
        return verdict("C_user_follow", ask(msgs, tools=TOOLS))
    except Exception as e:
        print(f"[C_user_follow ] ERROR: {str(e)[:300]}")
        return False


if __name__ == "__main__":
    print(f"=== Tool-image probe vs {BASE_URL} (model={MODEL}, secret={SECRET}) ===")
    a = control()
    b = tool_role()
    c = user_followup()
    print("\n=== VERDICT ===")
    if not a:
        print("INCONCLUSIVE: control failed — model can't read the image even as a plain user image.")
        sys.exit(2)
    print(f"role:tool image rendering  -> {'WORKS' if b else 'DROPPED'}")
    print(f"user-followup image        -> {'WORKS' if c else 'DROPPED'}")
    print("RECOMMEND default: " + ("role:'tool' image_url" if b else ("follow-up user turn" if c else "NEITHER — investigate chat template")))
