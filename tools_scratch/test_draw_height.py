#!/usr/bin/env python3
"""Test if VLM can draw the height of a triangle."""
import asyncio
import base64
import os
import sys
import json
import re
import tempfile
import shutil
from pathlib import Path

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

sys.path.insert(0, str(Path(__file__).parent / "eval"))

from openai import AsyncOpenAI

# Config
IMAGE_PATH = "/home/hw0171/reason-vlm-sft/triangle_test.png"
BASE_URL = os.getenv("OPENAI_API_BASE", "http://localhost:9001/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "qwen")
MODEL = os.getenv("MODEL_NAME", "qwen_vllm")
MCP_SERVER_PATH = str(Path(__file__).parent / "eval/lmms_eval/mcp/simplevqa_tools.py")

# Triangle coordinates in 0-1000 scale
COORDS = {
    "A": (167, 833),  # bottom left
    "B": (833, 833),  # bottom right
    "C": (500, 167),  # top
    "M": (500, 833),  # midpoint of base (for height)
}


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def main():
    from lmms_eval.mcp import MCPClient

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    mcp_client = MCPClient(MCP_SERVER_PATH)
    work_dir = tempfile.mkdtemp()

    image_work_path = os.path.join(work_dir, "triangle.png")
    shutil.copy(IMAGE_PATH, image_work_path)

    image_b64 = encode_image_base64(IMAGE_PATH)

    coords_str = "\n".join([f"  {name}: ({x}, {y})" for name, (x, y) in COORDS.items()])

    system_prompt = """You are a geometry assistant with drawing tools.

## Tools
To draw, write:
```tool
{"name": "draw_line", "arguments": {"image_path": "IMAGE_PATH", "x1": 100, "y1": 200, "x2": 300, "y2": 400, "color": "red"}}
```

Available: draw_line, draw_circle, draw_text
Coordinates: 0-1000 scale, (0,0)=top-left."""

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": f"""Image path: {image_work_path}

## Coordinates (0-1000 scale)
{coords_str}

## Task
Draw the HEIGHT of this triangle (the perpendicular line from vertex C to the base AB).

Use draw_line to draw a RED line from C to M (the midpoint of AB).
This represents the height of the triangle."""},
    ]

    print("=" * 60)
    print("Testing: Draw the height of a triangle")
    print("=" * 60)
    print(f"Work dir: {work_dir}")
    print(f"Coordinates: {COORDS}")
    print("-" * 60)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=2048,
        temperature=0,
    )

    content = response.choices[0].message.content or ""
    print(f"\nModel response:\n{content}")

    # Parse and execute tool calls
    tool_pattern = r'```tool\s*\n?(.*?)\n?```'
    tool_matches = re.findall(tool_pattern, content, re.DOTALL)

    if tool_matches:
        for tool_json in tool_matches:
            try:
                tool_data = json.loads(tool_json.strip())
                tool_name = tool_data["name"]
                tool_args = tool_data["arguments"]

                if "image_path" in tool_args and tool_args["image_path"] == "IMAGE_PATH":
                    tool_args["image_path"] = image_work_path

                print(f"\n>>> Executing: {tool_name}")
                print(f"    Args: {tool_args}")

                result = await mcp_client.run_tool(tool_name, tool_args)

                # Save result
                for c in result.content:
                    if hasattr(c, 'data'):
                        img_bytes = base64.b64decode(c.data)
                        with open(image_work_path, "wb") as f:
                            f.write(img_bytes)
                        print(f"    Saved to: {image_work_path}")

            except Exception as e:
                print(f"    Error: {e}")

    output_path = os.path.join(work_dir, "result.png")
    shutil.copy(image_work_path, output_path)
    print(f"\n{'=' * 60}")
    print(f"Result saved to: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
