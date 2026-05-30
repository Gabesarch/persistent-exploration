#!/usr/bin/env python3
"""Test if VLM can draw a line between two labeled points."""
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

IMAGE_PATH = "/home/hw0171/reason-vlm-sft/labeled_points.png"
BASE_URL = os.getenv("OPENAI_API_BASE", "http://localhost:9001/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "qwen")
MODEL = os.getenv("MODEL_NAME", "qwen_vllm")
MCP_SERVER_PATH = str(Path(__file__).parent / "eval/lmms_eval/mcp/simplevqa_tools.py")


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def main():
    from lmms_eval.mcp import MCPClient

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    mcp_client = MCPClient(MCP_SERVER_PATH)
    work_dir = tempfile.mkdtemp()

    image_work_path = os.path.join(work_dir, "points.png")
    shutil.copy(IMAGE_PATH, image_work_path)

    image_b64 = encode_image_base64(IMAGE_PATH)

    system_prompt = """You are a geometry assistant with drawing tools.

## Tools
To draw a line:
```tool
{"name": "draw_line", "arguments": {"image_path": "IMAGE_PATH", "x1": X1, "y1": Y1, "x2": X2, "y2": Y2, "color": "red"}}
```

Coordinate system (0-1000 scale):
- (0, 0) = top-left corner
- (1000, 1000) = bottom-right corner

Look at the image and estimate where the labeled points are located."""

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": f"""Image path: {image_work_path}

This image shows 4 labeled points: P, Q, R, S.

Draw a RED line from point P to point R.

Look at where P and R are in the image, estimate their coordinates, and use draw_line."""},
    ]

    print("=" * 60)
    print("Test: Draw a line from P to R")
    print("=" * 60)
    print(f"Work dir: {work_dir}")
    print("\nActual coordinates (for verification):")
    print("  P: (250, 333)")
    print("  R: (833, 667)")
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

                # Compare with actual
                if tool_name == "draw_line":
                    print(f"    Expected P: ~(250, 333), got: ({tool_args.get('x1')}, {tool_args.get('y1')})")
                    print(f"    Expected R: ~(833, 667), got: ({tool_args.get('x2')}, {tool_args.get('y2')})")

                result = await mcp_client.run_tool(tool_name, tool_args)

                for c in result.content:
                    if hasattr(c, 'data'):
                        img_bytes = base64.b64decode(c.data)
                        with open(image_work_path, "wb") as f:
                            f.write(img_bytes)
                        print(f"    Saved!")

            except Exception as e:
                print(f"    Error: {e}")
    else:
        print("\nNo tool calls found!")

    output_path = os.path.join(work_dir, "result.png")
    shutil.copy(image_work_path, output_path)
    print(f"\n{'=' * 60}")
    print(f"Result: {output_path}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    asyncio.run(main())
