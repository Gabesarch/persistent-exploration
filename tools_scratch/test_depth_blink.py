#!/usr/bin/env python3
"""
Test Qwen3-VL tool calling on BLINK Relative Depth dataset.
Uses depth estimation to determine which point is closer to the camera.

Usage:
    conda activate verovlm
    python test_depth_blink.py --sample-idx 0
"""
import argparse
import base64
import json
import os
import sys
import tempfile
from io import BytesIO
from pathlib import Path

os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

from openai import OpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "eval"))
from lmms_eval.mcp import MCPClient

# Config
MCP_SERVER_PATH = str(Path(__file__).parent / "eval/lmms_eval/mcp/vero2_mcp.py")

BASE_URL = os.getenv("OPENAI_API_BASE", "http://localhost:9001/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "qwen")
MODEL = os.getenv("MODEL_NAME", "qwen_vllm")


BLINK_PARQUET_PATH = "/scratch/gpfs/ZHUANGL/lc1556/hf_cache/hub/datasets--BLINK-Benchmark--BLINK/snapshots/a3666eb249237ba3d5eca8db21176cc47967e040/Relative_Depth/val-00000-of-00001.parquet"


def load_blink_sample(idx: int) -> dict:
    """Load a sample from BLINK Relative Depth parquet file."""
    import pyarrow.parquet as pq

    table = pq.read_table(BLINK_PARQUET_PATH)
    row = table.slice(idx, 1).to_pydict()

    # Load image from bytes
    img_data = row["image_1"][0]
    if isinstance(img_data, dict) and "bytes" in img_data:
        image = Image.open(BytesIO(img_data["bytes"]))
    else:
        image = Image.open(BytesIO(img_data))

    return {
        "image": image,
        "question": row["question"][0],
        "prompt": row["prompt"][0],
        "choices": row["choices"][0],
        "answer": row["answer"][0],
        "idx": row["idx"][0],
    }


def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_depth_tools(mcp_client: MCPClient) -> list:
    """Get depth estimation tools from MCP server."""
    import asyncio
    loop = asyncio.new_event_loop()
    functions = loop.run_until_complete(mcp_client.get_function_list())
    loop.close()
    # Depth tools (no overlay - it mixes colors and makes depth hard to read)
    tool_names = {"estimate_depth", "estimate_depth_at_points"}
    return [f for f in functions if f["function"]["name"] in tool_names]


def run_tool(mcp_client: MCPClient, name: str, args: dict) -> dict:
    """Execute a tool via MCP."""
    import asyncio
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(mcp_client.run_tool(name, args))
    loop.close()

    output = {"tool": name, "success": True}
    print(f"    [DEBUG] result.content has {len(result.content)} items")
    for i, content in enumerate(result.content):
        print(f"    [DEBUG] content[{i}] type: {type(content).__name__}")
        converted = mcp_client.convert_result_to_openai_format(content)
        print(f"    [DEBUG] converted: {[c['type'] for c in converted]}")
        for item in converted:
            if item["type"] == "image_url":
                output["image_b64"] = item["image_url"]["url"]
                print(f"    [DEBUG] Found image_b64, len={len(output['image_b64'])}")
            elif item["type"] == "text":
                output["text"] = item.get("text", "")
                print(f"    [DEBUG] Text content: {output['text'][:200]}")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--no-tools", action="store_true", help="Disable tools to compare baseline performance")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    # Load sample
    sample = load_blink_sample(args.sample_idx)
    work_dir = tempfile.mkdtemp()
    image_path = os.path.join(work_dir, "input.png")
    sample["image"].save(image_path)
    image_b64 = image_to_b64(sample["image"])

    print("=" * 70)
    print(f"Sample: {sample['idx']}")
    print(f"Question: {sample['question']}")
    print(f"Prompt: {sample['prompt']}")
    print(f"Choices: {sample['choices']}")
    print(f"Expected: {sample['answer']}")
    print(f"Work dir: {work_dir}")
    print("=" * 70)

    # Setup
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)

    if args.no_tools:
        tools = None
        mcp_client = None
        print("Tools: DISABLED (baseline mode)")
    else:
        mcp_client = MCPClient(MCP_SERVER_PATH)
        tools = get_depth_tools(mcp_client)
        print(f"Depth tools: {[t['function']['name'] for t in tools]}")

    # Initial message - let model decide which tools to use
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": f"Image path: {image_path}\n\n{sample['prompt']}\n\nChoices:\n(A) {sample['choices'][0]}\n(B) {sample['choices'][1]}"}
        ]}
    ]

    # Tool loop
    for iteration in range(args.max_iterations):
        print(f"\n--- Iteration {iteration + 1} ---")
        print(f"    [DEBUG] Sending {len(messages)} messages")
        for i, m in enumerate(messages):
            role = m["role"]
            if isinstance(m.get("content"), list):
                types = [c.get("type") for c in m["content"]]
                print(f"    [DEBUG] msg[{i}] role={role} content_types={types}")
            else:
                print(f"    [DEBUG] msg[{i}] role={role} content={str(m.get('content', ''))[:50]}")

        api_kwargs = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 2048,
            "temperature": 0,
        }
        if tools:
            api_kwargs["tools"] = tools
            api_kwargs["tool_choice"] = "auto"

        response = client.chat.completions.create(**api_kwargs)

        msg = response.choices[0].message
        content = msg.content or ""
        finish_reason = response.choices[0].finish_reason

        if args.verbose:
            print(f"Finish: {finish_reason}")
            print(f"Content: {content[:300]}..." if len(content) > 300 else f"Content: {content}")

        # No tool calls - done
        if finish_reason != "tool_calls" or not msg.tool_calls:
            print(f"\nFinal answer: {content}")
            # Check if answer is correct
            if "(A)" in content and sample["answer"] == "(A)":
                print("CORRECT!")
            elif "(B)" in content and sample["answer"] == "(B)":
                print("CORRECT!")
            else:
                print(f"Expected: {sample['answer']}")
            break

        # Add assistant message with tool calls
        messages.append({
            "role": "assistant",
            "content": content,
            "tool_calls": [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in msg.tool_calls
            ]
        })

        # Execute each tool call
        for tc in msg.tool_calls:
            name = tc.function.name
            args_dict = json.loads(tc.function.arguments)

            # Ensure image_path is set
            if "image_path" not in args_dict or not args_dict["image_path"]:
                args_dict["image_path"] = image_path

            print(f"  Tool: {name}({json.dumps(args_dict)[:100]})")

            try:
                result = run_tool(mcp_client, name, args_dict)

                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Tool {name} executed successfully."
                })

                # If image returned, add as user message
                if "image_b64" in result:
                    img_path = os.path.join(work_dir, f"iter{iteration}_{name}.png")
                    img_b64 = result["image_b64"]
                    # Ensure proper data URI format
                    if not img_b64.startswith("data:"):
                        img_b64 = f"data:image/png;base64,{img_b64}"

                    img_data = img_b64.split(",")[1] if "," in img_b64 else img_b64
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(img_data))
                    print(f"    -> Image saved: {img_path}")
                    print(f"    -> Image URL prefix: {img_b64[:50]}...")

                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": img_b64}},
                            {"type": "text", "text": "Tool result above."}
                        ]
                    })
            except Exception as e:
                print(f"    -> Error: {e}")
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error: {e}"
                })

    print("\n" + "=" * 70)
    print(f"Expected: {sample['answer']}")
    print(f"Work dir: {work_dir}")


if __name__ == "__main__":
    main()
