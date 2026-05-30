#!/usr/bin/env python3
"""
Test Qwen3-VL tool calling on SketchVLM Maze Navigation dataset.
Uses vLLM with hermes parser (--enable-auto-tool-choice --tool-call-parser hermes).

Offline usage (interactive node without internet):
    conda activate verovlm
    export HF_HOME=/scratch/gpfs/ZHUANGL/hf_cache
    export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

Usage:
    python test_maze.py --sample-idx 0
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

from datasets import load_dataset
from openai import OpenAI
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent / "eval"))
from lmms_eval.mcp import MCPClient
from conversation_logger import ConversationLogger

# Config
MAZE_CACHE_DIR = "/scratch/gpfs/ZHUANGL/hw0171/maze_nav"
MCP_SERVER_PATH = str(Path(__file__).parent / "eval/lmms_eval/mcp/vero2_mcp.py")

BASE_URL = os.getenv("OPENAI_API_BASE", "http://localhost:9001/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "qwen")
MODEL = os.getenv("MODEL_NAME", "qwen_vllm")


def load_sample(idx: int) -> dict:
    """Load a sample from maze navigation dataset."""
    ds = load_dataset("loganbolton/sketchvlm-maze-navigation", cache_dir=MAZE_CACHE_DIR)
    sample = ds["train"][idx]
    metadata = json.loads(sample["metadata_json"])
    return {
        "image": sample["maze_image"],
        "image_with_path": sample["maze_image_with_path"],
        "maze_id": sample["maze_id"],
        "entry": sample["entry_coordinate"],
        "exit": sample["exit_coordinate"],
        "path_directions": sample["shortest_path_directions"],
        "path_coordinates": sample["shortest_path_coordinates"],
        "rows": metadata["rows"],
        "cols": metadata["cols"],
    }


def image_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


def get_vision_tools(mcp_client: MCPClient) -> list:
    """Get all drawing tools from MCP server."""
    import asyncio
    loop = asyncio.new_event_loop()
    functions = loop.run_until_complete(mcp_client.get_function_list())
    loop.close()
    # All drawing tools for maze navigation
    tool_names = {
        "draw_arrow",           # Draw arrows to show direction/movement
        "draw_line",            # Draw lines
        "draw_circle",          # Mark positions (entry/exit)
        "draw_box",             # Highlight regions
        "draw_text",            # Add text annotations
        "draw_numbered_boxes",  # Mark multiple boxes with numbers
        "draw_numbered_points", # Mark points with numbers
        "draw_grid",            # Grid overlay
    }
    return [f for f in functions if f["function"]["name"] in tool_names]


def run_tool(mcp_client: MCPClient, name: str, args: dict) -> dict:
    """Execute a tool via MCP."""
    import asyncio
    loop = asyncio.new_event_loop()
    result = loop.run_until_complete(mcp_client.run_tool(name, args))
    loop.close()

    # Convert result to dict with image handling
    output = {"tool": name, "success": True}
    for content in result.content:
        converted = mcp_client.convert_result_to_openai_format(content)
        for item in converted:
            if item["type"] == "image_url":
                output["image_b64"] = item["image_url"]["url"]
            elif item["type"] == "text":
                output["text"] = item.get("text", "")
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-idx", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    # Load sample
    sample = load_sample(args.sample_idx)
    work_dir = tempfile.mkdtemp()
    image_path = os.path.join(work_dir, "input.png")
    sample["image"].save(image_path)
    image_b64 = image_to_b64(sample["image"])

    # Setup logger
    output_path = args.output or f"maze_{sample['maze_id']}.html"
    logger = ConversationLogger(output_path, title=f"Maze Navigation: {sample['maze_id']}")
    logger.set_metadata(
        sample_idx=args.sample_idx,
        maze_id=sample["maze_id"],
        grid_size=f"{sample['rows']}x{sample['cols']}",
        entry=str(sample["entry"]),
        exit=str(sample["exit"]),
        expected_path=" -> ".join(sample["path_directions"]),
        image=f"data:image/png;base64,{image_b64}",
        work_dir=work_dir,
    )

    print("=" * 70)
    print(f"Sample {args.sample_idx}: {sample['maze_id']}")
    print(f"Grid: {sample['rows']}x{sample['cols']}")
    print(f"Entry: {sample['entry']} -> Exit: {sample['exit']}")
    print(f"Expected path: {' -> '.join(sample['path_directions'])}")
    print(f"Work dir: {work_dir}")
    print("=" * 70)

    # Setup
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)
    mcp_client = MCPClient(MCP_SERVER_PATH)
    tools = get_vision_tools(mcp_client)

    print(f"Vision tools: {[t['function']['name'] for t in tools]}")

    # Initial message - ask model to find the path
    question = f"This is a {sample['rows']}x{sample['cols']} maze. Find the shortest path from the entry point (marked at grid position {sample['entry']}) to the exit point (at grid position {sample['exit']}). Describe the path as a sequence of directions (up, down, left, right)."
    user_text = f"Image path: {image_path}\n\nQuestion: {question}"
    messages = [
        {"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": user_text}
        ]}
    ]
    logger.log_user_message(user_text, [f"data:image/png;base64,{image_b64}"])

    # Tool loop
    for iteration in range(args.max_iterations):
        logger.log_iteration(iteration + 1)
        print(f"\n--- Iteration {iteration + 1} ---")

        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=tools,
            tool_choice="auto",
            max_tokens=2048,
            temperature=0,
        )

        msg = response.choices[0].message
        content = msg.content or ""
        finish_reason = response.choices[0].finish_reason

        print(f"Finish: {finish_reason}")
        print(f"Content: {content[:300]}..." if len(content) > 300 else f"Content: {content}")

        # Log assistant message
        tool_calls_for_log = []
        if msg.tool_calls:
            for tc in msg.tool_calls:
                tool_calls_for_log.append({
                    "id": tc.id,
                    "name": tc.function.name,
                    "arguments": tc.function.arguments,
                })
        logger.log_assistant_message(content, tool_calls_for_log)

        # No tool calls - done
        if finish_reason != "tool_calls" or not msg.tool_calls:
            print(f"\nFinal answer: {content}")
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

            print(f"  Tool: {name}({json.dumps(args_dict)})")

            try:
                result = run_tool(mcp_client, name, args_dict)

                # Log tool result
                logger.log_tool_result(
                    tool_name=name,
                    tool_call_id=tc.id,
                    result={"args": args_dict, "success": True},
                    image_b64=result.get("image_b64"),
                )

                # Add tool result
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Tool {name} executed successfully."
                })

                # If image returned, add as user message so model can see it
                if "image_b64" in result:
                    # Save image
                    img_path = os.path.join(work_dir, f"iter{iteration}_{name}.png")
                    img_data = result["image_b64"].split(",")[1] if "," in result["image_b64"] else result["image_b64"]
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(img_data))
                    print(f"    -> Image saved: {img_path}")

                    followup_text = "Tool result above. Continue with your analysis and provide your final answer."

                    messages.append({
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": result["image_b64"]}},
                            {"type": "text", "text": followup_text}
                        ]
                    })
                    logger.log_user_message(followup_text, [result["image_b64"]])

            except Exception as e:
                print(f"    -> Error: {e}")
                logger.log_tool_result(
                    tool_name=name,
                    tool_call_id=tc.id,
                    result={"args": args_dict, "error": str(e)},
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": f"Error: {e}"
                })

    # Save HTML log
    html_path = logger.save()
    print("\n" + "=" * 70)
    print(f"Expected: {' -> '.join(sample['path_directions'])}")
    print(f"Work dir: {work_dir}")
    print(f"HTML log: {html_path}")


if __name__ == "__main__":
    main()
