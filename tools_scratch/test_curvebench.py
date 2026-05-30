#!/usr/bin/env python3
"""
Test Qwen3-VL tool calling on CurveBench-Easy dataset (curve/tree tracing).
Uses vLLM with hermes parser (--enable-auto-tool-choice --tool-call-parser hermes).

Offline usage (interactive node without internet):
    conda activate verovlm
    export HF_HOME=/scratch/gpfs/ZHUANGL/hf_cache
    export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1

Usage:
    python test_curvebench.py --sample-idx 0 --level 3
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
CURVEBENCH_CACHE_DIR = "/scratch/gpfs/ZHUANGL/hw0171/curvebench"
MCP_SERVER_PATH = str(Path(__file__).parent / "eval/lmms_eval/mcp/vero2_mcp.py")

BASE_URL = os.getenv("OPENAI_API_BASE", "http://localhost:9001/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "qwen")
MODEL = os.getenv("MODEL_NAME", "qwen_vllm")


def load_sample(idx: int, level: int = 3, split: str = "train") -> dict:
    """Load a sample from CurveBench dataset."""
    ds = load_dataset("AmirMohseni/CurveBench-Easy", cache_dir=CURVEBENCH_CACHE_DIR)
    split_name = f"level_{level}_{split}"
    if split_name not in ds:
        split_name = f"total_{split}"
    sample = ds[split_name][idx]
    return {
        "image": sample["image"],
        "num_nodes": sample["num_nodes"],
        "tree": sample["tree"],
        "level": level,
        "split": split_name,
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
    # All drawing tools for curve tracing
    tool_names = {
        "draw_arrow",           # Draw arrows to show connections
        "draw_line",            # Draw lines to trace curves
        "draw_circle",          # Mark nodes/endpoints
        "draw_box",             # Highlight regions
        "draw_text",            # Add text annotations/labels
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
    parser.add_argument("--level", type=int, default=3, choices=[1, 2, 3])
    parser.add_argument("--split", type=str, default="train", choices=["train", "validation", "test"])
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--output", type=str, default=None, help="Output HTML path")
    args = parser.parse_args()

    # Load sample
    sample = load_sample(args.sample_idx, args.level, args.split)
    work_dir = tempfile.mkdtemp()
    image_path = os.path.join(work_dir, "input.png")
    sample["image"].save(image_path)
    image_b64 = image_to_b64(sample["image"])

    # Format tree as readable string
    tree_str = str(sample["tree"])

    # Setup logger
    output_path = args.output or f"curvebench_level{args.level}_{args.sample_idx}.html"
    logger = ConversationLogger(output_path, title=f"CurveBench Level {args.level} (Sample {args.sample_idx})")
    logger.set_metadata(
        sample_idx=args.sample_idx,
        level=args.level,
        split=sample["split"],
        num_nodes=sample["num_nodes"],
        expected_tree=tree_str,
        image=f"data:image/png;base64,{image_b64}",
        work_dir=work_dir,
    )

    print("=" * 70)
    print(f"Sample {args.sample_idx} (Level {args.level}, {sample['split']})")
    print(f"Num nodes: {sample['num_nodes']}")
    print(f"Expected tree: {tree_str}")
    print(f"Work dir: {work_dir}")
    print("=" * 70)

    # Setup
    client = OpenAI(api_key=API_KEY, base_url=BASE_URL, timeout=120)
    mcp_client = MCPClient(MCP_SERVER_PATH)
    tools = get_vision_tools(mcp_client)

    print(f"Vision tools: {[t['function']['name'] for t in tools]}")

    # Initial message - ask model to trace the curve structure
    question = """Analyze this image and extract the hierarchical tree structure representing the nested regions.
The image contains nested shapes/regions. Your task is to identify the parent-child relationships between these regions.
Return the tree structure as a list of edges, where each edge is represented as (child, parent).
• The root node is always 0.
• Each region is assigned a unique node number.
• Edges represent parent-child relationships, where a parent region contains a child region.
Format your response inside <answer>...</answer> tags.
The first line should be the number of nodes excluding the root. Each subsequent line should be u v, meaning an edge from v to u, where v is the parent and u is the child.
Example:
<answer>
3
1 0
2 0
3 1
</answer>
Make sure to include all edges that represent the hierarchical structure."""
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
    print(f"Expected tree: {tree_str}")
    print(f"Work dir: {work_dir}")
    print(f"HTML log: {html_path}")


if __name__ == "__main__":
    main()
