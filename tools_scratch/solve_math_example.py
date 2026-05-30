#!/usr/bin/env python3
"""
Solve math geometry problem using Qwen3-VL with Visual Sketchpad drawing tools.
Uses the async_openai model from lmms-eval with MCP tools.
"""
import asyncio
import base64
import os

# Bypass proxy for localhost (cluster proxy intercepts otherwise)
os.environ["no_proxy"] = "localhost,127.0.0.1"
os.environ["NO_PROXY"] = "localhost,127.0.0.1"
import sys
import tempfile
from pathlib import Path

# Add lmms_eval to path
sys.path.insert(0, str(Path(__file__).parent / "eval"))

from openai import AsyncOpenAI

# Problem setup - Parallel lines angle problem
IMAGE_PATH = "/home/hw0171/reason-vlm-sft/parallel_angle.png"
PROBLEM_TEXT = "Lines l1 and l2 are parallel. Point A is on l1, point B is on l2, and point P is between them. The angle between segment PA and line l1 is 35 degrees. The angle between segment PB and line l2 is 45 degrees. Find the measure of angle APB (the angle at point P)."

# Point coordinates for parallel_angle.png (0-1000 scale for MCP tools)
# Calculated from create_parallel_angle.py: xlim=(0,620), ylim=(0,600)
POINT_COORDS = {
    "A": (784, 167),   # On top line l1
    "P": (323, 500),   # Point between lines
    "B": (645, 833),   # On bottom line l2
}

# Qwen3-VL server config (matches serve_qwen3vl.sh)
BASE_URL = os.getenv("OPENAI_API_BASE", "http://localhost:9001/v1")
API_KEY = os.getenv("OPENAI_API_KEY", "qwen")
MODEL = os.getenv("MODEL_NAME", "qwen_vllm")

# MCP server path
MCP_SERVER_PATH = str(Path(__file__).parent / "eval/lmms_eval/mcp/vero2_mcp.py")


def encode_image_base64(image_path: str) -> str:
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


async def solve_with_mcp():
    """Solve the problem using MCP tools (like lmms-eval does)."""
    from lmms_eval.mcp import MCPClient

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    mcp_client = MCPClient(MCP_SERVER_PATH)
    work_dir = tempfile.mkdtemp()

    # Save image to work dir for MCP tools
    import shutil
    image_work_path = os.path.join(work_dir, "math_example.png")
    shutil.copy(IMAGE_PATH, image_work_path)

    # Get tool definitions from MCP server
    functions = await mcp_client.get_function_list()
    print(f"Available tools: {[f['function']['name'] for f in functions]}")
    import json
    print(f"Tool definitions:\n{json.dumps(functions[:2], indent=2)}...")  # Print first 2

    # Build messages
    image_b64 = encode_image_base64(IMAGE_PATH)

    system_prompt = """You are a geometry expert with drawing tools to annotate images.

## Available Tools
To use a tool, write a JSON block like this:
```tool
{"name": "draw_line", "arguments": {"image_path": "IMAGE_PATH", "x1": 100, "y1": 200, "x2": 500, "y2": 600, "color": "red"}}
```

Tools available:
- draw_line: Draw a line from (x1,y1) to (x2,y2). Args: image_path, x1, y1, x2, y2, color
- draw_circle: Draw a circle at (x,y) with radius. Args: image_path, x, y, radius, color
- draw_text: Add text label at (x,y). Args: image_path, x, y, text, color

Coordinates use 0-1000 scale: (0,0)=top-left, (1000,1000)=bottom-right.

## Instructions
1. Look at the image carefully and identify key points
2. Use draw_line to mark important lines or segments
3. Reason through the geometry step by step
4. Give your final answer

When you want to use a tool, output the ```tool block. I will execute it and show you the result."""

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": f"Image path: {image_work_path}"},
        {"type": "text", "text": f"""Solve this geometry problem:

{PROBLEM_TEXT}

Choices:
A) {CHOICES[0]}
B) {CHOICES[1]}
C) {CHOICES[2]}
D) {CHOICES[3]}

Use drawing tools if helpful. Explain your reasoning and give your final answer."""},
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    print("=" * 60)
    print("Solving geometry problem with Qwen3-VL + MCP Tools")
    print("=" * 60)
    print(f"\nProblem: {PROBLEM_TEXT}")
    print(f"Choices: {CHOICES}")
    print("\n" + "-" * 60)
    print("SYSTEM PROMPT:")
    print(system_prompt)
    print("-" * 60)
    print(f"USER CONTENT (text parts): {[c for c in user_content if c.get('type') == 'text']}")
    print("-" * 60)

    # First API call - use "required" to force tool use on first turn only
    first_call = True
    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        tools=functions,
        tool_choice="required",
        max_tokens=4096,
        temperature=0,
    )

    all_response = ""
    last_response = response.choices[0].message.content or ""
    all_response += last_response
    print(f"\nFinish reason: {response.choices[0].finish_reason}")
    print(f"Tool calls: {response.choices[0].message.tool_calls}")
    print(f"\nModel: {last_response}")

    # Handle tool calls
    max_iterations = 10
    iteration = 0

    while response.choices[0].finish_reason == "tool_calls" and iteration < max_iterations:
        iteration += 1
        print(f"\n[Tool iteration {iteration}]")

        # Add assistant message with tool calls (single message, not two)
        assistant_msg = {"role": "assistant", "content": last_response or None}
        if response.choices[0].message.tool_calls:
            assistant_msg["tool_calls"] = [
                {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                for tc in response.choices[0].message.tool_calls
            ]
        messages.append(assistant_msg)

        # Process each tool call and add tool result messages
        for call in response.choices[0].message.tool_calls:
            print(f"  >>> TOOL CALL: {call.function.name}")
            print(f"      Arguments: {call.function.arguments}")

            # Run the tool via MCP
            result = await mcp_client.run_tool(call.function.name, eval(call.function.arguments))

            all_response += f"<tool_call>{call.function.name} {call.function.arguments}</tool_call>"

            # Build tool result - inject image as user message so model can see it
            tool_content_text = f"Tool {call.function.name} executed successfully."
            image_content = None

            for content in result.content:
                tool_message = mcp_client.convert_result_to_openai_format(content)
                for c in tool_message:
                    if c["type"] == "image_url":
                        # Save image to disk for inspection
                        img_data = c["image_url"]["url"].split(",")[1]
                        img_bytes = base64.b64decode(img_data)
                        img_path = os.path.join(work_dir, f"tool_output_{iteration}_{call.function.name}.png")
                        with open(img_path, "wb") as f:
                            f.write(img_bytes)
                        print(f"      <<< IMAGE RETURNED: saved to {img_path}")
                        image_content = c["image_url"]["url"]
                        all_response += "<image>"
                    elif c["type"] == "text":
                        print(f"      <<< TEXT RETURNED: {c['text'][:200]}")
                        tool_content_text += " " + c["text"]
                        all_response += c["text"]
            all_response += "</tool_response>"

            # Add tool result message (text only for vLLM compatibility)
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": tool_content_text,
            })

            # Inject annotated image as user message so model can see it
            if image_content:
                messages.append({
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_content}},
                        {"type": "text", "text": "Here is the result of your drawing. Continue analyzing the geometry problem."},
                    ],
                })

        # Continue conversation - use "auto" after first call
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=functions,
            tool_choice="auto",
            max_tokens=4096,
            temperature=0,
        )

        last_response = response.choices[0].message.content or ""
        all_response += last_response
        print(f"\nModel: {last_response}")

    print("\n" + "=" * 60)
    print("Solution complete")
    print(f"Work directory with images: {work_dir}")
    print("=" * 60)

    return all_response


async def solve_with_text_tools():
    """Solve using text-based tool calling with MCP tools."""
    import json
    import re
    import shutil
    from lmms_eval.mcp import MCPClient

    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)
    mcp_client = MCPClient(MCP_SERVER_PATH)
    work_dir = tempfile.mkdtemp()

    # Save image to work dir
    image_work_path = os.path.join(work_dir, "geometry.png")
    shutil.copy(IMAGE_PATH, image_work_path)

    image_b64 = encode_image_base64(IMAGE_PATH)

    # Simple system prompt
    system_prompt = """You are a helpful multimodal AI assistant.
For each turn, you should first do a "THOUGHT", based on the images and text you see.
If you use a tool, STOP immediately after the tool call. Do NOT say TERMINATE - wait for the OBSERVATION.
Only use "ANSWER: <your answer>" followed by "TERMINATE" when you have the final numerical answer."""

    # Build user content with point coordinates (0-1000 scale for MCP tools)
    coords_str = "\n".join([f"  {name}: ({x}, {y})" for name, (x, y) in POINT_COORDS.items()])

    text_logic_form = """[
    "Parallel(Line(l1), Line(l2))",
    "PointLiesOnLine(A, Line(l1))",
    "PointLiesOnLine(B, Line(l2))",
    "Equals(MeasureOf(Angle(PA, l1)), 35)",
    "Equals(MeasureOf(Angle(PB, l2)), 45)",
    "Find(MeasureOf(Angle(APB)))"
]"""

    diagram_logic_form = """[
    "PointLiesOnLine(A, Line(l1))",
    "PointLiesOnLine(B, Line(l2))",
    "Parallel(Line(l1), Line(l2))",
    "PointBetweenLines(P, Line(l1), Line(l2))"
]"""

    # Check if we should include coordinates
    include_coords = os.getenv("INCLUDE_COORDS", "0") == "1"

    question_text = f"""Here are some tools that can help you draw auxiliary lines on the geometry image.
Notice that the upper left corner of the image is the origin (0, 0). Coordinates use a 0-1000 scale.

To use a tool, write a JSON block like this:
```tool
{{"name": "draw_line_simple", "arguments": {{"image_path": "{image_work_path}", "x1": X1, "y1": Y1, "x2": X2, "y2": Y2, "color": "red"}}}}
```

Available tools:
- draw_line_simple: Draw a line from (x1,y1) to (x2,y2). Args: image_path, x1, y1, x2, y2, color

# GOAL #: You MUST draw auxiliary lines to visualize the geometry before solving. Always draw at least one line.
# Hint #: When deciding what type of auxiliary line to draw, consider:
(1) Perpendicular line: Draw a line perpendicular to an existing line through a point.
(2) Parallel line: Draw a line parallel to an existing line through a point.
(3) Connecting points: Draw lines connecting key points to reveal angle relationships.
(4) Extending lines: Extend existing segments to identify alternate or corresponding angles.

# USER REQUEST #: Given the geometry diagram, solve the following question: [{PROBLEM_TEXT}]
"""

    if include_coords:
        question_text += f"""
Point coordinates (0-1000 scale):
{coords_str}
"""

    question_text += """
Look at the image to estimate where the points are located. Draw auxiliary lines to help solve the problem.

IMPORTANT: After each ACTION (tool call), STOP and wait for the OBSERVATION. Do not continue until you see the result.

Generate THOUGHT and ACTION:
# RESULT #:"""

    user_content = [
        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
        {"type": "text", "text": question_text},
    ]

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    print("=" * 60)
    print("Solving with TEXT-BASED MCP tool calling")
    print("=" * 60)
    print(f"Work dir: {work_dir}")
    print(f"Input image: {image_work_path}")
    print("-" * 60)

    # HTML log
    html_log = []
    html_log.append(f"<h2>System Prompt</h2><pre>{system_prompt}</pre>")
    html_log.append(f"<h2>User Message</h2>")
    html_log.append(f"<img src='data:image/png;base64,{image_b64}' style='max-width:600px'><br>")
    html_log.append(f"<pre>{question_text}</pre>")

    max_iterations = 5
    for iteration in range(max_iterations):
        response = await client.chat.completions.create(
            model=MODEL,
            messages=messages,
            max_tokens=4096,
            temperature=0,
        )

        content = response.choices[0].message.content or ""
        print(f"\n[Turn {iteration + 1}]")
        print(f"Model:\n{content}")

        # Log to HTML
        html_log.append(f"<h2>Turn {iteration + 1} - Model Response</h2>")
        html_log.append(f"<pre>{content}</pre>")

        # Check for tool calls in ```tool blocks
        tool_pattern = r'```tool\s*\n?(.*?)\n?```'
        tool_matches = re.findall(tool_pattern, content, re.DOTALL)

        # Only terminate if ANSWER is present and no tool calls
        if "TERMINATE" in content and "ANSWER" in content and not tool_matches:
            print("\nModel finished with ANSWER + TERMINATE")
            break

        if not tool_matches:
            print("\nNo tool calls found - model finished reasoning")
            break

        # Add assistant response
        messages.append({"role": "assistant", "content": content})

        # Execute each tool call via MCP
        tool_call_count = 0
        last_image_content = None
        for tool_json in tool_matches:
            try:
                tool_data = json.loads(tool_json.strip())
                tool_name = tool_data["name"]
                tool_args = tool_data["arguments"]

                # Update image_path to use the latest modified image
                if "image_path" in tool_args:
                    tool_args["image_path"] = image_work_path

                print(f"\n  >>> TOOL CALL: {tool_name}")
                print(f"      Arguments: {tool_args}")

                result = await mcp_client.run_tool(tool_name, tool_args)

                # Process result
                for c in result.content:
                    tool_message = mcp_client.convert_result_to_openai_format(c)
                    for item in tool_message:
                        if item["type"] == "image_url":
                            img_data = item["image_url"]["url"].split(",")[1]
                            img_bytes = base64.b64decode(img_data)
                            # Save with unique name and also update the working image
                            img_path = os.path.join(work_dir, f"action_{iteration}_{tool_call_count}_{tool_name}.png")
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                            # Update the working image so next tool call builds on this
                            with open(image_work_path, "wb") as f:
                                f.write(img_bytes)
                            last_image_content = item["image_url"]["url"]
                            tool_call_count += 1
                            print(f"      <<< IMAGE saved to: {img_path}")

            except json.JSONDecodeError as e:
                print(f"  !!! Failed to parse tool JSON: {e}")
            except Exception as e:
                print(f"  !!! Tool execution error: {e}")
                import traceback
                traceback.print_exc()

        # After all tool calls, add ONE observation with the final image
        if last_image_content:
            messages.append({
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": last_image_content}},
                    {"type": "text", "text": f"OBSERVATION: Drew {tool_call_count} line(s). The output is shown above.\n\nPlease generate the next THOUGHT and ACTION. If you can get the answer, reply with ANSWER: <your answer> and ends with TERMINATE."},
                ],
            })
            print(f"      <<< ADDED TO MESSAGES: final image with {tool_call_count} drawings")

            # Log to HTML
            html_log.append(f"<h3>Tool Result (Turn {iteration + 1})</h3>")
            html_log.append(f"<img src='{last_image_content}' style='max-width:600px'><br>")
            html_log.append(f"<p>Drew {tool_call_count} line(s)</p>")

        # Debug: show message count and structure before next API call
        print(f"\n  DEBUG: Message history ({len(messages)} messages):")
        for i, msg in enumerate(messages):
            role = msg["role"]
            if isinstance(msg["content"], str):
                content_preview = msg["content"][:80] + "..." if len(msg["content"]) > 80 else msg["content"]
                print(f"    [{i}] {role}: {content_preview}")
            elif isinstance(msg["content"], list):
                types = [c.get("type", "?") for c in msg["content"]]
                print(f"    [{i}] {role}: {types}")

    # Save HTML log
    html_path = os.path.join(work_dir, "conversation.html")
    with open(html_path, "w") as f:
        f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Geometry Problem Conversation Log</title>
    <style>
        body {{ font-family: Arial, sans-serif; max-width: 900px; margin: 0 auto; padding: 20px; }}
        pre {{ background: #f4f4f4; padding: 10px; overflow-x: auto; white-space: pre-wrap; }}
        img {{ border: 1px solid #ccc; margin: 10px 0; }}
        h2 {{ color: #333; border-bottom: 1px solid #ccc; }}
        h3 {{ color: #666; }}
    </style>
</head>
<body>
<h1>Geometry Problem Conversation Log</h1>
{''.join(html_log)}
</body>
</html>""")
    print(f"\nHTML log saved to: {html_path}")

    print("\n" + "=" * 60)
    print("Solution complete")
    print(f"Work directory: {work_dir}")
    print("=" * 60)

    return content


async def solve_simple():
    """Solve without MCP tools (simpler, just reasoning)."""
    client = AsyncOpenAI(api_key=API_KEY, base_url=BASE_URL)

    image_b64 = encode_image_base64(IMAGE_PATH)

    messages = [
        {"role": "system", "content": "You are a geometry expert. Solve problems step by step."},
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                {"type": "text", "text": f"""Solve this geometry problem:

{PROBLEM_TEXT}

Choices:
A) {CHOICES[0]}
B) {CHOICES[1]}
C) {CHOICES[2]}
D) {CHOICES[3]}

Think step by step:
1. What is given?
2. What theorem applies? (Hint: Equal chords have equal arcs)
3. What is the answer?"""},
            ],
        },
    ]

    print("=" * 60)
    print("Solving geometry problem with Qwen3-VL (no tools)")
    print("=" * 60)
    print(f"\nProblem: {PROBLEM_TEXT}")
    print(f"Choices: {CHOICES}")
    print("\n" + "-" * 60)

    response = await client.chat.completions.create(
        model=MODEL,
        messages=messages,
        max_tokens=2048,
        temperature=0,
    )

    result = response.choices[0].message.content
    print(f"\nModel response:\n{result}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-tools", action="store_true", help="Use vLLM native tool calling (requires --enable-auto-tool-choice)")
    parser.add_argument("--text-tools", action="store_true", help="Use text-based tool calling (no special vLLM flags needed)")
    args = parser.parse_args()

    if args.use_tools:
        asyncio.run(solve_with_mcp())
    elif args.text_tools:
        asyncio.run(solve_with_text_tools())
    else:
        asyncio.run(solve_simple())
