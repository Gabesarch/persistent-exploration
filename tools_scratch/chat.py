#!/usr/bin/env python3
"""
Interactive chat with tool calling using MMSearch's search utilities.

Usage:
    python chat.py --base-url http://localhost:8000 --model qwen
"""

import argparse
import base64
import datetime as dt
import json
import os
import re
import sys
import tempfile
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("Missing: pip install openai", file=sys.stderr)
    sys.exit(1)

# Add current dir to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import MMSearch utilities
from utils.utils import (
    search_text_brief_result,
    search_url_full_result,
    search_by_image,
    get_fullpage_content,
    BRIEF_TIMEOUT,
)

# Tool definitions for Qwen3 format
TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web for information. Returns titles, snippets, URLs, and screenshot paths of search results.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "num_results": {"type": "integer", "description": "Number of results (default: 5)"}
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "image_search",
            "description": "Reverse image search using Google Lens. Returns similar images and source pages.",
            "parameters": {
                "type": "object",
                "properties": {
                    "image_url": {"type": "string", "description": "URL of the image to search"}
                },
                "required": ["image_url"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "visit_webpage",
            "description": "Visit a webpage and extract its full content and screenshot.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "URL to visit"}
                },
                "required": ["url"]
            }
        }
    }
]

SYSTEM_PROMPT = """You are a helpful AI assistant with access to web search tools.

Available tools:
1. web_search(query, num_results=5) - Search the web, returns titles/snippets/URLs with screenshots
2. image_search(image_url) - Reverse image search via Google Lens
3. visit_webpage(url) - Get full content and screenshot of a webpage

To use a tool, output:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>

After receiving results, synthesize the information to answer the user's question.
For dates, use yyyy-mm-dd format. Be concise."""


def extract_tool_calls(content: str) -> list[dict] | None:
    """Extract tool calls from Qwen <tool_call> tags."""
    regex = re.compile(r"<tool_call>(.+?)</tool_call>", re.DOTALL)
    matches = regex.findall(content)
    if not matches:
        return None

    calls = []
    for m in matches:
        try:
            parsed = json.loads(m.strip())
            if "name" in parsed:
                calls.append(parsed)
        except json.JSONDecodeError:
            continue
    return calls if calls else None


def execute_web_search(query: str, num_results: int = 5, temp_dir: str = "/tmp/search") -> dict:
    """Execute web search using MMSearch's DuckDuckGo + Playwright."""
    screenshot_dir = os.path.join(temp_dir, f"search_{dt.datetime.now().strftime('%H%M%S')}")

    try:
        results = search_text_brief_result(query, num_results, screenshot_dir)
        if results is None:
            return {"query": query, "error": "Search returned no results", "results": []}

        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "url": r.get("url", ""),
                "screenshot_path": r.get("screenshot_path", "")
            })

        return {"query": query, "num_results": len(formatted), "results": formatted}
    except Exception as e:
        return {"query": query, "error": str(e), "results": []}


def execute_image_search(image_url: str, temp_dir: str = "/tmp/search") -> dict:
    """Execute reverse image search using Google Lens."""
    screenshot_path = os.path.join(temp_dir, f"lens_{dt.datetime.now().strftime('%H%M%S')}.png")

    try:
        results = search_by_image(image_url, screenshot_path)
        return {
            "image_url": image_url,
            "num_results": len(results),
            "results": results,
            "screenshot_path": screenshot_path
        }
    except Exception as e:
        return {"image_url": image_url, "error": str(e), "results": []}


def execute_visit_webpage(url: str, temp_dir: str = "/tmp/search") -> dict:
    """Visit webpage and extract content."""
    screenshot_dir = os.path.join(temp_dir, f"page_{dt.datetime.now().strftime('%H%M%S')}")
    os.makedirs(screenshot_dir, exist_ok=True)

    try:
        results = search_url_full_result([url], screenshot_dir)
        if results and len(results) > 0:
            r = results[0]
            return {
                "url": url,
                "content": r.get("content", "")[:8000],
                "screenshot_path": os.path.join(screenshot_dir, r.get("screenshot_fullpage_path", ""))
            }
        return {"url": url, "error": "Failed to fetch page", "content": ""}
    except Exception as e:
        return {"url": url, "error": str(e), "content": ""}


def execute_tool(name: str, args: dict, temp_dir: str) -> dict:
    """Execute a tool by name."""
    if name == "web_search":
        return execute_web_search(
            args.get("query", ""),
            args.get("num_results", 5),
            temp_dir
        )
    elif name == "image_search":
        return execute_image_search(args.get("image_url", ""), temp_dir)
    elif name == "visit_webpage":
        return execute_visit_webpage(args.get("url", ""), temp_dir)
    else:
        return {"error": f"Unknown tool: {name}"}


def encode_image_to_base64(path: str) -> str | None:
    """Encode image file to base64."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "rb") as f:
            return base64.b64encode(f.read()).decode("ascii")
    except Exception:
        return None


def build_user_content(text: str, image_path: str | None = None) -> list[dict]:
    """Build user message content with optional image."""
    content = []
    if image_path:
        if image_path.startswith(("http://", "https://")):
            content.append({"type": "image_url", "image_url": {"url": image_path}})
        elif os.path.exists(image_path):
            b64 = encode_image_to_base64(image_path)
            if b64:
                content.append({"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}})
    content.append({"type": "text", "text": text})
    return content


def extract_text(content: Any) -> str:
    """Extract text from message content."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    return str(content)


def call_model(client: OpenAI, messages: list, model: str, **kwargs) -> str:
    """Call model and return response text."""
    resp = client.chat.completions.create(
        model=model,
        messages=messages,
        **kwargs
    )
    return extract_text(resp.choices[0].message.content)


def run_tool_loop(
    client: OpenAI,
    messages: list,
    model: str,
    temp_dir: str,
    max_iterations: int = 3,
    verbose: bool = True,
    **kwargs
) -> tuple[str, list[dict]]:
    """Run conversation with tool calling loop."""
    tool_history = []

    for i in range(max_iterations):
        response = call_model(client, messages, model, **kwargs)

        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            return response, tool_history

        messages.append({"role": "assistant", "content": response})

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("arguments", {})

            if verbose:
                print(f"\n[Tool Call] {name}({json.dumps(args)[:80]}...)")

            result = execute_tool(name, args, temp_dir)
            tool_history.append({"tool": name, "args": args, "result": result})

            # Truncate large results
            result_str = json.dumps(result, ensure_ascii=False)
            if len(result_str) > 6000:
                result_str = result_str[:6000] + "...[truncated]"

            if verbose:
                print(f"[Result] {result_str[:200]}...")

            messages.append({
                "role": "tool",
                "content": result_str,
                "name": name
            })

    # Final response after tool calls
    final = call_model(client, messages, model, **kwargs)
    return final, tool_history


def main():
    parser = argparse.ArgumentParser(description="Chat with MMSearch tools")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", default="EMPTY", help="API key")
    parser.add_argument("--model", default="qwen", help="Model name")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temp-dir", default="/tmp/mmsearch_chat", help="Temp dir for screenshots")
    parser.add_argument("--verbose", action="store_true", default=True)
    args = parser.parse_args()

    os.makedirs(args.temp_dir, exist_ok=True)

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    client = OpenAI(api_key=args.api_key, base_url=base_url, timeout=120)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    print(f"Connected to {base_url}")
    print(f"Model: {args.model}")
    print(f"Temp dir: {args.temp_dir}")
    print("\nCommands: /help, /reset, /tools, /exit")
    print("-" * 50)

    while True:
        try:
            text = input("\nYou> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break

        if not text:
            continue
        if text.lower() == "/exit":
            break
        if text.lower() == "/help":
            print("Commands: /help, /reset, /tools, /exit")
            print("Optionally provide image path after your message")
            continue
        if text.lower() == "/reset":
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("Conversation reset.")
            continue
        if text.lower() == "/tools":
            for t in TOOLS:
                f = t["function"]
                print(f"  - {f['name']}: {f['description'][:60]}...")
            continue

        # Optional image input
        image_path = input("Image (optional)> ").strip() or None
        if image_path and not image_path.startswith("http") and not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue

        user_content = build_user_content(text, image_path)
        messages.append({"role": "user", "content": user_content})

        try:
            response, tool_history = run_tool_loop(
                client, messages, args.model, args.temp_dir,
                verbose=args.verbose,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )

            print(f"\nAssistant> {response}")
            messages.append({"role": "assistant", "content": response})

            if tool_history:
                print(f"\n[Used {len(tool_history)} tool(s)]")

        except Exception as e:
            print(f"Error: {e}")
            messages.pop()  # Remove failed user message


if __name__ == "__main__":
    main()
