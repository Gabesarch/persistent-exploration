#!/usr/bin/env python3
"""
Test tool-calling on SimpleVQA-EN dataset. 5/15

Usage:
    conda activate Behav
    python test_simplevqa.py --base-url http://localhost:8000 --model qwen --num-samples 5 --log-file detailed_log.jsonl
"""

import argparse
import base64
import datetime
import json
import logging
import os
import re
import sys
import io
import threading
import http.server
import socketserver
from typing import Any

try:
    from openai import OpenAI
except ImportError:
    print("Missing: pip install openai", file=sys.stderr)
    sys.exit(1)

import pyarrow.parquet as pq
from PIL import Image

# Add parent dir (tools_scratch) to path for utils import
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.utils import (
    search_text_brief_result,
    search_url_full_result,
    search_by_image,
    PROXY,
)


class ImageServer:
    """Upload images to imgbb for public hosting (needed for image_search)."""

    def __init__(self, imgbb_api_key: str = None, image_dir: str = "/tmp/hosted_images"):
        self.imgbb_api_key = imgbb_api_key
        self.image_dir = image_dir
        self.url_cache = {}  # Cache uploaded URLs
        os.makedirs(image_dir, exist_ok=True)

    def start(self):
        """No-op for compatibility."""
        if self.imgbb_api_key:
            print(f"Image hosting via imgbb enabled")
        else:
            print("WARNING: No imgbb API key provided. Image search will not work.")
            print("Get a free key at: https://api.imgbb.com/")

    def stop(self):
        """No-op for compatibility."""
        pass

    def host_image(self, image_bytes: bytes, filename: str = None) -> str:
        """Upload image to imgbb and return public URL."""
        if filename is None:
            filename = f"img_{datetime.datetime.now().strftime('%H%M%S%f')}.png"

        # Check cache first
        cache_key = hash(image_bytes)
        if cache_key in self.url_cache:
            return self.url_cache[cache_key]

        # Save locally as backup
        filepath = os.path.join(self.image_dir, filename)
        with open(filepath, "wb") as f:
            f.write(image_bytes)

        # Upload to imgbb if API key is available
        if self.imgbb_api_key:
            try:
                import requests
                b64_image = base64.b64encode(image_bytes).decode("ascii")
                response = requests.post(
                    "https://api.imgbb.com/1/upload",
                    data={
                        "key": self.imgbb_api_key,
                        "image": b64_image,
                        "name": filename.replace(".png", ""),
                    },
                    timeout=30
                )
                if response.status_code == 200:
                    url = response.json()["data"]["url"]
                    self.url_cache[cache_key] = url
                    return url
                else:
                    print(f"imgbb upload failed: {response.text}")
            except Exception as e:
                print(f"imgbb upload error: {e}")

        # Fallback to local URL (won't work for image_search)
        return f"file://{filepath}"

    def host_base64_image(self, b64_data: str, filename: str = None) -> str:
        """Decode base64 image and host it."""
        image_bytes = base64.b64decode(b64_data)
        return self.host_image(image_bytes, filename)


# Global image server instance
image_server: ImageServer | None = None


class DetailedLogger:
    """Logs detailed information about each sample's processing."""

    def __init__(self, log_file: str | None = None):
        self.log_file = log_file
        self.current_sample = {}
        self.entries = []

        if log_file:
            # Clear existing file
            with open(log_file, "w") as f:
                pass

    def start_sample(self, sample_id: int, question: str, gt_answer: str, source: str = "", image_base64: str = ""):
        self.current_sample = {
            "sample_id": sample_id,
            "question": question,
            "gt_answer": gt_answer,
            "source": source,
            "image_base64": image_base64,
            "start_time": datetime.datetime.now().isoformat(),
            "messages": [],
            "tool_calls": [],
            "model_responses": [],
        }

    def log_message(self, role: str, content: Any, timestamp: str | None = None):
        ts = timestamp or datetime.datetime.now().isoformat()
        # For content that contains images, mark them but don't store full base64
        if isinstance(content, list):
            logged_content = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "image_url":
                    url = item.get("image_url", {}).get("url", "")
                    if url.startswith("data:"):
                        logged_content.append({"type": "image_url", "image_url": {"url": f"[BASE64_IMAGE, {len(url)} chars]"}})
                    else:
                        logged_content.append(item)
                else:
                    logged_content.append(item)
        else:
            logged_content = content

        self.current_sample["messages"].append({
            "timestamp": ts,
            "role": role,
            "content": logged_content,
        })

    def log_tool_call(self, tool_name: str, arguments: dict, result: dict, result_sent_to_model: str = ""):
        ts = datetime.datetime.now().isoformat()
        entry = {
            "timestamp": ts,
            "tool_name": tool_name,
            "arguments": arguments,
            "result_full": result,
            "result_sent_to_model": result_sent_to_model,
            "result_length": len(json.dumps(result, ensure_ascii=False)),
        }
        self.current_sample["tool_calls"].append(entry)

        # Print detailed info
        print(f"\n    {'='*50}")
        print(f"    TOOL CALL: {tool_name}")
        print(f"    Arguments: {json.dumps(arguments, indent=2, ensure_ascii=False)}")
        print(f"    Result preview ({entry['result_length']} chars total):")
        result_str = json.dumps(result, indent=2, ensure_ascii=False)
        print(f"    {result_str[:1500]}{'...' if len(result_str) > 1500 else ''}")
        print(f"    {'='*50}\n")

    def log_model_response(self, response: str, iteration: int):
        ts = datetime.datetime.now().isoformat()
        self.current_sample["model_responses"].append({
            "timestamp": ts,
            "iteration": iteration,
            "response": response,
        })

    def end_sample(self, prediction: str, f1: float, exact_match: bool, error: str | None = None):
        self.current_sample["end_time"] = datetime.datetime.now().isoformat()
        self.current_sample["prediction"] = prediction
        self.current_sample["f1"] = f1
        self.current_sample["exact_match"] = exact_match
        if error:
            self.current_sample["error"] = error

        self.entries.append(self.current_sample)

        if self.log_file:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(self.current_sample, ensure_ascii=False) + "\n")

    def get_summary(self) -> dict:
        total = len(self.entries)
        correct = sum(1 for e in self.entries if e.get("exact_match"))
        total_tools = sum(len(e.get("tool_calls", [])) for e in self.entries)
        avg_f1 = sum(e.get("f1", 0) for e in self.entries) / total if total else 0

        return {
            "total_samples": total,
            "correct": correct,
            "accuracy": correct / total if total else 0,
            "avg_f1": avg_f1,
            "total_tool_calls": total_tools,
            "avg_tools_per_sample": total_tools / total if total else 0,
        }


# Global logger instance
detailed_logger: DetailedLogger | None = None


SYSTEM_PROMPT = """You are a helpful AI assistant with access to web search tools.

Available tools:
1. web_search(query, num_results=5) - Search the web. Returns titles, short snippets, and URLs.
2. image_search(image_url) - Reverse image search to identify things in images.
3. visit_webpage(url) - Get the full text content of a webpage.

To use a tool, output:
<tool_call>
{"name": "tool_name", "arguments": {"arg": "value"}}
</tool_call>

Use these tools when you need more information to answer the question accurately. Give a concise final answer.
"""


def extract_tool_calls(content: str) -> list[dict] | None:
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
    screenshot_dir = os.path.join(temp_dir, f"search_{datetime.datetime.now().strftime('%H%M%S%f')}")
    try:
        results = search_text_brief_result(query, num_results, screenshot_dir)
        if results is None:
            return {"query": query, "error": "No results", "results": []}
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "snippet": r.get("snippet", ""),
                "url": r.get("url", ""),
            })
        return {"query": query, "num_results": len(formatted), "results": formatted}
    except Exception as e:
        return {"query": query, "error": str(e), "results": []}


def execute_visit_webpage(url: str, temp_dir: str = "/tmp/search") -> dict:
    screenshot_dir = os.path.join(temp_dir, f"page_{datetime.datetime.now().strftime('%H%M%S%f')}")
    os.makedirs(screenshot_dir, exist_ok=True)
    try:
        results = search_url_full_result([url], screenshot_dir)
        if results and len(results) > 0:
            return {
                "url": url,
                "content": results[0].get("content", "")[:6000],
            }
        return {"url": url, "error": "Failed", "content": ""}
    except Exception as e:
        return {"url": url, "error": str(e), "content": ""}


def execute_image_search(image_url: str, temp_dir: str = "/tmp/search") -> dict:
    screenshot_dir = os.path.join(temp_dir, f"imgsearch_{datetime.datetime.now().strftime('%H%M%S%f')}")
    os.makedirs(screenshot_dir, exist_ok=True)
    screenshot_path = os.path.join(screenshot_dir, "lens_results.png")
    try:
        results = search_by_image(image_url, screenshot_path)
        if results is None or len(results) == 0:
            return {"image_url": image_url, "error": "No results from image search", "results": []}
        formatted = []
        for r in results:
            source_url = r.get("web_url", r.get("source_url", ""))
            # Extract actual URL from Bing's mediaurl parameter if present
            if "mediaurl=" in source_url:
                import urllib.parse
                parsed = urllib.parse.parse_qs(urllib.parse.urlparse(source_url).query)
                if "mediaurl" in parsed:
                    source_url = parsed["mediaurl"][0]
            # Make relative URLs absolute
            if source_url.startswith("/"):
                source_url = "https://www.bing.com" + source_url
            formatted.append({
                "snippet": r.get("snippet", ""),
                "source_url": source_url,
                "image_url": r.get("image_url", ""),
            })
        return {"image_url": image_url, "num_results": len(formatted), "results": formatted}
    except Exception as e:
        return {"image_url": image_url, "error": str(e), "results": []}


def execute_tool(name: str, args: dict, temp_dir: str, logger: DetailedLogger | None = None) -> dict:
    if name == "web_search":
        result = execute_web_search(args.get("query", ""), args.get("num_results", 5), temp_dir)
    elif name == "visit_webpage":
        result = execute_visit_webpage(args.get("url", ""), temp_dir)
    elif name == "image_search":
        result = execute_image_search(args.get("image_url", ""), temp_dir)
    else:
        result = {"error": f"Unknown tool: {name}"}

    if logger:
        logger.log_tool_call(name, args, result)

    return result


def image_to_base64(image_bytes: bytes) -> str:
    return base64.b64encode(image_bytes).decode("ascii")


def extract_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            item.get("text", "") for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    return str(content)


def call_model(client: OpenAI, messages: list, model: str, **kwargs) -> str:
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
    max_iterations: int = 5,
    verbose: bool = True,
    logger: DetailedLogger | None = None,
    **kwargs
) -> tuple[str, list[dict]]:
    tool_history = []

    for i in range(max_iterations):
        response = call_model(client, messages, model, **kwargs)

        if verbose:
            print(f"  [Response {i+1}] {response[:300]}...")

        if logger:
            logger.log_model_response(response, i + 1)

        tool_calls = extract_tool_calls(response)
        if not tool_calls:
            return response, tool_history

        messages.append({"role": "assistant", "content": response})
        if logger:
            logger.log_message("assistant", response)

        for tc in tool_calls:
            name = tc["name"]
            args = tc.get("arguments", {})

            if verbose:
                print(f"  [Tool] {name}({json.dumps(args, ensure_ascii=False)})")

            result = execute_tool(name, args, temp_dir, logger=None)  # Don't log here, log below with full context
            tool_history.append({"tool": name, "args": args, "result": result})

            result_str = json.dumps(result, ensure_ascii=False)
            result_str_for_model = result_str
            if len(result_str_for_model) > 4000:
                result_str_for_model = result_str_for_model[:4000] + "...[truncated]"

            # Log with both full result and what model sees
            if logger:
                logger.log_tool_call(name, args, result, result_sent_to_model=result_str_for_model)

            messages.append({
                "role": "tool",
                "content": result_str_for_model,
                "name": name
            })
            if logger:
                logger.log_message("tool", result_str_for_model)

    # Force final answer - add instruction to stop using tools
    messages.append({
        "role": "user",
        "content": "You have used the maximum number of tool calls. Based on the information gathered, provide your final answer now. Do NOT use any more tools. Just give the answer directly."
    })
    final = call_model(client, messages, model, **kwargs)

    # Strip any remaining tool calls from response, keep only the text answer
    if "<tool_call>" in final:
        # Extract text before/after tool calls
        import re
        final = re.sub(r'<tool_call>.*?</tool_call>', '', final, flags=re.DOTALL).strip()
        if not final:
            final = "[No answer provided after tool calls]"

    if logger:
        logger.log_model_response(final, max_iterations + 1)
    return final, tool_history


def compute_f1(pred: str, gt: str) -> float:
    pred_tokens = set(pred.lower().split())
    gt_tokens = set(gt.lower().split())
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = pred_tokens & gt_tokens
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def exact_match(pred: str, gt: str) -> bool:
    return gt.lower().strip() in pred.lower()


def load_dataset(path: str, num_samples: int = 5):
    table = pq.read_table(path)
    samples = []
    for i in range(min(num_samples, table.num_rows)):
        img_data = table.column("image")[i].as_py()
        image_bytes = img_data.get("bytes", b"") if isinstance(img_data, dict) else img_data
        samples.append({
            "id": i,
            "image_bytes": image_bytes,
            "question": table.column("question")[i].as_py(),
            "answer": table.column("answer")[i].as_py(),
            "source": table.column("source")[i].as_py(),
        })
    return samples


def main():
    global detailed_logger, image_server

    parser = argparse.ArgumentParser(description="Test tool-calling on SimpleVQA")
    parser.add_argument("--base-url", default="http://localhost:8000", help="API base URL")
    parser.add_argument("--api-key", default="qwen", help="API key")
    parser.add_argument("--model", default="qwen", help="Model name")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of samples to test")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temp-dir", default="/tmp/simplevqa_test", help="Temp dir for screenshots")
    parser.add_argument("--output", default="simplevqa_results.json", help="Output file")
    parser.add_argument("--log-file", default=None, help="Detailed JSONL log file for debugging (e.g., detailed_log.jsonl)")
    parser.add_argument("--dataset", default="/scratch/gpfs/ZHUANGL/lc1556/hf_cache/hub/datasets--gsarch--SimpleVQA-EN/snapshots/9f8bc1d1792667a8b29fda77726061acce3a3401/data/test-00000-of-00001.parquet")
    parser.add_argument("--verbose", action="store_true", default=True)
    parser.add_argument("--imgbb-api-key", type=str,
                        default=os.environ.get("IMGBB_API_KEY", "785fb8b3d1eb0f8b2de15a7031b61933"),
                        help="imgbb API key for public image hosting")
    args = parser.parse_args()

    # Initialize detailed logger
    detailed_logger = DetailedLogger(args.log_file)
    if args.log_file:
        print(f"Detailed logging enabled: {args.log_file}")

    # Start image server for hosting images (needed for image_search)
    image_server = ImageServer(imgbb_api_key=args.imgbb_api_key, image_dir=os.path.join(args.temp_dir, "hosted_images"))
    image_server.start()

    os.makedirs(args.temp_dir, exist_ok=True)

    base_url = args.base_url.rstrip("/")
    if not base_url.endswith("/v1"):
        base_url += "/v1"

    client = OpenAI(api_key=args.api_key, base_url=base_url, timeout=120)

    print(f"Loading {args.num_samples} samples from dataset...")
    samples = load_dataset(args.dataset, args.num_samples)

    print(f"Connected to {base_url}")
    print(f"Model: {args.model}")
    print(f"Testing {len(samples)} samples")
    print("=" * 60)

    results = []
    correct = 0

    for sample in samples:
        print(f"\n{'#'*60}")
        print(f"[Sample {sample['id']}]")
        print(f"  Question: {sample['question']}")
        print(f"  GT Answer: {sample['answer']}")
        print(f"  Source: {sample.get('source', 'N/A')}")

        # Build message with image
        image_b64 = image_to_base64(sample["image_bytes"])

        # Host image for image_search tool
        hosted_image_url = image_server.host_image(sample["image_bytes"], f"sample_{sample['id']}.png")
        print(f"  Hosted image URL: {hosted_image_url}")

        # Start logging for this sample (include image)
        detailed_logger.start_sample(
            sample["id"],
            sample["question"],
            sample["answer"],
            sample.get("source", ""),
            image_base64=image_b64
        )

        # Include the hosted URL in the text so model can use it for image_search
        question_with_url = f"{sample['question']}\n\n[Image URL for image_search: {hosted_image_url}]"

        user_content = [
            {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
            {"type": "text", "text": question_with_url}
        ]

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ]

        # Log initial messages
        detailed_logger.log_message("system", SYSTEM_PROMPT)
        detailed_logger.log_message("user", user_content)

        try:
            prediction, tool_history = run_tool_loop(
                client, messages, args.model, args.temp_dir,
                verbose=args.verbose,
                logger=detailed_logger,
                temperature=args.temperature,
                max_tokens=args.max_tokens
            )

            f1 = compute_f1(prediction, sample["answer"])
            em = exact_match(prediction, sample["answer"])

            if em:
                correct += 1

            print(f"\n  FINAL Prediction: {prediction[:300]}")
            print(f"  F1: {f1:.3f}, EM: {em}")
            print(f"  Tools used: {len(tool_history)}")

            # End logging for this sample
            detailed_logger.end_sample(prediction, f1, em)

            results.append({
                "id": sample["id"],
                "question": sample["question"],
                "gt_answer": sample["answer"],
                "prediction": prediction,
                "f1": f1,
                "exact_match": em,
                "num_tools": len(tool_history),
                "tool_history": tool_history,
            })

        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()
            detailed_logger.end_sample("", 0.0, False, error=str(e))
            results.append({
                "id": sample["id"],
                "question": sample["question"],
                "gt_answer": sample["answer"],
                "error": str(e),
            })

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Results: {correct}/{len(samples)} exact match")
    avg_f1 = sum(r.get("f1", 0) for r in results) / len(results) if results else 0
    print(f"Average F1: {avg_f1:.3f}")

    # Print logger summary
    if detailed_logger:
        summary = detailed_logger.get_summary()
        print(f"Total tool calls: {summary['total_tool_calls']}")
        print(f"Avg tools per sample: {summary['avg_tools_per_sample']:.2f}")

    # Save results
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "num_samples": len(samples),
            "exact_match": correct,
            "accuracy": correct / len(samples) if samples else 0,
            "avg_f1": avg_f1,
            "results": results,
        }, f, indent=2, ensure_ascii=False)

    print(f"\nResults saved to {args.output}")
    if args.log_file:
        print(f"Detailed log saved to {args.log_file}")
        print(f"  View with: cat {args.log_file} | python -m json.tool")

    # Cleanup
    if image_server:
        image_server.stop()


if __name__ == "__main__":
    main()
