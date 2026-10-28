#!/usr/bin/env python3
"""Exercise judge prefill, JSON grading, and sleep/wake before starting eval."""
import argparse
import json
from pathlib import Path
import urllib.request


def probe(base_url, api_key, model, timeout=180):
    base_url = base_url.rstrip("/")
    server_root = base_url.removesuffix("/v1")
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    template = (Path(__file__).resolve().parents[1]
                / "lmms_eval/tasks/visual_probe/prompts/judge_prompt.txt").read_text()

    def request(url, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        )
        with opener.open(req, timeout=timeout) as response:
            raw = response.read()
        return json.loads(raw) if raw else None

    for prediction, expected in (("white", True), ("red", False)):
        request(f"{server_root}/wake_up")
        prompt = (template.replace("[QUESTION]", "What color is the hat?")
                  .replace("[REFERENCE]", "white").replace("[PREDICTION]", prediction))
        response = request(f"{base_url}/chat/completions", {
            "model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": 512,
        })
        content = response["choices"][0]["message"]["content"]
        # Accept the fenced JSON form accepted by the evaluation parser too.
        start, end = content.find("{"), content.rfind("}")
        result = json.loads(content[start:end + 1])
        if result.get("is_equivalent") is not expected:
            raise RuntimeError(f"Judge canary expected {expected}, received {content!r}")
        request(f"{server_root}/sleep?level=1")
    print("Judge generation canary passed: correct/incorrect JSON grading and sleep/wake.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", default="qwen")
    parser.add_argument("--model", default="qwen_vllm")
    args = parser.parse_args()
    probe(args.base_url, args.api_key, args.model)
