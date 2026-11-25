#!/usr/bin/env python3
"""Verify forced-action tag processing through vLLM's OpenAI API."""

from __future__ import annotations

import argparse

from openai import OpenAI
from transformers import AutoTokenizer

from verl.utils.forced_action_tags import (
    FORCE_ACTION_TAG_IDS_KEY,
    VLLM_OPENAI_EXTRA_ARGS_KEY,
    build_forced_action_openai_extra_body,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--required-tag", choices=("<tool_call>", "<answer>"), default="<answer>")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    forced_extra_body = build_forced_action_openai_extra_body(tokenizer, args.required_tag)
    forced_args = forced_extra_body[VLLM_OPENAI_EXTRA_ARGS_KEY]
    response = OpenAI(api_key=args.api_key, base_url=args.base_url).chat.completions.create(
        model=args.model,
        messages=[
            {"role": "system", "content": "Follow the requested action protocol."},
            {"role": "assistant", "content": "<think>\nForced final-answer API canary.\n</think>"},
        ],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "canary_tool",
                    "description": "Canary tool for the forced-action transition.",
                    "parameters": {"type": "object", "properties": {}},
                },
            }
        ],
        tool_choice="none" if args.required_tag == "<answer>" else "auto",
        max_tokens=len(forced_args[FORCE_ACTION_TAG_IDS_KEY]) + 1,
        extra_body={
            "continue_final_message": True,
            "add_generation_prompt": False,
            **forced_extra_body,
        },
    )
    choice = response.choices[0]
    content = choice.message.content or ""
    parsed_tool_calls = choice.message.tool_calls or []
    tag_visible = content.startswith(f"\n{args.required_tag}")
    tag_parsed = args.required_tag == "<tool_call>" and bool(parsed_tool_calls)
    if args.required_tag == "<answer>":
        canary_passed = tag_visible and not parsed_tool_calls
    else:
        canary_passed = tag_visible or tag_parsed
    if not canary_passed:
        raise SystemExit(
            f"forced-action canary failed for {args.required_tag}: "
            f"finish_reason={choice.finish_reason!r}, content={content!r}, "
            f"tool_calls={parsed_tool_calls!r}"
        )
    print(f"Forced-action OpenAI API canary passed for {args.required_tag}.")


if __name__ == "__main__":
    main()
