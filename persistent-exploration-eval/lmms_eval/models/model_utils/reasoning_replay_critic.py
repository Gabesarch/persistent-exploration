"""OpenAI transport for the training replay critic's shared evidence contract."""

import importlib.util
import math
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def replay_contract():
    # Evaluation runs with persistent-exploration-eval on PYTHONPATH, without importing verl's
    # training dependencies. Load the single shared, CPU-only implementation.
    path = Path(__file__).resolve().parents[4] / "verl/utils/trajectory_replay.py"
    spec = importlib.util.spec_from_file_location("persistent_exploration_eval_replay_contract", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def reasoning_critic_request(
    payload, candidate_content, system_prompt, *, temperature=0.0, top_k=-1, top_p=1.0, max_tokens=5120
):
    if not math.isfinite(temperature) or temperature < 0:
        raise ValueError("reasoning critic temperature must be finite and nonnegative")
    if top_k != -1 and top_k < 1:
        raise ValueError("reasoning critic top_k must be -1 or positive")
    if not math.isfinite(top_p) or not 0 < top_p <= 1:
        raise ValueError("reasoning critic top_p must be in (0, 1]")
    if max_tokens < 1:
        raise ValueError("reasoning critic max_tokens must be positive")
    contract = replay_contract()
    history = payload.get("messages") or []
    question = contract.extract_question_text(history)
    answer = contract.extract_candidate_answer(candidate_content)
    initial_images, tool_images = [], []
    seen_user = False
    for message in history:
        content = message.get("content")
        images = (
            [p for p in content if isinstance(p, dict) and p.get("type") == "image_url"]
            if isinstance(content, list)
            else []
        )
        if message.get("role") == "user":
            (tool_images if seen_user else initial_images).extend(images)
            seen_user = True
        elif message.get("role") == "tool":
            tool_images.extend(images)
    turns = 1 + sum(m.get("role") == "assistant" for m in history)
    calls = sum(len(m.get("tool_calls") or []) for m in history if m.get("role") == "assistant")
    evidence_mode, images = contract.select_candidate_evidence(initial_images, tool_images, turns, calls)
    if not question or not answer or not images:
        raise ValueError("reasoning critic requires a question, candidate answer, and admissible visual evidence")
    messages = contract.build_judge_messages(question, answer, [], system_prompt)
    messages[1]["content"][1:1] = images
    messages.append({"role": "assistant", "content": "<think>\n"})
    request = dict(payload)
    for key in (
        "tools",
        "tool_choice",
        "parallel_tool_calls",
        "logprobs",
        "top_logprobs",
        "stop",
        "response_format",
        "_training_initial_tokens",
        "max_completion_tokens",
    ):
        request.pop(key, None)
    request.update(
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        n=1,
        extra_body={"top_k": top_k, "continue_final_message": True, "add_generation_prompt": False},
    )
    return request, {
        "critic_evidence_mode": evidence_mode,
        "critic_evidence_images": len(images),
        "critic_sampling_params": {
            "temperature": temperature, "top_k": top_k, "top_p": top_p, "n": 1, "max_tokens": max_tokens,
        },
    }


def reasoning_critic_response(response, evidence):
    choice = response.choices[0]
    text = "<think>\n" + (choice.message.content or "")
    decision = replay_contract().parse_judge_response(text)
    if decision is None or getattr(choice.message, "tool_calls", None):
        raise ValueError("reasoning critic must generate a nonempty think block followed by one A/R answer")
    generated = getattr(getattr(response, "usage", None), "completion_tokens", None)
    if generated is not None and generated > evidence.get("critic_sampling_params", {}).get("max_tokens", 5120):
        raise ValueError("reasoning critic exceeded its hard response budget")
    return {
        **evidence,
        "mode": "reasoning",
        "model_token": decision,
        "decision": decision,
        "selected_logprob": None,
        "p_accept": None,
        "p_retry": None,
        "threshold": None,
        "critic_response": text,
        "critic_generated_tokens": generated,
        "critic_reasoning_format_valid": True,
    }
