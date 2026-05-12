#!/usr/bin/env python3
"""Probe chart math reward behavior on saved validation generations.

This script isolates the verifier from rollout/training. It compares:

* current chart reward on the main thread
* current chart reward in a ThreadPoolExecutor, matching async reward manager use
* a small mathruler-based verifier
* the direct math_verify threaded timeout failure
"""

from __future__ import annotations

import argparse
import asyncio
import concurrent.futures
import json
from pathlib import Path
from typing import Any

from math_verify import parse

from verl.utils.reward_score.math_verify_reward_type_fixations_think_nonpoint import (
    _extract_reward_answer,
    _parse_and_validate_fixation_output,
    async_compute_score_from_data_source,
    compute_score_from_data_source,
)


DEFAULT_JSONL = (
    "/scratch/gpfs/ZHUANGL/gs1693/verl_gaze_rl/checkpoints/verl/"
    "GSPO_fp16_math_verify_chart_qwen3vl_onpolicy_20260420_155518/"
    "validation_data/160.jsonl"
)
DEFAULT_SOURCE_JSONL = (
    "/scratch/gpfs/ZHUANGL/gs1693/reason_vlm_datasets/chart_search_counting_rl/"
    "preprocessed_max_pixels_10240000/chart_search_counting_val.verl.preprocessed.jsonl"
)


def _thread_result(fn):
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        return executor.submit(fn).result(timeout=30)


def probe_math_verify_threading() -> None:
    def call(timeout):
        try:
            parsed = parse("1/2", parsing_timeout=timeout)
            return {"ok": True, "parsed": [str(x) for x in parsed]}
        except Exception as exc:  # noqa: BLE001 - this is a diagnostic script
            return {"ok": False, "error": type(exc).__name__, "message": str(exc)}

    print("== math_verify parse threading probe ==")
    print("main parsing_timeout=3 :", call(3))
    print("thread parsing_timeout=3:", _thread_result(lambda: call(3)))
    print("thread parsing_timeout=None:", _thread_result(lambda: call(None)))
    print()


def current_reward(output: str, gt: str, extra_info: dict[str, Any], format_score: float):
    return compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=output,
        ground_truth=gt,
        extra_info=extra_info,
        format_score=format_score,
    )


async def async_reward(output: str, gt: str, extra_info: dict[str, Any], format_score: float):
    return await async_compute_score_from_data_source(
        data_source="qwen3vl_filtering",
        solution_str=output,
        ground_truth=gt,
        extra_info=extra_info,
        format_score=format_score,
    )


def mathruler_reward(output: str, gt: str, reward_type: str, format_score: float) -> dict[str, float | str | None]:
    from mathruler.grader import extract_boxed_content, grade_answer

    formatting = 1.0 if _parse_and_validate_fixation_output(output)[0] else 0.0
    accuracy = 0.0
    answer = None
    if formatting:
        _, answer_text = _parse_and_validate_fixation_output(output)
        if answer_text:
            answer = _extract_reward_answer(answer_text, reward_type)
            if answer is not None:
                accuracy = 1.0 if grade_answer(answer, gt) else 0.0
    if answer is None:
        boxed = extract_boxed_content(output)
        answer = boxed if boxed != "" else None
    score = (1.0 - format_score) * accuracy + format_score * formatting
    return {"score": score, "accuracy": accuracy, "format": formatting, "answer": answer}


def synthetic_probe(format_score: float) -> None:
    cases = [
        ("exact int", "6", "6", None),
        ("decimal equivalent", "6.0", "6", None),
        ("relative tolerance", "6.66", "6.67", 0.05),
        ("fraction", "1/2", "0.5", None),
    ]
    print("== synthetic reward probe ==")
    for name, pred, gt, tolerance in cases:
        output = f"<think>Start. [1, 2] Then text. [3, 4]</think>\n\nThe answer is \\boxed{{{pred}}}"
        extra_info = {"reward_type": "numeric"}
        if tolerance is not None:
            extra_info["tolerance"] = tolerance
        main = current_reward(output, gt, extra_info, format_score)
        threaded = _thread_result(lambda: current_reward(output, gt, extra_info, format_score))
        awaited = asyncio.run(async_reward(output, gt, extra_info, format_score))
        ruler = mathruler_reward(output, gt, "numeric", format_score)
        print(f"{name:18s} pred={pred!r} gt={gt!r} tol={tolerance!r}")
        print(f"  current main   : {main}")
        print(f"  current thread : {threaded}")
        print(f"  async awaited  : {awaited}")
        print(f"  mathruler      : {ruler}")
    print()


def iter_rows(path: Path, limit: int | None):
    with path.open() as f:
        for idx, line in enumerate(f):
            if limit is not None and idx >= limit:
                break
            if line.strip():
                yield idx, json.loads(line)


def load_source_extra_info(path: Path | None, limit: int | None) -> dict[int, dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    extra_infos: dict[int, dict[str, Any]] = {}
    for idx, row in iter_rows(path, limit):
        extra_info = row.get("extra_info")
        if isinstance(extra_info, dict):
            extra_infos[idx] = dict(extra_info)
    return extra_infos


def audit_jsonl(
    path: Path,
    source_path: Path | None,
    limit: int | None,
    fallback_reward_type: str,
    format_score: float,
    fallback_tolerance: float | None,
) -> None:
    print(f"== JSONL audit: {path} ==")
    source_extra_infos = load_source_extra_info(source_path, limit)
    if source_extra_infos:
        print(f"using source extra_info from: {source_path}")
    else:
        print(f"using fallback extra_info: reward_type={fallback_reward_type!r}, tolerance={fallback_tolerance!r}")
    totals = {
        "rows": 0,
        "saved_current_mismatch": 0,
        "thread_main_mismatch": 0,
        "mathruler_differs": 0,
        "saved_zero_current_correct": 0,
    }
    examples = []

    for idx, row in iter_rows(path, limit):
        output = str(row.get("output", ""))
        gt = str(row.get("gts", ""))
        extra_info = dict(source_extra_infos.get(idx, {}))
        if "reward_type" not in extra_info:
            extra_info["reward_type"] = fallback_reward_type
        if fallback_tolerance is not None and extra_info.get("tolerance") is None:
            extra_info["tolerance"] = fallback_tolerance
        reward_type = str(extra_info.get("reward_type", fallback_reward_type))
        saved = {
            "score": float(row.get("score", 0.0)),
            "accuracy": float(row.get("accuracy", 0.0)),
            "format": float(row.get("format", 0.0)),
        }
        main = current_reward(output, gt, extra_info, format_score)
        threaded = _thread_result(lambda: current_reward(output, gt, extra_info, format_score))
        ruler = mathruler_reward(output, gt, reward_type, format_score)

        totals["rows"] += 1
        if abs(saved["score"] - float(main["score"])) > 1e-6 or abs(saved["accuracy"] - float(main["accuracy"])) > 1e-6:
            totals["saved_current_mismatch"] += 1
        if abs(float(threaded["score"]) - float(main["score"])) > 1e-6:
            totals["thread_main_mismatch"] += 1
        if abs(float(ruler["score"]) - float(main["score"])) > 1e-6:
            totals["mathruler_differs"] += 1
        if saved["accuracy"] == 0.0 and float(main["accuracy"]) == 1.0:
            totals["saved_zero_current_correct"] += 1

        if len(examples) < 12 and (
            saved["accuracy"] != float(main["accuracy"])
            or float(threaded["score"]) != float(main["score"])
            or float(ruler["score"]) != float(main["score"])
        ):
            examples.append(
                (idx, gt, reward_type, extra_info.get("tolerance"), saved, main, threaded, ruler, output[-180:].replace("\n", "\\n"))
            )

    print("summary:", totals)
    print("examples:")
    for idx, gt, reward_type, tolerance, saved, main, threaded, ruler, tail in examples:
        print(f"- row={idx} gt={gt!r} reward_type={reward_type!r} tolerance={tolerance!r}")
        print(f"  saved   : {saved}")
        print(f"  main    : {main}")
        print(f"  threaded: {threaded}")
        print(f"  mathruler: {ruler}")
        print(f"  tail    : {tail!r}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--jsonl", default=DEFAULT_JSONL)
    parser.add_argument("--source-jsonl", default=DEFAULT_SOURCE_JSONL)
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--reward-type", default="numeric")
    parser.add_argument("--format-score", type=float, default=0.2)
    parser.add_argument("--tolerance", type=float, default=None)
    args = parser.parse_args()

    probe_math_verify_threading()
    synthetic_probe(args.format_score)
    source_path = Path(args.source_jsonl) if args.source_jsonl else None
    audit_jsonl(Path(args.jsonl), source_path, args.limit, args.reward_type, args.format_score, args.tolerance)


if __name__ == "__main__":
    main()
