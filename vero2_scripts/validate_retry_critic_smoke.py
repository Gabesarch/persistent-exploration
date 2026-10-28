#!/usr/bin/env python3
"""Validate runtime evidence from the shared-weight retry-critic smoke."""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

QUESTION = (
    "Assuming another tool round is available, should the preceding candidate be accepted or retried? "
    'In the <answer> block, output only "A" or "R".'
)


def _jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{line_number}: invalid JSON: {exc}") from exc


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    args = parser.parse_args()

    failures: list[str] = []
    training_supervision = 0
    validation_decisions = 0
    feasible_retries = 0
    for record in _jsonl(args.traces):
        if record.get("event") != "sample_final":
            continue
        is_validation = bool((record.get("trajectory") or {}).get("validate"))
        if not is_validation:
            canonical_tokens = int(record.get("retry_critic_tokens") or 0)
            canonical_label = record.get("retry_critic_label")
            if canonical_tokens:
                training_supervision += canonical_tokens
                if canonical_tokens != 1 or canonical_label not in {"A", "R"}:
                    failures.append("canonical training critic target is not one A/R token")
            for branch in record.get("conditional_answer_retry_branch_outputs") or []:
                critic_tokens = int(branch.get("retry_critic_tokens") or 0)
                label = branch.get("retry_critic_label")
                if critic_tokens:
                    training_supervision += critic_tokens
                    if critic_tokens != 1 or label not in {"A", "R"}:
                        failures.append("training branch critic target is not one A/R token")
            continue

        final_response = str(record.get("final_response") or "")
        if QUESTION in final_response or "should the preceding candidate be accepted or retried" in final_response:
            failures.append("a validation response contains the discarded critic sidecar")
        events = record.get("debug_events") or []
        if any(event.get("event") == "incorrect_answer_retry" for event in events):
            failures.append("validation used a ground-truth retry event")
        for index, event in enumerate(events):
            if event.get("event") != "retry_critic_validation_decision":
                continue
            validation_decisions += 1
            decision = event.get("decision")
            if decision is not None and decision not in {"A", "R"}:
                failures.append(f"invalid critic decision {decision!r}")
            for key in ("p_accept", "p_retry"):
                value = event.get(key)
                if value is not None and (not math.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0):
                    failures.append(f"invalid {key}={value!r}")
            if event.get("controller_action") == "retry":
                feasible_retries += 1
                later_assistant_turns = [
                    later for later in events[index + 1 :] if later.get("event") == "assistant_turn"
                ]
                if not later_assistant_turns or "<tool_call>" not in str(later_assistant_turns[0].get("text") or ""):
                    failures.append("a feasible critic retry was not followed by the forced tool path")

    train_log = args.train_log.read_text(encoding="utf-8", errors="replace")
    required_metrics = (
        "actor/retry_critic_loss",
        "actor/retry_critic_accept_nll",
        "actor/retry_critic_retry_nll",
        "actor/retry_critic_weighted_loss",
        "actor/retry_critic_accept_examples",
        "actor/retry_critic_retry_examples",
        "actor/retry_critic_accept_parents",
        "actor/retry_critic_retry_parents",
    )
    for metric in required_metrics:
        matches = re.findall(re.escape(metric) + r":([^,}\s]+)", train_log)
        if not matches:
            failures.append(f"training log is missing {metric}")
            continue
        try:
            if not math.isfinite(float(matches[-1])):
                failures.append(f"training metric {metric} is non-finite")
        except ValueError:
            failures.append(f"training metric {metric} is not numeric: {matches[-1]!r}")

    if training_supervision == 0:
        failures.append("the smoke did not record a training critic target")
    if validation_decisions == 0:
        failures.append("the smoke did not record a validation critic decision")
    if feasible_retries == 0:
        failures.append("the smoke did not exercise a feasible validation critic retry")
    if failures:
        for failure in failures:
            print(f"FAIL: {failure}")
        raise SystemExit(1)
    print(
        "RETRY CRITIC SMOKE VALIDATION PASSED "
        f"training_targets={training_supervision} validation_decisions={validation_decisions} "
        f"feasible_retries={feasible_retries}"
    )


if __name__ == "__main__":
    main()
