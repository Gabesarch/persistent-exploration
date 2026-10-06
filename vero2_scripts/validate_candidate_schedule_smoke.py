#!/usr/bin/env python3
"""Check candidate-schedule runtime invariants and report sampled-path coverage."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path


def validate(traces: Path, train_log: Path) -> Counter:
    counts = Counter()
    for line in traces.read_text().splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("event") != "sample_final":
            continue
        schedule = record.get("candidate_answer_schedule")
        if (record.get("trajectory") or {}).get("validate"):
            counts["validation_rollouts"] += 1
            assert schedule is None, "training-only candidate scheduling leaked into validation"
            assert record.get("canonical_prefix_mask") is None, "training-only prefix mask leaked into validation"
            counts["validation_critic_decisions"] += len(record.get("retry_critic_validation_decision_log") or [])
            continue

        counts["training_rollouts"] += 1
        assert schedule is not None, "training rollout is missing candidate schedule metadata"
        assert schedule["first_deadline"] in (4, 5, 6, 7)
        history = schedule["history"]
        assert schedule["candidate_count"] == len(history)
        assert schedule["generated_policy_turns"] >= len(history)
        assert schedule["generated_policy_tokens"] > 0
        assert not any(event.get("event") == "forced_structure_error" for event in record.get("debug_events", [])), (
            "rollout contains a forced-action structure error"
        )
        followup = schedule["followup_deadline"]
        if history:
            assert history[0]["retained_turn"] <= schedule["first_deadline"], "first candidate missed its deadline"
            if history[0]["action"] == "retry":
                assert followup is not None, "feasible early retry was not assigned a follow-up deadline"
                assert history[0]["retained_turn"] + 2 <= followup < 11
            else:
                assert followup is None
        else:
            counts["rollouts_without_candidate"] += 1
        for index, candidate in enumerate(history):
            assert candidate["candidate_ordinal"] == index + 1
            assert 1 <= candidate["retained_turn"] <= 11
            assert math.isfinite(candidate["accuracy"])
            expected = schedule["first_deadline"] if index == 0 else followup if index == 1 else None
            assert candidate["sampled_deadline"] == expected
            if expected is not None:
                assert candidate["retained_turn"] <= expected, "candidate missed its deadline"
            if candidate["forcing_reason"] == "candidate_deadline":
                assert index < 2 and candidate["forced"]
                assert candidate["retained_turn"] == expected
                counts["forced_first" if index == 0 else "forced_followup"] += 1
            if index:
                assert history[index - 1]["action"] == "retry", "continued after a terminal candidate"
                assert candidate["generated_policy_turn"] > history[index - 1]["generated_policy_turn"]
            counts["candidates"] += 1
            counts["retried_candidates"] += int(candidate["action"] == "retry")
            if candidate["action"] == "accept":
                assert candidate["accuracy"] > 0
                counts["recovered_rollouts"] += int(index > 0)
            else:
                assert candidate["accuracy"] <= 0
        if len(history) >= 2:
            assert schedule["next_deadline"] is None, "schedule was rearmed after the one follow-up"
        prefix_mask = record.get("canonical_prefix_mask")
        if any(candidate["action"] == "retry" for candidate in history):
            assert prefix_mask is not None, "retried canonical rollout is missing prefix mask metadata"
            assert prefix_mask["active_prefix_policy_tokens"] == 0, "shared prefix has active policy loss"
            assert prefix_mask["prefix_tokens"] >= prefix_mask["masked_policy_tokens"] >= 0
            counts["prefix_masked_rollouts"] += 1
            counts["prefix_masked_policy_tokens"] += prefix_mask["masked_policy_tokens"]
        else:
            assert prefix_mask is None, "canonical prefix masked without an actual retry"
        counts["generated_policy_turns"] += schedule["generated_policy_turns"]
        counts["generated_policy_tokens"] += schedule["generated_policy_tokens"]

    assert counts["training_rollouts"], "no completed training rollouts"
    assert counts["validation_rollouts"], "no completed validation rollouts"
    assert counts["validation_critic_decisions"], "no validation critic decisions"
    assert counts["candidates"], "no graded training candidates"
    log = train_log.read_text(errors="replace")
    for metric in (
        "actor/pg_loss", "actor/retry_critic_loss", "actor/retry_critic_weighted_loss",
        *(f"actor/advantage_mass/{name}" for name in (
            "positive", "negative", "canonical_positive", "canonical_negative", "branch_positive", "branch_negative",
            "positive_fraction",
        )),
    ):
        values = re.findall(re.escape(metric) + r":([^,}\s]+)", log)
        assert values and math.isfinite(float(values[-1])), f"missing or non-finite optimizer metric: {metric}"
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", type=Path, required=True)
    parser.add_argument("--train-log", type=Path, required=True)
    args = parser.parse_args()
    counts = validate(args.traces, args.train_log)
    print("CANDIDATE SCHEDULE INVARIANTS PASSED " + json.dumps(dict(counts), sort_keys=True))
    missing = [key for key in ("forced_first", "forced_followup") if not counts[key]]
    if missing:
        print("COVERAGE INCOMPLETE: no observed " + ", ".join(missing) + "; rerun before claiming those paths tested.")
        raise SystemExit(2)


if __name__ == "__main__":
    main()
