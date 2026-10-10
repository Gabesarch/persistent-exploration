#!/usr/bin/env python3
"""Validate observed replay updates, reasoning decisions, and checkpoint resume."""

import argparse
import json
import math
import re
from collections import Counter
from pathlib import Path

METRIC_PATTERN = re.compile(
    r"(?P<name>[A-Za-z0-9_/]+):\s*(?:np\.(?:float|int|uint)\d+\()?"
    r"(?P<value>[-+0-9.eE]+)\)?(?=[,}\s]|$)"
)
REASONING_PATTERN = re.compile(
    r"\s*<think>\s*(?P<reasoning>\S.*?)\s*</think>\s*<answer>\s*(?P<decision>[AR])\s*</answer>\s*", re.S
)


def _jsonl(path):
    assert path.is_file(), f"missing smoke artifact: {path}"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _metric_rows(path):
    assert path.is_file(), f"missing training log: {path}"
    rows = []
    for line in path.read_text(errors="replace").splitlines():
        row = {match["name"]: float(match["value"]) for match in METRIC_PATTERN.finditer(line)}
        if "actor/pg_loss" in row:
            assert all(math.isfinite(value) for value in row.values()), "non-finite update metric"
            rows.append(row)
    assert rows, f"no optimizer update metrics: {path}"
    return rows


def _validate_update(row):
    for key in (
        "replay/prompt_count",
        "replay/accept_prompts",
        "replay/reject_prompts",
        "replay/loss_share",
        "replay/active_policy_tokens",
        "replay/reasoning_tokens",
        "replay/nonzero_advantage_tokens",
        "replay/prefill_active_tokens",
        "replay/supervised_critic_enabled",
    ):
        assert key in row, f"missing optimizer evidence: {key}"
    assert row["replay/prefill_active_tokens"] == 0, "injected thinking prefix receives policy loss"
    assert row["replay/supervised_critic_enabled"] == 0, "supervised critic loss is enabled"
    count = row["replay/prompt_count"]
    assert count == row["replay/accept_prompts"] + row["replay/reject_prompts"]
    assert row["replay/accept_prompts"] == row["replay/reject_prompts"], "replay classes are unbalanced"
    assert math.isclose(row["replay/loss_share"], count / 20, abs_tol=1e-6), (
        "judge objective share differs from realized prompt fraction"
    )
    if count:
        assert row["replay/active_policy_tokens"] > 0, "judge response has no active policy tokens"
        assert row["replay/reasoning_tokens"] > 0, "judge thinking has no active policy tokens"
        assert row["replay/nonzero_advantage_tokens"] >= 0
    assert "actor/retry_critic_weighted_loss" not in row or row["actor/retry_critic_weighted_loss"] == 0


def validate(run_dir):
    counts = Counter()
    initial = _metric_rows(run_dir / "train.log")
    resumed = _metric_rows(run_dir / "resume.log")
    assert initial[0].get("replay/prompt_count") == 0, "fresh smoke did not start with an empty replay cache"
    for row in initial + resumed:
        _validate_update(row)
        counts["optimizer_updates"] += 1
        counts["replay_updates"] += int(row["replay/prompt_count"] > 0)
        counts["replay_updates_with_nonzero_advantages"] += int(row["replay/nonzero_advantage_tokens"] > 0)
    assert counts["optimizer_updates"] == 4, "smoke must run three fresh steps and one resumed step"
    assert counts["replay_updates_with_nonzero_advantages"], "no judge update has advantage-bearing policy tokens"
    assert any(row["replay/prompt_count"] == 4 for row in initial), "no full balanced replay-bearing initial update"
    assert resumed[0].get("replay/cache_restore_step") == 3, "resume did not restore the selected step-3 cache"
    assert resumed[0].get("replay/restored_entries", 0) > 0, "resume restored an empty cache"
    assert resumed[0]["replay/prompt_count"] > 0, "resumed update did not sample restored replay"

    cache_root = run_dir / "checkpoint/trajectory_replay"
    for step in range(1, 5):
        snapshot = run_dir / f"checkpoint/global_step_{step}/trajectory_replay.json"
        assert snapshot.is_file(), f"checkpoint {step} has no matching replay snapshot"
        state = json.loads(snapshot.read_text())
        assert state.get("step") == step, f"checkpoint {step} has a replay snapshot from another step"
        entries = state["entries"]
        assert entries, f"checkpoint {step} cache contains no candidates"
        for entry in entries:
            assert entry["label"] in ("A", "R")
            assert "<think>" not in entry["question"] and "<think>" not in entry["answer"]
            for asset in entry.get("images", []):
                asset_path = Path(asset["path"])
                if not asset_path.is_absolute():
                    asset_path = cache_root / asset_path
                assert asset_path.is_file(), f"checkpoint cache image is unavailable: {asset_path}"
        counts["checkpoint_snapshots"] += 1

    for trace in _jsonl(run_dir / "debug/replay_judge_traces.jsonl"):
        assert trace["thinking_prefill_in_prompt"] is True
        assert trace["prefill_active_tokens"] == 0
        assert 0 <= trace["reasoning_policy_tokens"] <= trace["policy_tokens"]
        assert trace["generated_tokens"] == trace["policy_tokens"] > 0
        assert trace["generated_tokens"] <= 5120
        assert trace["retry_critic_tokens"] == trace["retry_sft_tokens"] == 0
        assert trace["prompt_tokens"] <= 16384
        if trace["source_turn"] > 1:
            assert trace["evidence_mode"] == "tool_outputs", "judge received the initial image for a later answer"
        counts["judge_rollouts"] += 1
        counts["judge_rollouts_with_active_reasoning"] += int(trace["reasoning_policy_tokens"] > 0)
        counts["well_formed_judge_rollouts"] += int(REASONING_PATTERN.fullmatch(trace["response_text"]) is not None)
    assert counts["judge_rollouts"] and counts["well_formed_judge_rollouts"], (
        "no sampled judge reasoning and A/R decision"
    )
    assert counts["judge_rollouts_with_active_reasoning"], "no observed masked policy tokens in judge reasoning"

    group_schedules, group_counts = {}, Counter()
    for record in _jsonl(run_dir / "debug/sample_traces.jsonl"):
        if record.get("event") != "sample_final":
            continue
        if not (record.get("trajectory") or {}).get("validate"):
            schedule = record.get("prompt_group_schedule")
            assert schedule is not None, "fresh training rollout is missing prompt-group scheduling metadata"
            turns = tuple(schedule["answer_turns"])
            assert len(turns) <= 2 and all(2 <= turn <= 9 for turn in turns)
            assert len(turns) < 2 or turns[1] - turns[0] >= 2
            assert schedule["force_first_tool_call"] == bool(turns)
            group_id = schedule["group_id"]
            immutable = (schedule["force_first_tool_call"], turns)
            assert group_schedules.setdefault(group_id, immutable) == immutable, (
                "samples of one prompt have different schedules"
            )
            group_counts[group_id] += 1
            consumed = schedule["consumed_answer_turns"]
            assert len(consumed) == len(set(consumed)) and set(consumed) <= set(turns)
            history_turns = [event["answer_turn"] for event in schedule["history"]]
            assert sorted(history_turns) == consumed, "a scheduled deadline was consumed incorrectly or fired twice"
            counts[f"schedule_{len(turns)}_deadline_rollouts"] += 1
            continue
        counts["validation_rollouts"] += 1
        assert record.get("prompt_group_schedule") is None, "training schedule leaked into validation"
        for decision in record.get("retry_critic_validation_decision_log", []):
            assert decision.get("critic_mode") == "reasoning", "validation did not use the reasoning judge"
            counts["validation_judge_decisions"] += 1
            if decision.get("decision") in ("A", "R"):
                assert REASONING_PATTERN.fullmatch(decision["critic_response"]), (
                    "validation accepted a malformed judge decision"
                )
                assert 0 < decision["critic_generated_tokens"] <= 5120
                if decision["candidate_turn"] > 1:
                    assert decision["critic_evidence_mode"] == "tool_outputs"
                counts["well_formed_validation_judge_decisions"] += 1
            else:
                assert decision["controller_action"] == "accept_fallback", "judge failure discarded the candidate"
    assert counts["validation_rollouts"] and counts["well_formed_validation_judge_decisions"], (
        "no observed validation reasoning decision"
    )
    assert group_counts and all(count == 8 for count in group_counts.values()), (
        "a fresh prompt lacks its eight scheduled samples"
    )
    assert all(counts[f"schedule_{deadlines}_deadline_rollouts"] for deadlines in (0, 1, 2)), (
        "one schedule category was not observed"
    )
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    print("REPLAY CRITIC GPU SMOKE PASSED " + json.dumps(dict(validate(args.run_dir)), sort_keys=True))


if __name__ == "__main__":
    main()
