"""The GPU smoke check must detect missing masking and advantage diagnostics."""

import json

import pytest

from vero2_scripts.validate_candidate_schedule_smoke import validate


def fixture_records():
    return [
        {
            "event": "sample_final",
            "trajectory": {"validate": False},
            "candidate_answer_schedule": {
                "first_deadline": 7, "followup_deadline": 9, "next_deadline": None,
                "candidate_count": 2, "generated_policy_turns": 10, "generated_policy_tokens": 100,
                "history": [
                    {"candidate_ordinal": 1, "retained_turn": 7, "sampled_deadline": 7,
                     "action": "retry", "accuracy": 0., "forcing_reason": "candidate_deadline",
                     "forced": True, "generated_policy_turn": 7},
                    {"candidate_ordinal": 2, "retained_turn": 9, "sampled_deadline": 9,
                     "action": "accept", "accuracy": 0.5, "forcing_reason": "candidate_deadline",
                     "forced": True, "generated_policy_turn": 10},
                ],
            },
            "canonical_prefix_mask": {
                "prefix_tokens": 50, "masked_policy_tokens": 40,
                "active_prefix_policy_tokens": 0, "active_suffix_policy_tokens": 20,
            },
        },
        {"event": "sample_final", "trajectory": {"validate": True},
         "retry_critic_validation_decision_log": [{"decision": "accept"}]},
    ]


@pytest.mark.parametrize("failure", [None, "active_prefix", "missing_mask", "missing_mass", "validation_mask"])
def test_smoke_checker_requires_mask_and_finite_mass(tmp_path, failure):
    records = fixture_records()
    if failure == "active_prefix":
        records[0]["canonical_prefix_mask"]["active_prefix_policy_tokens"] = 1
    elif failure == "missing_mask":
        del records[0]["canonical_prefix_mask"]
    elif failure == "validation_mask":
        records[1]["canonical_prefix_mask"] = records[0]["canonical_prefix_mask"]
    traces = tmp_path / "sample_traces.jsonl"
    traces.write_text("\n".join(json.dumps(record) for record in records))
    metrics = {"actor/pg_loss": 0.1, "actor/retry_critic_loss": 0.2, "actor/retry_critic_weighted_loss": 0.002}
    for name in ("positive", "negative", "canonical_positive", "canonical_negative", "branch_positive",
                 "branch_negative", "positive_fraction"):
        metrics[f"actor/advantage_mass/{name}"] = 0.5
    if failure == "missing_mass":
        del metrics["actor/advantage_mass/positive"]
    log = tmp_path / "train.log"
    log.write_text(", ".join(f"{key}:{value}" for key, value in metrics.items()))
    if failure:
        with pytest.raises(AssertionError):
            validate(traces, log)
    else:
        counts = validate(traces, log)
        assert counts["prefix_masked_rollouts"] == counts["forced_followup"] == 1
        assert counts["prefix_masked_policy_tokens"] == 40


@pytest.mark.parametrize("action,credit,valid", [("terminate", 0.5, True), ("accept", 0.5, False),
                                               ("accept", 1.0, True), ("terminate", 1.0, False)])
def test_smoke_checker_honors_full_credit_requirement(tmp_path, action, credit, valid):
    records = fixture_records()
    schedule = records[0]["candidate_answer_schedule"]
    schedule["accept_partial_credit"] = False
    schedule["correct_accuracy_threshold"] = 0.0
    schedule["history"][0]["accuracy"] = 0.75
    schedule["history"][-1].update(action=action, accuracy=credit)
    traces = tmp_path / "sample_traces.jsonl"
    traces.write_text("\n".join(json.dumps(record) for record in records))
    metrics = {"actor/pg_loss": 0.1, "actor/retry_critic_loss": 0.2, "actor/retry_critic_weighted_loss": 0.002}
    for name in ("positive", "negative", "canonical_positive", "canonical_negative", "branch_positive",
                 "branch_negative", "positive_fraction"):
        metrics[f"actor/advantage_mass/{name}"] = 0.5
    log = tmp_path / "train.log"
    log.write_text(", ".join(f"{key}:{value}" for key, value in metrics.items()))
    if valid:
        assert validate(traces, log)["retried_candidates"] == 1
    else:
        with pytest.raises(AssertionError):
            validate(traces, log)
