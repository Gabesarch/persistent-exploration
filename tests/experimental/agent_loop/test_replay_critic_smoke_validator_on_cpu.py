"""GPU smoke acceptance requires measured replay updates and snapshot restore."""

import json

import pytest

from vero2_scripts.validate_replay_critic_smoke import validate


def _fixture(run_dir):
    debug = run_dir / "debug"
    debug.mkdir()
    cache = run_dir / "checkpoint/trajectory_replay"
    (cache / "images").mkdir(parents=True)
    (cache / "images/asset.png").write_bytes(b"persistent fixture")
    for step in range(1, 5):
        checkpoint = run_dir / f"checkpoint/global_step_{step}"
        checkpoint.mkdir()
        (checkpoint / "trajectory_replay.json").write_text(
            json.dumps(
                {
                    "step": step,
                    "entries": [
                        {
                            "label": "A",
                            "question": "How many?",
                            "answer": "\\boxed{3}",
                            "images": [{"path": "images/asset.png"}],
                        },
                        {
                            "label": "R",
                            "question": "How many?",
                            "answer": "2",
                            "images": [{"path": "images/asset.png"}],
                        },
                    ],
                }
            )
        )
    rows = []
    for step, prompts in enumerate((0, 4, 4, 4), 1):
        row = {
            "actor/pg_loss": 0.01,
            "replay/prompt_count": prompts,
            "replay/accept_prompts": prompts / 2,
            "replay/reject_prompts": prompts / 2,
            "replay/loss_share": prompts / 20,
            "replay/active_policy_tokens": prompts * 50,
            "replay/reasoning_tokens": prompts * 40,
            "replay/nonzero_advantage_tokens": prompts * 20,
            "replay/prefill_active_tokens": 0,
            "replay/supervised_critic_enabled": 0,
        }
        if step == 4:
            row.update({"replay/cache_restore_step": 3, "replay/restored_entries": 2})
        rows.append(row)
    (run_dir / "train.log").write_text(
        "\n".join(", ".join(f"{key}:{value}" for key, value in row.items()) for row in rows[:3])
    )
    (run_dir / "resume.log").write_text(", ".join(f"{key}:{value}" for key, value in rows[3].items()))
    judge = {
        "thinking_prefill_in_prompt": True,
        "prefill_active_tokens": 0,
        "reasoning_policy_tokens": 40,
        "generated_tokens": 50,
        "policy_tokens": 50,
        "retry_critic_tokens": 0,
        "retry_sft_tokens": 0,
        "prompt_tokens": 100,
        "source_turn": 3,
        "evidence_mode": "tool_outputs",
        "response_text": "<think>There are three visible objects.</think><answer>A</answer>",
    }
    (debug / "replay_judge_traces.jsonl").write_text(json.dumps(judge) + "\n")
    validation = {
        "event": "sample_final",
        "trajectory": {"validate": True},
        "retry_critic_validation_decision_log": [
            {
                "critic_mode": "reasoning",
                "decision": "A",
                "critic_response": judge["response_text"],
                "critic_generated_tokens": 50,
                "critic_evidence_mode": "tool_outputs",
                "candidate_turn": 3,
                "controller_action": "accept",
            }
        ],
    }
    training = []
    for index, turns in enumerate(([], [3], [3, 7])):
        for _ in range(8):
            training.append(
                {
                    "event": "sample_final",
                    "trajectory": {"validate": False},
                    "prompt_group_schedule": {
                        "group_id": f"step1:task{index}",
                        "force_first_tool_call": bool(turns),
                        "answer_turns": turns,
                        "consumed_answer_turns": [],
                        "history": [],
                    },
                }
            )
    (debug / "sample_traces.jsonl").write_text(
        "\n".join(json.dumps(record) for record in [*training, validation]) + "\n"
    )
    return judge, validation


def test_observed_gpu_smoke_fixture_passes(tmp_path):
    _fixture(tmp_path)
    counts = validate(tmp_path)
    assert counts["optimizer_updates"] == counts["checkpoint_snapshots"] == 4
    assert counts["replay_updates"] == 3
    assert counts["well_formed_judge_rollouts"] == counts["well_formed_validation_judge_decisions"] == 1


@pytest.mark.parametrize(
    "failure",
    [
        "no_replay",
        "no_reasoning_pg",
        "nll_loss",
        "prefill_loss",
        "unbalanced",
        "wrong_share",
        "wrong_restore",
        "wrong_snapshot",
        "missing_image",
        "wrong_evidence",
        "empty_reasoning",
        "validation_schedule",
        "old_critic",
    ],
)
def test_smoke_rejects_missing_runtime_evidence(tmp_path, failure):
    judge, validation = _fixture(tmp_path)
    if failure in {"no_replay", "no_reasoning_pg", "nll_loss", "prefill_loss", "unbalanced", "wrong_share"}:
        replacements = {
            "no_replay": ("replay/prompt_count:4", "replay/prompt_count:0"),
            "no_reasoning_pg": ("replay/reasoning_tokens:160", "replay/reasoning_tokens:0"),
            "nll_loss": ("replay/supervised_critic_enabled:0", "replay/supervised_critic_enabled:1"),
            "prefill_loss": ("replay/prefill_active_tokens:0", "replay/prefill_active_tokens:1"),
            "unbalanced": ("replay/accept_prompts:2.0", "replay/accept_prompts:1.0"),
            "wrong_share": ("replay/loss_share:0.2", "replay/loss_share:0.19"),
        }
        old, new = replacements[failure]
        path = tmp_path / "train.log"
        path.write_text(path.read_text().replace(old, new))
    elif failure == "wrong_restore":
        path = tmp_path / "resume.log"
        path.write_text(path.read_text().replace("replay/cache_restore_step:3", "replay/cache_restore_step:4"))
    elif failure == "wrong_snapshot":
        path = tmp_path / "checkpoint/global_step_3/trajectory_replay.json"
        path.write_text(path.read_text().replace('"step": 3', '"step": 4'))
    elif failure == "missing_image":
        (tmp_path / "checkpoint/trajectory_replay/images/asset.png").unlink()
    elif failure in {"wrong_evidence", "empty_reasoning"}:
        if failure == "wrong_evidence":
            judge["evidence_mode"] = "initial"
        else:
            judge["response_text"] = "<think></think><answer>A</answer>"
        (tmp_path / "debug/replay_judge_traces.jsonl").write_text(json.dumps(judge))
    else:
        if failure == "validation_schedule":
            validation["prompt_group_schedule"] = {"answer_turns": [3]}
        else:
            validation["retry_critic_validation_decision_log"][0]["critic_mode"] = "token"
        path = tmp_path / "debug/sample_traces.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        records[-1] = validation
        path.write_text("\n".join(json.dumps(record) for record in records))
    with pytest.raises(AssertionError):
        validate(tmp_path)


@pytest.mark.parametrize(
    "failure", ["different_group_schedule", "missing_sample", "deadline_refired", "active_prefix_trace"]
)
def test_smoke_checks_runtime_schedule_and_prefix_masks(tmp_path, failure):
    _fixture(tmp_path)
    if failure == "active_prefix_trace":
        path = tmp_path / "debug/replay_judge_traces.jsonl"
        trace = json.loads(path.read_text())
        trace["prefill_active_tokens"] = 1
        path.write_text(json.dumps(trace))
    else:
        path = tmp_path / "debug/sample_traces.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        if failure == "different_group_schedule":
            records[8]["prompt_group_schedule"]["answer_turns"] = [4]
        elif failure == "missing_sample":
            records.pop(0)
        else:
            records[8]["prompt_group_schedule"].update(
                consumed_answer_turns=[3], history=[{"answer_turn": 3}, {"answer_turn": 3}]
            )
        path.write_text("\n".join(json.dumps(record) for record in records))
    with pytest.raises(AssertionError):
        validate(tmp_path)


def test_uniform_reward_replay_group_may_have_zero_advantages(tmp_path):
    _fixture(tmp_path)
    path = tmp_path / "train.log"
    path.write_text(
        path.read_text().replace("replay/nonzero_advantage_tokens:80", "replay/nonzero_advantage_tokens:0", 1)
    )
    assert validate(tmp_path)["replay_updates_with_nonzero_advantages"] == 2


def test_smoke_requires_at_least_one_judge_update_with_nonzero_advantages(tmp_path):
    _fixture(tmp_path)
    for name in ("train.log", "resume.log"):
        path = tmp_path / name
        path.write_text(
            path.read_text().replace("replay/nonzero_advantage_tokens:80", "replay/nonzero_advantage_tokens:0")
        )
    with pytest.raises(AssertionError, match="advantage-bearing"):
        validate(tmp_path)


def test_console_metrics_with_numpy_2_scalar_representations(tmp_path):
    _fixture(tmp_path)
    for name in ("train.log", "resume.log"):
        path = tmp_path / name
        path.write_text(
            path.read_text()
            .replace("actor/pg_loss:0.01", "actor/pg_loss:np.float64(0.01)")
            .replace("replay/loss_share:0.2", "replay/loss_share:np.float32(0.20000000298)")
        )
    assert validate(tmp_path)["optimizer_updates"] == 4
