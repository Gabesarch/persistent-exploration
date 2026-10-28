import json
from pathlib import Path

import pytest
from PIL import Image

from verl.utils.trajectory_replay import (
    ReplayCache,
    build_judge_messages,
    extract_candidate_answer,
    extract_question_text,
    judge_rewards,
    parse_judge_response,
    persist_candidate,
    select_candidate_evidence,
)


def candidate(root, *, label="A", turn=1, step=1, identity="row", color="red"):
    return persist_candidate(
        root,
        question="How many objects?",
        answer=r"\boxed{3}",
        initial_images=[Image.new("RGB", (64, 32), color)],
        tool_images=[Image.new("RGB", (32, 32), color)],
        candidate_turn=turn,
        tool_call_count=0 if turn == 1 else turn - 1,
        accuracy=1.0 if label == "A" else 0.5,
        source_step=step,
        source_ids={"request_id": identity},
        entry_id=identity,
    )


def test_candidate_sanitization_keeps_only_box_or_answer_body():
    trace = r"<think>private old reasoning</think><answer>Explanation. \boxed{\frac{1}{2}}</answer>"
    assert extract_candidate_answer(trace) == r"\boxed{\frac{1}{2}}"
    assert (
        extract_candidate_answer("<think>secret</think><answer>all unboxed answer text</answer>")
        == "all unboxed answer text"
    )
    assert extract_candidate_answer("<think>secret</think>") is None
    assert extract_candidate_answer("<answer><think>leaked</think>answer</answer>") is None
    assert (
        extract_question_text(
            [
                {"role": "system", "content": "do not copy system"},
                {
                    "role": "user",
                    "content": [{"type": "image", "image": "original.png"}, {"type": "text", "text": "Question?"}],
                },
                {"role": "assistant", "content": "private reasoning"},
                {"role": "tool", "content": "tool text"},
            ]
        )
        == "Question?"
    )


def test_first_turn_exception_preserves_multiple_initial_images_and_later_only_crops(tmp_path):
    initial = [Image.new("RGB", (64, 32), "red"), Image.new("RGB", (32, 64), "blue")]
    crops = [Image.new("RGB", (32, 32), "green")]
    assert select_candidate_evidence(initial, crops, 1, 0) == ("initial", initial)
    assert select_candidate_evidence(initial, crops, 2, 1) == ("tool_outputs", crops)
    messages = build_judge_messages("Question?", "Answer", initial)
    image_blocks = [block for block in messages[1]["content"] if block["type"] == "image"]
    assert [(block["resized_width"], block["resized_height"]) for block in image_blocks] == [(64, 32), (32, 64)]
    assert not any("ground_truth" in message for message in messages)
    entry = candidate(tmp_path, label="R")
    assert entry["label"] == "R" and entry["evidence_mode"] == "initial"
    with Image.open(entry["images"][0]["path"]) as saved:
        assert saved.size == (64, 32)


def test_worker_persistence_dedups_pixels_and_survives_temporary_crop_cleanup(tmp_path):
    temporary = tmp_path / "temporary.png"
    Image.new("RGB", (32, 32), "green").save(temporary)
    root = tmp_path / "cache"
    entry = persist_candidate(
        root,
        question="Q",
        answer="A",
        initial_images=[],
        tool_images=[{"path": str(temporary)}],
        candidate_turn=3,
        tool_call_count=2,
        accuracy=1,
        source_step=1,
        source_ids={"request_id": "source", "ground_truth": "hidden", "thinking": "private"},
    )
    temporary.unlink()
    again = candidate(root, turn=3, color="green", identity="other")
    assert entry["images"][0]["path"] == again["images"][0]["path"]
    assert Path(entry["images"][0]["path"]).is_file()
    assert entry["source_ids"] == {"request_id": "source"}
    assert len(list((root / "images").glob("*.png"))) == 1


def test_cache_balances_labels_turns_and_excludes_current_or_future_batches(tmp_path):
    root = tmp_path / "cache"
    cache = ReplayCache(root, capacity=20)
    entries = [
        candidate(root, label=label, turn=turn, identity=f"{label}{turn}")
        for label in ("A", "R")
        for turn in range(1, 6)
    ]
    entries.append(candidate(root, identity="current", step=2))
    cache.commit(entries, step=2)
    sampled = cache.sample(10, step=2)
    assert len(sampled) == 10
    assert {entry["entry_id"] for entry in sampled} == {
        f"{label}{turn}" for label in ("A", "R") for turn in range(1, 6)
    }
    assert cache.sample(26, step=1) == []
    assert sampled == cache.sample(10, step=2)
    assert sum(entry["label"] == "A" for entry in sampled) == 5
    cache.reload()
    assert len(cache) == 11


def test_shortage_uses_balanced_pairs_without_replacement(tmp_path):
    cache = ReplayCache(tmp_path, capacity=20)
    cache.commit([candidate(tmp_path, label="R", identity=str(i)) for i in range(8)], step=1)
    assert cache.sample(26, step=2) == []
    cache.commit([candidate(tmp_path, identity="only_A", step=2)], step=2)
    sampled = cache.sample(26, step=3)
    assert len(sampled) == 2 and {entry["label"] for entry in sampled} == {"A", "R"}


def test_eligibility_backfills_majority_class_despite_sparse_other_class(tmp_path):
    cache = ReplayCache(tmp_path, capacity=100)
    rows = [candidate(tmp_path, label="A", identity="only_A")]
    rows += [candidate(tmp_path, label="R", identity=f"R{i}", turn=2 + i % 6) for i in range(20)]
    cache.commit(rows, step=1)
    ordering = []
    cache.sample(26, step=2, eligible=lambda entry: ordering.append(entry["entry_id"]) or False)
    last_reject = ordering[-1]
    visited = []

    def eligible(entry):
        visited.append(entry["entry_id"])
        entry["prompt_tokens"] = 100
        return entry["label"] == "A" or entry["entry_id"] == last_reject

    sampled = cache.sample(26, step=2, eligible=eligible)
    assert {entry["entry_id"] for entry in sampled} == {"only_A", last_reject}
    assert len(visited) == len(set(visited)) == 21
    assert all(entry["prompt_tokens"] == 100 for entry in sampled)
    assert all("prompt_tokens" not in entry for entry in cache.entries)


def test_eviction_preserves_rare_turns_and_full_credit_labels(tmp_path):
    cache = ReplayCache(tmp_path, capacity=8)
    rows = [candidate(tmp_path, identity=f"A{i}", step=i + 1) for i in range(9)]
    rows.extend(candidate(tmp_path, turn=turn, identity=f"rare{turn}", step=1) for turn in (3, 7))
    rows.extend(candidate(tmp_path, label="R", identity=f"R{i}") for i in range(9))
    cache.commit(rows, step=10)
    assert len(cache) == 8
    assert sum(entry["label"] == "A" for entry in cache.entries) == 4
    assert {entry["turn"] for entry in cache.entries if entry["label"] == "A"} == {1, 3, 7}
    assert all(entry["accuracy"] == 0.5 for entry in cache.entries if entry["label"] == "R")


def test_checkpoint_restore_uses_selected_snapshot_and_gc_pins_its_assets(tmp_path):
    root = tmp_path / "trajectory_replay"
    cache = ReplayCache(root, capacity=2)
    first = candidate(root, identity="first", color="red")
    cache.commit([first], step=1)
    checkpoint = tmp_path / "global_step_1"
    cache.snapshot(checkpoint)
    second = candidate(root, identity="second", color="blue", step=2)
    cache.commit([second], step=2)
    orphan = candidate(root, identity="orphan", color="green")
    assert cache.garbage_collect() == 1
    assert Path(first["images"][0]["path"]).exists()
    assert not Path(orphan["images"][0]["path"]).exists()
    cache.restore(checkpoint)
    assert cache.step == 1 and cache.entries == [first]
    assert json.loads(cache.manifest_path.read_text())["step"] == 1
    with pytest.raises(FileNotFoundError):
        cache.restore(tmp_path / "global_step_0")
    with pytest.raises(ValueError, match="capacity/seed"):
        ReplayCache(root, capacity=4)


def test_pruning_retained_checkpoint_releases_only_its_unreferenced_assets(tmp_path):
    root = tmp_path / "trajectory_replay"
    cache = ReplayCache(root, capacity=2)
    first = candidate(root, identity="first", color="red")
    cache.commit([first], step=1)
    checkpoint = tmp_path / "global_step_1"
    cache.snapshot(checkpoint)
    current = candidate(root, identity="newer", color="blue", step=2)
    cache.commit([current], step=2)
    assert cache.garbage_collect() == 0
    assert Path(first["images"][0]["path"]).is_file()
    (checkpoint / cache.SNAPSHOT_NAME).unlink()
    assert cache.garbage_collect() == 1
    assert not Path(first["images"][0]["path"]).exists()
    assert Path(current["images"][0]["path"]).is_file()


@pytest.mark.parametrize(
    "bad",
    [
        "<answer>A</answer>",
        "<think></think><answer>A</answer>",
        "<think>x</think><answer>Accept</answer>",
        "<think>x<tool_call>{}</tool_call></think><answer>A</answer>",
        "<think>x</think><answer>A</answer>trailing",
    ],
)
def test_judge_format_requires_sampled_reasoning_and_one_decision(bad):
    assert parse_judge_response(bad) is None


def test_overlong_penalty_enters_format_component_at_full_strength():
    valid = "<think>Evidence supports it.</think><answer>A</answer>"
    short = judge_rewards(valid, "A", 4096)
    half = judge_rewards(valid, "A", 4608)
    capped = judge_rewards(valid, "A", 5120)
    assert short["score"] == 1.0 and short["format"] == 1.0
    assert half["score"] == 0.5 and half["format"] == -1.5
    assert capped["score"] == 0.0 and capped["format"] == -4.0
    assert capped["raw_format"] == 1.0 and capped["accuracy"] == 1.0
    assert 0.8 * half["accuracy"] + 0.2 * half["format"] == half["score"]


@pytest.mark.parametrize("text", ["<answer>A</answer>", "<think></think><answer>A</answer>"])
def test_decision_accuracy_is_independent_of_reasoning_format(text):
    reward = judge_rewards(text, "A", 20)
    assert reward["accuracy"] == 1.0 and reward["raw_format"] == 0.0
    assert reward["score"] == 0.8 and reward["decision"] == "A"
    assert parse_judge_response(text) is None


def test_wrong_or_ambiguous_decisions_keep_separate_format_accuracy_components():
    wrong = judge_rewards("<think>Evidence.</think><answer>R</answer>", "A", 20)
    assert wrong["accuracy"] == 0.0 and wrong["raw_format"] == 1.0 and wrong["score"] == 0.2
    ambiguous = judge_rewards("<think>Evidence.</think><answer>A</answer><answer>R</answer>", "A", 20)
    assert ambiguous["decision"] is None and ambiguous["accuracy"] == ambiguous["raw_format"] == 0.0
    incomplete = judge_rewards("<answer>A</answer><answer>", "A", 20)
    assert incomplete["decision"] is None and incomplete["accuracy"] == 0.0
