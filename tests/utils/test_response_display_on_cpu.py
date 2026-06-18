from omegaconf import OmegaConf

from verl.utils.response_display import (
    apply_response_display_prefix,
    apply_response_display_prefixes,
    get_response_display_prefix,
)


def test_response_display_prefix_from_config():
    cfg = OmegaConf.create({"reward": {"reward_kwargs": {"reward_response_prefix": "<think>\n"}}})

    assert get_response_display_prefix(cfg) == "<think>\n"
    assert get_response_display_prefix({}) == ""
    assert get_response_display_prefix(None) == ""


def test_response_display_prefix_is_idempotent_and_display_only():
    response_ids = [101, 102, 103]
    response_mask = [1, 1, 0]
    original_response_ids = list(response_ids)
    original_response_mask = list(response_mask)

    assert apply_response_display_prefix("answer", "<think>\n") == "<think>\nanswer"
    assert apply_response_display_prefix("<think>\nanswer", "<think>\n") == "<think>\nanswer"
    assert apply_response_display_prefix("answer", "") == "answer"
    assert apply_response_display_prefixes(["a", "<think>\nb"], "<think>\n") == ["<think>\na", "<think>\nb"]

    assert response_ids == original_response_ids
    assert response_mask == original_response_mask


def test_display_prefix_does_not_change_rollout_payload():
    rollout_payload = {
        "prompts": [[1, 2]],
        "responses": [[3, 4]],
        "response_mask": [[1, 1]],
        "attention_mask": [[1, 1, 1, 1]],
    }
    original_payload = {key: [list(row) for row in value] for key, value in rollout_payload.items()}

    displayed_outputs = apply_response_display_prefixes(["answer"], "<think>\n")

    assert displayed_outputs == ["<think>\nanswer"]
    assert rollout_payload == original_payload
