from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop


EXPLORE_SUFFIX = (
    "Even if you are confident in your answer, please explore more by outputting a zoom tool call in the next "
    "turn. You are encouraged to use depth first search (zooming deeper into previously explored regions) and "
    "breadth first search (zooming into image regions not yet examined), and are discouraged from similarly "
    "repeating previous zoom explorations."
)
FINAL_SUFFIX = "The maximum amount of tool calling has been reached. Please output a final answer in the next turn."


def _make_loop(enabled: bool = True, explore_turns: int = 10) -> ToolAgentLoop:
    loop = object.__new__(ToolAgentLoop)
    loop.tool_response_turn_suffix_enabled = enabled
    loop.tool_response_explore_turns = explore_turns
    loop.tool_response_explore_suffix = EXPLORE_SUFFIX
    loop.tool_response_final_suffix = FINAL_SUFFIX
    return loop


def test_tool_response_turn_suffix_uses_explore_suffix_before_final_turn_on_cpu():
    loop = _make_loop(explore_turns=10)

    assert loop._tool_response_turn_suffix(1) == EXPLORE_SUFFIX
    assert loop._tool_response_turn_suffix(9) == EXPLORE_SUFFIX


def test_tool_response_turn_suffix_uses_final_suffix_at_explore_limit_on_cpu():
    loop = _make_loop(explore_turns=10)

    assert loop._tool_response_turn_suffix(10) == FINAL_SUFFIX
    assert loop._tool_response_turn_suffix(11) == ""


def test_tool_response_turn_suffix_is_empty_when_disabled_on_cpu():
    loop = _make_loop(enabled=False, explore_turns=10)

    assert loop._tool_response_turn_suffix(1) == ""
    assert loop._tool_response_turn_suffix(10) == ""


def test_append_suffix_to_text_tool_message_on_cpu():
    loop = _make_loop()
    message = {"role": "tool", "content": "crop result"}

    updated = loop._append_suffix_to_tool_message(message, EXPLORE_SUFFIX)

    assert updated == {"role": "tool", "content": f"crop result\n\n{EXPLORE_SUFFIX}"}
    assert message == {"role": "tool", "content": "crop result"}


def test_append_suffix_to_structured_multimodal_tool_message_on_cpu():
    loop = _make_loop()
    message = {
        "role": "tool",
        "content": [
            {"type": "image"},
            {"type": "text", "text": "crop result"},
        ],
    }

    updated = loop._append_suffix_to_tool_message(message, FINAL_SUFFIX)

    assert updated["role"] == "tool"
    assert updated["content"][0] == {"type": "image"}
    assert updated["content"][1] == {"type": "text", "text": f"crop result\n\n{FINAL_SUFFIX}"}


def test_append_suffix_to_structured_multimodal_tool_message_without_text_on_cpu():
    loop = _make_loop()
    message = {"role": "tool", "content": [{"type": "image"}]}

    updated = loop._append_suffix_to_tool_message(message, EXPLORE_SUFFIX)

    assert updated["content"] == [{"type": "image"}, {"type": "text", "text": EXPLORE_SUFFIX}]
