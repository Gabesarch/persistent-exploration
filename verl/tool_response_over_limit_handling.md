# Tool Response Over-Limit Handling

## Problem

In multi-turn tool rollouts, a tool call can execute successfully but the rendered
tool-response message can exceed `actor_rollout_ref.rollout.response_length`.

Before the fix, `_handle_processing_tools_state` committed some tool-response
state before checking the response-length budget:

- tool messages were added to `agent_data.messages`
- crop refs were added to `agent_data.crop_image_refs`
- tool rewards were added to `agent_data.tool_rewards`

Only after that did the code render `response_ids` and check:

```python
if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
    return AgentState.TERMINATED
```

If this branch fired, the tool image was not added to `agent_data.image_data`,
and the tool-response tokens were not added to `prompt_ids` or `response_mask`.
This left the sample internally inconsistent.

The observed failure was:

```text
original_metadata image count = 5
rebuilt_metadata image count  = 6
crop_image_refs count         = 5 drawn images
```

The final response contained five `<tool_call>` blocks but only four consumed
`<tool_response>` blocks. The fifth tool image was saved in `crop_image_refs`,
but it was not part of the rollout multimodal inputs used for log-prob scoring.
Lazy multimodal rehydration then appended all crop refs and rebuilt one extra
image, causing an `image_grid_thw` mismatch.

## Fix

Tool-response state is now staged locally first:

```python
pending_crop_refs = []
pending_tool_rewards = []
add_messages = []
new_images_this_turn = []
```

While processing tool outputs, the code appends only to these local lists. It then
renders the would-be tool-response tokens:

```python
response_ids = await self.apply_chat_template(
    add_messages,
    images=new_images_this_turn if new_images_this_turn else None,
    videos=None,
    remove_system_prompt=True,
)
```

The response-length check happens before mutating persistent rollout state:

```python
if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
    return AgentState.TERMINATED
```

If the tool response is over limit, the whole tool-response transaction is
dropped from the training sample:

- no tool message is added to `agent_data.messages`
- no image is added to `agent_data.image_data`
- no crop ref is added to `agent_data.crop_image_refs`
- no tool reward is added
- no `prompt_ids` are added
- no `response_mask` zeros are added

The final sample consistently ends at the assistant `<tool_call>`.

If the tool response fits, all related state is committed together:

```python
agent_data.messages.extend(add_messages)
agent_data.crop_image_refs.extend(pending_crop_refs)
agent_data.tool_rewards.extend(pending_tool_rewards)

for img in new_images_this_turn:
    agent_data.image_data.append(img)

agent_data.prompt_ids += response_ids
agent_data.response_mask += [0] * len(response_ids)
agent_data.user_turns += 1
```

Crop-ref ordering also accounts for staged refs:

```python
crop_ref["order"] = len(agent_data.crop_image_refs) + len(pending_crop_refs)
```

This keeps ordering correct if a tool response returns multiple images.

## Debug Event

When a tool response is dropped because it would exceed the response budget, the
agent loop records:

```text
tool_response_dropped_over_limit
```

with:

- `dropped_tool_messages`
- `dropped_images`
- `dropped_crop_refs`
- `current_response_tokens`
- `tool_response_tokens`
- `max_response_tokens`

This makes future over-limit drops explicit in comprehensive debug logs.
