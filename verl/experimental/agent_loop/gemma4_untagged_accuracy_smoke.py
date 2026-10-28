"""Smoke-only judge extraction; native rollout and retry behavior are inherited."""

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.gemma4_tool_agent_loop import Gemma4ToolAgentLoop
from verl.utils.reward_score.gemma4_untagged_accuracy_smoke import parse_accuracy_trajectory


@register("gemma4_untagged_accuracy_smoke")
class Gemma4UntaggedAccuracySmokeAgent(Gemma4ToolAgentLoop):
    def _parse_solver_output(self, text):
        return parse_accuracy_trajectory(text)
