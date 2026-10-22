"""XML-aware trajectory parsing for the existing tool controller and search judge."""

from verl.experimental.agent_loop.agent_loop import register
from verl.experimental.agent_loop.tool_agent_loop import ToolAgentLoop
from verl.utils.reward_score.qwen35_multiturn import parse_trajectory


@register("qwen35_tool_agent")
class Qwen35ToolAgentLoop(ToolAgentLoop):
    def _parse_solver_output(self, text):
        return parse_trajectory(text)
