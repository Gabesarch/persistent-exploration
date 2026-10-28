"""Qwen3.5 XML calls with the control run's unchanged accuracy/format rewards."""

import json
import re

from verl.utils.reward_score import math_verify_reward_type_multiturn_tool_calling_optional_think as control


def canonicalize_tool_calls(text: str) -> str:
    """Translate complete XML calls for the JSON scorer; preserve malformed calls."""
    def replace(match):
        payload = match.group(1).strip()
        function = re.fullmatch(r"<function=([\w.-]+)>\s*(.*?)\s*</function>", payload, re.S)
        if function is None:
            # JSON is not the configured policy protocol. Keep it invalid for reward.
            return "<tool_call>INVALID_QWEN35_XML</tool_call>"
        arguments = {}
        pos = 0
        params = function.group(2)
        for parameter in re.finditer(r"<parameter=([\w.-]+)>(.*?)</parameter>", params, re.S):
            if params[pos:parameter.start()].strip() or parameter.group(1) in arguments:
                return match.group(0)
            key, value = parameter.group(1), parameter.group(2).strip()
            if key in {"bbox_2d", "img_idx"}:
                try:
                    value = json.loads(value)
                except json.JSONDecodeError:
                    return match.group(0)
            arguments[key] = value
            pos = parameter.end()
        if params[pos:].strip():
            return match.group(0)
        return '<tool_call>' + json.dumps({"name": function.group(1), "arguments": arguments}) + '</tool_call>'

    return re.sub(r"<tool_call>(.*?)</tool_call>", replace, str(text), flags=re.S)


def parse_trajectory(text):
    return control._parse_multiturn_tool_calling_output(canonicalize_tool_calls(text))


async def async_compute_score_from_data_source(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    return await control.async_compute_score_from_data_source(
        data_source=data_source,
        solution_str=canonicalize_tool_calls(solution_str),
        ground_truth=ground_truth,
        extra_info=extra_info,
        **kwargs,
    )
