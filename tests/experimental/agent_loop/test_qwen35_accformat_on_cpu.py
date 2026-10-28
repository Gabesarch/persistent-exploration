import asyncio
import json
import unittest
from pathlib import Path

from omegaconf import OmegaConf
from transformers import AutoTokenizer

from verl.experimental.agent_loop.tool_parser import ToolParser
from verl.tools.schemas import OpenAIFunctionToolSchema
from verl.utils.reward_score import math_verify_reward_type_multiturn_tool_calling_optional_think as control
from verl.utils.reward_score import qwen35_multiturn as native
from verl.utils.tokenizer import hf_processor


ROOT = Path(__file__).resolve().parents[3]
CALL = '<tool_call><function=image_zoom_in_tool>\n<parameter=bbox_2d>[100, 200, 600, 800]</parameter>\n<parameter=label>small object</parameter>\n<parameter=img_idx>0</parameter>\n</function></tool_call>'
TRAJECTORY = '<think>Inspect the object.</think>' + CALL + '<tool_response>Crop image</tool_response><think>The crop shows a cat.</think><answer>\\boxed{cat}</answer>'


def test_xml_scoring_matches_control():
    parsed = native.parse_trajectory(TRAJECTORY)
    assert parsed[:3] == (True, r'\boxed{cat}', 1)
    assert parsed[3][0]['arguments'] == {'bbox_2d': [100, 200, 600, 800], 'label': 'small object', 'img_idx': 0}
    kwargs = dict(data_source='test', ground_truth='cat', extra_info={'reward_type': 'search'}, require_boxed_answer_for_full_format=True)
    score = asyncio.run(native.async_compute_score_from_data_source(solution_str=TRAJECTORY, **kwargs))
    expected = asyncio.run(control.async_compute_score_from_data_source(solution_str=native.canonicalize_tool_calls(TRAJECTORY), **kwargs))
    assert score == expected
    assert score['format'] == 1.0
    assert score['accuracy'] == 1.0


def test_invalid_or_json_call_rejected():
    for bad in [CALL.replace('</function>', ''), CALL.replace('<parameter=img_idx>0', '<parameter=img_idx>oops'), '<tool_call>{"name":"image_zoom_in_tool","arguments":{}}</tool_call>']:
        assert not native.parse_trajectory(TRAJECTORY.replace(CALL, bad))[0]


def test_real_tokenizer_xml_parser_and_chat_template():
    tokenizer = AutoTokenizer.from_pretrained('/scratch/gpfs/ZHUANGL/gs1693/hf_cache/Qwen/Qwen3.5-9B', local_files_only=True)
    tools_config = OmegaConf.load(ROOT / 'vero2_scripts/config/tool_config/image_zoom_in_tool.yaml')
    tools = OmegaConf.to_container(tools_config, resolve=True)['tools']
    schemas = [OpenAIFunctionToolSchema.model_validate(tool['tool_schema']) for tool in tools]
    parser = ToolParser.get_tool_parser('qwen3_coder', tokenizer)
    loop = asyncio.new_event_loop()
    try:
        # Match the persistent worker loop; asyncio.run's executor shutdown can
        # stall in the network-restricted local sandbox after decode completes.
        _, calls = loop.run_until_complete(
            parser.extract_tool_calls(tokenizer.encode(CALL, add_special_tokens=False), schemas)
        )
    finally:
        loop.close()
    assert len(calls) == 1
    assert calls[0].name == 'image_zoom_in_tool'
    assert json.loads(calls[0].arguments) == native.parse_trajectory(TRAJECTORY)[3][0]['arguments']
    assert '<tool_call>' in tokenizer.decode(tokenizer.encode(CALL, add_special_tokens=False), skip_special_tokens=True)
    config = OmegaConf.load(ROOT / 'vero2_scripts/config/gdpo_fresh_lr1e5_noexplore_accformat_qwen35_9b.yaml')
    tokenizer.chat_template = config.actor_rollout_ref.model.custom_chat_template
    messages = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'What animal?'}]}]
    prompt = tokenizer.apply_chat_template(messages, tools=[tool['tool_schema'] for tool in tools], tokenize=False, add_generation_prompt=True)
    assert prompt.endswith('<|im_start|>assistant\n<think>\n')
    assert '<|vision_start|><|image_pad|><|vision_end|>' in prompt
    assert '<function=FUNCTION_NAME>' in prompt
    assert 'put only the json object' not in prompt
    messages += [{'role': 'assistant', 'content': '<think>Inspect.</think>', 'tool_calls': [{'function': native.parse_trajectory(TRAJECTORY)[3][0]}]}, {'role': 'tool', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'Crop index 1'}]}]
    history = tokenizer.apply_chat_template(messages, tools=[tool['tool_schema'] for tool in tools], tokenize=False, add_generation_prompt=True)
    assert '<function=image_zoom_in_tool>' in history
    assert history.count('<|image_pad|>') == 2
    assert '<tool_response>' in history


def test_multimodal_processor_positions():
    from PIL import Image
    import torch

    processor = hf_processor('/scratch/gpfs/ZHUANGL/gs1693/hf_cache/Qwen/Qwen3.5-9B', local_files_only=True)
    assert processor is not None and hasattr(processor, 'get_rope_index')
    processor.chat_template = OmegaConf.load(
        ROOT / 'vero2_scripts/config/gdpo_fresh_lr1e5_noexplore_accformat_qwen35_9b.yaml'
    ).actor_rollout_ref.model.custom_chat_template
    text = processor.apply_chat_template(
        [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'What color?'}]}],
        tokenize=False, add_generation_prompt=True,
    )
    inputs = processor(text=[text], images=[Image.new('RGB', (256, 256), 'blue')], return_tensors='pt')
    positions, _ = processor.get_rope_index(
        input_ids=inputs.input_ids, attention_mask=inputs.attention_mask,
        image_grid_thw=inputs.image_grid_thw, mm_token_type_ids=inputs.mm_token_type_ids,
    )
    assert positions.shape == (3, 1, inputs.input_ids.shape[1])
    assert torch.isfinite(positions).all()
    assert processor.config.model_type == 'qwen3_5'


if __name__ == '__main__':
    suite = unittest.TestSuite(unittest.FunctionTestCase(test) for test in [
        test_xml_scoring_matches_control,
        test_invalid_or_json_call_rejected,
        test_real_tokenizer_xml_parser_and_chat_template,
        test_multimodal_processor_positions,
    ])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    raise SystemExit(not result.wasSuccessful())
