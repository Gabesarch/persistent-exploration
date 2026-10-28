"""Native Gemma4 training/evaluation parity with the real cached tokenizer."""
import copy
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'vero_eval'))
from lmms_eval.models.chat.async_openai import AsyncOpenAIChat
from lmms_eval.models.model_utils.gemma4_training_alignment import Gemma4TrainingAlignment, THOUGHT, END_THOUGHT, CONTINUATION
from verl.models.gemma4 import protocol as native
from verl.utils.forced_action_tags import ForceActionTagAfterThinkRequestProcessor, FORCE_ACTION_TAG_IDS_KEY, FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY

BASE = Path('/scratch/gpfs/ZHUANGL/gs1693/vero2/gemma4_training')
REGISTRATIONS = [
    ('branchstrata20_fresh_lr1e5_critic_gemma4_e4b', '15884866', True),
    ('gdpo_fresh_lr1e5_noexplore_accformat_gemma4_e4b', '15910097', False),
]


@pytest.fixture(params=REGISTRATIONS)
def profile(request):
    name, job, _ = request.param
    return Gemma4TrainingAlignment(BASE / 'manifests' / name / job / 'resolved.yaml')


@pytest.fixture(scope='module')
def tokenizer():
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained('/scratch/gpfs/ZHUANGL/gs1693/hf_cache/google/gemma-4-E4B-it', local_files_only=True)


def model(profile, tokenizer):
    instance = object.__new__(AsyncOpenAIChat)
    instance.training_alignment = profile
    profile.configure(instance)
    instance._forced_action_tokenizer = tokenizer
    instance._retry_critic_accept_token_id = tokenizer.encode('A', add_special_tokens=False)[0]
    instance._retry_critic_retry_token_id = tokenizer.encode('R', add_special_tokens=False)[0]
    return instance


def render(profile, tokenizer, payload):
    extra = payload.get('extra_body', {})
    return tokenizer.apply_chat_template(payload['messages'], tools=payload.get('tools'),
        chat_template=profile.template + CONTINUATION, tokenize=False,
        add_generation_prompt=extra.get('add_generation_prompt', True),
        **{'enable_thinking': True, 'preserve_thinking': True, **extra.get('chat_template_kwargs', {})})


def payload(profile):
    return {'messages': [{'role': 'system', 'content': profile.system_prompt},
                         {'role': 'user', 'content': 'What is shown?\nImage 0 is available as img_idx=0.'}],
            'tools': [profile.tool_schema], 'tool_choice': 'auto', 'extra_body': {'top_k': 20}}


def test_validation_controller_and_prompt(profile, tokenizer):
    instance = model(profile, tokenizer)
    assert instance.critic_guided_answer_retry == ('critic' in profile.config['trainer']['experiment_name'])
    assert not instance.force_tool_calls_until_final_turn
    assert instance.force_answer_after_max_tool_iterations
    assert instance.max_tool_iterations == 10 and instance.tool_image_as_user_turn
    assert profile.validation_evidence
    assert tokenizer.encode('A', add_special_tokens=False) == [instance._retry_critic_accept_token_id]
    assert tokenizer.encode('R', add_special_tokens=False) == [instance._retry_critic_retry_token_id]
    for key in ('train_files', 'val_files'):
        rows = [json.loads(line) for line in Path(profile.data[key]).read_text().splitlines() if line.strip()]
        assert len(rows) > 0
        for row in rows:
            for message in row['prompt']:
                if message['role'] == 'system':
                    assert message['content'] == profile.system_prompt


def test_native_template_unchanged_for_ordinary_turns(profile, tokenizer):
    request = payload(profile)
    expected = tokenizer.apply_chat_template(request['messages'], tools=request['tools'],
        chat_template=profile.template, tokenize=False, add_generation_prompt=True,
        enable_thinking=True, preserve_thinking=True)
    assert render(profile, tokenizer, request) == expected
    assert '<|think|>' in expected and '<think>' not in expected


def test_forced_final_uses_training_native_channel(profile, tokenizer):
    instance = model(profile, tokenizer)
    request = payload(profile)
    expected = render(profile, tokenizer, request)
    final, prefill = instance._forced_final_answer_request(request, hard_fallback=False)
    assert render(profile, tokenizer, final) == expected + THOUGHT + profile.multi['force_final_answer_think_prefill']
    assert final['tool_choice'] == 'none' and final['tools'] == request['tools']
    assert prefill == THOUGHT + profile.multi['force_final_answer_think_prefill']
    force = final['extra_body']['vllm_xargs']
    assert tokenizer.decode(force[FORCE_ACTION_TAG_IDS_KEY]) == '<answer>'
    assert json.loads(force[FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY]) == [tokenizer.encode(END_THOUGHT, add_special_tokens=False)]


def test_critic_prompt_and_rollback_retry(profile, tokenizer):
    instance = model(profile, tokenizer)
    if not profile.eval_critic_enabled:
        return
    request = payload(profile)
    response = SimpleNamespace(content='<answer>candidate</answer>', reasoning='Visible evidence.', tool_calls=None)
    candidate = profile.candidate_content(response)
    critic = instance._retry_critic_request(request, candidate)
    text = render(profile, tokenizer, critic)
    assert text.endswith(native.USER + profile.multi['conditional_answer_retry_critic_question'] + '<turn|>\n'
                         + native.MODEL + profile.multi['conditional_answer_retry_critic_assistant_prefill'])
    assert THOUGHT + 'Visible evidence.' + END_THOUGHT + '<answer>candidate</answer>' in text
    with_end = instance._retry_critic_request(request, candidate + '<turn|>')
    assert render(profile, tokenizer, with_end) == text
    assert critic['extra_body']['allowed_token_ids'] == [instance._retry_critic_accept_token_id, instance._retry_critic_retry_token_id]
    assert critic['extra_body']['top_k'] == -1 and critic['max_tokens'] == 1
    retry, prefill = instance._retry_tool_call_request(request)
    assert render(profile, tokenizer, retry) == render(profile, tokenizer, request) + THOUGHT + profile.multi['conditional_answer_retry_assistant_prefill']
    assert all('candidate' not in str(message) for message in retry['messages'])
    force = retry['extra_body']['vllm_xargs']
    assert tokenizer.decode(force[FORCE_ACTION_TAG_IDS_KEY]) == '<|tool_call>call:'
    ids = tokenizer.encode(END_THOUGHT, add_special_tokens=False)
    action = ForceActionTagAfterThinkRequestProcessor([ids], force[FORCE_ACTION_TAG_IDS_KEY])
    assert action._forced_token_id([], ids) == force[FORCE_ACTION_TAG_IDS_KEY][0]
    history = profile.assistant_message(SimpleNamespace(content='', reasoning='Find another region.'), prefill)
    assert history['content'] == '' and history['reasoning'] == profile.multi['conditional_answer_retry_assistant_prefill'] + 'Find another region.'


def test_fractional_crop_response_preserves_training_wording(profile, tokenizer):
    instance = model(profile, tokenizer)
    args = {'bbox_2d': [100.5, 200, 700.5, 800], 'label': 'region'}
    assert instance._tool_image_registered_text('multi_index', 'image_zoom_in_tool', args, 1) == (
        'Zoomed in on the image to the region [100.5, 200, 700.5, 800] with label region. '
        'The cropped view is now available as img_idx 1.')


def test_shard_and_tracker_completeness(tmp_path):
    spec = importlib.util.spec_from_file_location('submit_gemma4', ROOT / 'vero_eval/examples/submit_zoom_eval_checkpoints_gemma4.py')
    submit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(submit)
    for number in (0, 1, 50, 99, 100, 101, 199, 200, 300):
        (tmp_path / f'global_step_{number}').mkdir()
    (tmp_path / 'global_step_400.incomplete').mkdir()
    assert [p.name for p in submit.selected_steps(tmp_path)] == ['global_step_100', 'global_step_200', 'global_step_300']
    step = tmp_path / 'global_step_100'
    hf = step / 'actor/huggingface'
    hf.mkdir(parents=True)
    for name in ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'processor_config.json', 'chat_template.jinja'):
        (hf / name).write_text('{}')
    for rank in range(2):
        (step / 'actor' / f'model_world_size_2_rank_{rank}.pt').write_bytes(b'shard')
    tracker = tmp_path / 'latest_checkpointed_iteration.txt'
    tracker.write_text('90')
    assert not submit.complete_checkpoint(step)
    tracker.write_text('100')
    assert submit.complete_checkpoint(step)
    (step / 'actor/model_world_size_2_rank_1.pt').unlink()
    assert not submit.complete_checkpoint(step)


def test_unsupported_forced_exploration_fails_closed(profile, tmp_path):
    import yaml
    config = copy.deepcopy(profile.config)
    config['actor_rollout_ref']['rollout']['multi_turn']['force_tool_calls_until_final_turn'] = True
    path = tmp_path / 'resolved.yaml'
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='no forced exploration'):
        Gemma4TrainingAlignment(path)
