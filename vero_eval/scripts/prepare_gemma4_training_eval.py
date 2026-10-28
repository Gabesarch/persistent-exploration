#!/usr/bin/env python3
"""Save native training protocol evidence and configure its evaluation runtime."""
import argparse
import json
import os
from pathlib import Path
import shlex
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lmms_eval.models.model_utils.gemma4_training_alignment import Gemma4TrainingAlignment, THOUGHT, END_THOUGHT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--environment', action='store_true')
    parser.add_argument('--probe-url')
    args = parser.parse_args()
    profile = Gemma4TrainingAlignment(os.environ['GEMMA_TRAINING_CONFIG'], os.environ['CHECKPOINT_PATH'])
    if args.probe_url:
        from openai import OpenAI
        from transformers import AutoTokenizer
        from verl.utils.forced_action_tags import FORCE_ACTION_TAG_IDS_KEY, VLLM_OPENAI_EXTRA_ARGS_KEY
        tokenizer = AutoTokenizer.from_pretrained(profile.checkpoint, local_files_only=True)
        force = profile.forced_action_extra_body(tokenizer, '<answer>')
        body = profile.continuation_request({'messages': [
            {'role': 'system', 'content': profile.system_prompt},
            {'role': 'user', 'content': 'Answer with the word ready.'},
        ]}, THOUGHT + 'Native final-answer API canary.' + END_THOUGHT)
        body['extra_body'].update(force)
        response = OpenAI(api_key=os.getenv('API_KEY', 'EMPTY'), base_url=args.probe_url).chat.completions.create(
            model=os.environ['MODEL_NAME'], **body, tools=[profile.tool_schema], tool_choice='none',
            temperature=0, max_tokens=len(force[VLLM_OPENAI_EXTRA_ARGS_KEY][FORCE_ACTION_TAG_IDS_KEY]) + 1)
        message = response.choices[0].message
        if not (message.content or '').startswith('<answer>') or message.tool_calls:
            raise SystemExit(f'Native Gemma forced-answer canary failed: {response}')
        print('Native Gemma4 forced-answer API canary passed.')
        return
    output = Path(os.environ['OUTPUT_ROOT']) / 'protocol'
    settings = output.parent / 'submitted_config.json'
    if settings.exists():
        identity = profile.identity()
        saved = json.loads(settings.read_text())
        for key in ('config_sha256', 'training_root', 'system_prompt_sha256', 'template_sha256', 'tool_schema', 'multi_turn'):
            if saved[key] != identity[key]:
                raise ValueError(f'Queued Gemma4 identity changed before execution: {key}')
    output.mkdir(parents=True, exist_ok=True)
    env = profile.shell_environment(output / 'served_chat_template.jinja')
    manifest_path = output / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    if manifest:
        if manifest['system_prompt'] != profile.system_prompt or manifest['tools'] != [profile.tool_schema]:
            raise ValueError('Base Gemma preflight prompt/schema differs from saved training')
    manifest.update(training_alignment=profile.identity(), forced_tool_calls=profile.multi['force_tool_calls_until_final_turn'],
                    forced_final_answer=profile.multi['force_answer_on_final_turn_validation'],
                    critic_guided_answer_retry=profile.eval_critic_enabled,
                    mm_processor_kwargs=profile.rollout['engine_kwargs']['vllm']['mm_processor_kwargs'],
                    max_model_len=profile.rollout['max_model_len'], response_budget=profile.data['max_response_length'],
                    served_chat_template=str(output / 'served_chat_template.jinja'))
    for task in manifest.get('tasks', {}).values():
        task['generation_kwargs'].update(profile.generation_kwargs())
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    if args.environment:
        for key, value in env.items():
            value = str(value).lower() if isinstance(value, bool) else str(value)
            print(f'export {key}={shlex.quote(value)}')


if __name__ == '__main__':
    main()
