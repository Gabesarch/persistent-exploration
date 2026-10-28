#!/usr/bin/env python3
"""Summarize actual tool calls, crop feedback and final answers from saved requests."""
import argparse
import json
import re
from pathlib import Path


def summarize(root):
    samples = []
    for directory in sorted((root / 'request_traces').glob('*')):
        if not directory.is_dir():
            continue
        records = [json.loads(p.read_text()) for p in sorted(directory.glob('*.json'))]
        if not records:
            continue
        calls = [call for record in records for choice in record['response']['choices']
                 for call in choice['message'].get('tool_calls') or []]
        last = records[-1]['response']['choices'][0]
        text = last['message'].get('content') or ''
        crop_messages = [message for message in records[-1]['request']['messages'] if message['role'] == 'tool']
        reasoning = [choice['message'].get('reasoning') or choice['message'].get('reasoning_content') or ''
                     for record in records for choice in record['response']['choices']]
        samples.append({'sample': directory.name, 'requests': len(records), 'tool_calls': len(calls),
                        'tool_arguments': [json.loads(call['function']['arguments']) for call in calls],
                        'crop_responses': len(crop_messages), 'finish_reason': last['finish_reason'],
                        'reasoning_turns': sum(bool(text) for text in reasoning),
                        'reasoning_characters': sum(len(text) for text in reasoning),
                        'answer': re.findall(r'<answer>(.*?)</answer>', text, flags=re.DOTALL),
                        'final_content': text})
    result = {'samples': len(samples), 'tool_calls': sum(s['tool_calls'] for s in samples),
              'samples_with_zoom': sum(s['crop_responses'] > 0 for s in samples),
              'samples_with_answer_tags': sum(bool(s['answer']) for s in samples),
              'samples_with_reasoning': sum(s['reasoning_turns'] > 0 for s in samples),
              'reasoning_turns': sum(s['reasoning_turns'] for s in samples),
              'truncated_final_responses': sum(s['finish_reason'] == 'length' for s in samples),
              'results_files': [str(p.relative_to(root)) for p in root.rglob('*results.json')],
              'per_sample': samples}
    (root / 'zoom_test_summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'per_sample'}, indent=2))
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output_root', type=Path)
    summarize(parser.parse_args().output_root)
