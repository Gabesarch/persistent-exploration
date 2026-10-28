#!/usr/bin/env python3
"""Check checkpoint, resolved prompts/schema, multimodal tool round trip and data.

This is a CPU preflight, not evidence that GPU inference works. Also saves the
literal system prompt, rendered prompts and run manifest beside the evaluation.
"""
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

from cache_gemma_zoom_models import validate

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def check_gemma3_image_spans(checkpoint, rendered):
    """Exercise vLLM's real CPU image preprocessing, including token prompts."""
    from PIL import Image
    from vllm.config import ModelConfig
    from vllm.multimodal import MULTIMODAL_REGISTRY
    from vllm.multimodal.processing.context import TimingContext
    from vllm.multimodal.processing.inputs import ProcessorInputs

    mm_kwargs = {'do_pan_and_scan': False}
    config = ModelConfig(model=str(checkpoint), dtype='bfloat16', max_model_len=49152,
                         mm_processor_kwargs=mm_kwargs)
    processor = MULTIMODAL_REGISTRY.create_processor(config)
    tokenizer = processor.info.get_tokenizer()
    checks = {}
    for label, text in rendered.items():
        count = 1 if label == 'initial' else 2
        images = [Image.new('RGB', (200, 100)) for _ in range(count)]
        for kind, prompt in [('text', text), ('tokens', tokenizer.encode(text, add_special_tokens=False))]:
            items = processor.info.parse_mm_data({'image': images})
            result = processor.apply(
                ProcessorInputs(prompt, items, hf_processor_mm_kwargs=mm_kwargs),
                TimingContext(enabled=False))
            spans = result['mm_placeholders'].get('image', [])
            assert len(spans) == count, f'{label}/{kind}: missing original/crop image spans'
            lengths = [int(span.is_embed.sum()) if span.is_embed is not None else span.length
                       for span in spans]
            assert lengths == [256] * count, (label, kind, lengths)
            checks[f'{label}_{kind}'] = {'images': count, 'image_embedding_tokens': lengths}
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, choices=['gemma3_4b', 'gemma4_e4b'])
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--skip-data', action='store_true')
    parser.add_argument('--tasks', default=os.getenv('TASKS'), help='comma-separated Gemma task names; defaults to VisualProbe')
    args = parser.parse_args()
    os.chdir(ROOT)
    validate(args.checkpoint)
    from transformers import AutoConfig, AutoProcessor, AutoTokenizer
    from lmms_eval.utils import load_yaml_config
    from lmms_eval.models.chat.async_openai import AsyncOpenAIChat, _zoom_bbox_to_xyxy, _composite_zoom_to_original, _preserve_assistant_reasoning
    import torch
    import transformers
    from importlib.metadata import version

    config = AutoConfig.from_pretrained(args.checkpoint, local_files_only=True)
    processor = AutoProcessor.from_pretrained(args.checkpoint, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, local_files_only=True)
    no_tools = os.getenv('GEMMA_EVAL_MODE', 'zoom') == 'thinking_only'
    family = args.model.split('_')[0]
    thinking = family == 'gemma4' and os.getenv('GEMMA4_ENABLE_THINKING', 'false').lower() in {'true', '1', 'yes', 'on'}
    template_path = ROOT / 'prompts/gemma3_zoom_tools.jinja' if family == 'gemma3' else args.checkpoint / 'chat_template.jinja'
    template = template_path.read_text()
    tokenizer.chat_template = template
    expected_type = 'gemma3' if family == 'gemma3' else 'gemma4'
    assert config.model_type == expected_type, (config.model_type, expected_type)
    assert torch.version.cuda, 'A CUDA torch build is required; do not use the CPU Gemma environment'
    args.output.mkdir(parents=True, exist_ok=True)
    tasks = {}
    resolved = None
    requested = args.tasks.split(',') if args.tasks else [f'visual_probe_{level}_{family}_zoom' for level in ['easy', 'medium', 'hard']]
    expected_counts = json.loads(os.getenv('GEMMA_EXPECTED_SAMPLE_COUNTS', '{}'))
    for task in requested:
        paths = list((ROOT / 'lmms_eval/tasks').glob(f'*/{task}.yaml'))
        if len(paths) != 1:
            raise ValueError(f'Expected one task definition for {task}, found {paths}')
        cfg = load_yaml_config(str(paths[0]))
        meta = cfg['metadata']
        if no_tools:
            assert meta['allowed_tools'] == [] and meta['system_prompt_files'] == []
            assert not meta.get('extra_tools') and not meta.get('show_initial_image_index_hint')
        else:
            assert meta['allowed_tools'] == ['image_zoom_in_tool']
            assert meta['image_indexing'] == 'multi_index'
            assert meta['restore_assistant_forced_prefill'] is False
            assert meta['zoom_bbox_coordinate_order'] == 'yxyx'
            overrides = meta['tool_schema_overrides']['image_zoom_in_tool']
            assert set(overrides['parameters']['properties']) == {'bbox_2d', 'label', 'img_idx'}
        # Exercise the same prompt/schema resolver as the actual async runner.
        model = object.__new__(AsyncOpenAIChat)
        model.training_alignment = None
        model.system_prompt = None
        model.task_metadata = {task: meta}
        model._task_system_prompt_cache = {}
        model._task_tools_cache = {}
        model._tool_functions = [] if no_tools else [{'type': 'function', 'function': {'name': 'image_zoom_in_tool'}}]
        system = model._resolve_task_system_prompt(task)
        tools, terminal = model._resolve_task_tools(task)
        assert not terminal
        if no_tools:
            assert system is None and tools == []
        if resolved is not None:
            assert (system, tools) == resolved, 'Protocol must match across difficulty splits'
        resolved = system, tools
        selected = []
        dataset_count = None
        if not args.skip_data:
            import datasets
            dataset_kwargs = dict(cfg.get('dataset_kwargs') or {})
            builder = dataset_kwargs.pop('builder_fn', None)
            if builder:
                ds = builder()[cfg['test_split']]
            else:
                ds = datasets.load_dataset(cfg['dataset_path'], cfg.get('dataset_name'),
                                           split=cfg['test_split'], cache_dir=os.environ.get('HF_DATASETS_CACHE'), **dataset_kwargs)
            if cfg.get('process_docs'):
                ds = cfg['process_docs'](ds)
            dataset_count = len(ds)
            if task in expected_counts:
                assert dataset_count == expected_counts[task], (task, dataset_count, expected_counts[task])
            limit = os.getenv('LIMIT', '5')
            count = int(limit) if limit else len(ds)
            for idx in range(min(count, len(ds))):
                row = ds[idx]
                images = cfg['doc_to_visual'](row)
                assert images, f'No image loaded for {task} doc {idx}'
                selected.append({'doc_id': idx, 'id': str(row.get('id', row.get('doc_id', idx))),
                                 'question': cfg['doc_to_text'](row, cfg.get('lmms_eval_specific_kwargs', {}).get('default')),
                                 'image_sizes': [list(im.size) for im in images]})
                for image in images:
                    image.close()
        tasks[task] = {'dataset_path': cfg['dataset_path'], 'split': cfg['test_split'],
                       'dataset_count': dataset_count,
                       'generation_kwargs': cfg['generation_kwargs'], 'metadata': meta,
                       'selected_samples': selected}

    # An asymmetric box on a non-square source catches axis swaps and scale errors.
    xyxy = _zoom_bbox_to_xyxy([100, 200, 700, 800], 'yxyx')
    assert xyxy == [200, 100, 800, 700]
    first = _composite_zoom_to_original({0: (0, [0, 0, 2000, 1000])}, 0, xyxy)
    assert first[1] == [400, 100, 1600, 700]
    nested = _composite_zoom_to_original({0: (0, [0, 0, 2000, 1000]), 1: (0, first[1])}, 1,
                                         _zoom_bbox_to_xyxy([0, 500, 500, 1000], 'yxyx'))
    assert nested[1] == [1000, 100, 1600, 400]
    system, tools = resolved
    initial = [{'role': 'system', 'content': system},
               {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': '\nWhat is in the marked region?\nImage 0 is available as img_idx=0.'}]}]
    tool_args = {'bbox_2d': [100, 200, 500, 600], 'label': 'region of interest', 'img_idx': 0}
    followup = initial + [
        {'role': 'assistant', 'content': '', 'tool_calls': [{'id': 'call_zoom_1', 'type': 'function',
          'function': {'name': 'image_zoom_in_tool', 'arguments': tool_args}}]},
        {'role': 'tool', 'tool_call_id': 'call_zoom_1', 'name': 'image_zoom_in_tool',
         'content': [{'type': 'text', 'text': 'Zoomed in on the image to the region [100, 200, 500, 600] with label region of interest. The cropped view is now available as img_idx 1.'}]},
        {'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': 'Above are the tool-returned image(s).'}]}]
    kwargs = {} if family == 'gemma3' else {'enable_thinking': thinking}
    if thinking:
        from types import SimpleNamespace
        kwargs['preserve_thinking'] = True
        probe_reasoning = 'I should inspect this region before answering.'
        if not no_tools:
            _preserve_assistant_reasoning(SimpleNamespace(reasoning=probe_reasoning), followup[2])
            assert followup[2].get('reasoning') == probe_reasoning, 'Lost API reasoning in assistant tool turn'
    if no_tools:
        assert thinking, 'thinking_only requires native thinking'
        first_task = next(iter(tasks.values()))
        question = (first_task['selected_samples'][0]['question'] if first_task['selected_samples']
                    else 'What is in the marked region?')
        initial = [{'role': 'user', 'content': [{'type': 'image'}, {'type': 'text', 'text': question}]}]
    rendered = {}
    for label, messages in ([('initial', initial)] if no_tools else [('initial', initial), ('after_zoom', followup)]):
        text = tokenizer.apply_chat_template(messages, tools=tools, tokenize=False, add_generation_prompt=True, **kwargs)
        if no_tools:
            assert 'image_zoom_in_tool' not in text and 'img_idx' not in text and '<|tool_call>' not in text
        else:
            assert 'image_zoom_in_tool' in text and '<answer>' in text
        if thinking:
            assert '<|think|>' in text, 'Thinking mode was not activated in the native system turn'
            if label == 'after_zoom':
                assert '<|channel>thought\n' + probe_reasoning in text, 'Lost thought channel before the tool call'
        image_token = '<start_of_image>' if family == 'gemma3' else '<|image|>'
        assert text.count(image_token) == (1 if label == 'initial' else 2), 'Lost input/crop image placeholder'
        if label == 'after_zoom':
            assert 'img_idx 1' in text and 'region of interest' in text, 'Lost tool response'
            assert ('<tool_call>' if family == 'gemma3' else '<|tool_call>call:') in text
        (args.output / f'{label}_rendered.txt').write_text(text)
        rendered[label] = text
    mm_preprocessing = check_gemma3_image_spans(args.checkpoint, rendered) if family == 'gemma3' else None
    (args.output / 'system_prompt.txt').write_text(system or '')
    (args.output / 'tool_schema.json').write_text(json.dumps(tools, indent=2) + '\n')
    (args.output / 'chat_template.jinja').write_text(template)
    manifest = {'model': args.model, 'checkpoint': str(args.checkpoint), 'model_type': config.model_type,
                'transformers': transformers.__version__, 'transformers_source': transformers.__file__,
                'torch': torch.__version__, 'vllm': version('vllm'), 'processor': type(processor).__name__,
                'system_prompt': system, 'system_prompt_sha256': hashlib.sha256((system or '').encode()).hexdigest(),
                'chat_template_source': str(template_path), 'chat_template_sha256': hashlib.sha256(template.encode()).hexdigest(),
                'tools': tools, 'tool_parser': None if no_tools else ('hermes' if family == 'gemma3' else 'gemma4'),
                'thinking': thinking, 'preserve_thinking': thinking,
                'reasoning_parser': 'gemma4' if thinking else None,
                'chat_template_kwargs': kwargs, 'assistant_prefill': '', 'forced_tool_calls': False,
                'forced_final_answer': False, 'max_tool_iterations': int(os.getenv('MAX_TOOL_ITERATIONS', '10')),
                'zoom_bbox_coordinate_order': None if no_tools else 'yxyx', 'coordinate_grid': None if no_tools else [1000, 1000],
                'max_tool_images': int(os.getenv('MAX_TOOL_IMAGES', '10')), 'tool_image_as_user_turn': not no_tools, 'eval_mode': 'thinking_only' if no_tools else 'zoom',
                'mm_processor_kwargs': json.loads(os.getenv('MM_PROCESSOR_KWARGS', '{}')),
                'multimodal_preprocessing_checks': mm_preprocessing,
                'judge': os.getenv('JUDGE_MODEL_PATH'), 'slurm_job_id': os.getenv('SLURM_JOB_ID'), 'tasks': tasks}
    (args.output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(f'CPU preflight PASS: {args.model}; protocol artifacts: {args.output}')


if __name__ == '__main__':
    main()
