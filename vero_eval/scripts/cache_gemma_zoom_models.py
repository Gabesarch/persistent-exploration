#!/usr/bin/env python3
"""Cache and validate Gemma zoom policy checkpoints without changing their templates."""
import argparse
import json
from pathlib import Path

MODELS = {'gemma3_4b': 'google/gemma-3-4b-it', 'gemma4_e4b': 'google/gemma-4-E4B-it'}
DEFAULT_CACHE = '/scratch/gpfs/ZHUANGL/gs1693/hf_cache'


def validate(path):
    config = json.loads((path / 'config.json').read_text())
    index = path / 'model.safetensors.index.json'
    weights = set(json.loads(index.read_text())['weight_map'].values()) if index.exists() else {'model.safetensors'}
    for filename in weights:
        if not (path / filename).is_file() or (path / filename).stat().st_size == 0:
            raise ValueError(f'Missing/empty checkpoint shard: {path / filename}')
    if not (path / 'tokenizer_config.json').is_file():
        raise ValueError(f'Missing tokenizer config: {path}')
    if not any((path / f).exists() for f in ['processor_config.json', 'preprocessor_config.json']):
        raise ValueError(f'Missing multimodal processor: {path}')
    print(json.dumps({'path': str(path), 'model_type': config['model_type'], 'shards': len(weights)}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', choices=['both', *MODELS], default='both')
    parser.add_argument('--cache-root', type=Path, default=Path(DEFAULT_CACHE))
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--revision', default='main')
    args = parser.parse_args()
    for key, repo in MODELS.items():
        if args.model not in ('both', key):
            continue
        local = args.cache_root / repo
        if not args.verify_only:
            from huggingface_hub import snapshot_download
            # Explicitly excludes duplicate alternate weights, not model/processor assets.
            snapshot_download(repo_id=repo, revision=args.revision, local_dir=str(local),
                              ignore_patterns=['*.bin', '*.msgpack', '*.h5', '*.tflite', '*.gguf', 'original/*'])
        validate(local)


if __name__ == '__main__':
    main()
