#!/usr/bin/env python3
"""Snapshot only Transformers into an isolated overlay; keep the CUDA eval stack.

The current vero2_eval has vLLM 0.23 but Transformers 5.3 (no Gemma 4).
An installed, pure-Python Transformers 5.8 snapshot supplies Gemma 4 support.
No packages in either Conda environment are changed, and no CPU torch is copied.
"""
import argparse
import json
import shutil
from pathlib import Path

DEFAULT_OVERLAY = '/scratch/gpfs/ZHUANGL/gs1693/vero2/runtime/gemma_zoom_transformers'
DEFAULT_SOURCE = '/scratch/gpfs/ZHUANGL/gs1693/conda/reason-vlm-gemma4/lib/python3.10/site-packages'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path(DEFAULT_OVERLAY))
    parser.add_argument('--source', type=Path, default=Path(DEFAULT_SOURCE))
    args = parser.parse_args()
    manifest = args.output / 'runtime_source.json'
    if manifest.exists():
        print(manifest.read_text().strip())
        return
    config = args.source / 'transformers/models/gemma4/configuration_gemma4.py'
    if not config.is_file():
        raise SystemExit(f'Missing Gemma 4 Transformers source: {config}')
    args.output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(args.source / 'transformers', args.output / 'transformers',
                    dirs_exist_ok=True, ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    metadata = list(args.source.glob('transformers-*.dist-info'))
    if len(metadata) != 1:
        raise SystemExit(f'Expected one Transformers distribution, found {metadata}')
    shutil.copytree(metadata[0], args.output / metadata[0].name, dirs_exist_ok=True)
    record = {'source': str(args.source), 'overlay': str(args.output), 'distribution': metadata[0].name,
              'purpose': 'Gemma 4 Transformers support over the CUDA vero2_eval environment; torch/vLLM unchanged'}
    manifest.write_text(json.dumps(record, indent=2) + '\n')
    print(manifest.read_text().strip())


if __name__ == '__main__':
    main()
