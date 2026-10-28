"""Submit both Gemma 4 modes using the selected200 dataset and scoring suite.

Pinpoint and VStar run by default; HRBench-8K is opt-in.
The shared Gemma runtime supplies native model templates, yxyx coordinates,
optional reasoning replay, exact request traces and a sequential Qwen3.5 judge.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from submit_zoom_eval_selected200_generalization import UNSET, complete, digest, verify_datasets, verify_weights

EVAL = Path(__file__).resolve().parents[1]
REFERENCE_CONFIG = EVAL / 'examples/configs/selected200_generalization.json'
DEFAULT_OUTPUT = '/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_gemma_zoom/generalization_v1'
BENCHMARKS = ('pinpoint_bench', 'vstar_bench', 'hrbench_8k')


def make_job(config, mode, output_root, walltime):
    no_tools = mode == 'thinking_only'
    thinking = mode in ('thinking', 'thinking_only')
    variant = 'gemma4_e4b_thinking_only' if no_tools else ('gemma4_e4b_thinking' if thinking else 'gemma4_e4b')
    out = Path(output_root).resolve() / variant
    model = str(Path(config['root_path']) / 'hf_cache/google/gemma-4-E4B-it')
    suffix = 'thinking_only' if no_tools else 'zoom'
    counts = {f"{b['name']}_gemma4_{suffix}": b['count'] for b in config['benchmarks']}
    env = {
        'ROOT_PATH': config['root_path'], 'VERO_EVAL_DIR': str(EVAL),
        'GEMMA_FAMILY': 'gemma4_e4b', 'GEMMA4_ENABLE_THINKING': str(thinking).lower(),
        'GEMMA_EVAL_MODE': 'thinking_only' if no_tools else 'zoom',
        'CHECKPOINT_PATH': model, 'OUTPUT_ROOT': str(out),
        'TASKS': ','.join(counts), 'LIMIT': '', 'GEMMA_EXPECTED_SAMPLE_COUNTS': json.dumps(counts),
        'GEMMA_TRANSFORMERS_OVERLAY': config['root_path'] + '/vero2/runtime/gemma_zoom_transformers',
        'WORKERS': '2', 'MAX_TOOL_ITERATIONS': '0' if no_tools else '10', 'MAX_TOOL_IMAGES': '0' if no_tools else '10',
        'MAX_MODEL_LEN': '49152', 'VLLM_JUDGE_MODEL_PATH': config['judge'],
        'CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE': '1', 'CHARXIV_JUDGE_DTYPE': 'bfloat16',
        'CHARXIV_JUDGE_MAX_MODEL_LEN': '18432', 'VLLM_JUDGE_MAX_NUM_SEQS': '32',
        'CHARXIV_JUDGE_MAX_TOKENS': '2048', 'CHARXIV_JUDGE_TEMPERATURE': '0',
        'CHARXIV_JUDGE_TOP_P': '0.8', 'CHARXIV_JUDGE_TOP_K': '20',
        'CHARXIV_JUDGE_PRESENCE_PENALTY': '0', 'CHARXIV_JUDGE_GPU_MEMORY_UTILIZATION': '0.9',
        'VLLM_JUDGE_ENFORCE_EAGER': 'true', 'VLLM_JUDGE_GDN_PREFILL_BACKEND': 'triton',
        'LMMS_EVAL_USE_CACHE': 'False',
    }
    # Source hashes make submissions inspectable and detect a changed protocol
    # before reusing a fixed output directory. The runtime snapshots resolved YAML.
    paths = [REFERENCE_CONFIG, EVAL / 'prompts/tools/system_gemma_zoom.txt',
             Path(model) / 'chat_template.jinja', EVAL / 'scripts/run_gemma_zoom_eval.sbatch',
             EVAL / 'scripts/preflight_gemma_zoom.py',
             EVAL / 'scripts/serve_qwen_vllm.sh', EVAL / 'examples/eval_domain.sh',
             EVAL / 'lmms_eval/models/chat/async_openai.py',
             EVAL / 'lmms_eval/tasks/_tool_templates/_toolset_zoom_gemma_yaml']
    for benchmark in (b['name'] for b in config['benchmarks']):
        paths += [EVAL / f'lmms_eval/tasks/gemma4_generalization/{benchmark}_gemma4_{suffix}.yaml',
                  EVAL / f'lmms_eval/tasks/selected200_generalization/{benchmark}_selected200_zoom.yaml']
        if no_tools:
            paths += [EVAL / f'lmms_eval/tasks/selected200_generalization/{benchmark}_selected200_qwen_zs.yaml']
    if no_tools:
        paths.remove(EVAL / 'prompts/tools/system_gemma_zoom.txt')
        paths.remove(EVAL / 'lmms_eval/tasks/_tool_templates/_toolset_zoom_gemma_yaml')
    hashes = {str(path): digest(path) for path in paths}
    identity = {'mode': mode, 'env': env, 'hashes': hashes,
                'image_processor': {'max_soft_tokens': 1120},
                'coordinate_order': None if no_tools else 'yxyx', 'coordinate_grid': None if no_tools else [1000, 1000],
                'tools': not no_tools, 'system_prompt': None if no_tools else 'prompts/tools/system_gemma_zoom.txt',
                'chat_template_kwargs': {'enable_thinking': thinking, 'preserve_thinking': thinking}}
    protocol_hash = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    return {'condition': variant, 'mode': mode, 'tasks': list(counts), 'sample_counts': counts,
            'samples': sum(counts.values()), 'job_name': f'gemma4-generalization-{mode}-{protocol_hash}',
            'walltime': walltime, 'env': env, 'protocol': identity,
            'source_reference': str(REFERENCE_CONFIG),
            'system_prompt_sha256': None if no_tools else hashlib.sha256((EVAL / 'prompts/tools/system_gemma_zoom.txt').read_text().strip().encode()).hexdigest()}


def clean_environment():
    unset = set(UNSET) | {
        'TASKS', 'LIMIT', 'LIMIT_PER_TASK', 'GEMMA_MODELS', 'GEMMA4_ENABLE_THINKING', 'GEMMA_EVAL_MODE',
        'GEMMA_EXPECTED_SAMPLE_COUNTS', 'GEMMA_TRANSFORMERS_OVERLAY', 'MODEL', 'TOOLS',
        'MCP_SERVER_PATH', 'TOOL_CALL_PARSER', 'CHAT_TEMPLATE', 'MM_PROCESSOR_KWARGS',
        'DEFAULT_CHAT_TEMPLATE_KWARGS', 'REASONING_PARSER', 'OUTPUT_ROOT', 'OUTPUT_BASE',
        'MAX_TOOL_ITERATIONS', 'MAX_TOOL_IMAGES', 'MAX_MODEL_LEN', 'WORKERS',
        'LMMS_EVAL_REQUEST_TRACE_DIR', 'LMMS_EVAL_PRESERVE_REASONING',
        'HF_HUB_OFFLINE', 'HF_DATASETS_OFFLINE', 'TRANSFORMERS_OFFLINE',
    }
    unset |= {key for key in os.environ if key.startswith(('SBATCH_', 'CHARXIV_JUDGE_', 'VLLM_JUDGE_', 'VERO_EVAL_'))}
    return {key: value for key, value in os.environ.items() if key not in unset}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--include-hrbench-8k', action='store_true', help='include HRBench-8K (excluded by default)')
    parser.add_argument('--dry-run', action='store_true', default=os.getenv('DRY_RUN') == '1')
    parser.add_argument('--modes', default='nonthinking,thinking')
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT)
    parser.add_argument('--time', default='16:00:00')
    parser.add_argument('--manifest', type=Path)
    args = parser.parse_args()
    modes = args.modes.split(',')
    if len(set(modes)) != len(modes) or set(modes) - {'nonthinking', 'thinking', 'thinking_only'}:
        parser.error('--modes must select nonthinking, thinking, thinking_only, or a comma-separated combination')
    try:
        h, m, s = map(int, args.time.split(':'))
        assert h >= 0 and 0 <= m < 60 and 0 <= s < 60 and 0 < h*3600+m*60+s <= 72*3600
    except (ValueError, AssertionError):
        parser.error('--time must be HH:MM:SS and at most 72 hours')
    config = json.loads(REFERENCE_CONFIG.read_text())
    if tuple(b['name'] for b in config['benchmarks']) != BENCHMARKS:
        raise ValueError('Reference benchmark scope changed; review the Gemma suite before submitting')
    if not args.include_hrbench_8k:
        config['benchmarks'] = [b for b in config['benchmarks'] if b['name'] != 'hrbench_8k']
    counts, evidence = verify_datasets(config)
    verify_weights(Path(config['root_path']) / 'hf_cache/google/gemma-4-E4B-it')
    verify_weights(config['judge'])
    overlay = Path(config['root_path']) / 'vero2/runtime/gemma_zoom_transformers/runtime_source.json'
    if not overlay.is_file():
        raise ValueError(f'Prepare the Gemma Transformers overlay first: {overlay}')
    jobs = [make_job(config, mode, args.output_root, args.time) for mode in modes]
    plan = {'reference_config': str(REFERENCE_CONFIG), 'dataset_counts': counts,
            'dataset_evidence': evidence, 'jobs': jobs}
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Validated {len(jobs)} modes; {sum(counts.values())} samples/mode: {counts}', flush=True)
    if args.preflight:
        print('Input preflight passed; no jobs submitted.')
        return
    try:
        result = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-r', '-o', '%j', '-t', 'PD,R'],
                                check=True, capture_output=True, text=True, timeout=30)
        queued = set(result.stdout.splitlines())
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        if not args.dry_run:
            raise RuntimeError('Cannot verify queued duplicates; refusing to submit') from exc
        queued = set()
        print('Queue unavailable in dry-run; duplicate check deferred.')
    for job in jobs:
        if complete(job):
            print(f"SKIP complete: {job['condition']}")
            continue
        if job['job_name'] in queued:
            print(f"SKIP queued/running: {job['condition']}")
            continue
        out = Path(job['env']['OUTPUT_ROOT'])
        cmd = ['sbatch', '--parsable', '--partition=pli-c', '--account=pli', '--constraint=h100',
               '--nodes=1', '--ntasks=1', '--gres=gpu:1', '--cpus-per-task=8', '--mem=180G',
               f"--time={job['walltime']}", f"--job-name={job['job_name']}",
               f'--chdir={EVAL}', f'--output={out}/slurm_%j.out', f'--error={out}/slurm_%j.err',
               '--export=ALL', str(EVAL / 'scripts/run_gemma_zoom_eval.sbatch')]
        print(shlex.join(['env'] + [f'{key}={value}' for key, value in job['env'].items()] + cmd), flush=True)
        if args.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / 'submission_plan.json').write_text(json.dumps(job, indent=2) + '\n')
        env = clean_environment()
        env.update(job['env'])
        result = subprocess.run(cmd, cwd=EVAL, env=env, check=True, text=True, capture_output=True)
        job_id = result.stdout.strip().split(';')[0]
        (out / 'job_id.txt').write_text(job_id + '\n')
        (out / 'submission.json').write_text(json.dumps({'job_id': job_id, **job}, indent=2) + '\n')
        print(f"Submitted {job_id}: {job['condition']}", flush=True)
    print('Dry-run complete; no jobs submitted.' if args.dry_run else 'Submission pass complete.')


if __name__ == '__main__':
    main()
