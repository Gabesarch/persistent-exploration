"""Submit full Visual Probe, Pinpoint and V* for three Qwen3.5-9B conditions."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess

from submit_zoom_eval_selected200_generalization import complete, digest, verify_weights, verify_datasets
from submit_zoom_eval_gemma4_generalization import clean_environment

EVAL = Path(__file__).resolve().parents[1]
ROOT = Path('/scratch/gpfs/ZHUANGL/gs1693')
BASE = ROOT / 'hf_cache/Qwen/Qwen3.5-9B'
VERO = ROOT / 'checkpoints/verl/FINAL_CKPTS/Vero-Qwen35-9B'
JUDGE = ROOT / 'hf_cache/Qwen/Qwen3.5-27B'
COUNTS = {'visual_probe_easy': 141, 'visual_probe_medium': 268, 'visual_probe_hard': 106,
          'pinpoint_bench': 433, 'vstar_bench': 191}
CONDITIONS = ('qwen35_9b_zoom', 'qwen35_9b_no_tools', 'vero_qwen35_9b_no_tools')


def inputs():
    counts, evidence = verify_datasets({'root_path': str(ROOT), 'benchmarks': [
        {'name': 'pinpoint_bench', 'count': 433}, {'name': 'vstar_bench', 'count': 191}]})
    for base in list(COUNTS)[:3]:
        paths = sorted((ROOT / 'hf_cache/datasets').glob(f'Mini-o3___{base}/*/*/*/dataset_info.json'))
        found = [(p, json.loads(p.read_text())['splits']['validation']['num_examples']) for p in paths]
        if not found or {n for _, n in found} != {COUNTS[base]}:
            raise ValueError(f'Missing or changed dataset: {base}: {found}')
        counts[base], evidence[base] = found[0][1], str(found[0][0])
    for path in (BASE, VERO, JUDGE):
        verify_weights(path)
        if json.loads((path / 'config.json').read_text())['model_type'] != 'qwen3_5':
            raise ValueError(f'Unexpected model architecture: {path}')
        if not (path / 'chat_template.jinja').is_file():
            raise ValueError(f'Missing chat template: {path}')
    if not (JUDGE / 'chat_template_no_think.jinja').is_file():
        raise ValueError('Missing judge no-thinking template')
    return counts, evidence


def make_job(condition, output, walltime):
    zoom = condition.endswith('_zoom')
    model = VERO if condition.startswith('vero') else BASE
    tasks = {f'{base}_{condition}': count for base, count in COUNTS.items()}
    env = {
        'ROOT_PATH': str(ROOT), 'VERO_EVAL_DIR': str(EVAL), 'MODEL': 'vero',
        'CHECKPOINT_PATH': str(model), 'CHAT_TEMPLATE': str(model / 'chat_template.jinja'),
        'LMMS_EVAL_HF_TOKENIZER': str(model), 'LMMS_EVAL_HF_TOKENIZER_LOCAL_ONLY': '1',
        'TOOLS': 'on' if zoom else 'off', 'MCP_SERVER_PATH': 'lmms_eval/mcp/vero2_mcp.py' if zoom else '',
        'TASKS': ','.join(tasks), 'SUFFIX': '', 'LIMIT': '',
        'OUTPUT_ROOT': str(Path(output).resolve() / condition),
        'MAX_TOOL_ITERATIONS': '10' if zoom else '0', 'MAX_TOOL_IMAGES': '10' if zoom else '0',
        'MM_MAX_PIXELS': '10000000', 'MM_MIN_PIXELS': '', 'MM_PROCESSOR_KWARGS': '{"max_pixels":10000000}',
        'MAX_MODEL_LEN': '49152', 'VLLM_MAX_NUM_SEQS': '4', 'NUM_GPUS': '1', 'WORKERS': '4', 'BATCH_SIZE': '1',
        'LIMIT_MM_PER_PROMPT': json.dumps({'image': 11 if zoom else 1, 'video': 0}),
        'TOOL_CALL_PARSER': 'qwen3_coder', 'REASONING_PARSER': 'qwen3',
        'DEFAULT_CHAT_TEMPLATE_KWARGS': '{"enable_thinking":true}',
        'LMMS_EVAL_PRESERVE_REASONING': 'true',
        'VLLM_ADDITIONAL_CONFIG': '{"gdn_prefill_backend":"triton"}',
        'RUN_SYSTEM_PROMPT_MODE': 'task_metadata', 'SYSTEM_PROMPT': '', 'TRAINING_SYSTEM_PROMPT': '',
        'VERO_TRAINING_CONFIG': '', 'FORCE_TOOL_CALLS_UNTIL_FINAL_TURN': 'false',
        'FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS': 'false', 'FORCE_FINAL_ANSWER_PREFIX': '',
        'FORCE_FINAL_ANSWER_THINK_PREFILL': '', 'CRITIC_GUIDED_ANSWER_RETRY': 'false',
        'CRITIC_GUIDED_ANSWER_RETRY_SOURCE': 'policy', 'FIRST_CANDIDATE_ANSWER_TURN': '0',
        'FOLLOWUP_CANDIDATE_ANSWER_TURN': '0', 'ALWAYS_RETRY_ANSWERS': 'false', 'RECORD_ANSWER_CANDIDATES': 'false',
        'JUDGE_BACKEND': 'engine', 'API_TYPE': 'vllm', 'JUDGE_MODEL_PATH': str(JUDGE),
        'VLLM_JUDGE_MODEL_PATH': str(JUDGE), 'VLLM_JUDGE_CHAT_TEMPLATE': str(JUDGE / 'chat_template_no_think.jinja'),
        'VLLM_JUDGE_ENFORCE_EAGER': 'true', 'VLLM_JUDGE_GDN_PREFILL_BACKEND': 'triton',
        'CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE': '1', 'CHARXIV_JUDGE_DTYPE': 'bfloat16',
        'CHARXIV_JUDGE_MAX_MODEL_LEN': '18432', 'VLLM_JUDGE_MAX_NUM_SEQS': '32',
        'CHARXIV_JUDGE_MAX_TOKENS': '2048', 'CHARXIV_JUDGE_TEMPERATURE': '0',
        'CHARXIV_JUDGE_TOP_P': '0.8', 'CHARXIV_JUDGE_TOP_K': '20', 'CHARXIV_JUDGE_PRESENCE_PENALTY': '0',
        'VLLM_USE_FLASHINFER_SAMPLER': '0', 'LMMS_EVAL_USE_CACHE': 'False',
        'LMMS_EVAL_SAVE_PREJUDGE_SAMPLES': 'true', 'OVERRIDE_EXISTING': 'false',
    }
    paths = [Path(__file__), EVAL / 'scripts/run_qwen35_9b_generalization.sbatch',
             EVAL / 'run_eval.sbatch', EVAL / 'scripts/serve_qwen_vllm.sh',
             EVAL / 'examples/eval_domain.sh', EVAL / 'lmms_eval/models/chat/async_openai.py',
             model / 'chat_template.jinja', JUDGE / 'chat_template_no_think.jinja']
    paths += [EVAL / f'lmms_eval/tasks/qwen35_9b_generalization/{task}.yaml' for task in tasks]
    for base in COUNTS:
        family = 'visual_probe' if base.startswith('visual_probe') else base
        paths += [EVAL / f'lmms_eval/tasks/{family}/{base}_reasoning.yaml',
                  EVAL / f'lmms_eval/tasks/{family}/_default_template_yaml',
                  EVAL / f'lmms_eval/tasks/{family}/utils.py']
    if zoom:
        paths += [EVAL / 'lmms_eval/tasks/_tool_templates/_toolset_zoom_yaml',
                  EVAL / 'prompts/tools/system_base.txt', EVAL / 'prompts/tools/system_zoom.txt']
    hashes = {str(p): digest(p) for p in paths}
    protocol = {'env': env, 'hashes': hashes, 'sample_counts': tasks}
    identity = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()[:12]
    return dict(condition=condition, env=env, sample_counts=tasks, tasks=list(tasks), samples=sum(tasks.values()),
                job_name=f'q35-eval-{condition}-{identity}', walltime=walltime, protocol=protocol)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--conditions', default=os.getenv('EVAL_NAME', 'all'))
    p.add_argument('--output-root', default=os.getenv('OUTPUT_ROOT', str(ROOT / 'lmms_eval_vero2_qwen35_27b_judge/qwen35_9b_visual_probe_pinpoint_vstar_v1')))
    p.add_argument('--time', default=os.getenv('TIME_OVERRIDE', '24:00:00'))
    p.add_argument('--preflight', action='store_true')
    p.add_argument('--dry-run', action='store_true', default=os.getenv('DRY_RUN') == '1')
    p.add_argument('--manifest', type=Path)
    args = p.parse_args()
    selected = CONDITIONS if args.conditions == 'all' else args.conditions.split(',')
    if not selected or len(set(selected)) != len(selected) or set(selected) - set(CONDITIONS):
        p.error(f'--conditions must select from {CONDITIONS}')
    try:
        h, m, s = map(int, args.time.split(':'))
        assert 0 <= h <= 72 and 0 <= m < 60 and 0 <= s < 60 and 0 < h*3600+m*60+s <= 72*3600
    except (ValueError, AssertionError):
        p.error('--time must be positive HH:MM:SS, at most 72 hours')
    counts, evidence = inputs()
    jobs = [make_job(c, args.output_root, args.time) for c in selected]
    plan = {'dataset_counts': counts, 'dataset_evidence': evidence, 'jobs': jobs}
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(plan, indent=2) + '\n')
    print(f'Validated {len(jobs)} conditions, {sum(counts.values())} samples each: {counts}', flush=True)
    if args.preflight:
        return
    result = subprocess.run(['squeue', '-u', os.environ['USER'], '-h', '-r', '-o', '%j', '-t', 'PD,R'],
                            check=True, capture_output=True, text=True, timeout=30)
    queued = set(result.stdout.splitlines())
    for job in jobs:
        if complete(job) or job['job_name'] in queued:
            print(f"SKIP complete or queued: {job['condition']}", flush=True)
            continue
        out = Path(job['env']['OUTPUT_ROOT'])
        cmd = ['sbatch', '--parsable', '--partition=pli-c', '--account=pli', '--constraint=h100',
               '--nodes=1', '--ntasks=1', '--gres=gpu:1', '--cpus-per-task=8', '--mem=180G',
               f'--time={args.time}', f"--job-name={job['job_name']}", f'--chdir={EVAL}',
               f'--output={out}/slurm_%j.out', f'--error={out}/slurm_%j.err', '--export=ALL',
               str(EVAL / 'scripts/run_qwen35_9b_generalization.sbatch')]
        print(shlex.join(cmd), flush=True)
        if args.dry_run:
            continue
        out.mkdir(parents=True, exist_ok=True)
        (out / 'submission_plan.json').write_text(json.dumps(job, indent=2) + '\n')
        env = clean_environment()
        for key in ('GENERATION_KWARGS', 'TOOL_IMAGE_AS_USER_TURN', 'MM_PROCESSOR_KWARGS', 'LMMS_EVAL_HF_TOKENIZER'):
            env.pop(key, None)
        env.update(job['env'])
        result = subprocess.run(cmd, cwd=EVAL, env=env, check=True, capture_output=True, text=True)
        job_id = result.stdout.strip().split(';')[0]
        (out / 'job_id.txt').write_text(job_id + '\n')
        (out / 'submission.json').write_text(json.dumps({'job_id': job_id, **job}, indent=2) + '\n')
        print(f"Submitted {job_id}: {job['condition']}", flush=True)


if __name__ == '__main__':
    main()
