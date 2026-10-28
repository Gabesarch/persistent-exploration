#!/usr/bin/env python3
"""Register the two trained Gemma4 conditions; select complete steps 100, 200, ..."""

import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from lmms_eval.models.model_utils.gemma4_training_alignment import Gemma4TrainingAlignment

BASE = Path('/scratch/gpfs/ZHUANGL/gs1693/vero2/gemma4_training')
RUNS = {
    '15809082': ('branchstrata20_fresh_lr1e5_critic_gemma4_e4b', '15884866'),
    '15884796': ('gdpo_fresh_lr1e5_noexplore_accformat_gemma4_e4b', '15910097'),
}
# A replacement continuation array resumes the same critic output/protocol.
JOB_ALIASES = {'16193714': '15809082'}
COUNTS = {f'visual_probe_{level}_gemma4_zoom': n for level, n in [('easy', 141), ('medium', 268), ('hard', 106)]}
OUTPUT = Path('/scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2_gemma_zoom/training_v1')


def selected_steps(root, requested_steps=None):
    if requested_steps is not None:
        return [root / f'global_step_{step}' for step in sorted(set(requested_steps))]
    steps = [p for p in root.glob('global_step_*') if re.fullmatch(r'global_step_\d+', p.name)]
    return [p for p in sorted(steps, key=lambda p: int(p.name.rsplit('_', 1)[1]))
            if int(p.name.rsplit('_', 1)[1]) > 0 and int(p.name.rsplit('_', 1)[1]) % 100 == 0]


def complete_checkpoint(step):
    tracker = step.parent / 'latest_checkpointed_iteration.txt'
    if not tracker.is_file() or int(tracker.read_text().strip()) < int(step.name.rsplit('_', 1)[1]):
        return False
    actor = step / 'actor'
    files = list(actor.glob('model_world_size_*_rank_*.pt'))
    matches = [re.fullmatch(r'model_world_size_(\d+)_rank_(\d+)\.pt', p.name) for p in files]
    if not matches:
        return False
    world = int(matches[0][1])
    return (len(files) == world and {int(m[1]) for m in matches} == {world}
            and {int(m[2]) for m in matches} == set(range(world))
            and all(p.stat().st_size > 0 for p in files)
            and all((actor / 'huggingface' / f).is_file() for f in
                    ('config.json', 'tokenizer_config.json', 'tokenizer.json', 'processor_config.json', 'chat_template.jinja')))


def completed(output):
    for result in output.rglob('*_results.json') if output.exists() else []:
        try:
            data = json.loads(result.read_text())
            if not all(t in data.get('results', {}) and data.get('n-samples', {}).get(t, {}).get('effective') == n
                       for t, n in COUNTS.items()):
                continue
            for task, count in COUNTS.items():
                samples = list(result.parent.glob(f'*samples_{task}.jsonl'))
                if len(samples) != 1:
                    break
                rows = [json.loads(line) for line in samples[0].read_text().splitlines() if line.strip()]
                if len(rows) != count or len({row['doc_id'] for row in rows}) != count:
                    break
            else:
                return True
        except (ValueError, KeyError, OSError):
            continue
    return False


def queued(output):
    result = subprocess.run(['squeue', '-u', 'gs1693', '-h', '-r', '-o', '%Z', '-t', 'PD,R,CF,CG,S'],
                            check=True, text=True, capture_output=True, timeout=45)
    return str(output.resolve()) in {str(Path(line.strip()).resolve()) for line in result.stdout.splitlines() if line.strip()}


def plan(job, requested_steps=None):
    name, recorded_job = RUNS[job]
    config = BASE / 'manifests' / name / recorded_job / 'resolved.yaml'
    # Both runs have saved validation proof even before step 100 appears.
    profile = Gemma4TrainingAlignment(config)
    root = Path(profile.config['trainer']['default_local_dir'])
    entries = []
    for step in selected_steps(root, requested_steps):
        if not complete_checkpoint(step):
            if requested_steps is not None:
                raise ValueError(f'Requested checkpoint is missing or incomplete: {step}')
            print(f'[INFO] {name}/{step.name}: incomplete model shards; skipped', flush=True)
            continue
        checkpoint_profile = Gemma4TrainingAlignment(config, step / 'actor/huggingface')
        output = OUTPUT / name / step.name
        export = BASE / 'eval_exports' / name / step.name
        identity = {**checkpoint_profile.identity(), 'checkpoint': str(step), 'export': str(export),
                    'tasks': COUNTS, 'interval': 100}
        # Validation evidence can grow while training continues; it is audit
        # evidence, not a different protocol or checkpoint identity.
        identity.pop('validation_evidence')
        entries.append((checkpoint_profile, step, output, export, identity))
    return profile, entries


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', nargs='+', choices=list(RUNS) + list(JOB_ALIASES), default=list(RUNS))
    parser.add_argument('--steps', nargs='+', type=int,
                        help='Evaluate only these positive checkpoint steps; default: every 100 steps')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--preflight', action='store_true', help='validate saved training identities and shards without Slurm or writes')
    args = parser.parse_args()
    if args.steps is not None and any(step <= 0 for step in args.steps):
        parser.error('--steps must contain positive integers')
    args.dry_run = args.dry_run or os.getenv('DRY_RUN') == '1'
    for job in dict.fromkeys(JOB_ALIASES.get(job, job) for job in args.jobs):
        profile, entries = plan(job, args.steps)
        print(json.dumps({'job': job, 'training_root': profile.config['trainer']['default_local_dir'],
                          'critic': profile.eval_critic_enabled,
                          'every_steps': 100 if args.steps is None else None,
                          'requested_steps': args.steps,
                          'available': [step.name for _, step, *_ in entries],
                          'validation_evidence': profile.validation_evidence}), flush=True)
        for checkpoint_profile, step, output, export, identity in entries:
            settings = output / 'submitted_config.json'
            if output.exists() and not settings.exists() and any(output.rglob('*_results.json')):
                raise ValueError(f'Existing Gemma4 results have no saved submission identity: {output}')
            if settings.exists() and json.loads(settings.read_text()) != identity:
                raise ValueError(f'Existing Gemma4 submitted identity differs: {settings}')
            if completed(output):
                print(f'[INFO] Skipping {output}: complete 515-example results', flush=True)
                continue
            if args.preflight:
                continue
            if queued(output):
                print(f'[INFO] Skipping {output}: already queued/running', flush=True)
                continue
            options = ['sbatch', '--parsable', f'--job-name=gemma4-eval-{job}-{step.name}',
                       '--partition=ailab', '--account=zhuangl', '--constraint=h200', '--time=24:00:00',
                       '--nodes=1', '--ntasks=1', '--gres=gpu:1', '--cpus-per-task=8', '--mem=240G',
                       f'--chdir={output}', f'--output={output}/slurm_%j.out', f'--error={output}/slurm_%j.err',
                       '--export=ALL', str(ROOT / 'scripts/run_gemma4_training_eval.sbatch')]
            env = os.environ.copy()
            # These jobs have fixed training identities; inherited experiment,
            # task, prompt and debug selectors must not alter them.
            for key in list(env):
                if key.startswith(('VERO_EVAL_', 'CRITIC_GUIDED_', 'REPLAY_CRITIC_', 'FORCE_', 'GEMMA_', 'GEMMA4_')):
                    env.pop(key)
            env.update(VERO_EVAL_DIR=str(ROOT), GEMMA_FAMILY='gemma4_e4b', GEMMA4_ENABLE_THINKING='true',
                       GEMMA_TRAINING_CONFIG=str(checkpoint_profile.path),
                       GEMMA_TRAINING_CHECKPOINT=str(step / 'actor/huggingface'),
                       GEMMA_RL_ACTOR_DIR=str(step / 'actor'), CHECKPOINT_PATH=str(export),
                       OUTPUT_ROOT=str(output), TASKS=','.join(COUNTS), LIMIT='',
                       GEMMA_EXPECTED_SAMPLE_COUNTS=json.dumps(COUNTS),
                       LMMS_EVAL_USE_CACHE='False', OVERRIDE_EXISTING='true')
            if args.dry_run:
                print(json.dumps({'action': 'would_submit', 'identity': identity, 'command': options}), flush=True)
                continue
            output.mkdir(parents=True, exist_ok=True)
            settings.write_text(json.dumps(identity, indent=2) + '\n')
            result = subprocess.run(options, env=env, check=True, text=True, capture_output=True, timeout=60)
            job_id = result.stdout.strip().split(';')[0]
            (output / 'job_id.txt').write_text(job_id + '\n')
            print(f'Submitted batch job {job_id} ({job}/{step.name})', flush=True)


if __name__ == '__main__':
    main()
