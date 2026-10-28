import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[3]
EVAL = REPO / 'vero_eval'
SCRIPT = EVAL / 'examples/submit_zoom_eval_checkpoints_qwen35_27b_judge_visual_probe_hstar_pinpoint.sh'
RUN = 'branchstrata20_fresh_lr1e5_critic_zoom_joint3_datasetbalanced'


def run_submission(tmp_path, **overrides):
    train = tmp_path / 'train'
    (train / 'global_step_100/actor/huggingface').mkdir(parents=True, exist_ok=True)
    judge = tmp_path / 'judge'
    judge.mkdir(exist_ok=True)
    (judge / 'template.jinja').write_text('test')
    env = {k: v for k, v in os.environ.items() if not k.startswith(('CRITIC_', 'VERO_EVAL_CRITIC'))}
    env.update(EVAL_NAME=RUN, TRAIN_DIR=str(train), OUTPUT_ROOT=str(tmp_path/'results'),
               VLLM_JUDGE_MODEL_PATH=str(judge), VLLM_JUDGE_CHAT_TEMPLATE=str(judge/'template.jinja'),
               DRY_RUN='1', SKIP_IF_EXISTS='true', SKIP_IF_QUEUED='false', PORT_OFFSET='1')
    env.update(overrides)
    result = subprocess.run(['bash', str(SCRIPT)], cwd=EVAL, env=env, capture_output=True, text=True)
    return result


def submitted(result):
    assert result.returncode == 0, result.stdout + result.stderr
    line = next(line.strip() for line in result.stdout.splitlines() if line.strip().startswith('env '))
    parts = shlex.split(line)
    stop = parts.index('sbatch')
    env = dict(part.split('=', 1) for part in parts[1:stop])
    job = next(part.split('=', 1)[1] for part in parts[stop:] if part.startswith('--job-name='))
    return env, job


def test_changed_threshold_uses_new_results_and_queue_identity(tmp_path):
    first, job = submitted(run_submission(tmp_path))
    assert json.loads(first['VERO_EVAL_CRITIC_DOMAIN_THRESHOLDS']) == dict(
        visual_probe=.65, hstar=.65, pinpoint=.5, screenspot=.35)
    assert 'critic_vp0p65_hs0p65_pp0p5_ss0p35_fb0p65' in first['OUTPUT_ROOT']
    result_dir = Path(first['OUTPUT_ROOT']) / 'qwen_vllm'
    result_dir.mkdir(parents=True)
    (result_dir/'test_results.json').write_text(json.dumps({'results': {t: {} for t in first['TASKS'].split(',')}}))
    same = run_submission(tmp_path)
    assert same.returncode == 0
    assert 'No checkpoints to submit' in same.stdout
    changed, changed_job = submitted(run_submission(tmp_path, CRITIC_RETRY_THRESHOLD_SCREENSPOT='.4'))
    assert changed['OUTPUT_ROOT'] != first['OUTPUT_ROOT']
    assert changed_job != job
    assert 'ss0p4' in changed['OUTPUT_ROOT']
    equivalent = run_submission(tmp_path, CRITIC_RETRY_THRESHOLD_SCREENSPOT='0.3500')
    assert 'No checkpoints to submit' in equivalent.stdout
    # A queued job with the old settings must not suppress the new settings.
    binary = tmp_path/'bin'
    binary.mkdir()
    queue = binary/'squeue'
    queue.write_text('#!/bin/sh\nprintf "%s\\n" ' + shlex.quote(job) + '\n')
    queue.chmod(0o755)
    path = str(binary) + ':' + os.environ['PATH']
    queued_same = run_submission(tmp_path, PATH=path, SKIP_IF_EXISTS='false', SKIP_IF_QUEUED='true')
    assert 'No checkpoints to submit' in queued_same.stdout
    submitted(run_submission(tmp_path, PATH=path, SKIP_IF_EXISTS='false', SKIP_IF_QUEUED='true',
                             CRITIC_RETRY_THRESHOLD_SCREENSPOT='.4'))


def test_all_critic_runs_get_domains_but_no_critic_runs_do_not(tmp_path):
    critic, job = submitted(run_submission(tmp_path, EVAL_NAME='branchstrata20_fresh_lr1e5_critic_zoom_joint3'))
    assert json.loads(critic['VERO_EVAL_CRITIC_DOMAIN_THRESHOLDS'])['screenspot'] == .35
    other, other_job = submitted(run_submission(tmp_path))
    assert job != other_job
    no_critic, _ = submitted(run_submission(tmp_path, EVAL_NAME='hardonly_llmjudge_lr5e6'))
    assert no_critic['VERO_EVAL_CRITIC_DOMAIN_THRESHOLDS'] == ''
    assert Path(no_critic['OUTPUT_ROOT']).parent.name == 'results'


@pytest.mark.parametrize('value', ['nan', 'inf', '-0.1', '1.1', 'bad'])
def test_invalid_domain_threshold_fails_before_submission(tmp_path, value):
    result = run_submission(tmp_path, CRITIC_RETRY_THRESHOLD_SCREENSPOT=value)
    assert result.returncode != 0
    assert 'Submitted 1 job' not in result.stdout
