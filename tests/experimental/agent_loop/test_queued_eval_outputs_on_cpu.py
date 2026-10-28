"""Resource variants and renamed wrappers share an output identity."""
import importlib.util
import os
from pathlib import Path
import subprocess

SCRIPT = Path(__file__).resolve().parents[3] / 'vero_eval/examples/queued_eval_outputs.py'
spec = importlib.util.spec_from_file_location('queued_outputs', SCRIPT)
queued = importlib.util.module_from_spec(spec)
spec.loader.exec_module(queued)


def test_renamed_wrapper_and_alias_resolve_to_output(tmp_path):
    output = tmp_path / 'global_step_200'
    output.mkdir()
    alias = tmp_path / 'global_step_200__ailab_1gpu'
    alias.symlink_to(output, target_is_directory=True)
    wrapper = tmp_path / 'worker.sh'
    wrapper.write_text('#!/bin/bash\nexport OUTPUT_ROOT=' + str(alias) + '\n')
    identities = set(queued.identities(f'old-resource-name|{tmp_path}|{wrapper}\n'))
    assert ('OUTPUT', str(output)) in identities
    assert ('NAME', 'old-resource-name') in identities
    assert ('OUTPUT', str(tmp_path / 'global_step_400')) not in identities


def test_dynamic_shell_expressions_are_not_evaluated(tmp_path):
    wrapper = tmp_path / 'worker.sh'
    marker = tmp_path / 'must-not-exist'
    wrapper.write_text('export OUTPUT_ROOT=/scratch/$(touch ' + str(marker) + ')\n')
    identities = set(queued.identities(f'name|{tmp_path}|{wrapper}\n'))
    assert identities == {('NAME', 'name'), ('OUTPUT', str(tmp_path))}
    assert not marker.exists()


def test_submitter_skips_renamed_wrapper_for_same_output(tmp_path):
    evaluation = SCRIPT.parents[1]
    train = tmp_path / 'train'
    (train / 'global_step_200/actor/huggingface').mkdir(parents=True)
    output = tmp_path / 'results/global_step_200'
    output.mkdir(parents=True)
    alias = output.with_name(output.name + '__ailab_1gpu')
    alias.symlink_to(output, target_is_directory=True)
    worker = tmp_path / 'worker.sh'
    worker.write_text('export OUTPUT_ROOT=' + str(alias) + '\n')
    binary = tmp_path / 'bin'
    binary.mkdir()
    squeue = binary / 'squeue'
    squeue.write_text('#!/bin/sh\nprintf "%s\\n" "renamed-job|' + str(tmp_path) + '|' + str(worker) + '"\n')
    squeue.chmod(0o700)
    env = {k: v for k, v in os.environ.items() if not k.startswith((
        'ONLY_EVAL_', 'CRITIC_', 'REPLAY_CRITIC_', 'VERO_', 'FORCE_', 'FIRST_CANDIDATE_'))}
    env.update(PATH=str(binary) + ':' + env['PATH'], EVAL_NAME='hardonly_llmjudge_lr5e6',
               TRAIN_DIR=str(train), OUTPUT_ROOT=str(output.parent), TASK_GROUP='visual_probe',
               EVAL_PROFILE='ailab', DRY_RUN='1', SKIP_IF_EXISTS='false', SKIP_IF_QUEUED='true',
               PORT_OFFSET='1')
    result = subprocess.run(['bash', str(SCRIPT.parent / 'submit_zoom_eval_checkpoints_qwen35_27b_judge.sh')],
                            cwd=evaluation, env=env, text=True, capture_output=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'already queued/running' in result.stdout
    assert not any(line.strip().startswith('env ') for line in result.stdout.splitlines())
