"""Gemma plotting must reject smokes and recover tool traces without false zeros."""
import importlib.util
import json
import os
from pathlib import Path
from types import SimpleNamespace
import sys
import subprocess

import pytest

REPO = Path(__file__).resolve().parents[3]
EXAMPLES = REPO / 'vero_eval/examples'
sys.path.insert(0, str(EXAMPLES))
spec = importlib.util.spec_from_file_location('gemma_plotter', EXAMPLES / 'plot_eval_accuracy_over_steps_gemma4_e4b.py')
plotter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(plotter)


def payload(*, limit=None, accuracy=.5):
    return {
        'model_name': 'gemma4_e4b', 'config': {'limit': limit},
        'results': {f'visual_probe_{d}_gemma4_zoom': {'judge_accuracy,none': accuracy}
                    for d in plotter.COUNTS},
        'n-samples': {f'visual_probe_{d}_gemma4_zoom': {'effective': n}
                      for d, n in plotter.COUNTS.items()},
    }


def write_result(root, name='001_results.json', **kwargs):
    root.mkdir(parents=True, exist_ok=True)
    path = root / name
    path.write_text(json.dumps(payload(**kwargs)))
    return path


def test_full_suite_excludes_smokes_and_missing_difficulties():
    assert plotter.full_suite(payload(), 'judge_accuracy')
    assert plotter.full_suite(payload(limit=5), 'judge_accuracy') is None
    partial = payload()
    partial['n-samples']['visual_probe_medium_gemma4_zoom']['effective'] = 5
    assert plotter.full_suite(partial, 'judge_accuracy') is None
    del partial['results']['visual_probe_medium_gemma4_zoom']
    assert plotter.full_suite(partial, 'judge_accuracy') is None


def test_discovery_ignores_newer_partial_results_and_keeps_conditions_separate(tmp_path):
    gdpo = tmp_path / 'gdpo_run/global_step_10/model'
    expected = write_result(gdpo)
    write_result(gdpo, '999_results.json', limit=5)
    write_result(tmp_path / 'critic_maxcalls5/global_step_20/model')
    write_result(tmp_path / 'critic_maxcalls20/global_step_20/model')
    write_result(tmp_path / 'pretrained/model')
    args = SimpleNamespace(root=tmp_path, condition=[], baseline=[], metric='judge_accuracy', latest_by='name')
    chosen, _ = plotter.discover(args)
    assert expected in [path for _, path in chosen]
    labels = {key[1] for key, _ in chosen}
    assert {'critic_maxcalls5', 'critic_maxcalls20'} <= labels
    assert len(chosen) == 4
    assert next(key[2] for key, _ in chosen if key[0] == 'baseline') is None


def test_tool_count_uses_matching_prejudge_rows_and_refuses_wrong_hashes(tmp_path, monkeypatch):
    monkeypatch.setitem(plotter.COUNTS, 'easy', 2)
    result = write_result(tmp_path / 'run/model')
    task = 'visual_probe_easy_gemma4_zoom'
    final_path = plotter.shared.sample_file_for_benchmark(result, task)
    rows = [{'doc_id': i, 'doc_hash': 'hash-' + str(i)} for i in range(2)]
    final_path.write_text('\n'.join(json.dumps(row) for row in rows))
    assert plotter.tool_rows(result, task) is None
    prej = tmp_path / 'run/prejudge_samples/job/prejudge_samples_visual_probe_easy_gemma4_zoom.jsonl'
    prej.parent.mkdir(parents=True)
    trajectories = [{**row, 'resps': ['<tool_call>x</tool_call>' * (i + 1)]} for i, row in enumerate(rows)]
    prej.write_text('\n'.join(json.dumps(row) for row in trajectories))
    recovered, source = plotter.tool_rows(result, task)
    assert source == prej
    assert sum(len(plotter.shared.TOOL_CALL_RE.findall(str(row['resps']))) for row in recovered) == 3
    trajectories[0]['doc_hash'] = 'different-question'
    prej.write_text('\n'.join(json.dumps(row) for row in trajectories))
    assert plotter.tool_rows(result, task) is None


def test_weighted_overall_and_real_checkpoint_steps(tmp_path):
    result = write_result(tmp_path / 'gdpo/global_step_20/model')
    d = json.loads(result.read_text())
    d['results']['visual_probe_easy_gemma4_zoom']['judge_accuracy,none'] = 1.0
    d['results']['visual_probe_medium_gemma4_zoom']['judge_accuracy,none'] = 0.0
    d['results']['visual_probe_hard_gemma4_zoom']['judge_accuracy,none'] = 0.0
    result.write_text(json.dumps(d))
    records, tools, _ = plotter.load_selected([(('checkpoint', 'GDPO', 20), result)], 'judge_accuracy')
    overall = next(row for row in records if row['difficulty'] == 'overall')
    assert overall['accuracy'] == pytest.approx(141 / 515)
    assert all(row['step'] == 20 for row in records)
    assert tools == []  # Missing logs must not become a zero-tool series.


def test_wrapper_defaults_register_both_training_conditions_without_duplicate_points(tmp_path):
    capture = tmp_path / 'arguments.json'
    python = tmp_path / 'capture-python'
    python.write_text('#!' + sys.executable + '\nimport json,sys\n'
                      + 'open(' + repr(str(capture)) + ', "w").write(json.dumps(sys.argv[1:]))\n')
    python.chmod(0o700)
    root = tmp_path / 'results'
    env = dict(os.environ, PYTHON=str(python), EVAL_OUTPUT_ROOT=str(root), OUTPUT_DIR=str(tmp_path / 'plots'))
    for key in ('CONDITION_GDPO_DIR', 'CONDITION_CRITIC_DIR', 'CONDITION_GDPO_LABEL', 'CONDITION_CRITIC_LABEL'):
        env.pop(key, None)
    subprocess.run(['bash', str(EXAMPLES / 'plot_eval_accuracy_over_steps_qwen35_27b_judge_gemma4_e4b.sh')],
                   check=True, env=env, text=True, capture_output=True)
    arguments = json.loads(capture.read_text())
    conditions = [arguments[i + 1] for i, value in enumerate(arguments) if value == '--condition']
    assert len(conditions) == 2
    for item in conditions:
        label, directory = item.split('=', 1)
        assert Path(directory).parent == root / 'training_v1'
        for step in (100, 200):
            result = write_result(Path(directory) / f'global_step_{step}/model')
            data = json.loads(result.read_text())
            data['model_name'] = 'gemma4_e4b_thinking'
            result.write_text(json.dumps(data))
    chosen, _ = plotter.discover(SimpleNamespace(root=root, condition=conditions, baseline=[],
                                               metric='judge_accuracy', latest_by='name'))
    assert len(chosen) == 4
    assert {key[2] for key, _ in chosen} == {100, 200}
