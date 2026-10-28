#!/usr/bin/env python3
"""Gemma4 E4B Visual Probe plots with full-suite and tool-trace coverage checks.

Reuse selected9's records, metric handling, step parsing and weighted aggregate.
Pretrained runs are reference bars until checkpoint evaluations exist, then
dotted baselines. Native Gemma final logs can omit resps; match the saved
pre-judge trajectories by doc_id AND doc_hash rather than reporting zero calls.
"""
import argparse
import csv
import json
import math
from pathlib import Path
import sys

import plot_eval_accuracy_over_steps as shared

COUNTS = {'easy': 141, 'medium': 268, 'hard': 106}
DIFFICULTIES = (*COUNTS, 'overall')


def full_suite(payload, metric):
    if payload.get('config', {}).get('limit') is not None:
        return None
    selected = {}
    for task, values in payload.get('results', {}).items():
        difficulty = shared.visual_probe_difficulty(task)
        if difficulty is None:
            continue
        found = shared.metric_value(values, metric)
        count = payload.get('n-samples', {}).get(task, {}).get('effective')
        if difficulty in selected or found is None or count != COUNTS[difficulty]:
            return None
        if not math.isfinite(found[1]) or not 0 <= found[1] <= 1:
            return None
        selected[difficulty] = (task, found)
    return selected if set(selected) == set(COUNTS) else None


def family_matches(payload):
    return ('gemma4_e4b' in str(payload.get('model_name', '')).lower() or
            'model_version=gemma4_e4b' in str(payload.get('config', {}).get('model_args', '')).lower())


def thinking_enabled(payload):
    name = str(payload.get('model_name', '')).lower()
    return 'thinking' in name and 'nonthinking' not in name and 'non-thinking' not in name


def mode_label(payload):
    mode = 'thinking' if thinking_enabled(payload) else 'non-thinking'
    return 'Gemma4 E4B (' + mode + ')'


def checkpoint_label(path, root, payload):
    parts = path.relative_to(root).parts
    index = next(i for i, part in enumerate(parts) if shared.STEP_RE.search(part))
    run = parts[index - 1] if index else root.name
    normalized = run.removeprefix('zoom_eval_')
    for suffix in ('_visual_probe_critic_retry_thr0p5', '_visual_probe'):
        normalized = normalized.removesuffix(suffix)
    label = {
        'gdpo_fresh_lr1e5_noexplore_accformat_gemma4_e4b': 'Gemma4 E4B GDPO',
        'branchstrata20_fresh_lr1e5_critic_gemma4_e4b': 'Gemma4 E4B retry + critic',
    }.get(normalized, run)
    if thinking_enabled(payload):
        label += ' (thinking)'
    return label


def discover(args):
    candidates = {}
    skipped = 0
    overrides = {
        'checkpoint': {shared.parse_condition(raw).label for raw in args.condition},
        'baseline': {shared.parse_condition(raw).label for raw in args.baseline},
    }
    searches = [(args.root, None, None)]
    searches += [(shared.parse_condition(raw).path, shared.parse_condition(raw).label, 'checkpoint')
                 for raw in args.condition]
    searches += [(shared.parse_condition(raw).path, shared.parse_condition(raw).label, 'baseline')
                 for raw in args.baseline]
    for root, label, requested_role in searches:
        if not root.is_dir():
            print('WARNING: no results directory:', root, file=sys.stderr)
            continue
        for path in sorted(root.rglob('*_results.json')):
            try:
                payload = json.loads(path.read_text())
                tasks = full_suite(payload, args.metric)
                if tasks is None or (label is None and not family_matches(payload)):
                    skipped += 1
                    continue
                step = shared.extract_step(path, root)
                role = requested_role or ('checkpoint' if step is not None else 'baseline')
                if role == 'checkpoint' and step is None:
                    continue
                selected_label = label or (checkpoint_label(path, root, payload)
                                           if role == 'checkpoint' else mode_label(payload))
                if label is None and selected_label in overrides[role]:
                    continue
                key = (role, selected_label, step if role == 'checkpoint' else None)
                candidates.setdefault(key, []).append(path)
            except (OSError, ValueError, TypeError) as exc:
                print('WARNING: skipping', path, str(exc), file=sys.stderr)
    chosen = [(key, shared.choose_latest(paths, args.latest_by)) for key, paths in candidates.items()]
    return chosen, skipped


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def row_key(row):
    return str(row['doc_id']), row.get('doc_hash')


def tool_rows(result, task):
    final_path = shared.sample_file_for_benchmark(result, task)
    if not final_path.is_file():
        return None
    try:
        final = read_rows(final_path)
    except (OSError, ValueError):
        return None
    difficulty = shared.visual_probe_difficulty(task)
    if len(final) != COUNTS[difficulty] or len({row_key(row) for row in final}) != len(final):
        return None
    if all('resps' in row and row['resps'] is not None for row in final):
        return final, final_path
    # Stop at the closest run's trajectory directory, avoiding adjacent protocols.
    trace_root = next((parent / 'prejudge_samples' for parent in list(result.parents)[:3]
                       if (parent / 'prejudge_samples').is_dir()), None)
    if trace_root is None:
        return None
    expected = {row_key(row) for row in final}
    for path in sorted(trace_root.rglob('prejudge_samples_' + task + '.jsonl'), reverse=True):
        try:
            rows = read_rows(path)
        except (OSError, ValueError):
            continue
        if (len(rows) == len(final) and {row_key(row) for row in rows} == expected and
                all(row.get('doc_hash') and 'resps' in row and row['resps'] is not None for row in rows)):
            return rows, path
    return None


def load_selected(chosen, metric):
    accuracy, tools, sources = [], [], []
    for (role, label, step), path in chosen:
        payload = json.loads(path.read_text())
        tasks = full_suite(payload, metric)
        for difficulty, (task, (metric_key, value)) in tasks.items():
            accuracy.append(dict(condition=label, step=step, difficulty=difficulty,
                                 benchmark='visual_probe_' + difficulty + '_reasoning_zoom',
                                 metric=metric_key, accuracy=value, result_file=str(path), role=role))
        overall = shared.visual_probe_overall_record(shared.Condition(label, path.parent), step or 0,
                                                     path, payload, metric)
        accuracy.append(dict(condition=label, step=step, difficulty='overall', benchmark=overall.benchmark,
                             metric=overall.metric, accuracy=overall.accuracy, result_file=str(path), role=role))
        task_tools = []
        for difficulty, (task, _) in tasks.items():
            loaded = tool_rows(path, task)
            if loaded is None:
                print('WARNING: complete tool trajectories unavailable:', label, task, file=sys.stderr)
                continue
            rows, source = loaded
            total = sum(len(shared.TOOL_CALL_RE.findall(str(row['resps']))) for row in rows)
            task_tools.append(dict(condition=label, step=step, difficulty=difficulty,
                                   avg_tool_calls=total / len(rows), sample_count=len(rows),
                                   sample_files=str(source), role=role, total_calls=total))
        if len(task_tools) == 3:
            count = sum(row['sample_count'] for row in task_tools)
            total = sum(row['total_calls'] for row in task_tools)
            task_tools.append(dict(condition=label, step=step, difficulty='overall', avg_tool_calls=total / count,
                                   sample_count=count, sample_files=';'.join(row['sample_files'] for row in task_tools),
                                   role=role, total_calls=total))
        tools.extend(task_tools)
        sources.append(dict(condition=label, role=role, step=step, result_file=str(path), samples=515,
                            tool_sources=[row['sample_files'] for row in task_tools if row['difficulty'] != 'overall']))
    return accuracy, tools, sources


def plot(rows, output, title, field, ylabel, percent=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    plt.style.use('default')
    labels = sorted({row['condition'] for row in rows})
    colors = {label: plt.get_cmap('tab10')(index % 10) for index, label in enumerate(labels)}
    checkpoint_mode = any(row['role'] == 'checkpoint' for row in rows)
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.8), facecolor='white')
    for ax, difficulty in zip(axes.flat, DIFFICULTIES):
        panel = [row for row in rows if row['difficulty'] == difficulty]
        if checkpoint_mode:
            for label in labels:
                points = sorted((row for row in panel if row['condition'] == label and row['role'] == 'checkpoint'),
                                key=lambda row: row['step'])
                if points:
                    ax.plot([row['step'] for row in points],
                            [row[field] * (100 if percent else 1) for row in points],
                            marker='o', linewidth=2, label=label, color=colors[label])
                for row in panel:
                    if row['condition'] == label and row['role'] == 'baseline':
                        ax.axhline(row[field] * (100 if percent else 1), linestyle=':', linewidth=2,
                                   label=label, color=colors[label])
            ax.set_xlabel('Training step')
        else:
            panel = sorted(panel, key=lambda row: row['condition'])
            bars = ax.bar(range(len(panel)), [row[field] * (100 if percent else 1) for row in panel],
                          color=[colors[row['condition']] for row in panel], width=.55)
            ax.set_xticks(range(len(panel)), [row['condition'].replace('Gemma4 E4B ', '') for row in panel])
            ax.bar_label(bars, fmt='%.1f' if percent else '%.3f', padding=3)
        ax.set_title(difficulty.title())
        ax.set_ylabel(ylabel)
        maximum = max((row[field] * (100 if percent else 1) for row in panel), default=0)
        ax.set_ylim(0, maximum * 1.18 if maximum > 0 else 1)
        ax.grid(axis='y', alpha=.25)
        ax.set_axisbelow(True)
    fig.suptitle(title, y=.99)
    if checkpoint_mode:
        handles, names = axes[0, 0].get_legend_handles_labels()
        fig.legend(handles, names, loc='upper center', bbox_to_anchor=(.5, .95), ncol=min(3, len(names)), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, .87 if checkpoint_mode else .94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def write_csv(rows, path, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--condition', action='append', default=[], metavar='LABEL=RUN_DIR')
    parser.add_argument('--baseline', action='append', default=[], metavar='LABEL=RUN_DIR')
    parser.add_argument('--metric', default='judge_accuracy')
    parser.add_argument('--latest-by', choices=['name', 'mtime'], default='name')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--csv', type=Path, required=True)
    parser.add_argument('--tool-calls-output', type=Path, required=True)
    parser.add_argument('--tool-calls-csv', type=Path, required=True)
    parser.add_argument('--manifest', type=Path, required=True)
    parser.add_argument('--title', default='Gemma4 E4B Visual Probe Accuracy')
    parser.add_argument('--tool-calls-title', default='Gemma4 E4B Visual Probe Tool Calls')
    parser.add_argument('--percent', action='store_true')
    args = parser.parse_args()
    chosen, skipped = discover(args)
    if not chosen:
        parser.error('No complete Gemma4 E4B Visual Probe results: need easy=141, medium=268, hard=106 and no sample limit.')
    accuracy, tools, sources = load_selected(chosen, args.metric)
    plot(accuracy, args.output, args.title, 'accuracy', 'Accuracy (%)' if args.percent else 'Accuracy', args.percent)
    write_csv(accuracy, args.csv, ['condition', 'step', 'benchmark', 'metric', 'accuracy', 'result_file'])
    write_csv(tools, args.tool_calls_csv, ['condition', 'step', 'difficulty', 'avg_tool_calls', 'sample_count', 'sample_files'])
    if tools:
        plot(tools, args.tool_calls_output, args.tool_calls_title, 'avg_tool_calls', 'Average tool calls per sample')
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.write_text(json.dumps(dict(sources=sources, skipped_incomplete_or_other_results=skipped,
                                            checkpoint_conditions=len({source['condition'] for source in sources if source['role'] == 'checkpoint'}),
                                            checkpoint_evaluations=sum(source['role'] == 'checkpoint' for source in sources)), indent=2) + '\n')
    for path in [args.output, args.csv, args.tool_calls_csv, args.manifest] + ([args.tool_calls_output] if tools else []):
        print('Wrote:', path)
    if not any(source['role'] == 'checkpoint' for source in sources):
        print('No checkpoint evaluations yet; showing complete pretrained baselines as comparisons, without invented training steps.')


if __name__ == '__main__':
    main()
