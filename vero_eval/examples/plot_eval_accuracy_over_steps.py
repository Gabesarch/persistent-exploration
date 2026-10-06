#!/usr/bin/env python3
"""Plot benchmark accuracy over training steps for multiple eval conditions.

Example:
  python examples/plot_eval_accuracy_over_steps.py \
    /scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2/zoom_eval_answer_retry_forcedprefill_visual_probe \
    /scratch/gpfs/ZHUANGL/gs1693/lmms_eval_vero2/zoom_eval_hardonly_visual_probe \
    --output visual_probe_accuracy_over_steps.png \
    --csv visual_probe_accuracy_over_steps.csv

Condition arguments can be either paths or label=path pairs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any


STEP_RE = re.compile(r"(?:global[_-]?step|step)[_-]?(\d+)", re.IGNORECASE)


@dataclass(frozen=True)
class Condition:
    label: str
    path: Path


@dataclass(frozen=True)
class Record:
    condition: str
    step: int
    benchmark: str
    metric: str
    accuracy: float
    result_file: Path


@dataclass(frozen=True)
class BaselineRecord:
    condition: str
    benchmark: str
    metric: str
    accuracy: float
    result_file: Path


@dataclass(frozen=True)
class ToolCallRecord:
    condition: str
    step: int
    difficulty: str
    avg_tool_calls: float
    sample_count: int
    sample_files: tuple[Path, ...]


VISUAL_PROBE_DIFFICULTIES = ("easy", "medium", "hard")
TOOL_CALL_RE = re.compile(r"<tool_call>")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Read lmms-eval *_results.json files from condition folders and "
            "plot each benchmark's accuracy over checkpoint steps."
        )
    )
    parser.add_argument(
        "conditions",
        nargs="+",
        help="Condition folders, optionally as label=/path/to/folder.",
    )
    parser.add_argument(
        "--restart-from",
        action="append",
        default=[],
        metavar="LABEL=SOURCE_LABEL:STEP",
        help=(
            "Align a weights-only restart with its source run: add STEP to its "
            "local checkpoint steps and prepend the source result at STEP. "
            "Applies to both plots and CSVs. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--metric",
        default="judge_accuracy",
        help="Metric prefix to plot. The script matches keys like '<metric>,none'.",
    )
    parser.add_argument(
        "--baseline",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help=(
            "Completed non-checkpoint eval folder to draw as dotted horizontal "
            "lines. Can be passed multiple times."
        ),
    )
    parser.add_argument(
        "--output",
        default="accuracy_over_steps.png",
        help="Path for the output figure.",
    )
    parser.add_argument(
        "--csv",
        default="accuracy_over_steps.csv",
        help="Path for the parsed CSV table. Use an empty string to disable.",
    )
    parser.add_argument(
        "--title",
        default="Benchmark Accuracy over Training Steps",
        help="Figure title.",
    )
    parser.add_argument(
        "--percent",
        action="store_true",
        help="Plot accuracy as percentages instead of fractions.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=[],
        help="Only include benchmarks matching this regex. Can be passed multiple times.",
    )
    parser.add_argument(
        "--exclude",
        action="append",
        default=[],
        help="Exclude benchmarks matching this regex. Can be passed multiple times.",
    )
    parser.add_argument(
        "--latest-by",
        choices=("mtime", "name"),
        default="name",
        help=(
            "If multiple results files exist for a condition/step/model directory, "
            "choose the newest by filename or modification time."
        ),
    )
    parser.add_argument(
        "--visual-probe-overall",
        action="store_true",
        help=(
            "Add a Visual Probe Overall subplot. For each checkpoint, this is "
            "the sample-weighted accuracy across easy, medium, and hard."
        ),
    )
    parser.add_argument(
        "--tool-calls-output",
        default="",
        help=(
            "Optional path for a second figure showing average Visual Probe tool "
            "calls per sample for each difficulty and overall."
        ),
    )
    parser.add_argument(
        "--tool-calls-csv",
        default="",
        help="Optional path for the average tool-calls CSV table.",
    )
    parser.add_argument(
        "--tool-calls-title",
        default="Average Tool Calls over Training Steps",
        help="Title for the average tool-calls figure.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=180,
        help="Figure DPI.",
    )
    parser.add_argument(
        "--fig-width",
        type=float,
        default=12.0,
        help="Figure width in inches.",
    )
    parser.add_argument(
        "--subplot-height",
        type=float,
        default=3.2,
        help="Height in inches per subplot row.",
    )
    return parser.parse_args()


def parse_condition(raw: str) -> Condition:
    if "=" in raw:
        label, path = raw.split("=", 1)
        label = label.strip()
        if not label:
            raise ValueError(f"Empty condition label in {raw!r}")
    else:
        path = raw
        label = Path(path).name
    return Condition(label=label, path=Path(path).expanduser())


def extract_step(path: Path, root: Path) -> int | None:
    try:
        candidates = [part for part in path.relative_to(root).parts]
    except ValueError:
        candidates = list(path.parts)
    for part in candidates:
        match = STEP_RE.search(part)
        if match:
            return int(match.group(1))
    return None


def metric_value(metrics: dict[str, Any], metric: str) -> tuple[str, float] | None:
    preferred_keys = (f"{metric},none", metric)
    for key in preferred_keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)

    matching_keys = sorted(k for k in metrics if k.startswith(f"{metric},"))
    for key in matching_keys:
        value = metrics.get(key)
        if isinstance(value, (int, float)):
            return key, float(value)
    return None


def choose_latest(paths: list[Path], latest_by: str) -> Path:
    if latest_by == "mtime":
        return max(paths, key=lambda path: (path.stat().st_mtime, path.name))
    return max(paths, key=lambda path: path.name)


def discover_result_files(condition: Condition, latest_by: str) -> list[tuple[int, Path]]:
    if not condition.path.is_dir():
        raise FileNotFoundError(f"Condition folder does not exist: {condition.path}")

    by_step_and_parent: dict[tuple[int, Path], list[Path]] = {}
    for result_file in condition.path.rglob("*_results.json"):
        step = extract_step(result_file, condition.path)
        if step is None:
            continue
        by_step_and_parent.setdefault((step, result_file.parent), []).append(result_file)

    chosen: list[tuple[int, Path]] = []
    for (step, _parent), paths in by_step_and_parent.items():
        chosen.append((step, choose_latest(paths, latest_by)))
    return sorted(chosen, key=lambda item: (item[0], str(item[1])))


def discover_baseline_result_file(
    condition: Condition,
    latest_by: str,
) -> Path | None:
    """Return the latest results file for an eval without checkpoint steps."""
    if not condition.path.is_dir():
        raise FileNotFoundError(f"Baseline folder does not exist: {condition.path}")
    paths = list(condition.path.rglob("*_results.json"))
    if not paths:
        return None
    return choose_latest(paths, latest_by)


def load_records(
    condition: Condition,
    metric: str,
    latest_by: str,
    add_visual_probe_overall: bool = False,
) -> list[Record]:
    records: list[Record] = []
    for step, result_file in discover_result_files(condition, latest_by):
        with result_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload.get("results", {})
        if not isinstance(results, dict):
            continue
        for benchmark, metrics in results.items():
            if not isinstance(metrics, dict):
                continue
            found = metric_value(metrics, metric)
            if found is None:
                continue
            metric_key, accuracy = found
            records.append(
                Record(
                    condition=condition.label,
                    step=step,
                    benchmark=str(benchmark),
                    metric=metric_key,
                    accuracy=accuracy,
                    result_file=result_file,
                )
            )
        if add_visual_probe_overall:
            overall = visual_probe_overall_record(
                condition=condition,
                step=step,
                result_file=result_file,
                payload=payload,
                metric=metric,
            )
            if overall is not None:
                records.append(overall)
    return records


def load_baseline_records(
    condition: Condition,
    metric: str,
    latest_by: str,
    add_visual_probe_overall: bool = False,
) -> list[BaselineRecord]:
    result_file = discover_baseline_result_file(condition, latest_by)
    if result_file is None:
        return []

    with result_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    results = payload.get("results", {})
    if not isinstance(results, dict):
        return []

    records: list[BaselineRecord] = []
    for benchmark, metrics in results.items():
        if not isinstance(metrics, dict):
            continue
        found = metric_value(metrics, metric)
        if found is None:
            continue
        metric_key, accuracy = found
        records.append(
            BaselineRecord(
                condition=condition.label,
                benchmark=str(benchmark),
                metric=metric_key,
                accuracy=accuracy,
                result_file=result_file,
            )
        )

    if add_visual_probe_overall:
        overall = visual_probe_overall_record(
            condition=condition,
            step=0,
            result_file=result_file,
            payload=payload,
            metric=metric,
        )
        if overall is not None:
            records.append(
                BaselineRecord(
                    condition=condition.label,
                    benchmark=overall.benchmark,
                    metric=overall.metric,
                    accuracy=overall.accuracy,
                    result_file=result_file,
                )
            )
    return records


def visual_probe_difficulty(benchmark: str) -> str | None:
    """Return the difficulty for a Visual Probe task, if the name is recognized."""
    normalized = benchmark.lower()
    if "visual_probe" not in normalized:
        return None
    for difficulty in VISUAL_PROBE_DIFFICULTIES:
        if re.search(rf"(?:^|_)visual_probe_{difficulty}(?:_|$)", normalized):
            return difficulty
    return None


def visual_probe_overall_record(
    condition: Condition,
    step: int,
    result_file: Path,
    payload: dict[str, Any],
    metric: str,
) -> Record | None:
    """Build a sample-weighted easy/medium/hard Visual Probe aggregate."""
    results = payload.get("results", {})
    sample_counts = payload.get("n-samples", {})
    if not isinstance(results, dict) or not isinstance(sample_counts, dict):
        return None

    by_difficulty: dict[str, tuple[float, int, str]] = {}
    for benchmark, metrics in results.items():
        difficulty = visual_probe_difficulty(str(benchmark))
        if difficulty is None or not isinstance(metrics, dict):
            continue
        found = metric_value(metrics, metric)
        counts = sample_counts.get(benchmark, {})
        if found is None or not isinstance(counts, dict):
            continue
        count = counts.get("effective")
        if not isinstance(count, int) or isinstance(count, bool) or count <= 0:
            continue
        metric_key, accuracy = found
        by_difficulty[difficulty] = (accuracy, count, metric_key)

    if set(by_difficulty) != set(VISUAL_PROBE_DIFFICULTIES):
        return None

    total_samples = sum(count for _, count, _ in by_difficulty.values())
    weighted_accuracy = sum(
        accuracy * count for accuracy, count, _ in by_difficulty.values()
    ) / total_samples
    metric_keys = {metric_key for _, _, metric_key in by_difficulty.values()}
    aggregate_metric = metric_keys.pop() if len(metric_keys) == 1 else metric
    return Record(
        condition=condition.label,
        step=step,
        benchmark="visual_probe_overall",
        metric=aggregate_metric,
        accuracy=weighted_accuracy,
        result_file=result_file,
    )


def sample_file_for_benchmark(result_file: Path, benchmark: str) -> Path:
    suffix = "_results.json"
    if not result_file.name.endswith(suffix):
        raise ValueError(f"Unexpected result filename: {result_file}")
    run_prefix = result_file.name[: -len(suffix)]
    return result_file.with_name(f"{run_prefix}_samples_{benchmark}.jsonl")


def load_tool_call_records(
    condition: Condition,
    latest_by: str,
) -> list[ToolCallRecord]:
    """Compute per-sample Visual Probe tool-call averages from JSONL logs."""
    records: list[ToolCallRecord] = []
    for step, result_file in discover_result_files(condition, latest_by):
        with result_file.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        results = payload.get("results", {})
        if not isinstance(results, dict):
            continue

        totals = {difficulty: 0 for difficulty in VISUAL_PROBE_DIFFICULTIES}
        counts = {difficulty: 0 for difficulty in VISUAL_PROBE_DIFFICULTIES}
        files: dict[str, list[Path]] = {
            difficulty: [] for difficulty in VISUAL_PROBE_DIFFICULTIES
        }
        for benchmark in results:
            difficulty = visual_probe_difficulty(str(benchmark))
            if difficulty is None:
                continue
            sample_file = sample_file_for_benchmark(result_file, str(benchmark))
            if not sample_file.is_file():
                print(
                    f"WARNING: missing sample log for {benchmark!r}: {sample_file}",
                    file=sys.stderr,
                )
                continue
            files[difficulty].append(sample_file)
            with sample_file.open("r", encoding="utf-8") as handle:
                for line_number, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError as exc:
                        raise ValueError(
                            f"Invalid JSON in {sample_file}:{line_number}"
                        ) from exc
                    totals[difficulty] += len(
                        TOOL_CALL_RE.findall(str(sample.get("resps") or ""))
                    )
                    counts[difficulty] += 1

        for difficulty in VISUAL_PROBE_DIFFICULTIES:
            if counts[difficulty] == 0:
                continue
            records.append(
                ToolCallRecord(
                    condition=condition.label,
                    step=step,
                    difficulty=difficulty,
                    avg_tool_calls=totals[difficulty] / counts[difficulty],
                    sample_count=counts[difficulty],
                    sample_files=tuple(files[difficulty]),
                )
            )

        if all(counts[difficulty] > 0 for difficulty in VISUAL_PROBE_DIFFICULTIES):
            total_calls = sum(totals.values())
            total_samples = sum(counts.values())
            records.append(
                ToolCallRecord(
                    condition=condition.label,
                    step=step,
                    difficulty="overall",
                    avg_tool_calls=total_calls / total_samples,
                    sample_count=total_samples,
                    sample_files=tuple(
                        sample_file
                        for difficulty in VISUAL_PROBE_DIFFICULTIES
                        for sample_file in files[difficulty]
                    ),
                )
            )
    return records


def keep_benchmark(name: str, include: list[str], exclude: list[str]) -> bool:
    if include and not any(re.search(pattern, name) for pattern in include):
        return False
    if exclude and any(re.search(pattern, name) for pattern in exclude):
        return False
    return True


def benchmarks_match(left: str, right: str) -> bool:
    """Match VisualProbe task variants by difficulty for baseline overlays."""
    if left == right:
        return True
    left_difficulty = visual_probe_difficulty(left)
    right_difficulty = visual_probe_difficulty(right)
    return left_difficulty is not None and left_difficulty == right_difficulty


def write_csv(
    records: list[Record],
    baseline_records: list[BaselineRecord],
    output: Path,
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["condition", "step", "benchmark", "metric", "accuracy", "result_file"],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "condition": record.condition,
                    "step": record.step,
                    "benchmark": record.benchmark,
                    "metric": record.metric,
                    "accuracy": record.accuracy,
                    "result_file": str(record.result_file),
                }
            )
        for record in baseline_records:
            writer.writerow(
                {
                    "condition": record.condition,
                    "step": "",
                    "benchmark": record.benchmark,
                    "metric": record.metric,
                    "accuracy": record.accuracy,
                    "result_file": str(record.result_file),
                }
            )


def write_tool_calls_csv(records: list[ToolCallRecord], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "condition",
                "step",
                "difficulty",
                "avg_tool_calls",
                "sample_count",
                "sample_files",
            ],
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "condition": record.condition,
                    "step": record.step,
                    "difficulty": record.difficulty,
                    "avg_tool_calls": record.avg_tool_calls,
                    "sample_count": record.sample_count,
                    "sample_files": ";".join(str(path) for path in record.sample_files),
                }
            )


def nice_benchmark_name(name: str) -> str:
    cleaned = name
    for token in ("_reasoning_zoom", "visual_probe_"):
        cleaned = cleaned.replace(token, "")
    return cleaned.replace("_", " ").title()


def plot_records(
    records: list[Record],
    baseline_records: list[BaselineRecord],
    output: Path,
    title: str,
    percent: bool,
    dpi: int,
    fig_width: float,
    subplot_height: float,
) -> None:
    import matplotlib.pyplot as plt

    benchmarks = sorted({record.benchmark for record in records})
    conditions = sorted({record.condition for record in records})
    baseline_conditions = sorted({record.condition for record in baseline_records})
    ncols = 1 if len(benchmarks) <= 3 else 2
    nrows = math.ceil(len(benchmarks) / ncols)
    fig_height = max(3.5, subplot_height * nrows)

    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, fig_height),
        sharex=True,
        squeeze=False,
    )

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    series_labels = sorted(set(conditions) | set(baseline_conditions))
    colors = {
        label: color_cycle[i % len(color_cycle)]
        for i, label in enumerate(series_labels)
    }

    for index, benchmark in enumerate(benchmarks):
        ax = axes[index // ncols][index % ncols]
        for condition in conditions:
            points = [
                record
                for record in records
                if record.benchmark == benchmark and record.condition == condition
            ]
            points = sorted(points, key=lambda record: record.step)
            if not points:
                continue
            values = [record.accuracy * 100.0 if percent else record.accuracy for record in points]
            ax.plot(
                [record.step for record in points],
                values,
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                label=condition,
                color=colors.get(condition),
            )
        for condition in baseline_conditions:
            points = [
                record
                for record in baseline_records
                if record.condition == condition
                and benchmarks_match(record.benchmark, benchmark)
            ]
            for point in points:
                value = point.accuracy * 100.0 if percent else point.accuracy
                ax.axhline(
                    value,
                    linestyle=":",
                    linewidth=2.0,
                    label=condition,
                    color=colors.get(condition),
                )
        ax.set_title(nice_benchmark_name(benchmark))
        ax.set_ylabel("Accuracy (%)" if percent else "Accuracy")
        ax.grid(True, axis="both", alpha=0.25)

    for index in range(len(benchmarks), nrows * ncols):
        axes[index // ncols][index % ncols].axis("off")

    for ax in axes[-1]:
        ax.set_xlabel("Training step")

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.suptitle(title, y=0.99)
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(len(labels), 4),
            frameon=False,
        )
        top = 0.88
    else:
        fig.suptitle(title, y=0.99)
        top = 0.94
    fig.tight_layout(rect=(0, 0, 1, top))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def plot_tool_call_records(
    records: list[ToolCallRecord],
    output: Path,
    title: str,
    dpi: int,
    fig_width: float,
    subplot_height: float,
) -> None:
    import matplotlib.pyplot as plt

    difficulties = (*VISUAL_PROBE_DIFFICULTIES, "overall")
    conditions = sorted({record.condition for record in records})
    ncols = 2
    nrows = 2
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=(fig_width, max(3.5, subplot_height * nrows)),
        sharex=True,
        sharey=True,
        squeeze=False,
    )

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
    colors = {
        condition: color_cycle[i % len(color_cycle)]
        for i, condition in enumerate(conditions)
    }
    for index, difficulty in enumerate(difficulties):
        ax = axes[index // ncols][index % ncols]
        for condition in conditions:
            points = sorted(
                (
                    record
                    for record in records
                    if record.difficulty == difficulty
                    and record.condition == condition
                ),
                key=lambda record: record.step,
            )
            if not points:
                continue
            ax.plot(
                [record.step for record in points],
                [record.avg_tool_calls for record in points],
                marker="o",
                linewidth=2.0,
                markersize=4.5,
                label=condition,
                color=colors.get(condition),
            )
        ax.set_title(difficulty.title())
        ax.set_ylabel("Average tool calls per sample")
        ax.grid(True, axis="both", alpha=0.25)

    for ax in axes[-1]:
        ax.set_xlabel("Training step")

    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.suptitle(title, y=0.99)
    if handles:
        fig.legend(
            handles,
            labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 0.955),
            ncol=min(len(labels), 4),
            frameon=False,
        )
        top = 0.88
    else:
        top = 0.94
    fig.tight_layout(rect=(0, 0, 1, top))

    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    plt.close(fig)


def align_restart_steps(records, restarts):
    """Use cumulative steps and the evaluated source checkpoint as the anchor."""
    aligned = list(records)
    for label, source_label, step in restarts:
        if not any(record.condition == label for record in records):
            continue
        anchors = [
            record for record in records
            if record.condition == source_label and record.step == step
        ]
        if not anchors:
            raise ValueError(
                f"Restart {label!r} has no source records at {source_label}:{step}"
            )
        aligned = [
            replace(record, step=record.step + step)
            if record.condition == label else record
            for record in aligned
        ]
        # A real local step-zero evaluation takes precedence over the source.
        existing = {
            (record.benchmark if isinstance(record, Record) else record.difficulty)
            for record in records if record.condition == label and record.step == 0
        }
        aligned.extend(
            replace(record, condition=label)
            for record in anchors
            if (record.benchmark if isinstance(record, Record) else record.difficulty)
            not in existing
        )
    return aligned


def main() -> int:
    args = parse_args()
    conditions = [parse_condition(raw) for raw in args.conditions]
    baseline_conditions = [parse_condition(raw) for raw in args.baseline]
    restarts = []
    labels = {condition.label for condition in conditions}
    for raw in args.restart_from:
        label, source = raw.split("=", 1)
        source_label, step = source.rsplit(":", 1)
        if label not in labels or source_label not in labels or label == source_label:
            raise ValueError(f"Restart needs distinct known condition labels: {raw!r}")
        restarts.append((label, source_label, int(step)))

    records: list[Record] = []
    for condition in conditions:
        if not condition.path.is_dir():
            print(
                f"WARNING: skipping missing condition {condition.label!r}: "
                f"{condition.path}",
                file=sys.stderr,
            )
            continue
        records.extend(
            load_records(
                condition,
                args.metric,
                args.latest_by,
                add_visual_probe_overall=args.visual_probe_overall,
            )
        )

    records = [
        record
        for record in records
        if keep_benchmark(record.benchmark, args.include, args.exclude)
    ]
    if not records:
        raise SystemExit(
            "No matching result records found. Check condition paths, --metric, "
            "and --include/--exclude filters."
        )

    records = align_restart_steps(records, restarts)

    baseline_records: list[BaselineRecord] = []
    for condition in baseline_conditions:
        if not condition.path.is_dir():
            print(
                f"WARNING: skipping missing baseline {condition.label!r}: "
                f"{condition.path}",
                file=sys.stderr,
            )
            continue
        loaded = load_baseline_records(
            condition,
            args.metric,
            args.latest_by,
            add_visual_probe_overall=args.visual_probe_overall,
        )
        if not loaded:
            print(
                f"WARNING: no matching results found for baseline "
                f"{condition.label!r}: {condition.path}",
                file=sys.stderr,
            )
        baseline_records.extend(loaded)
    baseline_records = [
        record
        for record in baseline_records
        if keep_benchmark(record.benchmark, args.include, args.exclude)
    ]

    output = Path(args.output).expanduser()
    plot_records(
        records=records,
        baseline_records=baseline_records,
        output=output,
        title=args.title,
        percent=args.percent,
        dpi=args.dpi,
        fig_width=args.fig_width,
        subplot_height=args.subplot_height,
    )

    if args.csv:
        write_csv(records, baseline_records, Path(args.csv).expanduser())

    if args.tool_calls_output:
        tool_call_records: list[ToolCallRecord] = []
        for condition in conditions:
            if condition.path.is_dir():
                tool_call_records.extend(
                    load_tool_call_records(condition, args.latest_by)
                )
        if not tool_call_records:
            raise SystemExit(
                "No Visual Probe sample logs found for the average tool-calls plot."
            )
        tool_call_records = align_restart_steps(tool_call_records, restarts)
        tool_calls_output = Path(args.tool_calls_output).expanduser()
        plot_tool_call_records(
            records=tool_call_records,
            output=tool_calls_output,
            title=args.tool_calls_title,
            dpi=args.dpi,
            fig_width=args.fig_width,
            subplot_height=args.subplot_height,
        )
        print(f"Wrote tool-calls figure: {tool_calls_output}")
        if args.tool_calls_csv:
            tool_calls_csv = Path(args.tool_calls_csv).expanduser()
            write_tool_calls_csv(tool_call_records, tool_calls_csv)
            print(f"Wrote tool-calls CSV: {tool_calls_csv}")

    print(f"Wrote figure: {output}")
    if args.csv:
        print(f"Wrote CSV: {Path(args.csv).expanduser()}")
    print(
        f"Plotted {len(records)} points across "
        f"{len({record.benchmark for record in records})} benchmarks and "
        f"{len({record.condition for record in records})} conditions, plus "
        f"{len({record.condition for record in baseline_records})} baselines."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
