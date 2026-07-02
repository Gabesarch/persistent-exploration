#!/usr/bin/env python3
"""Render row-aligned validation JSONL files from multiple runs as one HTML page."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import random
import re
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Condition:
    name: str
    root: Path


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc
    return rows


def as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def is_correct(row: dict[str, Any]) -> bool:
    accuracy = as_float(row.get("accuracy"))
    return accuracy == 1.0


def image_src(path: str, embed_images: bool) -> str:
    if path.startswith(("http://", "https://", "data:")):
        return path
    p = Path(path)
    if not embed_images:
        return p.as_posix()
    try:
        data = p.read_bytes()
    except OSError:
        return p.as_posix()
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def render_images(paths: list[str], embed_images: bool, css_class: str = "images") -> str:
    blocks = []
    seen = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        exists = path.startswith(("http://", "https://", "data:")) or Path(path).exists()
        status_class = "" if exists else " missing"
        blocks.append(
            "<figure>"
            f'<a href="{escape(Path(path).as_posix())}">'
            f'<img src="{escape(image_src(path, embed_images))}" alt="{escape(path)}">'
            "</a>"
            f'<figcaption class="path{status_class}">{escape(path)}</figcaption>'
            "</figure>"
        )
    return f'<div class="{escape(css_class)}">' + "\n".join(blocks) + "</div>" if blocks else ""


def extract_question(prompt: str) -> str:
    match = re.search(r"\nuser\n\n(?P<question>.*?)(?:\nassistant\n|\Z)", prompt, re.DOTALL)
    if match:
        return match.group("question").strip()
    return prompt.strip()


def shorten(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + "\n\n...[truncated]...\n\n" + text[-keep:]


def render_pre(text: Any, max_text_chars: int | None = None) -> str:
    return f"<pre>{escape(shorten(str(text or ''), max_text_chars))}</pre>"


def render_metric(value: Any) -> str:
    if value is None:
        return ""
    number = as_float(value)
    if number is None:
        return escape(str(value))
    return escape(f"{number:g}")


def crop_paths(row: dict[str, Any]) -> list[str]:
    refs = row.get("crop_image_refs") or []
    if not isinstance(refs, list):
        return []
    return [str(ref.get("path")) for ref in refs if isinstance(ref, dict) and ref.get("path")]


def render_crop_refs(row: dict[str, Any], embed_images: bool) -> str:
    refs = row.get("crop_image_refs") or []
    if not isinstance(refs, list) or not refs:
        return '<div class="empty">No crop images</div>'

    figures = []
    for ref in refs:
        if not isinstance(ref, dict) or not ref.get("path"):
            continue
        path = str(ref["path"])
        meta = {
            "order": ref.get("order"),
            "tool": ref.get("tool_name") or ref.get("tool"),
            "assistant_turn": ref.get("assistant_turn"),
            "label": ref.get("label"),
            "bbox_2d": ref.get("bbox_2d"),
            "img_idx": ref.get("img_idx"),
            "size": (
                [ref.get("width"), ref.get("height")]
                if ref.get("width") is not None or ref.get("height") is not None
                else None
            ),
        }
        meta = {k: v for k, v in meta.items() if v is not None}
        exists = Path(path).exists()
        status_class = "" if exists else " missing"
        figures.append(
            "<figure>"
            f'<a href="{escape(Path(path).as_posix())}">'
            f'<img src="{escape(image_src(path, embed_images))}" alt="{escape(path)}">'
            "</a>"
            f'<figcaption class="path{status_class}">{escape(path)}</figcaption>'
            f"<pre>{escape(json.dumps(meta, ensure_ascii=False, indent=2))}</pre>"
            "</figure>"
        )
    return '<div class="crop-images">' + "\n".join(figures) + "</div>" if figures else '<div class="empty">No crop images</div>'


def render_condition(condition: Condition, row: dict[str, Any], embed_images: bool, max_text_chars: int | None) -> str:
    correct = is_correct(row)
    badge_class = "correct" if correct else "incorrect"
    badge_text = "Correct" if correct else "Incorrect"
    metrics = [
        ("accuracy", row.get("accuracy")),
        ("score", row.get("score")),
        ("reward", row.get("reward")),
        ("format", row.get("format")),
        ("tool calls", row.get("tool_call_count")),
        ("tool efficiency", row.get("tool_efficiency")),
        ("tool bonus", row.get("tool_call_bonus")),
    ]
    metric_items = "\n".join(
        f"<tr><th>{escape(label)}</th><td>{render_metric(value)}</td></tr>"
        for label, value in metrics
        if value is not None
    )
    return (
        f'<article class="condition-card {badge_class}">'
        f"<h3>{escape(condition.name)} <span>{badge_text}</span></h3>"
        f"<table>{metric_items}</table>"
        "<details open><summary>Output transcript</summary>"
        f"{render_pre(row.get('output'), max_text_chars)}"
        "</details>"
        "<details open><summary>Crop images</summary>"
        f"{render_crop_refs(row, embed_images)}"
        "</details>"
        "</article>"
    )


def build_sample_section(
    row_index: int,
    random_index: int,
    rows_by_condition: list[dict[str, Any]],
    conditions: list[Condition],
    embed_images: bool,
    max_text_chars: int | None,
) -> str:
    first = rows_by_condition[0]
    correctness_attrs = " ".join(
        f'data-correct-{idx}="{str(is_correct(row)).lower()}"'
        for idx, row in enumerate(rows_by_condition)
    )
    question = extract_question(str(first.get("input") or ""))
    image_paths = [str(path) for path in first.get("images") or [] if path]
    correct_count = sum(1 for row in rows_by_condition if is_correct(row))
    condition_blocks = "\n".join(
        render_condition(condition, row, embed_images, max_text_chars)
        for condition, row in zip(conditions, rows_by_condition, strict=True)
    )
    summary = (
        f"row {row_index} · {correct_count}/{len(conditions)} correct · "
        f"ground truth: {first.get('gts')}"
    )
    return (
        f'<section class="sample" data-row-index="{row_index}" '
        f'data-random-index="{random_index}" {correctness_attrs}>'
        f"<h2>Sample {random_index + 1}</h2>"
        f'<div class="sample-summary">{escape(summary)}</div>'
        "<div class=\"question-block\">"
        "<h3>Question</h3>"
        f"{render_pre(question, max_text_chars)}"
        "</div>"
        f"{render_images(image_paths, embed_images, 'original-images')}"
        "<details><summary>Full prompt</summary>"
        f"{render_pre(first.get('input'), max_text_chars)}"
        "</details>"
        f'<div class="condition-grid">{condition_blocks}</div>'
        "</section>"
    )


def validate_aligned(rows_by_condition: list[list[dict[str, Any]]], paths: list[Path]) -> None:
    lengths = [len(rows) for rows in rows_by_condition]
    if len(set(lengths)) != 1:
        detail = ", ".join(f"{path}: {length}" for path, length in zip(paths, lengths, strict=True))
        raise SystemExit(f"Validation files have different row counts: {detail}")
    for row_index, rows in enumerate(zip(*rows_by_condition, strict=True)):
        inputs = [row.get("input") for row in rows]
        if any(value != inputs[0] for value in inputs[1:]):
            raise SystemExit(f"Validation inputs are not row-aligned at row {row_index}")


def controls_html(conditions: list[Condition]) -> str:
    filter_groups = []
    sort_options = [
        '<option value="random">Random order</option>',
        '<option value="original">Original row order</option>',
    ]
    for idx, condition in enumerate(conditions):
        safe_name = escape(condition.name)
        filter_groups.append(
            '<fieldset class="filter-group">'
            f"<legend>{safe_name}</legend>"
            f'<label><input type="checkbox" data-filter-condition="{idx}" value="correct"> Correct</label>'
            f'<label><input type="checkbox" data-filter-condition="{idx}" value="incorrect"> Incorrect</label>'
            "</fieldset>"
        )
        sort_options.append(f'<option value="correct-{idx}">{safe_name}: correct first</option>')
        sort_options.append(f'<option value="incorrect-{idx}">{safe_name}: incorrect first</option>')
    return (
        '<div class="controls">'
        '<div class="control-row">'
        f'{"".join(filter_groups)}'
        "</div>"
        '<label class="sort-control">Sort '
        f'<select id="sort-mode">{"".join(sort_options)}</select>'
        "</label>"
        '<div id="visible-count" class="visible-count"></div>'
        "</div>"
    )


def build_html(
    conditions: list[Condition],
    step: int,
    selected_indices: list[int],
    rows_by_condition: list[list[dict[str, Any]]],
    embed_images: bool,
    max_text_chars: int | None,
) -> str:
    samples = []
    for random_index, row_index in enumerate(selected_indices):
        rows = [condition_rows[row_index] for condition_rows in rows_by_condition]
        samples.append(
            build_sample_section(
                row_index,
                random_index,
                rows,
                conditions,
                embed_images,
                max_text_chars,
            )
        )

    counts = []
    for condition, rows in zip(conditions, rows_by_condition, strict=True):
        correct = sum(1 for row in rows if is_correct(row))
        counts.append(f"{condition.name}: {correct}/{len(rows)} correct")

    css = """
    :root { color-scheme: light; }
    body { margin: 0; background: #f5f7fa; color: #1f2933; font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
    main { max-width: 1520px; margin: 0 auto; padding: 24px; }
    h1 { margin: 0 0 6px; font-size: 24px; }
    h2 { margin: 0 0 8px; font-size: 20px; }
    h3 { margin: 0 0 10px; font-size: 13px; color: #394b59; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font: 12px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    table { width: 100%; border-collapse: collapse; margin-bottom: 12px; background: white; }
    th, td { border: 1px solid #d0d7de; padding: 6px 8px; text-align: left; vertical-align: top; }
    th { width: 120px; background: #f0f3f6; font-weight: 600; }
    summary { cursor: pointer; font-weight: 650; margin: 10px 0; }
    .subhead { margin: 0 0 16px; color: #57606a; }
    .controls { position: sticky; top: 0; z-index: 2; background: #ffffff; border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; margin: 18px 0 24px; box-shadow: 0 1px 2px rgba(16, 24, 40, 0.04); }
    .control-row { display: flex; flex-wrap: wrap; gap: 12px; align-items: stretch; }
    .filter-group { border: 1px solid #d0d7de; border-radius: 6px; padding: 8px 10px; min-width: 210px; }
    .filter-group legend { font-weight: 650; padding: 0 4px; }
    label { display: inline-flex; gap: 6px; align-items: center; margin-right: 10px; }
    .sort-control { margin-top: 12px; }
    select { margin-left: 6px; padding: 4px 8px; }
    .visible-count { display: inline-block; margin-left: 16px; color: #57606a; font-weight: 650; }
    .sample { border: 1px solid #d0d7de; border-radius: 8px; background: #ffffff; margin: 0 0 24px; padding: 16px; }
    .sample[hidden] { display: none; }
    .sample-summary { color: #57606a; margin-bottom: 14px; }
    .question-block { border-left: 4px solid #0969da; background: #f6f8fa; padding: 12px; margin-bottom: 14px; }
    .condition-grid { display: grid; grid-template-columns: repeat(3, minmax(0, 1fr)); gap: 14px; align-items: start; }
    .condition-card { border: 1px solid #d0d7de; border-radius: 8px; padding: 12px; background: #fbfcfe; min-width: 0; }
    .condition-card.correct { border-top: 5px solid #1a7f37; }
    .condition-card.incorrect { border-top: 5px solid #cf222e; }
    .condition-card h3 { display: flex; justify-content: space-between; gap: 8px; align-items: center; }
    .condition-card h3 span { border-radius: 999px; padding: 2px 8px; font-size: 12px; color: white; background: #cf222e; }
    .condition-card.correct h3 span { background: #1a7f37; }
    .original-images, .crop-images { display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 12px; margin: 12px 0; }
    figure { margin: 0; border: 1px solid #d0d7de; border-radius: 6px; overflow: hidden; background: #f6f8fa; }
    img { display: block; max-width: 100%; max-height: 520px; object-fit: contain; margin: 0 auto; background: #111; }
    figcaption { border-top: 1px solid #d0d7de; padding: 7px; color: #57606a; font: 11px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }
    figure pre { border-top: 1px solid #d0d7de; padding: 8px; background: #ffffff; }
    .missing { color: #cf222e; }
    .empty { color: #57606a; font-style: italic; }
    @media (max-width: 1100px) { .condition-grid { grid-template-columns: 1fr; } }
    """
    js = """
    const samples = Array.from(document.querySelectorAll('.sample'));
    const sortMode = document.getElementById('sort-mode');
    const count = document.getElementById('visible-count');
    const checkboxes = Array.from(document.querySelectorAll('[data-filter-condition]'));

    function wantedState(conditionIdx) {
      const boxes = checkboxes.filter((box) => Number(box.dataset.filterCondition) === conditionIdx);
      const correct = boxes.some((box) => box.value === 'correct' && box.checked);
      const incorrect = boxes.some((box) => box.value === 'incorrect' && box.checked);
      if ((correct && incorrect) || (!correct && !incorrect)) return 'any';
      return correct ? 'correct' : 'incorrect';
    }

    function sampleMatches(sample) {
      const conditionCount = Number(document.body.dataset.conditionCount);
      for (let idx = 0; idx < conditionCount; idx += 1) {
        const wanted = wantedState(idx);
        if (wanted === 'any') continue;
        const actualCorrect = sample.getAttribute(`data-correct-${idx}`) === 'true';
        if (wanted === 'correct' && !actualCorrect) return false;
        if (wanted === 'incorrect' && actualCorrect) return false;
      }
      return true;
    }

    function orderValue(sample, mode) {
      if (mode === 'random') return Number(sample.dataset.randomIndex);
      if (mode === 'original') return Number(sample.dataset.rowIndex);
      const [direction, conditionText] = mode.split('-');
      const conditionIdx = Number(conditionText);
      const actualCorrect = sample.getAttribute(`data-correct-${conditionIdx}`) === 'true';
      const primary = direction === 'correct' ? (actualCorrect ? 0 : 1) : (actualCorrect ? 1 : 0);
      return primary * 1000000 + Number(sample.dataset.randomIndex);
    }

    function applyControls() {
      const mode = sortMode.value;
      let visible = 0;
      for (const sample of samples) {
        const show = sampleMatches(sample);
        sample.hidden = !show;
        sample.style.order = String(orderValue(sample, mode));
        if (show) visible += 1;
      }
      count.textContent = `${visible} / ${samples.length} samples visible`;
    }

    checkboxes.forEach((box) => box.addEventListener('change', applyControls));
    sortMode.addEventListener('change', applyControls);
    document.getElementById('samples').style.display = 'flex';
    document.getElementById('samples').style.flexDirection = 'column';
    applyControls();
    """
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>Validation Comparison Step {step}</title><style>{css}</style></head>"
        f'<body data-condition-count="{len(conditions)}"><main>'
        f"<h1>Validation Comparison: Step {step}</h1>"
        f'<p class="subhead">{escape(" · ".join(counts))}</p>'
        f"{controls_html(conditions)}"
        f'<div id="samples">{"".join(samples)}</div>'
        f"<script>{js}</script>"
        "</main></body></html>\n"
    )


def parse_condition(value: str) -> Condition:
    if "=" not in value:
        raise argparse.ArgumentTypeError("conditions must be NAME=ROOT")
    name, root = value.split("=", 1)
    name = name.strip()
    root = root.strip()
    if not name or not root:
        raise argparse.ArgumentTypeError("conditions must be NAME=ROOT")
    return Condition(name=name, root=Path(root).expanduser())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--condition", action="append", type=parse_condition, required=True, help="Condition in NAME=CHECKPOINT_ROOT format")
    parser.add_argument("--step", type=int, required=True, help="Validation step number")
    parser.add_argument("--num-samples", type=int, default=24, help="Number of row-aligned samples to render")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for row selection")
    parser.add_argument("--output", type=Path, required=True, help="Output HTML path")
    parser.add_argument("--embed-images", action="store_true", help="Embed original and crop images as base64 data URIs")
    parser.add_argument("--max-text-chars", type=int, default=0, help="Truncate long prompt/output text; 0 means no truncation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    conditions: list[Condition] = args.condition
    if len(conditions) < 2:
        raise SystemExit("At least two --condition values are required")
    if args.num_samples < 1:
        raise SystemExit("--num-samples must be >= 1")

    paths = [condition.root / "validation_data" / f"{args.step}.jsonl" for condition in conditions]
    for path in paths:
        if not path.exists():
            raise SystemExit(f"Missing validation file: {path}")

    rows_by_condition = [read_jsonl(path) for path in paths]
    validate_aligned(rows_by_condition, paths)

    total_rows = len(rows_by_condition[0])
    sample_count = min(args.num_samples, total_rows)
    rng = random.Random(args.seed)
    selected_indices = rng.sample(range(total_rows), sample_count)

    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    max_text_chars = args.max_text_chars if args.max_text_chars > 0 else None
    html = build_html(
        conditions=conditions,
        step=args.step,
        selected_indices=selected_indices,
        rows_by_condition=rows_by_condition,
        embed_images=args.embed_images,
        max_text_chars=max_text_chars,
    )
    output.write_text(html, encoding="utf-8")

    print(f"Wrote {output}")
    print(f"step={args.step}")
    print(f"num_samples={sample_count}")
    print(f"seed={args.seed}")
    for condition, rows in zip(conditions, rows_by_condition, strict=True):
        correct = sum(1 for row in rows if is_correct(row))
        print(f"{condition.name}: {correct}/{len(rows)} correct by accuracy == 1.0")


if __name__ == "__main__":
    main()
