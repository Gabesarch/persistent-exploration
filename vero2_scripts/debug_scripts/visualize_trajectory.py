#!/usr/bin/env python3
"""Render a verl/Vero2 debug trajectory as a local HTML page."""

from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import random
from html import escape
from pathlib import Path
from typing import Any


EVENT_ORDER = {
    "sample_start": 0,
    "assistant_turn": 1,
    "tool_call": 2,
    "tool_response": 3,
    "sample_final": 4,
}

SUCCESS_SCORE_THRESHOLD = 1.0


def read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON in {path}:{line_no}: {exc}") from exc


def first_scalar(value: Any) -> Any:
    if isinstance(value, list) and value:
        return first_scalar(value[0])
    return value


def flatten_paths(value: Any) -> list[str]:
    paths: list[str] = []
    if isinstance(value, str):
        paths.append(value)
    elif isinstance(value, list):
        for item in value:
            paths.extend(flatten_paths(item))
    return paths


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks = []
    for item in content:
        if isinstance(item, dict) and "text" in item:
            chunks.append(str(item["text"]))
    return "\n".join(chunks)


def image_paths_from_content(content: Any) -> list[str]:
    if not isinstance(content, list):
        return []
    paths = []
    for item in content:
        if not isinstance(item, dict):
            continue
        for key in ("image", "image_url"):
            if key in item and isinstance(item[key], str):
                paths.append(item[key])
    return paths


def image_src(path: str, embed_images: bool) -> str:
    if path.startswith(("http://", "https://", "data:")):
        return path
    if not embed_images:
        return Path(path).as_posix()

    p = Path(path)
    try:
        data = p.read_bytes()
    except OSError:
        return p.as_posix()
    mime = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
    encoded = base64.b64encode(data).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def shorten(text: str, max_chars: int | None) -> str:
    if max_chars is None or max_chars <= 0 or len(text) <= max_chars:
        return text
    keep = max_chars // 2
    return text[:keep] + "\n\n...[truncated]...\n\n" + text[-keep:]


def find_sample(
    sample_path: Path,
    request_id: str | None,
    sample_index: int | None,
    rollout_n: int | None,
    select: str,
) -> dict[str, Any]:
    samples = [r for r in read_jsonl(sample_path) if r.get("event") == "sample_final"]
    if not samples:
        raise SystemExit(f"No sample_final records found in {sample_path}")

    if request_id:
        for sample in samples:
            if sample.get("request_id") == request_id:
                return sample
        raise SystemExit(f"request_id not found: {request_id}")

    if sample_index is not None:
        matches = [
            s
            for s in samples
            if s.get("index") == sample_index
            or s.get("trajectory", {}).get("sample_index") == sample_index
        ]
        if rollout_n is not None:
            matches = [s for s in matches if s.get("trajectory", {}).get("rollout_n") == rollout_n]
        if not matches:
            raise SystemExit(f"No sample found for sample_index={sample_index}, rollout_n={rollout_n}")
        return sorted(matches, key=sample_score, reverse=True)[0]

    if select == "first":
        return samples[0]
    if select == "last":
        return samples[-1]
    if select == "most-tools":
        return sorted(samples, key=lambda s: (len(s.get("tool_rewards") or []), sample_score(s)), reverse=True)[0]
    if select == "longest":
        return sorted(samples, key=sample_score, reverse=True)[0]
    if select == "random":
        return random.choice(samples)
    raise SystemExit(f"Unsupported --select value: {select}")


def read_reward_assignments(reward_path: Path) -> dict[str, dict[str, Any]]:
    if not reward_path.exists():
        return {}
    rewards: dict[str, dict[str, Any]] = {}
    for row in read_jsonl(reward_path):
        request_id = row.get("request_id")
        if request_id:
            rewards[str(request_id)] = row
    return rewards


def reward_score(reward: dict[str, Any] | None) -> float | None:
    if not reward:
        return None
    value = reward.get("score", reward.get("final_reward"))
    if value is None:
        raw_result = reward.get("raw_result") or reward.get("reward_extra_info") or {}
        value = raw_result.get("score") if isinstance(raw_result, dict) else None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def reward_extra(reward: dict[str, Any] | None, key: str) -> Any:
    if not reward:
        return None
    if key in reward:
        return reward.get(key)
    for field in ("raw_result", "reward_extra_info"):
        values = reward.get(field)
        if isinstance(values, dict) and key in values:
            return values.get(key)
    return None


def reward_outcome(reward: dict[str, Any] | None) -> str | None:
    score = reward_score(reward)
    if score is None:
        return None
    return "successful" if score >= SUCCESS_SCORE_THRESHOLD else "incorrect"


def attach_reward_assignments(samples: list[dict[str, Any]], rewards: dict[str, dict[str, Any]]) -> None:
    for sample in samples:
        request_id = sample.get("request_id")
        if request_id and str(request_id) in rewards:
            sample["_reward_assignment"] = rewards[str(request_id)]


def filter_by_outcome(samples: list[dict[str, Any]], outcome: str) -> list[dict[str, Any]]:
    if outcome == "all":
        return samples
    return [
        sample
        for sample in samples
        if reward_outcome(sample.get("_reward_assignment")) == outcome
    ]


def find_samples(
    sample_path: Path,
    reward_path: Path | None,
    request_id: str | None,
    sample_index: int | None,
    rollout_n: int | None,
    select: str,
    num_samples: int,
    outcome: str,
) -> list[dict[str, Any]]:
    samples = [r for r in read_jsonl(sample_path) if r.get("event") == "sample_final"]
    if not samples:
        raise SystemExit(f"No sample_final records found in {sample_path}")
    total_samples = len(samples)

    rewards: dict[str, dict[str, Any]] = {}
    if reward_path is not None and (outcome != "all" or reward_path.exists()):
        rewards = read_reward_assignments(reward_path)
        attach_reward_assignments(samples, rewards)
    if outcome != "all":
        if reward_path is None or not reward_path.exists():
            raise SystemExit(f"--outcome={outcome} requires reward assignments, but {reward_path} is missing")
        samples = filter_by_outcome(samples, outcome)
        if not samples:
            raise SystemExit(
                f"No {outcome} samples found in {sample_path}; "
                f"checked {total_samples} samples and {len(rewards)} reward assignments"
            )

    if request_id:
        matches = [sample for sample in samples if sample.get("request_id") == request_id]
        if not matches:
            suffix = f" with outcome={outcome}" if outcome != "all" else ""
            raise SystemExit(f"request_id not found{suffix}: {request_id}")
        return matches[:1]

    if sample_index is not None:
        matches = [
            s
            for s in samples
            if s.get("index") == sample_index
            or s.get("trajectory", {}).get("sample_index") == sample_index
        ]
        if rollout_n is not None:
            matches = [s for s in matches if s.get("trajectory", {}).get("rollout_n") == rollout_n]
        if not matches:
            suffix = f", outcome={outcome}" if outcome != "all" else ""
            raise SystemExit(f"No sample found for sample_index={sample_index}, rollout_n={rollout_n}{suffix}")
        return sorted(matches, key=sample_score, reverse=True)[: max(1, num_samples)]

    if select == "first":
        return samples[: max(1, num_samples)]
    if select == "last":
        return samples[-max(1, num_samples) :]
    if select == "most-tools":
        return sorted(samples, key=lambda s: (len(s.get("tool_rewards") or []), sample_score(s)), reverse=True)[
            : max(1, num_samples)
        ]
    if select == "longest":
        return sorted(samples, key=sample_score, reverse=True)[: max(1, num_samples)]
    if select == "random":
        count = min(max(1, num_samples), len(samples))
        return random.sample(samples, count)
    raise SystemExit(f"Unsupported --select value: {select}")


def sample_score(sample: dict[str, Any]) -> tuple[int, int, int]:
    return (
        int(sample.get("num_turns") or 0),
        len(sample.get("tool_rewards") or []),
        len(sample.get("final_response") or ""),
    )


def events_for_request(agent_path: Path, request_id: str) -> list[dict[str, Any]]:
    return events_for_requests(agent_path, [request_id]).get(request_id, [])


def events_for_requests(agent_path: Path, request_ids: list[str]) -> dict[str, list[dict[str, Any]]]:
    events_by_request_id = {request_id: [] for request_id in request_ids}
    if not agent_path.exists() or not request_ids:
        return events_by_request_id

    wanted = set(request_ids)
    for seq, event in enumerate(read_jsonl(agent_path)):
        request_id = str(event.get("request_id"))
        if request_id in wanted:
            event["_seq"] = seq
            events_by_request_id[request_id].append(event)

    for request_id, events in events_by_request_id.items():
        events_by_request_id[request_id] = sorted(
            events,
            key=lambda e: (e.get("assistant_turn") or 0, EVENT_ORDER.get(e.get("event"), 99), e["_seq"]),
        )
    return events_by_request_id


def render_meta(sample: dict[str, Any], debug_dir: Path) -> str:
    extra = sample.get("extra_info") or {}
    traj = sample.get("trajectory") or {}
    reward = sample.get("_reward_assignment")
    rows = [
        ("request_id", sample.get("request_id")),
        ("sample index", sample.get("index") or traj.get("sample_index")),
        ("rollout_n", traj.get("rollout_n")),
        ("step", traj.get("step")),
        ("dataset id", extra.get("id")),
        ("question", extra.get("question")),
        ("ground_truth", extra.get("ground_truth") or (sample.get("reward_model") or {}).get("ground_truth")),
        ("answer", extra.get("answer")),
        ("num_turns", sample.get("num_turns")),
        ("tool_calls", len(sample.get("tool_rewards") or [])),
        ("reward_outcome", reward_outcome(reward)),
        ("score", reward_score(reward)),
        ("format", reward_extra(reward, "format")),
        ("accuracy", reward_extra(reward, "accuracy")),
        ("tool_call_bonus", reward_extra(reward, "tool_call_bonus")),
        ("tool_call_count", reward_extra(reward, "tool_call_count")),
        ("debug_dir", str(debug_dir)),
    ]
    body = "\n".join(
        f"<tr><th>{escape(str(k))}</th><td>{escape(str(v))}</td></tr>"
        for k, v in rows
        if v is not None
    )
    return f"<table>{body}</table>"


def render_images(paths: list[str], embed_images: bool) -> str:
    blocks = []
    seen = set()
    for path in paths:
        if not path or path in seen:
            continue
        seen.add(path)
        exists = Path(path).exists() if not path.startswith(("http://", "https://", "data:")) else True
        status = "" if exists else " missing"
        blocks.append(
            "<figure>"
            f'<a href="{escape(Path(path).as_posix())}">'
            f'<img src="{escape(image_src(path, embed_images))}" alt="{escape(path)}">'
            "</a>"
            f"<figcaption class=\"path{status}\">{escape(path)}</figcaption>"
            "</figure>"
        )
    return "<div class=\"images\">" + "\n".join(blocks) + "</div>" if blocks else ""


def render_message(role: str, text: str, image_paths: list[str], embed_images: bool, max_text_chars: int | None) -> str:
    text_html = f"<pre>{escape(shorten(text, max_text_chars))}</pre>" if text else ""
    images_html = render_images(image_paths, embed_images)
    return f'<section class="message {escape(role)}"><h3>{escape(role)}</h3>{text_html}{images_html}</section>'


def render_tool_response(event: dict[str, Any], embed_images: bool, max_text_chars: int | None) -> str:
    info = event.get("tool_info") or {}
    paths = flatten_paths(info.get("saved_images"))
    meta = {
        "tool": event.get("tool_name") or info.get("tool"),
        "assistant_turn": event.get("assistant_turn"),
        "label": info.get("label"),
        "bbox_2d": info.get("bbox_2d"),
        "pixel_bbox": info.get("pixel_bbox"),
        "source_image_size": info.get("source_image_size"),
        "cropped_image_size": info.get("cropped_image_size"),
        "success": info.get("success"),
    }
    meta_text = json.dumps({k: v for k, v in meta.items() if v is not None}, indent=2)
    text = (event.get("tool_response_text") or "") + "\n\n" + meta_text
    return render_message("tool response", text, paths, embed_images, max_text_chars)


def build_html(
    sample: dict[str, Any],
    events: list[dict[str, Any]],
    debug_dir: Path,
    embed_images: bool,
    max_text_chars: int | None,
) -> str:
    request_id = sample.get("request_id", "")
    blocks = [render_meta(sample, debug_dir)]

    for message in sample.get("raw_prompt") or sample.get("messages") or []:
        role = message.get("role", "message")
        content = message.get("content")
        blocks.append(
            render_message(
                role,
                text_from_content(content),
                image_paths_from_content(content),
                embed_images,
                max_text_chars,
            )
        )

    event_blocks = []
    for event in events:
        name = event.get("event")
        if name == "sample_start":
            continue
        if name == "assistant_turn":
            event_blocks.append(
                render_message(
                    f"assistant turn {event.get('assistant_turn')}",
                    event.get("text") or "",
                    [],
                    embed_images,
                    max_text_chars,
                )
            )
        elif name == "tool_call":
            text = json.dumps(event.get("tool_arguments") or event, indent=2, ensure_ascii=False)
            event_blocks.append(render_message("tool call", text, [], embed_images, max_text_chars))
        elif name == "tool_response":
            event_blocks.append(render_tool_response(event, embed_images, max_text_chars))
        elif name == "sample_final":
            # sample_traces has a richer final record; render that below instead.
            continue

    if event_blocks:
        blocks.append("<h2>Interleaved Events</h2>")
        blocks.extend(event_blocks)

    blocks.append("<h2>Final Transcript</h2>")
    blocks.append(render_message("final_response", sample.get("final_response") or "", [], embed_images, max_text_chars))

    image_refs = flatten_paths(sample.get("image_references")) + flatten_paths((sample.get("extra_info") or {}).get("images"))
    if image_refs:
        blocks.append("<h2>Referenced Original Images</h2>")
        blocks.append(render_images(image_refs, embed_images))

    css = """
    body { color: #1f2933; font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f8fa; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 24px; margin: 0 0 16px; }
    h2 { border-top: 1px solid #d0d7de; margin-top: 28px; padding-top: 20px; }
    h3 { margin: 0 0 10px; text-transform: uppercase; letter-spacing: .04em; font-size: 12px; color: #57606a; }
    table { border-collapse: collapse; width: 100%; background: white; margin: 12px 0 20px; }
    th, td { border: 1px solid #d0d7de; padding: 8px 10px; vertical-align: top; text-align: left; }
    th { width: 160px; background: #f0f3f6; }
    .message { background: white; border: 1px solid #d0d7de; border-radius: 8px; margin: 14px 0; padding: 14px; }
    .system { border-left: 5px solid #6e7781; }
    .user { border-left: 5px solid #0969da; }
    .assistant, .final_response { border-left: 5px solid #1a7f37; }
    .tool, .tool\\ response, .tool\\ call { border-left: 5px solid #bc4c00; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .images { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 12px; }
    figure { margin: 0; border: 1px solid #d0d7de; border-radius: 6px; overflow: hidden; background: #f6f8fa; }
    img { display: block; max-width: 100%; max-height: 620px; margin: 0 auto; object-fit: contain; background: #111; }
    figcaption { border-top: 1px solid #d0d7de; padding: 8px; color: #57606a; font: 12px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }
    .missing { color: #cf222e; }
    """
    title = f"Trajectory {request_id}"
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>{escape(title)}</title><style>{css}</style></head><body><main>"
        f"<h1>{escape(title)}</h1>"
        + "\n".join(blocks)
        + "</main></body></html>\n"
    )


def build_multi_sample_html(
    samples: list[dict[str, Any]],
    events_by_request_id: dict[str, list[dict[str, Any]]],
    debug_dir: Path,
    embed_images: bool,
    max_text_chars: int | None,
) -> str:
    if len(samples) == 1:
        request_id = str(samples[0].get("request_id"))
        return build_html(samples[0], events_by_request_id.get(request_id, []), debug_dir, embed_images, max_text_chars)

    sections = []
    nav_items = []
    for idx, sample in enumerate(samples, start=1):
        request_id = str(sample.get("request_id"))
        anchor = f"sample-{idx}"
        traj = sample.get("trajectory") or {}
        sample_index = sample.get("index") or traj.get("sample_index")
        rollout_n = traj.get("rollout_n")
        tool_count = len(sample.get("tool_rewards") or [])
        reward = sample.get("_reward_assignment")
        score = reward_score(reward)
        outcome = reward_outcome(reward)
        reward_label = ""
        if score is not None:
            reward_label = f", {outcome}, score {score:g}"
        nav_items.append(
            f'<li><a href="#{anchor}">Sample {idx}: '
            f'{escape(str(sample_index))} / rollout {escape(str(rollout_n))} '
            f'({tool_count} tools{escape(reward_label)})</a></li>'
        )
        sample_html = build_html(
            sample,
            events_by_request_id.get(request_id, []),
            debug_dir,
            embed_images,
            max_text_chars,
        )
        inner = sample_html.split("<main>", 1)[1].rsplit("</main>", 1)[0]
        sections.append(f'<section id="{anchor}" class="sample-section">{inner}</section>')

    css = """
    body { color: #1f2933; font: 14px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; margin: 0; background: #f6f8fa; }
    main { max-width: 1180px; margin: 0 auto; padding: 24px; }
    h1 { font-size: 24px; margin: 0 0 16px; }
    .sample-nav, .sample-section { background: white; border: 1px solid #d0d7de; border-radius: 8px; padding: 14px; margin: 14px 0 28px; }
    .sample-section { background: #f6f8fa; }
    .sample-section > h1 { border-top: 3px solid #0969da; padding-top: 18px; }
    h2 { border-top: 1px solid #d0d7de; margin-top: 28px; padding-top: 20px; }
    h3 { margin: 0 0 10px; text-transform: uppercase; letter-spacing: .04em; font-size: 12px; color: #57606a; }
    table { border-collapse: collapse; width: 100%; background: white; margin: 12px 0 20px; }
    th, td { border: 1px solid #d0d7de; padding: 8px 10px; vertical-align: top; text-align: left; }
    th { width: 160px; background: #f0f3f6; }
    .message { background: white; border: 1px solid #d0d7de; border-radius: 8px; margin: 14px 0; padding: 14px; }
    .system { border-left: 5px solid #6e7781; }
    .user { border-left: 5px solid #0969da; }
    .assistant, .final_response { border-left: 5px solid #1a7f37; }
    .tool, .tool\\ response, .tool\\ call { border-left: 5px solid #bc4c00; }
    pre { white-space: pre-wrap; overflow-wrap: anywhere; margin: 0; font: 13px/1.45 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; }
    .images { display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 14px; margin-top: 12px; }
    figure { margin: 0; border: 1px solid #d0d7de; border-radius: 6px; overflow: hidden; background: #f6f8fa; }
    img { display: block; max-width: 100%; max-height: 620px; margin: 0 auto; object-fit: contain; background: #111; }
    figcaption { border-top: 1px solid #d0d7de; padding: 8px; color: #57606a; font: 12px/1.35 ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; overflow-wrap: anywhere; }
    .missing { color: #cf222e; }
    """
    return (
        "<!doctype html><html><head><meta charset=\"utf-8\">"
        f"<title>Trajectory Samples</title><style>{css}</style></head><body><main>"
        f"<h1>Trajectory Samples ({len(samples)})</h1>"
        f'<nav class="sample-nav"><ol>{"".join(nav_items)}</ol></nav>'
        + "\n".join(sections)
        + "</main></body></html>\n"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--debug-log-dir", type=Path, required=True, help="Directory containing sample_traces.jsonl and agent_events.jsonl")
    parser.add_argument("--request-id", help="Exact rollout request_id to render")
    parser.add_argument("--sample-index", type=int, help="Dataset/sample index to render")
    parser.add_argument("--rollout-n", type=int, help="Disambiguate --sample-index by rollout number")
    parser.add_argument(
        "--select",
        choices=("longest", "most-tools", "first", "last", "random"),
        default="most-tools",
        help="Selection strategy when --request-id is omitted",
    )
    parser.add_argument("--output", type=Path, help="Output HTML path")
    parser.add_argument("--max-text-chars", type=int, default=0, help="Truncate long text blocks; 0 means no truncation")
    parser.add_argument("--num-samples", type=int, default=1, help="Number of samples to render when not selecting an exact request")
    parser.add_argument("--embed-images", action="store_true", help="Embed images as base64 data URIs; produces large HTML")
    parser.add_argument(
        "--outcome",
        choices=("all", "successful", "incorrect"),
        default="all",
        help=(
            "Filter by reward outcome using reward_assignments.jsonl. "
            "successful means score >= 1.0; incorrect means score < 1.0."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    debug_dir = args.debug_log_dir.expanduser().resolve()
    sample_path = debug_dir / "sample_traces.jsonl"
    agent_path = debug_dir / "agent_events.jsonl"
    reward_path = debug_dir / "reward_assignments.jsonl"
    if not sample_path.exists():
        raise SystemExit(f"Missing {sample_path}")

    samples = find_samples(
        sample_path,
        reward_path,
        args.request_id,
        args.sample_index,
        args.rollout_n,
        args.select,
        args.num_samples,
        args.outcome,
    )
    request_ids = [str(sample.get("request_id")) for sample in samples]
    events_by_request_id = events_for_requests(agent_path, request_ids)

    output = args.output
    if output is None:
        sample = samples[0]
        request_id = str(sample.get("request_id"))
        output = debug_dir / f"trajectory_{request_id}.html"
    output = output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    max_text_chars = args.max_text_chars if args.max_text_chars > 0 else None
    html = build_multi_sample_html(samples, events_by_request_id, debug_dir, args.embed_images, max_text_chars)
    output.write_text(html, encoding="utf-8")

    print(f"Wrote {output}")
    print(f"num_samples={len(samples)}")
    print(f"outcome={args.outcome}")
    for sample in samples:
        request_id = str(sample.get("request_id"))
        reward = sample.get("_reward_assignment")
        print(f"request_id={request_id}")
        print(f"sample_index={sample.get('index') or sample.get('trajectory', {}).get('sample_index')}")
        print(f"rollout_n={sample.get('trajectory', {}).get('rollout_n')}")
        print(f"tool_calls={len(sample.get('tool_rewards') or [])}")
        if reward:
            print(f"score={reward_score(reward)}")
            print(f"reward_outcome={reward_outcome(reward)}")
            print(f"format={reward_extra(reward, 'format')}")
            print(f"accuracy={reward_extra(reward, 'accuracy')}")


if __name__ == "__main__":
    main()
