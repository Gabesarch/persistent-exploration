#!/usr/bin/env python3
"""Turn collected GPT-5.5 ball-drop traces into a distillation SFT dataset.

Input : the per-doc trace JSONLs from collect_ball_drop_traces.py
Filter: keep only CORRECT traces (default --filter trainable = ran clean AND the
        boxed answer in the FINAL turn equals ground truth; the strict SFT gate).
Output: one multimodal multi-turn conversation per kept trace, with assistant turns
        rendered in this project's hermes style (<think>...</think>, <tool_call>{json}
        </tool_call>, <answer>...$\\boxed{N}$...</answer>), images addressed by img_idx
        and listed (in img_idx order) under a flat top-level `images`. This maps
        directly onto the student tool interface (Qwen3-VL / Vero) and converts
        cleanly to verl / ms-swift / LLaMA-Factory SFT loaders.

Usage:
  python build_distill_dataset.py
  python build_distill_dataset.py --filter correct --require-tools \
      --root /scratch/gpfs/ZHUANGL/lc1556/distillation/ball_drop
"""
from __future__ import annotations

import argparse
import glob
import json
import os

# hermes / Qwen tool-call + response markers (what the student is trained to emit/read).
THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
TOOLCALL_OPEN, TOOLCALL_CLOSE = "<tool_call>", "</tool_call>"
ANSWER_OPEN, ANSWER_CLOSE = "<answer>", "</answer>"


def _toolset_of(task: str) -> str:
    if task and task.endswith("_zoomdraw"):
        return "zoomdraw"
    if task and task.endswith("_draw"):
        return "draw"
    if task and task.endswith("_zoom"):
        return "zoom"
    return "unknown"


def _render_tool_call(tc: dict) -> str:
    """One hermes <tool_call> block from a recorded function call."""
    try:
        args = json.loads(tc["arguments"]) if isinstance(tc.get("arguments"), str) else (tc.get("arguments") or {})
    except Exception:
        args = {"_raw": tc.get("arguments")}
    payload = {"name": tc.get("name"), "arguments": args}
    return f"{TOOLCALL_OPEN}\n{json.dumps(payload, ensure_ascii=False)}\n{TOOLCALL_CLOSE}"


def build_conversation(rec: dict) -> dict | None:
    """Reconstruct a trainable multimodal conversation from one trace record."""
    turns = rec.get("turns") or []
    if not turns:
        return None
    registry = rec.get("image_registry") or {}
    # images in img_idx order; verify they resolve on disk
    ordered_idx = sorted(registry.keys(), key=lambda k: int(k))
    images = [registry[k] for k in ordered_idx]
    if not images or not all(os.path.isfile(p) for p in images):
        return None
    input_image = registry.get("0", rec.get("input_image_path"))

    messages = []
    if rec.get("system_prompt"):
        messages.append({"role": "system", "content": rec["system_prompt"]})
    messages.append({"role": "user", "content": [
        {"type": "image", "image": input_image, "img_idx": 0},
        {"type": "text", "text": rec.get("user_text", "")},
    ]})

    n = len(turns)
    for ti, turn in enumerate(turns):
        is_last = ti == n - 1
        think = (turn.get("reasoning_summary") or "").strip()
        msg = (turn.get("message_text") or "").strip()
        tcs = turn.get("tool_calls") or []

        parts = []
        if think:
            parts.append(f"{THINK_OPEN}\n{think}\n{THINK_CLOSE}")
        if tcs:
            if msg:
                parts.append(msg)
            parts.extend(_render_tool_call(tc) for tc in tcs)
        else:
            # answer turn: prefer the recorded final_text (guaranteed to carry the boxed N)
            answer_body = (rec.get("final_text") or msg).strip()
            parts.append(f"{ANSWER_OPEN}\n{answer_body}\n{ANSWER_CLOSE}")
        messages.append({"role": "assistant", "content": "\n".join(parts)})

        # one tool message per tool call (carries the returned image by img_idx)
        for tc in tcs:
            content = [{"type": "text", "text": tc.get("result_text", "")}]
            if tc.get("result_image_path") and os.path.isfile(tc["result_image_path"]):
                content.append({"type": "image", "image": tc["result_image_path"],
                                "img_idx": tc.get("result_img_idx")})
            messages.append({"role": "tool", "content": content})

    return {
        "id": f"{rec.get('task')}::{rec.get('doc_id')}",
        "source_task": rec.get("task"),
        "toolset": _toolset_of(rec.get("task")),
        "teacher_model": rec.get("teacher_model"),
        "reasoning_effort": rec.get("reasoning_effort"),
        "ground_truth": rec.get("gt"),
        "category": rec.get("category"),
        "answer": rec.get("final_text"),
        "images": images,
        "messages": messages,
        "num_turns": rec.get("num_turns"),
        "num_tool_calls": rec.get("num_tool_calls"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/scratch/gpfs/ZHUANGL/lc1556/distillation/ball_drop")
    ap.add_argument("--traces-glob", default=None, help="default: <root>/traces/*.jsonl")
    ap.add_argument("--filter", choices=["trainable", "correct"], default="trainable",
                    help="trainable = correct AND clean AND final-turn boxed==gt (default)")
    ap.add_argument("--require-tools", action="store_true",
                    help="keep only trajectories that called at least one tool")
    ap.add_argument("--output", default=None, help="default: <root>/sft/<root-name>_distill.jsonl")
    args = ap.parse_args()

    name = os.path.basename(args.root.rstrip("/")) or "distill"
    traces_glob = args.traces_glob or os.path.join(args.root, "traces", "*.jsonl")
    out_path = args.output or os.path.join(args.root, "sft", f"{name}_distill.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    out_base = os.path.splitext(os.path.basename(out_path))[0]

    files = sorted(glob.glob(traces_glob))
    if not files:
        raise SystemExit(f"no trace files matched {traces_glob}")

    stats = {"files": files, "filter": args.filter, "require_tools": args.require_tools,
             "seen": 0, "kept": 0, "dropped_filter": 0, "dropped_no_tools": 0,
             "dropped_unbuildable": 0, "by_toolset": {}, "tool_calls_kept": 0,
             "with_tools_kept": 0}
    examples = []
    with open(out_path, "w") as out:
        for f in files:
            for line in open(f):
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                stats["seen"] += 1
                keep = rec.get("trainable") if args.filter == "trainable" else rec.get("correct")
                if not keep:
                    stats["dropped_filter"] += 1
                    continue
                if args.require_tools and not (rec.get("num_tool_calls") or 0) > 0:
                    stats["dropped_no_tools"] += 1
                    continue
                conv = build_conversation(rec)
                if conv is None:
                    stats["dropped_unbuildable"] += 1
                    continue
                out.write(json.dumps(conv, ensure_ascii=False) + "\n")
                stats["kept"] += 1
                ts = conv["toolset"]
                stats["by_toolset"][ts] = stats["by_toolset"].get(ts, 0) + 1
                stats["tool_calls_kept"] += conv.get("num_tool_calls") or 0
                stats["with_tools_kept"] += int((conv.get("num_tool_calls") or 0) > 0)
                if len(examples) < 3 and (conv.get("num_tool_calls") or 0) > 0:
                    examples.append(conv)

    with open(os.path.join(os.path.dirname(out_path), f"{out_base}.stats.json"), "w") as fh:
        json.dump(stats, fh, indent=2)

    # human-readable preview of a few rendered conversations
    preview_path = os.path.join(os.path.dirname(out_path), f"{out_base}.preview.md")
    with open(preview_path, "w") as fh:
        fh.write(f"# Distillation dataset preview\n\n`{out_path}`\n\nStats: {json.dumps(stats, indent=2)}\n\n")
        for ex in examples:
            fh.write(f"\n---\n## {ex['id']}  (gt {ex['ground_truth']}, "
                     f"{ex['num_tool_calls']} tool calls, toolset {ex['toolset']})\n\n")
            fh.write(f"images ({len(ex['images'])}): {[os.path.basename(p) for p in ex['images']]}\n\n")
            for m in ex["messages"]:
                role = m["role"]
                if isinstance(m["content"], str):
                    body = m["content"]
                else:
                    body = " ".join(
                        (c.get("text", "") if c.get("type") == "text" else f"[IMAGE img_idx={c.get('img_idx')} {os.path.basename(c.get('image',''))}]")
                        for c in m["content"]
                    )
                if len(body) > 1200:
                    body = body[:1200] + " …(truncated)"
                fh.write(f"**{role}:** {body}\n\n")

    print("[done]", json.dumps(stats, indent=2))
    print("[out]", out_path)
    print("[preview]", preview_path)


if __name__ == "__main__":
    main()
