#!/usr/bin/env python3
"""Convert hstarbench JSONL (test/RL/SFT) to verl format (single merged file per split)."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any, Iterable

TASK_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(TASK_DIR))

from utils import get_doc_ground_bboxes  # noqa: E402

_DEFAULT_BENCH_ROOT = Path("/scratch/gpfs/ZHUANGL/shared/Hstarbench")
_DEFAULT_OUT_DIR = _DEFAULT_BENCH_ROOT / "data" / "verl"
WEB_GROUNDING_POST_PROMPT = 'Output the point in a JSON array format: [{"point_2d": [x, y]}].'


def _load_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _natural_instruction(doc: dict[str, Any]) -> str:
    instruction = str(doc.get("task") or "").replace("？", "?").replace("，", ",")
    instruction = re.sub(r"\s+", " ", instruction).strip()
    instruction = re.sub(r"\s+([,.;:!?])", r"\1", instruction)
    if not instruction:
        raise ValueError(f"missing task for {doc.get('doc_id')}")
    context = re.split(r"\s*,?\s*(?=where(?:\b|')|which direction\b)", instruction, maxsplit=1, flags=re.I)[0]
    context_lower = context.lower()
    if "scooter across the street" in context_lower:
        return "Which direction should you take to safely reach the scooter across the street?"
    if re.fullmatch(r"if you queue up to buy (?:the )?tickets", context, flags=re.I):
        return "Where should you queue to buy the tickets?"
    if re.fullmatch(r"you have lost something that is important", context, flags=re.I):
        return "Which direction should you take to recover something important you lost?"
    if re.fullmatch(r"you have something to print", context, flags=re.I):
        return "Which direction should you take to print something?"
    context = re.sub(r"^you\s+wan\s+to\b", "you want to", context, flags=re.I)
    goal_patterns = (
        (r"^(?:if\s+)?you\s+want\s+to\s+(.+)$", "action"),
        (r"^you\s+need\s+to\s+(.+)$", "action"),
        (r"^you\s+want\s+((?:go|exit|leave|take|buy|enter|throw|order|cross|ride|find|move)\b.+)$", "action"),
        (r"^you\s+want\s+(.+)$", "object"),
        (r"^you\s+need\s+(.+)$", "object"),
        (r"^you\s+are\s+going\s+to\s+(.+)$", "destination"),
    )
    for pattern, kind in goal_patterns:
        match = re.fullmatch(pattern, context, flags=re.I)
        if not match:
            continue
        goal = match.group(1).strip(" ,.?!")
        goal = re.sub(
            r"^go to (restroom|toilet|bathroom|subway platform|opposite platform|check-in counter)\b",
            r"go to the \1",
            goal,
            flags=re.I,
        )
        goal = re.sub(r"^take (elevator|escalator|subway)\b", r"take the \1", goal, flags=re.I)
        if kind == "destination":
            return f"Which direction leads to {goal}?"
        if kind == "object":
            return f"Which direction should you take to get {goal}?"
        return f"Which direction should you take to {goal}?"
    find_match = re.fullmatch(r"find\s+(.+)", instruction, flags=re.I)
    if find_match:
        target = find_match.group(1).strip(" .")
        first_word = target.split(maxsplit=1)[0].lower()
        if first_word not in {"a", "an", "and", "the", "your"}:
            instruction = f"Find the {target}"
    instruction = re.sub(r"^look for area\b", "Look for the area", instruction, flags=re.I)
    instruction = instruction[0].upper() + instruction[1:]
    if instruction[-1] not in ".?!":
        instruction += "."
    return instruction


def _user_prompt(doc: dict[str, Any]) -> str:
    return f"<image>\n{_natural_instruction(doc)} {WEB_GROUNDING_POST_PROMPT}"


def _image_dimensions(doc: dict[str, Any]) -> tuple[int, int]:
    width = int(doc.get("strip_width") or 0)
    height = int(doc.get("strip_height") or 0)
    if width <= 0 or height <= 0:
        raise ValueError(f"missing strip dimensions for {doc.get('doc_id')}")
    return width, height


def _click_ground_truth(doc: dict[str, Any]) -> tuple[str, list[list[float]], int, int]:
    width, height = _image_dimensions(doc)
    normalized_boxes = [
        list(box)
        for box in get_doc_ground_bboxes(doc)
        if float(box[0]) < float(box[2]) and float(box[1]) < float(box[3])
    ]
    if not normalized_boxes:
        raise ValueError(f"no nondegenerate target boxes for {doc.get('doc_id')}")
    native_boxes = [
        [
            round(box[0] * width / 1000, 6),
            round(box[1] * height / 1000, 6),
            round(box[2] * width / 1000, 6),
            round(box[3] * height / 1000, 6),
        ]
        for box in normalized_boxes
    ]
    target = json.dumps(
        [{"bbox_2d": box, "label": "target"} for box in native_boxes],
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return target, normalized_boxes, width, height


def _to_verl_row(doc: dict[str, Any], *, split_tag: str) -> dict[str, Any]:
    strip_path = doc.get("panorama_strip_path") or doc.get("view_path")
    if not strip_path:
        raise ValueError(f"missing image path for {doc.get('doc_id')}")
    data_source = f"hstarbench_{doc.get('split', split_tag)}"
    prompt = _user_prompt(doc)
    target, normalized_boxes, width, height = _click_ground_truth(doc)
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": prompt}],
        "images": [strip_path],
        "ability": "search",
        "reward_model": {
            "style": "rule",
            "ground_truth": target,
        },
        "extra_info": {
            "split": split_tag,
            "index": int(doc.get("seed", 0)),
            "id": doc.get("doc_id"),
            "doc_id": doc.get("doc_id"),
            "reward_type": "clicking",
            "question": prompt,
            "answer": target,
            "original_instruction": _natural_instruction(doc),
            "image_sizes": [[width, height]],
            "normalize_bbox_to_1000": True,
            "acceptance_bboxes_normalized_1000": normalized_boxes,
            "coordinate_format": "point_2d_normalized_1000",
            "target_yaw": doc.get("target_yaw"),
            "target_pitch": doc.get("target_pitch"),
            "level": doc.get("level"),
            "eval_set": doc.get("eval_set"),
            "is_nav": doc.get("is_nav"),
        },
    }


def _write_jsonl(rows: Iterable[dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def export_verl(
    *,
    src_path: Path,
    out_dir: Path,
    split_tag: str,
    limit: int = 0,
) -> int:
    rows = _load_rows(src_path)
    if limit > 0:
        rows = rows[:limit]
    verl_rows = [_to_verl_row(doc, split_tag=split_tag) for doc in rows]

    stem = src_path.stem.replace("hstarbench_", "")
    out_path = out_dir / f"hstarbench_{stem}.verl.jsonl"
    n = _write_jsonl(verl_rows, out_path)
    print(f"Wrote {n} -> {out_path}")
    return n


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        choices=("rl", "sft", "test", "all"),
        default="rl",
        help="Which hstarbench JSONL split to convert",
    )
    p.add_argument("--bench-root", default=str(_DEFAULT_BENCH_ROOT))
    p.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR))
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    data_dir = Path(args.bench_root) / "data"
    out_dir = Path(args.out_dir)
    specs = {
        "rl": [
            ("mixed_rl", data_dir / "hstarbench_mixed_rl.jsonl"),
            ("hos_rl", data_dir / "hstarbench_hos_rl.jsonl"),
            ("hps_rl", data_dir / "hstarbench_hps_rl.jsonl"),
        ],
        "sft": [
            ("mixed_sft", data_dir / "hstarbench_mixed_sft.jsonl"),
            ("hos_sft", data_dir / "hstarbench_hos_sft.jsonl"),
            ("hps_sft", data_dir / "hstarbench_hps_sft.jsonl"),
        ],
        "test": [
            ("mixed_test", data_dir / "hstarbench_mixed_test.jsonl"),
            ("hos_test", data_dir / "hstarbench_hos_test.jsonl"),
            ("hps_test", data_dir / "hstarbench_hps_test.jsonl"),
        ],
    }
    chosen = specs["rl"] + specs["sft"] + specs["test"] if args.source == "all" else specs[args.source]
    for tag, src in chosen:
        if not src.is_file():
            print(f"skip missing {src}")
            continue
        print(f"=== {tag} ===")
        export_verl(
            src_path=src,
            out_dir=out_dir,
            split_tag=tag,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
