#!/usr/bin/env python3
"""Convert hstarbench JSONL (test/RL/SFT) to verl training/validation format."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from utils import hstarbench_bbox_zoom_doc_to_text, hstarbench_doc_to_target  # noqa: E402

_DEFAULT_BENCH_ROOT = Path("/scratch/gpfs/ZHUANGL/shared/Hstarbench")
_DEFAULT_OUT_DIR = _DEFAULT_BENCH_ROOT / "data" / "verl"


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _user_prompt(doc: Dict[str, Any], *, pre_prompt: str = "\n") -> str:
    question = hstarbench_bbox_zoom_doc_to_text(doc)
    return f"<image>\n{pre_prompt}{question}"


def _to_verl_row(doc: Dict[str, Any], *, split_tag: str) -> Dict[str, Any]:
    strip_path = doc.get("panorama_strip_path") or doc.get("view_path")
    if not strip_path:
        raise ValueError(f"missing image path for {doc.get('doc_id')}")
    data_source = f"hstarbench_{doc.get('split', split_tag)}"
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": _user_prompt(doc)}],
        "images": [strip_path],
        "ability": "search",
        "reward_model": {
            "style": "rule",
            "ground_truth": hstarbench_doc_to_target(doc),
        },
        "extra_info": {
            "split": split_tag,
            "index": int(doc.get("seed", 0)),
            "id": doc.get("doc_id"),
            "doc_id": doc.get("doc_id"),
            "reward_type": "hstar_strip",
            "target_yaw": doc.get("target_yaw"),
            "target_pitch": doc.get("target_pitch"),
            "level": doc.get("level"),
            "eval_set": doc.get("eval_set"),
            "is_nav": doc.get("is_nav"),
        },
    }


def _write_jsonl(rows: Iterable[Dict[str, Any]], out_path: Path) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with out_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            n += 1
    return n


def _split_train_val(rows: List[Dict[str, Any]], val_ratio: float, seed: int) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    shuffled = list(rows)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    if val_ratio <= 0:
        return shuffled, []
    val_n = max(1, int(len(shuffled) * val_ratio))
    return shuffled[val_n:], shuffled[:val_n]


def export_verl(
    *,
    src_path: Path,
    out_dir: Path,
    split_tag: str,
    val_ratio: float,
    seed: int,
    limit: int = 0,
) -> Dict[str, int]:
    rows = _load_rows(src_path)
    if limit > 0:
        rows = rows[:limit]
    verl_rows = [_to_verl_row(doc, split_tag=split_tag) for doc in rows]
    train_rows, val_rows = _split_train_val(verl_rows, val_ratio, seed)

    stem = src_path.stem.replace("hstarbench_", "")
    counts = {}
    if train_rows:
        train_path = out_dir / f"hstarbench_{stem}_train.verl.jsonl"
        counts["train"] = _write_jsonl(train_rows, train_path)
        print(f"Wrote {counts['train']} -> {train_path}")
    if val_rows:
        val_path = out_dir / f"hstarbench_{stem}_val.verl.jsonl"
        counts["val"] = _write_jsonl(val_rows, val_path)
        print(f"Wrote {counts['val']} -> {val_path}")
    full_path = out_dir / f"hstarbench_{stem}.verl.jsonl"
    counts["all"] = _write_jsonl(verl_rows, full_path)
    print(f"Wrote {counts['all']} -> {full_path}")
    return counts


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
    p.add_argument("--val-ratio", type=float, default=0.1, help="Hold-out ratio for val.verl.jsonl")
    p.add_argument("--seed", type=int, default=42)
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
            val_ratio=args.val_ratio,
            seed=args.seed,
            limit=args.limit,
        )


if __name__ == "__main__":
    main()
