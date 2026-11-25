#!/usr/bin/env python3
"""Export the held-out VisualProbe splits to the verl validation format."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path, PurePosixPath


SPLITS = (
    ("easy", "Mini-o3/VisualProbe_Easy", "0e665a57946d4473deb9975a13f8ddba973b8026", 141),
    ("medium", "Mini-o3/VisualProbe_Medium", "25d5ff5bce48147be34fd2655617ab17dc55b3fa", 268),
    ("hard", "Mini-o3/VisualProbe_Hard", "bbe9b7a6c41d49844f45e51916ec0a60d1f0e378", 106),
)
OUTPUT_NAME = "visual_probe_validation.verl.jsonl"


def export_validation(snapshots: dict[str, Path], output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / OUTPUT_NAME
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    index = 0
    seen_ids: set[str] = set()
    try:
        with temporary_path.open("w", encoding="utf-8") as output:
            for difficulty, _, revision, expected_count in SPLITS:
                snapshot = snapshots[difficulty]
                source_rows = json.loads((snapshot / "val.json").read_text(encoding="utf-8"))
                if not isinstance(source_rows, list) or len(source_rows) != expected_count:
                    raise ValueError(f"{difficulty}: expected {expected_count} rows")
                for source_index, source in enumerate(source_rows):
                    doc_id = source["doc_id"]
                    if not isinstance(doc_id, str) or doc_id in seen_ids:
                        raise ValueError(f"{difficulty} row {source_index}: invalid or repeated doc_id")
                    seen_ids.add(doc_id)
                    if source["data_source"] != f"visual_probe_{difficulty}":
                        raise ValueError(f"{difficulty} row {source_index}: unexpected data_source")
                    if not isinstance(source["images"], list) or len(source["images"]) != 1:
                        raise ValueError(f"{difficulty} row {source_index}: expected one image")
                    image_ref = PurePosixPath(source["images"][0])
                    if image_ref.is_absolute() or ".." in image_ref.parts or image_ref.parts[:2] != (f"VisualProbe_{difficulty.title()}", "data"):
                        raise ValueError(f"{difficulty} row {source_index}: unsafe image path")
                    image_source = snapshot / "data" / image_ref.name
                    if not image_source.is_file():
                        raise FileNotFoundError(image_source)
                    image_target = output_dir / "images" / difficulty / image_ref.name
                    image_target.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(image_source, image_target)
                    question = source["problem"]
                    answer = source["solution"]
                    if not isinstance(question, str) or not isinstance(answer, str):
                        raise ValueError(f"{difficulty} row {source_index}: invalid question or answer")
                    answer = answer.strip()
                    row = {
                        "data_source": "visual_probe",
                        "prompt": [{"role": "user", "content": question}],
                        "images": [str(image_target)],
                        "ability": "search",
                        "reward_model": {"style": "rule", "ground_truth": answer},
                        "extra_info": {
                            "split": "validation", "index": index, "answer": answer,
                            "question": question, "tolerance": None, "id": doc_id,
                            "reward_type": "search", "difficulty": difficulty,
                            "source_dataset": source["data_source"],
                            "source_index": source_index, "source_revision": revision,
                        },
                    }
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    index += 1
        temporary_path.replace(output_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output_dir", type=Path, help="Destination outside this Git checkout")
    parser.add_argument("--cache-dir", type=Path, help="Optional Hugging Face Hub cache directory")
    parser.add_argument("--local-files-only", action="store_true", help="Use already cached datasets")
    args = parser.parse_args()
    from huggingface_hub import snapshot_download

    snapshots = {
        difficulty: Path(snapshot_download(
            repo_id=repo_id, repo_type="dataset", revision=revision,
            allow_patterns=["val.json", "data/*.jpg"],
            cache_dir=str(args.cache_dir) if args.cache_dir else None,
            local_files_only=args.local_files_only,
        ))
        for difficulty, repo_id, revision, _ in SPLITS
    }
    output = export_validation(snapshots, args.output_dir)
    print(f"Wrote {sum(1 for _ in output.open(encoding='utf-8'))} rows to {output}")


if __name__ == "__main__":
    main()
