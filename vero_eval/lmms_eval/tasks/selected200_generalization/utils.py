"""Read the existing MME-RealWorld-Lite cache without rebuilding or downloading."""
import os
from pathlib import Path

from datasets import DatasetDict, load_from_disk


def mme_realworld_lite_cached_builder():
    cache = Path(os.environ.get("HF_HOME", "/scratch/gpfs/ZHUANGL/gs1693/hf_cache"))
    path = cache / "datasets/yifanzhang114__MME-RealWorld-lite-lmms-eval/default/train"
    train = load_from_disk(str(path))
    if len(train) != 1919:
        raise ValueError(f"Expected 1919 MME-RealWorld-Lite samples in {path}, got {len(train)}")
    return DatasetDict({"train": train})
