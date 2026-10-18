import json
from pathlib import Path
from typing import Any, Optional

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from lmms_eval.tasks.visual_probe import utils as visual_probe_utils
from PIL import Image as PILImage

PINPOINT_BENCH_REPO = "godx7/Pinpoint-Bench"
PINPOINT_BENCH_METADATA = "val.json"
_REQUIRED_FIELDS = {"images", "doc_id", "problem", "solution", "data_source"}


def _load_metadata_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        rows = json.load(handle)

    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{PINPOINT_BENCH_METADATA} must contain a non-empty JSON list.")

    seen_doc_ids = set()
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Pinpoint-Bench row {index} must be a JSON object.")
        missing = sorted(_REQUIRED_FIELDS.difference(row))
        if missing:
            raise ValueError(f"Pinpoint-Bench row {index} is missing required fields: {', '.join(missing)}")
        if not isinstance(row["images"], list) or not row["images"]:
            raise ValueError(f"Pinpoint-Bench row {index} must contain at least one image.")

        doc_id = str(row["doc_id"])
        if doc_id in seen_doc_ids:
            raise ValueError(f"Pinpoint-Bench contains duplicate doc_id {doc_id!r}.")
        seen_doc_ids.add(doc_id)

    return rows


def pinpoint_bench_builder() -> DatasetDict:
    """Build the validation split without parsing the malformed Hub card schema."""
    metadata_path = hf_hub_download(
        repo_id=PINPOINT_BENCH_REPO,
        filename=PINPOINT_BENCH_METADATA,
        repo_type="dataset",
    )
    rows = _load_metadata_rows(Path(metadata_path))
    return DatasetDict({"validation": Dataset.from_list(rows)})


def _open_image(path: Path) -> Optional[PILImage.Image]:
    if not path.is_file():
        return None
    try:
        with PILImage.open(path) as image:
            return image.convert("RGB")
    except Exception:
        return None


def _coerce_image(image: Any) -> Optional[PILImage.Image]:
    if isinstance(image, PILImage.Image):
        return image.convert("RGB")
    if not isinstance(image, str):
        return None

    normalized = image.replace("\\", "/").lstrip("./")
    for candidate in (Path(image), Path(normalized)):
        loaded = _open_image(candidate)
        if loaded is not None:
            return loaded

    try:
        local_path = hf_hub_download(
            repo_id=PINPOINT_BENCH_REPO,
            filename=normalized,
            repo_type="dataset",
        )
    except Exception:
        return None
    return _open_image(Path(local_path))


def pinpoint_bench_doc_to_visual(doc: dict[str, Any]) -> list[PILImage.Image]:
    visuals = []
    for image in doc.get("images", []):
        coerced = _coerce_image(image)
        if coerced is not None:
            visuals.append(coerced)
    return visuals


# Pinpoint-Bench deliberately uses VisualProbe's exact prompt construction,
# answer extraction, judge prompts, failure handling, and detail reporting.
pinpoint_bench_doc_to_text = visual_probe_utils.visual_probe_doc_to_text
pinpoint_bench_doc_to_target = visual_probe_utils.visual_probe_doc_to_target
pinpoint_bench_process_results = visual_probe_utils.visual_probe_process_results
pinpoint_bench_process_results_json = visual_probe_utils.visual_probe_process_results_json
pinpoint_bench_aggregate_judge_results = visual_probe_utils.visual_probe_aggregate_judge_results
pinpoint_bench_aggregate_judge_results_strict = visual_probe_utils.visual_probe_aggregate_judge_results_strict
