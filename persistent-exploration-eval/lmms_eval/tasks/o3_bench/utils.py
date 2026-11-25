"""Pinned O3-Bench test split and official MCQ extraction with the local judge."""
from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path, PurePosixPath
import re

from datasets import Dataset, DatasetDict
from huggingface_hub import hf_hub_download
from PIL import Image

from lmms_eval.tasks._task_utils.vllm_judge import get_judge_engine

REPO_ID = "m-Just/O3-Bench"
REVISION = "478bd2150344654eaddb49a9473206a9698dec8f"
EXPECTED_COUNTS = {"map": 182, "chart": 163}
FAILURE_MARKERS = (
    "[LMMS_EVAL_FORCED_FINAL_ANSWER_FAILED:",
    "[LMMS_EVAL_BUDGET_FORCING_FAILED:",
    "[LMMS_EVAL_REQUEST_FAILED",
    "[LMMS_EVAL_BUDGET_EXCEEDED]",
    "[MAX_TOOL_ITERATIONS_REACHED]",
)


def image_filename(row):
    path = PurePosixPath(row["file_name"])
    if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != "images":
        raise ValueError(f"Invalid O3-Bench image path: {path}")
    return f"test/{path.as_posix()}"


def load_rows(metadata_path):
    rows = [json.loads(line) for line in Path(metadata_path).read_text().splitlines() if line.strip()]
    if Counter(row["subset"] for row in rows) != Counter(EXPECTED_COUNTS):
        raise ValueError("O3-Bench split counts differ from the pinned 345-question test set")
    seen = set()
    for row in rows:
        image_filename(row)
        letters = re.findall(r"^([A-F])\. ", row["options"], flags=re.MULTILINE)
        # Question 223 has five choices (A-E); all other questions have A-F.
        if letters not in (list("ABCDE"), list("ABCDEF")) or row["answer"] not in letters or not row["question"].strip():
            raise ValueError("Invalid O3-Bench question/options/answer")
        identity = (row["file_name"], row["question"], row["options"])
        if identity in seen:
            raise ValueError("Duplicate O3-Bench question")
        seen.add(identity)
    return rows


def o3_bench_builder():
    metadata = hf_hub_download(REPO_ID, "test/metadata.jsonl", repo_type="dataset", revision=REVISION)
    # Explanations, target_layouts and layout_infos are privileged annotations.
    rows = [
        {"doc_id": f"o3_bench_test_{i:03d}", **{key: row[key] for key in
         ("file_name", "subset", "question", "options", "answer")}}
        for i, row in enumerate(load_rows(metadata))
    ]
    return DatasetDict({"test": Dataset.from_list(rows)})


def o3_bench_doc_to_visual(doc):
    path = hf_hub_download(REPO_ID, image_filename(doc), repo_type="dataset", revision=REVISION)
    # Missing/corrupt images must fail, never become text-only evaluations.
    with Image.open(path) as image:
        return [image.convert("RGB")]


def o3_bench_doc_to_text(doc, lmms_eval_specific_kwargs=None):
    kwargs = lmms_eval_specific_kwargs or {}
    return kwargs.get("pre_prompt", "") + doc["question"] + "\n" + doc["options"] + kwargs.get("post_prompt", "")


def o3_bench_doc_to_target(doc):
    return doc["answer"]


def final_answer(raw):
    """Official extraction: last answer tag, then last boxed answer.

    async_openai returns a tool transcript; first isolate its final assistant
    message so an earlier rejected candidate can never receive credit.
    """
    text = str(raw or "")
    for marker in ("</tool_response>", "</tool_call>"):
        if marker in text:
            text = text.rsplit(marker, 1)[-1]
    matches = re.findall(r"<answer>(.*?)</answer>", text, re.DOTALL)
    text = matches[-1].strip() if matches else text.strip()
    matches = re.findall(r"\\boxed\{(.*?)\}", text, re.DOTALL)
    return matches[-1].strip() if matches else text


def o3_bench_process_results(doc, results):
    raw = str(results[0] or "") if results else ""
    return {"judge_accuracy": {
        "doc_id": doc["doc_id"], "subset": doc["subset"],
        "question": doc["question"], "options": doc["options"],
        "reference": doc["answer"], "prediction": final_answer(raw),
        "raw_prediction": raw,
        "generation_failed": not raw.strip() or any(marker in raw for marker in FAILURE_MARKERS),
    }}


def o3_bench_aggregate_results(results):
    # Exact same MCQA extraction prompt as InSight-o3 at
    # 7de6458e749c748177b29f1a979fdf881265d53a, prompts.py.
    template = (Path(__file__).parent / "mcqa_extraction_prompt.txt").read_text()
    pending, prompts = [], []
    for item in results:
        item.update(extracted_answer="", judge_output="", extraction_method="generation_failed", score=0.0)
        if item["generation_failed"]:
            continue
        if item["prediction"] == item["reference"]:
            item.update(extracted_answer=item["prediction"], extraction_method="exact", score=1.0)
            continue
        prediction = item["prediction"]
        if len(prediction) > 1000:
            prediction = "[cropped] ... " + prediction[-1000:]
        pending.append(item)
        prompts.append(template.format(question=item["question"], options=item["options"], model_answer=prediction))
    if prompts:
        outputs = get_judge_engine(os.getenv("JUDGE_MODEL_PATH", "")).generate_json_batch(
            prompts, max_tokens=2048, use_tqdm=True,
        )
        if len(outputs) != len(pending):
            raise RuntimeError("O3-Bench judge returned incomplete coverage")
        for item, output in zip(pending, outputs):
            answer = output.strip()
            item.update(judge_output=output, extracted_answer=answer, extraction_method="qwen35_27b",
                        score=float(answer == item["reference"]))
    details = {item["doc_id"]: dict(item) for item in results}
    o3_bench_aggregate_results.details = list(details.values())
    o3_bench_aggregate_results.individual_scores = details
    o3_bench_aggregate_results.subset_scores = {
        subset: {"correct": sum(item["score"] for item in results if item["subset"] == subset),
                 "total": sum(item["subset"] == subset for item in results)}
        for subset in EXPECTED_COUNTS
    }
    return sum(item["score"] for item in results) / len(results) if results else 0.0
