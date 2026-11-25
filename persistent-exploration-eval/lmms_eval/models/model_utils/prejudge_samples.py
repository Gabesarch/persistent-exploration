"""Persist generated samples before a local judge can fail during aggregation."""
import json
import os
from pathlib import Path
import re
import time


def save_prejudge_samples(task_outputs, output_path, *, serializer=str):
    """Write a separate, unscored snapshot; never create final result artifacts."""
    if not output_path:
        raise ValueError("Pre-judge snapshots require an output path")
    job_id = re.sub(r"[^A-Za-z0-9_.-]", "_", os.getenv("SLURM_JOB_ID", "local"))
    root = Path(output_path) / "prejudge_samples" / f"job_{job_id}_{time.time_ns()}"
    root.mkdir(parents=True, exist_ok=False)
    manifest = {
        "status": "generation_complete_judging_pending",
        "checkpoint_path": os.getenv("CHECKPOINT_PATH"),
        "slurm_job_id": os.getenv("SLURM_JOB_ID"),
        "tasks": [],
    }
    for task in task_outputs:
        name = re.sub(r"[^A-Za-z0-9_.-]", "_", task.task_name)
        path = root / f"prejudge_samples_{name}.jsonl"
        temporary = path.with_suffix(".jsonl.tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            for sample in task.logged_samples:
                handle.write(json.dumps(sample, ensure_ascii=False, default=serializer) + "\n")
        temporary.replace(path)
        manifest["tasks"].append({
            "task": task.task_name, "file": path.name,
            "num_samples": len(task.logged_samples), "task_config": task.task_config,
        })
    path = root / "manifest.json"
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2, default=serializer) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path
