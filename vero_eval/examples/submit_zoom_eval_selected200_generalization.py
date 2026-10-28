"""Submit the frozen selected200 generalization profiles through run_eval.sbatch.

One job per condition: Pinpoint and VStar by default; HRBench-8K is opt-in.
Configuration is deliberately
independent of ambient training/evaluation overrides. No jobs in --preflight or
--dry-run mode. The restart's local step 120 is cumulative training step 200.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import shlex
import subprocess
import sys


EVAL = Path(__file__).resolve().parents[1]
CONFIG = EVAL / "examples/configs/selected200_generalization.json"
# run_eval exports ALL: explicitly remove inherited configuration that could
# change sampling, prompt, critic, scheduling, or checkpoint conversion behavior.
UNSET = (
    "VERO_TRAINING_CONFIG", "VERO_EVAL_DISABLE_CRITIC", "VERO_EVAL_CRITIC_THRESHOLD",
    "VERO_EVAL_CRITIC_DOMAIN_THRESHOLDS", "CUSTOM_CHAT_TEMPLATE", "MODEL_ARGS",
    "VLLM_SERVER_JUDGE", "VLLM_JUDGE_BASE_URL", "VLLM_JUDGE_PORT",
    "VLLM_POLICY_CUDA_VISIBLE_DEVICES", "VLLM_JUDGE_CUDA_VISIBLE_DEVICES",
    "API_KEY", "MODEL_NAME", "MODEL_VERSION", "OUTPUT_PATH", "BATCH_SIZE",
    "MM_MIN_PIXELS", "MAX_IMAGE_PIXELS", "USE_CACHE", "GEN_KWARGS", "LMMS_EVAL_JUDGE_MODE",
    "FORCED_TOOL_PREFILL_ENABLED", "RETRY_MODE", "FINAL_PREFILL_MODE",
    "REPLAY_CRITIC_SYSTEM_PROMPT", "REPLAY_CRITIC_TEMPERATURE",
    "REPLAY_CRITIC_TOP_K", "REPLAY_CRITIC_TOP_P", "TRAINING_SYSTEM_PROMPT",
)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def resolve(path):
    return str((EVAL / path).resolve()) if path else ""


def verify_weights(root):
    root = Path(root)
    if not (root / "config.json").is_file() or not (root / "tokenizer_config.json").is_file():
        raise ValueError(f"Missing model/tokenizer metadata: {root}")
    indexes = list(root.glob("*.safetensors.index.json")) + list(root.glob("pytorch_model.bin.index.json"))
    if indexes:
        for index in indexes:
            metadata = json.loads(index.read_text())
            files = [root / name for name in set(metadata["weight_map"].values())]
            if not files or any(not p.is_file() or p.stat().st_size == 0 for p in files):
                raise ValueError(f"Incomplete indexed weights: {index}")
            if sum(p.stat().st_size for p in files) < metadata.get("metadata", {}).get("total_size", 0):
                raise ValueError(f"Truncated weights: {index}")
    elif not any(p.stat().st_size > 0 for pattern in ("*.safetensors", "pytorch_model*.bin") for p in root.glob(pattern)):
        raise ValueError(f"No complete Hugging Face weights: {root}; merge separately before submission")


def verify_datasets(config):
    cache = Path(config["root_path"]) / "hf_cache"
    expected = {b["name"]: b["count"] for b in config["benchmarks"]}
    pin = cache / "hub/datasets--godx7--Pinpoint-Bench"
    revision = (pin / "refs/main").read_text().strip()
    metadata = pin / "snapshots" / revision / "val.json"
    rows = json.loads(metadata.read_text())
    if len({str(r["doc_id"]) for r in rows}) != len(rows):
        raise ValueError("Duplicate Pinpoint doc IDs")
    if any(not (metadata.parent / image).is_file() for row in rows for image in row["images"]):
        raise ValueError("Missing cached Pinpoint images")
    counts = {"pinpoint_bench": len(rows)}
    evidence = {"pinpoint_bench": str(metadata)}
    specs = [
        ("mme_realworld_lite", "yifanzhang114__MME-RealWorld-lite-lmms-eval/default/train/dataset_info.json", "train"),
        ("vstar_bench", "lmms-lab___vstar-bench/*/*/*/dataset_info.json", "test"),
        ("hrbench_8k", "DreamMr___hr-bench/*/*/*/dataset_info.json", "hrbench_8k"),
        ("hrbench_4k", "DreamMr___hr-bench/*/*/*/dataset_info.json", "hrbench_4k"),
    ]
    for name, pattern, split in specs:
        if name not in expected:
            continue
        paths = sorted((cache / "datasets").glob(pattern))
        found = [(p, json.loads(p.read_text())["splits"][split]["num_examples"]) for p in paths]
        if not found or len({n for _, n in found}) != 1:
            raise ValueError(f"Missing or ambiguous cached sample count: {name}")
        evidence[name], counts[name] = str(found[0][0]), found[0][1]
    if counts != expected:
        raise ValueError(f"Dataset counts drifted: {counts} != {expected}")
    return counts, evidence


def make_job(config, condition, output_root, time_override=None):
    c = condition
    tasks = [f'{b["name"]}_selected200_{c["task_family"]}' for b in config["benchmarks"]]
    samples = sum(b["count"] for b in config["benchmarks"])
    seconds_per_sample = c["reference_seconds"] / c["reference_samples"]
    projected_hours = samples * seconds_per_sample / 3600
    hours = math.ceil(projected_hours * c["walltime_factor"] + 0.5)
    time = time_override or f"{hours}:00:00"
    identity = "global_step_200" if c["step"] else "baseline"
    out = Path(output_root).resolve() / c["key"] / identity
    prompt = resolve(c["system_prompt"])
    template = resolve(c["chat_template"])
    env = {
        "ROOT_PATH": config["root_path"], "MODEL": "vero", "SUFFIX": "reasoning_zoom",
        "CHECKPOINT_PATH": c["checkpoint"], "TASKS": ",".join(tasks),
        "TOOLS": "on" if c["tools"] else "off", "MCP_SERVER_PATH": "lmms_eval/mcp/vero2_mcp.py" if c["tools"] else "",
        "MAX_TOOL_ITERATIONS": str(c["max_tool_iterations"]), "MAX_TOOL_IMAGES": str(c["max_tool_iterations"]),
        "MM_MAX_PIXELS": "10000000", "MM_MIN_PIXELS": "", "NUM_GPUS": "1", "WORKERS": "10",
        "MAX_MODEL_LEN": "32768", "LIMIT": "", "LMMS_EVAL_USE_CACHE": "False",
        "LMMS_EVAL_SAVE_PREJUDGE_SAMPLES": "true", "LMMS_EVAL_HF_TOKENIZER": c["checkpoint"],
        "LMMS_EVAL_HF_TOKENIZER_LOCAL_ONLY": "1", "BATCH_SIZE": "1",
        "OUTPUT_ROOT": str(out), "LOG_DIR": str(out / "runtime_logs"),
        "TOOL_CALL_PARSER": c["tool_parser"], "CHAT_TEMPLATE": template,
        "SYSTEM_PROMPT": prompt, "TRAINING_SYSTEM_PROMPT": prompt,
        "RUN_SYSTEM_PROMPT_MODE": "run_env" if prompt else ("task_metadata" if c["tools"] else "none"),
        "FORCE_TOOL_CALLS_UNTIL_FINAL_TURN": str(c["force_tools"]).lower(),
        "FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS": str(c["force_final"]).lower(),
        "FORCE_FINAL_ANSWER_PREFIX": "", "FORCE_FINAL_ANSWER_THINK_PREFILL": c["final_prefill"],
        "ALWAYS_RETRY_ANSWERS": str(c["always_retry"]).lower(),
        "CRITIC_GUIDED_ANSWER_RETRY": str(c["critic"]).lower(),
        "CRITIC_GUIDED_ANSWER_RETRY_SOURCE": "policy", "CRITIC_GUIDED_ANSWER_RETRY_MODE": "token",
        "CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD": "0.5", "CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE": "persist",
        "FIRST_CANDIDATE_ANSWER_TURN": "0", "FOLLOWUP_CANDIDATE_ANSWER_TURN": "0",
        "RECORD_ANSWER_CANDIDATES": str(c["always_retry"]).lower(),
        "JUDGE_BACKEND": "engine", "API_TYPE": "vllm", "JUDGE_MODEL_PATH": config["judge"],
        "VLLM_JUDGE_MODEL_PATH": config["judge"], "VLLM_JUDGE_CHAT_TEMPLATE": config["judge"] + "/chat_template_no_think.jinja",
        "CHARXIV_JUDGE_TENSOR_PARALLEL_SIZE": "1", "VLLM_USE_FLASHINFER_SAMPLER": "0",
        "CHARXIV_JUDGE_DTYPE": "bfloat16", "CHARXIV_JUDGE_MAX_MODEL_LEN": "18432",
        "VLLM_JUDGE_MAX_NUM_SEQS": "32", "CHARXIV_JUDGE_MAX_TOKENS": "2048",
        "CHARXIV_JUDGE_TEMPERATURE": "0", "CHARXIV_JUDGE_TOP_P": "0.8", "CHARXIV_JUDGE_TOP_K": "20",
        "CHARXIV_JUDGE_PRESENCE_PENALTY": "0", "CHARXIV_JUDGE_GPU_MEMORY_UTILIZATION": "0.9",
        "VLLM_JUDGE_ENFORCE_EAGER": "true", "VLLM_JUDGE_GDN_PREFILL_BACKEND": "triton",
        "MERGE_BACKEND": "fsdp", "OVERRIDE_EXISTING": "true",
    }
    # Bind duplicate identity to protocol content and templates, not truncated labels.
    protocol_hash = hashlib.sha256(json.dumps({"condition": c, "env": env,
        "tasks": {t: digest(EVAL / "lmms_eval/tasks/selected200_generalization" / f"{t}.yaml") for t in tasks},
        "prompt_hash": digest(prompt) if prompt else None,
        "template_hash": digest(template) if template else None,
    }, sort_keys=True).encode()).hexdigest()[:12]
    job_name = f"selected200-{c['key'][:36]}-{protocol_hash}"
    # Different conditions get distinct ports even before Slurm assigns a node.
    port_offset = int(protocol_hash, 16) % 18000
    env.update(PORT=str(15000 + port_offset), MASTER_PORT=str(35000 + port_offset))
    return dict(condition=c["key"], label=c["label"], step=c["step"], local_step=c["local_step"],
                tasks=tasks, sample_counts={t: b["count"] for t, b in zip(tasks, config["benchmarks"])},
                samples=samples, seconds_per_sample=seconds_per_sample,
                projected_hours=projected_hours, walltime_factor=c["walltime_factor"],
                walltime=time, reference_result=c["reference_result"], timing_source=c["timing_source"],
                timing_kind=c["timing_kind"], job_name=job_name, env=env,
                prompt_sha256=digest(prompt) if prompt else None,
                template_sha256=digest(template) if template else None)


def complete(job):
    plan = Path(job["env"]["OUTPUT_ROOT"]) / "submission_plan.json"
    if plan.exists() and json.loads(plan.read_text()).get("job_name") != job["job_name"]:
        raise ValueError(f"Existing output uses a different protocol: {plan}; choose a new --output-root")
    for path in Path(job["env"]["OUTPUT_ROOT"]).rglob("*_results.json"):
        try:
            data = json.loads(path.read_text())
            if data.get("config", {}).get("limit") is not None:
                continue
            if all(t in data.get("results", {}) and data.get("n-samples", {}).get(t, {}).get("effective") == n
                   for t, n in job["sample_counts"].items()):
                return True
        except (OSError, ValueError, TypeError):
            continue
    return False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=CONFIG, help="condition registry JSON (default: frozen selected200 suite)")
    parser.add_argument("--include-hrbench-8k", action="store_true", help="include HRBench-8K from the registry (excluded by default)")
    parser.add_argument("--dry-run", action="store_true", default=os.getenv("DRY_RUN") == "1")
    parser.add_argument("--preflight", action="store_true", help="validate all selected inputs; never submit")
    parser.add_argument("--conditions", default=os.getenv("EVAL_NAME", "all"), help="comma-separated profile keys, or all")
    parser.add_argument("--output-root", default=os.getenv("OUTPUT_ROOT"))
    parser.add_argument("--time", default=os.getenv("TIME_OVERRIDE"), help="override HH:MM:SS; max 72h on pli-c")
    parser.add_argument("--manifest", type=Path, help="save reviewable JSON plan")
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    if not args.include_hrbench_8k:
        config["benchmarks"] = [b for b in config["benchmarks"] if b["name"] != "hrbench_8k"]
    conditions = config["conditions"]
    if args.conditions != "all":
        requested = args.conditions.split(",")
        unknown = set(requested) - {c["key"] for c in conditions}
        if unknown:
            parser.error(f"Unknown conditions: {sorted(unknown)}")
        conditions = [c for c in conditions if c["key"] in requested]
    if args.time:
        try:
            h, m, s = map(int, args.time.split(":"))
            assert h >= 0 and 0 <= m < 60 and 0 <= s < 60 and 0 < h * 3600 + m * 60 + s <= 72 * 3600
        except (ValueError, AssertionError):
            parser.error("--time must be HH:MM:SS, positive and at most 72 hours")
    counts, evidence = verify_datasets(config)
    verify_weights(config["judge"])
    jobs = []
    for c in conditions:
        verify_weights(c["checkpoint"])
        if c["system_prompt"] and not Path(resolve(c["system_prompt"])).is_file():
            raise ValueError(f"Missing system prompt for {c['key']}")
        if not Path(config["judge"] + "/chat_template_no_think.jinja").is_file():
            raise ValueError("Missing Qwen3.5 no-think judge template")
        jobs.append(make_job(config, c, args.output_root or config["output_root"], args.time))
    plan = dict(config=str(args.config.resolve()), config_sha256=digest(args.config), dataset_counts=counts,
                dataset_evidence=evidence, jobs=jobs)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(json.dumps(plan, indent=2) + "\n")
    print(f"Verified {len(jobs)} profiles; {sum(counts.values())} samples each; {sum(j['samples'] for j in jobs)} total.")
    for j in jobs:
        print(f"{j['condition']}: {j['seconds_per_sample']:.2f} s/sample, {j['projected_hours']:.1f}h projected, limit {j['walltime']}")
    if args.preflight:
        print("Input preflight passed. No jobs submitted; GPU execution has not been tested.")
        return
    queued = set()
    try:
        result = subprocess.run(["squeue", "-u", os.environ["USER"], "-h", "-r", "-o", "%j", "-t", "PD,R"], check=True, text=True, capture_output=True)
        queued = set(result.stdout.splitlines())
    except (OSError, subprocess.CalledProcessError) as exc:
        if not args.dry_run:
            raise RuntimeError("Cannot verify queued duplicates; refusing to submit") from exc
        print("WARNING: queue unavailable in dry-run; queued duplicates not checked.", file=sys.stderr)
    for job in jobs:
        if complete(job):
            print(f"SKIP complete: {job['condition']}")
            continue
        if job["job_name"] in queued:
            print(f"SKIP queued/running: {job['condition']}")
            continue
        out = Path(job["env"]["OUTPUT_ROOT"])
        cmd = ["sbatch", "--partition=pli-c", "--account=pli", "--constraint=h100",
               "--nodes=1", "--ntasks=1", "--gres=gpu:1", "--cpus-per-task=8", "--mem=180G",
               f"--time={job['walltime']}", f"--job-name={job['job_name']}",
               f"--output={out}/slurm_%j.out", f"--error={out}/slurm_%j.err", "--export=ALL", "run_eval.sbatch"]
        unset = sorted(set(UNSET) | {k for k in os.environ if k.startswith(("SBATCH_", "CHARXIV_JUDGE_", "VLLM_JUDGE_", "VERO_EVAL_"))})
        display = ["env"] + [f"--unset={k}" for k in unset] + [f"{k}={v}" for k, v in job["env"].items()] + cmd
        print(shlex.join(display), flush=True)
        if not args.dry_run:
            out.mkdir(parents=True, exist_ok=True)
            (out / "submission_plan.json").write_text(json.dumps(job, indent=2) + "\n")
            environment = {k: v for k, v in os.environ.items() if k not in unset}
            environment.update(job["env"])
            subprocess.run(cmd, cwd=EVAL, env=environment, check=True)
    print("Dry-run finished; no jobs submitted." if args.dry_run else "Submission pass finished.")


if __name__ == "__main__":
    main()
