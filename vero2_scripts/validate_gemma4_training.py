#!/usr/bin/env python3
"""Record pinned runtime/protocol evidence and audit Gemma-only smoke artifacts."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL = Path(os.environ.get("HF_HOME", "/scratch/gpfs/ZHUANGL/gs1693/hf_cache")) / "google/gemma-4-E4B-it"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def preflight(out):
    import torch
    import transformers
    import vllm

    assert torch.__version__ == "2.11.0+cu129", torch.__version__
    assert vllm.__version__ == "0.23.0", vllm.__version__
    assert transformers.__version__ == "5.8.0.dev0", transformers.__version__
    assert "vero2_gemma4_train" in sys.prefix, sys.prefix
    overlay = Path(transformers.__file__).parent
    files = [
        MODEL / "chat_template.jinja",
        MODEL / "config.json",
        MODEL / "generation_config.json",
        ROOT / "vero_eval/prompts/tools/system_gemma_zoom.txt",
        ROOT / "vero2_scripts/config/tool_config/gemma4_image_zoom_in_tool.yaml",
    ]
    source = {str(p.relative_to(overlay)): sha(p) for p in sorted(overlay.rglob("*.py"))}
    manifest = {
        "python": sys.executable,
        "runtime_versions": {
            "torch": torch.__version__,
            "vllm": vllm.__version__,
            "transformers": transformers.__version__,
        },
        "installed_distributions": [
            {"name": d.metadata["Name"], "version": d.version} for d in importlib.metadata.distributions()
        ],
        "protocol_hashes": {str(p): sha(p) for p in files},
        "transformers_source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
        "transformers_files": source,
        "enable_thinking": True,
        "preserve_thinking": True,
        "coordinate_order": "yxyx",
        "max_soft_tokens": 1120,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
    }
    import yaml

    recipe_path = ROOT / "vero2_scripts/config/branchstrata20_fresh_lr1e5_critic_gemma4_e4b_recipe.yaml"
    recipe = yaml.safe_load(recipe_path.read_text())
    train_path = Path(recipe["data"]["train_files"])
    rows = [json.loads(line) for line in train_path.read_text().splitlines() if line.strip()]
    from collections import Counter

    domains = dict(Counter(row.get("ability") for row in rows))
    assert len(rows) == 2042 and domains == {"search": 2042}, (len(rows), domains)
    manifest["training_source"] = {
        "path": str(train_path),
        "sha256": sha(train_path),
        "rows": len(rows),
        "domains": domains,
        "smoke_row_indices": [0, 1, 2, 3],
    }
    manifest["recipe_sha256"] = sha(recipe_path)
    implementation_files = {
        p
        for pattern in (
            "verl/models/gemma4/*.py",
            "verl/experimental/agent_loop/*gemma4*.py",
            "verl/tools/*gemma4*.py",
            "verl/utils/dataset/*gemma4*.py",
            "verl/utils/reward_score/*gemma4*.py",
            "vero2_scripts/config/*gemma4*.yaml",
            "vero2_scripts/config/tool_config/*gemma4*.yaml",
            "vero2_scripts/*gemma4*.py",
            "vero2_scripts/*gemma4*.sh",
            "tests/experimental/agent_loop/*gemma4*.py",
        )
        for p in sorted(ROOT.glob(pattern))
    }
    shared_changes = subprocess.check_output(["git", "diff", "--name-only", "--", "verl"], cwd=ROOT, text=True)
    implementation_files.update(ROOT / name for name in shared_changes.splitlines())
    implementation_files.add(ROOT / "verl/experimental/agent_loop/tool_parser.py")
    from verl.models.gemma4.protocol import TOOL_CALL_EOS_POLICY

    manifest["tool_call_eos_policy"] = TOOL_CALL_EOS_POLICY
    manifest["implementation_hashes"] = {str(p.relative_to(ROOT)): sha(p) for p in sorted(implementation_files)}
    vllm_root = Path(vllm.__file__).parent
    manifest["vllm_gemma_source_hashes"] = {
        str(p.relative_to(vllm_root)): sha(p) for p in sorted(vllm_root.rglob("*gemma4*.py"))
    }
    (out / "runtime_protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")


def fixtures(out):
    tests = [
        "tests/experimental/agent_loop/test_gemma4_native_on_cpu.py",
        "tests/experimental/agent_loop/test_gemma4_fsdp_buffer_sync_on_cpu.py",
        "tests/experimental/agent_loop/test_gemma4_shared_kv_on_cpu.py",
        "tests/experimental/agent_loop/test_gemma4_vllm_state_on_cpu.py",
    ]
    proc = subprocess.run(
        [sys.executable, "-m", "pytest", "-q", *tests, "--junitxml=" + str(out / "fixtures.xml")],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (out / "fixtures.log").write_text(proc.stdout)
    print(proc.stdout)
    if proc.returncode:
        raise SystemExit(proc.returncode)


def audit(out):
    from omegaconf import OmegaConf

    cfg = OmegaConf.load(out / "resolved.yaml")
    run = Path(cfg.trainer.default_local_dir)
    report = {"run": str(run), "fixture_coverage": "fixtures.xml", "live_validation_passed": False}
    checkpoints = list(run.glob("global_step_*/actor/model_world_size_*_rank_0.pt"))
    report["checkpoints"] = [str(p) for p in checkpoints]
    text = (out / "training.log").read_text()
    report["training_complete"] = "training_complete" in text or "step:2" in text
    report["limitations"] = []
    if not report["training_complete"]:
        report["limitations"].append("No completed second optimizer step observed.")
    # Saved metrics and raw trajectories must be inspected; never declare success from scheduler exit.
    report["debug_artifacts"] = [str(p) for p in (run / "debug_logs").rglob("*") if p.is_file()]
    if checkpoints:
        reload_check = subprocess.run(
            [
                sys.executable,
                str(ROOT / "vero2_scripts/gemma4_checkpoint_audit.py"),
                "--run",
                str(run),
                "--model",
                str(MODEL),
                "--output",
                str(out / "checkpoint_audit.json"),
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        (out / "checkpoint_audit.log").write_text(reload_check.stdout)
        if reload_check.returncode:
            report["limitations"].append("Checkpoint update/reload audit failed; see checkpoint_audit.log.")
        else:
            report["checkpoint_audit"] = json.loads((out / "checkpoint_audit.json").read_text())
    else:
        report["limitations"].append("No smoke checkpoint saved.")
    traces = run / "debug_logs/gemma4_raw_trajectories.jsonl"
    records = [json.loads(line) for line in traces.read_text().splitlines() if line.strip()] if traces.exists() else []
    samples = [sample for record in records for sample in [record["main"], *record["branches"]]]
    report["live_samples"] = len(records)
    report["live_critic_labels"] = {
        label: sum(s["extra_fields"].get("retry_critic_label") == label for s in samples) for label in ("A", "R")
    }
    report["live_branches"] = sum(len(r["branches"]) for r in records)
    report["live_crop_returns"] = sum(bool(r.get("crop_image_refs")) for r in records)
    if not sum(report["live_critic_labels"].values()):
        report["limitations"].append("No live critic supervision observed.")
    if not report["live_branches"]:
        report["limitations"].append("No live retry branch observed; deterministic fixture coverage only.")
    if not report["live_crop_returns"]:
        report["limitations"].append("No live crop return observed; deterministic fixture coverage only.")
    import math
    import re

    number = r"([-+]?(?:[0-9]+(?:\.[0-9]*)?(?:[eE][-+]?[0-9]+)?|nan|inf))"
    # The console logger prints NumPy scalar reprs on the pinned NumPy version.
    number = r"(?:np\.float(?:16|32|64)\(\s*)?" + number

    def metric_values(key):
        return [float(x) for x in re.findall(re.escape(key) + r"[:\s]+" + number, text, flags=re.I)]

    gradients = metric_values("actor/grad_norm")
    report["gradient_norms"] = gradients
    losses = {}
    for key, value in re.findall(r"(actor/[\w/]*loss[\w/]*)[:\s]+" + number, text, flags=re.I):
        losses.setdefault(key, []).append(float(value))
    report["loss_metrics"] = losses
    if not losses or not all(math.isfinite(value) for values in losses.values() for value in values):
        report["limitations"].append("Missing or nonfinite loss evidence.")
    report["active_layer_metrics"] = {}
    for group in ("text", "vision"):
        for suffix in ("grad_norm", "update_max_abs", "param_bf16", "grad_fp32", "grad_bf16"):
            key = f"actor/gemma4_{group}_{suffix}"
            values = metric_values(key)
            report["active_layer_metrics"][key] = values
            if not values or not all(math.isfinite(value) for value in values):
                report["limitations"].append(f"Missing or nonfinite active-layer metric: {key}.")
            elif suffix == "grad_norm" and not any(value > 0 for value in values):
                report["limitations"].append(f"All-zero live {group} gradient signal.")
    rollout_steps = sorted(
        {r.get("trajectory", {}).get("step") for r in records if isinstance(r.get("trajectory", {}).get("step"), int)}
    )
    report["live_rollout_steps"] = rollout_steps
    report["weight_sync_evidence"] = "live rollout after step-one update" if 2 in rollout_steps else None
    if 2 not in rollout_steps:
        report["limitations"].append("No live second-step rollout confirming the update/synchronization path.")
    if not gradients or not all(math.isfinite(x) for x in gradients):
        report["limitations"].append("Missing or nonfinite gradient evidence.")
    elif not any(x > 0 for x in gradients):
        report["limitations"].append("All-zero gradient signal.")
    report["live_validation_passed"] = bool(checkpoints and records and gradients and not report["limitations"])
    (out / "acceptance.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["live_validation_passed"]:
        raise SystemExit("Smoke acceptance incomplete; see acceptance.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--fixtures", action="store_true")
    ap.add_argument("--audit", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.preflight:
        preflight(args.output)
    if args.fixtures:
        fixtures(args.output)
    if args.audit:
        audit(args.output)


if __name__ == "__main__":
    main()
