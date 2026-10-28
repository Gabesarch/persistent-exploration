#!/usr/bin/env python3
"""Controlled Gemma generation, sleep/reload, and saved-actor comparisons."""

import argparse
import json
from pathlib import Path


def audit_checkpoint(args):
    import torch
    from safetensors import safe_open

    from vero2_scripts.gemma4_checkpoint_audit import load_smoke_state

    state, shards = load_smoke_state(args.run / "global_step_2/actor", args.model)
    report = {"shards": list(map(str, shards)), "equal": [], "different": {}, "missing": [], "extra": []}
    with safe_open(str(args.model / "model.safetensors"), framework="pt", device="cpu") as source:
        keys = set(source.keys())
        for name in sorted(keys):
            if name not in state:
                report["missing"].append(name)
                continue
            base, actor = source.get_tensor(name), state[name]
            if base.shape != actor.shape:
                report["different"][name] = {"base_shape": list(base.shape), "actor_shape": list(actor.shape)}
                continue
            if torch.equal(base, actor) or torch.allclose(base, actor, rtol=0, atol=0, equal_nan=True):
                report["equal"].append(name)
            else:
                report["different"][name] = {
                    "base_shape": list(base.shape),
                    "actor_shape": list(actor.shape),
                    "max_abs": float((base.float() - actor.float()).abs().max()),
                }
        report["extra"] = sorted(set(state) - keys)
    (args.output / "actor_vs_source.json").write_text(json.dumps(report, indent=2))
    print("CHECKPOINT_COMPARISON", {k: len(v) for k, v in report.items()}, flush=True)


def generation_probe(args):
    from PIL import Image
    from vllm import LLM, SamplingParams

    from verl.models.gemma4.processing import collapse_image_tokens
    from verl.utils.tokenizer import hf_processor

    processor = hf_processor(str(args.model))
    row = json.loads((args.run / "debug_logs/gemma4_raw_trajectories.jsonl").open().readline())
    event = row["events"][0]
    image = Image.open(event["extra_info"]["images"][0]).convert("RGB")
    saved_ids = row["main"]["prompt_ids"]
    collapsed = collapse_image_tokens(saved_ids, processor)
    text = processor.tokenizer.decode(collapsed, skip_special_tokens=False)
    (args.output / "saved_prompt.txt").write_text(text)
    control = processor.apply_chat_template(
        [{"role": "user", "content": "Reply with exactly the word HELLO."}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    prompts = [
        {"prompt_token_ids": processor.tokenizer.encode(control, add_special_tokens=False)},
        {"prompt_token_ids": collapsed, "multi_modal_data": {"image": [image]}},
    ]
    llm = LLM(
        model=str(args.model),
        tensor_parallel_size=2,
        dtype="bfloat16",
        load_format=args.load_format,
        enforce_eager=True,
        enable_sleep_mode=True,
        max_model_len=8192,
        max_num_seqs=2,
        gpu_memory_utilization=0.6,
        enable_prefix_caching=False,
        enable_chunked_prefill=True,
        mm_processor_kwargs={"max_soft_tokens": 1120},
        limit_mm_per_prompt={"image": 11, "audio": 0, "video": 0},
        worker_extension_cls="verl.models.gemma4.sleep_diagnostics.Gemma4SleepDiagnosticWorker",
        seed=42,
    )
    source = str(args.model / "model.safetensors")
    if args.load_format == "dummy":
        llm.collective_rpc("gemma_reload_source", args=(source,))
    report = {"load_format": args.load_format, "stages": {}}

    def measure(stage):
        # Re-encode the image after weight changes; cached embeddings from the
        # deliberately corrupted stage must not contaminate the repaired stage.
        llm.reset_mm_cache()
        snapshot = llm.collective_rpc("gemma_snapshot")
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=256, skip_special_tokens=False),
            use_tqdm=False,
        )
        result = {"workers": snapshot, "generations": []}
        for output in outputs:
            choice = output.outputs[0]
            result["generations"].append(
                {
                    "text": choice.text,
                    "token_ids": list(choice.token_ids),
                    "finish_reason": choice.finish_reason,
                }
            )
        report["stages"][stage] = result
        (args.output / f"generation_{args.load_format}.json").write_text(json.dumps(report, indent=2))
        print("STAGE", stage, "OUTPUTS", [x["text"][:400] for x in result["generations"]], flush=True)
        print(
            "CONSTANTS", stage, [sum(not c["all_ones"] for c in w["constants"].values()) for w in snapshot], flush=True
        )

    measure("before_sleep")
    llm.sleep(level=2)
    llm.wake_up()
    llm.collective_rpc("gemma_reload_source", args=(source,))
    measure("unprotected_sleep_reload")
    llm.collective_rpc("gemma_register_constants")
    measure("restored_constants")
    for cycle in range(2):
        llm.sleep(level=2)
        llm.wake_up()
        llm.collective_rpc("gemma_reload_source", args=(source,))
        measure(f"protected_sleep_reload_{cycle + 1}")
    baseline = report["stages"]["before_sleep"]
    report["summary"] = {
        "baseline_hello": "HELLO" in baseline["generations"][0]["text"],
        "stage_matches_baseline": {
            name: [
                a["token_ids"] == b["token_ids"]
                for a, b in zip(
                    baseline["generations"],
                    stage["generations"],
                    strict=True,
                )
            ]
            for name, stage in report["stages"].items()
        },
        "parameter_hashes_match_baseline": {
            name: [
                a["parameter_sha256"] == b["parameter_sha256"]
                for a, b in zip(
                    baseline["workers"],
                    stage["workers"],
                    strict=True,
                )
            ]
            for name, stage in report["stages"].items()
        },
    }
    (args.output / f"generation_{args.load_format}.json").write_text(json.dumps(report, indent=2))
    print("SUMMARY", json.dumps(report["summary"]), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--load-format", choices=["auto", "dummy"], default="auto")
    ap.add_argument("--audit-checkpoint", action="store_true")
    args = ap.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"diagnose_source_{args.load_format}.py").write_text(Path(__file__).read_text())
    if args.audit_checkpoint:
        audit_checkpoint(args)
    else:
        generation_probe(args)


if __name__ == "__main__":
    main()
