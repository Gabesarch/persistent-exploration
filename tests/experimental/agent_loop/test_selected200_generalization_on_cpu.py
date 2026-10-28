"""Regression coverage for the two controllers and submission protocol boundary."""
import asyncio
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

REPO = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "forced_helpers", Path(__file__).with_name("test_eval_forced_final_answer_on_cpu.py")
)
h = importlib.util.module_from_spec(spec)
spec.loader.exec_module(h)


def test_unconditional_retry_discards_candidates_without_any_critic_request(tmp_path):
    model = h._budget_model(
        tmp_path,
        h._response("<think>guess</think><answer>discard me</answer>"),
        h._response(" inspect elsewhere\n</think>", tool_calls=[h._tool_call("retry")], finish_reason="tool_calls"),
        h._response("<think>done</think><answer>keep at cap</answer>"),
        max_tool_iterations=1, max_tool_images=10,
    )
    model.always_retry_answers = True
    model.force_tool_calls_until_final_turn = False
    text, _, _, metadata = asyncio.run(model.maybe_forward_with_tool(h._budget_request(), 0))
    assert "discard me" not in text
    assert "keep at cap" in text
    assert "Let’s see. I still don’t have decisive evidence" in text
    assert len(model.mcp_client.attempts) == 1
    assert len(model.client.chat.completions.requests) == 3
    assert "retry_critic" not in metadata
    assert [r["controller_action"] for r in metadata["answer_retry"]["decisions"]] == ["retry", "forced_accept"]
    assert all(not r["critic_used"] for r in metadata["answer_retry"]["decisions"])
    assert all("allowed_token_ids" not in r.get("extra_body", {}) for r in model.client.chat.completions.requests)


def test_forced_tool_mode_keeps_final_answer_text_prefill_empty(tmp_path):
    model = h._budget_model(
        tmp_path,
        h._response("<think>look</think>", tool_calls=[h._tool_call("zoom")], finish_reason="tool_calls"),
        h._response("<think>done</think><answer>final</answer>"),
        max_tool_iterations=1, max_tool_images=10,
    )
    model.force_final_answer_think_prefill = ""
    text, _, _, _ = asyncio.run(model.maybe_forward_with_tool(h._budget_request(), 0))
    requests = model.client.chat.completions.requests
    assert len(model.mcp_client.attempts) == 1
    assert h._forced_tag(requests[0]) == "<tool_call>"
    assert h._forced_tag(requests[1]) == "<answer>"
    assert "This is the final turn" not in text
    assert all("This is the final turn" not in str(r["messages"]) for r in requests)


@pytest.mark.parametrize("prefill", ["", "explicit final prefill"])
def test_runner_preserves_prefill_and_configures_pinpoint_judge(tmp_path, prefill):
    ev = tmp_path / "vero_eval"
    (ev / "examples").mkdir(parents=True)
    (ev / "set_paths.sh").write_text("")
    shutil.copyfile(REPO / "vero_eval/examples/eval_domain.sh", ev / "examples/eval_domain.sh")
    bindir = tmp_path / "bin"
    bindir.mkdir()
    python = bindir / "python"
    python.write_text(f"#!{sys.executable}\nimport os,json,sys\nprint(json.dumps(dict(argv=sys.argv[1:],prefill=os.environ['FORCE_FINAL_ANSWER_THINK_PREFILL'])))\n")
    python.chmod(0o755)
    env = dict(os.environ, PATH=str(bindir) + os.pathsep + os.environ["PATH"],
               FORCE_FINAL_ANSWER_THINK_PREFILL=prefill, ALWAYS_RETRY_ANSWERS="true",
               JUDGE_BACKEND="engine", API_TYPE="vllm", HF_HOME=str(tmp_path))
    result = subprocess.run(["bash", str(ev / "examples/eval_domain.sh"), "--tasks", "pinpoint_bench_selected200_zoom"],
                            env=env, check=True, text=True, capture_output=True)
    payload = json.loads(result.stdout.splitlines()[-1])
    assert payload["prefill"] == prefill
    model_args = payload["argv"][payload["argv"].index("--model_args") + 1]
    assert "manage_vllm_server_lifecycle=true" in model_args
    assert "always_retry_answers=true" in model_args


def test_submission_uses_cumulative_200_and_keeps_protocols_separate():
    spec = importlib.util.spec_from_file_location("selected_submitter", REPO / "vero_eval/examples/submit_zoom_eval_selected200_generalization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    config = json.loads(module.CONFIG.read_text())
    jobs = {c["key"]: module.make_job(config, c, "/tmp/test-selected200") for c in config["conditions"]}
    assert len(jobs) == 11
    assert len({j["job_name"] for j in jobs.values()}) == 11
    assert all(j["samples"] == 1424 for j in jobs.values())
    assert {b["name"]: b["count"] for b in config["benchmarks"]} == {
        "pinpoint_bench": 433, "vstar_bench": 191, "hrbench_8k": 800,
    }
    restart = jobs["branchstrata20_fresh_lr1e5_restart_step80"]
    assert restart["step"] == 200 and restart["local_step"] == 120
    assert "/global_step_120/" in restart["env"]["CHECKPOINT_PATH"]
    critic = jobs["branchstrata20_fresh_lr1e5_critic_maxcalls5"]["env"]
    assert critic["MAX_TOOL_ITERATIONS"] == critic["MAX_TOOL_IMAGES"] == "5"
    assert critic["CRITIC_GUIDED_ANSWER_RETRY"] == "true"
    assert critic["CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD"] == "0.5"
    for name in ["qwen3vl_8b_instruct_no_tools", "vero_qwen3i_8b_no_tools"]:
        assert jobs[name]["env"]["TOOLS"] == "off"
        assert jobs[name]["env"]["MCP_SERVER_PATH"] == ""
    retry = jobs["gdpo_fresh_lr1e5_noexplore_accformat_always_retry_answers_no_critic_final_prefill"]["env"]
    assert retry["ALWAYS_RETRY_ANSWERS"] == "true" and retry["CRITIC_GUIDED_ANSWER_RETRY"] == "false"


def test_completion_requires_full_sample_coverage_and_matching_protocol(tmp_path):
    spec = importlib.util.spec_from_file_location("selected_completion", REPO / "vero_eval/examples/submit_zoom_eval_selected200_generalization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    job = {"env": {"OUTPUT_ROOT": str(tmp_path)}, "job_name": "expected", "sample_counts": {"one": 433, "two": 800}}
    path = tmp_path / "eval_results.json"
    data = {"config": {}, "results": {"one": {}, "two": {}}, "n-samples": {"one": {"effective": 433}, "two": {"effective": 799}}}
    path.write_text(json.dumps(data))
    assert not module.complete(job)
    data["n-samples"]["two"]["effective"] = 800
    path.write_text(json.dumps(data))
    assert module.complete(job)
    data["config"]["limit"] = 100
    path.write_text(json.dumps(data))
    assert not module.complete(job)
    (tmp_path / "submission_plan.json").write_text(json.dumps({"job_name": "different"}))
    with pytest.raises(ValueError, match="different protocol"):
        module.complete(job)


def test_requested_step200_rows_preserve_existing_identities_and_minio3_schema():
    from lmms_eval.utils import load_yaml_config

    spec = importlib.util.spec_from_file_location("requested_submitter", REPO / "vero_eval/examples/submit_zoom_eval_selected200_generalization.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = json.loads(module.CONFIG.read_text())
    requested = json.loads((module.CONFIG.parent / "selected200_requested_step200.json").read_text())
    old_jobs = {c['key']: module.make_job(original, c, original['output_root']) for c in original['conditions']}
    jobs = {c['key']: module.make_job(requested, c, requested['output_root']) for c in requested['conditions']}
    assert len(jobs) == 8
    for key, job in jobs.items():
        if key in old_jobs:
            assert job == old_jobs[key]  # Queued full jobs must still deduplicate.
    mini = jobs['minio3_vero2_step200']
    assert mini['step'] == mini['local_step'] == 200
    assert mini['env']['TOOL_CALL_PARSER'] == 'hermes'
    assert mini['env']['RUN_SYSTEM_PROMPT_MODE'] == 'run_env'
    assert mini['env']['CRITIC_GUIDED_ANSWER_RETRY'] == 'false'
    source = load_yaml_config(str(REPO / 'vero_eval/lmms_eval/tasks/visual_probe/visual_probe_easy_reasoning_zoom_minio3_step200.yaml'))
    for task in mini['tasks']:
        config = load_yaml_config(str(REPO / 'vero_eval/lmms_eval/tasks/selected200_generalization' / f'{task}.yaml'))
        assert config['metadata'] == source['metadata']
        assert config['generation_kwargs'] == source['generation_kwargs']
        params = config['metadata']['tool_schema_overrides']['image_zoom_in_tool']['parameters']
        assert params['required'] == ['bbox_2d', 'label']
        assert not {'items', 'minItems', 'maxItems'} & params['properties']['bbox_2d'].keys()
