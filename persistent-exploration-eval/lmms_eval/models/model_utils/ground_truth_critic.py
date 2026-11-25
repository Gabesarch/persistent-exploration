"""Grade retry candidates against VisualProbe references with the evaluation judge."""
import os


def grade_visual_probe_candidate(doc, candidate_content, task):
    # A resident, separate judge is required during policy generation. The
    # sequential in-process judge cannot share the active policy GPU here.
    if os.getenv("VLLM_SERVER_JUDGE", "").lower() not in {"1", "true", "yes"}:
        raise ValueError("ground_truth critic requires a dedicated server judge")
    if task not in {
        "visual_probe_easy_reasoning_zoom",
        "visual_probe_medium_reasoning_zoom",
        "visual_probe_hard_reasoning_zoom",
    }:
        raise ValueError(f"ground_truth critic does not support task {task!r}")
    from lmms_eval.tasks.visual_probe import utils

    item = utils.visual_probe_process_results(doc, [candidate_content])["judge_accuracy"]
    reference = utils.visual_probe_doc_to_target(doc)
    if not reference:
        raise ValueError("ground_truth critic requires a nonempty reference answer")
    prompt = utils._format_judge_prompt(item["question"], reference, item["prediction"])
    engine = utils.get_judge_engine(utils.MODEL_HINT)
    outputs = engine.generate_json_batch([prompt], use_tqdm=False)
    raw = outputs[0] if len(outputs) == 1 else ""
    parsed = utils.extract_json_candidate(raw)
    if not isinstance(parsed, dict) or not isinstance(parsed.get("is_equivalent"), bool):
        raise RuntimeError(f"ground_truth critic returned invalid grading JSON: {raw!r}")
    correct = parsed["is_equivalent"]
    return {
        "source": "ground_truth",
        "model_token": None,
        "decision": "A" if correct else "R",
        "selected_logprob": None,
        "p_accept": float(correct),
        "p_retry": float(not correct),
        "threshold": None,
        "ground_truth_score": float(correct),
        "ground_truth_reference": reference,
        "candidate_prediction": item["prediction"],
        "judge_output": raw,
    }
