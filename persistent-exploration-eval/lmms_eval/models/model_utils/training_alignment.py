"""Load the explicit zoom training recipe used by a checkpoint evaluation.

This profile follows training's validation controller (learned A/R decisions),
not its ground-truth oracle used to create supervised training branches.
"""

import argparse
import json
import math
import os
import shlex
from pathlib import Path

import yaml


def _overlay(base, overrides):
    result = dict(base)
    for key, value in overrides.items():
        result[key] = (_overlay(result[key], value)
                       if isinstance(value, dict) and isinstance(result.get(key), dict) else value)
    return result


class TrainingAlignment:
    def __init__(self, path, checkpoint_path=None):
        self.path = Path(path).resolve()
        self.root = self.path.parents[2]
        self.config = yaml.safe_load(self.path.read_text())
        # Use the recipe recorded by this run, including its literal retry text.
        if checkpoint_path:
            checkpoint = Path(checkpoint_path)
            step = next((p for p in (checkpoint, *checkpoint.parents) if p.name.startswith("global_step_")), None)
            if step is not None:
                saved = step.parent / "debug_logs/config_resolved.yaml"
                if saved.is_file():
                    registration = self.config
                    self.config = yaml.safe_load(saved.read_text())
                    domain = registration.get("evaluation_training_domain")
                    if domain:
                        profiles = self.config["data"]["domain_profiles"]["profiles"]
                        if domain not in profiles:
                            raise ValueError(f"Saved training config has no evaluation domain {domain!r}")
                        profile = profiles[domain]
                        self.config["data"] = _overlay(self.config["data"], profile["data"])
                        self.config["actor_rollout_ref"]["rollout"] = _overlay(
                            self.config["actor_rollout_ref"]["rollout"], profile["rollout"])
                        self.config["actor_rollout_ref"]["model"]["custom_chat_template"] = profile["data"]["domain_tool_chat_template"]
                        self.config.update({key: value for key, value in registration.items()
                                            if key.startswith("evaluation_")})
        self.data = self.config["data"]
        self.rollout = self.config["actor_rollout_ref"]["rollout"]
        self.multi = self.rollout["multi_turn"]
        self.template = self.config["actor_rollout_ref"]["model"]["custom_chat_template"]
        self.system_prompt = self.resolve(self.data["system_prompt"]).read_text()
        tool = yaml.safe_load(self.resolve(self.multi["tool_config_path"]).read_text())["tools"][0]
        self.tool_schema = tool["tool_schema"]
        self.image_indexing = tool["config"]["image_indexing"]
        # These are protocol assumptions of the evaluator; fail if the recipe changes.
        assert self.multi["max_parallel_calls"] == 1
        assert self.multi["conditional_answer_retry_suffix"] == ""
        assert self.multi["conditional_answer_retry_force_next_tool_call"]
        assert self.multi["conditional_answer_retry_critic_accept_token"] == "A"
        assert self.multi["conditional_answer_retry_critic_retry_token"] == "R"
        assert tool["config"]["cropped_min_pixels"] == 518400
        assert tool["config"]["cropped_max_pixels"] == 1048576
        self.validation_evidence = None
        if checkpoint_path:
            self._load_checkpoint_evidence(Path(checkpoint_path))
        if self.validation_evidence is None and self.config.get("evaluation_validation_evidence"):
            self._load_validation_evidence(self.resolve(self.config["evaluation_validation_evidence"]))

    def _load_checkpoint_evidence(self, checkpoint):
        # Config dependencies can change after a job starts. Prefer the template
        # saved with the weights and the schema actually rendered in validation.
        step_dir = next((p for p in (checkpoint, *checkpoint.parents) if p.name.startswith("global_step_")), None)
        if step_dir is None:
            return
        template = step_dir / "actor/huggingface/chat_template.jinja"
        if template.is_file():
            self.template = template.read_text()
        validation_root = self.config.get("trainer", {}).get("validation_data_dir")
        validation_root = self.resolve(validation_root) if validation_root else step_dir.parent / "validation_data"
        if not validation_root.is_absolute() or "${" in str(validation_root):
            validation_root = step_dir.parent / "validation_data"
        validation = validation_root / (step_dir.name.removeprefix("global_step_") + ".jsonl")
        if not validation.is_file():
            return
        self._load_validation_evidence(validation)

    def _load_validation_evidence(self, validation):
        with validation.open() as stream:
            for line in stream:
                record = json.loads(line)
                domain = self.config.get("evaluation_training_domain")
                if domain and record.get("training_domain") != domain:
                    continue
                debug = record.get("debug", {}).get("comprehensive_debug", {})
                events = debug.get("debug_events", [])
                start = next((event for event in events if event.get("event") == "sample_start"), {})
                schemas = start.get("tool_prompt", [])
                responses = [
                    event.get("tool_response_text", "") for event in events if event.get("event") == "tool_response"
                ]
                if len(schemas) != 1 or not responses:
                    continue
                if schemas[0]["function"]["name"] != self.tool_schema["function"]["name"]:
                    raise ValueError("Checkpoint validation uses a different tool")
                self.tool_schema = schemas[0]
                self.image_indexing = (
                    "multi_index" if any("The cropped view is now available as img_idx" in text for text in responses)
                    else "training_faithful"
                )
                system = next(m for m in debug["raw_prompt"] if m["role"] == "system")["content"]
                self.system_prompt = system if isinstance(system, str) else "".join(c.get("text", "") for c in system)
                self.validation_evidence = str(validation)
                return
        raise ValueError(f"No usable prompt/tool evidence in {validation}")

    def align_user_prompt(self, messages, task, doc):
        """Render held-out questions with the combined recipe's training wrapper."""
        if task.startswith(("screenspotpro_", "screenspot_")):
            question = doc.get("instruction") or doc.get("instruction_cn") or ""
        elif task.startswith("hstarbench_"):
            from lmms_eval.tasks.hstarbench.hstar_instruction import naturalize_hstar_instruction

            question = naturalize_hstar_instruction(str(doc.get("task", "")))
        elif task.startswith(("visual_probe_", "pinpoint_bench_")):
            question = (doc.get("problem") or doc.get("extra_info", {}).get("question")
                        or doc.get("prompt") or doc.get("instruction") or "")
        else:
            return messages
        question = str(question).replace("<image>", "").strip()
        if task.startswith(("screenspotpro_", "screenspot_", "hstarbench_")):
            # Deliberately preserve the training dataset's literal wording/typo.
            question += ' In the your final answer, output only this JSON array: [{"point_2d": [x, y]}].'
        result = []
        for message in messages:
            if message["role"] != "user":
                result.append(message)
                continue
            content = message["content"]
            media = [part for part in content if part.get("type") != "text"] if isinstance(content, list) else []
            result.append({**message, "content": [*media, {"type": "text", "text": "\n" + question}]})
        return result

    def resolve(self, path):
        path = self.config.get("evaluation_path_overrides", {}).get(path, path)
        local_path = Path(path.replace("${oc.env:VERL_ROOT}", str(self.root)))
        return local_path if local_path.is_absolute() else self.path.parent / local_path

    @property
    def max_tool_iterations(self):
        return min(self.multi["max_assistant_turns"] - 1, self.multi["max_user_turns"])

    @property
    def eval_critic_threshold(self):
        override = os.getenv("VERO_EVAL_CRITIC_THRESHOLD")
        value = float(override) if override else self.multi["conditional_answer_retry_critic_decision_threshold"]
        if not math.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("VERO_EVAL_CRITIC_THRESHOLD must be a finite number in [0, 1]")
        return value

    @property
    def eval_critic_enabled(self):
        return (self.multi["conditional_answer_retry_critic_enable_validation"]
                and os.getenv("VERO_EVAL_DISABLE_CRITIC", "false").lower() != "true")

    def configure(self, model):
        model.max_tool_iterations = self.max_tool_iterations
        model.max_tool_images = self.max_tool_iterations
        model.tool_image_as_user_turn = False
        model.force_tool_calls_until_final_turn = self.multi["force_tool_calls_until_final_turn"]
        model.force_answer_after_max_tool_iterations = self.multi["force_answer_on_final_turn_validation"]
        model.force_final_answer_prefix = self.multi.get("force_final_answer_prefix", "")
        model.force_final_answer_think_prefill = self.multi["force_final_answer_think_prefill"]
        model.critic_guided_answer_retry = self.eval_critic_enabled
        model.critic_guided_answer_retry_threshold = self.eval_critic_threshold
        model.critic_guided_answer_retry_prefill_profile = "persist"

    @property
    def reasoning_critic_settings(self):
        from lmms_eval.models.model_utils.reasoning_replay_critic import replay_contract

        replay = self.config.get("trainer", {}).get("trajectory_replay", {})
        prompt_path = replay.get("system_prompt_path")
        prompt = self.resolve(prompt_path) if prompt_path else replay.get("system_prompt")
        normal = int(replay.get("normal_budget", 4096))
        buffer = int(replay.get("overlong_buffer", 1024))
        limit = int(self.data["max_prompt_length"])
        if min(normal, buffer, limit) <= 0:
            raise ValueError("Training reasoning critic budgets must be positive")
        return {
            "system_prompt": replay_contract().load_judge_system_prompt(prompt),
            "prompt_limit": limit,
            "max_tokens": normal + buffer,
        }

    def generation_kwargs(self):
        # Keep the evaluation's sampling distribution; only align the turn budget.
        return {"max_new_tokens": self.multi["max_assistant_tokens_per_turn"]}

    def shell_environment(self, template_path):
        Path(template_path).write_text(self.template)
        return {
            "CHAT_TEMPLATE": str(template_path),
            "CUSTOM_CHAT_TEMPLATE": "",
            "MAX_MODEL_LEN": self.rollout["max_model_len"],
            "MM_MAX_PIXELS": self.data["max_pixels"],
            "MM_MIN_PIXELS": "",
            "MAX_TOOL_ITERATIONS": self.max_tool_iterations,
            "MAX_TOOL_IMAGES": self.max_tool_iterations,
            "FORCE_TOOL_CALLS_UNTIL_FINAL_TURN": self.multi["force_tool_calls_until_final_turn"],
            "FORCE_ANSWER_AFTER_MAX_TOOL_ITERATIONS": self.multi["force_answer_on_final_turn_validation"],
            "FORCE_FINAL_ANSWER_PREFIX": self.multi.get("force_final_answer_prefix", ""),
            "FORCE_FINAL_ANSWER_THINK_PREFILL": self.multi["force_final_answer_think_prefill"],
            "CRITIC_GUIDED_ANSWER_RETRY": self.eval_critic_enabled,
            "CRITIC_GUIDED_ANSWER_RETRY_THRESHOLD": self.eval_critic_threshold,
            "CRITIC_GUIDED_ANSWER_RETRY_PREFILL_PROFILE": "persist",
            "SYSTEM_PROMPT": "",
        }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--template", required=True)
    args = parser.parse_args()
    profile = TrainingAlignment(os.environ["VERO_TRAINING_CONFIG"], os.getenv("CHECKPOINT_PATH"))
    for key, value in profile.shell_environment(args.template).items():
        value = str(value).lower() if isinstance(value, bool) else str(value)
        print(f"export {key}={shlex.quote(value)}")
