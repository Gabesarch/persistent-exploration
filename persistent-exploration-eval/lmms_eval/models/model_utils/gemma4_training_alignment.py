"""Native Gemma4 protocol with an explicit evaluation-only critic override."""

import copy
import hashlib
import json
import os
from pathlib import Path

import yaml

from .training_alignment import TrainingAlignment

THOUGHT = "<|channel>thought\n"
END_THOUGHT = "<channel|>"
CONTINUATION = "{{- gemma4_continuation | default('') -}}"


class Gemma4TrainingAlignment(TrainingAlignment):
    native_gemma4 = True

    def __init__(self, path, checkpoint_path=None, *, eval_critic=None):
        if eval_critic is None and 'GEMMA_EVAL_CRITIC' in os.environ:
            value = os.environ['GEMMA_EVAL_CRITIC'].lower()
            if value not in ('true', 'false'):
                raise ValueError('GEMMA_EVAL_CRITIC must be true or false')
            eval_critic = value == 'true'
        if eval_critic is not None and type(eval_critic) is not bool:
            raise ValueError('eval_critic must be a boolean or None')
        self._eval_critic_override = eval_critic
        self.path = Path(path).resolve()
        self.root = Path(__file__).resolve().parents[4]
        self.config = yaml.safe_load(self.path.read_text())
        self.data = self.config["data"]
        self.rollout = self.config["actor_rollout_ref"]["rollout"]
        self.multi = self.rollout["multi_turn"]
        self.system_prompt = Path(self.data["system_prompt"]).read_text().strip()
        tool = yaml.safe_load(Path(self.multi["tool_config_path"]).read_text())["tools"][0]
        self.tool_schema = tool["tool_schema"]
        self.image_indexing = tool["config"]["image_indexing"]
        self.base_model = Path(self.config["actor_rollout_ref"]["model"]["path"])
        self.checkpoint = Path(checkpoint_path) if checkpoint_path else self.base_model
        self.template = (self.checkpoint / "chat_template.jinja").read_text()
        self.validation_evidence = None
        self._validate(tool)
        self._load_validation_evidence()

    def _validate(self, tool):
        model = self.config["actor_rollout_ref"]["model"]
        checks = {
            "native model": json.loads((self.checkpoint / "config.json").read_text())["model_type"] == "gemma4",
            "native template": model["custom_chat_template"] is None and self.template == (self.base_model / "chat_template.jinja").read_text(),
            "thinking": self.data["apply_chat_template_kwargs"] == {"enable_thinking": True, "preserve_thinking": True},
            "agent": self.rollout["agent"]["default_agent_loop"] == "gemma4_tool_agent",
            "format": self.multi["format"] == "gemma4",
            "one call": self.multi["max_parallel_calls"] == 1,
            "turn budget": self.multi["max_assistant_turns"] == 11 and self.multi["max_user_turns"] == 10,
            "response budget": self.data["max_response_length"] == 24576 and self.multi["max_assistant_tokens_per_turn"] == 4096,
            "context": self.rollout["max_model_len"] == 40960,
            "no forced exploration": self.multi["force_tool_calls_until_final_turn"] is False and self.multi["tool_response_turn_suffix_enabled"] is False,
            "final answer": self.multi["force_answer_on_final_turn_validation"] is True,
            "no oracle at evaluation": self.multi["conditional_answer_retry_enable_validation"] is False and self.multi["conditional_answer_retry_branch_enable_validation"] is False,
            "no retry suffix": self.multi["conditional_answer_retry_suffix"] == "",
            "native critic": not self.multi["conditional_answer_retry_critic_enable_validation"] or (
                self.multi["conditional_answer_retry_force_next_tool_call"] and
                self.multi["conditional_answer_retry_critic_assistant_prefill"].startswith(THOUGHT) and
                END_THOUGHT in self.multi["conditional_answer_retry_critic_assistant_prefill"]),
            "critic labels": self.multi["conditional_answer_retry_critic_accept_token"] == "A" and self.multi["conditional_answer_retry_critic_retry_token"] == "R",
            "crop": tool["config"]["cropped_min_pixels"] == 518400 and tool["config"]["cropped_max_pixels"] == 1048576 and tool["config"]["use_original_resolution_source"] is True,
            "indices": self.image_indexing == "multi_index",
            "processor": self.rollout["engine_kwargs"]["vllm"]["mm_processor_kwargs"] == {"max_soft_tokens": 1120},
        }
        failed = [name for name, ok in checks.items() if not ok]
        if failed:
            raise ValueError(f"Unsupported Gemma4 training contract: {failed}")

    def _load_validation_evidence(self):
        root = Path(self.config["trainer"]["default_local_dir"])
        validations = sorted((root / "validation_data").glob("*.jsonl"), key=lambda p: int(p.stem))
        for validation in reversed(validations):
            with validation.open() as stream:
                for line in stream:
                    debug = json.loads(line).get("debug", {}).get("comprehensive_debug", {})
                    raw = debug.get("raw_prompt", [])
                    start = next((e for e in debug.get("debug_events", []) if e.get("event") == "sample_start"), {})
                    if not raw or not start.get("tool_prompt"):
                        continue
                    system = next(m["content"] for m in raw if m["role"] == "system")
                    if system != self.system_prompt or start["tool_prompt"] != [self.tool_schema]:
                        raise ValueError(f"Saved prompt/schema disagrees with training recipe: {validation}")
                    user = next(m["content"] for m in raw if m["role"] == "user")
                    if not any("Image 0 is available as img_idx=0." in p.get("text", "") for p in user):
                        raise ValueError(f"Missing saved image-index wrapper: {validation}")
                    self.validation_evidence = str(validation)
                    return
        raise ValueError(f"No saved Gemma4 validation prompt/schema evidence under {root}")

    def configure(self, model):
        super().configure(model)
        model.tool_image_as_user_turn = True
        model.critic_guided_answer_retry_mode = "token"
        model.critic_guided_answer_retry_source = "policy"
        model.first_candidate_answer_turn = 0
        model.followup_candidate_answer_turn = 0
        model.always_retry_answers = False

    @property
    def eval_critic_enabled(self):
        return (self.multi["conditional_answer_retry_critic_enable_validation"]
                and self._eval_critic_override is not False)

    @property
    def eval_critic_threshold(self):
        return self.multi["conditional_answer_retry_critic_decision_threshold"]

    def align_user_prompt(self, messages, task, doc):
        messages = super().align_user_prompt(messages, task, doc)
        for message in messages:
            if message["role"] == "user":
                texts = [p for p in message["content"] if p.get("type") == "text"]
                texts[-1]["text"] = texts[-1]["text"].rstrip() + "\nImage 0 is available as img_idx=0."
                break
        return messages

    def continuation_request(self, payload, text):
        payload = copy.deepcopy(payload)
        extra = payload.setdefault("extra_body", {})
        # Render the native generation header, then append the exact training
        # continuation. Chat API partial-assistant rendering rewrites channels.
        kwargs = extra.setdefault("chat_template_kwargs", {})
        kwargs.update(enable_thinking=True, preserve_thinking=True, gemma4_continuation=text)
        extra.update(add_generation_prompt=True, continue_final_message=False)
        return payload

    def forced_action_extra_body(self, tokenizer, tag, suffix=""):
        from verl.utils.forced_action_tags import (
            FORCE_ACTION_REQUIRED_TAG_KEY, FORCE_ACTION_TAG_IDS_KEY,
            FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY, VLLM_OPENAI_EXTRA_ARGS_KEY,
        )
        action = "<|tool_call>call:" if tag == "<tool_call>" else "<answer>" + suffix
        return {VLLM_OPENAI_EXTRA_ARGS_KEY: {
            FORCE_ACTION_REQUIRED_TAG_KEY: tag,
            FORCE_ACTION_TAG_IDS_KEY: tokenizer.encode(action, add_special_tokens=False),
            FORCE_ACTION_AFTER_THINK_IDS_OPTIONS_KEY: json.dumps([tokenizer.encode(END_THOUGHT, add_special_tokens=False)]),
        }}

    @staticmethod
    def assistant_message(message, prefill=""):
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None) or ""
        if prefill.startswith(THOUGHT):
            reasoning = prefill[len(THOUGHT):] + reasoning
        result = {"role": "assistant", "content": message.content or ""}
        if reasoning:
            result["reasoning"] = reasoning
        return result

    def candidate_content(self, message, prefill=""):
        msg = self.assistant_message(message, prefill)
        return THOUGHT + msg.get("reasoning", "") + END_THOUGHT + msg["content"]

    def critic_request(self, payload, candidate):
        # Native Gemma drops an earlier answer's reasoning when a later user
        # question exists, even with preserve_thinking. Training appends the
        # critic sidecar to the candidate tokens instead of re-rendering it.
        candidate = candidate.rstrip()
        boundary = "" if candidate.endswith(("<turn|>", "<eos>")) else "<turn|>"
        suffix = (candidate + boundary + "\n<|turn>user\n"
                  + self.multi["conditional_answer_retry_critic_question"]
                  + "<turn|>\n<|turn>model\n"
                  + self.multi["conditional_answer_retry_critic_assistant_prefill"])
        return self.continuation_request(payload, suffix)

    def shell_environment(self, template_path):
        values = super().shell_environment(template_path)
        Path(template_path).write_text(self.template + CONTINUATION)
        values.update(MM_MAX_PIXELS="", TOOL_CALL_PARSER="gemma4", REASONING_PARSER="gemma4",
                      GEMMA_EVAL_CRITIC=str(self.eval_critic_enabled).lower(),
                      MM_PROCESSOR_KWARGS=json.dumps(self.rollout["engine_kwargs"]["vllm"]["mm_processor_kwargs"]),
                      DEFAULT_CHAT_TEMPLATE_KWARGS=json.dumps(self.data["apply_chat_template_kwargs"]),
                      LMMS_EVAL_PRESERVE_REASONING="true", TOOL_IMAGE_AS_USER_TURN="true",
                      LMMS_EVAL_HF_TOKENIZER=str(self.checkpoint))
        return values

    def identity(self):
        sha = lambda text: hashlib.sha256(text.encode()).hexdigest()
        return {"training_config": str(self.path), "config_sha256": sha(self.path.read_text()),
                "training_root": self.config["trainer"]["default_local_dir"],
                "system_prompt_sha256": sha(self.system_prompt), "template_sha256": sha(self.template),
                "tool_schema": self.tool_schema, "multi_turn": self.multi,
                "validation_evidence": self.validation_evidence,
                "generation_kwargs": self.generation_kwargs(), "eval_critic_enabled": self.eval_critic_enabled}
