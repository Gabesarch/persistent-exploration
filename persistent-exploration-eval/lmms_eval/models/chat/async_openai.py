import asyncio
import fnmatch
import json
import math
import os
import re
import shutil
import tempfile
import time
import urllib.parse
import urllib.request
import uuid
from functools import lru_cache
from multiprocessing import cpu_count
from typing import Dict, List, Optional, Tuple

from accelerate import Accelerator, DistributedType
from dotenv import load_dotenv
from lmms_eval.api.instance import GenerationResult, Instance, TokenCounts
from lmms_eval.api.model import lmms
from lmms_eval.api.registry import register_model
from lmms_eval.imports import optional_import
from lmms_eval.models.model_utils.concurrency_control import (
    AdaptiveConcurrencyConfig,
    decide_next_concurrency,
    extract_text_prefix_from_chat_messages,
    is_rate_limit_error,
    make_prefix_hash,
    parse_bool,
)
from lmms_eval.models.model_utils.usage_metrics import (
    get_running_totals,
    is_budget_exceeded,
    log_usage,
)
from lmms_eval.protocol import ChatMessages
from loguru import logger as eval_logger
from openai import AsyncOpenAI
from tqdm import tqdm
from transformers import AutoTokenizer

from verl.utils.forced_action_tags import build_forced_action_openai_extra_body

VideoReader, _ = optional_import("decord", "VideoReader")
cpu, _ = optional_import("decord", "cpu")

# Sentinel emitted by MCP tools (see lmms_eval/mcp/vero2_mcp.py) so the
# client can discover and clean up tool-generated image files after each
# request finishes.
_TOOL_SAVED_IMAGE_RE = re.compile(r"Saved drawn image to:\s*(\S+)")
_HIDDEN_TOOL_ARGS = {"image_path"}
_IMAGE_REGISTRY_ENV = "LMMS_EVAL_IMAGE_REGISTRY"
_ASSISTANT_THINK_PREFIX = "<think>\n"
_ANSWER_BLOCK_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_VLLM_ADMIN_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
_RETRY_CRITIC_QUESTION = (
    'Assuming another tool round is available, should the preceding candidate be accepted or retried? '
    'In the <answer> block, output only "A" or "R".'
)
_RETRY_CRITIC_ASSISTANT_PREFILL = (
    "I need to classify whether the preceding candidate should be accepted or retried.\n"
    "</think>\n"
    "<answer>\n"
)
_RETRY_TOOL_ASSISTANT_PREFILL = (
    "Let’s see. I still don’t have decisive evidence, so I should inspect a substantially different region. "
    "If one area looks especially promising, I can examine it more closely, but I should look elsewhere once "
    "further zooming stops revealing new evidence."
)
_RETRY_CRITIC_ACCEPT_TOKEN = "A"
_RETRY_CRITIC_RETRY_TOKEN = "R"
_RETRY_PREFILL_PROFILES = {"persist", "teacher_only"}


class VLLMServerLifecycleError(RuntimeError):
    """Raised when the policy server cannot complete a required lifecycle transition."""


# Env var carrying the per-call image registry (img_idx -> file path) to the MCP
# tool subprocess, so tools resolve img_idx the way training does. Keep in sync
# with lmms_eval/mcp/vero2_mcp.py _IMAGE_REGISTRY_ENV.
_IMAGE_REGISTRY_ENV = "LMMS_EVAL_IMAGE_REGISTRY"

# persistent-exploration-eval package root (parent of lmms_eval), used to resolve the relative
# system_prompt_files paths declared in task yamls (e.g. prompts/tools/*.txt).
_VERO_EVAL_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))


def _hf_tokenizer_local_files_only() -> bool:
    override = os.getenv("LMMS_EVAL_HF_TOKENIZER_LOCAL_ONLY")
    if override is not None:
        return override.lower() not in {"0", "false", "no", "off"}
    return os.getenv("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"} or os.getenv(
        "TRANSFORMERS_OFFLINE", ""
    ).lower() in {"1", "true", "yes", "on"}


def _resolve_hf_tokenizer_path(name: str) -> str:
    if os.path.isabs(name) or os.path.isdir(name):
        return name
    if not _hf_tokenizer_local_files_only():
        return name
    hf_home = os.path.expanduser(os.getenv("HF_HOME", ""))
    local = os.path.join(hf_home, name) if hf_home else ""
    if local and os.path.isdir(local):
        return local
    return name


@lru_cache(maxsize=4)
def _get_forced_action_tokenizer(name: str):
    use_fast = os.getenv("LMMS_EVAL_HF_TOKENIZER_USE_FAST", "1").lower() not in {"0", "false", "no", "off"}
    trust_remote_code = os.getenv("LMMS_EVAL_HF_TRUST_REMOTE_CODE", "0").lower() in {"1", "true", "yes", "on"}
    return AutoTokenizer.from_pretrained(
        _resolve_hf_tokenizer_path(name),
        use_fast=use_fast,
        trust_remote_code=trust_remote_code,
        local_files_only=_hf_tokenizer_local_files_only(),
    )


def _composite_zoom_to_original(image_provenance, img_idx, bbox):
    """Remap a zoom-into-img_idx call so it crops the ROOT original at NATIVE resolution.

    Each img_idx has provenance (orig_idx, region=[X1,Y1,X2,Y2]) = the native-pixel rectangle of
    original `orig_idx` that this image represents. The model's 0-1000 bbox (on the image it sees)
    is composited into the original's native pixels (region_origin + bbox/1000*region_span), then
    re-expressed as a 0-1000 bbox of the ORIGINAL so the (unchanged) MCP tool, called with
    img_idx=orig_idx, crops the original PNG at full resolution instead of re-cropping a resized
    crop. Mirrors the training tool's in-tool compositing. Returns
    (orig_idx, native_subregion, bbox_in_original_0_1000) or None if no provenance.
    For an original (region = full image) this is a no-op: bbox' == bbox, orig_idx == img_idx."""
    prov = image_provenance.get(img_idx)
    root = image_provenance.get(prov[0]) if prov is not None else None
    if prov is None or root is None:
        return None
    orig_idx, region = prov
    x1, y1, x2, y2 = region
    rw, rh = x2 - x1, y2 - y1
    nsub = [
        x1 + bbox[0] / 1000.0 * rw, y1 + bbox[1] / 1000.0 * rh,
        x1 + bbox[2] / 1000.0 * rw, y1 + bbox[3] / 1000.0 * rh,
    ]
    w_orig, h_orig = root[1][2], root[1][3]  # root region is [0,0,W,H] (full original, native)
    bbox_prime = [
        nsub[0] / w_orig * 1000.0, nsub[1] / h_orig * 1000.0,
        nsub[2] / w_orig * 1000.0, nsub[3] / h_orig * 1000.0,
    ]
    return orig_idx, nsub, bbox_prime


def _zoom_bbox_to_xyxy(bbox, coordinate_order="xyxy"):
    """Convert a model's normalized box into the crop tool's x/y order."""
    if len(bbox) != 4:
        raise ValueError("Zoom bbox must contain exactly four coordinates")
    values = [float(value) for value in bbox]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Zoom bbox coordinates must be finite")
    if coordinate_order == "yxyx":
        values = [values[1], values[0], values[3], values[2]]
    elif coordinate_order != "xyxy":
        raise ValueError(f"Unknown zoom coordinate order: {coordinate_order}")
    if not (values[0] < values[2] and values[1] < values[3]):
        raise ValueError("Zoom bbox must have positive width and height")
    return values


def _restore_assistant_forced_prefill(content: str, prefix: str = _ASSISTANT_THINK_PREFIX) -> str:
    """Restore forced assistant prefill that is present in the prompt but omitted from completions."""
    if not content or not prefix:
        return content
    stripped = content.lstrip()
    if stripped.startswith(prefix.strip()):
        return content
    if "</think>" not in content:
        return content
    leading_ws_len = len(content) - len(stripped)
    return content[:leading_ws_len] + prefix + stripped


def _preserve_assistant_reasoning(message, assistant_dict):
    """Replay the separately parsed thought channel for opt-in thinking runs."""
    if os.getenv("LMMS_EVAL_PRESERVE_REASONING", "false").lower() in {"1", "true", "yes", "on"}:
        reasoning = getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        if reasoning:
            assistant_dict["reasoning"] = reasoning
    return assistant_dict


load_dotenv(verbose=True)


@register_model("async_openai")
class AsyncOpenAIChat(lmms):
    is_simple = False

    def __init__(
        self,
        model_version: str = "grok-2-latest",
        model: Optional[str] = None,
        base_url: str = None,
        api_key: str = None,
        timeout: int = 600,
        retry_backoff_s: Optional[float] = None,
        max_retries: int = 5,
        max_size_in_mb: int = 20,
        mcp_server_path: str = None,
        num_cpus: int = None,
        work_dir: str = None,
        fps: Optional[int] = None,
        nframes: Optional[int] = 64,
        max_frames: Optional[int] = 768,
        max_pixels: Optional[int] = 151200,
        min_pixels: Optional[int] = 28 * 28,
        is_qwen3_vl: bool = False,
        adaptive_concurrency: bool = False,
        adaptive_min_concurrency: int = 1,
        adaptive_max_concurrency: int = 128,
        adaptive_target_latency_s: float = 15.0,
        adaptive_increase_step: float = 0.1,
        adaptive_decrease_factor: float = 0.7,
        adaptive_failure_threshold: float = 0.05,
        prefix_aware_queue: bool = True,
        prefix_hash_chars: int = 256,
        system_prompt: Optional[str] = None,
        max_tool_iterations: int = 10,
        max_tool_images: int = 10,
        tool_image_as_user_turn: bool = False,
        force_tool_calls_until_final_turn: bool = False,
        force_answer_after_max_tool_iterations: bool = False,
        force_final_answer_prefix: str = "",
        force_final_answer_think_prefill: Optional[str] = None,
        critic_guided_answer_retry: bool = False,
        critic_guided_answer_retry_source: str = "policy",
        critic_guided_answer_retry_mode: str = "token",
        critic_guided_answer_retry_temperature: float = 0.0,
        critic_guided_answer_retry_top_k: int = -1,
        critic_guided_answer_retry_top_p: float = 1.0,
        critic_guided_answer_retry_threshold: float = 0.5,
        critic_guided_answer_retry_prefill_profile: str = "persist",
        first_candidate_answer_turn: int = 0,
        followup_candidate_answer_turn: int = 0,
        record_answer_candidates: bool = False,
        always_retry_answers: bool = False,
        manage_vllm_server_lifecycle: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        if model is not None:
            model_version = model
        if kwargs:
            eval_logger.warning(f"Unknown model_args ignored: {list(kwargs.keys())}. " f"Check the supported parameters for the 'async_openai' backend.")
        self.model_version = model_version
        self.timeout = timeout
        self.retry_backoff_s = max(0.0, float(1.0 if retry_backoff_s is None else retry_backoff_s))
        self.max_retries = max_retries
        self.max_size_in_mb = max_size_in_mb  # some models have a limit on the size of the image
        if num_cpus is None:
            self.num_cpus = cpu_count() // 2
        else:
            self.num_cpus = num_cpus
        self.work_dir = work_dir if work_dir is not None else tempfile.mkdtemp()
        self.fps = fps
        self.nframes = nframes
        self.base_url = base_url if base_url is not None else os.getenv("OPENAI_API_BASE")
        self.api_key = api_key if api_key is not None else os.getenv("OPENAI_API_KEY")
        self.client = AsyncOpenAI(api_key=self.api_key, base_url=self.base_url, timeout=timeout)
        self.max_pixels = max_pixels
        self.min_pixels = min_pixels
        self.max_frames = max_frames
        self.is_qwen3_vl = is_qwen3_vl
        self.adaptive_concurrency = parse_bool(adaptive_concurrency)
        self.adaptive_config = AdaptiveConcurrencyConfig.from_raw(
            min_concurrency=adaptive_min_concurrency,
            max_concurrency=adaptive_max_concurrency,
            target_latency_s=adaptive_target_latency_s,
            increase_step=adaptive_increase_step,
            decrease_factor=adaptive_decrease_factor,
            failure_threshold=adaptive_failure_threshold,
        )
        self.prefix_aware_queue = parse_bool(prefix_aware_queue)
        self.prefix_hash_chars = max(32, int(prefix_hash_chars))
        if system_prompt is not None:
            self.system_prompt = self._resolve_system_prompt(system_prompt)
        else:
            self.system_prompt = None
        if mcp_server_path is not None:
            from lmms_eval.mcp import MCPClient

            self.mcp_client = MCPClient(mcp_server_path)
            os.makedirs(self.work_dir, exist_ok=True)
        else:
            self.mcp_client = None
        # Multi-turn tool loop controls (model_args). Tool list is fetched once
        # and cached to avoid forking the MCP server subprocess on every request.
        self.max_tool_iterations = int(max_tool_iterations)
        self.max_tool_images = int(max_tool_images)
        self.tool_image_as_user_turn = parse_bool(tool_image_as_user_turn)
        self.force_tool_calls_until_final_turn = parse_bool(force_tool_calls_until_final_turn)
        self.force_answer_after_max_tool_iterations = parse_bool(force_answer_after_max_tool_iterations)
        # Opt-in repaired evaluation protocol; historical/training-aligned runs
        # retain their original response budgeting unless explicitly selected.
        self.context_budget_enabled = parse_bool(os.getenv("LMMS_EVAL_CONTEXT_BUDGET", "false"))
        self.context_budget_limit = int(os.getenv("MAX_MODEL_LEN", "0"))
        self.context_budget_reserve = int(os.getenv("LMMS_EVAL_CONTEXT_RESERVE_TOKENS", "16384"))
        if self.context_budget_enabled and not (
            1024 <= self.context_budget_reserve < self.context_budget_limit
            and self.force_answer_after_max_tool_iterations
        ):
            raise ValueError("Context budgeting requires MAX_MODEL_LEN, a smaller reserve, and forced final answering")
        self.first_candidate_answer_turn = int(first_candidate_answer_turn)
        self.followup_candidate_answer_turn = int(followup_candidate_answer_turn)
        if self.followup_candidate_answer_turn and not (
            2 <= self.first_candidate_answer_turn < self.followup_candidate_answer_turn <= self.max_tool_iterations
        ):
            raise ValueError("followup_candidate_answer_turn requires first turn < followup turn <= max_tool_iterations")
        self.record_answer_candidates = parse_bool(record_answer_candidates)
        if self.first_candidate_answer_turn and not 2 <= self.first_candidate_answer_turn <= self.max_tool_iterations:
            raise ValueError("first_candidate_answer_turn must be 0 (disabled) or between 2 and max_tool_iterations")
        if self.first_candidate_answer_turn and self.force_tool_calls_until_final_turn:
            raise ValueError("first_candidate_answer_turn is incompatible with force_tool_calls_until_final_turn")
        if self.force_tool_calls_until_final_turn and self.max_tool_iterations < 1:
            raise ValueError("max_tool_iterations must be positive when force_tool_calls_until_final_turn is enabled")
        self.force_final_answer_prefix = force_final_answer_prefix or ""
        self.manage_vllm_server_lifecycle = parse_bool(manage_vllm_server_lifecycle)
        if force_final_answer_think_prefill is None:
            force_final_answer_think_prefill = os.getenv("FORCE_FINAL_ANSWER_THINK_PREFILL", "")
        self.force_final_answer_think_prefill = force_final_answer_think_prefill or ""
        self._forced_action_tokenizer = None
        self.critic_guided_answer_retry = parse_bool(critic_guided_answer_retry)
        self.always_retry_answers = parse_bool(always_retry_answers)
        if self.always_retry_answers and (self.critic_guided_answer_retry or self.force_tool_calls_until_final_turn):
            raise ValueError("always_retry_answers requires critic and forced-tool modes to be disabled")
        if self.always_retry_answers and (self.mcp_client is None or self.max_tool_iterations < 1):
            raise ValueError("always_retry_answers requires tools and a positive tool budget")
        self.critic_guided_answer_retry_source = str(critic_guided_answer_retry_source)
        self.critic_guided_answer_retry_mode = str(critic_guided_answer_retry_mode)
        self.critic_guided_answer_retry_temperature = float(critic_guided_answer_retry_temperature)
        self.critic_guided_answer_retry_top_k = int(critic_guided_answer_retry_top_k)
        self.critic_guided_answer_retry_top_p = float(critic_guided_answer_retry_top_p)
        if self.critic_guided_answer_retry_mode not in {"token", "reasoning"}:
            raise ValueError("critic_guided_answer_retry_mode must be token or reasoning")
        if self.critic_guided_answer_retry_source not in {"policy", "ground_truth"}:
            raise ValueError("critic_guided_answer_retry_source must be policy or ground_truth")
        if self.critic_guided_answer_retry_source == "ground_truth":
            if not self.critic_guided_answer_retry:
                raise ValueError("ground_truth critic requires critic_guided_answer_retry=true")
            if os.getenv("VLLM_SERVER_JUDGE", "").lower() not in {"1", "true", "yes"}:
                raise ValueError("ground_truth critic requires a dedicated server judge")
        self._ground_truth_critic_lock = asyncio.Lock()
        self.critic_guided_answer_retry_threshold = float(critic_guided_answer_retry_threshold)
        from lmms_eval.models.model_utils.critic_thresholds import parse_domain_thresholds

        self.critic_domain_thresholds = parse_domain_thresholds(os.getenv("VERO_EVAL_CRITIC_DOMAIN_THRESHOLDS", ""))
        self.critic_guided_answer_retry_prefill_profile = str(critic_guided_answer_retry_prefill_profile)
        self.training_alignment = None
        if os.getenv("GEMMA_TRAINING_CONFIG"):
            from lmms_eval.models.model_utils.gemma4_training_alignment import Gemma4TrainingAlignment

            self.training_alignment = Gemma4TrainingAlignment(os.environ["GEMMA_TRAINING_CONFIG"], os.getenv("CHECKPOINT_PATH"))
            self.training_alignment.configure(self)
        elif os.getenv("VERO_TRAINING_CONFIG"):
            from lmms_eval.models.model_utils.training_alignment import TrainingAlignment

            self.training_alignment = TrainingAlignment(os.environ["VERO_TRAINING_CONFIG"], os.getenv("CHECKPOINT_PATH"))
            self.training_alignment.configure(self)
        if not 0.0 <= self.critic_guided_answer_retry_threshold <= 1.0:
            raise ValueError("critic_guided_answer_retry_threshold must be in [0, 1]")
        if self.critic_guided_answer_retry_prefill_profile not in _RETRY_PREFILL_PROFILES:
            raise ValueError(
                "critic_guided_answer_retry_prefill_profile must be one of "
                f"{sorted(_RETRY_PREFILL_PROFILES)}"
            )
        self._retry_critic_accept_token_id = None
        self._retry_critic_retry_token_id = None
        if self.critic_guided_answer_retry and self.critic_guided_answer_retry_mode == "token":
            tokenizer_name = os.getenv("LMMS_EVAL_HF_TOKENIZER", "Qwen/Qwen3-8B")
            self._forced_action_tokenizer = _get_forced_action_tokenizer(tokenizer_name)
            accept_ids = self._forced_action_tokenizer.encode(
                _RETRY_CRITIC_ACCEPT_TOKEN, add_special_tokens=False
            )
            retry_ids = self._forced_action_tokenizer.encode(
                _RETRY_CRITIC_RETRY_TOKEN, add_special_tokens=False
            )
            if len(accept_ids) != 1 or len(retry_ids) != 1 or accept_ids[0] == retry_ids[0]:
                raise ValueError(
                    "retry critic A/R labels must encode as distinct single tokens; "
                    f"got {accept_ids!r} and {retry_ids!r}"
                )
            self._retry_critic_accept_token_id = int(accept_ids[0])
            self._retry_critic_retry_token_id = int(retry_ids[0])
        self._retry_critic_stats = {
            "decisions": 0,
            "accepts": 0,
            "retries": 0,
            "forced_accepts": 0,
            "failures": 0,
            "retry_probability_sum": 0.0,
            "retry_probability_count": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "reasoning_tokens": 0,
        }
        self._forced_final_stats = {
            "primary_attempts": 0,
            "primary_successes": 0,
            "fallback_attempts": 0,
            "fallback_successes": 0,
            "primary_input_tokens": 0,
            "primary_output_tokens": 0,
            "fallback_input_tokens": 0,
            "fallback_output_tokens": 0,
        }
        self._forced_final_failures: dict[str, int] = {}
        self._tool_functions = None
        self._task_tools_cache = {}
        self._task_system_prompt_cache = {}

        # If system_prompt contains placeholders, fill them in
        if self.system_prompt and self.mcp_client is not None:
            from datetime import datetime
            # Handle YAML-like format (Section: | content) by extracting just the content
            if "Role: |" in self.system_prompt or "Tools: |" in self.system_prompt:
                import re
                sections = re.split(r'\n[A-Z][a-z]+:\s*\|\n', self.system_prompt)
                # Remove empty first element and join sections
                self.system_prompt = "\n".join(s.strip() for s in sections if s.strip())

            if "{tools}" in self.system_prompt:
                tool_list = self._rewrite_tool_schemas_for_model(self.mcp_client.get_function_list_sync())
                self._tool_functions = tool_list  # Cache for later use
                tools_str = "\n".join(json.dumps(t) for t in tool_list)
                self.system_prompt = self.system_prompt.replace("{tools}", tools_str)
                eval_logger.info(f"Injected {len(tool_list)} tools into system prompt")

            # Handle other placeholders
            if "{date}" in self.system_prompt:
                self.system_prompt = self.system_prompt.replace("{date}", datetime.now().strftime("%Y-%m-%d"))
            if "{examples}" in self.system_prompt:
                self.system_prompt = self.system_prompt.replace("{examples}", "(No examples provided)")
            if "{schema}" in self.system_prompt:
                schema = '{"name": "<function-name>", "arguments": <args-json-object>}'
                self.system_prompt = self.system_prompt.replace("{schema}", schema)

        accelerator = Accelerator()
        # assert self.batch_size_per_gpu == 1, "Llava currently does not support batched generation. See https://github.com/haotian-liu/LLaVA/issues/754. HF Llava also has this issue."
        if accelerator.num_processes > 1:
            assert accelerator.distributed_type in [
                DistributedType.FSDP,
                DistributedType.MULTI_GPU,
                DistributedType.DEEPSPEED,
            ], "Unsupported distributed type provided. Only DDP and FSDP are supported."
            self.accelerator = accelerator
            if self.accelerator.is_local_main_process:
                eval_logger.info(f"Using {accelerator.num_processes} devices with data parallelism")
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes
        else:
            self.accelerator = accelerator
            self._rank = self.accelerator.local_process_index
            self._world_size = self.accelerator.num_processes

        self.device = self.accelerator.device

    def _vllm_server_root(self) -> str:
        if not self.base_url:
            raise RuntimeError(
                "manage_vllm_server_lifecycle requires an OpenAI-compatible base_url"
            )
        server_root = str(self.base_url).rstrip("/")
        if server_root.endswith("/v1"):
            server_root = server_root[:-3]
        return server_root

    def _is_local_vllm_server(self) -> bool:
        if not self.base_url:
            return False
        hostname = urllib.parse.urlparse(str(self.base_url)).hostname
        return hostname in {"localhost", "127.0.0.1", "::1"}

    def _post_vllm_admin(self, action: str, *, level: Optional[int] = None) -> None:
        params = {}
        if level is not None:
            params["level"] = str(level)
        query = f"?{urllib.parse.urlencode(params)}" if params else ""
        url = f"{self._vllm_server_root()}/{action}{query}"
        request = urllib.request.Request(url, method="POST")
        request_timeout = max(0.1, min(float(self.timeout), 30.0))
        with _VLLM_ADMIN_OPENER.open(request, timeout=request_timeout):
            pass

    def _vllm_is_sleeping(self, request_timeout_s: Optional[float] = None) -> bool:
        url = f"{self._vllm_server_root()}/is_sleeping"
        if request_timeout_s is None:
            request_timeout_s = 10.0
        request_timeout = max(
            0.001,
            min(float(self.timeout), 10.0, float(request_timeout_s)),
        )
        with _VLLM_ADMIN_OPENER.open(url, timeout=request_timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        if isinstance(payload, bool):
            return payload
        if isinstance(payload, dict):
            for key in ("is_sleeping", "sleeping"):
                value = payload.get(key)
                if isinstance(value, bool):
                    return value
        raise RuntimeError(f"Unexpected response from vLLM /is_sleeping: {payload!r}")

    def _wait_for_vllm_sleep_state(
        self,
        *,
        expected_sleeping: bool,
        timeout_s: float = 300.0,
    ) -> None:
        deadline = time.monotonic() + timeout_s
        last_observation = "no response"
        while True:
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0:
                state = "sleeping" if expected_sleeping else "awake"
                raise TimeoutError(
                    f"Timed out after {timeout_s:g}s waiting for vLLM server to become "
                    f"{state}; last observation: {last_observation}"
                )
            try:
                sleeping = self._vllm_is_sleeping(request_timeout_s=remaining_s)
                last_observation = f"is_sleeping={sleeping}"
                if sleeping is expected_sleeping:
                    return
            except Exception as exc:
                last_observation = f"{type(exc).__name__}: {exc}"
            time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))

    def _wake_vllm_server(self) -> None:
        try:
            if not self._vllm_is_sleeping():
                return
        except Exception:
            pass
        self._post_vllm_admin("wake_up")
        self._wait_for_vllm_sleep_state(expected_sleeping=False)

    def _sleep_vllm_server(self) -> None:
        try:
            if self._vllm_is_sleeping():
                return
        except Exception:
            pass
        self._post_vllm_admin("sleep", level=2)
        self._wait_for_vllm_sleep_state(expected_sleeping=True)

    def _sleep_vllm_server_best_effort(self, context: str) -> None:
        try:
            self._sleep_vllm_server()
        except BaseException as exc:
            eval_logger.error(
                f"Best-effort vLLM deep-sleep cleanup failed after {context}: {exc}"
            )

    @property
    def model(self):
        # returns the model, unwrapping it if using Accelerate
        return self.client

    @property
    def batch_size(self):
        return self.batch_size_per_gpu

    def _forced_action_extra_body(self, required_tag: str) -> dict:
        tokenizer_name = os.getenv("LMMS_EVAL_HF_TOKENIZER", "Qwen/Qwen3-8B")
        if self._forced_action_tokenizer is None:
            self._forced_action_tokenizer = _get_forced_action_tokenizer(tokenizer_name)

        profile = getattr(self, "training_alignment", None)
        if getattr(profile, "native_gemma4", False):
            return profile.forced_action_extra_body(self._forced_action_tokenizer, required_tag, self.force_final_answer_prefix)

        return build_forced_action_openai_extra_body(
            self._forced_action_tokenizer,
            required_tag,
            forced_suffix=self.force_final_answer_prefix if required_tag == "<answer>" else "",
        )

    def _with_forced_action_extra_body(self, payload: dict, required_tag: str) -> dict:
        payload = dict(payload)
        extra_body = dict(payload.get("extra_body") or {})
        extra_body.update(self._forced_action_extra_body(required_tag))
        payload["extra_body"] = extra_body
        return payload

    def _with_final_answer_think_prefill(self, payload: dict, *, close_think: bool = False) -> tuple[dict, str]:
        if not self.force_final_answer_think_prefill:
            return payload, ""

        profile = getattr(self, "training_alignment", None)
        if getattr(profile, "native_gemma4", False):
            from lmms_eval.models.model_utils.gemma4_training_alignment import THOUGHT

            prefill = THOUGHT + self.force_final_answer_think_prefill
            return profile.continuation_request(payload, prefill), prefill

        payload = dict(payload)
        messages = list(payload.get("messages") or [])
        prefill_content = f"{_ASSISTANT_THINK_PREFIX}{self.force_final_answer_think_prefill}"
        if close_think:
            prefill_content += "\n</think>"
        messages.append({"role": "assistant", "content": prefill_content})
        payload["messages"] = messages
        extra_body = dict(payload.get("extra_body") or {})
        extra_body.update({"continue_final_message": True, "add_generation_prompt": False})
        payload["extra_body"] = extra_body
        return payload, prefill_content

    def _forced_final_answer_request(self, payload: dict, *, hard_fallback: bool) -> tuple[dict, str]:
        request_payload = dict(payload)
        # vLLM retains the tool schemas in the rendered prompt by default while
        # disabling OpenAI/Hermes tool-call extraction for this request.
        request_payload["tool_choice"] = "none"
        request_payload = self._with_forced_action_extra_body(request_payload, "<answer>")
        return self._with_final_answer_think_prefill(request_payload, close_think=hard_fallback)

    def _retry_critic_request(self, payload: dict, candidate_content: str) -> dict:
        request_payload = dict(payload)
        profile = getattr(self, "training_alignment", None)
        native = getattr(profile, "native_gemma4", False)
        messages = list(payload.get("messages") or [])
        messages.extend(
            [
                {"role": "assistant", "content": candidate_content},
                {"role": "user", "content": self._training_text("conditional_answer_retry_critic_question", _RETRY_CRITIC_QUESTION)},
                {
                    "role": "assistant",
                    "content": _ASSISTANT_THINK_PREFIX + self._training_text("conditional_answer_retry_critic_assistant_prefill", _RETRY_CRITIC_ASSISTANT_PREFILL),
                },
            ]
        )
        request_payload.update(
            {
                "messages": messages,
                "max_tokens": 1,
                "temperature": 0.0,
                "top_p": 1.0,
                "logprobs": True,
                "top_logprobs": 2,
                "tool_choice": "none",
            }
        )
        extra_body = dict(request_payload.get("extra_body") or {})
        extra_body.update(
            {
                "allowed_token_ids": [
                    int(self._retry_critic_accept_token_id),
                    int(self._retry_critic_retry_token_id),
                ],
                "top_k": -1,
                "continue_final_message": True,
                "add_generation_prompt": False,
            }
        )
        request_payload["extra_body"] = extra_body
        if native:
            # Replace Qwen's partial-assistant block with the literal native
            # Gemma critic prefix from the saved validation configuration.
            native_payload = profile.critic_request(payload, candidate_content)
            request_payload["messages"] = native_payload["messages"]
            request_payload["extra_body"].update(native_payload["extra_body"])
            request_payload["extra_body"]["top_k"] = -1
        return request_payload

    @staticmethod
    def _retry_critic_logprob_items(choice) -> list:
        logprobs = getattr(choice, "logprobs", None)
        content = getattr(logprobs, "content", None) if logprobs is not None else None
        if not content:
            raise ValueError("critic response did not include token logprobs")
        return list(content)

    def _parse_retry_critic_response(self, response) -> tuple[str, float, float, float]:
        choice = response.choices[0]
        model_token = (choice.message.content or "").strip()
        if model_token not in {_RETRY_CRITIC_ACCEPT_TOKEN, _RETRY_CRITIC_RETRY_TOKEN}:
            raise ValueError(f"critic returned invalid token {model_token!r}")

        token_items = self._retry_critic_logprob_items(choice)
        selected_item = token_items[0]
        selected_logprob = float(selected_item.logprob)
        class_logprobs = {}
        for candidate in list(getattr(selected_item, "top_logprobs", None) or []):
            token = str(getattr(candidate, "token", "")).strip()
            if token in {_RETRY_CRITIC_ACCEPT_TOKEN, _RETRY_CRITIC_RETRY_TOKEN}:
                class_logprobs[token] = float(candidate.logprob)

        if {_RETRY_CRITIC_ACCEPT_TOKEN, _RETRY_CRITIC_RETRY_TOKEN} <= class_logprobs.keys():
            accept_logprob = class_logprobs[_RETRY_CRITIC_ACCEPT_TOKEN]
            retry_logprob = class_logprobs[_RETRY_CRITIC_RETRY_TOKEN]
            normalizer = max(accept_logprob, retry_logprob)
            accept_weight = math.exp(accept_logprob - normalizer)
            retry_weight = math.exp(retry_logprob - normalizer)
            p_retry = retry_weight / (accept_weight + retry_weight)
        else:
            selected_probability = min(max(math.exp(selected_logprob), 0.0), 1.0)
            p_retry = (
                selected_probability
                if model_token == _RETRY_CRITIC_RETRY_TOKEN
                else 1.0 - selected_probability
            )

        # With only A/R allowed, the selected token is an argmax. Enforce that
        # invariant against tiny complement/softmax rounding errors at a 0.5 tie.
        if model_token == _RETRY_CRITIC_ACCEPT_TOKEN:
            p_retry = min(p_retry, 0.5)
        else:
            p_retry = max(p_retry, 0.5)
        p_retry = min(max(float(p_retry), 0.0), 1.0)
        return model_token, selected_logprob, 1.0 - p_retry, p_retry

    def _critic_threshold_for_task(self, task):
        from lmms_eval.models.model_utils.critic_thresholds import threshold_for_task

        return threshold_for_task(
            task, getattr(self, "critic_domain_thresholds", {}), self.critic_guided_answer_retry_threshold
        )

    async def _obtain_retry_critic_decision(self, payload: dict, candidate_content: str, *, task=""):
        if getattr(self, "critic_guided_answer_retry_mode", "token") == "reasoning":
            from lmms_eval.models.model_utils.reasoning_replay_critic import (
                reasoning_critic_request,
                reasoning_critic_response,
            )

            profile = getattr(self, "training_alignment", None)
            settings = profile.reasoning_critic_settings if profile else {
                "system_prompt": os.environ["REPLAY_CRITIC_SYSTEM_PROMPT"],
                "prompt_limit": 16384,
                "max_tokens": 5120,
            }
            request, evidence = reasoning_critic_request(
                payload, candidate_content, settings["system_prompt"],
                temperature=getattr(self, "critic_guided_answer_retry_temperature", 0.0),
                top_k=getattr(self, "critic_guided_answer_retry_top_k", -1),
                top_p=getattr(self, "critic_guided_answer_retry_top_p", 1.0),
                max_tokens=settings["max_tokens"],
            )
            evidence["critic_prompt_limit"] = settings["prompt_limit"]
            response = None
            try:
                if await self._training_prompt_tokens(request) > settings["prompt_limit"]:
                    raise ValueError("reasoning critic prompt exceeds configured prompt_length")
                response = await self._create_training_completion(request, critic=True)
                return response, reasoning_critic_response(response, evidence)
            except Exception as exc:
                if response is not None:
                    exc.retry_critic_response = response
                exc.retry_critic_details = evidence
                raise
        threshold = self._critic_threshold_for_task(task)
        request_payload = self._retry_critic_request(payload, candidate_content)
        response = await self._create_training_completion(request_payload, critic=True)
        try:
            model_token, selected_logprob, p_accept, p_retry = self._parse_retry_critic_response(response)
        except Exception as exc:
            exc.retry_critic_response = response
            raise
        decision = (
            _RETRY_CRITIC_RETRY_TOKEN
            if p_retry > threshold
            else _RETRY_CRITIC_ACCEPT_TOKEN
        )
        return response, {
            "model_token": model_token,
            "decision": decision,
            "selected_logprob": selected_logprob,
            "p_accept": p_accept,
            "p_retry": p_retry,
            "threshold": threshold,
        }

    def _training_text(self, key, default):
        profile = getattr(self, "training_alignment", None)
        return profile.multi[key] if profile else default

    def _retry_tool_call_request(self, payload: dict) -> tuple[dict, str]:
        profile = getattr(self, "training_alignment", None)
        if getattr(profile, "native_gemma4", False):
            from lmms_eval.models.model_utils.gemma4_training_alignment import THOUGHT

            prefill = THOUGHT + profile.multi["conditional_answer_retry_assistant_prefill"]
            request_payload = profile.continuation_request(payload, prefill)
            request_payload["tool_choice"] = "auto"
            return self._with_forced_action_extra_body(request_payload, "<tool_call>"), prefill
        request_payload = dict(payload)
        request_payload["tool_choice"] = "auto"
        messages = list(payload.get("messages") or [])
        messages.append(
            {
                "role": "assistant",
                "content": _ASSISTANT_THINK_PREFIX + self._training_text("conditional_answer_retry_assistant_prefill", _RETRY_TOOL_ASSISTANT_PREFILL),
            }
        )
        request_payload["messages"] = messages
        request_payload = self._with_forced_action_extra_body(request_payload, "<tool_call>")
        extra_body = dict(request_payload.get("extra_body") or {})
        extra_body.update({"continue_final_message": True, "add_generation_prompt": False})
        request_payload["extra_body"] = extra_body
        persisted_prefill = _ASSISTANT_THINK_PREFIX
        if self.critic_guided_answer_retry_prefill_profile == "persist":
            persisted_prefill += self._training_text("conditional_answer_retry_assistant_prefill", _RETRY_TOOL_ASSISTANT_PREFILL)
        return request_payload, persisted_prefill

    async def _obtain_retry_tool_call(self, payload: dict):
        request_payload, persisted_prefill = self._retry_tool_call_request(payload)
        response = await self._create_training_completion(request_payload)
        return response, persisted_prefill

    def _record_retry_critic_stats(self, record: dict, usage_response=None) -> None:
        self._retry_critic_stats["decisions"] += 1
        action = record.get("controller_action")
        if action == "accept":
            self._retry_critic_stats["accepts"] += 1
        elif action == "retry":
            self._retry_critic_stats["retries"] += 1
        elif action == "forced_accept":
            self._retry_critic_stats["forced_accepts"] += 1
        elif action == "accept_fallback":
            self._retry_critic_stats["failures"] += 1
        p_retry = record.get("p_retry")
        if isinstance(p_retry, int | float):
            self._retry_critic_stats["retry_probability_sum"] += float(p_retry)
            self._retry_critic_stats["retry_probability_count"] += 1
        if usage_response is not None:
            input_tokens, output_tokens, reasoning_tokens = self._response_token_counts(usage_response)
            self._retry_critic_stats["input_tokens"] += input_tokens
            self._retry_critic_stats["output_tokens"] += output_tokens
            self._retry_critic_stats["reasoning_tokens"] += reasoning_tokens

    @staticmethod
    def _forced_final_answer_failure(response) -> Optional[str]:
        choice = response.choices[0]
        message = choice.message
        if getattr(message, "tool_calls", None):
            return "unexpected_message_tool_calls"
        content = message.content or ""
        if "<tool_call>" in content or "</tool_call>" in content or "<|tool_call>" in content:
            return "textual_tool_call"
        if content.count("<answer>") == 0:
            return "missing_answer_tag"
        if content.count("<answer>") != 1:
            return "multiple_answer_tags"
        if content.count("</answer>") == 0:
            return "missing_answer_close"
        if content.count("</answer>") != 1:
            return "multiple_answer_closes"
        matches = _ANSWER_BLOCK_RE.findall(content)
        if len(matches) != 1:
            return "malformed_answer_block"
        if not matches[0].strip():
            return "empty_answer"
        return None

    @staticmethod
    def _forced_tool_call_failure(response) -> Optional[str]:
        choice = response.choices[0]
        message = choice.message
        content = message.content or ""
        if "<answer>" in content or "</answer>" in content:
            return "early_answer_tag"
        if "</think>" not in content and not (
            getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None)
        ):
            return "missing_think_close"
        if choice.finish_reason != "tool_calls":
            return f"finish_reason_{choice.finish_reason or 'missing'}"
        if not (getattr(message, "tool_calls", None) or []):
            return "missing_tool_call"
        return None

    @staticmethod
    def _budget_forcing_failure_marker(reason: str) -> str:
        return f"[LMMS_EVAL_BUDGET_FORCING_FAILED: {reason}]"

    def _record_forced_final_failure(self, reason: str) -> None:
        self._forced_final_failures[reason] = self._forced_final_failures.get(reason, 0) + 1

    @staticmethod
    def _response_token_counts(response) -> tuple[int, int, int]:
        if not getattr(response, "usage", None):
            return 0, 0, 0
        input_tokens = getattr(response.usage, "prompt_tokens", 0) or 0
        output_tokens = getattr(response.usage, "completion_tokens", 0) or 0
        reasoning_tokens = 0
        details = getattr(response.usage, "completion_tokens_details", None)
        if details:
            reasoning_tokens = getattr(details, "reasoning_tokens", 0) or 0
        return input_tokens, output_tokens, reasoning_tokens

    @staticmethod
    def _tool_call_diagnostic(response) -> str:
        tool_calls = getattr(response.choices[0].message, "tool_calls", None) or []
        rendered = []
        for call in tool_calls:
            function = getattr(call, "function", None)
            rendered.append(
                {
                    "name": getattr(function, "name", None),
                    "arguments": getattr(function, "arguments", None),
                }
            )
        return json.dumps(rendered, ensure_ascii=False)

    async def _obtain_forced_final_answer(self, payload: dict, request_idx: int):
        self._forced_final_stats["primary_attempts"] += 1
        primary_payload, primary_prefill = self._forced_final_answer_request(
            payload,
            hard_fallback=False,
        )
        primary_response = await self._create_training_completion(primary_payload, final=True)
        responses_for_usage = [("primary", primary_response)]
        primary_failure = self._forced_final_answer_failure(primary_response)
        if primary_failure is None:
            self._forced_final_stats["primary_successes"] += 1
            return primary_response, primary_prefill, responses_for_usage

        self._record_forced_final_failure(f"primary_{primary_failure}")
        if getattr(self, "training_alignment", None):
            # Training gives this turn one generation, with no second closed-think attempt.
            return primary_response, primary_prefill, responses_for_usage
        eval_logger.warning(
            "Forced final-answer primary request failed "
            f"for request {request_idx}: reason={primary_failure} "
            f"finish_reason={primary_response.choices[0].finish_reason} "
            f"tool_calls={self._tool_call_diagnostic(primary_response)} "
            f"content={repr((primary_response.choices[0].message.content or '')[-1000:])}"
        )
        self._forced_final_stats["fallback_attempts"] += 1
        fallback_payload, fallback_prefill = self._forced_final_answer_request(
            payload,
            hard_fallback=True,
        )
        fallback_response = await self._create_training_completion(fallback_payload, final=True)
        responses_for_usage.append(("fallback", fallback_response))
        fallback_failure = self._forced_final_answer_failure(fallback_response)
        if fallback_failure is not None:
            self._record_forced_final_failure(f"fallback_{fallback_failure}")
            eval_logger.error(
                "Hard forced final-answer retry failed "
                f"for request {request_idx}: reason={fallback_failure}, "
                f"finish_reason={fallback_response.choices[0].finish_reason}, "
                f"tool_calls={self._tool_call_diagnostic(fallback_response)}, "
                f"content={repr((fallback_response.choices[0].message.content or '')[-1000:])}"
            )
            # Treat a malformed forced answer as a bad sample, not a bad run.
            # Replacing the response also prevents permissive task parsers from
            # extracting a plausible-looking answer from malformed content.
            fallback_response.choices[0].message.content = (
                f"[LMMS_EVAL_FORCED_FINAL_ANSWER_FAILED: {fallback_failure}]"
            )
            return fallback_response, "", responses_for_usage

        self._forced_final_stats["fallback_successes"] += 1
        return fallback_response, fallback_prefill, responses_for_usage

    async def _training_prompt_tokens(self, payload):
        # Ask the serving processor so crop image tokens count toward the response
        # budget, just as tool-response tokens do in training's response_mask.
        body = {key: payload[key] for key in ("model", "messages", "tools") if key in payload}
        extra = payload.get("extra_body") or {}
        for key in ("continue_final_message", "add_generation_prompt", "chat_template_kwargs"):
            if key in extra:
                body[key] = extra[key]

        def tokenize():
            request = urllib.request.Request(
                f"{self._vllm_server_root()}/tokenize", data=json.dumps(body).encode(),
                headers={"Content-Type": "application/json", "Authorization": f"Bearer {self.client.api_key}"},
                method="POST",
            )
            with _VLLM_ADMIN_OPENER.open(request, timeout=float(self.timeout)) as response:
                result = json.loads(response.read())
                if getattr(self, "context_budget_enabled", False) and result.get("max_model_len") != self.context_budget_limit:
                    raise ValueError(f"Serving context differs from configured limit: {result.get('max_model_len')} != {self.context_budget_limit}")
                return int(result["count"])

        return await asyncio.to_thread(tokenize)

    async def _context_final_due(self, payload):
        if not getattr(self, "context_budget_enabled", False):
            return False
        prompt_tokens = await self._training_prompt_tokens(payload)
        # Keep room for the final prefill and answer before another tool turn.
        return self.context_budget_limit - prompt_tokens <= self.context_budget_reserve + 1024

    async def _create_training_completion(self, payload, *, critic=False, final=False):
        request = dict(payload)
        trace_id = request.pop("_eval_trace_id", None)
        initial_tokens = request.pop("_training_initial_tokens", None)
        profile = getattr(self, "training_alignment", None)
        if profile and not critic and initial_tokens is not None:
            consumed = await self._training_prompt_tokens(request) - initial_tokens
            remaining = profile.data["max_response_length"] - consumed
            if remaining <= 0:
                raise RuntimeError("Training response budget exhausted before generation")
            request["max_tokens"] = min(request["max_tokens"], remaining)
        if getattr(self, "context_budget_enabled", False):
            # Use the serving multimodal processor, including every accumulated
            # crop and the actual template/prefill, rather than a text estimate.
            prompt_tokens = await self._training_prompt_tokens(request)
            reserve = 256 if final or critic else self.context_budget_reserve
            available = self.context_budget_limit - prompt_tokens - reserve
            if available <= 0:
                raise RuntimeError(f"Context budget exhausted: input_tokens={prompt_tokens}, reserve={reserve}")
            if request["max_tokens"] > available:
                eval_logger.info(f"Context budget: reducing max_tokens={request['max_tokens']} to {available}; input_tokens={prompt_tokens}, final={final}, critic={critic}")
                request["max_tokens"] = available
        response = await self.client.chat.completions.create(**request)
        # Opt-in debug artifacts retain exact requests, images and parsed tool calls.
        # The internal trace key is never sent to the OpenAI-compatible server.
        trace_dir = os.getenv("LMMS_EVAL_REQUEST_TRACE_DIR")
        if trace_dir and trace_id:
            path = os.path.join(trace_dir, trace_id)
            os.makedirs(path, exist_ok=True)
            with open(os.path.join(path, f"{time.time_ns()}_{uuid.uuid4().hex}.json"), "w", encoding="utf-8") as stream:
                json.dump({"request": request, "response": response.model_dump(mode="json")}, stream, ensure_ascii=False)
        return response

    def _rewrite_tool_schemas_for_model(self, tool_list):
        rewritten = []
        for tool in tool_list:
            if tool.get("type") != "function":
                rewritten.append(tool)
                continue
            function = tool.get("function", {})
            parameters = function.get("parameters")
            if not isinstance(parameters, dict):
                rewritten.append(tool)
                continue
            properties = parameters.get("properties")
            if not isinstance(properties, dict):
                rewritten.append(tool)
                continue

            tool_copy = json.loads(json.dumps(tool))
            properties = tool_copy["function"]["parameters"].setdefault("properties", {})
            for hidden_arg in _HIDDEN_TOOL_ARGS:
                properties.pop(hidden_arg, None)
            required = tool_copy["function"]["parameters"].get("required")
            if isinstance(required, list):
                tool_copy["function"]["parameters"]["required"] = [key for key in required if key not in _HIDDEN_TOOL_ARGS]
            rewritten.append(tool_copy)
        return rewritten

    @property
    def rank(self):
        return self._rank

    @property
    def world_size(self):
        return self._world_size

    def _resolve_task_system_prompt(self, task: str) -> Optional[str]:
        """Build the turn-0 system prompt for a task by concatenating, in order:

        1. the run-wide ``system_prompt`` model_arg (if set, e.g. via SYSTEM_PROMPT);
        2. each file listed in the task yaml's ``metadata.system_prompt_files``.

        This is what makes the prompt modular: a task includes a toolset template
        (``prompts/tools/system_base.txt`` + ``system_zoom.txt`` / ``system_draw.txt``)
        and the fragments concatenate here. The model's chat template then appends
        its own ``# Tools`` / ``<tools>`` schema block after this system text, so the
        final system message is base (+ run-wide) (+ toolset) (+ rendered schemas).

        Paths in ``system_prompt_files`` resolve as-is if absolute/cwd-relative,
        else against the persistent-exploration-eval package root. A listed-but-missing file is a
        hard error (a silently dropped system prompt would corrupt the whole run).
        Returns ``None`` when neither source yields any text. Cached per task.
        """
        if getattr(self, "training_alignment", None):
            return self.training_alignment.system_prompt
        if task in self._task_system_prompt_cache:
            return self._task_system_prompt_cache[task]
        # A run-specific protocol (e.g. Qwen3 XML) must replace task fragments,
        # since concatenating it with the default JSON rules gives conflicting calls.
        if os.getenv("RUN_SYSTEM_PROMPT_MODE", "task_metadata").lower() == "run_env":
            path = os.getenv("TRAINING_SYSTEM_PROMPT") or os.getenv("SYSTEM_PROMPT")
            if not path:
                raise ValueError("RUN_SYSTEM_PROMPT_MODE=run_env requires a system prompt file")
            if not os.path.isfile(path):
                path = os.path.join(_VERO_EVAL_ROOT, path)
            with open(path, "r", encoding="utf-8", newline="") as stream:
                resolved = stream.read()
            self._task_system_prompt_cache[task] = resolved
            return resolved
        fragments: List[str] = []
        if self.system_prompt:
            fragments.append(self.system_prompt.strip())
        meta = self.task_metadata.get(task) or {}
        files = meta.get("system_prompt_files") or []
        if isinstance(files, str):
            files = [files]
        # Some checkpoints' chat templates append a fixed "\n\n" + tool block right
        # after the system message, so a fragment's own trailing newline is part of the
        # byte-exact seam (e.g. Vero training: Layer A ends "...```\n" and the template
        # adds "\n\n" -> 3 newlines before "# Tools"). Opt in per task to preserve it;
        # default strips (intended for the single training-faithful fragment case).
        preserve_trailing_newline = bool(meta.get("system_prompt_preserve_trailing_newline"))
        for rel in files:
            path = rel if os.path.isabs(rel) else rel
            if not os.path.isfile(path):
                path = os.path.join(_VERO_EVAL_ROOT, rel)
            if not os.path.isfile(path):
                raise ValueError(f"Task '{task}': system_prompt_files entry '{rel}' not found " f"(looked at '{rel}' and '{os.path.join(_VERO_EVAL_ROOT, rel)}'). Fix the path in the task yaml metadata.")
            with open(path, "r") as fh:
                content = fh.read()
            fragments.append(content if preserve_trailing_newline else content.strip())
        resolved = "\n\n".join(f for f in fragments if f) or None
        self._task_system_prompt_cache[task] = resolved
        if resolved:
            eval_logger.info(f"Task '{task}': system prompt = {len(fragments)} fragment(s), {len(resolved)} chars")
        return resolved

    def _resolve_task_tools(self, task: str):
        """Resolve the per-task tool view from the task yaml's `metadata` block:

        - ``allowed_tools``: list of MCP tool names or fnmatch patterns (e.g.
          ``["image_zoom_in_tool"]`` or ``["draw_*"]``). Absent/empty = offer
          every tool the MCP server exposes (backwards-compatible default).
        - ``extra_tools``: extra OpenAI function specs offered to the model but
          NOT backed by the MCP server (answer-protocol tools such as
          screenspot's ``computer_use``). Given verbatim in the yaml.
        - ``terminal_tools``: tool names whose call ENDS the episode — the call
          is recorded in the transcript as the final answer and never executed.

        Returns ``(functions, terminal_tool_names)``; cached per task.
        """
        if getattr(self, "training_alignment", None):
            return [self.training_alignment.tool_schema], set()
        if task in self._task_tools_cache:
            return self._task_tools_cache[task]
        meta = self.task_metadata.get(task) or {}
        functions = list(self._tool_functions or [])
        allowed = meta.get("allowed_tools")
        if allowed:
            if isinstance(allowed, str):
                allowed = [p.strip() for p in allowed.split(",") if p.strip()]
            functions = [f for f in functions if any(fnmatch.fnmatchcase(f["function"]["name"], pat) for pat in allowed)]
            if not functions:
                server_tools = [f["function"]["name"] for f in (self._tool_functions or [])]
                raise ValueError(f"Task '{task}': allowed_tools={allowed} matched none of the MCP server tools {server_tools}. " f"Fix the task yaml metadata or the MCP server at the configured mcp_server_path.")
        extra = meta.get("extra_tools") or []
        functions = functions + list(extra)
        # Per-task tool-schema overrides: replace the FastMCP-introspected `description`
        # and/or `parameters` of a named tool with an authored schema. Used to render the
        # offered tool JSON byte-identical to the RL training tool schema (whose property
        # order/types/keys differ from the Python-signature-derived one). The tool stays
        # backed by the MCP server (execution by name); only the advertised schema changes.
        overrides = meta.get("tool_schema_overrides") or {}
        if overrides:
            pinned = []
            for f in functions:
                ov = overrides.get(f.get("function", {}).get("name"))
                if ov:
                    f = json.loads(json.dumps(f))  # deep copy; don't mutate the shared MCP spec
                    if "description" in ov:
                        f["function"]["description"] = ov["description"]
                    if "parameters" in ov:
                        f["function"]["parameters"] = json.loads(json.dumps(ov["parameters"]))
                pinned.append(f)
            functions = pinned
        terminal = {str(name) for name in (meta.get("terminal_tools") or [])}
        self._task_tools_cache[task] = (functions, terminal)
        offered = [f["function"]["name"] for f in functions]
        eval_logger.info(f"Task '{task}': offering tools {offered}" + (f" (terminal: {sorted(terminal)})" if terminal else ""))
        return functions, terminal

    def _tool_image_registered_text(self, mode, tool_name, tool_args, assigned_idx):
        """Model-visible text that replaces a tool's 'Saved drawn image to:' sentinel.

        For the zoom tool under the training-faithful image_indexing modes, reproduce the
        RL environment's <tool_response> wording 'Zoomed in on the image to the region
        [x1, y1, x2, y2] with label {label}.' (built from the call's bbox_2d/label):
          - ``training_faithful``: that sentence ONLY (training never announced new
            indices; 581/581 training calls used img_idx 0).
          - ``multi_index``: that sentence + a hint that the crop is available as a new
            img_idx and can be zoomed into (the only divergence from training).
        Otherwise (default) keep the generic 'Registered image as img_idx N.'.
        """
        if mode in ("training_faithful", "multi_index") and tool_name == "image_zoom_in_tool":
            bbox = (tool_args or {}).get("bbox_2d")
            label = (tool_args or {}).get("label")
            if isinstance(bbox, (list, tuple)):
                if getattr(getattr(self, "training_alignment", None), "native_gemma4", False):
                    bbox_str = str(list(bbox))
                else:
                    bbox_str = "[" + ", ".join(str(int(v)) for v in bbox) + "]"
            else:
                bbox_str = str(bbox)
            wording = f"Zoomed in on the image to the region {bbox_str} with label {label}."
            if mode == "multi_index":
                # Inform-only: announce the crop is addressable, without nudging the model to
                # always zoom further (avoids biasing it toward an extra "closer look").
                wording += f" The cropped view is now available as img_idx {assigned_idx}."
            return wording
        return f"Registered image as img_idx {assigned_idx}."

    def loglikelihood(self, requests: List[Instance]) -> List[Tuple[float, bool]]:
        assert False, "TODO, not implemented"

    def generate_until_multi_round(self, requests) -> List[str]:
        raise NotImplementedError("TODO: Implement multi-round generation for LLaVAHF")

    async def maybe_forward_with_tool(self, request: Instance, idx: int):
        """
        Forward the request to the OpenAI API, using tools if available.
        This method is designed to handle chat messages and tool calls in an asynchronous manner.
        It retrieves the chat messages from the request, prepares the payload, and sends it to the OpenAI API.
        If the response indicates that tool calls are needed, it will call the tools using the MCP client and continue the conversation until a final response is received.
        :param request: The request instance containing the chat messages and other parameters.
        :param idx: The index of the request in the batch. (Use to restore the original order of responses)
        """
        ctx, doc_to_messages, gen_kwargs, doc_id, task, split = request.args
        chat_messages = doc_to_messages(self.task_dict[task][split][doc_id])
        chat_messages: ChatMessages = ChatMessages(**{"messages": chat_messages})
        video_kwargs = {"max_pixels": self.max_pixels, "min_pixels": self.min_pixels}
        if self.fps is not None:
            video_kwargs["fps"] = self.fps
        else:
            video_kwargs["nframes"] = self.nframes
        if self.max_frames is not None:
            video_kwargs["max_frames"] = self.max_frames
        if self.is_qwen3_vl:
            messages = chat_messages.to_qwen3_vl_openai_messages(video_kwargs)
        else:
            messages = chat_messages.to_openai_messages(video_kwargs)
        profile = getattr(self, "training_alignment", None)
        if profile:
            messages = profile.align_user_prompt(messages, task, self.task_dict[task][split][doc_id])
        task_system_prompt = self._resolve_task_system_prompt(task)
        messages = self._apply_system_prompt(messages, task_system_prompt) if task_system_prompt else messages
        images, videos, audios = chat_messages.extract_media()
        # Track every image file that belongs to this request so we can
        # delete them after the model finishes answering (avoids unbounded
        # accumulation of input copies and chained tool outputs on disk).
        request_image_paths: List[str] = []
        image_registry: Dict[int, str] = {}
        next_img_idx = 0

        def _cleanup_request_images():
            for path in request_image_paths:
                try:
                    os.remove(path)
                except FileNotFoundError:
                    pass
                except OSError as exc:
                    eval_logger.debug(f"Failed to remove tool image {path}: {exc}")

        # Image addressing by index (img_idx), matching training: the model is told
        # each image's img_idx and never sees a path; tools resolve img_idx -> path
        # via the registry passed in LMMS_EVAL_IMAGE_REGISTRY. Tool-produced images
        # get appended with a new img_idx below (mirrors agent_data.image_data growth).
        image_registry: Dict[int, str] = {}
        # Provenance for original-resolution crop compositing: {img_idx: (orig_idx, [x1,y1,x2,y2])}
        # = the native-pixel rectangle of original `orig_idx` that this image represents. Seeded
        # for each initial image as (itself, full native bounds); crops get composited regions.
        # A zoom into a crop is remapped to crop the ROOT original at native resolution (see
        # _composite_zoom_to_original) so chained / multi-index zooms keep full detail.
        image_provenance: Dict[int, tuple] = {}
        next_img_idx = 0
        task_metadata = self.task_metadata.get(task) or {}
        critic_threshold = self._critic_threshold_for_task(task)
        profile = getattr(self, "training_alignment", None)
        if profile:
            task_metadata = dict(task_metadata, show_initial_image_index_hint=False,
                                 restore_assistant_forced_prefill=not getattr(profile, "native_gemma4", False),
                                 image_indexing=profile.image_indexing)
            gen_kwargs = {**gen_kwargs, **profile.generation_kwargs()}
        show_initial_image_index_hint = parse_bool(task_metadata.get("show_initial_image_index_hint", True))

        if self.mcp_client is not None:
            for image_idx, image in enumerate(images):
                # Save losslessly at ORIGINAL resolution (PNG), matching training's
                # use_original_resolution_source — NOT a lossy/downscaled JPEG.
                image_path = os.path.join(self.work_dir, f"{uuid.uuid4()}.png")
                image.save(image_path, format="PNG")
                request_image_paths.append(image_path)
                image_registry[next_img_idx] = image_path
                image_provenance[next_img_idx] = (next_img_idx, [0.0, 0.0, float(image.width), float(image.height)])
                next_img_idx += 1
                if show_initial_image_index_hint:
                    messages[-1]["content"].append(
                        {
                            "type": "text",
                            "text": f"\nImage {image_idx} is available as img_idx={image_idx}.",
                        }
                    )
            for video_idx, video in enumerate(videos):
                messages[-1]["content"].append(
                    {
                        "type": "text",
                        "text": f"\nVideo {video_idx} has video path: {video}",
                    }
                )

        try:
            payload = {"messages": messages}
            if os.getenv("LMMS_EVAL_REQUEST_TRACE_DIR"):
                safe_task = re.sub(r"[^A-Za-z0-9_.-]", "_", str(task))
                payload["_eval_trace_id"] = f"{safe_task}_{doc_id}_{idx}_{uuid.uuid4().hex}"
            payload["model"] = self.model_version
            all_response = ""
            total_input_tokens = 0
            total_output_tokens = 0
            total_reasoning_tokens = 0
            critic_input_tokens = 0
            critic_output_tokens = 0
            critic_reasoning_tokens = 0

            if "max_new_tokens" not in gen_kwargs:
                gen_kwargs["max_new_tokens"] = 1024
            if "temperature" not in gen_kwargs:
                gen_kwargs["temperature"] = 0
            payload["max_tokens"] = gen_kwargs["max_new_tokens"]
            payload["temperature"] = gen_kwargs["temperature"]
            if gen_kwargs.get("top_p") is not None:
                payload["top_p"] = gen_kwargs["top_p"]
            for _sp in ("presence_penalty", "frequency_penalty"):
                if gen_kwargs.get(_sp) is not None:
                    payload[_sp] = gen_kwargs[_sp]
            _extra_body = {}
            for _sp in ("top_k", "repetition_penalty", "min_p"):
                if gen_kwargs.get(_sp) is not None:
                    _extra_body[_sp] = gen_kwargs[_sp]
            if _extra_body:
                payload["extra_body"] = _extra_body

            functions = []
            terminal_tools = set()
            if self.mcp_client is not None:
                if self._tool_functions is None:
                    self._tool_functions = await self.mcp_client.get_function_list()
                functions, terminal_tools = self._resolve_task_tools(task)
                payload["tools"] = functions
                payload["tool_choice"] = "auto"
            if self.force_tool_calls_until_final_turn:
                if self.mcp_client is None or not payload.get("tools"):
                    raise ValueError(
                        "force_tool_calls_until_final_turn requires at least one non-terminal MCP tool"
                    )
                if terminal_tools:
                    raise ValueError(
                        "force_tool_calls_until_final_turn requires a final <answer> block and is incompatible "
                        f"with terminal tools for task {task!r}: {sorted(terminal_tools)}"
                    )
            if getattr(self, "first_candidate_answer_turn", 0) and (not functions or terminal_tools):
                raise ValueError("first_candidate_answer_turn requires MCP tools and an <answer> final-answer protocol")
            if getattr(self, "always_retry_answers", False) and (not functions or terminal_tools):
                raise ValueError("always_retry_answers requires MCP tools and an <answer> final-answer protocol")
            # Per-task image-indexing mode controls the model-visible tool_response wording
            # (training_faithful | multi_index | None=default). See _tool_image_registered_text.
            image_indexing_mode = task_metadata.get("image_indexing")
            restore_assistant_forced_prefill = parse_bool(task_metadata.get("restore_assistant_forced_prefill", False))

            tool_iteration = 0
            tool_image_count = 0
            terminal_tool_called = False
            forced_final_answer_requested = False
            forced_final_answer_valid = False
            retry_depth = 0
            retry_critic_records = []
            answer_retry_records = []
            first_candidate_seen_turn = None
            first_candidate_forced_turn = None
            candidate_answer_turns = []
            candidate_forced_turns = []
            context_final_due = False
            training_remaining = None
            if profile:
                payload["_training_initial_tokens"] = await self._training_prompt_tokens(payload)
                training_remaining = profile.data["max_response_length"]

            def _retry_critic_metadata():
                if getattr(self, "always_retry_answers", False):
                    return {"answer_retry": {
                        "enabled": True,
                        "critic_used": False,
                        "prefill_profile": self.critic_guided_answer_retry_prefill_profile,
                        "decisions": answer_retry_records,
                    }}
                if not self.critic_guided_answer_retry:
                    return None
                probabilities = [
                    float(record["p_retry"])
                    for record in retry_critic_records
                    if isinstance(record.get("p_retry"), int | float)
                ]
                metadata = {
                    "retry_critic": {
                        "enabled": True,
                        "source": getattr(self, "critic_guided_answer_retry_source", "policy"),
                        "threshold": critic_threshold,
                        "prefill_profile": self.critic_guided_answer_retry_prefill_profile,
                        "decisions": retry_critic_records,
                        "summary": {
                            "decisions": len(retry_critic_records),
                            "accepts": sum(r.get("controller_action") == "accept" for r in retry_critic_records),
                            "retries": sum(r.get("controller_action") == "retry" for r in retry_critic_records),
                            "forced_accepts": sum(
                                r.get("controller_action") == "forced_accept" for r in retry_critic_records
                            ),
                            "failures": sum(
                                r.get("controller_action") == "accept_fallback" for r in retry_critic_records
                            ),
                            "mean_retry_probability": (
                                sum(probabilities) / len(probabilities) if probabilities else None
                            ),
                            "critic_token_usage": {
                                "input_tokens": critic_input_tokens,
                                "output_tokens": critic_output_tokens,
                                "reasoning_tokens": critic_reasoning_tokens,
                                "total_tokens": (
                                    critic_input_tokens
                                    + critic_output_tokens
                                    + critic_reasoning_tokens
                                ),
                            },
                        },
                    }
                }
                if getattr(self, "critic_guided_answer_retry_mode", "token") == "reasoning":
                    metadata["retry_critic"]["mode"] = "reasoning"
                    metadata["retry_critic"]["sampling_params"] = {
                        "temperature": getattr(self, "critic_guided_answer_retry_temperature", 0.0),
                        "top_k": getattr(self, "critic_guided_answer_retry_top_k", -1),
                        "top_p": getattr(self, "critic_guided_answer_retry_top_p", 1.0),
                        "n": 1,
                    }
                if getattr(self, "first_candidate_answer_turn", 0) or getattr(self, "record_answer_candidates", False):
                    metadata["first_candidate_schedule"] = {
                        "deadline_turn": getattr(self, "first_candidate_answer_turn", 0),
                        "forced_turn": first_candidate_forced_turn,
                        "first_candidate_turn": first_candidate_seen_turn,
                    }
                if getattr(self, "followup_candidate_answer_turn", 0):
                    metadata["candidate_answer_schedule"] = {
                        "deadline_turns": [self.first_candidate_answer_turn, self.followup_candidate_answer_turn],
                        "candidate_turns": list(candidate_answer_turns),
                        "forced_turns": list(candidate_forced_turns),
                    }
                return metadata

            def _add_usage(usage_response, *, source: str):
                nonlocal total_input_tokens, total_output_tokens, total_reasoning_tokens
                nonlocal critic_input_tokens, critic_output_tokens, critic_reasoning_tokens
                input_tokens, output_tokens, reasoning_tokens = self._response_token_counts(usage_response)
                total_input_tokens += input_tokens
                total_output_tokens += output_tokens
                total_reasoning_tokens += reasoning_tokens
                if source == "retry_critic":
                    critic_input_tokens += input_tokens
                    critic_output_tokens += output_tokens
                    critic_reasoning_tokens += reasoning_tokens
                log_usage(
                    model_name=self.model_version,
                    task_name=task,
                    input_tokens=input_tokens,
                    output_tokens=output_tokens,
                    reasoning_tokens=reasoning_tokens,
                    source=source,
                )
                return input_tokens, output_tokens, reasoning_tokens

            def _retry_skip_reason():
                if context_final_due:
                    return "context_final_answer_reserve"
                if profile and training_remaining <= profile.multi["final_answer_token_margin"]:
                    return "final_answer_token_margin"
                if tool_iteration >= self.max_tool_iterations:
                    return "max_tool_iterations"
                if tool_image_count >= self.max_tool_images:
                    return "max_tool_images"
                retry_tools = [
                    function
                    for function in payload.get("tools", [])
                    if function.get("function", {}).get("name") not in terminal_tools
                ]
                if not retry_tools:
                    return "no_retry_tool_available"
                return None

            async def _maybe_replace_answer_with_retry_tool(current_response, current_prefill):
                nonlocal retry_depth, first_candidate_seen_turn
                message = current_response.choices[0].message
                if getattr(message, "tool_calls", None):
                    return current_response, current_prefill, False
                candidate_content = f"{current_prefill}{message.content or ''}"
                if getattr(profile, "native_gemma4", False):
                    candidate_content = profile.candidate_content(message, current_prefill)
                if "<answer>" not in candidate_content:
                    return current_response, current_prefill, False
                if first_candidate_seen_turn is None:
                    first_candidate_seen_turn = tool_iteration + 1
                candidate_answer_turns.append(tool_iteration + 1)
                if getattr(self, "always_retry_answers", False):
                    # Same controller as the saved step-200 ablation: discard each
                    # candidate, force a zoom with persist prefill, never query a critic.
                    skipped_reason = _retry_skip_reason()
                    if skipped_reason is None:
                        retry_depth += 1
                    record = {
                        "candidate_turn": tool_iteration + 1,
                        "candidate_content": candidate_content,
                        "controller": "always_retry_answer",
                        "critic_used": False,
                        "retry_feasible": skipped_reason is None,
                        "controller_action": "retry" if skipped_reason is None else "forced_accept",
                        "retry_depth": retry_depth,
                        "skip_or_error_reason": skipped_reason,
                    }
                    answer_retry_records.append(record)
                    eval_logger.info(f"Unconditional answer retry: request={idx} task={task} record={json.dumps(record, sort_keys=True)}")
                    if skipped_reason is not None:
                        return current_response, current_prefill, False
                    retry_response, persisted_prefill = await self._obtain_retry_tool_call(payload)
                    _add_usage(retry_response, source="model")
                    return retry_response, persisted_prefill, True
                if not self.critic_guided_answer_retry:
                    return current_response, current_prefill, False
                critic_candidate = candidate_content
                if restore_assistant_forced_prefill:
                    critic_candidate = _restore_assistant_forced_prefill(critic_candidate)

                critic_response = None
                try:
                    critic_payload = dict(payload)
                    if functions:
                        critic_payload["tools"] = functions
                        critic_payload["tool_choice"] = "auto"
                    if getattr(self, "critic_guided_answer_retry_source", "policy") == "ground_truth":
                        from lmms_eval.models.model_utils.ground_truth_critic import grade_visual_probe_candidate

                        # Serialize wake/grade/sleep cycles across concurrent
                        # trajectories; leave policy generation asynchronous.
                        async with self._ground_truth_critic_lock:
                            record = await asyncio.to_thread(
                                grade_visual_probe_candidate,
                                self.task_dict[task][split][doc_id],
                                critic_candidate,
                                task,
                            )
                    else:
                        critic_response, record = await self._obtain_retry_critic_decision(
                            critic_payload,
                            critic_candidate,
                            task=task,
                        )
                    _add_usage(critic_response, source="retry_critic")
                    skipped_reason = _retry_skip_reason()
                    retry_feasible = skipped_reason is None
                    if record["decision"] == "R" and retry_feasible:
                        action = "retry"
                        retry_depth += 1
                    elif record["decision"] == "R":
                        action = "forced_accept"
                    else:
                        action = "accept"
                    record.update(
                        {
                            "candidate_turn": tool_iteration + 1,
                            "retry_feasible": retry_feasible,
                            "controller_action": action,
                            "retry_depth": retry_depth,
                            "skip_or_error_reason": skipped_reason,
                        }
                    )
                except Exception as exc:  # Critic failures retain the candidate.
                    if getattr(self, "critic_guided_answer_retry_source", "policy") == "ground_truth":
                        # An unavailable GT grader must not silently become an
                        # accept-all critic and produce a mislabeled evaluation.
                        raise RuntimeError("ground_truth critic grading failed") from exc
                    if critic_response is None:
                        critic_response = getattr(exc, "retry_critic_response", None)
                    if critic_response is not None:
                        _add_usage(critic_response, source="retry_critic")
                    record = {
                        "candidate_turn": tool_iteration + 1,
                        "model_token": None,
                        "decision": None,
                        "selected_logprob": None,
                        "p_accept": None,
                        "p_retry": None,
                        "threshold": critic_threshold,
                        "retry_feasible": None,
                        "controller_action": "accept_fallback",
                        "retry_depth": retry_depth,
                        "skip_or_error_reason": f"{type(exc).__name__}: {exc}",
                    }
                    record.update(getattr(exc, "retry_critic_details", {}))

                if getattr(self, "record_answer_candidates", False):
                    record["candidate_content"] = critic_candidate
                retry_critic_records.append(record)
                self._record_retry_critic_stats(record, usage_response=critic_response)
                eval_logger.info(
                    "Retry critic decision: "
                    f"request={idx} task={task} record={json.dumps(record, sort_keys=True)}"
                )
                if record["controller_action"] != "retry":
                    return current_response, current_prefill, False

                retry_response, persisted_prefill = await self._obtain_retry_tool_call(payload)
                _add_usage(retry_response, source="model")
                return retry_response, persisted_prefill, True

            initial_request_payload = payload
            if self.force_tool_calls_until_final_turn:
                initial_request_payload = self._with_forced_action_extra_body(payload, "<tool_call>")
            response_prefill = ""
            context_final_due = await self._context_final_due(initial_request_payload)
            if context_final_due and not terminal_tools:
                response, response_prefill, initial_attempts = await self._obtain_forced_final_answer(payload, idx)
                for _, initial_response in initial_attempts:
                    _add_usage(initial_response, source="model")
            else:
                response = await self._create_training_completion(initial_request_payload)
                _add_usage(response, source="model")
            response, response_prefill, critic_forced_tool = await _maybe_replace_answer_with_retry_tool(
                response,
                response_prefill,
            )
            last_response = response.choices[0].message.content or ""
            try:
                if response_prefill:
                    all_response += response_prefill
                all_response += last_response
            except Exception as e:
                all_response += f"Error: {str(e)}"

            if (self.force_tool_calls_until_final_turn and not context_final_due) or critic_forced_tool:
                forced_tool_failure = self._forced_tool_call_failure(response)
                if forced_tool_failure is not None:
                    eval_logger.error(
                        "Budget-forced initial tool turn failed "
                        f"for request {idx}: reason={forced_tool_failure}, "
                        f"finish_reason={response.choices[0].finish_reason}, "
                        f"tool_calls={self._tool_call_diagnostic(response)}, "
                        f"content={repr((response.choices[0].message.content or '')[-1000:])}"
                    )
                    all_response += f"\n{self._budget_forcing_failure_marker(forced_tool_failure)}"
                    return all_response, idx, TokenCounts(
                        input_tokens=total_input_tokens,
                        output_tokens=total_output_tokens,
                        reasoning_tokens=total_reasoning_tokens,
                    ), _retry_critic_metadata()
            while response.choices[0].finish_reason == "tool_calls" and tool_iteration < self.max_tool_iterations:
                tool_iteration += 1
                assistant_message = response.choices[0].message
                assistant_content = f"{response_prefill}{assistant_message.content or ''}"
                if restore_assistant_forced_prefill:
                    assistant_content = _restore_assistant_forced_prefill(assistant_content)
                assistant_dict = {"role": "assistant", "content": assistant_content}
                _preserve_assistant_reasoning(assistant_message, assistant_dict)
                if getattr(profile, "native_gemma4", False):
                    assistant_dict = profile.assistant_message(assistant_message, response_prefill)
                if assistant_message.tool_calls:
                    assistant_dict["tool_calls"] = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                        }
                        for tc in assistant_message.tool_calls
                    ]
                messages.append(assistant_dict)

                pending_user_images = []
                active_calls = assistant_message.tool_calls or []
                if profile:
                    active_calls = active_calls[:profile.multi["max_parallel_calls"]]
                for call in active_calls:
                    eval_logger.debug(f"Calling {call.function.name}...")
                    all_response += f"<tool_call>{call.function.name} {call.function.arguments}</tool_call>"
                    if call.function.name in terminal_tools:
                        # Answer-protocol tool (e.g. screenspot's computer_use): the
                        # call IS the final answer. It is recorded in the transcript
                        # above for the task's parser; never executed, episode ends.
                        terminal_tool_called = True
                        break
                    try:
                        tool_args = json.loads(call.function.arguments) if call.function.arguments else {}
                        call_args = tool_args  # what the tool actually runs (may be remapped below)
                        if call.function.name == "image_zoom_in_tool" and "bbox_2d" in tool_args:
                            # Keep model-visible calls/responses in the model's coordinate order.
                            # Convert before native-resolution crop compositing, exactly once.
                            call_args = dict(tool_args)
                            call_args["bbox_2d"] = _zoom_bbox_to_xyxy(
                                tool_args["bbox_2d"], task_metadata.get("zoom_bbox_coordinate_order", "xyxy")
                            )
                            # Training defaults an omitted image index to the primary
                            # image. Apply it before compositing so crop provenance
                            # is retained, without changing the model-visible call.
                            call_args.setdefault("img_idx", 0)
                        zoom_provenance = None   # (orig_idx, native_subregion) of the crop this call yields
                        if "img_idx" in call_args:
                            _idx = int(call_args["img_idx"])
                            if _idx not in image_registry:
                                raise ValueError(f"img_idx {_idx} is not available. Available indices: {sorted(image_registry)}.")
                            # Original-resolution crop: remap a zoom into ANY image to crop the ROOT
                            # original at native resolution (composite the 0-1000 bbox through
                            # provenance). No-op for originals (img_idx whose region is the full image).
                            # tool_args is kept as-is for the model-visible tool_response wording; only
                            # the executed call_args is remapped.
                            if call.function.name == "image_zoom_in_tool" and isinstance(tool_args.get("bbox_2d"), (list, tuple)) and len(tool_args["bbox_2d"]) >= 4:
                                composited = _composite_zoom_to_original(image_provenance, _idx, call_args["bbox_2d"])
                                if composited is not None:
                                    orig_idx, nsub, bbox_prime = composited
                                    zoom_provenance = (orig_idx, nsub)
                                    call_args = dict(call_args)
                                    call_args["img_idx"] = orig_idx
                                    call_args["bbox_2d"] = bbox_prime
                        # Snapshot the registry into the tool subprocess so it can map img_idx -> path.
                        tool_env = {_IMAGE_REGISTRY_ENV: json.dumps({str(k): v for k, v in image_registry.items()})}
                        result = await self.mcp_client.run_tool(call.function.name, call_args, env=tool_env)
                        raw_parts = []
                        for content in result.content:
                            raw_parts.extend(self.mcp_client.convert_result_to_openai_format(content))
                    except Exception as exc:
                        eval_logger.warning(f"Tool {call.function.name} failed: {exc}")
                        raw_parts = [{"type": "text", "text": f"Tool {call.function.name} failed: {exc}"}]

                    all_response += "<tool_response>"
                    kept_parts = []
                    for part in raw_parts:
                        if part.get("type") == "image_url":
                            if tool_image_count >= self.max_tool_images:
                                continue
                            tool_image_count += 1
                            all_response += "<image_url>"
                            kept_parts.append(part)
                        elif part.get("type") == "text":
                            text = part["text"]
                            # A tool-produced image announces itself with the
                            # "Saved drawn image to: <path>" sentinel. Register each
                            # under a NEW img_idx (so the model can address it in a
                            # later tool call, like training appends crops to
                            # image_data), track it for cleanup, and replace the raw
                            # path the model sees with a clean img_idx message.
                            for match in _TOOL_SAVED_IMAGE_RE.finditer(text):
                                saved_path = match.group(1)
                                request_image_paths.append(saved_path)
                                assigned_idx = next_img_idx
                                image_registry[assigned_idx] = saved_path
                                if zoom_provenance is not None:
                                    # Carry provenance so a later zoom into this crop composites
                                    # back to the ROOT original at native resolution again.
                                    image_provenance[assigned_idx] = zoom_provenance
                                next_img_idx += 1
                                text = text.replace(match.group(0), self._tool_image_registered_text(image_indexing_mode, call.function.name, tool_args, assigned_idx))
                            if profile and len(text) > profile.multi["max_tool_response_length"]:
                                length = profile.multi["max_tool_response_length"]
                                side = profile.multi["tool_response_truncate_side"]
                                if side == "left":
                                    text = text[:length] + "...(truncated)"
                                elif side == "right":
                                    text = "(truncated)..." + text[-length:]
                                else:
                                    text = text[:length // 2] + "...(truncated)..." + text[-length // 2:]
                            part = {"type": "text", "text": text}
                            all_response += text
                            kept_parts.append(part)
                        else:
                            kept_parts.append(part)
                    all_response += "</tool_response>"

                    if self.tool_image_as_user_turn:
                        text_parts = [p for p in kept_parts if p.get("type") == "text"]
                        if not text_parts:
                            text_parts = [{"type": "text", "text": "(tool returned an image; shown below)"}]
                        messages.append({"role": "tool", "tool_call_id": call.id, "name": call.function.name, "content": text_parts})
                        pending_user_images.extend([p for p in kept_parts if p.get("type") == "image_url"])
                    else:
                        messages.append({"role": "tool", "tool_call_id": call.id, "name": call.function.name, "content": kept_parts})

                if terminal_tool_called:
                    break

                if pending_user_images:
                    messages.append({"role": "user", "content": pending_user_images + [{"type": "text", "text": "Above are the tool-returned image(s)."}]})

                if profile:
                    consumed = await self._training_prompt_tokens({**payload, "messages": messages}) - payload["_training_initial_tokens"]
                    training_remaining = profile.data["max_response_length"] - consumed
                    if training_remaining <= 0:
                        break
                context_final_due = await self._context_final_due({**payload, "messages": messages})
                final_turn_due = (
                    (self.force_answer_after_max_tool_iterations or self.force_tool_calls_until_final_turn)
                    and (context_final_due or tool_iteration >= self.max_tool_iterations or (
                        getattr(self, "context_budget_enabled", False) and tool_image_count >= self.max_tool_images
                    ) or (
                        profile and training_remaining <= profile.multi["final_answer_token_margin"]
                    ))
                )
                force_final_answer_now = final_turn_due and not terminal_tools
                force_terminal_answer_now = final_turn_due and bool(terminal_tools)
                first_deadline = getattr(self, "first_candidate_answer_turn", 0)
                followup_deadline = getattr(self, "followup_candidate_answer_turn", 0)
                force_first_candidate_now = bool(
                    first_deadline and tool_iteration + 1 >= first_deadline
                    and first_candidate_forced_turn is None
                    and (followup_deadline or first_candidate_seen_turn is None)
                    and not final_turn_due
                )
                force_followup_candidate_now = bool(
                    followup_deadline and followup_deadline not in candidate_forced_turns
                    and tool_iteration + 1 >= followup_deadline and not final_turn_due
                )

                # Once the image budget is spent, stop offering image-producing tools
                # so the model answers. Terminal (answer-protocol) tools stay offered:
                # they return no images and may be the task's only answer channel
                # (e.g. screenspot's computer_use).
                if (
                    tool_image_count >= self.max_tool_images
                    and not force_final_answer_now
                    and not self.force_tool_calls_until_final_turn
                ):
                    terminal_only = [f for f in payload.get("tools", []) if f["function"]["name"] in terminal_tools]
                    if terminal_only:
                        payload["tools"] = terminal_only
                    else:
                        payload.pop("tools", None)
                        payload.pop("tool_choice", None)

                payload["messages"] = messages
                if force_terminal_answer_now:
                    payload["tools"] = [
                        function
                        for function in payload.get("tools", [])
                        if function["function"]["name"] in terminal_tools
                    ]
                    payload["tool_choice"] = "auto"
                request_payload = payload
                response_prefill = ""
                forced_final_answer_valid = False
                responses_for_usage = []
                if force_final_answer_now:
                    forced_final_answer_requested = True
                    eval_logger.info(f"Forcing final answer on final allowed assistant turn for request {idx}")
                    response, response_prefill, responses_for_usage = await self._obtain_forced_final_answer(
                        payload,
                        idx,
                    )
                    forced_final_answer_valid = self._forced_final_answer_failure(response) is None
                elif force_first_candidate_now or force_followup_candidate_now:
                    # Match training's candidate deadline: constrain the action to
                    # <answer>, without the final-turn prefill or changing the cap.
                    if force_first_candidate_now:
                        first_candidate_forced_turn = tool_iteration + 1
                    candidate_forced_turns.append(tool_iteration + 1)
                    request_payload = self._with_forced_action_extra_body(payload, "<answer>")
                    request_payload["tool_choice"] = "none"
                    response = await self._create_training_completion(request_payload)
                    responses_for_usage.append(("primary", response))
                    failure = self._forced_final_answer_failure(response)
                    if failure is not None:
                        response.choices[0].message.content = f"[LMMS_EVAL_FORCED_FINAL_ANSWER_FAILED: early_candidate_{failure}]"
                else:
                    if self.force_tool_calls_until_final_turn:
                        request_payload = self._with_forced_action_extra_body(payload, "<tool_call>")
                    response = await self._create_training_completion(request_payload)
                    responses_for_usage.append(("primary", response))

                for force_stage, usage_response in responses_for_usage:
                    input_tokens, output_tokens, _ = _add_usage(usage_response, source="model")
                    if force_final_answer_now:
                        self._forced_final_stats[f"{force_stage}_input_tokens"] += input_tokens
                        self._forced_final_stats[f"{force_stage}_output_tokens"] += output_tokens
                response, response_prefill, critic_forced_tool = await _maybe_replace_answer_with_retry_tool(
                    response,
                    response_prefill,
                )
                last_response = response.choices[0].message.content or ""
                try:
                    if response_prefill:
                        all_response += response_prefill
                    all_response += last_response
                except Exception as e:
                    all_response += str(e)
                if (self.force_tool_calls_until_final_turn and not force_final_answer_now) or critic_forced_tool:
                    forced_tool_failure = self._forced_tool_call_failure(response)
                    if forced_tool_failure is not None:
                        eval_logger.error(
                            "Budget-forced tool turn failed "
                            f"for request {idx} after {tool_iteration} completed tool turn(s): "
                            f"reason={forced_tool_failure}, "
                            f"finish_reason={response.choices[0].finish_reason}, "
                            f"tool_calls={self._tool_call_diagnostic(response)}, "
                            f"content={repr((response.choices[0].message.content or '')[-1000:])}"
                        )
                        all_response += f"\n{self._budget_forcing_failure_marker(forced_tool_failure)}"
                        return all_response, idx, TokenCounts(
                            input_tokens=total_input_tokens,
                            output_tokens=total_output_tokens,
                            reasoning_tokens=total_reasoning_tokens,
                        ), _retry_critic_metadata()

                if force_terminal_answer_now:
                    for call in response.choices[0].message.tool_calls or []:
                        all_response += f"<tool_call>{call.function.name} {call.function.arguments}</tool_call>"
                        if call.function.name in terminal_tools:
                            terminal_tool_called = True
                            break

            if self.mcp_client is not None and tool_iteration >= self.max_tool_iterations and not terminal_tool_called:
                if forced_final_answer_requested and forced_final_answer_valid:
                    eval_logger.info(f"Final answer was forced on max tool turn for request {idx}")
                else:
                    eval_logger.warning(f"Max tool iterations ({self.max_tool_iterations}) reached for request {idx}")
                    all_response += f"\n[MAX_TOOL_ITERATIONS_REACHED]"
            return (
                all_response,
                idx,
                TokenCounts(
                    input_tokens=total_input_tokens,
                    output_tokens=total_output_tokens,
                    reasoning_tokens=total_reasoning_tokens,
                ),
                _retry_critic_metadata(),
            )
        finally:
            _cleanup_request_images()

    def generate_until(self, requests) -> List[GenerationResult]:
        results = []

        async def run():
            if getattr(self, "critic_guided_answer_retry_source", "policy") == "ground_truth":
                # Each task calls asyncio.run anew. Locks with concurrent
                # waiters bind to a loop and cannot be reused by the next task.
                self._ground_truth_critic_lock = asyncio.Lock()
            res: List[Tuple[GenerationResult, int]] = []
            pbar = tqdm(total=len(requests), disable=(self.rank != 0), desc="Model Responding")
            current_concurrency = (
                min(
                    max(1, self.num_cpus),
                    self.adaptive_config.max_concurrency,
                )
                if self.adaptive_concurrency
                else max(1, self.num_cpus)
            )
            dispatch_order = list(range(len(requests)))
            if self.prefix_aware_queue:
                prefix_hashes = {}
                for idx in dispatch_order:
                    request_obj = requests[idx]
                    prefix_text = request_obj.args[0] if isinstance(request_obj.args[0], str) else ""
                    if not prefix_text:
                        _, doc_to_messages, _, doc_id, task, split = request_obj.args
                        chat_messages_raw = doc_to_messages(self.task_dict[task][split][doc_id])
                        prefix_text = extract_text_prefix_from_chat_messages(chat_messages_raw, self.prefix_hash_chars)
                    prefix_hashes[idx] = make_prefix_hash(prefix_text, self.prefix_hash_chars)
                dispatch_order.sort(key=lambda idx: (prefix_hashes[idx], idx))
            cursor = 0

            async def _process(req, idx):
                if is_budget_exceeded():
                    return "[LMMS_EVAL_BUDGET_EXCEEDED]", idx, TokenCounts(), None, True, False, 0.0
                started_at = time.time()
                rate_limited = False
                last_error_msg = "unknown error"
                for attempt in range(self.max_retries):
                    try:
                        content, original_idx, token_counts, metadata = await self.maybe_forward_with_tool(req, idx)
                        elapsed = time.time() - started_at
                        return content, original_idx, token_counts, metadata, True, rate_limited, elapsed
                    except Exception as exc:
                        error_msg = str(exc)
                        last_error_msg = error_msg
                        rate_limited = rate_limited or is_rate_limit_error(error_msg)
                        eval_logger.info(f"Attempt {attempt + 1}/{self.max_retries} failed for request {idx} with error: {error_msg}")
                        # Skip retries for errors that won't succeed on retry
                        if "context length" in error_msg.lower() or "input_tokens" in error_msg or "maximum" in error_msg.lower() or "Already borrowed" in error_msg:
                            eval_logger.warning(f"Non-retriable error - skipping retries for request {idx}")
                            break
                        if attempt == self.max_retries - 1:
                            eval_logger.error(f"All {self.max_retries} attempts failed. Last error: {error_msg}")
                        else:
                            await asyncio.sleep(self.retry_backoff_s)

                elapsed = time.time() - started_at
                error_preview = last_error_msg.replace("\n", " ")[:200]
                failure_content = f"[LMMS_EVAL_REQUEST_FAILED after {self.max_retries} retries] {error_preview}"
                return failure_content, idx, TokenCounts(), None, False, rate_limited, elapsed

            failed_requests = 0
            rate_limited_requests = 0
            request_latencies: List[float] = []
            completed_since_adapt = 0
            in_flight: dict[asyncio.Task, int] = {}

            def maybe_update_concurrency(force: bool = False) -> None:
                nonlocal current_concurrency
                nonlocal failed_requests
                nonlocal rate_limited_requests
                nonlocal request_latencies
                nonlocal completed_since_adapt

                if not self.adaptive_concurrency:
                    return

                sample_threshold = max(4, current_concurrency)
                if not force and completed_since_adapt < sample_threshold:
                    return
                if completed_since_adapt <= 0:
                    return

                decision = decide_next_concurrency(
                    current_concurrency=current_concurrency,
                    total_requests=completed_since_adapt,
                    failed_requests=failed_requests,
                    rate_limited_requests=rate_limited_requests,
                    latencies=request_latencies,
                    config=self.adaptive_config,
                )
                if decision.next_concurrency != decision.current_concurrency:
                    eval_logger.info(
                        "Adaptive concurrency update: "
                        f"{decision.current_concurrency} -> "
                        f"{decision.next_concurrency} "
                        f"(fail_rate={decision.failure_rate:.3f}, "
                        f"rate_limit_rate={decision.rate_limit_rate:.3f}, "
                        f"p95_latency={decision.p95_latency_s:.3f}s)"
                    )
                current_concurrency = decision.next_concurrency
                failed_requests = 0
                rate_limited_requests = 0
                request_latencies = []
                completed_since_adapt = 0

            while cursor < len(dispatch_order) or in_flight:
                while cursor < len(dispatch_order) and len(in_flight) < max(1, current_concurrency):
                    request_index = dispatch_order[cursor]
                    task = asyncio.create_task(_process(requests[request_index], request_index))
                    in_flight[task] = request_index
                    cursor += 1

                if not in_flight:
                    break

                done, _ = await asyncio.wait(in_flight, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    in_flight.pop(task, None)
                    (
                        content,
                        request_idx,
                        token_counts,
                        metadata,
                        success,
                        rate_limited,
                        elapsed,
                    ) = task.result()
                    res.append(
                        (
                            GenerationResult(
                                text=content,
                                token_counts=token_counts,
                                metadata=metadata,
                            ),
                            request_idx,
                        )
                    )
                    if not success:
                        failed_requests += 1
                    if rate_limited:
                        rate_limited_requests += 1
                    request_latencies.append(elapsed)
                    completed_since_adapt += 1
                    totals = get_running_totals()
                    pbar.set_postfix({"tokens": f"{totals['total_tokens']:,}"}, refresh=False)
                    pbar.update(1)
                    maybe_update_concurrency(force=False)

            maybe_update_concurrency(force=True)

            pbar.close()
            if self._forced_final_stats["primary_attempts"]:
                eval_logger.info(
                    "Forced final-answer summary: "
                    f"stats={json.dumps(self._forced_final_stats, sort_keys=True)} "
                    f"failures={json.dumps(self._forced_final_failures, sort_keys=True)}"
                )
            if self.critic_guided_answer_retry:
                critic_stats = dict(self._retry_critic_stats)
                probability_count = critic_stats.pop("retry_probability_count")
                probability_sum = critic_stats.pop("retry_probability_sum")
                critic_stats["mean_retry_probability"] = (
                    probability_sum / probability_count if probability_count else None
                )
                eval_logger.info(
                    "Retry critic summary: "
                    f"stats={json.dumps(critic_stats, sort_keys=True)}"
                )
            return res

        manage_lifecycle = self.manage_vllm_server_lifecycle and bool(requests)
        if manage_lifecycle and not self._is_local_vllm_server():
            eval_logger.warning(
                "Ignoring manage_vllm_server_lifecycle for non-local base_url "
                f"{self.base_url!r}; vLLM admin endpoints are restricted to localhost"
            )
            manage_lifecycle = False
        if manage_lifecycle:
            try:
                self._wake_vllm_server()
            except BaseException:
                self._sleep_vllm_server_best_effort("wake failure")
                raise
        try:
            eval_results = asyncio.run(run())
        except BaseException:
            if manage_lifecycle:
                self._sleep_vllm_server_best_effort("generation failure")
            raise
        if manage_lifecycle:
            # A successful generation is not safe to hand to the judge until the
            # shared vLLM server has released weights and KV cache.
            try:
                self._sleep_vllm_server()
            except Exception as exc:
                raise VLLMServerLifecycleError(
                    "Policy generation succeeded, but the managed vLLM server did not "
                    "confirm level-2 deep sleep; refusing to initialize the judge"
                ) from exc
        eval_results.sort(key=lambda x: x[1])  # Sort by index to restore original
        results = results + [content for content, _ in eval_results]
        if self.mcp_client is not None:
            shutil.rmtree(self.work_dir)
        return results
