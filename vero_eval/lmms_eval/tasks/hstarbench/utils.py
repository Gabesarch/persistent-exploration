"""hstarbench — agentic humanoid visual search for lmms-eval."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, TYPE_CHECKING

from PIL import Image

if TYPE_CHECKING:
    from vagen.env.hstar.env import HstarEnv
    from vagen.env.hstar.env_config import HstarEnvConfig

_TASK_DIR = Path(__file__).resolve().parent
if str(_TASK_DIR) not in sys.path:
    sys.path.insert(0, str(_TASK_DIR))
from image_io import patched_hstar_cv2_imread  # noqa: E402
from panorama_strip import erp_to_strip_pil  # noqa: E402
from scoring import (  # noqa: E402
    evaluate_episode_outcome,
    get_doc_ground_bboxes,
    inject_submit_from_bbox,
    parse_bboxes_from_text,
    score_final_answer_bbox,
)

_VAGEN = None


def _vagen():
    """Lazy import so strip+zoom tasks work without the hstarbench simulator checkout."""
    global _VAGEN
    if _VAGEN is None:
        root = Path(os.environ.get("HSTAR_VAGEN_ROOT", os.environ.get("THINKIN360_ROOT", "")))
        if root.is_dir() and str(root) not in sys.path:
            sys.path.insert(0, str(root))
        from vagen.env.hstar.env import HstarEnv
        from vagen.env.hstar.env_config import HstarEnvConfig
        from vagen.env.hstar.prompt import (
            action_template,
            format_prompt,
            init_observation_template,
            system_prompt,
        )

        _VAGEN = {
            "HstarEnv": HstarEnv,
            "HstarEnvConfig": HstarEnvConfig,
            "action_template": action_template,
            "format_prompt": format_prompt,
            "init_observation_template": init_observation_template,
            "system_prompt": system_prompt,
        }
    return _VAGEN

PANORAMA_STRIP_TOOL_HINT = """
Visual layout for tool use:
- Image 0: full 360° cylindrical panorama (iPhone-style horizontal strip). Left→right is yaw 0°→360° (seamless wrap). Green line marks your initial heading.
- Image 1 (when shown): current camera view (~100° FOV).
Use image_zoom_in_tool on image 0 with bbox_2d in 0–1000 coordinates to inspect the scene.
To answer, either:
  (A) submit_bbox(bbox_2d=[x1,y1,x2,y2]) — center of the box on image 0 is scored as your target heading, or
  (B) rotate(yaw,pitch) / submit(yaw,pitch) as in the original benchmark.
Success on strip tasks is judged by whether your final zoom/submit bbox center falls in the target yaw/pitch region.
"""

ZOOM_BBOX_SYSTEM_PROMPT = """You are a 360° visual search agent.

Setup:
- Image 0 is a horizontal 360° panorama strip. Left to right spans yaw 0°→360°. Use 0–1000 bbox coordinates on image 0.
- You may call image_zoom_in_tool ONLY on image 0 (img_idx=0) to inspect regions before answering.

Rules:
- Do NOT use rotate(), submit(), or submit_bbox().
- When you are ready to finish, your last message MUST include an <answer>...</answer> block with exactly one target bbox on image 0, e.g.:
  <answer>bbox_2d=[x1,y1,x2,y2]</answer>
- Success is judged by (1) whether the center of your final answer bbox falls in the hidden target yaw/pitch cone on the strip, and (2) IoU against the ground-truth strip bbox region.
"""


def _bbox_zoom_only(kwargs: Optional[Dict[str, Any]], doc: Dict[str, Any]) -> bool:
    kw = kwargs or {}
    if kw.get("bbox_zoom_only"):
        return True
    return str(kw.get("score_mode", doc.get("score_mode", ""))) in ("answer_bbox", "answer_bbox_iou")


def _env_config_from_doc(doc: Dict[str, Any]):
    v = _vagen()
    return v["HstarEnvConfig"](
        data_path=doc["data_path"],
        eval_set=doc.get("eval_set", "object_search"),
        resolution=int(doc.get("resolution", 1080)),
        prompt_format=doc.get("prompt_format", "free_think"),
        seed=int(doc.get("dataset_seed", 42)),
    )


def _make_env(doc: Dict[str, Any]):
    v = _vagen()
    return v["HstarEnv"](config=_env_config_from_doc(doc))


def _use_panorama_strip(doc: Dict[str, Any], kwargs: Optional[Dict[str, Any]] = None) -> bool:
    kw = kwargs or {}
    return bool(kw.get("use_panorama_strip", doc.get("use_panorama_strip", False)))


def _panorama_strip_image(doc: Dict[str, Any]) -> Image.Image:
    loaded = _load_image(doc.get("panorama_strip_path", ""), doc)
    if loaded is not None:
        return loaded
    erp_path = doc.get("erp_path", "")
    if erp_path:
        return erp_to_strip_pil(erp_path, start_yaw_deg=doc.get("start_yaw"))
    raise FileNotFoundError("panorama strip missing; run build_lmms_dataset.py")


def _global_image_for_tool(
    doc: Dict[str, Any],
    kwargs: Optional[Dict[str, Any]],
    state_info: Optional[Dict[str, Any]] = None,
) -> Image.Image:
    if state_info is not None and state_info.get("cached_global") is not None:
        return state_info["cached_global"]
    if _use_panorama_strip(doc, kwargs):
        img = _panorama_strip_image(doc)
    else:
        img = _load_image(doc.get("erp_path", ""), doc)
        if img is None:
            raise FileNotFoundError(f"no global image for doc {doc.get('doc_id')}")
    if state_info is not None:
        state_info["cached_global"] = img
    return img


def _view_from_env(env, init_obs: bool) -> Image.Image:
    obs = env._render(init_obs=init_obs)
    return obs["multi_modal_data"][env.config.image_placeholder][0]


def _view_from_step_obs(obs: Dict[str, Any], env) -> Image.Image:
    return obs["multi_modal_data"][env.config.image_placeholder][0]


def _visuals_for_step(
    doc: Dict[str, Any],
    obs: Dict[str, Any],
    include_global: bool,
    kwargs: Optional[Dict[str, Any]],
    state_info: Dict[str, Any],
) -> List[Image.Image]:
    view = _view_from_step_obs(obs, state_info["env"])
    if include_global:
        return [_global_image_for_tool(doc, kwargs, state_info), view]
    return [view]


def _system_text(env) -> str:
    v = _vagen()
    fmt = env.format_prompt_func(max_actions_per_step=env.config.max_actions_per_step, add_example=True)
    return v["system_prompt"](format=env.config.prompt_format, is_nav=env.is_nav) + "\n" + fmt


def _initial_context(env) -> str:
    v = _vagen()
    obs_str = v["init_observation_template"](
        observation=env.config.image_placeholder,
        instruction=env.episode_language_instruction,
    )
    return _system_text(env) + "\n\n" + obs_str


def _initial_context_from_doc(doc: Dict[str, Any], kwargs: Optional[Dict[str, Any]] = None) -> str:
    """Build first-turn prompt without instantiating HstarEnv (safe for lmms-eval request build)."""
    v = _vagen()
    is_nav = doc.get("eval_set") == "navigation" or bool(doc.get("is_nav"))
    prompt_format = doc.get("prompt_format", "free_think")
    fmt = v["format_prompt"][prompt_format](max_actions_per_step=1, add_example=True)
    sys_txt = v["system_prompt"](format=prompt_format, is_nav=is_nav) + "\n" + fmt
    obs_str = v["init_observation_template"](
        observation="<image>",
        instruction=doc.get("task", ""),
    )
    text = sys_txt + "\n\n" + obs_str
    if _use_panorama_strip(doc, kwargs) and bool((kwargs or {}).get("include_erp_for_mcp", doc.get("include_erp_for_mcp"))):
        text += "\n" + PANORAMA_STRIP_TOOL_HINT.strip()
    return text


def _step_context(env, env_feedback: str, valid_action: str, done: bool) -> str:
    v = _vagen()
    return v["action_template"](
        observation=env.config.image_placeholder,
        instruction=env.episode_language_instruction,
        valid_action=valid_action,
        env_feedback=env_feedback,
        done=done,
    )


def _score_mode(kwargs: Optional[Dict[str, Any]], doc: Dict[str, Any]) -> str:
    kw = kwargs or {}
    if kw.get("score_mode"):
        return str(kw["score_mode"])
    if doc.get("score_mode"):
        return str(doc["score_mode"])
    if _use_panorama_strip(doc, kw) and kw.get("include_erp_for_mcp", doc.get("include_erp_for_mcp")):
        return "zoom_align"
    return "env"


def _preprocess_model_response(
    text: str,
    doc: Dict[str, Any],
    kwargs: Optional[Dict[str, Any]],
) -> str:
    if not _use_panorama_strip(doc, kwargs):
        return text
    target_yaw = doc.get("target_yaw") or [0, 360]
    target_pitch = doc.get("target_pitch") or [-90, 90]
    return inject_submit_from_bbox(text, target_yaw, target_pitch)


def _record_zoom_from_response(state_info: Dict[str, Any], model_response: str) -> None:
    boxes = parse_bboxes_from_text(model_response)
    if boxes:
        state_info["zoom_bboxes"] = state_info.get("zoom_bboxes", []) + boxes


def _new_state_info(env, include_erp: bool, doc: Dict[str, Any], kwargs: Dict[str, Any]) -> Dict[str, Any]:
    state = {
        "env": env,
        "steps": 0,
        "valid_actions": 0,
        "invalid_steps": 0,
        "include_erp": include_erp,
        "zoom_bboxes": [],
        "cached_global": None,
    }
    if include_erp:
        try:
            state["cached_global"] = _global_image_for_tool(doc, kwargs)
        except FileNotFoundError:
            pass
    return state


_DEFAULT_HSTARBENCH_ROOT = "/scratch/gpfs/ZHUANGL/shared/Hstarbench"


def _resolve_asset_path(path: str, doc: Optional[Dict[str, Any]] = None) -> Path:
    if not path:
        return Path()
    p = Path(path)
    if p.is_file():
        return p
    bench_root = ""
    if doc:
        bench_root = str(doc.get("hstarbench_root", "") or "")
    if not bench_root:
        bench_root = os.environ.get("HSTARBENCH_ROOT", _DEFAULT_HSTARBENCH_ROOT)
    if bench_root:
        candidate = Path(bench_root) / path
        if candidate.is_file():
            return candidate
    if not p.is_absolute():
        task_relative = Path(__file__).resolve().parent / p
        if task_relative.is_file():
            return task_relative
    return p


def _load_image(path: str, doc: Optional[Dict[str, Any]] = None) -> Optional[Image.Image]:
    if not path:
        return None
    p = _resolve_asset_path(path, doc)
    if not p.is_file():
        return None
    try:
        with Image.open(p) as im:
            return im.convert("RGB")
    except Exception:
        return None


def _init_env_for_episode(doc: Dict[str, Any], kwargs: Dict[str, Any], include_global: bool):
    with patched_hstar_cv2_imread():
        env = _make_env(doc)
        env.reset(seed=int(doc["seed"]))
    state_info = _new_state_info(env, include_global, doc, kwargs)
    return env, state_info


def hstarbench_doc_to_visual(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None) -> List[Image.Image]:
    kwargs = lmms_eval_specific_kwargs or {}
    if _bbox_zoom_only(kwargs, doc):
        return hstarbench_bbox_zoom_doc_to_visual(doc, kwargs)
    include_global = bool(kwargs.get("include_erp_for_mcp", doc.get("include_erp_for_mcp", False)))
    use_strip = _use_panorama_strip(doc, kwargs)
    visuals: List[Image.Image] = []
    if include_global:
        if use_strip:
            strip = _load_image(doc.get("panorama_strip_path", ""), doc)
            if strip is None and doc.get("erp_path"):
                strip = erp_to_strip_pil(doc["erp_path"], start_yaw_deg=doc.get("start_yaw"))
            if strip is not None:
                visuals.append(strip)
        else:
            erp = _load_image(doc.get("erp_path", ""), doc)
            if erp is not None:
                visuals.append(erp)
    view = _load_image(doc.get("view_path", ""), doc)
    if view is not None:
        visuals.append(view)
    elif not visuals:
        fallback = _load_image(doc.get("panorama_strip_path", ""), doc) or _load_image(doc.get("erp_path", ""), doc)
        if fallback is not None:
            visuals.append(fallback)
    return visuals


def hstarbench_bbox_zoom_doc_to_visual(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None,
) -> List[Image.Image]:
    """Panorama strip only — single global image for zoom-in tool (img_idx=0)."""
    kwargs = lmms_eval_specific_kwargs or {}
    if _use_panorama_strip(doc, kwargs):
        strip = _load_image(doc.get("panorama_strip_path", ""), doc)
        if strip is None and doc.get("erp_path"):
            strip = erp_to_strip_pil(doc["erp_path"], start_yaw_deg=doc.get("start_yaw"))
        if strip is not None:
            return [strip]
    erp = _load_image(doc.get("erp_path", ""), doc)
    if erp is not None:
        return [erp]
    raise FileNotFoundError(f"no panorama image for doc {doc.get('doc_id')}")


def hstarbench_bbox_zoom_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None,
) -> str:
    instruction = doc.get("task", "").strip()
    return (
        f"Find the target described below in the 360° panorama (image 0).\n"
        f"Task: {instruction}\n\n"
        "You may zoom in with image_zoom_in_tool on image 0. "
        "When finished, respond with <answer>bbox_2d=[x1,y1,x2,y2]</answer> on image 0."
    )


def hstarbench_bbox_zoom_doc_to_messages(doc: Dict[str, Any], lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None):
    kwargs = lmms_eval_specific_kwargs or {}
    question = hstarbench_bbox_zoom_doc_to_text(doc, kwargs)
    visuals = hstarbench_bbox_zoom_doc_to_visual(doc, kwargs)
    system_messages = [{"role": "system", "content": [{"type": "text", "text": ZOOM_BBOX_SYSTEM_PROMPT.strip()}]}]
    user_content = []
    for visual in visuals:
        user_content.append({"type": "image", "url": visual})
    user_content.append({"type": "text", "text": question})
    return system_messages + [{"role": "user", "content": user_content}]


def hstarbench_bbox_zoom_doc_to_messages_vero(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None,
):
    """User turn only — Vero system + tool schema come from task yaml metadata."""
    kwargs = lmms_eval_specific_kwargs or {}
    pre = str(kwargs.get("pre_prompt", ""))
    post = str(kwargs.get("post_prompt", ""))
    question = pre + hstarbench_bbox_zoom_doc_to_text(doc, kwargs) + post
    visuals = hstarbench_bbox_zoom_doc_to_visual(doc, kwargs)
    user_content = []
    for visual in visuals:
        user_content.append({"type": "image", "url": visual})
    user_content.append({"type": "text", "text": question})
    return [{"role": "user", "content": user_content}]


def hstarbench_bbox_zoom_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    raw = results[0] if results else ""
    if isinstance(raw, list):
        raw = raw[0] if raw else ""
    outcome = score_final_answer_bbox(doc, str(raw))
    level = int(doc.get("level", 0))
    split = str(doc.get("split", "unknown"))
    cone_success = 1.0 if outcome.get("cone_success") else 0.0
    iou_success = 1.0 if outcome.get("iou_success") else 0.0
    has_bbox = 1.0 if outcome.get("has_answer_bbox") else 0.0
    bbox_iou = float(outcome.get("bbox_iou") or 0.0)
    return {
        "hstarbench_success": cone_success,
        "hstarbench_cone_success": cone_success,
        "hstarbench_zoom_align_success": cone_success,
        "hstarbench_bbox_iou_success": iou_success,
        "hstarbench_answer_bbox_success": iou_success,
        "hstarbench_has_answer_bbox": has_bbox,
        "hstarbench_bbox_iou": bbox_iou,
        "hstarbench_env_success": 0.0,
        "hstarbench_submit_bbox_success": iou_success,
        f"hstarbench_{split}_success": cone_success,
        f"hstarbench_{split}_cone_success": cone_success,
        f"hstarbench_{split}_iou_success": iou_success,
        f"hstarbench_level_{level}_success": cone_success,
        f"hstarbench_level_{level}_cone_success": cone_success,
        f"hstarbench_level_{level}_iou_success": iou_success,
    }


def hstarbench_doc_to_target(doc: Dict[str, Any]) -> str:
    ground_bboxes = [list(box) for box in get_doc_ground_bboxes(doc)]
    return json.dumps(
        {
            "ground_bboxes": ground_bboxes,
            "ground_bbox": ground_bboxes[0] if len(ground_bboxes) == 1 else None,
            "target_yaw": doc.get("target_yaw"),
            "target_pitch": doc.get("target_pitch"),
            "level": doc.get("level"),
        }
    )


def hstarbench_doc_to_text(
    doc: Dict[str, Any],
    lmms_eval_specific_kwargs: Optional[Dict[str, Any]] = None,
    previous_output: Optional[List[str]] = None,
    round_idx: Optional[int] = None,
    previous_round_info: Optional[Dict[str, Any]] = None,
) -> Any:
    """Agentic callback; also returns initial prompt string when ``round_idx is None``."""
    kwargs = lmms_eval_specific_kwargs or {}
    include_global = bool(kwargs.get("include_erp_for_mcp", doc.get("include_erp_for_mcp", False)))

    # Build initial context for request construction (fewshot_context -> ctx).
    if round_idx is None:
        return _initial_context_from_doc(doc, kwargs)

    # First agentic round after model's first generation: create env.
    if previous_round_info is None:
        try:
            env, state_info = _init_env_for_episode(doc, kwargs, include_global)
        except Exception as exc:
            return _error_terminal_tuple(doc, previous_output, str(exc))
        model_response = previous_output[-1] if previous_output else ""
        _record_zoom_from_response(state_info, model_response)
        model_response = _preprocess_model_response(model_response, doc, kwargs)
        with patched_hstar_cv2_imread():
            obs, reward, done, info = env.step(model_response)
        state_info["steps"] = 1
        if info.get("metrics", {}).get("turn_metrics", {}).get("action_is_valid"):
            state_info["valid_actions"] = 1
        else:
            state_info["invalid_steps"] = 1
        if done:
            return _terminal_tuple(doc, env, state_info, previous_output, info, kwargs)
        next_context = obs["obs_str"]
        visuals = _visuals_for_step(doc, obs, include_global, kwargs, state_info)
        return visuals, next_context, False, previous_output, state_info

    state_info = previous_round_info
    env = state_info["env"]
    model_response = previous_output[-1] if previous_output else ""
    _record_zoom_from_response(state_info, model_response)
    model_response = _preprocess_model_response(model_response, doc, kwargs)

    with patched_hstar_cv2_imread():
        obs, reward, done, info = env.step(model_response)
    state_info["steps"] = state_info.get("steps", 0) + 1
    if info.get("metrics", {}).get("turn_metrics", {}).get("action_is_valid"):
        state_info["valid_actions"] = state_info.get("valid_actions", 0) + 1
    else:
        state_info["invalid_steps"] = state_info.get("invalid_steps", 0) + 1

    if done:
        return _terminal_tuple(doc, env, state_info, previous_output, info, kwargs)

    next_context = obs["obs_str"]
    visuals = _visuals_for_step(
        doc,
        obs,
        state_info.get("include_erp", include_global),
        kwargs,
        state_info,
    )
    return visuals, next_context, False, previous_output, state_info


def _error_terminal_tuple(
    doc: Dict[str, Any],
    previous_output: Optional[List[str]],
    error: str,
) -> tuple:
    payload = {
        "success": False,
        "env_success": False,
        "zoom_align_success": False,
        "submit_bbox_align_success": False,
        "error": error,
        "steps": 0,
        "seed": doc.get("seed"),
        "doc_id": doc.get("doc_id"),
        "trace": previous_output,
    }
    return None, None, True, [json.dumps(payload, ensure_ascii=False)], None


def _terminal_tuple(
    doc: Dict[str, Any],
    env,
    state_info: Dict[str, Any],
    previous_output: Optional[List[str]],
    info: Dict[str, Any],
    kwargs: Optional[Dict[str, Any]] = None,
) -> tuple:
    env_success = bool(info.get("task_success") or info.get("metrics", {}).get("traj_metrics", {}).get("success"))
    use_strip = _use_panorama_strip(doc, kwargs)
    outcome = evaluate_episode_outcome(
        doc,
        env_success,
        previous_output,
        score_mode=_score_mode(kwargs, doc),
        use_strip=use_strip,
    )
    payload = {
        "success": outcome["success"],
        "env_success": outcome["env_success"],
        "zoom_align_success": outcome["zoom_align_success"],
        "submit_bbox_align_success": outcome["submit_bbox_align_success"],
        "zoom_bbox": outcome.get("zoom_bbox"),
        "zoom_yaw": outcome.get("zoom_yaw"),
        "zoom_pitch": outcome.get("zoom_pitch"),
        "zoom_bbox_count": outcome.get("zoom_bbox_count", 0),
        "score_mode": outcome.get("score_mode"),
        "steps": state_info.get("steps", 0),
        "valid_actions": state_info.get("valid_actions", 0),
        "invalid_steps": state_info.get("invalid_steps", 0),
        "level": doc.get("level"),
        "split": doc.get("split"),
        "seed": doc.get("seed"),
        "doc_id": doc.get("doc_id"),
        "final_yaw": env.yaw,
        "final_pitch": env.pitch,
        "target_yaw": doc.get("target_yaw"),
        "target_pitch": doc.get("target_pitch"),
        "trace": previous_output,
    }
    try:
        env.close()
    except Exception:
        pass
    return None, None, True, [json.dumps(payload, ensure_ascii=False)], state_info


def hstarbench_process_results(doc: Dict[str, Any], results: List[str]) -> Dict[str, float]:
    raw = results[0] if results else ""
    success = env_success = zoom_success = submit_bbox_success = 0.0
    steps = 0.0
    valid_actions = 0.0
    invalid_steps = 0.0

    if isinstance(raw, str):
        try:
            payload = json.loads(raw)
            success = 1.0 if payload.get("success") else 0.0
            env_success = 1.0 if payload.get("env_success") else 0.0
            zoom_success = 1.0 if payload.get("zoom_align_success") else 0.0
            submit_bbox_success = 1.0 if payload.get("submit_bbox_align_success") else 0.0
            steps = float(payload.get("steps", 0))
            valid_actions = float(payload.get("valid_actions", 0))
            invalid_steps = float(payload.get("invalid_steps", 0))
        except json.JSONDecodeError:
            pass

    level = int(doc.get("level", 0))
    split = str(doc.get("split", "unknown"))
    return {
        "hstarbench_success": success,
        "hstarbench_env_success": env_success,
        "hstarbench_zoom_align_success": zoom_success,
        "hstarbench_submit_bbox_success": submit_bbox_success,
        "hstarbench_steps": steps,
        "hstarbench_valid_actions": valid_actions,
        "hstarbench_invalid_steps": invalid_steps,
        f"hstarbench_{split}_success": success,
        f"hstarbench_level_{level}_success": success,
    }


def _mean_metric(results: List[float]) -> float:
    return sum(results) / len(results) if results else 0.0


def hstarbench_aggregate_success(results: List[float]) -> float:
    return _mean_metric(results)


def hstarbench_aggregate_steps(results: List[float]) -> float:
    return _mean_metric(results)


def hstarbench_aggregate_valid_actions(results: List[float]) -> float:
    return _mean_metric(results)


def hstarbench_aggregate_invalid_steps(results: List[float]) -> float:
    return _mean_metric(results)


def hstarbench_aggregate_env_success(results: List[float]) -> float:
    return _mean_metric(results)


def hstarbench_aggregate_zoom_align_success(results: List[float]) -> float:
    return _mean_metric(results)


def hstarbench_aggregate_submit_bbox_success(results: List[float]) -> float:
    return _mean_metric(results)


def hstarbench_aggregate_has_answer_bbox(results: List[float]) -> float:
    return _mean_metric(results)


def hstarbench_aggregate_bbox_iou(results: List[float]) -> float:
    return _mean_metric(results)
