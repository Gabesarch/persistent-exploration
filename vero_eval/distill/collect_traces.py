#!/usr/bin/env python3
"""Distill GPT-5.5 reasoning + tool-use traces on ANY tool-calling task.

Generic successor to collect_ball_drop_traces.py — same Responses-API multi-turn
tool loop, now task-agnostic. The task-specific bits (how to render the question,
get the image, name a doc, and score an answer) come from a torch-free
``distill_adapter.py`` in the task's directory; everything else (the loop, the
img_idx registry, image persistence, the trace schema) is shared.

WHY the Responses API: gpt-5.5 tool-calling WITH reasoning_effort is rejected on
/v1/chat/completions ("use /v1/responses instead"), and lmms_eval's async_openai
backend speaks chat.completions — so we reimplement the loop here.

WHAT it reuses (single source of truth, consistent with the modular eval):
  * the task YAML (e.g. vstar_bench_gpt55_zoomdraw.yaml): dataset_path/test_split,
    the toolset (metadata.allowed_tools + system_prompt_files), the prompt
    (lmms_eval_specific_kwargs.default), and metadata.distill (teacher knobs +
    optional adapter overrides).
  * <task_dir>/distill_adapter.py: doc_to_text / doc_to_visual / doc_id / score
    (torch-free; loaded by file path, so the heavy lmms_eval package is not imported).
  * the MCP tool server vero2_mcp.py via MCPClient (img_idx registry exactly as the
    eval; tool images announce themselves with the "Saved drawn image to:" sentinel).

Run:
  python collect_traces.py --task vstar_bench_gpt55_zoomdraw \
      [--limit N] [--workers 8] [--output-root <dir>] [--self-test]
"""
from __future__ import annotations

import argparse
import base64
import glob
import importlib.util
import io
import json
import os
import re
import sys
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from PIL import Image

# --- repo paths -------------------------------------------------------------
_THIS = os.path.abspath(__file__)
DISTILL_DIR = os.path.dirname(_THIS)
VERO_EVAL_ROOT = os.path.dirname(DISTILL_DIR)                       # .../vero_eval
TASKS_ROOT = os.path.join(VERO_EVAL_ROOT, "lmms_eval", "tasks")
MCP_SERVER = os.path.join(VERO_EVAL_ROOT, "lmms_eval", "mcp", "vero2_mcp.py")
MCP_CLIENT_PY = os.path.join(VERO_EVAL_ROOT, "lmms_eval", "mcp", "client.py")
SANDBOX_DIR = "/home/lc1556/repo/AI-sandbox"

DEFAULT_OUTPUT_BASE = "/scratch/gpfs/ZHUANGL/lc1556/distillation"
DEFAULT_HF_HOME = "/scratch/gpfs/ZHUANGL/lc1556/hf_cache"

# Keep in sync with vero2_mcp.py / async_openai.py.
_IMAGE_REGISTRY_ENV = "LMMS_EVAL_IMAGE_REGISTRY"
_TOOL_SAVED_IMAGE_RE = re.compile(r"Saved drawn image to:\s*(\S+)")

# Default adapter convention (override via metadata.distill.adapter in the yaml).
_ADAPTER_DEFAULTS = {"module": "distill_adapter", "doc_to_text": "doc_to_text",
                     "doc_to_visual": "doc_to_visual", "doc_id": "doc_id", "score": "score"}


# --- dynamic imports (avoid importing the torch-heavy lmms_eval package) -----
def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _get_mcp_client_cls():
    return _load_module("vero_mcp_client", MCP_CLIENT_PY).MCPClient


def _get_sandbox_client():
    if SANDBOX_DIR not in sys.path:
        sys.path.insert(0, SANDBOX_DIR)
    import sandbox  # noqa: E402

    return sandbox.get_client()


def resolve_task_yaml(task):
    matches = glob.glob(os.path.join(TASKS_ROOT, "*", f"{task}.yaml"))
    if not matches:
        raise SystemExit(f"task yaml '{task}.yaml' not found under {TASKS_ROOT}/*/")
    return matches[0]


# --- minimal YAML config resolver (include + metadata deep-merge) ------------
# Mirrors lmms_eval.utils.load_yaml_config(mode="simple") WITHOUT importing it
# (it pulls in torch). !function tags are read as their "module.name" string.
class _FuncTag:
    def __init__(self, ref):
        self.ref = ref

    def __repr__(self):
        return f"!function {self.ref}"


def _func_constructor(loader, node):
    return _FuncTag(loader.construct_scalar(node))


_DEEP_MERGE_KEYS = ("metadata", "lmms_eval_specific_kwargs")


def _coerce_meta(m):
    """Some task templates write `metadata` as a LIST of single-key dicts
    (e.g. `- version: 0.0`); coerce to a dict so it deep-merges with a toolset
    template's dict metadata instead of clobbering it."""
    if isinstance(m, list):
        out = {}
        for it in m:
            if isinstance(it, dict):
                out.update(it)
        return out
    return m if isinstance(m, dict) else {}


def _merge(dst, src):
    for k, v in src.items():
        if k == "metadata":
            v = _coerce_meta(v)
            if k in dst:
                dst[k] = _coerce_meta(dst[k])
        if k in _DEEP_MERGE_KEYS and isinstance(v, dict) and isinstance(dst.get(k), dict):
            merged = dict(dst[k])
            for kk, vv in v.items():
                if isinstance(vv, dict) and isinstance(merged.get(kk), dict):
                    m2 = dict(merged[kk]); m2.update(vv); merged[kk] = m2
                else:
                    merged[kk] = vv
            dst[k] = merged
        else:
            dst[k] = v


def load_task_config(yaml_path):
    loader = yaml.FullLoader
    yaml.add_constructor("!function", _func_constructor, Loader=loader)
    with open(yaml_path, "rb") as fh:
        cfg = yaml.load(fh, Loader=loader)
    yaml_dir = os.path.dirname(yaml_path)
    if "include" in cfg:
        includes = cfg.pop("include")
        if isinstance(includes, str):
            includes = [includes]
        includes = list(reversed(includes))
        final = {}
        for inc in includes:
            path = inc if os.path.isfile(inc) else os.path.join(yaml_dir, inc)
            _merge(final, load_task_config(path))
        _merge(final, cfg)
        return final
    return cfg


# --- prompt / system / tools resolution -------------------------------------
def resolve_system_prompt(system_prompt_files):
    """Concatenate the toolset system fragments, exactly like async_openai."""
    if not system_prompt_files:
        return None
    if isinstance(system_prompt_files, str):
        system_prompt_files = [system_prompt_files]
    fragments = []
    for rel in system_prompt_files:
        path = rel if os.path.isfile(rel) else os.path.join(VERO_EVAL_ROOT, rel)
        if not os.path.isfile(path):
            raise FileNotFoundError(f"system_prompt_files entry '{rel}' not found at '{path}'")
        with open(path) as fh:
            fragments.append(fh.read().strip())
    return "\n\n".join(f for f in fragments if f) or None


def _fnmatch_any(name, patterns):
    import fnmatch

    return any(fnmatch.fnmatchcase(name, p) for p in patterns)


def build_responses_tools(mcp_client, allowed_tools):
    """MCP function list -> Responses-API flat tool specs, filtered by allowed_tools."""
    fns = mcp_client.get_function_list_sync()  # chat-completions shape
    if allowed_tools:
        if isinstance(allowed_tools, str):
            allowed_tools = [p.strip() for p in allowed_tools.split(",") if p.strip()]
        fns = [f for f in fns if _fnmatch_any(f["function"]["name"], allowed_tools)]
        if not fns:
            raise ValueError(f"allowed_tools={allowed_tools} matched no MCP tool")
    tools = []
    for f in fns:
        fn = f["function"]
        tools.append({
            "type": "function",
            "name": fn["name"],
            "description": fn.get("description", ""),
            "parameters": fn.get("parameters", {"type": "object", "properties": {}}),
        })
    return tools


# --- image helpers ----------------------------------------------------------
def pil_to_data_url(image, fmt="JPEG", quality=90, max_pixels=1_048_576):
    image = image.convert("RGB")
    if max_pixels and image.width * image.height > max_pixels:
        scale = (max_pixels / (image.width * image.height)) ** 0.5
        image = image.resize((max(1, round(image.width * scale)), max(1, round(image.height * scale))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    if fmt == "JPEG":
        image.save(buf, format="JPEG", quality=quality)
        mime = "image/jpeg"
    else:
        image.save(buf, format="PNG")
        mime = "image/png"
    return f"data:{mime};base64," + base64.b64encode(buf.getvalue()).decode()


def path_to_data_url(path):
    return pil_to_data_url(Image.open(path))


def _to_dict(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump(exclude_none=True)
    if isinstance(obj, dict):
        return obj
    return dict(obj)


def _reasoning_summary_text(item):
    return "\n".join(p["text"] for p in (item.get("summary") or [])
                     if isinstance(p, dict) and p.get("text"))


def _message_text(item):
    return "\n".join(c["text"] for c in (item.get("content") or [])
                     if isinstance(c, dict) and c.get("type") in ("output_text", "text") and c.get("text"))


# --- the Responses-API multi-turn tool loop ---------------------------------
def run_one(doc, index, *, client, mcp_client, tools, system_prompt, lmms_kwargs,
            distill, images_dir, tool_images_dir, adapter, task_name=None,
            allowed_tools=None, system_prompt_files=None):
    doc_id = adapter.doc_id(doc, index)

    # 1) Persist the input image at ORIGINAL resolution (tools zoom into this).
    os.makedirs(images_dir, exist_ok=True)
    input_image_path = os.path.join(images_dir, f"{doc_id}.png")
    pil_img = adapter.doc_to_visual(doc)[0]
    pil_img.save(input_image_path, format="PNG")
    registry = {0: input_image_path}
    next_idx = 1

    # 2) Build the opening Responses-API input (system + user text + image).
    user_text = adapter.doc_to_text(doc, lmms_kwargs) + "\nImage 0 is available as img_idx=0."
    input_items = []
    if system_prompt:
        input_items.append({"role": "system", "content": [{"type": "input_text", "text": system_prompt}]})
    input_items.append({"role": "user", "content": [
        {"type": "input_text", "text": user_text},
        # Show the teacher the FULL-resolution PNG (no downscale), matching what the
        # student sees at inference. The 0-1000 tool coords stay valid; tools read this PNG.
        {"type": "input_image", "image_url": pil_to_data_url(pil_img, fmt="PNG", max_pixels=None)},
    ]})

    model = distill["teacher_model"]
    reasoning = {"effort": distill.get("reasoning_effort", "medium")}
    if distill.get("reasoning_summary"):
        reasoning["summary"] = distill["reasoning_summary"]
    max_output_tokens = int(distill.get("max_output_tokens", 8000))
    max_iters = int(distill.get("max_tool_iterations", 8))
    max_tool_images = int(distill.get("max_tool_images", 8))

    def _create(items):
        last_err = None
        for attempt in range(5):
            kwargs = dict(model=model, input=items, tools=tools, tool_choice="auto",
                          reasoning=reasoning, max_output_tokens=max_output_tokens,
                          include=["reasoning.encrypted_content"], store=False)
            try:
                return client.responses.create(**kwargs)
            except Exception as e:
                msg = str(e)
                last_err = e
                if attempt == 0 and ("include" in msg or "store" in msg or "encrypted" in msg):
                    kwargs.pop("include", None); kwargs.pop("store", None)
                    try:
                        return client.responses.create(**kwargs)
                    except Exception as e2:
                        last_err = e2
                time.sleep(2 * (attempt + 1))
        raise last_err

    turns = []
    tool_image_count = 0
    tok_in = tok_out = tok_reason = 0
    final_text = ""
    status = "ok"

    for it in range(max_iters + 1):
        resp = _create(input_items)
        resp_status = getattr(resp, "status", None)
        out_items = [_to_dict(o) for o in (resp.output or [])]
        u = _to_dict(getattr(resp, "usage", {}) or {})
        turn_in = u.get("input_tokens", 0) or 0
        turn_out = u.get("output_tokens", 0) or 0
        turn_reason = (u.get("output_tokens_details") or {}).get("reasoning_tokens", 0) or 0
        # input_tokens are CUMULATIVE per turn (store=false re-sends the growing prefix);
        # the per-turn value is kept on each turn, last turn = true final prompt size.
        tok_in += turn_in
        tok_out += turn_out
        tok_reason += turn_reason

        reasoning_text = "\n".join(_reasoning_summary_text(o) for o in out_items if o.get("type") == "reasoning").strip()
        msg_text = "\n".join(_message_text(o) for o in out_items if o.get("type") == "message").strip()
        fcs = [o for o in out_items if o.get("type") == "function_call"]

        turn = {"iteration": it, "reasoning_summary": reasoning_text, "message_text": msg_text,
                "tool_calls": [],
                "usage": {"input_tokens": turn_in, "output_tokens": turn_out, "reasoning_tokens": turn_reason}}

        # Truncated turn (hit max_output_tokens): record and stop — do NOT echo/execute a
        # half-emitted tool call. Checked regardless of whether fcs is present.
        if resp_status == "incomplete":
            status = "incomplete"
            final_text = msg_text or final_text
            turns.append(turn)
            break

        if not fcs:
            final_text = msg_text or final_text
            turns.append(turn)
            break

        if it >= max_iters:
            turns.append(turn)
            status = "max_tool_iterations"
            final_text = msg_text or final_text
            break

        # Echo ALL output items (reasoning + function_call) back — required for
        # reasoning models to continue a tool chain.
        input_items.extend(out_items)

        pending_images = []
        for fc in fcs:
            name = fc.get("name")
            call_id = fc.get("call_id")
            args_raw = fc.get("arguments") or "{}"
            rec = {"name": name, "arguments": args_raw, "result_text": "", "result_image_path": None}
            try:
                args = json.loads(args_raw) if isinstance(args_raw, str) else (args_raw or {})
                if "img_idx" in args:
                    idx = int(args["img_idx"])
                    if idx not in registry:
                        raise ValueError(f"img_idx {idx} is not available. Available: {sorted(registry)}.")
                env = {_IMAGE_REGISTRY_ENV: json.dumps({str(k): v for k, v in registry.items()})}
                result = mcp_client.run_tool_sync(name, args, env=env)
                result_text_parts = []
                produced_img_path = None
                produced_img_b64 = None
                for content in result.content:
                    cd = _to_dict(content)
                    if cd.get("type") == "text":
                        result_text_parts.append(cd.get("text", ""))
                    elif cd.get("type") == "image":
                        produced_img_b64 = cd.get("data")
                joined = "\n".join(result_text_parts)
                m = _TOOL_SAVED_IMAGE_RE.search(joined)
                if m:
                    produced_img_path = m.group(1)
                if produced_img_path and os.path.isfile(produced_img_path):
                    assigned = next_idx
                    next_idx += 1
                    registry[assigned] = produced_img_path
                    rec["result_img_idx"] = assigned
                    # persist the EXACT bytes the model saw (MCP already bounded to <=1MP q85)
                    os.makedirs(os.path.join(tool_images_dir, doc_id), exist_ok=True)
                    persist = os.path.join(tool_images_dir, doc_id, f"turn{it}_call{len(turn['tool_calls'])}_idx{assigned}.jpg")
                    persisted_ok = False
                    try:
                        if produced_img_b64:
                            with open(persist, "wb") as _ph:
                                _ph.write(base64.b64decode(produced_img_b64))
                        else:
                            Image.open(produced_img_path).convert("RGB").save(persist, format="JPEG", quality=90)
                        rec["result_image_path"] = persist
                        persisted_ok = True
                    except Exception:
                        rec["result_image_path"] = produced_img_path
                    if persisted_ok:
                        registry[assigned] = persist  # point at the persisted copy
                        try:
                            os.remove(produced_img_path)  # drop MCP /tmp temp (inode hygiene)
                        except OSError:
                            pass
                    rec["result_text"] = f"Registered tool-output image as img_idx {assigned}."
                    input_items.append({"type": "function_call_output", "call_id": call_id,
                                        "output": rec["result_text"] + " The image is shown next."})
                    if tool_image_count < max_tool_images:
                        tool_image_count += 1
                        data_url = ("data:image/jpeg;base64," + produced_img_b64) if produced_img_b64 else path_to_data_url(produced_img_path)
                        pending_images.append((assigned, data_url))
                else:
                    rec["result_text"] = joined or "(no output)"
                    input_items.append({"type": "function_call_output", "call_id": call_id, "output": rec["result_text"]})
            except Exception as exc:
                err = f"Tool {name} failed: {exc}"
                rec["result_text"] = err
                input_items.append({"type": "function_call_output", "call_id": call_id, "output": err})
            turn["tool_calls"].append(rec)

        turns.append(turn)

        if pending_images:
            content = []
            for assigned, data_url in pending_images:
                content.append({"type": "input_text", "text": f"Tool-output image (img_idx={assigned}):"})
                content.append({"type": "input_image", "image_url": data_url})
            input_items.append({"role": "user", "content": content})

    # Score the FINAL turn (proper final-answer supervision); a whole-transcript scan
    # is only a softer fallback for the REPORTED accuracy, never for `trainable`.
    sc_final = adapter.score(doc, final_text or "")
    gt = sc_final.get("gt")
    pred = sc_final.get("pred")
    correct = bool(sc_final.get("correct"))
    if not correct:
        whole = "\n".join((t.get("message_text") or "") for t in turns)
        sc_whole = adapter.score(doc, whole)
        if pred is None:
            pred = sc_whole.get("pred")
        correct = bool(sc_whole.get("correct"))
    trainable = bool(status == "ok" and sc_final.get("correct"))

    return {
        "doc_id": doc_id,
        "run_id": doc.get("run_id"),
        "index": index,
        "task": task_name,
        "category": doc.get("category"),
        "gt": gt,
        "pred": pred,
        "correct": correct,
        "trainable": trainable,
        "teacher_model": model,
        "reasoning_effort": reasoning.get("effort"),
        "prompt_version": lmms_kwargs.get("prompt_version"),
        "allowed_tools": allowed_tools,
        "system_prompt_files": system_prompt_files,
        "system_prompt": system_prompt,
        "user_text": user_text,
        "num_turns": len(turns),
        "num_tool_calls": sum(len(t["tool_calls"]) for t in turns),
        "tool_image_count": tool_image_count,
        "input_image_path": input_image_path,
        "image_registry": {str(k): v for k, v in registry.items()},
        "final_text": final_text,
        "status": status,
        "turns": turns,
        "usage": {"input_tokens": tok_in, "output_tokens": tok_out, "reasoning_tokens": tok_reason},
    }


# --- dataset / driver -------------------------------------------------------
def load_hf_dataset(dataset_path, split, limit=None):
    os.environ.setdefault("HF_HOME", DEFAULT_HF_HOME)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    from datasets import load_dataset

    ds = load_dataset(dataset_path, split=split)
    if limit:
        ds = ds.select(range(min(limit, len(ds))))
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, help="task yaml stem (searched under tasks/*/)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--output-root", default=None, help="default: <base>/<task-stem-without-gpt55>")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    yaml_path = resolve_task_yaml(args.task)
    task_dir = os.path.dirname(yaml_path)
    cfg = load_task_config(yaml_path)
    md = cfg.get("metadata", {})
    distill = dict(md.get("distill") or {})
    distill.setdefault("teacher_model", "gpt-5.5")
    distill.setdefault("reasoning_effort", "medium")
    distill.setdefault("reasoning_summary", "auto")
    distill.setdefault("max_output_tokens", 16384)
    distill.setdefault("max_tool_iterations", 8)
    distill.setdefault("max_tool_images", 8)

    adapter_cfg = dict(_ADAPTER_DEFAULTS)
    adapter_cfg.update(distill.get("adapter") or {})
    adapter = _load_module(f"{args.task}_adapter", os.path.join(task_dir, f"{adapter_cfg['module']}.py"))
    # rebind chosen function names to the standard attribute names
    for std, name in (("doc_to_text", adapter_cfg["doc_to_text"]), ("doc_to_visual", adapter_cfg["doc_to_visual"]),
                      ("doc_id", adapter_cfg["doc_id"]), ("score", adapter_cfg["score"])):
        setattr(adapter, std, getattr(adapter, name))

    allowed_tools = md.get("allowed_tools")
    system_prompt_files = md.get("system_prompt_files")
    system_prompt = resolve_system_prompt(system_prompt_files)
    lmms_kwargs = (cfg.get("lmms_eval_specific_kwargs", {}) or {}).get("default", {}) or {}
    dataset_path = cfg["dataset_path"]
    split = (distill.get("adapter") or {}).get("dataset_split") or cfg.get("test_split") \
        or getattr(adapter, "DATASET_SPLIT", "train")

    MCPClient = _get_mcp_client_cls()
    mcp_client = MCPClient(MCP_SERVER)
    tools = build_responses_tools(mcp_client, allowed_tools)

    print(f"[config] task={args.task} dir={os.path.basename(task_dir)} adapter={adapter_cfg['module']}")
    print(f"[config] teacher={distill['teacher_model']} effort={distill['reasoning_effort']} "
          f"summary={distill['reasoning_summary']} max_out={distill['max_output_tokens']} "
          f"max_iters={distill['max_tool_iterations']} max_tool_images={distill['max_tool_images']}")
    print(f"[config] tools={[t['name'] for t in tools]}")
    print(f"[config] system_prompt={'<%d chars>' % len(system_prompt) if system_prompt else None}")
    print(f"[config] dataset={dataset_path} split={split} lmms_kwargs={lmms_kwargs}")

    if args.self_test:
        ds = load_hf_dataset(dataset_path, split, limit=2)
        d0 = ds[0]
        print(f"[self-test] rows(sampled)={len(ds)} keys={list(d0.keys())[:8]}")
        print(f"[self-test] doc_id={adapter.doc_id(d0,0)} img={adapter.doc_to_visual(d0)[0].size}")
        print(f"[self-test] user_text head={adapter.doc_to_text(d0, lmms_kwargs)[:90]!r}")
        probe = adapter.score(d0, "\\boxed{A}")
        print(f"[self-test] score(probe boxed-A)={probe}")
        print("[self-test] OK")
        return

    stem = args.task.replace("_gpt55_zoomdraw", "").replace("_gpt55_zoom", "").replace("_gpt55_draw", "")
    output_root = args.output_root or os.path.join(DEFAULT_OUTPUT_BASE, stem)
    images_dir = os.path.join(output_root, "images")
    tool_images_dir = os.path.join(output_root, "tool_images")
    traces_dir = os.path.join(output_root, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    trace_path = os.path.join(traces_dir, f"{args.task}.jsonl")

    done_ids = set()
    if args.resume and os.path.isfile(trace_path):
        with open(trace_path) as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("status") != "error" and "error" not in obj:
                    done_ids.add(obj["doc_id"])
        print(f"[resume] {len(done_ids)} docs already done (errored docs retried); skipping them")

    ds = load_hf_dataset(dataset_path, split, limit=args.limit)
    todo = [(i, ds[i]) for i in range(len(ds)) if adapter.doc_id(ds[i], i) not in done_ids]
    print(f"[run] {len(todo)} docs to process with {args.workers} workers -> {trace_path}")

    client = _get_sandbox_client()
    write_lock = threading.Lock()
    counters = {"done": 0, "correct": 0, "err": 0, "trainable": 0}

    def _work(item):
        index, doc = item
        try:
            rec = run_one(doc, index, client=client, mcp_client=mcp_client, tools=tools,
                          system_prompt=system_prompt, lmms_kwargs=lmms_kwargs, distill=distill,
                          images_dir=images_dir, tool_images_dir=tool_images_dir, adapter=adapter,
                          task_name=args.task, allowed_tools=allowed_tools,
                          system_prompt_files=system_prompt_files)
        except Exception as exc:
            rec = {"doc_id": adapter.doc_id(doc, index), "index": index, "task": args.task,
                   "status": "error", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()[-1500:], "pred": None,
                   "correct": False, "trainable": False, "turns": [], "final_text": ""}
        with write_lock:
            with open(trace_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            counters["done"] += 1
            counters["correct"] += int(bool(rec.get("correct")))
            counters["err"] += int("error" in rec)
            counters["trainable"] += int(bool(rec.get("trainable")))
            if counters["done"] % 10 == 0 or counters["done"] == len(todo):
                non_err = counters["done"] - counters["err"]
                acc = counters["correct"] / max(1, non_err)
                print(f"[progress] {counters['done']}/{len(todo)} acc={acc:.3f} "
                      f"trainable={counters['trainable']} errs={counters['err']}", flush=True)
        return rec

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for _ in as_completed([pool.submit(_work, it) for it in todo]):
            pass

    total = counters["done"] + len(done_ids)
    non_err = counters["done"] - counters["err"]
    acc = counters["correct"] / max(1, non_err)
    summary = {"task": args.task, "teacher_model": distill["teacher_model"],
               "reasoning_effort": distill["reasoning_effort"], "processed_now": counters["done"],
               "resumed_skipped": len(done_ids), "total_in_trace": total,
               "acc_this_run": acc, "trainable_this_run": counters["trainable"],
               "errors": counters["err"], "elapsed_s": round(time.time() - t0, 1),
               "trace_path": trace_path}
    with open(os.path.join(traces_dir, f"{args.task}.summary.json"), "w") as fh:
        json.dump(summary, fh, indent=2)
    print("[summary]", json.dumps(summary))


if __name__ == "__main__":
    main()
