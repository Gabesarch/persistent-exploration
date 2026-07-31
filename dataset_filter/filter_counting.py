#!/usr/bin/env python
"""Ground-truth filtering for the combined_counting_hq dataset using Gemini.

Goal: keep hard-but-correct counting examples, drop examples whose ground-truth
count is wrong or unanswerable. We do NOT reject an example just because it is
hard (see prompts.py for the anti-laziness / anti-sycophancy design).

Strategies (--strategy):
  blind   : independent count, image only. keep if GT in [count_low, count_high].
  verify  : answer-aware audit, image + GT shown. keep if verdict in CORRECT/PLAUSIBLE.
  smart   : blind count; if it exactly matches GT -> CORRECT (1 call); else run the
            answer-aware audit (2 calls). Default. Best accuracy/cost trade-off.

Output: one JSON line per example in --out, resumable (re-run skips done ids).
Each line records the per-stage raw model output, the final verdict, and `keep`.

Example:
  python filter_counting.py --n 200 --strategy smart \
      --out /scratch/.../runs/pilot200.jsonl
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import os
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from PIL import Image, ImageDraw

import prompts as P

# ---- wire up the Princeton AI-sandbox client (key auto-loaded from its .env) ----
SANDBOX_DIR = os.environ.get(
    "AI_SANDBOX_DIR", str(Path(__file__).resolve().parents[3] / "AI-sandbox")
)
sys.path.insert(0, SANDBOX_DIR)
from sandbox import get_client, chat  # noqa: E402

DEFAULT_DATA = (
    "/scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq"
)
DEFAULT_MODEL = "gemini-3.1-pro-preview"


# --------------------------------------------------------------------------- IO

def load_records(data_dir: str, split: str, local: bool = False,
                 dataset: str = "combined_counting"):
    """Read <data_dir>/<dataset>_<split>.verl.jsonl (or the .local. variant).

    `dataset` is the metadata file prefix — change it (or pass --dataset) to filter a
    different dataset with the same tooling; nothing else here is counting-specific.
    """
    ext = "local.verl.jsonl" if local else "verl.jsonl"
    path = os.path.join(data_dir, f"{dataset}_{split}.{ext}")
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def record_view(rec: dict) -> dict:
    """Flatten a verl record to the few fields we need for filtering + reports."""
    ei = rec.get("extra_info", {})
    meta = ei.get("metadata", {})
    q = ei.get("question") or (rec.get("prompt", [{}])[0].get("content", ""))
    q = q.replace("<image>", "").strip()
    gt_raw = rec.get("reward_model", {}).get("ground_truth", ei.get("answer"))
    try:
        gt = int(str(gt_raw).strip())
    except (ValueError, TypeError):
        gt = None
    return {
        "id": ei.get("id"),
        "index": ei.get("index"),
        "image": rec["images"][0],
        "question": q,
        "ground_truth": gt,
        "gt_raw": gt_raw,
        "category": meta.get("annotation_category"),
        "source_dataset": ei.get("source_dataset") or meta.get("source_dataset"),
        "original_size": meta.get("original_size"),
        "processed_size": meta.get("processed_size"),
        "points": meta.get("original_points") or [],
        "n_points": len(meta.get("original_points") or []),
    }


# ------------------------------------------------------------------- image prep

def _load_rgb(path, image_root_remap=None):
    if image_root_remap:
        src, dst = image_root_remap
        if path.startswith(src):
            path = dst + path[len(src):]
    return Image.open(path).convert("RGB")


def _encode(img, max_side):
    w, h = img.size
    long_side = max(w, h)
    if long_side > max_side:  # downscale only
        scale = max_side / long_side
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def image_data_url(path: str, max_side: int, image_root_remap=None) -> str:
    return _encode(_load_rgb(path, image_root_remap), max_side)


def points_data_url(path, points, original_size, max_side, image_root_remap=None):
    """Draw the dataset's GT annotation points as magenta rings, then encode.

    Points are in `original_size` coordinates; the on-disk image is the upscaled
    `processed_size`, so we scale points by processed/original before drawing.
    """
    img = _load_rgb(path, image_root_remap)
    W, H = img.size
    ow, oh = (original_size if original_size and len(original_size) == 2 else (W, H))
    sx, sy = W / max(1, ow), H / max(1, oh)
    draw = ImageDraw.Draw(img)
    r = max(4, round(max(W, H) / 220))
    for p in points:
        try:
            x, y = float(p[0]) * sx, float(p[1]) * sy
        except (TypeError, ValueError, IndexError):
            continue
        draw.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 255), width=max(2, r // 3))
    return _encode(img, max_side)


# Failover key order: primary, then AI_SANDBOX_KEY2, KEY3, ... (add more in .env anytime).
KEY_VARS = ["AI_SANDBOX_KEY"] + [f"AI_SANDBOX_KEY{i}" for i in range(2, 10)]


def load_keys():
    """[primary, backup, ...] from env (overrides) + AI-sandbox/.env. Values never printed."""
    src = {}
    env = Path(SANDBOX_DIR) / ".env"
    if env.exists():
        for line in env.read_text().splitlines():
            s = line.strip()
            if s.startswith("export "):
                s = s[len("export "):].strip()
            if "=" in s:
                k = s.split("=", 1)[0].strip()
                if k in KEY_VARS:
                    v = s.split("=", 1)[1].strip().strip('"').strip("'")
                    if v:
                        src.setdefault(k, v)
    for k in KEY_VARS:
        v = os.getenv(k)
        if v:
            src[k] = v.strip()
    keys, seen = [], set()
    for k in KEY_VARS:
        if k in src and src[k] not in seen:
            keys.append(src[k]); seen.add(src[k])
    if not keys:
        raise SystemExit("[filter] no AI_SANDBOX_KEY* in env or .env")
    return keys


def _should_failover(err) -> bool:
    """True only for key-level exhaustion / auth problems — NOT transient per-minute
    rate limits (those are handled by backoff+retry on the same key)."""
    s = str(err).lower()
    if "rate limit" in s and "quota" not in s and "insufficient" not in s:
        return False
    return any(t in s for t in ("insufficient", "quota", "exhaust", "billing",
                                "usage limit", "limit exceeded", "412",  # Portkey key usage cap
                                "401", "403", "unauthorized", "permission",
                                "invalid api key", "expired", "credit"))


class FailoverClient:
    """Thread-safe client shared across workers. On a quota/auth error it marks that key
    'dead' and moves to the next live key. A capped key is skipped for the rest of the run
    (so we don't keep wasting calls on it); if EVERY key looks dead we clear the set and
    re-probe from the top — usage windows may have cleared. We never permanently give up:
    a transient all-keys-limited burst just backs off and retries (rows stay retryable)."""
    def __init__(self, keys):
        self.keys = keys
        self.i = 0
        self._c = None
        self.dead = set()          # key indices that returned a usage/auth error
        self._lock = threading.Lock()

    def _client(self):
        if self._c is None:
            from portkey_ai import Portkey
            self._c = Portkey(api_key=self.keys[self.i])
        return self._c

    def create(self, **kw):
        return self._client().chat.completions.create(**kw)

    def on_error(self, err, used_i=None):
        if not _should_failover(err):
            return
        with self._lock:
            # Only react to the key that actually failed; a burst of simultaneous errors
            # from key N must not skip past the key another worker already moved us to.
            if used_i is not None and used_i != self.i:
                return
            self.dead.add(self.i)
            n = len(self.keys)
            nxt = next((j for step in range(1, n + 1)
                        for j in [(self.i + step) % n] if j not in self.dead), None)
            if nxt is None:                 # every key looks dead -> re-probe from the top
                self.dead.clear()
                nxt = 0
            self.i = nxt
            self._c = None
            print(f"[filter] key usage/auth limit; dead={sorted(self.dead)} -> now on key {self.i}")


def raw_chat(client, model, msgs, retries=6, reasoning_effort=None):
    """Like sandbox.chat but also returns token usage: (content, {prompt,completion,total}).

    reasoning_effort (minimal|low|medium|high) controls how many hidden "thinking"
    tokens the model spends — the dominant cost driver. None = provider default.
    Works with a FailoverClient (has .create/.on_error) or a raw Portkey client.
    """
    kw = {"reasoning_effort": reasoning_effort} if reasoning_effort else {}
    create = getattr(client, "create", None) or client.chat.completions.create
    on_error = getattr(client, "on_error", None)
    last = None
    for attempt in range(retries):
        try:
            resp = create(model=model, messages=msgs, **kw)
            u = getattr(resp, "usage", None)
            usage = {
                "prompt": getattr(u, "prompt_tokens", None),
                "completion": getattr(u, "completion_tokens", None),
                "total": getattr(u, "total_tokens", None),
            } if u else {}
            return resp.choices[0].message.content, usage
        except Exception as err:
            last = err
            if on_error:
                on_error(err, getattr(client, "i", None))  # mark the failed key dead, move to a live one
            if attempt == retries - 1:
                raise
            time.sleep(min(30, 2 ** attempt))
    raise last


# ------------------------------------------------------------------ json parsing

def parse_json(text: str):
    """Tolerant: pull the first balanced {...} block and json.loads it."""
    if not text:
        return None
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except json.JSONDecodeError:
                        return None
    return None


def as_int(v):
    try:
        return int(round(float(v)))
    except (ValueError, TypeError):
        return None


# ------------------------------------------------------------------- strategies

VALID_VERDICTS = {"CORRECT", "PLAUSIBLE", "WRONG", "AMBIGUOUS", "UNANSWERABLE"}


def call_blind(client, model, view, data_url, retries=4, reasoning_effort=None):
    msgs = [
        {"role": "system", "content": P.BLIND_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": P.BLIND_USER.format(question=view["question"])},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]
    raw, usage = raw_chat(client, model, msgs, retries=retries, reasoning_effort=reasoning_effort)
    j = parse_json(raw) or {}
    return {
        "count": as_int(j.get("count")),
        "count_low": as_int(j.get("count_low")) if j.get("count_low") is not None else as_int(j.get("count")),
        "count_high": as_int(j.get("count_high")) if j.get("count_high") is not None else as_int(j.get("count")),
        "difficulty": j.get("difficulty"),
        "confidence": j.get("confidence"),
        "reasoning": j.get("reasoning"),
        "usage": usage,
        "raw": raw,
    }


def call_verify(client, model, view, data_url, blind=None, n_points=0, retries=4, reasoning_effort=None):
    bb = P.blind_block(blind) if blind else ""
    pb = P.points_block(n_points)
    msgs = [
        {"role": "system", "content": P.VERIFY_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": P.VERIFY_USER.format(
                question=view["question"], gt=view["ground_truth"],
                blind_block=bb, points_block=pb)},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]
    raw, usage = raw_chat(client, model, msgs, retries=retries, reasoning_effort=reasoning_effort)
    j = parse_json(raw) or {}
    verdict = (j.get("verdict") or "").strip().upper()
    if verdict not in VALID_VERDICTS:
        verdict = "PARSE_ERROR"
    return {
        "your_count": as_int(j.get("your_count")),
        "verdict": verdict,
        "confidence": j.get("confidence"),
        "reasoning": j.get("reasoning"),
        "used_points": bool(n_points),
        "usage": usage,
        "raw": raw,
    }


def num_ok(model_count, gt, abs_tol, rel_tol):
    """Is the model's own count close enough to GT to treat GT as confirmed?

    True when |model - gt| <= max(abs_tol, floor(rel_tol*gt)). We round the
    relative band DOWN and default abs_tol=0 so that small counts must match
    (essentially) exactly: floor(0.15*gt)=0 for gt<=6, so GT=1/model=2 or
    GT=2/model=4 is NOT rescued. The 15% band only opens up on larger counts
    (gt=100 -> +/-15). This overrides an over-strict WRONG on genuinely dense
    images without papering over real small-count disagreements.
    """
    import math
    if model_count is None or gt is None:
        return None
    return abs(model_count - gt) <= max(abs_tol, math.floor(rel_tol * abs(gt)))


def run_one(view, *, client, model, strategy, max_side, image_root_remap,
            retries=4, abs_tol=2, rel_tol=0.15, use_points=True, reasoning_effort=None):
    gt = view["ground_truth"]
    points = view.get("points") or []
    out = dict(view)
    out.pop("points", None)  # keep n_points only; raw coords bloat the decisions file
    out["strategy"] = strategy
    out["model"] = model
    out["abs_tol"] = abs_tol
    out["rel_tol"] = rel_tol
    out["reasoning_effort"] = reasoning_effort
    if gt is None:  # non-integer GT -> can't do count match; let verify judge it
        strategy = "verify"
    data_url = image_data_url(view["image"], max_side, image_root_remap)

    # Stage-2 image: overlay GT points (audit guide) when available + requested.
    with_points = bool(use_points and points)
    if with_points:
        verify_url = points_data_url(view["image"], points, view.get("original_size"),
                                     max_side, image_root_remap)
        n_pts = len(points)
    else:
        verify_url, n_pts = data_url, 0

    def verify(blind):
        return call_verify(client, model, view, verify_url, blind=blind,
                           n_points=n_pts, retries=retries, reasoning_effort=reasoning_effort)

    if strategy == "blind":
        b = call_blind(client, model, view, data_url, retries=retries, reasoning_effort=reasoning_effort)
        out["blind"] = b
        in_band = (b["count_low"] is not None and b["count_high"] is not None
                   and gt is not None and b["count_low"] <= gt <= b["count_high"])
        exact = (b["count"] == gt)
        out["verdict"] = "CORRECT" if exact else ("PLAUSIBLE" if in_band else "WRONG")
        out["keep"] = bool(exact or in_band)
        out["best_count"] = b["count"]

    elif strategy == "verify":
        v = verify(None)
        out["verify"] = v
        out["best_count"] = v["your_count"]
        out["verdict"], out["keep"], out["override"] = combine_verify(
            v["verdict"], v["your_count"], gt, abs_tol, rel_tol)

    else:  # smart
        b = call_blind(client, model, view, data_url, retries=retries, reasoning_effort=reasoning_effort)
        out["blind"] = b
        if gt is not None and b["count"] == gt:
            out["verdict"] = "CORRECT"
            out["keep"] = True
            out["best_count"] = b["count"]
            out["short_circuit"] = True
        else:
            v = verify(b)
            out["verify"] = v
            out["best_count"] = v["your_count"]
            out["short_circuit"] = False
            out["verdict"], out["keep"], out["override"] = combine_verify(
                v["verdict"], v["your_count"], gt, abs_tol, rel_tol)

    out["used_points"] = with_points
    out["usage_total"] = _sum_usage(out.get("blind"), out.get("verify"))
    return out


def _sum_usage(*calls):
    tot = {"prompt": 0, "completion": 0, "total": 0, "calls": 0}
    for c in calls:
        u = (c or {}).get("usage") or {}
        if not u:
            continue
        tot["calls"] += 1
        for k in ("prompt", "completion", "total"):
            if u.get(k) is not None:
                tot[k] += u[k]
    return tot


def combine_verify(verdict, model_count, gt, abs_tol, rel_tol):
    """Fold the auditor verdict with our numeric tolerance.

    Trust the model on UNANSWERABLE / category mismatch, but if it says WRONG yet
    its own count is within tolerance of GT, override to PLAUSIBLE+keep (the
    off-by-small over-rejection we saw in the pilot). Returns (verdict, keep, override).
    """
    ok = num_ok(model_count, gt, abs_tol, rel_tol)
    # Only override an over-strict WRONG when the model actually counted SOME
    # objects close to GT. model_count == 0 means the model says the category is
    # absent / mismatched (e.g. "those are geese, not cranes") -> a real drop
    # signal we must never paper over with numeric tolerance.
    if verdict == "WRONG" and ok and model_count and model_count > 0:
        return "PLAUSIBLE", True, "num_tol"  # model's own count actually matches GT
    keep = verdict in {"CORRECT", "PLAUSIBLE"}
    return verdict, keep, None


# Rows that failed (API/image error or unparseable JSON) are NOT final decisions:
# they must be re-run, never treated as a silent drop and never applied.
NONTERMINAL_VERDICTS = {"ERROR", "PARSE_ERROR"}


def is_terminal(row: dict) -> bool:
    """A row is a real, final decision only if it has an explicit keep AND a valid verdict."""
    return (row.get("keep") in (True, False)
            and row.get("verdict") not in NONTERMINAL_VERDICTS)


# ------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", default=DEFAULT_DATA)
    ap.add_argument("--dataset", default="combined_counting",
                    help="metadata file prefix: <dataset>_<split>.verl.jsonl (reusable across datasets)")
    ap.add_argument("--split", default="train", choices=["train", "val"])
    ap.add_argument("--local-meta", action="store_true",
                    help="read the self-contained <dataset>_<split>.local.verl.jsonl")
    ap.add_argument("--strategy", default="smart", choices=["smart", "blind", "verify"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--n", type=int, default=200, help="number of records (0 = all)")
    ap.add_argument("--sample", choices=["random", "first"], default="random")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-side", type=int, default=1536, help="downscale long side to this")
    ap.add_argument("--abs-tol", type=int, default=0, help="abs count tolerance for keep (0 = small counts exact)")
    ap.add_argument("--rel-tol", type=float, default=0.15, help="relative count tolerance for keep")
    ap.add_argument("--force", action="store_true",
                    help="resume even if the run config differs from <out>.config.json")
    ap.add_argument("--no-points", action="store_true",
                    help="do NOT overlay GT annotation points in stage-2 verify")
    ap.add_argument("--reasoning-effort", default=None,
                    choices=["minimal", "low", "medium", "high"],
                    help="thinking effort (cost driver); default = provider default")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--retries", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-root-remap", default=None,
                    help="SRC:DST prefix remap for image paths")
    args = ap.parse_args()

    remap = None
    if args.image_root_remap:
        src, dst = args.image_root_remap.split(":", 1)
        remap = (src, dst)

    recs = load_records(args.data_dir, args.split, local=args.local_meta, dataset=args.dataset)
    views = [record_view(r) for r in recs]
    if args.sample == "random":
        rng = random.Random(args.seed)
        rng.shuffle(views)
    if args.n and args.n > 0:
        views = views[: args.n]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    # --- run-config compatibility guard (fail closed on drift) ---
    cfg = {"dataset": args.dataset, "split": args.split, "strategy": args.strategy,
           "model": args.model, "reasoning_effort": args.reasoning_effort,
           "max_side": args.max_side, "abs_tol": args.abs_tol, "rel_tol": args.rel_tol,
           "use_points": not args.no_points, "local_meta": args.local_meta}
    cfg_path = args.out + ".config.json"
    if os.path.exists(cfg_path):
        old = json.load(open(cfg_path))
        # tolerant: only flag keys the old sidecar actually recorded, so adding new
        # config fields later never spuriously blocks resuming an older run.
        diff = {k: (old.get(k), cfg[k]) for k in cfg if k in old and old.get(k) != cfg[k]}
        if diff and not args.force:
            raise SystemExit(
                f"[filter] run-config mismatch vs {cfg_path}: {diff}\n"
                f"  Reusing this --out with different settings would mix decision "
                f"semantics. Use a fresh --out, or pass --force to override.")
    else:
        json.dump(cfg, open(cfg_path, "w"), indent=2)

    # --- resumable: only TERMINAL rows count as done; compact the file so any
    #     prior non-terminal (ERROR/PARSE_ERROR) or duplicate rows get re-run. ---
    done = set()
    if os.path.exists(args.out):
        by_id = {}
        with open(args.out) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                by_id[r.get("id")] = r          # last write wins
        terminal = {i: r for i, r in by_id.items() if is_terminal(r)}
        with open(args.out, "w") as f:          # rewrite: keep only terminal decisions
            for r in terminal.values():
                f.write(json.dumps(r) + "\n")
        done = set(terminal)
        n_requeue = len(by_id) - len(terminal)
        if n_requeue:
            print(f"[filter] compacted {args.out}: re-queued {n_requeue} non-terminal/duplicate row(s)")
    todo = [v for v in views if v["id"] not in done]
    print(f"[filter] split={args.split} strategy={args.strategy} model={args.model}")
    print(f"[filter] {len(views)} selected, {len(done)} already done, {len(todo)} to run")
    if not todo:
        summarize(args.out)
        return

    keys = load_keys()
    client = FailoverClient(keys)
    print(f"[filter] {len(keys)} API key(s) available "
          f"({'failover ON' if len(keys) > 1 else 'single key, NO failover'})")
    lock = threading.Lock()
    fout = open(args.out, "a")
    t0 = time.time()
    n_done = [0]

    def work(v):
        try:
            return run_one(v, client=client, model=args.model, strategy=args.strategy,
                           max_side=args.max_side, image_root_remap=remap, retries=args.retries,
                           abs_tol=args.abs_tol, rel_tol=args.rel_tol,
                           use_points=not args.no_points, reasoning_effort=args.reasoning_effort)
        except Exception as e:  # never lose the batch over one bad image / API blip
            return {**v, "error": repr(e)[:300], "verdict": "ERROR", "keep": None}

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futs = {pool.submit(work, v): v for v in todo}
        for fut in as_completed(futs):
            res = fut.result()
            with lock:
                fout.write(json.dumps(res) + "\n")
                fout.flush()
                n_done[0] += 1
                if n_done[0] % 10 == 0 or n_done[0] == len(todo):
                    rate = n_done[0] / max(1e-9, time.time() - t0)
                    print(f"  {n_done[0]}/{len(todo)}  ({rate:.2f}/s)  last={res.get('verdict')}")
    fout.close()
    summarize(args.out)


def summarize(out_path: str):
    rows = []
    with open(out_path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    n = len(rows)
    keep = sum(1 for r in rows if r.get("keep") is True)
    drop = sum(1 for r in rows if r.get("keep") is False)
    err = sum(1 for r in rows if r.get("keep") is None)
    from collections import Counter
    vc = Counter(r.get("verdict") for r in rows)
    diff = Counter((r.get("blind") or {}).get("difficulty") for r in rows if r.get("blind"))
    used_pts = sum(1 for r in rows if r.get("used_points"))
    tot_tokens = sum((r.get("usage_total") or {}).get("total", 0) for r in rows)
    tot_calls = sum((r.get("usage_total") or {}).get("calls", 0) for r in rows)
    print("\n===== SUMMARY =====")
    print(f"total={n}  keep={keep} ({keep/max(1,n):.1%})  drop={drop} ({drop/max(1,n):.1%})  error={err}")
    print("verdicts:", dict(vc))
    if diff:
        print("blind difficulty:", dict(diff))
    print(f"used_points: {used_pts}/{n}")
    if tot_tokens:
        print(f"tokens: total={tot_tokens:,}  calls={tot_calls}  "
              f"avg/example={tot_tokens/max(1,n):,.0f} tok, {tot_calls/max(1,n):.2f} calls")


if __name__ == "__main__":
    main()
