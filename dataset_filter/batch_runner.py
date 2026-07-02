#!/usr/bin/env python
"""Resumable, batch-API version of the two-phase counting-dataset GT filter.

Runs the SAME smart pipeline as filter_counting.py (stage-1 blind image-only count;
stage-2 answer-aware verify with GT points overlaid + ambiguity check) but submits the
work to the Portkey **batch API** (~50% cheaper, <=24h) instead of synchronous calls.

Two rounds, because verify depends on blind:
  Round A: blind count for every example -> blind_results.jsonl
           blind == GT  -> short-circuit CORRECT/keep (no verify)
  Round B: verify the rest (blind hint + points overlay) -> verify_results.jsonl
  Combine -> decisions.jsonl (identical schema to filter_counting, so filter_apply.py
             / make_report.py / make_exam_report.py all work unchanged).

Resumable: everything is tracked in <out-dir>/manifest.json; re-running reattaches to
in-flight server-side batches instead of resubmitting. The manifest's config block is
validated against the CLI on resume and refuses to continue on a mismatch. Only rows
that parsed to a valid result count as "done"; API/parse failures are re-submitted for
up to --max-passes rounds, then surfaced as undecided (never silently kept/dropped).

Two API keys with failover: AI_SANDBOX_KEY (primary) + AI_SANDBOX_KEY2 (backup), read
from the env or the gitignored ../../AI-sandbox/.env. Backup key is never written here.

FIRST run `--selftest` to confirm this key is even allowed to use files+batches.

  python batch_runner.py --selftest --data-dir <dir>
  python batch_runner.py --data-dir <dir> --local-meta --splits train val \
      --model gemini-3.1-pro-preview --reasoning-effort high \
      --max-side 1536 --abs-tol 0 --rel-tol 0.15 --use-points --wait --out-dir <dir>/full_run
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# Reuse the filter's helpers — do NOT duplicate prompt/encode/decision logic.
sys.path.insert(0, str(Path(__file__).resolve().parent))
import prompts as P
from filter_counting import (  # noqa: E402
    load_records, record_view, image_data_url, points_data_url,
    parse_json, as_int, combine_verify, _sum_usage, VALID_VERDICTS,
)

SANDBOX_DIR = os.environ.get(
    "AI_SANDBOX_DIR", str(Path(__file__).resolve().parents[3] / "AI-sandbox"))

DEFAULT_MODEL = "gemini-3.1-pro-preview"
TAG = {"blind": "b", "verify": "v"}
TERMINAL_REMOTE = {"completed", "failed", "expired", "cancelled"}
# gemini-3.1-pro-preview list price per token (batch is ~50% off)
PRICE_IN, PRICE_OUT = 2 / 1e6, 12 / 1e6


# --------------------------------------------------------------------------- keys
KEY_VARS = ["AI_SANDBOX_KEY"] + [f"AI_SANDBOX_KEY{i}" for i in range(2, 10)]


def load_keys():
    """[primary, backup, ...] from env (overrides) + ../../AI-sandbox/.env. Never printed."""
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
        raise SystemExit(f"[batch] no AI_SANDBOX_KEY* in env or {env} — cannot authenticate.")
    return keys


def client_for(key):
    from portkey_ai import Portkey
    return Portkey(api_key=key)


def _is_quota_or_auth(err) -> bool:
    s = str(err).lower()
    return any(t in s for t in ("429", "quota", "rate limit", "rate-limit",
                                "insufficient", "401", "403", "unauthorized",
                                "permission", "exhaust"))


class ClientManager:
    """Holds the key list + current index; fails over to the backup on quota/auth errors."""
    def __init__(self, keys, start_index=0):
        self.keys = keys
        self.i = min(start_index, len(keys) - 1)
        self._clients = {}

    def client(self, index=None):
        idx = self.i if index is None else index
        if idx not in self._clients:
            self._clients[idx] = client_for(self.keys[idx])
        return self._clients[idx]

    def call(self, fn):
        """Run fn(client) on the current key; on quota/auth error advance and retry once.
        Returns (result, key_index_used)."""
        try:
            return fn(self.client(self.i)), self.i
        except Exception as e:
            if _is_quota_or_auth(e) and self.i + 1 < len(self.keys):
                print(f"[batch] key {self.i} hit quota/auth ({str(e)[:120]}); failing over to key {self.i + 1}")
                self.i += 1
                return fn(self.client(self.i)), self.i
            raise


# ----------------------------------------------------------------- request bodies
def blind_body(view, args):
    data_url = image_data_url(view["image"], args.max_side)
    msgs = [
        {"role": "system", "content": P.BLIND_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": P.BLIND_USER.format(question=view["question"])},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]},
    ]
    body = {"model": args.model, "messages": msgs}
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    return body


def verify_body(view, blind, args):
    points = view.get("points") or []
    with_points = bool(args.use_points and points)
    if with_points:
        url = points_data_url(view["image"], points, view.get("original_size"), args.max_side)
        n_pts = len(points)
    else:
        url = image_data_url(view["image"], args.max_side)
        n_pts = 0
    bb = P.blind_block(blind) if blind else ""
    pb = P.points_block(n_pts)
    msgs = [
        {"role": "system", "content": P.VERIFY_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": P.VERIFY_USER.format(
                question=view["question"], gt=view["ground_truth"],
                blind_block=bb, points_block=pb)},
            {"type": "image_url", "image_url": {"url": url}},
        ]},
    ]
    body = {"model": args.model, "messages": msgs}
    if args.reasoning_effort:
        body["reasoning_effort"] = args.reasoning_effort
    return body


def batch_line(phase, ex_id, body):
    return {"custom_id": f"{TAG[phase]}#{ex_id}", "method": "POST",
            "url": "/v1/chat/completions", "body": body}


def id_from_custom(cid):
    return cid.split("#", 1)[1]


# --------------------------------------------------------------------- chunking
def chunk_requests(items, max_bytes, max_count):
    """items: list of (id, line_dict). Greedy pack by serialized bytes AND count."""
    chunks, cur, cur_bytes = [], [], 0
    for ex_id, line in items:
        b = len(json.dumps(line)) + 1
        if cur and (cur_bytes + b > max_bytes or len(cur) >= max_count):
            chunks.append(cur); cur, cur_bytes = [], 0
        if b > max_bytes:
            print(f"[batch] WARN: single request {ex_id} is {b/1e6:.1f} MB > cap {max_bytes/1e6:.0f} MB; sending alone")
        cur.append((ex_id, line)); cur_bytes += b
    if cur:
        chunks.append(cur)
    return chunks


def write_chunk(path, lines):
    with open(path, "w") as f:
        for _, line in lines:
            f.write(json.dumps(line) + "\n")


# ------------------------------------------------------------- submit / fetch
def submit(client, path):
    f = client.files.create(file=open(path, "rb"), purpose="batch")
    b = client.batches.create(input_file_id=f.id, endpoint="/v1/chat/completions",
                              completion_window="24h")
    return f.id, b.id


def _read_content(obj):
    if obj is None:
        return ""
    if isinstance(obj, (bytes, bytearray)):
        return obj.decode("utf-8", "replace")
    if isinstance(obj, str):
        return obj
    v = getattr(obj, "text", None)
    if isinstance(v, str):
        return v
    if hasattr(obj, "read"):
        d = obj.read()
        return d.decode("utf-8", "replace") if isinstance(d, (bytes, bytearray)) else str(d)
    return str(obj)


def _parse_output_line(r):
    """-> (custom_id, content|None, usage|None, err|None)"""
    cid = r.get("custom_id")
    resp = r.get("response") or {}
    err = r.get("error")
    body = resp.get("body") if isinstance(resp, dict) else None
    code = resp.get("status_code") if isinstance(resp, dict) else None
    content = usage = None
    if body and (code in (200, None)):
        try:
            content = body["choices"][0]["message"]["content"]
        except Exception:
            content = None
        usage = body.get("usage")
    return cid, content, usage, (err or (None if content else "no_content"))


def fetch(client, batch_id):
    """Return (status, {custom_id: (content, usage, err)}) for a batch."""
    b = client.batches.retrieve(batch_id)
    status = getattr(b, "status", None)
    out = {}
    if status == "completed":
        for fid in (getattr(b, "output_file_id", None), getattr(b, "error_file_id", None)):
            if not fid:
                continue
            text = _read_content(client.files.content(fid))
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                cid, content, usage, err = _parse_output_line(r)
                if cid and (cid not in out or content is not None):
                    out[cid] = (content, usage, err)
    return status, out


# -------------------------------------------------------------- results / parse
def usage_dict(u):
    if not u:
        return {}
    def g(k):
        return u.get(k) if isinstance(u, dict) else getattr(u, k, None)
    return {"prompt": g("prompt_tokens"), "completion": g("completion_tokens"),
            "total": g("total_tokens")}


def parse_blind(content):
    j = parse_json(content) or {}
    c = as_int(j.get("count"))
    return {"count": c,
            "count_low": as_int(j.get("count_low")) if j.get("count_low") is not None else c,
            "count_high": as_int(j.get("count_high")) if j.get("count_high") is not None else c,
            "difficulty": j.get("difficulty"), "confidence": j.get("confidence"),
            "reasoning": j.get("reasoning")}


def parse_verify(content):
    j = parse_json(content) or {}
    verdict = (j.get("verdict") or "").strip().upper()
    if verdict not in VALID_VERDICTS:
        verdict = "PARSE_ERROR"
    return {"your_count": as_int(j.get("your_count")), "verdict": verdict,
            "confidence": j.get("confidence"), "reasoning": j.get("reasoning")}


def blind_ok(r):
    return bool(r and r.get("blind") and r["blind"].get("count") is not None)


def verify_ok(r):
    return bool(r and r.get("verify") and r["verify"].get("verdict") in VALID_VERDICTS)


def load_results(path):
    d = {}
    if os.path.exists(path):
        for line in open(path):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except Exception:
                continue
            d[r["id"]] = r
    return d


def append_jsonl(path, rows):
    if not rows:
        return
    with open(path, "a") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def record_phase_results(phase, fetched, results_path):
    rows = []
    for cid, (content, usage, err) in fetched.items():
        rid = id_from_custom(cid)
        u = usage_dict(usage)
        if phase == "blind":
            b = parse_blind(content) if content else None
            if b and b.get("count") is not None:
                rows.append({"id": rid, "blind": b, "usage": u})
        else:
            v = parse_verify(content) if content else None
            if v and v.get("verdict") in VALID_VERDICTS:
                rows.append({"id": rid, "verify": v, "usage": u})
    append_jsonl(results_path, rows)
    return len(rows)


# --------------------------------------------------------------------- manifest
def save_manifest(manifest, out_dir):
    tmp = os.path.join(out_dir, "manifest.json.tmp")
    with open(tmp, "w") as f:
        json.dump(manifest, f, indent=2)
    os.replace(tmp, os.path.join(out_dir, "manifest.json"))


def cfg_of(args):
    return {"model": args.model, "reasoning_effort": args.reasoning_effort,
            "abs_tol": args.abs_tol, "rel_tol": args.rel_tol,
            "use_points": bool(args.use_points), "max_side": args.max_side,
            "splits": list(args.splits), "local_meta": bool(args.local_meta),
            "data_dir": args.data_dir}


# ----------------------------------------------------------------- phase driver
def wait_for_chunks(chunks, cm, args):
    pending = {ch["batch_id"]: ch for ch in chunks}
    t0 = time.time()
    while pending:
        time.sleep(args.poll_interval)
        for bid, ch in list(pending.items()):
            try:
                b = cm.client(ch["key_index"]).batches.retrieve(bid)
                st = getattr(b, "status", None)
            except Exception as e:
                st = f"retrieve_error:{str(e)[:60]}"
            ch["status_remote"] = st
            if st in TERMINAL_REMOTE:
                del pending[bid]
        print(f"[batch] ... {len(pending)} chunk(s) still running "
              f"({int(time.time()-t0)}s elapsed)")
        if args.wait_timeout and time.time() - t0 > args.wait_timeout:
            print("[batch] wait-timeout reached; returning to reconcile.")
            return


def run_phase(phase, all_ids, build_body, cm, manifest, out_dir, results_path, args):
    """Drive one batch round to completion (or one reconcile pass if not --wait).
    Returns True when every id in all_ids has a parsed result."""
    ph = manifest["phases"].setdefault(phase, {"chunks": [], "done": False})
    ok = blind_ok if phase == "blind" else verify_ok
    all_ids = list(all_ids)

    for _pass in range(args.max_passes):
        # (a) fetch + parse any completed-but-unparsed chunks
        for ch in ph["chunks"]:
            if ch.get("status") == "parsed":
                continue
            try:
                status, fetched = fetch(cm.client(ch["key_index"]), ch["batch_id"])
            except Exception as e:
                print(f"[batch] fetch error on chunk {ch['idx']} ({phase}): {str(e)[:120]}")
                continue
            ch["status_remote"] = status
            if status == "completed":
                n = record_phase_results(phase, fetched, results_path)
                ch["status"] = "parsed"
                print(f"[batch] {phase} chunk {ch['idx']}: parsed {n}/{len(ch['ids'])}")
                save_manifest(manifest, out_dir)
            elif status in TERMINAL_REMOTE:
                ch["status"] = status
                print(f"[batch] {phase} chunk {ch['idx']}: {status} (will re-queue its ids)")
                save_manifest(manifest, out_dir)

        # (b) status
        results = load_results(results_path)
        done_ids = {i for i in all_ids if ok(results.get(i))}
        inflight = set()
        for ch in ph["chunks"]:
            if ch.get("status") not in ("parsed",) and ch.get("status") not in TERMINAL_REMOTE:
                inflight.update(ch["ids"])
        if len(done_ids) == len(all_ids):
            ph["done"] = True
            save_manifest(manifest, out_dir)
            print(f"[batch] phase {phase} COMPLETE: {len(done_ids)}/{len(all_ids)}")
            return True

        # (c) submit new chunks for anything not done and not already in flight
        todo = [i for i in all_ids if i not in done_ids and i not in inflight]
        if todo:
            print(f"[batch] {phase}: building {len(todo)} request(s) ...")
            lines = [(i, batch_line(phase, i, build_body(i))) for i in todo]
            for reqs in chunk_requests(lines, args.chunk_max_bytes, args.chunk_max_count):
                idx = len(ph["chunks"])
                path = os.path.join(out_dir, f"{phase}_{idx:03d}.jsonl")
                write_chunk(path, reqs)
                (fid, bid), ki = cm.call(lambda c, p=path: submit(c, p))
                ph["chunks"].append({"idx": idx, "path": path,
                                     "ids": [i for i, _ in reqs],
                                     "file_id": fid, "batch_id": bid,
                                     "key_index": ki, "status": "submitted"})
                manifest["current_key_index"] = cm.i
                save_manifest(manifest, out_dir)
                print(f"[batch] {phase} chunk {idx}: submitted {len(reqs)} reqs "
                      f"batch={bid} key={ki}")

        # (d) wait for in-flight chunks, or return for a manual re-run
        inflight_chunks = [ch for ch in ph["chunks"]
                           if ch.get("status") not in ("parsed",)
                           and ch.get("status_remote") not in TERMINAL_REMOTE]
        if not args.wait:
            done = len({i for i in all_ids if ok(load_results(results_path).get(i))})
            print(f"[batch] {phase}: {done}/{len(all_ids)} done, "
                  f"{len(inflight_chunks)} chunk(s) in flight. Re-run to advance "
                  f"(or add --wait to block until finished).")
            return False
        if inflight_chunks:
            print(f"[batch] {phase}: waiting on {len(inflight_chunks)} chunk(s), "
                  f"polling every {args.poll_interval}s ...")
            wait_for_chunks(inflight_chunks, cm, args)

    results = load_results(results_path)
    left = [i for i in all_ids if not ok(results.get(i))]
    ph["done"] = (len(left) == 0)
    save_manifest(manifest, out_dir)
    if left:
        print(f"[batch] phase {phase}: {len(left)} id(s) unresolved after "
              f"{args.max_passes} passes — left UNDECIDED (not kept, not dropped).")
    return ph["done"]


# ------------------------------------------------------------------- combine
def build_decision(view, blind, verify, args):
    gt = view["ground_truth"]
    out = {k: view.get(k) for k in ("id", "index", "image", "question", "ground_truth",
                                    "category", "source_dataset", "n_points",
                                    "original_size", "processed_size")}
    out.update({"strategy": "smart", "model": args.model, "abs_tol": args.abs_tol,
                "rel_tol": args.rel_tol, "reasoning_effort": args.reasoning_effort})
    points = view.get("points") or []
    out["used_points"] = bool(args.use_points and points)
    if blind:
        out["blind"] = blind
    if gt is not None and blind and blind.get("count") == gt:
        out.update({"verdict": "CORRECT", "keep": True, "best_count": blind["count"],
                    "short_circuit": True})
    elif verify:
        out["verify"] = verify
        out["short_circuit"] = False
        out["best_count"] = verify.get("your_count")
        v, keep, override = combine_verify(verify.get("verdict"), verify.get("your_count"),
                                           gt, args.abs_tol, args.rel_tol)
        out.update({"verdict": v, "keep": keep, "override": override})
    else:  # no verify result and no short-circuit -> failed/undecided
        out.update({"verdict": "ERROR", "keep": None, "best_count": None})
    out["usage_total"] = _sum_usage(out.get("blind"), out.get("verify"))
    return out


def combine(views_by_id, blind_path, verify_path, decisions_path, args):
    blind = load_results(blind_path)
    verify = load_results(verify_path)
    n = keep = drop = undecided = 0
    with open(decisions_path, "w") as f:
        for eid, view in views_by_id.items():
            b = (blind.get(eid) or {}).get("blind")
            v = (verify.get(eid) or {}).get("verify")
            # carry usage from the results rows onto the sub-dicts for _sum_usage
            if b is not None:
                b = dict(b); b["usage"] = (blind.get(eid) or {}).get("usage") or {}
            if v is not None:
                v = dict(v); v["usage"] = (verify.get(eid) or {}).get("usage") or {}
            row = build_decision(view, b, v, args)
            f.write(json.dumps(row) + "\n")
            n += 1
            if row["keep"] is True:
                keep += 1
            elif row["keep"] is False:
                drop += 1
            else:
                undecided += 1
    print(f"[batch] decisions: total={n} keep={keep} ({keep/max(1,n):.1%}) "
          f"drop={drop} ({drop/max(1,n):.1%}) undecided={undecided}")
    if undecided:
        print(f"[batch] WARNING: {undecided} undecided — rerun to resolve before applying.")
    return decisions_path


def cost_report(blind_path, verify_path):
    Pt = Ct = 0
    for path in (blind_path, verify_path):
        for r in load_results(path).values():
            u = r.get("usage") or {}
            Pt += u.get("prompt") or 0
            Ct += u.get("completion") or 0
    sync = Pt * PRICE_IN + Ct * PRICE_OUT
    print(f"[cost] prompt={Pt:,} completion={Ct:,} tok  |  "
          f"est list-price: sync=${sync:,.2f}  batch(-50%)=${sync*0.5:,.2f}")


# ------------------------------------------------------------------- selftest
def run_selftest(views, cm, args):
    two = views[:2]
    lines = [(v["id"], batch_line("blind", v["id"], blind_body(v, args))) for v in two]
    path = os.path.join(args.out_dir, "selftest.jsonl")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    write_chunk(path, lines)
    print(f"[selftest] submitting a 2-request blind batch to confirm files+batches are permitted ...")
    try:
        (fid, bid), ki = cm.call(lambda c: submit(c, path))
    except Exception as e:
        print(f"[selftest] FAILED to submit (batch API likely NOT permitted on this key):\n  {e}")
        sys.exit(2)
    print(f"[selftest] OK: submission accepted. file_id={fid} batch_id={bid} key={ki}")
    t0 = time.time()
    while True:
        try:
            status, fetched = fetch(cm.client(ki), bid)
        except Exception as e:
            print(f"[selftest] retrieve error: {e}"); status, fetched = "retrieve_error", {}
        print(f"[selftest] status={status} ({int(time.time()-t0)}s)")
        if status == "completed":
            for cid, (content, usage, err) in fetched.items():
                b = parse_blind(content) if content else None
                print(f"[selftest]   {cid}: count={(b or {}).get('count')} err={err}")
            print("[selftest] PASS — batch submit + retrieve + parse all work.")
            return
        if status in TERMINAL_REMOTE:
            print(f"[selftest] batch ended as {status}; submission works but job did not complete cleanly.")
            return
        if time.time() - t0 > args.selftest_timeout:
            print(f"[selftest] still {status} after {args.selftest_timeout}s. Submission/permission "
                  f"CONFIRMED working; batch just hasn't finished. Safe to proceed.")
            return
        time.sleep(args.poll_interval)


# ---------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir",
                    default="/scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq")
    ap.add_argument("--local-meta", action="store_true")
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reasoning-effort", default="high",
                    choices=["minimal", "low", "medium", "high"])
    ap.add_argument("--max-side", type=int, default=1536)
    ap.add_argument("--abs-tol", type=int, default=0)
    ap.add_argument("--rel-tol", type=float, default=0.15)
    ap.add_argument("--use-points", action="store_true", default=True)
    ap.add_argument("--no-points", dest="use_points", action="store_false")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--wait", action="store_true",
                    help="poll until batches finish and drive both rounds end-to-end")
    ap.add_argument("--poll-interval", type=int, default=60)
    ap.add_argument("--wait-timeout", type=int, default=0, help="0 = no limit")
    ap.add_argument("--max-passes", type=int, default=3, help="retry rounds for failed requests")
    ap.add_argument("--chunk-max-bytes", type=int, default=180_000_000)
    ap.add_argument("--chunk-max-count", type=int, default=8000)
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-timeout", type=int, default=900)
    ap.add_argument("--force", action="store_true", help="ignore manifest config mismatch")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # load dataset views (self-contained local metadata when --local-meta)
    views = []
    for split in args.splits:
        for rec in load_records(args.data_dir, split, local=args.local_meta):
            v = record_view(rec); v["split"] = split
            views.append(v)
    for v in views:
        assert "#" not in str(v["id"]), f"id contains '#': {v['id']}"
    views_by_id = {v["id"]: v for v in views}
    print(f"[batch] loaded {len(views)} views from splits={args.splits} "
          f"(local_meta={args.local_meta})")

    keys = load_keys()
    print(f"[batch] {len(keys)} API key(s) available (primary + {'backup' if len(keys) > 1 else 'NO backup'})")

    if args.selftest:
        cm = ClientManager(keys)
        run_selftest(views, cm, args)
        return

    # manifest: init or resume with config validation
    man_path = os.path.join(args.out_dir, "manifest.json")
    cfg = cfg_of(args)
    if os.path.exists(man_path):
        manifest = json.load(open(man_path))
        old = manifest.get("config", {})
        diff = {k: (old.get(k), cfg[k]) for k in cfg if old.get(k) != cfg[k]}
        if diff and not args.force:
            raise SystemExit(f"[batch] manifest config mismatch: {diff}\n"
                             f"  Use a fresh --out-dir or pass --force.")
        manifest["config"] = cfg
    else:
        manifest = {"config": cfg, "current_key_index": 0, "phases": {}}
    cm = ClientManager(keys, manifest.get("current_key_index", 0))
    save_manifest(manifest, args.out_dir)

    blind_path = os.path.join(args.out_dir, "blind_results.jsonl")
    verify_path = os.path.join(args.out_dir, "verify_results.jsonl")
    decisions_path = os.path.join(args.out_dir, "decisions.jsonl")

    # ---- Round A: blind ----
    all_ids = [v["id"] for v in views]
    a_done = run_phase("blind", all_ids, lambda i: blind_body(views_by_id[i], args),
                       cm, manifest, args.out_dir, blind_path, args)
    if not a_done:
        print("[batch] Round A (blind) not finished. Re-run to advance.")
        return

    # ---- verify subset: everything that did NOT short-circuit (blind==GT) ----
    blind = load_results(blind_path)
    verify_ids = []
    short = 0
    for eid, v in views_by_id.items():
        b = (blind.get(eid) or {}).get("blind")
        gt = v["ground_truth"]
        if gt is not None and b and b.get("count") == gt:
            short += 1
        else:
            verify_ids.append(eid)
    print(f"[batch] blind==GT short-circuit: {short}; need verify: {len(verify_ids)}")

    # ---- Round B: verify ----
    def vbuild(i):
        b = (blind.get(i) or {}).get("blind")
        return verify_body(views_by_id[i], b, args)

    b_done = run_phase("verify", verify_ids, vbuild, cm, manifest, args.out_dir, verify_path, args)
    if not b_done:
        print("[batch] Round B (verify) not finished. Re-run to advance.")
        return

    # ---- combine + report ----
    combine(views_by_id, blind_path, verify_path, decisions_path, args)
    cost_report(blind_path, verify_path)
    print(f"[batch] wrote {decisions_path}")
    print("[batch] next: python filter_apply.py --decisions "
          f"{decisions_path} --data-dir {args.data_dir}"
          f"{' --local-meta' if args.local_meta else ''} --splits {' '.join(args.splits)} "
          f"--out-dir {os.path.join(args.out_dir, 'filtered')}")


if __name__ == "__main__":
    main()
