#!/usr/bin/env python
"""Build a self-contained before/after HTML report from a decisions JSONL.

  BEFORE = a random sample of K examples (kept AND dropped, dropped flagged red).
  AFTER  = the kept subset of that same sample.

Images are embedded as small base64 JPEGs so the file is portable.

  python make_report.py --decisions runs/train_smart.jsonl --k 40 \
      --out runs/report.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import random
from pathlib import Path

from PIL import Image, ImageDraw

VERDICT_COLOR = {
    "CORRECT": "#1a7f37", "PLAUSIBLE": "#9a6700",
    "WRONG": "#cf222e", "AMBIGUOUS": "#bc4c00", "UNANSWERABLE": "#8250df",
    "PARSE_ERROR": "#6e7781", "ERROR": "#6e7781",
}


def load_points_index(data_dir):
    """id -> (points, original_size) from the verl splits, for drawing GT marks."""
    idx = {}
    for split in ("train", "val"):
        path = f"{data_dir}/combined_counting_{split}.verl.jsonl"
        try:
            f = open(path)
        except OSError:
            continue
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            ei = r.get("extra_info", {})
            meta = ei.get("metadata", {})
            pts = meta.get("original_points") or []
            if pts:
                idx[ei.get("id")] = (pts, meta.get("original_size"))
        f.close()
    return idx


def thumb(path, max_side, remap=None, points=None, original_size=None):
    if remap and path.startswith(remap[0]):
        path = remap[1] + path[len(remap[0]):]
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return None, str(e)
    if points:
        W, H = img.size
        ow, oh = (original_size if original_size and len(original_size) == 2 else (W, H))
        sx, sy = W / max(1, ow), H / max(1, oh)
        d = ImageDraw.Draw(img)
        r = max(4, round(max(W, H) / 220))
        for p in points:
            try:
                x, y = float(p[0]) * sx, float(p[1]) * sy
            except (TypeError, ValueError, IndexError):
                continue
            d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 255), width=max(2, r // 3))
    w, h = img.size
    s = max_side / max(w, h)
    if s < 1:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), None


def card(r, max_side, remap, points_index=None):
    v = r.get("verdict", "?")
    color = VERDICT_COLOR.get(v, "#6e7781")
    keep = r.get("keep")
    keep_badge = ("KEEP" if keep else "DROP" if keep is False else "ERR")
    keep_color = "#1a7f37" if keep else "#cf222e" if keep is False else "#6e7781"
    pts, osize = (points_index or {}).get(r.get("id"), (None, None))
    src, err = thumb(r["image"], max_side, remap, points=pts, original_size=osize)
    img_html = (f'<img src="{src}" loading="lazy">' if src
                else f'<div class="imgerr">image error: {html.escape(err or "")}</div>')
    pts_badge = (f'<span class="pts">◯ {len(pts)} GT marks</span>' if pts else
                 ('<span class="pts">used points</span>' if r.get("used_points") else ''))

    b = r.get("blind") or {}
    vf = r.get("verify") or {}
    blind_line = ""
    if b:
        band = f'[{b.get("count_low")},{b.get("count_high")}]'
        blind_line = (f'<div class="row"><b>blind</b>: count={b.get("count")} '
                      f'band={band} diff={b.get("difficulty")} conf={b.get("confidence")}'
                      f'<div class="reason">{html.escape(str(b.get("reasoning") or ""))}</div></div>')
    verify_line = ""
    if vf:
        verify_line = (f'<div class="row"><b>verify</b>: your_count={vf.get("your_count")} '
                       f'verdict={vf.get("verdict")} conf={vf.get("confidence")}'
                       f'<div class="reason">{html.escape(str(vf.get("reasoning") or ""))}</div></div>')

    cat = f' &middot; cat={html.escape(str(r.get("category")))}' if r.get("category") else ""
    return f"""
    <div class="card" style="border-color:{color}">
      <div class="imgwrap">{img_html}</div>
      <div class="meta">
        <div class="hdr">
          <span class="badge" style="background:{keep_color}">{keep_badge}</span>
          <span class="verdict" style="color:{color}">{v}</span>
          <span class="gt">GT={r.get("ground_truth")}</span>
          <span class="best">model={r.get("best_count")}</span>
          {pts_badge}
        </div>
        <div class="q">{html.escape(str(r.get("question") or ""))}</div>
        <div class="sub">{html.escape(str(r.get("source_dataset") or ""))}{cat} &middot; id={html.escape(str(r.get("id")))}</div>
        {blind_line}
        {verify_line}
      </div>
    </div>"""


PAGE = """<!doctype html><html><head><meta charset="utf-8">
<title>Counting GT filter — before/after</title>
<style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}}
header{{position:sticky;top:0;background:#24292f;color:#fff;padding:12px 20px;z-index:10}}
header h1{{margin:0;font-size:18px}} header .stats{{font-size:13px;opacity:.85;margin-top:4px}}
.tabs{{display:flex;gap:8px;margin-top:10px}}
.tab{{cursor:pointer;padding:6px 14px;border-radius:6px;background:#444c56;color:#fff;font-size:13px;border:none}}
.tab.active{{background:#2da44e}}
section{{padding:16px 20px;display:none}} section.active{{display:block}}
.grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px}}
.card{{background:#fff;border:2px solid #ddd;border-radius:10px;overflow:hidden;display:flex;flex-direction:column}}
.imgwrap{{background:#0d1117;display:flex;align-items:center;justify-content:center;max-height:320px;overflow:hidden}}
.imgwrap img{{max-width:100%;max-height:320px;object-fit:contain}}
.imgerr{{color:#cf222e;padding:20px;font-size:12px}}
.meta{{padding:10px 12px;font-size:13px}}
.hdr{{display:flex;align-items:center;gap:10px;margin-bottom:6px;flex-wrap:wrap}}
.badge{{color:#fff;border-radius:4px;padding:1px 8px;font-weight:700;font-size:12px}}
.verdict{{font-weight:700}} .gt{{background:#eef;border-radius:4px;padding:1px 6px}}
.best{{color:#57606a}}
.pts{{background:#fbe9ff;color:#a000c0;border:1px solid #e9b8f5;border-radius:4px;padding:1px 6px;font-size:11px}}
.q{{font-weight:600;margin:4px 0}}
.sub{{color:#8b949e;font-size:11px;margin-bottom:6px;word-break:break-all}}
.row{{margin-top:6px;border-top:1px solid #eee;padding-top:6px}}
.reason{{color:#444;margin-top:3px;font-size:12px;line-height:1.4}}
</style></head><body>
<header>
  <h1>Counting ground-truth filter — before / after</h1>
  <div class="stats">{stats}</div>
  <div class="tabs">
    <button class="tab active" onclick="show('before',this)">BEFORE — raw sample ({n_before})</button>
    <button class="tab" onclick="show('after',this)">AFTER — kept only ({n_after})</button>
    <button class="tab" onclick="show('dropped',this)">DROPPED only ({n_dropped})</button>
  </div>
</header>
<section id="before" class="active"><div class="grid">{before}</div></section>
<section id="after"><div class="grid">{after}</div></section>
<section id="dropped"><div class="grid">{dropped}</div></section>
<script>
function show(id,btn){{
  document.querySelectorAll('section').forEach(s=>s.classList.remove('active'));
  document.getElementById(id).classList.add('active');
  document.querySelectorAll('.tab').forEach(t=>t.classList.remove('active'));
  btn.classList.add('active');
}}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--k", type=int, default=40, help="random sample size for BEFORE")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--max-side", type=int, default=560)
    ap.add_argument("--out", required=True)
    ap.add_argument("--image-root-remap", default=None)
    ap.add_argument("--points-from", default=None,
                    help="dataset dir; overlay GT annotation points where available")
    args = ap.parse_args()

    remap = tuple(args.image_root_remap.split(":", 1)) if args.image_root_remap else None
    points_index = load_points_index(args.points_from) if args.points_from else None
    rows = [json.loads(l) for l in open(args.decisions) if l.strip()]
    total = len(rows)
    kept = [r for r in rows if r.get("keep") is True]
    dropped_all = [r for r in rows if r.get("keep") is False]

    rng = random.Random(args.seed)
    sample = sorted(rows, key=lambda r: r.get("id") or "")  # stable across files w/ same ids
    rng.shuffle(sample)
    sample = sample[: args.k]
    sample_kept = [r for r in sample if r.get("keep") is True]
    sample_drop = [r for r in sample if r.get("keep") is False]

    def render(rs):
        return "\n".join(card(r, args.max_side, remap, points_index) for r in rs)

    from collections import Counter
    vc = Counter(r.get("verdict") for r in rows)
    stats = (f"decisions={total} &middot; keep={len(kept)} ({len(kept)/max(1,total):.1%}) "
             f"&middot; drop={len(dropped_all)} ({len(dropped_all)/max(1,total):.1%}) "
             f"&middot; verdicts={dict(vc)} &middot; sample seed={args.seed}")

    page = PAGE.format(
        stats=stats, n_before=len(sample), n_after=len(sample_kept), n_dropped=len(sample_drop),
        before=render(sample), after=render(sample_kept), dropped=render(sample_drop))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(page)
    print(f"wrote {args.out}  (sample={len(sample)}, kept={len(sample_kept)}, dropped={len(sample_drop)})")


if __name__ == "__main__":
    main()
