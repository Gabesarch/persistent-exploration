#!/usr/bin/env python
"""Manual-examination report: you decide the TRUE count; GT is only a candidate.

Unlike a normal report, nothing here treats the dataset ground truth as correct —
GT is shown neutrally as "candidate, verify it". For each shared example you get:
  * a high-res, click-to-zoom image (essential for counting dense scenes),
  * a toggle to show/hide the GT annotation marks (count unbiased, then reveal),
  * every model's independent count + verdict + reasoning, side by side,
  * the original image path, so you can open the full-res file if needed.
Examples where the models disagree are shown first.

  python make_exam_report.py \
      --run "gemini-3.1-pro=runs/train_points.jsonl" \
      --run "gpt-5=runs/train_points_gpt5.jsonl" \
      --run "gpt-5.5=runs/train_points_gpt55.jsonl" \
      --run "gemini-3.5-flash=runs/train_points_gem35flash.jsonl" \
      --points-from <data_dir> --out runs/MANUAL_EXAM.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

VC = {"CORRECT": "#1a7f37", "PLAUSIBLE": "#9a6700", "WRONG": "#cf222e",
      "AMBIGUOUS": "#bc4c00", "UNANSWERABLE": "#8250df", "PARSE_ERROR": "#6e7781", "ERROR": "#6e7781"}


def load(f):
    return {json.loads(l)["id"]: json.loads(l) for l in open(f) if l.strip()}


def load_points_index(data_dir):
    idx = {}
    for split in ("train", "val"):
        try:
            fh = open(f"{data_dir}/combined_counting_{split}.verl.jsonl")
        except OSError:
            continue
        for line in fh:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line); ei = r.get("extra_info", {}); meta = ei.get("metadata", {})
            if meta.get("original_points"):
                idx[ei.get("id")] = (meta["original_points"], meta.get("original_size"))
        fh.close()
    return idx


def encode(img, side):
    w, h = img.size
    s = side / max(w, h)
    if s < 1:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def two_images(path, side, points, original_size):
    """Return (clean_data_url, marked_data_url) at `side` px long edge."""
    try:
        base = Image.open(path).convert("RGB")
    except Exception:
        return None, None
    clean = encode(base, side)
    if points:
        m = base.copy(); W, H = m.size
        ow, oh = (original_size if original_size and len(original_size) == 2 else (W, H))
        sx, sy = W / max(1, ow), H / max(1, oh)
        d = ImageDraw.Draw(m); r = max(4, round(max(W, H) / 240))
        for p in points:
            try:
                x, y = float(p[0]) * sx, float(p[1]) * sy
            except (TypeError, ValueError, IndexError):
                continue
            d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 255), width=max(2, r // 3))
        marked = encode(m, side)
    else:
        marked = clean
    return clean, marked


def model_panel(label, r):
    if r is None:
        return f'<div class="mp none"><b>{html.escape(label)}</b> — no data</div>'
    v = r.get("verdict", "?"); color = VC.get(v, "#6e7781")
    keep = r.get("keep"); kb = "KEEP" if keep else "DROP" if keep is False else "ERR"
    kc = "#1a7f37" if keep else "#cf222e" if keep is False else "#6e7781"
    reason = (r.get("verify") or {}).get("reasoning") or (r.get("blind") or {}).get("reasoning") or ""
    eff = r.get("reasoning_effort") or "default"
    return f"""<div class="mp" style="border-left:4px solid {color}">
      <div class="ml"><b>{html.escape(label)}</b> <span class="eff">[{eff}]</span>
        <span class="badge" style="background:{kc}">{kb}</span></div>
      <div class="mv">says <span class="cnt">{r.get('best_count')}</span>
        &middot; <span style="color:{color};font-weight:700">{v}</span></div>
      <div class="mr">{html.escape(str(reason))}</div>
    </div>"""


def card(eid, labels, recs, gt, clean, marked, path, n_marks, i):
    any_r = next((r for r in recs if r is not None), None)
    counts = [r.get("best_count") for r in recs if r is not None and r.get("best_count") is not None]
    keeps = [bool(r.get("keep")) for r in recs if r is not None]
    disagree = len(set(keeps)) > 1
    flag = '<span class="flag">⚠ models disagree</span>' if disagree else '<span class="agree">models agree</span>'
    spread = f' &middot; model counts {min(counts)}–{max(counts)}' if counts else ""
    q = html.escape(str(any_r.get("question") or ""))
    nmarks = f'{n_marks} GT marks' if n_marks else 'no GT marks'
    img = (f'<img class="z" src="{marked}" data-clean="{clean}" data-marked="{marked}" '
           f'data-on="1" onclick="zoom(this.src)">' if clean
           else '<div class="imgerr">image error</div>')
    panels = "\n".join(model_panel(l, r) for l, r in zip(labels, recs))
    return f"""<div class="card" id="c{i}">
      <div class="left">
        <div class="imgwrap">{img}</div>
        <div class="ctrls"><button onclick="toggleMarks('c{i}')">show/hide GT marks</button>
          <span class="hint">click image to zoom &middot; {nmarks}</span></div>
        <div class="q">{q}</div>
        <div class="gtline"><span class="gt">candidate GT = {gt} (verify!)</span> {flag}{spread}</div>
        <div class="sub">{html.escape(str(any_r.get('source_dataset') or ''))} &middot; id={html.escape(str(eid))}
          <br>img: {html.escape(str(path))}</div>
      </div>
      <div class="models">{panels}</div>
    </div>"""


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Manual examination</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}}
header{{position:sticky;top:0;background:#24292f;color:#fff;padding:12px 20px;z-index:10}}
header h1{{margin:0;font-size:18px}} .stats{{font-size:13px;opacity:.9;margin-top:4px}}
.warn{{background:#7d4e00;color:#fff;font-size:12px;margin-top:6px;padding:4px 8px;border-radius:5px}}
main{{padding:16px 20px;display:flex;flex-direction:column;gap:16px}}
.card{{background:#fff;border:1px solid #d0d7de;border-radius:10px;display:flex;gap:16px;padding:12px}}
.left{{flex:0 0 560px;max-width:560px}}
.imgwrap{{background:#0d1117;border-radius:8px;display:flex;align-items:center;justify-content:center;max-height:460px;overflow:hidden}}
.imgwrap img{{max-width:100%;max-height:460px;object-fit:contain;cursor:zoom-in}}
.ctrls{{margin-top:6px}} .ctrls button{{font-size:12px;padding:3px 8px;border:1px solid #d0d7de;border-radius:5px;background:#f6f8fa;cursor:pointer}}
.hint{{color:#8b949e;font-size:11px;margin-left:8px}}
.q{{font-weight:700;font-size:15px;margin-top:8px}}
.gtline{{margin-top:4px;font-size:13px}}
.gt{{background:#fff8c5;border:1px solid #eac54f;border-radius:5px;padding:1px 8px;font-weight:700}}
.flag{{color:#bc4c00;font-weight:700}} .agree{{color:#57606a}}
.sub{{color:#8b949e;font-size:11px;margin-top:4px;word-break:break-all}}
.models{{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:8px;align-content:start}}
.mp{{background:#f6f8fa;border:1px solid #eaeef2;border-radius:8px;padding:8px 10px}} .mp.none{{opacity:.5}}
.ml{{font-size:13px;display:flex;gap:6px;align-items:center;flex-wrap:wrap}} .eff{{color:#8b949e;font-weight:400;font-size:11px}}
.badge{{color:#fff;border-radius:4px;padding:0 7px;font-size:11px;font-weight:700}}
.mv{{margin:3px 0;font-size:13px}} .cnt{{font-weight:700;font-size:15px;font-variant-numeric:tabular-nums}}
.mr{{font-size:12px;color:#444;line-height:1.4}}
#lb{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.92);z-index:99;overflow:auto;cursor:zoom-out;text-align:center}}
#lb img{{margin:10px auto;display:block}}
@media(max-width:1200px){{.card{{flex-direction:column}}.left{{flex:none;max-width:none}}}}
</style></head><body>
<header><h1>Manual examination — you decide the true count</h1>
<div class="stats">{stats}</div>
<div class="warn">⚠ The "candidate GT" is NOT trusted — it is the dataset's number that we are auditing. Count the image yourself (zoom in), use the GT marks to check completeness, then judge whether GT is right and which model agrees with reality.</div>
</header>
<main>{cards}</main>
<div id="lb" onclick="this.style.display='none'"><img id="lbi"></div>
<script>
function zoom(src){{document.getElementById('lbi').src=src;document.getElementById('lb').style.display='block';}}
function toggleMarks(id){{var im=document.querySelector('#'+id+' img.z');var on=im.dataset.on==='1';
  im.src= on? im.dataset.clean : im.dataset.marked; im.dataset.on= on?'0':'1';}}
</script></body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=path (repeatable)")
    ap.add_argument("--points-from", default=None)
    ap.add_argument("--side", type=int, default=1500, help="embedded image long edge (for zoom)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    labels, data = [], []
    for spec in args.run:
        lbl, path = spec.split("=", 1)
        labels.append(lbl); data.append(load(path))
    shared = set(data[0])
    for d in data[1:]:
        shared &= set(d)
    pidx = load_points_index(args.points_from) if args.points_from else None

    def gt_of(eid):
        for d in data:
            if eid in d and d[eid].get("ground_truth") is not None:
                return d[eid]["ground_truth"]
        return None

    def disagree(eid):
        ks = [bool(d[eid]["keep"]) for d in data if eid in d]
        return len(set(ks)) > 1

    ids = sorted(shared, key=lambda e: (not disagree(e), e))
    cards = []
    for i, eid in enumerate(ids):
        recs = [d.get(eid) for d in data]
        any_r = next((r for r in recs if r is not None), None)
        pts, osize = (pidx or {}).get(eid, (None, None))
        clean, marked = two_images(any_r["image"], args.side, pts, osize)
        cards.append(card(eid, labels, recs, gt_of(eid), clean, marked,
                          any_r["image"], len(pts) if pts else 0, i))
    n_dis = sum(1 for e in ids if disagree(e))
    stats = (f"models: {', '.join(labels)} &middot; shared examples: {len(ids)} "
             f"&middot; disagree on keep/drop: {n_dis} (shown first)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(PAGE.format(stats=stats, cards="\n".join(cards)))
    print(f"wrote {args.out}  ({len(ids)} shared, {n_dis} disagreements)")


if __name__ == "__main__":
    main()
