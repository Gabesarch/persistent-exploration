#!/usr/bin/env python
"""Side-by-side multi-model comparison report.

One card per shared example: the image (with GT annotation points overlaid),
the question + GT, and a panel PER MODEL showing that model's count, verdict,
keep/drop, and reasoning — so a human can look at the picture and decide which
model is right. Examples where the models DISAGREE are shown first.

  python make_compare_report.py \
      --run "gemini-3.1-pro=runs/train_points.jsonl" \
      --run "gpt-5=runs/train_points_gpt5.jsonl" \
      --run "gpt-5.5=runs/train_points_gpt55.jsonl" \
      --run "gemini-3.5-flash=runs/train_points_gem35flash.jsonl" \
      --points-from <data_dir> --out runs/compare.html
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
from pathlib import Path

from PIL import Image, ImageDraw

VERDICT_COLOR = {
    "CORRECT": "#1a7f37", "PLAUSIBLE": "#9a6700",
    "WRONG": "#cf222e", "AMBIGUOUS": "#bc4c00", "UNANSWERABLE": "#8250df",
    "PARSE_ERROR": "#6e7781", "ERROR": "#6e7781",
}


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
            r = json.loads(line)
            ei = r.get("extra_info", {}); meta = ei.get("metadata", {})
            if meta.get("original_points"):
                idx[ei.get("id")] = (meta["original_points"], meta.get("original_size"))
        fh.close()
    return idx


def thumb(path, max_side, points=None, original_size=None):
    try:
        img = Image.open(path).convert("RGB")
    except Exception as e:
        return None, str(e)
    if points:
        W, H = img.size
        ow, oh = (original_size if original_size and len(original_size) == 2 else (W, H))
        sx, sy = W / max(1, ow), H / max(1, oh)
        d = ImageDraw.Draw(img); r = max(4, round(max(W, H) / 220))
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
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=82)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode(), None


def model_panel(label, r, gt):
    if r is None:
        return f'<div class="mp none"><div class="ml">{html.escape(label)}</div><div class="mv">— no data —</div></div>'
    v = r.get("verdict", "?"); color = VERDICT_COLOR.get(v, "#6e7781")
    keep = r.get("keep"); kb = "KEEP" if keep else "DROP" if keep is False else "ERR"
    kc = "#1a7f37" if keep else "#cf222e" if keep is False else "#6e7781"
    cnt = r.get("best_count")
    match = ""
    if cnt is not None and gt is not None:
        match = ' style="color:#1a7f37;font-weight:700"' if cnt == gt else ' style="color:#cf222e"'
    reason = (r.get("verify") or {}).get("reasoning") or (r.get("blind") or {}).get("reasoning") or ""
    eff = r.get("reasoning_effort") or "default"
    return f"""<div class="mp" style="border-left:4px solid {color}">
      <div class="ml">{html.escape(label)} <span class="eff">[{eff}]</span></div>
      <div class="mv"><span class="badge" style="background:{kc}">{kb}</span>
        <span style="color:{color};font-weight:700">{v}</span>
        <span class="cnt"{match}>count={cnt}</span></div>
      <div class="mr">{html.escape(str(reason))}</div>
    </div>"""


def card(eid, models, recs, gt, max_side, pidx):
    pts, osize = (pidx or {}).get(eid, (None, None))
    any_r = next((r for r in recs if r is not None), None)
    src, err = thumb(any_r["image"], max_side, pts, osize)
    img_html = f'<img src="{src}" loading="lazy">' if src else f'<div class="imgerr">{html.escape(err or "")}</div>'
    counts = [r.get("best_count") for r in recs if r is not None]
    keeps = [bool(r.get("keep")) for r in recs if r is not None]
    disagree = len(set(keeps)) > 1
    spread = (max(counts) - min(counts)) if counts and None not in counts else None
    flag = '<span class="flag">⚠ disagree</span>' if disagree else '<span class="agree">✓ agree</span>'
    q = html.escape(str(any_r.get("question") or ""))
    cat = f' &middot; cat={html.escape(str(any_r.get("category")))}' if any_r.get("category") else ""
    npts = f' &middot; {len(pts)} GT marks' if pts else ""
    panels = "\n".join(model_panel(lbl, r, gt) for lbl, r in zip(models, recs))
    return f"""<div class="card">
      <div class="left">
        <div class="imgwrap">{img_html}</div>
        <div class="info">
          <div class="q">{q}</div>
          <div class="gtline"><span class="gt">GT = {gt}</span> {flag}
             {f'&middot; count spread={spread}' if spread is not None else ''}{npts}</div>
          <div class="sub">{html.escape(str(any_r.get("source_dataset") or ""))}{cat} &middot; id={html.escape(str(eid))}</div>
        </div>
      </div>
      <div class="models">{panels}</div>
    </div>"""


PAGE = """<!doctype html><html><head><meta charset="utf-8"><title>Model comparison</title><style>
body{{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}}
header{{position:sticky;top:0;background:#24292f;color:#fff;padding:12px 20px;z-index:10}}
header h1{{margin:0;font-size:18px}} header .stats{{font-size:13px;opacity:.85;margin-top:4px}}
.legend{{font-size:12px;margin-top:6px;opacity:.9}}
main{{padding:16px 20px;display:flex;flex-direction:column;gap:16px}}
.card{{background:#fff;border:1px solid #d0d7de;border-radius:10px;display:flex;gap:14px;padding:12px;overflow:hidden}}
.left{{flex:0 0 420px;max-width:420px}}
.imgwrap{{background:#0d1117;border-radius:8px;display:flex;align-items:center;justify-content:center;max-height:360px;overflow:hidden}}
.imgwrap img{{max-width:100%;max-height:360px;object-fit:contain}}
.imgerr{{color:#cf222e;padding:20px}}
.info{{margin-top:8px}} .q{{font-weight:700;font-size:14px}}
.gtline{{margin-top:4px;font-size:13px}} .gt{{background:#ddf4ff;border:1px solid #b6e3ff;border-radius:5px;padding:1px 8px;font-weight:700}}
.flag{{color:#bc4c00;font-weight:700}} .agree{{color:#1a7f37;font-weight:700}}
.sub{{color:#8b949e;font-size:11px;margin-top:4px;word-break:break-all}}
.models{{flex:1;display:grid;grid-template-columns:1fr 1fr;gap:8px;align-content:start}}
.mp{{background:#f6f8fa;border:1px solid #eaeef2;border-radius:8px;padding:8px 10px}}
.mp.none{{opacity:.5}}
.ml{{font-weight:700;font-size:13px}} .eff{{color:#8b949e;font-weight:400;font-size:11px}}
.mv{{margin:4px 0;font-size:13px;display:flex;gap:8px;align-items:center;flex-wrap:wrap}}
.badge{{color:#fff;border-radius:4px;padding:0 7px;font-size:11px;font-weight:700}}
.cnt{{font-variant-numeric:tabular-nums}}
.mr{{font-size:12px;color:#444;line-height:1.4;margin-top:2px}}
@media(max-width:1100px){{.card{{flex-direction:column}}.left{{flex:none;max-width:none}}.models{{grid-template-columns:1fr 1fr}}}}
</style></head><body>
<header><h1>Per-example model comparison — which model is right?</h1>
<div class="stats">{stats}</div>
<div class="legend">Magenta rings = dataset GT points. Per model: <b>count</b> green if it equals GT (GT may itself be wrong — judge from the image). Disagreement cards first.</div>
</header>
<main>{cards}</main>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="append", required=True, help="label=path (repeatable)")
    ap.add_argument("--points-from", default=None)
    ap.add_argument("--max-side", type=int, default=620)
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

    ids = sorted(shared, key=lambda e: (not disagree(e), e))  # disagreements first
    cards = "\n".join(
        card(eid, labels, [d.get(eid) for d in data], gt_of(eid), args.max_side, pidx)
        for eid in ids)
    n_dis = sum(1 for e in ids if disagree(e))
    stats = (f"models: {', '.join(labels)} &middot; shared examples: {len(ids)} "
             f"&middot; models disagree on keep/drop: {n_dis}")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(PAGE.format(stats=stats, cards=cards))
    print(f"wrote {args.out}  ({len(ids)} shared, {n_dis} disagreements)")


if __name__ == "__main__":
    main()
