#!/usr/bin/env python
"""Build the PLAUSIBLE-tier manual-review website (one self-contained HTML per reviewer).

The Gemini GT-filter labels the "GT is believable but unverifiable" middle band
`PLAUSIBLE` and keeps it. This tool hands that tier to humans: it reads a
`decisions.jsonl`, keeps `verdict == "PLAUSIBLE"`, splits the ids deterministically
across the reviewers, and emits ONE dependency-free HTML page per reviewer plus an
`index.html` hub. Each page embeds its images as base64 JPEGs (like the exam report),
so a reviewer can either download the single file or serve the folder over HTTP.

Per card the reviewer sees the image (lazy, clean/points-overlay toggle, click-to-zoom
lightbox), the question, the labeled count (shown neutrally — it is what we are auditing),
the source/category, and both model passes (blind count+range, answer-aware verify). They
record **keep / drop** (required to count as reviewed), an optional **corrected count**, and
an optional note. Decisions live in `localStorage` keyed by record id (survive refreshes)
and export to a JSON that `merge_reviews.py` consumes.

  python make_review_site.py                       # defaults: full_sync decisions -> manual_review/
  python make_review_site.py --limit 8 --out-dir /tmp/smoke   # quick smoke of one small shard set

Dependency-free: stdlib + PIL, vanilla JS. Regeneration is idempotent (same inputs -> same
shard assignment; output files overwritten in place). See `dataset_filter/README.md`.
"""
from __future__ import annotations

import argparse
import base64
import html
import io
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image, ImageDraw

# --- defaults tied to the combined_counting_hq full_sync run (override on the CLI) ---
DEF_DECISIONS = ("/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/"
                 "full_sync/decisions.jsonl")
DEF_DATA_DIR = "/scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq"
DEF_OUT_DIR = ("/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/"
               "full_sync/manual_review")
DEF_REVIEWERS = ["Chris Cai", "Gabe Sarch", "Yipeng Zhang", "Haoyang Wu"]
DEF_VERDICT = "PLAUSIBLE"

VC = {"CORRECT": "#1a7f37", "PLAUSIBLE": "#9a6700", "WRONG": "#cf222e",
      "AMBIGUOUS": "#bc4c00", "UNANSWERABLE": "#8250df",
      "PARSE_ERROR": "#6e7781", "ERROR": "#6e7781"}


def slugify(name: str) -> str:
    """'Chris Cai' -> 'chris-cai' (stable filename + localStorage key component)."""
    s = re.sub(r"[^a-z0-9]+", "-", name.strip().lower()).strip("-")
    return s or "reviewer"


def load_decisions(path: str) -> list:
    return [json.loads(l) for l in open(path) if l.strip()]


def load_points_index(data_dir: str) -> dict:
    """id -> (points, original_size) from the verl splits, for drawing GT marks.

    Mirrors dataset_filter/make_exam_report.py::load_points_index so the overlay is
    identical to the audit report.
    """
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
            ei = r.get("extra_info", {})
            meta = ei.get("metadata", {})
            if meta.get("original_points"):
                idx[ei.get("id")] = (meta["original_points"], meta.get("original_size"))
        fh.close()
    return idx


def encode(img: Image.Image, side: int) -> str:
    w, h = img.size
    s = side / max(w, h)
    if s < 1:
        img = img.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=85)
    return "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode()


def two_images(path: str, side: int, points, original_size):
    """Return (clean_data_url, marked_data_url_or_None) at `side` px long edge.

    marked is None when there are no GT points (the toggle is then a no-op / hidden),
    which avoids embedding the same base64 blob twice.
    """
    try:
        base = Image.open(path).convert("RGB")
    except Exception:
        return None, None
    clean = encode(base, side)
    if not points:
        return clean, None
    m = base.copy()
    W, H = m.size
    ow, oh = (original_size if original_size and len(original_size) == 2 else (W, H))
    sx, sy = W / max(1, ow), H / max(1, oh)
    d = ImageDraw.Draw(m)
    r = max(4, round(max(W, H) / 240))
    for p in points:
        try:
            x, y = float(p[0]) * sx, float(p[1]) * sy
        except (TypeError, ValueError, IndexError):
            continue
        d.ellipse([x - r, y - r, x + r, y + r], outline=(255, 0, 255), width=max(2, r // 3))
    return clean, encode(m, side)


# --------------------------------------------------------------------------- shard split
def split_ids(ids: list, n: int, mode: str) -> list:
    """Split an already-sorted id list into n deterministic shards.

    round-robin (default): ids[k::n] -> balances source datasets across reviewers
      (ids are prefixed by source, so contiguous chunks would be lopsided).
    contiguous: near-equal contiguous slices, remainder front-loaded.
    """
    if mode == "roundrobin":
        return [ids[k::n] for k in range(n)]
    if mode == "contiguous":
        q, rem = divmod(len(ids), n)
        out, i = [], 0
        for k in range(n):
            size = q + (1 if k < rem else 0)
            out.append(ids[i:i + size])
            i += size
        return out
    raise ValueError(f"unknown split mode: {mode}")


# --------------------------------------------------------------------------- card render
def _reason(text) -> str:
    return html.escape(str(text or "")) or "<span class='muted'>—</span>"


def card_html(rec: dict, clean: str, marked, n_marks: int) -> str:
    eid = str(rec.get("id"))
    v = rec.get("verdict", "?")
    color = VC.get(v, "#6e7781")
    b = rec.get("blind") or {}
    vf = rec.get("verify") or {}
    gt = rec.get("ground_truth")
    override = rec.get("override")
    cat = html.escape(str(rec.get("category") or "")) or "—"
    src = html.escape(str(rec.get("source_dataset") or "")) or "—"
    q = html.escape(str(rec.get("question") or ""))
    path = html.escape(str(rec.get("image") or ""))

    if clean is None:
        img = '<div class="imgerr">image failed to load</div>'
    else:
        # clean lives only in src; marked (if any) once in data-marked. On first
        # toggle the JS stashes clean into data-clean, so it is never embedded twice.
        marked_attr = f' data-marked="{marked}"' if marked else ""
        img = (f'<img class="z" loading="lazy" src="{clean}"'
               f'{marked_attr} data-on="0" onclick="zoom(this)">')

    marks_ctrl = (f'<button type="button" onclick="toggleMarks(this)">show GT marks</button>'
                  f'<span class="hint">{n_marks} GT marks</span>'
                  if marked else '<span class="hint muted">no GT marks for this image</span>')

    band = f'[{b.get("count_low")}, {b.get("count_high")}]'
    blind_line = (
        f'<div class="pass"><span class="tag blind">blind</span> '
        f'count <b>{b.get("count")}</b> &middot; range {band} &middot; '
        f'difficulty {html.escape(str(b.get("difficulty") or "?"))} &middot; '
        f'conf {html.escape(str(b.get("confidence") or "?"))}'
        f'<div class="reason">{_reason(b.get("reasoning"))}</div></div>')
    verify_line = (
        f'<div class="pass"><span class="tag verify">verify</span> '
        f'count <b>{vf.get("your_count")}</b> &middot; '
        f'verdict <b style="color:{color}">{html.escape(str(vf.get("verdict") or "?"))}</b> &middot; '
        f'conf {html.escape(str(vf.get("confidence") or "?"))}'
        f'{" &middot; used GT points" if vf.get("used_points") else ""}'
        f'<div class="reason">{_reason(vf.get("reasoning"))}</div></div>')
    override_badge = (f'<span class="ov" title="flipped WRONG->PLAUSIBLE by numeric-tolerance rescue">'
                      f'override: {html.escape(str(override))}</span>' if override else "")

    return f"""<div class="card" data-id="{html.escape(eid)}" data-src="{src}">
  <div class="left">
    <div class="imgwrap">{img}</div>
    <div class="ctrls">{marks_ctrl}<span class="hint">click image to zoom</span></div>
  </div>
  <div class="mid">
    <div class="q">{q}</div>
    <div class="facts">
      <span class="gt" title="the dataset's labeled count — this is what you are auditing, not ground truth">labeled count = {gt}</span>
      {override_badge}
      <span class="chip">source: {src}</span>
      <span class="chip">category: {cat}</span>
    </div>
    {blind_line}
    {verify_line}
    <div class="sub">id={html.escape(eid)}<br>img: {path}</div>
  </div>
  <div class="right">
    <div class="decision">
      <button type="button" class="keep" data-act="keep" onclick="setDecision('{html.escape(eid)}','keep')">✓ keep</button>
      <button type="button" class="drop" data-act="drop" onclick="setDecision('{html.escape(eid)}','drop')">✕ drop</button>
    </div>
    <label class="cc">corrected count
      <input type="number" step="1" inputmode="numeric" placeholder="(if labeled count is wrong)"
             oninput="setCount('{html.escape(eid)}',this.value)"></label>
    <label class="nt">note
      <textarea rows="2" placeholder="(optional)" oninput="setNote('{html.escape(eid)}',this.value)"></textarea></label>
    <button type="button" class="clear" onclick="clearCard('{html.escape(eid)}')">clear</button>
  </div>
</div>"""


# --------------------------------------------------------------------------- page shell
HEAD = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PLAUSIBLE review — __REVIEWER__</title>
<style>
:root{--bg:#f6f8fa;--fg:#1f2328;--mut:#8b949e;--bd:#d0d7de;--keep:#1a7f37;--drop:#cf222e}
*{box-sizing:border-box}
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:var(--bg);color:var(--fg)}
header{position:sticky;top:0;background:#24292f;color:#fff;padding:10px 18px;z-index:20;box-shadow:0 1px 6px rgba(0,0,0,.3)}
header h1{margin:0;font-size:17px} header h1 small{opacity:.7;font-weight:400;font-size:13px}
.bar{display:flex;align-items:center;gap:14px;flex-wrap:wrap;margin-top:7px;font-size:13px}
.prog{font-variant-numeric:tabular-nums}
.progwrap{flex:1;min-width:160px;max-width:360px;height:8px;background:#444c56;border-radius:5px;overflow:hidden}
.progbar{height:100%;width:0;background:#2da44e;transition:width .2s}
header button,header label{font-size:12px}
header button{padding:4px 10px;border:1px solid #57606a;border-radius:6px;background:#32383f;color:#fff;cursor:pointer}
header button.primary{background:#2da44e;border-color:#2da44e;font-weight:600}
header button:hover{filter:brightness(1.12)}
.filterlbl{display:flex;align-items:center;gap:5px;cursor:pointer;user-select:none}
.tip{background:#7d4e00;color:#fff;font-size:12px;padding:5px 18px;line-height:1.4}
main{padding:14px 18px;display:flex;flex-direction:column;gap:14px;max-width:1500px;margin:0 auto}
.card{background:#fff;border:1px solid var(--bd);border-radius:10px;display:flex;gap:14px;padding:12px;scroll-margin-top:96px}
.card.reviewed{border-left:5px solid var(--mut)}
.card.kept{border-left:5px solid var(--keep)} .card.dropped{border-left:5px solid var(--drop)}
.card.focus{box-shadow:0 0 0 3px #54aeff}
body.hideReviewed .card.reviewed{display:none}
.left{flex:0 0 480px;max-width:480px}
.imgwrap{background:#0d1117;border-radius:8px;display:flex;align-items:center;justify-content:center;max-height:430px;overflow:hidden}
.imgwrap img{max-width:100%;max-height:430px;object-fit:contain;cursor:zoom-in}
.imgerr{color:var(--drop);padding:30px;font-size:13px}
.ctrls{margin-top:6px;display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.ctrls button{font-size:12px;padding:3px 9px;border:1px solid var(--bd);border-radius:6px;background:var(--bg);cursor:pointer}
.hint{color:var(--mut);font-size:11px} .muted{color:var(--mut)}
.mid{flex:1;min-width:240px}
.q{font-weight:700;font-size:15px;margin-bottom:7px;line-height:1.35}
.facts{display:flex;gap:7px;flex-wrap:wrap;align-items:center;margin-bottom:8px}
.gt{background:#fff8c5;border:1px solid #eac54f;border-radius:6px;padding:2px 9px;font-weight:700;font-size:13px}
.ov{background:#ddf4ff;border:1px solid #54aeff;color:#0969da;border-radius:6px;padding:2px 8px;font-size:11px}
.chip{background:#eef1f4;border:1px solid #e2e6ea;border-radius:6px;padding:2px 8px;font-size:12px;color:#57606a}
.pass{background:#f6f8fa;border:1px solid #eaeef2;border-radius:8px;padding:7px 10px;margin-bottom:7px;font-size:13px}
.tag{display:inline-block;color:#fff;border-radius:4px;padding:0 7px;font-size:11px;font-weight:700;margin-right:6px}
.tag.blind{background:#57606a} .tag.verify{background:#9a6700}
.reason{color:#3a3f45;font-size:12px;line-height:1.45;margin-top:3px}
.sub{color:var(--mut);font-size:11px;margin-top:6px;word-break:break-all}
.right{flex:0 0 220px;max-width:220px;display:flex;flex-direction:column;gap:8px}
.decision{display:flex;gap:8px}
.decision button{flex:1;padding:9px 0;border-radius:8px;border:2px solid var(--bd);background:#fff;font-weight:700;font-size:14px;cursor:pointer}
.decision .keep.on{background:var(--keep);border-color:var(--keep);color:#fff}
.decision .drop.on{background:var(--drop);border-color:var(--drop);color:#fff}
.right label{font-size:12px;color:#57606a;display:flex;flex-direction:column;gap:3px}
.right input,.right textarea{font-size:13px;padding:5px 7px;border:1px solid var(--bd);border-radius:6px;font-family:inherit;resize:vertical}
.clear{align-self:flex-start;font-size:11px;padding:2px 8px;border:1px solid var(--bd);border-radius:6px;background:var(--bg);color:#57606a;cursor:pointer}
#lb{display:none;position:fixed;inset:0;background:rgba(0,0,0,.94);z-index:99;overflow:auto;cursor:zoom-out;text-align:center}
#lb img{margin:10px auto;display:block;max-width:98%}
.toast{position:fixed;bottom:18px;left:50%;transform:translateX(-50%);background:#1f2328;color:#fff;padding:9px 16px;border-radius:8px;font-size:13px;z-index:120;opacity:0;transition:opacity .25s;pointer-events:none}
.toast.show{opacity:1}
@media(max-width:1100px){.card{flex-direction:column}.left,.right{flex:none;max-width:none}}
</style></head><body>
<header>
  <h1>PLAUSIBLE manual review &mdash; __REVIEWER__ <small>__TOTAL__ cards &middot; audit the dataset's labeled count</small></h1>
  <div class="bar">
    <span class="prog"><b id="ndone">0</b>/<b>__TOTAL__</b> reviewed</span>
    <span class="progwrap"><span class="progbar" id="pbar"></span></span>
    <label class="filterlbl"><input type="checkbox" id="fhide" onchange="toggleHide(this.checked)"> hide reviewed</label>
    <button type="button" onclick="jumpNext()">next unreviewed</button>
    <button type="button" class="primary" onclick="exportJSON()">Export decisions JSON</button>
    <button type="button" onclick="showHelp()">help</button>
  </div>
</header>
<div class="tip">The <b>labeled count</b> is the dataset's number — the thing you are auditing, <b>not</b> trusted ground truth. Count the image yourself (zoom, toggle GT marks), then <b>keep</b> if the labeled count is believable or <b>drop</b> if it's wrong. Add a <b>corrected count</b> when you know the right number. Keys: <b>k</b>=keep, <b>d</b>=drop, <b>j</b>/<b>n</b>=next, <b>z</b>=zoom.</div>
<main id="cards">
__CARDS__
</main>
<div id="lb" onclick="this.style.display='none'"><img id="lbi"></div>
<div class="toast" id="toast"></div>
<script>
var REVIEWER=__REVIEWER_JSON__;
var STORAGE_KEY=__STORAGE_KEY_JSON__;
var TOTAL=__TOTAL__;
var STATE={};
try{STATE=JSON.parse(localStorage.getItem(STORAGE_KEY))||{};}catch(e){STATE={};}
var focusEl=null;

function save(){try{localStorage.setItem(STORAGE_KEY,JSON.stringify(STATE));}catch(e){toast('localStorage full/blocked — export often!');}}
function ent(id){return STATE[id]||(STATE[id]={decision:null,corrected_count:null,note:""});}
function cardEl(id){return document.querySelector('.card[data-id="'+CSS.escape(id)+'"]');}

function paint(id){
  var e=STATE[id]||{},c=cardEl(id);if(!c)return;
  c.querySelectorAll('.decision button').forEach(function(b){b.classList.toggle('on',b.dataset.act===e.decision);});
  c.classList.toggle('reviewed',e.decision==='keep'||e.decision==='drop');
  c.classList.toggle('kept',e.decision==='keep');
  c.classList.toggle('dropped',e.decision==='drop');
}
function setDecision(id,act){
  var e=ent(id);e.decision=(e.decision===act?null:act);save();paint(id);updateProg();
}
function setCount(id,v){
  var e=ent(id);v=v.trim();
  e.corrected_count=(v===''?null:(isNaN(parseInt(v,10))?null:parseInt(v,10)));save();
}
function setNote(id,v){ent(id).note=v;save();}
function clearCard(id){
  delete STATE[id];save();
  var c=cardEl(id);if(c){var i=c.querySelector('input[type=number]');if(i)i.value='';var t=c.querySelector('textarea');if(t)t.value='';}
  paint(id);updateProg();
}
function reviewedCount(){var n=0;for(var k in STATE){var d=STATE[k]&&STATE[k].decision;if(d==='keep'||d==='drop')n++;}return n;}
function updateProg(){
  var n=reviewedCount();document.getElementById('ndone').textContent=n;
  document.getElementById('pbar').style.width=(TOTAL?100*n/TOTAL:0)+'%';
}
function toggleHide(on){document.body.classList.toggle('hideReviewed',on);}
function jumpNext(){
  var cards=document.querySelectorAll('.card');
  for(var i=0;i<cards.length;i++){
    var id=cards[i].dataset.id,e=STATE[id];
    if(!e||(e.decision!=='keep'&&e.decision!=='drop')){focus(cards[i]);cards[i].scrollIntoView({behavior:'smooth',block:'center'});return;}
  }
  toast('all '+TOTAL+' reviewed — export your JSON');
}
function focus(c){if(focusEl)focusEl.classList.remove('focus');focusEl=c;if(c)c.classList.add('focus');}

function zoom(img){document.getElementById('lbi').src=img.src;document.getElementById('lb').style.display='block';}
function toggleMarks(btn){
  var img=btn.closest('.card').querySelector('img.z');if(!img||!img.dataset.marked)return;
  if(img.dataset.on==='1'){img.src=img.dataset.clean;img.dataset.on='0';btn.textContent='show GT marks';}
  else{img.dataset.clean=img.src;img.src=img.dataset.marked;img.dataset.on='1';btn.textContent='hide GT marks';}
}
function toast(m){var t=document.getElementById('toast');t.textContent=m;t.classList.add('show');
  clearTimeout(t._h);t._h=setTimeout(function(){t.classList.remove('show');},2200);}

function exportJSON(){
  var out={reviewer:REVIEWER,exported_at:new Date().toISOString(),
           n_reviewed:reviewedCount(),n_total:TOTAL,decisions:STATE};
  var blob=new Blob([JSON.stringify(out,null,2)],{type:'application/json'});
  var url=URL.createObjectURL(blob),a=document.createElement('a');
  a.href=url;a.download='review_'+SLUG+'.json';document.body.appendChild(a);a.click();
  document.body.removeChild(a);URL.revokeObjectURL(url);
  toast('exported '+reviewedCount()+'/'+TOTAL+' decisions');
}
var SLUG=__SLUG_JSON__;
function showHelp(){alert(
  'PLAUSIBLE manual review — '+REVIEWER+'\\n\\n'+
  '1. For each card, count the objects yourself (zoom in; toggle GT marks to check the annotation).\\n'+
  '2. KEEP if the dataset\\'s labeled count is believable; DROP if it is clearly wrong.\\n'+
  '3. Optionally type the CORRECTED COUNT and a note.\\n'+
  '4. Progress is saved in this browser (localStorage). Refreshing is safe.\\n'+
  '5. Click \"Export decisions JSON\" and send the downloaded file back for merging.\\n\\n'+
  'Keys: k=keep, d=drop, j/n=next unreviewed, z=zoom focused image.');}

document.addEventListener('keydown',function(ev){
  var t=ev.target.tagName;if(t==='INPUT'||t==='TEXTAREA')return;
  var k=ev.key.toLowerCase();
  if(k==='j'||k==='n'){ev.preventDefault();jumpNext();}
  else if(k==='k'&&focusEl){setDecision(focusEl.dataset.id,'keep');}
  else if(k==='d'&&focusEl){setDecision(focusEl.dataset.id,'drop');}
  else if(k==='z'&&focusEl){var im=focusEl.querySelector('img.z');if(im)zoom(im);}
});
document.querySelectorAll('.card').forEach(function(c){
  c.addEventListener('click',function(ev){if(ev.target.closest('button,input,textarea,img'))return;focus(c);});
});
// hydrate UI from saved STATE
document.querySelectorAll('.card').forEach(function(c){
  var id=c.dataset.id,e=STATE[id];if(!e)return;
  if(e.corrected_count!=null){var i=c.querySelector('input[type=number]');if(i)i.value=e.corrected_count;}
  if(e.note){var tx=c.querySelector('textarea');if(tx)tx.value=e.note;}
  paint(id);
});
updateProg();
</script></body></html>"""


def render_page(reviewer: str, cards: list) -> str:
    slug = slugify(reviewer)
    page = HEAD
    page = page.replace("__CARDS__", "\n".join(cards))
    # non-JS text substitutions
    page = page.replace("__REVIEWER__", html.escape(reviewer))
    page = page.replace("__TOTAL__", str(len(cards)))
    # JS literals (json.dumps -> safe quoting)
    page = page.replace("__REVIEWER_JSON__", json.dumps(reviewer))
    page = page.replace("__SLUG_JSON__", json.dumps(slug))
    page = page.replace("__STORAGE_KEY_JSON__", json.dumps(f"plausible_review_v1::{slug}"))
    return page


INDEX = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PLAUSIBLE manual review &mdash; hub</title>
<style>
body{font-family:-apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;margin:0;background:#f6f8fa;color:#1f2328}
header{background:#24292f;color:#fff;padding:16px 24px}
header h1{margin:0;font-size:20px} header p{margin:6px 0 0;opacity:.85;font-size:14px}
main{max-width:900px;margin:0 auto;padding:22px 24px}
.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:14px;margin:8px 0 26px}
a.rev{display:block;text-decoration:none;color:inherit;background:#fff;border:1px solid #d0d7de;border-radius:10px;padding:16px 18px;transition:.15s}
a.rev:hover{border-color:#2da44e;box-shadow:0 2px 10px rgba(0,0,0,.08)}
a.rev b{font-size:16px} a.rev .n{color:#57606a;font-size:13px;margin-top:4px}
h2{font-size:16px;margin:22px 0 8px;border-bottom:1px solid #d0d7de;padding-bottom:5px}
ol,ul{line-height:1.6;font-size:14px} code{background:#eef1f4;border-radius:4px;padding:1px 5px;font-size:13px}
pre{background:#0d1117;color:#e6edf3;border-radius:8px;padding:12px 14px;overflow:auto;font-size:12.5px;line-height:1.5}
.note{background:#fff8c5;border:1px solid #eac54f;border-radius:8px;padding:10px 14px;font-size:13px}
</style></head><body>
<header><h1>PLAUSIBLE tier &mdash; manual review</h1>
<p>__TOTAL__ counting examples the GT-filter labeled PLAUSIBLE (believable-but-unverifiable), split across __NREV__ reviewers. Generated __GENERATED__.</p></header>
<main>
<h2>Your shard</h2>
<div class="grid">__LINKS__</div>
<div class="note">Progress is stored per-page in your browser (<code>localStorage</code>). Work through your shard, then click <b>Export decisions JSON</b> and send the downloaded <code>review_&lt;name&gt;.json</code> back for merging. Refreshing the page is safe; your decisions survive.</div>
<h2>What you're doing</h2>
<ol>
<li>Each card shows an image, the question, and the dataset's <b>labeled count</b> — the number we are auditing (it is <b>not</b> trusted ground truth).</li>
<li>Count the objects yourself. Zoom in (click the image); toggle the magenta <b>GT marks</b> to see where the annotation placed points.</li>
<li><b>Keep</b> if the labeled count is believable; <b>Drop</b> if it is clearly wrong. Optionally record a <b>corrected count</b> and a note. Keep/Drop is required for a card to count as reviewed.</li>
</ol>
<h2>Two ways to open your shard</h2>
<p><b>A. Download the single HTML file.</b> Everything (images included) is embedded, so you can download your <code>review_&lt;name&gt;.html</code> and open it locally in any browser — no server, no Della session needed while reviewing.</p>
<p><b>B. Serve the folder from a Della login node + VS Code port-forward.</b></p>
<pre>ssh &lt;netid&gt;@della.princeton.edu
cd __OUTDIR__
python -m http.server 8000          # pick any free port
# In VS Code (Remote-SSH to della): Ports panel -> Forward Port -> 8000
# then open http://localhost:8000/  in your local browser</pre>
<p class="note">The files are group <code>zhuangl</code>-readable, so anyone in the group can serve this same folder.</p>
</main></body></html>"""


def build(args):
    reviewers = [r.strip() for r in args.reviewers.split(",") if r.strip()]
    n = len(reviewers)
    if n < 1:
        sys.exit("[review-site] FATAL: need at least one reviewer")
    slugs = [slugify(r) for r in reviewers]
    if len(set(slugs)) != n:
        sys.exit(f"[review-site] FATAL: reviewer names collide after slugify: {slugs}")

    rows = load_decisions(args.decisions)
    plausible = [r for r in rows if r.get("verdict") == args.verdict]
    plausible.sort(key=lambda r: str(r.get("id")))
    if args.limit:
        plausible = plausible[: args.limit]
    ids = [str(r.get("id")) for r in plausible]
    if len(set(ids)) != len(ids):
        sys.exit("[review-site] FATAL: duplicate ids among selected records")
    by_id = {str(r.get("id")): r for r in plausible}
    print(f"[review-site] {len(rows)} decisions -> {len(plausible)} {args.verdict} records")

    shards = split_ids(ids, n, args.split_mode)

    # ---- shard-math assertions (verify step 1) ----
    union = set()
    for s in shards:
        assert not (union & set(s)), "shard overlap detected"
        union |= set(s)
    assert union == set(ids), "shard union != selected ids"
    assert sum(len(s) for s in shards) == len(ids), "shard sizes don't sum to total"
    print(f"[review-site] split '{args.split_mode}' -> sizes {[len(s) for s in shards]} "
          f"(union {len(union)} unique, no overlap) ✓")

    pidx = load_points_index(args.data_dir) if args.data_dir else {}
    n_pts = sum(1 for i in ids if i in pidx)
    print(f"[review-site] GT-point overlays available for {n_pts}/{len(ids)} records")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    manifest = {"generated_at": generated, "decisions": args.decisions,
                "verdict": args.verdict, "split_mode": args.split_mode,
                "side": args.side, "total": len(ids), "reviewers": {}}
    links = []
    n_img_err = 0
    for reviewer, slug, shard in zip(reviewers, slugs, shards):
        cards = []
        for k, eid in enumerate(shard):
            rec = by_id[eid]
            pts, osize = pidx.get(eid, (None, None))
            clean, marked = two_images(rec["image"], args.side, pts, osize)
            if clean is None:
                n_img_err += 1
            cards.append(card_html(rec, clean, marked, len(pts) if pts else 0))
            if (k + 1) % 50 == 0:
                print(f"    {reviewer}: {k + 1}/{len(shard)} images encoded")
        fname = f"review_{slug}.html"
        (out_dir / fname).write_text(render_page(reviewer, cards))
        sz_mb = (out_dir / fname).stat().st_size / 1e6
        print(f"[review-site] wrote {fname}  ({len(shard)} cards, {sz_mb:.1f} MB)")
        links.append(f'<a class="rev" href="{fname}"><b>{html.escape(reviewer)}</b>'
                     f'<div class="n">{len(shard)} cards &middot; {sz_mb:.0f} MB</div></a>')
        manifest["reviewers"][reviewer] = {"file": fname, "slug": slug,
                                           "count": len(shard), "ids": shard}

    index = (INDEX.replace("__TOTAL__", str(len(ids)))
                  .replace("__NREV__", str(n))
                  .replace("__GENERATED__", generated)
                  .replace("__LINKS__", "\n".join(links))
                  .replace("__OUTDIR__", html.escape(str(out_dir))))
    (out_dir / "index.html").write_text(index)
    (out_dir / "shards.json").write_text(json.dumps(manifest, indent=2))
    print(f"[review-site] wrote index.html + shards.json into {out_dir}")
    if n_img_err:
        print(f"[review-site] WARNING: {n_img_err} image(s) failed to load (rendered as errors)")
    print(f"[review-site] done. Open {out_dir}/index.html")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", default=DEF_DECISIONS,
                    help=f"decisions JSONL (train∪val, keyed by id). default: {DEF_DECISIONS}")
    ap.add_argument("--verdict", default=DEF_VERDICT,
                    help=f"verdict tier to review (default {DEF_VERDICT})")
    ap.add_argument("--data-dir", default=DEF_DATA_DIR,
                    help="dataset dir holding combined_counting_<split>.verl.jsonl for GT-point "
                         f"overlays (default {DEF_DATA_DIR}); pass '' to skip overlays")
    ap.add_argument("--reviewers", default=",".join(DEF_REVIEWERS),
                    help="comma-separated reviewer names (default the 4 assigned reviewers)")
    ap.add_argument("--split-mode", choices=["roundrobin", "contiguous"], default="roundrobin",
                    help="how to split sorted ids into shards (default roundrobin: balances "
                         "source datasets across reviewers)")
    ap.add_argument("--side", type=int, default=1100,
                    help="embedded image long edge in px (default 1100; higher = sharper zoom, "
                         "bigger file)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only take the first N PLAUSIBLE records (0 = all; for smoke tests)")
    ap.add_argument("--out-dir", default=DEF_OUT_DIR,
                    help=f"output directory for the HTML site (default {DEF_OUT_DIR})")
    args = ap.parse_args()
    build(args)


if __name__ == "__main__":
    main()
