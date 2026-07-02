#!/usr/bin/env python
"""Measure how reasoning_effort trades off thinking-token cost vs counting accuracy.

For each model x effort level, runs the BLIND count prompt on a few known-GT images
and reports avg input/output tokens and how close the count lands to GT. Output
tokens (mostly hidden 'thinking') are the dominant $ driver, so this tells us how
low we can push effort before accuracy degrades.

  python probe_effort.py --n-images 4 --reps 1
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time

sys.path.insert(0, ".")
from filter_counting import image_data_url, parse_json, as_int, record_view, raw_chat  # noqa: E402
sys.path.insert(0, "/home/lc1556/repo/AI-sandbox")
from sandbox import get_client  # noqa: E402
import prompts as P  # noqa: E402

DATA = "/scratch/gpfs/ZHUANGL/lc1556/vero2/datasets/combined_counting_hq/combined_counting_train.verl.jsonl"


def pick_images(n, seed=0):
    views = [record_view(json.loads(l)) for l in open(DATA)]
    random.Random(seed).shuffle(views)
    out = []
    for v in views:
        if v["ground_truth"] and 5 <= v["ground_truth"] <= 400:
            out.append(v)
        if len(out) >= n:
            break
    return out


def blind(client, model, view, effort, max_side=1536):
    url = image_data_url(view["image"], max_side)
    msgs = [
        {"role": "system", "content": P.BLIND_SYSTEM},
        {"role": "user", "content": [
            {"type": "text", "text": P.BLIND_USER.format(question=view["question"])},
            {"type": "image_url", "image_url": {"url": url}},
        ]},
    ]
    raw, u = raw_chat(client, model, msgs, retries=1, reasoning_effort=effort)
    j = parse_json(raw) or {}
    return u.get("prompt"), u.get("completion"), as_int(j.get("count"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", default=["gemini-3.1-pro-preview", "gpt-5"])
    ap.add_argument("--efforts", nargs="+", default=["minimal", "low", "medium", "high"])
    ap.add_argument("--n-images", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    imgs = pick_images(args.n_images, args.seed)
    print("probe images:", [(v["question"][:24], "GT", v["ground_truth"]) for v in imgs])
    client = get_client()

    print(f"\n{'model':24} {'effort':8} {'in':>6} {'out':>7} {'avg|err|':>9} {'exact':>6}")
    for model in args.models:
        for effort in args.efforts:
            ins, outs, errs, exact, fails = [], [], [], 0, 0
            for v in imgs:
                try:
                    pin, pout, c = blind(client, model, v, effort)
                    if pin: ins.append(pin)
                    if pout: outs.append(pout)
                    if c is not None:
                        errs.append(abs(c - v["ground_truth"]))
                        if c == v["ground_truth"]:
                            exact += 1
                except Exception as e:
                    fails += 1
                    last = repr(e)[:60]
                time.sleep(0.5)
            if not outs and fails:
                print(f"{model:24} {effort:8} ERR ({fails}/{len(imgs)}) {last}")
                continue
            ai = sum(ins) / len(ins) if ins else 0
            ao = sum(outs) / len(outs) if outs else 0
            ae = sum(errs) / len(errs) if errs else float("nan")
            print(f"{model:24} {effort:8} {ai:>6.0f} {ao:>7.0f} {ae:>9.1f} {exact}/{len(imgs):>3}"
                  + (f"  ({fails} fail)" if fails else ""))


if __name__ == "__main__":
    main()
