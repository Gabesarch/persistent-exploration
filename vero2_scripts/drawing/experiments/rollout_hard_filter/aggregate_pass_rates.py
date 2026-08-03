#!/usr/bin/env python3
"""Aggregate the val-only rollout dumps into per-sample pass rates and emit the hard set.

Reads every `validation_data/0.jsonl` produced by the shard runs (one json row per rollout;
all n rollouts of a sample share `debug.uid`, and `debug.extra_info.id` is the stable
dataset key), computes per-sample pass rates with `correct := accuracy >= threshold`
(graded counting accuracy is exactly 1.0 inside the scale-aware full-credit band, so the
default 0.999 equals training's notion of "correct"), joins them back onto the source
jsonl, and writes:

  <source stem>.annotated_passrate.verl.jsonl   every source row + avg_pass_rate etc.
                                                (any future threshold re-cut is free)
  <source stem>.rollout_passrate_leq<thr>.verl.jsonl   rows with avg_pass_rate <= threshold
  stats.md / stats.json                         histogram, GT x pass-rate crosstab, cuts

Usage:
  python aggregate_pass_rates.py \
    --dumps "${SAVE_PATH}/checkpoints/vero2_verl/basefilter_counting_drawonly_shard0*/validation_data/0.jsonl" \
    --source /scratch/.../combined_counting_train.correct.partial.verl.jsonl \
    --out-dir /scratch/gpfs/ZHUANGL/lc1556/vero2/hard_filter_runs/output \
    --expected-n 8 --correct-threshold 0.999 --keep-max-pass-rate 0.5 \
    [--allow-incomplete] [--annotate-only]
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

GT_BUCKETS = (
    ("lt10", lambda v: v < 10),
    ("10to19", lambda v: 10 <= v < 20),
    ("20to49", lambda v: 20 <= v < 50),
    ("ge50", lambda v: v >= 50),
)
REPORT_CUTS = (0.5, 0.625, 0.75)


def bucket_of(gt_value: float) -> str:
    for name, pred in GT_BUCKETS:
        if pred(gt_value):
            return name
    raise AssertionError(f"unbucketed ground truth {gt_value!r}")


def rollout_accuracy(row: dict) -> float:
    if "accuracy" in row and row["accuracy"] is not None:
        acc = row["accuracy"]
    else:
        acc = ((row.get("debug") or {}).get("reward_extra_info") or {}).get("accuracy")
    if acc is None:
        raise SystemExit(f"rollout row has no accuracy field: keys={sorted(row)[:20]}")
    acc = float(acc)
    if not 0.0 <= acc <= 1.0:
        raise SystemExit(f"accuracy out of [0,1]: {acc}")
    return acc


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dumps", required=True, help="glob over the shard dump jsonl files")
    parser.add_argument("--source", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--expected-n", type=int, default=8)
    parser.add_argument("--correct-threshold", type=float, default=0.999)
    parser.add_argument("--keep-max-pass-rate", type=float, default=0.5)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="tolerate missing ids / rollout counts != expected-n (smoke)")
    parser.add_argument("--annotate-only", action="store_true", help="skip the filtered output")
    args = parser.parse_args()

    dump_files = sorted(glob.glob(args.dumps))
    if not dump_files:
        raise SystemExit(f"no dump files match {args.dumps!r}")
    print(f"reading {len(dump_files)} dump file(s):")
    for f in dump_files:
        print(f"  {f}")

    # uid -> rollout rows; then re-key by the stable dataset id.
    by_uid: dict[str, list[dict]] = defaultdict(list)
    for dump_file in dump_files:
        with open(dump_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                debug = row.get("debug") or {}
                uid = debug.get("uid")
                if uid is None:
                    raise SystemExit(
                        f"dump row without debug.uid in {dump_file} — was "
                        "trainer.comprehensive_debug_logging.enabled true for this run?"
                    )
                by_uid[uid].append(row)

    per_id: dict[str, dict] = {}
    for uid, rows in by_uid.items():
        ids = {((r.get("debug") or {}).get("extra_info") or {}).get("id") for r in rows}
        if len(ids) != 1 or None in ids:
            raise SystemExit(f"uid {uid} maps to inconsistent extra_info.id set: {ids}")
        row_id = ids.pop()
        if row_id in per_id:
            raise SystemExit(f"extra_info.id {row_id!r} appears under two uids (duplicate shard run?)")
        accs = [rollout_accuracy(r) for r in rows]
        n = len(accs)
        if n != args.expected_n and not args.allow_incomplete:
            raise SystemExit(f"id {row_id!r}: {n} rollouts, expected {args.expected_n}")
        n_correct = sum(1 for a in accs if a >= args.correct_threshold)
        num_turns = [((r.get("debug") or {}).get("__num_turns__")) for r in rows]
        num_turns = [t for t in num_turns if t is not None]
        tool_calls = [r.get("tool_call_count") for r in rows if r.get("tool_call_count") is not None]
        per_id[row_id] = {
            "n_rollouts": n,
            "n_correct": n_correct,
            "avg_pass_rate": n_correct / n,
            "mean_graded_accuracy": sum(accs) / n,
            "mean_score": sum(float(r.get("score", 0.0)) for r in rows) / n,
            "mean_tool_calls": sum(tool_calls) / len(tool_calls) if tool_calls else None,
            "mean_num_turns": sum(num_turns) / len(num_turns) if num_turns else None,
        }

    source_rows: list[dict] = []
    with open(args.source, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                source_rows.append(json.loads(line))
    source_ids = {row["extra_info"]["id"] for row in source_rows}
    if len(source_ids) != len(source_rows):
        raise SystemExit("duplicate extra_info.id in source")

    missing = source_ids - per_id.keys()
    extra = per_id.keys() - source_ids
    print(f"coverage: {len(per_id)}/{len(source_ids)} source ids measured"
          f" ({len(missing)} missing, {len(extra)} not in source)")
    if missing and not args.allow_incomplete:
        preview = sorted(missing)[:10]
        raise SystemExit(f"{len(missing)} source ids missing from dumps, e.g. {preview} "
                         "(rerun the failed shard, or pass --allow-incomplete)")
    if extra:
        raise SystemExit(f"{len(extra)} dump ids not in source, e.g. {sorted(extra)[:10]}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(args.source).name.replace(".verl.jsonl", "")

    annotated_path = out_dir / f"{stem}.annotated_passrate.verl.jsonl"
    thr_tag = f"{args.keep_max_pass_rate:g}".replace("0.", "0").replace(".", "")
    filtered_path = out_dir / f"{stem}.rollout_passrate_leq{thr_tag}.verl.jsonl"

    annotated = 0
    kept_rows: list[dict] = []
    hist: dict[str, int] = defaultdict(int)
    crosstab: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    by_source_ds: dict[str, list[float]] = defaultdict(list)
    with open(annotated_path, "w", encoding="utf-8") as f:
        for row in source_rows:
            row_id = row["extra_info"]["id"]
            stats = per_id.get(row_id)
            if stats is None:
                continue  # only reachable with --allow-incomplete
            out_row = dict(row)
            out_row.update(stats)
            f.write(json.dumps(out_row, ensure_ascii=False) + "\n")
            annotated += 1

            rate_bin = f"{stats['n_correct']}/{stats['n_rollouts']}"
            hist[rate_bin] += 1
            gt = int(float(str(row["reward_model"]["ground_truth"]).strip()))
            crosstab[bucket_of(gt)][rate_bin] += 1
            by_source_ds[row["extra_info"].get("source_dataset", "?")].append(stats["avg_pass_rate"])
            if stats["avg_pass_rate"] <= args.keep_max_pass_rate:
                kept_rows.append(out_row)

    if not args.annotate_only:
        with open(filtered_path, "w", encoding="utf-8") as f:
            for row in kept_rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    all_rates = [s["avg_pass_rate"] for s in per_id.values()]
    stats_obj = {
        "dumps": dump_files,
        "source": args.source,
        "measured_rows": annotated,
        "expected_n": args.expected_n,
        "correct_threshold": args.correct_threshold,
        "keep_max_pass_rate": args.keep_max_pass_rate,
        "mean_pass_rate": sum(all_rates) / len(all_rates),
        "histogram": dict(sorted(hist.items())),
        "crosstab_gt_bucket_x_pass": {b: dict(sorted(c.items())) for b, c in crosstab.items()},
        "per_source_dataset": {
            ds: {
                "rows": len(rates),
                "mean_pass_rate": sum(rates) / len(rates),
                "kept_fraction": sum(1 for r in rates if r <= args.keep_max_pass_rate) / len(rates),
            }
            for ds, rates in sorted(by_source_ds.items())
        },
        "cut_sizes": {
            f"keep_leq_{cut:g}": sum(1 for r in all_rates if r <= cut) for cut in REPORT_CUTS
        },
        "kept_rows": len(kept_rows),
    }
    with open(out_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats_obj, f, indent=2)

    lines = [
        "# Rollout pass-rate filter stats",
        "",
        f"- source: `{args.source}` ({len(source_rows)} rows, {annotated} measured)",
        f"- rollouts per sample: {args.expected_n}, correct := accuracy >= {args.correct_threshold}",
        f"- mean pass rate: {stats_obj['mean_pass_rate']:.3f}",
        f"- KEPT (avg_pass_rate <= {args.keep_max_pass_rate:g}): **{len(kept_rows)}** rows",
        "",
        "## Pass-rate histogram (n_correct / n)",
        "",
        "| pass | rows |",
        "|---|---|",
    ]
    def bin_key(bin_label: str) -> tuple[int, int]:
        num, den = bin_label.split("/")
        return int(den), int(num)

    ordered_bins = sorted(hist, key=bin_key)
    lines += [f"| {k} | {hist[k]} |" for k in ordered_bins]
    lines += ["", "## GT bucket x pass rate", "", "| bucket | " + " | ".join(ordered_bins) + " | mean pass |", "|" + "---|" * (len(ordered_bins) + 2)]
    for bucket, _ in GT_BUCKETS:
        row_counts = crosstab.get(bucket, {})
        total = sum(row_counts.values())
        if not total:
            continue
        mean_rate = sum(int(k.split("/")[0]) / int(k.split("/")[1]) * v for k, v in row_counts.items()) / total
        lines.append(f"| {bucket} | " + " | ".join(str(row_counts.get(k, 0)) for k in ordered_bins) + f" | {mean_rate:.3f} |")
    lines += ["", "## Per source dataset", "", "| source | rows | mean pass | kept frac |", "|---|---|---|---|"]
    for ds, info in stats_obj["per_source_dataset"].items():
        lines.append(f"| {ds} | {info['rows']} | {info['mean_pass_rate']:.3f} | {info['kept_fraction']:.3f} |")
    lines += ["", "## Alternative cuts", ""]
    lines += [f"- keep <= {cut:g}: {stats_obj['cut_sizes'][f'keep_leq_{cut:g}']} rows" for cut in REPORT_CUTS]
    (out_dir / "stats.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"annotated -> {annotated_path} ({annotated} rows)")
    if not args.annotate_only:
        print(f"filtered  -> {filtered_path} ({len(kept_rows)} rows)")
    print(f"stats     -> {out_dir / 'stats.md'}")
    print(f"mean pass rate {stats_obj['mean_pass_rate']:.3f}; cuts: {stats_obj['cut_sizes']}")


if __name__ == "__main__":
    main()
