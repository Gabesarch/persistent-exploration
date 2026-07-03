#!/usr/bin/env python
"""Offline ablation harness for the scale-aware counting reward.

Replays real ``(pred, gt)`` counting pairs through the shipped reward and a sweep
of variants (old exact-match vs new graded, band width ``rel_tol``, decay width
``rel_sigma``/``min_sigma``, band-only vs graded, Gaussian vs Laplacian decay) and
emits distribution plots + summary tables. It imports the *real*
``counting_reward_value`` from ``verl.utils.reward_score.math_verify_reward_type_boxed``
(the Phase-B-plumbed function), so every variant is exactly a set of Hydra
``reward_kwargs.counting_*`` overrides -- no reward math is re-implemented here
except the old exact-match baseline.

Realistic pairs come from the Gemini ground-truth filter's ``decisions.jsonl``:
``blind.count`` (a strong VLM's count, produced without ever seeing the GT) vs
``ground_truth``. blind counts are a stand-in for policy predictions -- their error
distribution is not identical to an RL policy's, but it is the best realistic proxy
of "how far off is a capable counter, and by how much" that we have offline.

Regenerate everything with one command:

    python vero2_scripts/debug_scripts/counting_reward_ablation.py

Outputs land in ``docs/vero2/assets/counting-reward-ablation/`` by default:
PNG plots, ``variant_summary.{csv,md}``, ``by_gt_bucket.csv``, and ``summary.json``.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# Import the real, Phase-B-plumbed reward so variants == config overrides.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from verl.utils.reward_score.math_verify_reward_type_boxed import (  # noqa: E402
    _COUNTING_MIN_SIGMA,
    _COUNTING_REL_SIGMA,
    _COUNTING_REL_TOL,
    counting_reward_value,
)

DEFAULT_DECISIONS = (
    "/scratch/gpfs/ZHUANGL/lc1556/vero2/dataset_filter_runs/full_sync/decisions.jsonl"
)
DEFAULT_OUTDIR = _REPO_ROOT / "docs/vero2/assets/counting-reward-ablation"

# Reference values (TASK.md) -- the harness must reproduce these for the default config.
REFERENCE_VALUES = [
    ((110, 100), 0.8948),
    ((8, 7), 0.4037),
    ((0, 1), 0.0183),
]

GT_BUCKETS = [
    ("1-4", 1, 5),
    ("5-9", 5, 10),
    ("10-19", 10, 20),
    ("20-49", 20, 50),
    ("50-99", 50, 100),
    ("100+", 100, math.inf),
]
ABS_ERR_BUCKETS = [
    ("0", 0, 1),
    ("1", 1, 2),
    ("2-3", 2, 4),
    ("4-9", 4, 10),
    ("10+", 10, math.inf),
]


# --------------------------------------------------------------------------- #
# Reward variants                                                             #
# --------------------------------------------------------------------------- #
def _old_exact(pred: int, gt: int) -> float:
    """The baseline arm: exact integer match -> 1.0 else 0.0.

    Mirrors ``_short_string_reward``'s integer path (the training entrypoint before
    PR #40): no partial credit, no band.
    """
    return 1.0 if int(pred) == int(gt) else 0.0


def _graded(**knobs):
    def fn(pred: int, gt: int) -> float:
        return counting_reward_value(pred, gt, **knobs)

    return fn


def build_variants() -> dict[str, dict]:
    """Return ``{name: {"fn", "group", "knobs", "label"}}`` for every ablation arm."""
    variants: dict[str, dict] = {}

    def add(name, fn, group, knobs, label):
        variants[name] = {"fn": fn, "group": group, "knobs": knobs, "label": label}

    # (1) headline: old exact-match vs new graded default
    add("old_exact", _old_exact, "headline", {"reward": "exact_match"}, "old (exact match)")
    add("new_default", _graded(), "headline", {}, "new (graded, shipped)")

    # (2) rel_tol (band fraction) sweep
    for rt in (0.0, 0.02, 0.05, 0.10):
        add(
            f"rel_tol_{rt:g}",
            _graded(rel_tol=rt),
            "rel_tol",
            {"counting_rel_tol": rt},
            f"rel_tol={rt:g}" + (" (shipped)" if rt == _COUNTING_REL_TOL else ""),
        )

    # (3) rel_sigma x min_sigma decay-width grid
    for rs in (0.05, 0.15, 0.30):
        for ms in (0.25, 0.5, 1.0):
            add(
                f"sigma_rs{rs:g}_ms{ms:g}",
                _graded(rel_sigma=rs, min_sigma=ms),
                "sigma_grid",
                {"counting_rel_sigma": rs, "counting_min_sigma": ms},
                f"rel_sigma={rs:g}, min_sigma={ms:g}"
                + (" (shipped)" if rs == _COUNTING_REL_SIGMA and ms == _COUNTING_MIN_SIGMA else ""),
            )

    # (4) band-only vs graded -- isolates gradient signal from leniency
    add("graded", _graded(), "band_vs_graded", {}, "graded (Gaussian tail)")
    add("band_only", _graded(graded=False), "band_vs_graded", {"counting_graded": False}, "band-only (hard 0)")

    # (5) Gaussian (exp 2) vs Laplacian (exp 1) decay
    add("gaussian_exp2", _graded(decay_exp=2.0), "decay_shape", {}, "Gaussian (exp 2, shipped)")
    add("laplacian_exp1", _graded(decay_exp=1.0), "decay_shape", {"counting_decay_exp": 1.0}, "Laplacian (exp 1)")

    return variants


# --------------------------------------------------------------------------- #
# Data loading                                                                #
# --------------------------------------------------------------------------- #
def _as_int(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    if isinstance(x, float) and float(x).is_integer():
        return int(x)
    return None


def load_pairs_from_decisions(path: str, max_records: int | None = None) -> list[dict]:
    """Load ``(pred=blind.count, gt=ground_truth)`` pairs from a filter decisions.jsonl."""
    pairs: list[dict] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            gt = _as_int(rec.get("ground_truth"))
            blind = rec.get("blind") or {}
            pred = _as_int(blind.get("count"))
            if gt is None or pred is None or gt < 0 or pred < 0:
                continue
            pairs.append(
                {
                    "pred": pred,
                    "gt": gt,
                    "verdict": rec.get("verdict"),
                    "source_dataset": rec.get("source_dataset"),
                }
            )
            if max_records and len(pairs) >= max_records:
                break
    return pairs


def _parse_boxed_int(text: str):
    if not isinstance(text, str):
        return None
    m = None
    for m in _re_boxed().finditer(text):
        pass
    if m is None:
        return None
    return _as_int_str(m.group(1))


def _re_boxed():
    import re

    return re.compile(r"\\boxed\{\s*([0-9][0-9,]*)\s*\}")


def _as_int_str(s: str):
    try:
        return int(s.replace(",", ""))
    except (TypeError, ValueError):
        return None


def load_pairs_from_reward_assignments(paths: list[str], max_records: int | None = None) -> list[dict]:
    """Load counting ``(pred, gt)`` pairs from real training reward_assignments.jsonl logs.

    Only rows with ``extra_info.reward_type == "counting"`` and a parseable boxed
    integer prediction + integer ground truth are kept. Returns [] if none found.
    """
    pairs: list[dict] = []
    for path in paths:
        try:
            fh = open(path)
        except OSError:
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ei = rec.get("extra_info") or {}
                if (ei.get("reward_type") or rec.get("reward_type")) != "counting":
                    continue
                gt = _as_int_str(str(rec.get("ground_truth")))
                pred = _parse_boxed_int(rec.get("response") or "")
                if gt is None or pred is None:
                    continue
                pairs.append({"pred": pred, "gt": gt, "verdict": None, "source_dataset": rec.get("data_source")})
                if max_records and len(pairs) >= max_records:
                    return pairs
    return pairs


# --------------------------------------------------------------------------- #
# Aggregation                                                                 #
# --------------------------------------------------------------------------- #
def _bucket(value, buckets):
    for name, lo, hi in buckets:
        if lo <= value < hi:
            return name
    return buckets[-1][0]


def evaluate(pairs: list[dict], variants: dict[str, dict]) -> dict:
    preds = np.array([p["pred"] for p in pairs], dtype=float)
    gts = np.array([p["gt"] for p in pairs], dtype=float)
    abs_err = np.abs(preds - gts)
    gt_bucket = np.array([_bucket(g, GT_BUCKETS) for g in gts])
    err_bucket = np.array([_bucket(e, ABS_ERR_BUCKETS) for e in abs_err])
    exact_mask = abs_err == 0

    results: dict[str, dict] = {}
    for name, spec in variants.items():
        fn = spec["fn"]
        rewards = np.array([fn(p["pred"], p["gt"]) for p in pairs], dtype=float)
        in_band = (rewards >= 1.0) & (~exact_mask)
        in_decay = (rewards > 0.0) & (rewards < 1.0)
        zero = rewards <= 0.0
        nonexact = ~exact_mask

        by_gt = {}
        for bname, _, _ in GT_BUCKETS:
            m = gt_bucket == bname
            by_gt[bname] = float(rewards[m].mean()) if m.any() else float("nan")
        by_err = {}
        for bname, _, _ in ABS_ERR_BUCKETS:
            m = err_bucket == bname
            by_err[bname] = float(rewards[m].mean()) if m.any() else float("nan")

        results[name] = {
            "label": spec["label"],
            "group": spec["group"],
            "knobs": spec["knobs"],
            "mean_reward": float(rewards.mean()),
            "frac_exact": float(exact_mask.mean()),
            "frac_in_band": float(in_band.mean()),
            "frac_in_decay": float(in_decay.mean()),
            "frac_zero": float(zero.mean()),
            # signal density: among NON-exact preds, share that get any partial credit,
            # and their mean reward -- the "learning signal on near-misses" the ablation cares about.
            "nonexact_partial_frac": float((in_decay | in_band)[nonexact].mean()) if nonexact.any() else 0.0,
            "nonexact_mean_reward": float(rewards[nonexact].mean()) if nonexact.any() else 0.0,
            "mean_reward_by_gt": by_gt,
            "mean_reward_by_err": by_err,
            "_rewards": rewards,  # kept in-memory for plotting; stripped before JSON
        }
    return {
        "n_pairs": len(pairs),
        "exact_frac": float(exact_mask.mean()),
        "gt_bucket_counts": {b: int((gt_bucket == b).sum()) for b, _, _ in GT_BUCKETS},
        "err_bucket_counts": {b: int((err_bucket == b).sum()) for b, _, _ in ABS_ERR_BUCKETS},
        "variants": results,
    }


# --------------------------------------------------------------------------- #
# Plots                                                                       #
# --------------------------------------------------------------------------- #
def _savefig(fig, outdir: Path, name: str):
    path = outdir / name
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return path


def plot_headline(res: dict, outdir: Path):
    v = res["variants"]
    cats = ["frac_exact", "frac_in_band", "frac_in_decay", "frac_zero"]
    labels = ["exact", "in-band", "in-decay", "zero"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # (a) reward-category composition, old vs new
    names = ["old_exact", "new_default"]
    x = np.arange(len(names))
    bottom = np.zeros(len(names))
    colors = ["#2a9d8f", "#8ab17d", "#e9c46a", "#e76f51"]
    for cat, lab, c in zip(cats, labels, colors):
        vals = np.array([v[n][cat] for n in names])
        axes[0].bar(x, vals, bottom=bottom, label=lab, color=c)
        bottom += vals
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([v[n]["label"] for n in names])
    axes[0].set_ylabel("fraction of pairs")
    axes[0].set_title("Reward composition: old vs new")
    axes[0].legend(fontsize=8, loc="center right")
    # (b) mean reward by gt bucket
    bnames = [b for b, _, _ in GT_BUCKETS]
    for n, style in zip(names, ["o-", "s--"]):
        axes[1].plot(bnames, [v[n]["mean_reward_by_gt"][b] for b in bnames], style, label=v[n]["label"])
    axes[1].set_xlabel("ground-truth count bucket")
    axes[1].set_ylabel("mean reward")
    axes[1].set_title("Mean reward vs count magnitude")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    return _savefig(fig, outdir, "headline_old_vs_new.png")


def plot_reward_histograms(res: dict, outdir: Path, names: list[str], title: str, fname: str):
    v = res["variants"]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bins = np.linspace(0, 1, 26)
    for n in names:
        r = v[n]["_rewards"]
        ax.hist(r, bins=bins, histtype="step", linewidth=1.8, label=v[n]["label"], density=True)
    ax.set_xlabel("reward")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return _savefig(fig, outdir, fname)


def plot_rel_tol(res: dict, outdir: Path):
    v = res["variants"]
    names = [n for n in v if v[n]["group"] == "rel_tol"]
    names.sort(key=lambda n: v[n]["knobs"].get("counting_rel_tol", 0))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    x = np.arange(len(names))
    axes[0].bar(x, [v[n]["mean_reward"] for n in names], color="#457b9d")
    axes[0].bar(x, [v[n]["frac_exact"] + v[n]["frac_in_band"] for n in names], color="#a8dadc", alpha=0.7, label="frac full-credit")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([v[n]["label"] for n in names], rotation=15, fontsize=8)
    axes[0].set_ylabel("value")
    axes[0].set_title("rel_tol: mean reward & full-credit share")
    axes[0].legend(fontsize=8)
    bnames = [b for b, _, _ in GT_BUCKETS]
    for n in names:
        axes[1].plot(bnames, [v[n]["mean_reward_by_gt"][b] for b in bnames], "o-", label=v[n]["label"])
    axes[1].set_xlabel("ground-truth count bucket")
    axes[1].set_ylabel("mean reward")
    axes[1].set_title("rel_tol: mean reward vs count magnitude")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=7)
    axes[1].grid(alpha=0.3)
    return _savefig(fig, outdir, "rel_tol_sweep.png")


def plot_sigma_heatmap(res: dict, outdir: Path):
    v = res["variants"]
    rss = [0.05, 0.15, 0.30]
    mss = [0.25, 0.5, 1.0]
    grid = np.full((len(rss), len(mss)), np.nan)
    for i, rs in enumerate(rss):
        for j, ms in enumerate(mss):
            grid[i, j] = v[f"sigma_rs{rs:g}_ms{ms:g}"]["mean_reward"]
    fig, ax = plt.subplots(figsize=(6.2, 5))
    im = ax.imshow(grid, cmap="viridis", aspect="auto", origin="lower")
    ax.set_xticks(range(len(mss)))
    ax.set_xticklabels([f"{m:g}" for m in mss])
    ax.set_yticks(range(len(rss)))
    ax.set_yticklabels([f"{r:g}" for r in rss])
    ax.set_xlabel("min_sigma (width floor)")
    ax.set_ylabel("rel_sigma (width fraction)")
    ax.set_title("Mean reward over decay-width grid")
    for i in range(len(rss)):
        for j in range(len(mss)):
            ax.text(j, i, f"{grid[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)
    fig.colorbar(im, ax=ax, label="mean reward")
    return _savefig(fig, outdir, "sigma_grid_heatmap.png")


def plot_reward_curves(outdir: Path):
    """Illustrative reward(pred) curves for a few gt values under key variants."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    for ax, gt in zip(axes, (5, 20, 100)):
        preds = np.arange(max(0, gt - int(1.2 * gt) - 3), gt + int(1.2 * gt) + 4)
        ax.plot(preds, [counting_reward_value(int(p), gt) for p in preds], "-", label="default (Gaussian)", color="#1d3557")
        ax.plot(preds, [counting_reward_value(int(p), gt, decay_exp=1.0) for p in preds], "--", label="Laplacian (exp1)", color="#e76f51")
        ax.plot(preds, [counting_reward_value(int(p), gt, graded=False) for p in preds], ":", label="band-only", color="#2a9d8f")
        ax.axvline(gt, color="gray", lw=0.8, alpha=0.6)
        ax.set_title(f"gt = {gt}")
        ax.set_xlabel("predicted count")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("reward")
    axes[0].legend(fontsize=8)
    return _savefig(fig, outdir, "reward_curves.png")


def plot_band_vs_graded_and_decay(res: dict, outdir: Path):
    v = res["variants"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    # band-only vs graded: mean reward by abs-error bucket
    bnames = [b for b, _, _ in ABS_ERR_BUCKETS]
    for n, style in (("graded", "o-"), ("band_only", "s--")):
        axes[0].plot(bnames, [v[n]["mean_reward_by_err"][b] for b in bnames], style, label=v[n]["label"])
    axes[0].set_xlabel("abs error bucket")
    axes[0].set_ylabel("mean reward")
    axes[0].set_title("Band-only vs graded (signal on near-misses)")
    axes[0].set_ylim(0, 1.02)
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)
    # gaussian vs laplacian: mean reward by abs-error bucket
    for n, style in (("gaussian_exp2", "o-"), ("laplacian_exp1", "s--")):
        axes[1].plot(bnames, [v[n]["mean_reward_by_err"][b] for b in bnames], style, label=v[n]["label"])
    axes[1].set_xlabel("abs error bucket")
    axes[1].set_ylabel("mean reward")
    axes[1].set_title("Gaussian vs Laplacian decay")
    axes[1].set_ylim(0, 1.02)
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3)
    return _savefig(fig, outdir, "band_vs_graded_and_decay.png")


# --------------------------------------------------------------------------- #
# Tables                                                                      #
# --------------------------------------------------------------------------- #
def write_tables(res: dict, outdir: Path):
    v = res["variants"]
    cols = ["mean_reward", "frac_exact", "frac_in_band", "frac_in_decay", "frac_zero", "nonexact_partial_frac", "nonexact_mean_reward"]
    header = ["variant", "group"] + cols
    # CSV
    csv_lines = [",".join(header)]
    for name, d in v.items():
        row = [name, d["group"]] + [f"{d[c]:.4f}" for c in cols]
        csv_lines.append(",".join(row))
    (outdir / "variant_summary.csv").write_text("\n".join(csv_lines) + "\n")

    # Markdown
    md = ["| variant | group | mean R | exact | in-band | in-decay | zero | near-miss signal | near-miss mean R |",
          "|---|---|---|---|---|---|---|---|---|"]
    for name, d in v.items():
        md.append(
            f"| `{name}` | {d['group']} | {d['mean_reward']:.3f} | {d['frac_exact']:.3f} | "
            f"{d['frac_in_band']:.3f} | {d['frac_in_decay']:.3f} | {d['frac_zero']:.3f} | "
            f"{d['nonexact_partial_frac']:.3f} | {d['nonexact_mean_reward']:.3f} |"
        )
    (outdir / "variant_summary.md").write_text("\n".join(md) + "\n")

    # by-gt-bucket CSV
    bnames = [b for b, _, _ in GT_BUCKETS]
    lines = ["variant," + ",".join(bnames)]
    for name, d in v.items():
        lines.append(name + "," + ",".join(f"{d['mean_reward_by_gt'][b]:.4f}" for b in bnames))
    (outdir / "by_gt_bucket.csv").write_text("\n".join(lines) + "\n")


def write_summary_json(res: dict, outdir: Path, meta: dict):
    clean = json.loads(json.dumps(res, default=lambda o: None))  # drop numpy arrays
    for name in clean["variants"]:
        clean["variants"][name].pop("_rewards", None)
    clean["meta"] = meta
    (outdir / "summary.json").write_text(json.dumps(clean, indent=2) + "\n")


# --------------------------------------------------------------------------- #
# Reference-value sanity check                                                #
# --------------------------------------------------------------------------- #
def check_reference_values(tol: float = 5e-4) -> bool:
    ok = True
    print("Reference-value check (default config, TASK.md):")
    for (pred, gt), expected in REFERENCE_VALUES:
        got = counting_reward_value(pred, gt)
        passed = abs(got - expected) <= tol
        ok = ok and passed
        print(f"  gt={gt:>4} pred={pred:>4}: got {got:.4f}  expected {expected:.4f}  {'OK' if passed else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------- #
# Main                                                                        #
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--decisions", default=DEFAULT_DECISIONS, help="filter decisions.jsonl (blind.count vs ground_truth)")
    ap.add_argument("--reward-assignments", nargs="*", default=None, help="optional reward_assignments.jsonl logs to add real policy counting preds")
    ap.add_argument("--output-dir", default=str(DEFAULT_OUTDIR))
    ap.add_argument("--max-records", type=int, default=None)
    ap.add_argument("--check-only", action="store_true", help="only run the reference-value sanity check and exit")
    args = ap.parse_args()

    ref_ok = check_reference_values()
    if args.check_only:
        sys.exit(0 if ref_ok else 1)
    if not ref_ok:
        print("ERROR: default reward no longer reproduces the reference values.", file=sys.stderr)
        sys.exit(1)

    pairs = load_pairs_from_decisions(args.decisions, args.max_records)
    source = f"decisions.jsonl ({len(pairs)} blind-count pairs)"
    if args.reward_assignments:
        extra = load_pairs_from_reward_assignments(args.reward_assignments, args.max_records)
        print(f"Loaded {len(extra)} counting pairs from reward_assignments logs.")
        pairs += extra
        source += f" + {len(extra)} policy pairs"
    if not pairs:
        print("ERROR: no usable (pred, gt) pairs loaded.", file=sys.stderr)
        sys.exit(1)
    print(f"Loaded {len(pairs)} (pred, gt) pairs from {args.decisions}")

    variants = build_variants()
    res = evaluate(pairs, variants)

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    written = [
        plot_headline(res, outdir),
        plot_reward_histograms(res, outdir, ["old_exact", "new_default"], "Reward distribution: old vs new", "hist_old_vs_new.png"),
        plot_rel_tol(res, outdir),
        plot_sigma_heatmap(res, outdir),
        plot_reward_curves(outdir),
        plot_band_vs_graded_and_decay(res, outdir),
    ]
    write_tables(res, outdir)
    write_summary_json(res, outdir, {"source": source, "n_pairs": len(pairs), "decisions_path": args.decisions})

    print(f"\nWrote {len(written)} plots + tables + summary.json to {outdir}")
    print("\nVariant summary (mean reward | exact | in-band | in-decay | zero):")
    for name, d in res["variants"].items():
        print(
            f"  {name:24s} R={d['mean_reward']:.3f}  exact={d['frac_exact']:.3f}  "
            f"band={d['frac_in_band']:.3f}  decay={d['frac_in_decay']:.3f}  zero={d['frac_zero']:.3f}"
        )


if __name__ == "__main__":
    main()
