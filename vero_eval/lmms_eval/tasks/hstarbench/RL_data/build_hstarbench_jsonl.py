#!/usr/bin/env python3
"""Export Hstar episodes (benchmark, RL, or SFT) to hstarbench test JSONL format."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

TASK_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TASK_DIR))

from panorama_strip import save_strip  # noqa: E402

_DEFAULT_VAGEN_ROOT = Path(os.environ.get("HSTAR_VAGEN_ROOT", os.environ.get("THINKIN360_ROOT", "/home/yz5121/vero/thinkin360")))
_DEFAULT_BENCH_ROOT = Path(os.environ.get("HSTARBENCH_ROOT", "/scratch/gpfs/ZHUANGL/shared/Hstarbench"))
_DEFAULT_HOS_RL = Path("/scratch/gpfs/ZHUANGL/yz5121/thinkin360/data/hos_train")
_DEFAULT_HPS_RL = Path("/scratch/gpfs/ZHUANGL/yz5121/thinkin360/data/hps_train")
_DEFAULT_HOS_BENCH = Path("/scratch/gpfs/ZHUANGL/yz5121/thinkin360/data/hos_bench")
_DEFAULT_HPS_BENCH = Path("/scratch/gpfs/ZHUANGL/yz5121/thinkin360/data/hps_bench")
_DEFAULT_HOS_SFT = Path("/scratch/gpfs/ZHUANGL/yz5121/thinkin360/data/hos_sft_panorama")
_DEFAULT_HPS_SFT = Path("/scratch/gpfs/ZHUANGL/yz5121/thinkin360/data/hps_sft_pano")
_YAW_TOLERANCE = 30
_PITCH_TOLERANCE = 20
_SUBMIT_RE = re.compile(r"submit\s*\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)")


def _import_hstar_env(vagen_root: Path):
    sys.path.insert(0, str(vagen_root))
    from vagen.env.hstar.env import HstarEnv  # noqa: E402
    from vagen.env.hstar.env_config import HstarEnvConfig  # noqa: E402

    return HstarEnv, HstarEnvConfig


def _normalize_angle(angle: float) -> float:
    return (angle + 360) % 360


def _target_cone_from_submit(submit_yaw: int, submit_pitch: int) -> Tuple[List[float], List[float]]:
    yaw = _normalize_angle(submit_yaw)
    pitch = max(-90, min(90, submit_pitch))
    target_yaw = [_normalize_angle(yaw - _YAW_TOLERANCE), _normalize_angle(yaw + _YAW_TOLERANCE)]
    target_pitch = [max(-90, pitch - _PITCH_TOLERANCE), min(90, pitch + _PITCH_TOLERANCE)]
    return target_yaw, target_pitch


def _parse_submit(action: str) -> Optional[Tuple[int, int]]:
    match = _SUBMIT_RE.search(action or "")
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _load_hos_sft_episodes(data_path: Path) -> List[Dict[str, Any]]:
    episodes: List[Dict[str, Any]] = []
    for inst_dir in sorted(data_path.iterdir(), key=lambda p: p.name):
        if not inst_dir.is_dir():
            continue
        sft_path = inst_dir / "sft.json"
        if not sft_path.is_file():
            continue
        imgs = sorted(inst_dir.glob("*.jpg")) + sorted(inst_dir.glob("*.png"))
        if not imgs:
            continue
        erp_path = imgs[0]
        for traj in json.loads(sft_path.read_text(encoding="utf-8")):
            outputs = traj.get("outputs") or []
            if not outputs:
                continue
            submit = None
            for out in reversed(outputs):
                submit = _parse_submit(str(out.get("action", "")))
                if submit is not None:
                    break
            if submit is None:
                continue
            start_yaw, start_pitch = outputs[0].get("input_angles", [0, 0])
            target_yaw, target_pitch = _target_cone_from_submit(submit[0], submit[1])
            episodes.append(
                {
                    "task": traj["task"],
                    "erp_path": erp_path,
                    "start_yaw": int(start_yaw),
                    "start_pitch": int(start_pitch),
                    "target_yaw": target_yaw,
                    "target_pitch": target_pitch,
                    "level": 0,
                    "is_nav": False,
                }
            )
    return episodes


def _render_view(env, erp_path: Path, yaw: int, pitch: int):
    import cv2
    from PIL import Image

    pano_img = cv2.imread(str(erp_path))
    view_img = env._e2p(
        pano_img,
        env.fov,
        _normalize_angle(yaw),
        pitch,
        0,
        env.resolution[0],
        env.resolution[1],
    )
    return Image.fromarray(cv2.cvtColor(view_img, cv2.COLOR_BGR2RGB))


def export_hos_sft_split(
    *,
    split_name: str,
    data_path: Path,
    eval_set: str,
    out_path: Path,
    bench_root: Path,
    vagen_root: Path,
    limit: int = 0,
    start_seed: int = 0,
    skip_existing_assets: bool = False,
) -> int:
    HstarEnv, HstarEnvConfig = _import_hstar_env(vagen_root)
    cfg = HstarEnvConfig(
        data_path=str(data_path),
        eval_set=eval_set,
        resolution=1080,
        prompt_format="free_think",
    )
    env = HstarEnv(config=cfg)
    episodes = _load_hos_sft_episodes(data_path)
    total = len(episodes) if limit <= 0 else min(limit, len(episodes))
    if start_seed >= total:
        env.close()
        return 0

    strip_dir = bench_root / "strips"
    view_dir = bench_root / "views"
    strip_dir.mkdir(parents=True, exist_ok=True)
    view_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if start_seed > 0 and out_path.is_file() else "w"
    written = 0
    with out_path.open(mode, encoding="utf-8") as out_f:
        for seed in range(start_seed, total):
            ep = episodes[seed]
            doc_id = f"{split_name}_{seed}"
            start_yaw = int(ep["start_yaw"])
            view_abs = view_dir / f"{doc_id}_view.jpg"
            strip_abs = strip_dir / f"{doc_id}_strip.jpg"
            assets_exist = view_abs.is_file() and strip_abs.is_file()
            if not (skip_existing_assets and assets_exist):
                _render_view(env, ep["erp_path"], start_yaw, int(ep["start_pitch"])).save(view_abs, quality=90)
                save_strip(str(ep["erp_path"]), strip_abs, start_yaw_deg=start_yaw)

            row = {
                "doc_id": doc_id,
                "seed": seed,
                "split": split_name,
                "eval_set": eval_set,
                "level": int(ep["level"]),
                "task": ep["task"],
                "start_yaw": start_yaw,
                "is_nav": bool(ep["is_nav"]),
                "data_path": str(data_path),
                "erp_path": str(ep["erp_path"]),
                "panorama_strip_path": str(strip_abs),
                "view_path": str(view_abs),
                "target_yaw": ep["target_yaw"],
                "target_pitch": ep["target_pitch"],
                "prompt_format": "free_think",
                "score_mode": "zoom_align",
                "hstar_bench_root": str(bench_root),
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1
            if (seed + 1) % 100 == 0 or seed + 1 == total:
                print(f"[{split_name}] {seed + 1}/{total}", flush=True)

    env.close()
    return written


def export_split(
    *,
    split_name: str,
    data_path: Path,
    eval_set: str,
    out_path: Path,
    bench_root: Path,
    vagen_root: Path,
    limit: int = 0,
    start_seed: int = 0,
    skip_existing_assets: bool = False,
) -> int:
    HstarEnv, HstarEnvConfig = _import_hstar_env(vagen_root)
    cfg = HstarEnvConfig(
        data_path=str(data_path),
        eval_set=eval_set,
        resolution=1080,
        prompt_format="free_think",
    )
    env = HstarEnv(config=cfg)
    total = env.number_of_episodes if limit <= 0 else min(limit, env.number_of_episodes)
    if start_seed >= total:
        env.close()
        return 0

    strip_dir = bench_root / "strips"
    view_dir = bench_root / "views"
    strip_dir.mkdir(parents=True, exist_ok=True)
    view_dir.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mode = "a" if start_seed > 0 and out_path.is_file() else "w"
    written = 0
    with out_path.open(mode, encoding="utf-8") as out_f:
        for seed in range(start_seed, total):
            obs, _ = env.reset(seed=seed)
            erp_path = str(env.img_buf[env.episode_data["image_path_id"]])
            doc_id = f"{split_name}_{seed}"
            start_yaw = int(env.episode_data.get("start_yaw", env.yaw))

            view_abs = view_dir / f"{doc_id}_view.jpg"
            strip_abs = strip_dir / f"{doc_id}_strip.jpg"
            assets_exist = view_abs.is_file() and strip_abs.is_file()
            if not (skip_existing_assets and assets_exist):
                obs["multi_modal_data"][env.config.image_placeholder][0].save(view_abs, quality=90)
                save_strip(erp_path, strip_abs, start_yaw_deg=start_yaw)

            row = {
                "doc_id": doc_id,
                "seed": seed,
                "split": split_name,
                "eval_set": eval_set,
                "level": int(env.level),
                "task": env.episode_language_instruction,
                "start_yaw": start_yaw,
                "is_nav": bool(env.is_nav),
                "data_path": str(data_path),
                "erp_path": erp_path,
                "panorama_strip_path": str(strip_abs),
                "view_path": str(view_abs),
                "target_yaw": env.target_yaw,
                "target_pitch": env.target_pitch,
                "prompt_format": "free_think",
                "score_mode": "zoom_align",
                "hstar_bench_root": str(bench_root),
            }
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
            out_f.flush()
            written += 1

            env.close()
            env = HstarEnv(config=cfg)
            if (seed + 1) % 100 == 0 or seed + 1 == total:
                print(f"[{split_name}] {seed + 1}/{total}", flush=True)

    env.close()
    return written


def _concat_jsonl(sources: list[Path], dest: Path) -> int:
    dest.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with dest.open("w", encoding="utf-8") as out:
        for src in sources:
            with src.open(encoding="utf-8") as inp:
                for line in inp:
                    out.write(line)
                    n += 1
    return n


def _resolve_start_seed(out_path: Path, cli_start: int) -> int:
    if cli_start > 0:
        return cli_start
    if not out_path.is_file():
        return 0
    seeds = []
    with out_path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                seeds.append(json.loads(line)["seed"])
    return max(seeds) + 1 if seeds else 0


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--split",
        choices=("rl", "bench", "sft", "all"),
        default="rl",
        help="rl/sft/bench exports; all = bench+rl (not sft)",
    )
    p.add_argument("--only", default="", help="Run a single split name, e.g. hos_rl or hps_sft")
    p.add_argument("--hos-data", default="")
    p.add_argument("--hps-data", default="")
    p.add_argument("--bench-root", default=str(_DEFAULT_BENCH_ROOT))
    p.add_argument("--vagen-root", default=str(_DEFAULT_VAGEN_ROOT))
    p.add_argument("--limit", type=int, default=0, help="0 = all episodes per split")
    p.add_argument("--start-seed", type=int, default=0, help="Resume from this episode seed")
    p.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip strip/view regeneration when both asset files already exist",
    )
    p.add_argument(
        "--auto-resume",
        action="store_true",
        help="If --start-seed is 0, resume from max seed in output JSONL",
    )
    args = p.parse_args()

    bench_root = Path(args.bench_root)
    vagen_root = Path(args.vagen_root)
    data_dir = bench_root / "data"

    specs: list[tuple[str, Path, str, Path, str]] = []
    if args.split in ("rl", "all"):
        hos_rl = Path(args.hos_data) if args.hos_data else _DEFAULT_HOS_RL
        hps_rl = Path(args.hps_data) if args.hps_data else _DEFAULT_HPS_RL
        specs.extend(
            [
                ("hos_rl", hos_rl, "object_search", data_dir / "hstarbench_hos_rl.jsonl", "env"),
                ("hps_rl", hps_rl, "navigation", data_dir / "hstarbench_hps_rl.jsonl", "env"),
            ]
        )
    if args.split in ("sft",):
        hos_sft = Path(args.hos_data) if args.hos_data else _DEFAULT_HOS_SFT
        hps_sft = Path(args.hps_data) if args.hps_data else _DEFAULT_HPS_SFT
        specs.extend(
            [
                ("hos_sft", hos_sft, "object_search", data_dir / "hstarbench_hos_sft.jsonl", "hos_sft"),
                ("hps_sft", hps_sft, "navigation", data_dir / "hstarbench_hps_sft.jsonl", "env"),
            ]
        )
    if args.split in ("bench", "all"):
        hos_bench = Path(args.hos_data) if args.hos_data else _DEFAULT_HOS_BENCH
        hps_bench = Path(args.hps_data) if args.hps_data else _DEFAULT_HPS_BENCH
        specs.extend(
            [
                ("hos", hos_bench, "object_search", data_dir / "hstarbench_hos_test.jsonl", "env"),
                ("hps", hps_bench, "navigation", data_dir / "hstarbench_hps_test.jsonl", "env"),
            ]
        )

    if args.only:
        specs = [s for s in specs if s[0] == args.only]
        if not specs:
            raise ValueError(f"unknown --only split: {args.only}")

    counts = {}
    for split_name, data_path, eval_set, out_path, exporter in specs:
        if not data_path.is_dir():
            raise FileNotFoundError(f"missing data_path for {split_name}: {data_path}")
        start_seed = args.start_seed
        if args.auto_resume and start_seed == 0:
            start_seed = _resolve_start_seed(out_path, 0)
        elif split_name == "hps_rl" and args.start_seed > 0 and not args.only:
            start_seed = 0
        elif split_name == "hps_sft" and args.start_seed > 0 and not args.only:
            start_seed = 0

        export_fn = export_hos_sft_split if exporter == "hos_sft" else export_split
        counts[split_name] = export_fn(
            split_name=split_name,
            data_path=data_path,
            eval_set=eval_set,
            out_path=out_path,
            bench_root=bench_root,
            vagen_root=vagen_root,
            limit=args.limit,
            start_seed=start_seed,
            skip_existing_assets=args.skip_existing,
        )
        print(f"Wrote {counts[split_name]} -> {out_path}")

    if args.split == "rl" and not args.only:
        mixed = data_dir / "hstarbench_mixed_rl.jsonl"
        total = _concat_jsonl(
            [data_dir / "hstarbench_hos_rl.jsonl", data_dir / "hstarbench_hps_rl.jsonl"],
            mixed,
        )
        print(f"Wrote {total} -> {mixed}")
    elif args.split == "sft" and not args.only:
        mixed = data_dir / "hstarbench_mixed_sft.jsonl"
        total = _concat_jsonl(
            [data_dir / "hstarbench_hos_sft.jsonl", data_dir / "hstarbench_hps_sft.jsonl"],
            mixed,
        )
        print(f"Wrote {total} -> {mixed}")
    elif args.split == "bench" and not args.only:
        mixed = data_dir / "hstarbench_mixed_test.jsonl"
        total = _concat_jsonl(
            [data_dir / "hstarbench_hos_test.jsonl", data_dir / "hstarbench_hps_test.jsonl"],
            mixed,
        )
        print(f"Wrote {total} -> {mixed}")


if __name__ == "__main__":
    main()
