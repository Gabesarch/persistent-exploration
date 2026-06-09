#!/usr/bin/env python3
"""OPTIONAL: pin task-yaml `dataset_path:` HF ids to absolute local-snapshot dirs.

You usually do NOT need this: the loader (lmms_eval/api/task.py _resolve_local_dataset_path)
already resolves HF ids to the local HF cache via snapshot_download, so tasks load under
HF_HUB_OFFLINE=1 with plain HF ids. Use this only to force a *specific* local copy of one
or more datasets (the loader passes existing absolute paths through unchanged).

Idempotent: a dataset_path that is already an absolute path (or a bare local name) is left
alone. Keeps the upstream id as a trailing comment for portability.

  python scripts/pin_datasets.py --dry-run        # show what would change
  python scripts/pin_datasets.py                  # apply (uses $HF_HOME/hub)
"""
import argparse
import os
import re
from pathlib import Path

DSPATH_RE = re.compile(r'^(\s*dataset_path:\s*)(["\']?)([^"\'\n#]+?)\2\s*$')


def find_snapshot(hf_home: str, repo_id: str):
    base = Path(hf_home) / "hub" / f"datasets--{repo_id.replace('/', '--')}" / "snapshots"
    if not base.is_dir():
        return None
    snaps = [p for p in base.iterdir() if p.is_dir()]
    if not snaps:
        return None
    return str(sorted(snaps, key=lambda p: p.stat().st_mtime)[-1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--hf-home", default=os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"))
    ap.add_argument("--tasks-dir", default=None)
    args = ap.parse_args()

    pkg = Path(__file__).resolve().parent.parent  # vero_eval/
    tasks_dir = Path(args.tasks_dir) if args.tasks_dir else pkg / "lmms_eval" / "tasks"
    pinned = skipped = missing = 0
    miss_ids = set()

    for yml in sorted(tasks_dir.glob("*/*.yaml")):
        lines = yml.read_text().splitlines(keepends=True)
        changed = False
        for i, line in enumerate(lines):
            m = DSPATH_RE.match(line.rstrip("\n"))
            if not m:
                continue
            val = m.group(3).strip()
            if val.startswith("/") or "/" not in val:   # already a path / a bare local name
                skipped += 1
                continue
            snap = find_snapshot(args.hf_home, val)
            if not snap:
                missing += 1
                miss_ids.add(val)
                continue
            lines[i] = f"{m.group(1)}{snap}  # upstream: {val}\n"
            changed = True
            pinned += 1
            print(f"  pin  {val:42s} -> {snap}")
        if changed and not args.dry_run:
            yml.write_text("".join(lines))

    print(f"\npinned={pinned}  already_local={skipped}  missing_from_cache={missing}  dry_run={args.dry_run}")
    if miss_ids:
        print("MISSING from cache (download or pin manually):")
        for mid in sorted(miss_ids):
            print(f"  - {mid}")


if __name__ == "__main__":
    main()
