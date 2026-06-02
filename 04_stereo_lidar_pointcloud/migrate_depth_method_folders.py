"""
Reorganize flat depth/ files into shared/ + per-method subfolders.

    python migrate_depth_method_folders.py --run checkerboard_data/pair_001 --dry-run
    python migrate_depth_method_folders.py --all
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from depth_layout import migrate_depth_folder
from output_runs import RUNS_ROOT, iter_run_dirs, resolve_run_dir


def parse_args():
    p = argparse.ArgumentParser(description="Migrate depth/ to shared/ + method subfolders")
    p.add_argument("--run", default=None, help="Single run id, e.g. checkerboard_data/pair_001")
    p.add_argument("--all", action="store_true", help="Migrate every run under outputs/runs/")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.run:
        runs = [(args.run, resolve_run_dir(args.run))]
    elif args.all:
        runs = iter_run_dirs()
    else:
        raise SystemExit("Pass --run <id> or --all")

    total_moves = 0
    for run_id, run_dir in runs:
        depth = run_dir / "depth"
        if not depth.is_dir():
            continue
        lines = migrate_depth_folder(depth, dry_run=args.dry_run)
        if lines:
            print(f"\n{run_id}:")
            for line in lines:
                print(f"  {line}")
            total_moves += len(lines)

    print(f"\n{'Would move' if args.dry_run else 'Moved'} {total_moves} file(s) across {len(runs)} run(s).")


if __name__ == "__main__":
    main()
