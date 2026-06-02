"""
Per-scene chart: ROI |Z_est − GT| vs ground-truth depth (pooled annotated captures).

    python 11_roi_scene_error_vs_distance.py
    python 11_roi_scene_error_vs_distance.py --scene checkerboard_data
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

from evaluation.roi_error_vs_distance import collect_scene_roi_pairs, process_scene
from output_runs import RUNS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description="ROI error vs GT distance — one chart per scene")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--n-bins", type=int, default=8)
    return p.parse_args()


def scenes_with_roi(runs_root: Path) -> list[Path]:
    out = []
    for scene_dir in sorted(runs_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        if collect_scene_roi_pairs(scene_dir):
            out.append(scene_dir)
    return out


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)

    if args.scene:
        scene_dirs = [runs_root / args.scene]
    else:
        scene_dirs = scenes_with_roi(runs_root)

    if not scene_dirs:
        print("No scenes with ROI per-point CSVs found.")
        return 1

    ok = skip = 0
    for scene_dir in scene_dirs:
        if not scene_dir.is_dir():
            print(f"  SKIP {scene_dir.name}: not found")
            skip += 1
            continue
        summary = process_scene(scene_dir, n_bins=args.n_bins)
        if summary is None:
            print(f"  SKIP {scene_dir.name}: insufficient ROI data")
            skip += 1
            continue
        ok += 1
        n = summary["n_pairs"]
        pts = sum(summary["methods"][m]["n_points"] for m in summary["methods"])
        print(f"  OK   {scene_dir.name} ({n} pairs, {pts} pixels) -> roi_error_vs_gt_distance.png")

    print(f"\nDone: {ok} scenes, {skip} skipped")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
