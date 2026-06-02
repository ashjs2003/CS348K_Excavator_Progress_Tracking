"""
Compare ROI depth to ruler GT (.txt) and wall GT (100 cm); writes grids + heatmaps.

    python 10_evaluate_roi_gt_depth.py --run checkerboard_data/pair_001
    python 10_evaluate_roi_gt_depth.py --scene checkerboard_data --all

For all annotated pairs use: python 10_batch_roi_gt_eval_grid.py
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

from evaluation.roi_gt_compare import discover_data_pairs, roi_json_path, run_roi_gt_evaluation
from output_runs import RUNS_ROOT, run_dir_from_id


def parse_args():
    p = argparse.ArgumentParser(description="ROI GT evaluation for one run or scene")
    p.add_argument("--run", type=str, default=None)
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--all", action="store_true")
    p.add_argument("--data-root", type=Path, default=_REPO_ROOT / "data")
    p.add_argument("--tolerance-cm", type=float, default=5.0)
    p.add_argument("--heatmap-vmax-cm", type=float, default=30.0)
    return p.parse_args()


def main():
    args = parse_args()
    tol = args.tolerance_cm / 100.0

    if args.run:
        run_dir = run_dir_from_id(args.run)
        pair_id = run_dir.name.replace("pair_", "", 1)
        scene = run_dir.parent.name
        roi_path = roi_json_path(Path(args.data_root) / scene, pair_id)
        result = run_roi_gt_evaluation(
            run_dir, roi_path, _REPO_ROOT,
            tolerance_m=tol, heatmap_vmax_cm=args.heatmap_vmax_cm,
        )
        if not result or result.get("error"):
            print(result)
            return 1
        print(f"Saved grids under {run_dir / 'validation'}")
        return 0

    if args.scene and args.all:
        ok = 0
        for pair_id in discover_data_pairs(Path(args.data_root) / args.scene):
            run_id = f"{args.scene}/pair_{pair_id}"
            run_dir = run_dir_from_id(run_id)
            roi_path = roi_json_path(Path(args.data_root) / args.scene, pair_id)
            if not roi_path.is_file():
                continue
            r = run_roi_gt_evaluation(
                run_dir, roi_path, _REPO_ROOT,
                tolerance_m=tol, heatmap_vmax_cm=args.heatmap_vmax_cm,
            )
            if r and not r.get("error"):
                ok += 1
        print(f"Done: {ok} pairs")
        return 0

    print("Use --run <scene>/pair_XXX or --scene NAME --all, or run 10_batch_roi_gt_eval_grid.py")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
