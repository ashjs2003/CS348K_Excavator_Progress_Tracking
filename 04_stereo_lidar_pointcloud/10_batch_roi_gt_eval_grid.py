"""
Batch ROI GT evaluation: numeric grid + heatmaps for every annotated pair.

Skips 100 cm wall GT on excavator_M and excavator_S (ruler .txt only there).

    python 10_batch_roi_gt_eval_grid.py
    python 10_batch_roi_gt_eval_grid.py --scene checkerboard_data
    python 10_batch_roi_gt_eval_grid.py --run checkerboard_data/pair_001
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.roi_gt_compare import discover_all_roi_jobs, run_roi_gt_evaluation
from output_runs import RUNS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description="Batch ROI GT grids (table + heatmap)")
    p.add_argument("--data-root", type=Path, default=_REPO_ROOT / "data")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--run", type=str, default=None)
    p.add_argument("--scene", type=str, default=None)
    p.add_argument("--tolerance-cm", type=float, default=5.0)
    p.add_argument("--heatmap-vmax-cm", type=float, default=30.0)
    return p.parse_args()


def main():
    args = parse_args()
    jobs = discover_all_roi_jobs(args.data_root, args.runs_root)

    if args.run:
        scene, pair_id = args.run.rsplit("/", 1)[-2], args.run.rsplit("/", 1)[-1].replace("pair_", "")
        jobs = [j for j in jobs if j[0] == scene and j[1] == pair_id]
    elif args.scene:
        jobs = [j for j in jobs if j[0] == args.scene]

    if not jobs:
        print("No ROI + run pairs found.")
        return 1

    print(f"Processing {len(jobs)} captures...\n")
    ok = fail = 0
    scene_summaries: dict[str, list] = {}

    for scene, pair_id, roi_path, run_dir in jobs:
        run_id = f"{scene}/pair_{pair_id}"
        result = run_roi_gt_evaluation(
            run_dir,
            roi_path,
            _REPO_ROOT,
            tolerance_m=args.tolerance_cm / 100.0,
            heatmap_vmax_cm=args.heatmap_vmax_cm,
        )
        if result is None or result.get("error"):
            print(f"  FAIL {run_id}: {result.get('error', 'unknown')}")
            fail += 1
            continue
        ok += 1
        wall = "wall" if result.get("use_wall_gt") else "no wall"
        print(f"  OK   {run_id} ({wall}) -> validation/roi_gt_eval_grid.png")
        scene_summaries.setdefault(scene, []).append(
            {"pair_id": pair_id, "run_id": run_id, "summary": result}
        )

    for scene, entries in scene_summaries.items():
        out = args.runs_root / scene / "roi_gt_eval_batch_summary.json"
        out.write_text(json.dumps(entries, indent=2, default=str) + "\n")

    print(f"\nDone: {ok} ok, {fail} failed")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
