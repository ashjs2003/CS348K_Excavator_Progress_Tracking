"""
ROI error vs GT distance — combined views.

  roi_error_pooled_median.png       — all scenes merged into one curve per method
  roi_error_by_method_faceted.png   — 2×2: one panel per method, one line per scene

Also refreshes per-scene roi_error_vs_gt_distance.png.

Use with heatmaps: 14_roi_error_all_scenes_dashboard.py, 15_roi_error_grouped_visuals.py
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

from evaluation.roi_error_vs_distance import (
    aggregate_per_scene_roi_error_by_method,
    aggregate_pooled_roi_error_vs_gt,
    process_scene,
    render_method_faceted_roi_error_chart,
    render_pooled_roi_error_chart,
)
from output_runs import RUNS_ROOT

# Cardboard + checkerboard scenes with ruler ROI GT (skip excavator).
DEFAULT_SCENES = [
    "L_carboard_box",
    "L_cardboard_box_30",
    "M_cardboard_box",
    "M_cardboardbox_30",
    "S_cardboard_box",
    "S_cardboard_box_30",
    "checkerboard_data",
    "checkerboard_data_30",
    "checkerboard_data_60",
]


def parse_args():
    p = argparse.ArgumentParser(description="Pooled ROI error overview + refresh per-scene charts")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--scenes", nargs="*", default=None, help="Scenes to pool (default: cardboard+checkerboard list)")
    p.add_argument("--skip-per-scene", action="store_true", help="Only write combined charts (no per-scene refresh)")
    p.add_argument("--n-bins", type=int, default=8, help="Bins for per-scene charts")
    return p.parse_args()


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    scene_list = args.scenes if args.scenes else DEFAULT_SCENES

    if not args.skip_per_scene:
        ok = skip = 0
        for name in scene_list:
            scene_dir = runs_root / name
            if not scene_dir.is_dir():
                skip += 1
                continue
            summary = process_scene(scene_dir, n_bins=args.n_bins)
            if summary is None:
                skip += 1
                print(f"  SKIP {name}: no ROI data")
            else:
                ok += 1
                print(f"  OK   {name} -> roi_error_vs_gt_distance.png")
        print(f"Per-scene: {ok} ok, {skip} skipped\n")

    out_dir = runs_root / "_combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    by_method = aggregate_per_scene_roi_error_by_method(runs_root, scenes=scene_list)
    if by_method is None:
        print("No ROI data for combined charts.")
        return 1

    out_faceted_json = out_dir / "roi_error_by_method_faceted.json"
    out_faceted_png = out_dir / "roi_error_by_method_faceted.png"
    out_faceted_json.write_text(json.dumps(by_method, indent=2) + "\n")
    render_method_faceted_roi_error_chart(by_method, out_faceted_png)
    print(f"Wrote {out_faceted_png}")
    print(f"Wrote {out_faceted_json}")

    pooled = aggregate_pooled_roi_error_vs_gt(runs_root, scenes=scene_list)
    if pooled is not None:
        out_json = out_dir / "roi_error_pooled_median.json"
        out_png = out_dir / "roi_error_pooled_median.png"
        out_json.write_text(json.dumps(pooled, indent=2) + "\n")
        render_pooled_roi_error_chart(pooled, out_png)
        print(f"Wrote {out_png}")
        print(f"Wrote {out_json}")

    print(f"Scenes ({by_method['n_scenes']}): {', '.join(by_method['scenes'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
