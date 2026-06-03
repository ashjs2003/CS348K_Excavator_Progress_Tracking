"""
Box depth (ROI − flap) vs catalog GT for cardboard S / M / L at 0° and −30°.

Output:
  outputs/runs/_combined/box_depth_vs_gt_distance.png
  outputs/runs/_combined/box_depth_vs_gt_distance.json
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

from evaluation.depth_vs_gt import (
    collect_depth_points,
    render_depth_vs_gt_chart,
    write_summary_json,
)
from output_runs import RUNS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description="Box depth vs GT distance chart")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--out-dir", type=Path, default=RUNS_ROOT / "_combined")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    runs_root = Path(args.runs_root)
    out_dir = Path(args.out_dir)
    points = collect_depth_points(runs_root)
    if not any(points.values()):
        print(
            "No depth rows found. Run 12_estimate_box_volume_from_roi.py for cardboard scenes first."
        )
        return 1

    png = out_dir / "box_depth_vs_gt_distance.png"
    json_path = out_dir / "box_depth_vs_gt_distance.json"
    render_depth_vs_gt_chart(points, png)
    write_summary_json(points, json_path)
    print(f"Wrote {png}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
