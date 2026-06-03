"""
LiDAR edge-profile width vs ruler GT distance for cardboard S / M / L at 0° and −30°.

Output:
  outputs/runs/_combined/lidar_width_vs_gt_distance.png
  outputs/runs/_combined/lidar_width_vs_gt_distance.json
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

from evaluation.lidar_width_vs_gt import (
    collect_lidar_width_points,
    render_width_vs_gt_chart,
    write_summary_json,
)
from output_runs import RUNS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description="LiDAR width vs ruler GT distance chart")
    p.add_argument("--data-root", type=Path, default=_REPO_ROOT / "data")
    p.add_argument("--out-dir", type=Path, default=RUNS_ROOT / "_combined")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    data_root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    points = collect_lidar_width_points(data_root)
    if not any(points.values()):
        print("No cardboard LiDAR captures found under data/")
        return 1

    png = out_dir / "lidar_width_vs_gt_distance.png"
    json_path = out_dir / "lidar_width_vs_gt_distance.json"
    render_width_vs_gt_chart(points, png)
    write_summary_json(points, json_path)
    print(f"Wrote {png}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
