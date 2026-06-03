"""
Polar LiDAR setup grid (checkerboard example, then cardboard S/M/L).

    python 19_lidar_polar_setup_grid.py --preset checkerboard
    python 19_lidar_polar_setup_grid.py --preset cardboard --size L

Outputs under outputs/runs/_combined/:
    lidar_polar_checkerboard_setup.png
    lidar_polar_L_box_setup.png  (etc.)
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

from evaluation.lidar_polar_setup_grid import (
    CARDBOARD_DISTANCE_COLUMNS_CM,
    DEFAULT_DISTANCE_COLUMNS_CM,
    render_cardboard_sml_combined_grid,
    render_setup_grid,
)
from output_runs import RUNS_ROOT

PRESETS = {
    "checkerboard": {
        "scenes": ["checkerboard_data", "checkerboard_data_30", "checkerboard_data_60"],
        "target_label": "Checkerboard",
        "title": "LiDAR setup — 2D checkerboard (GT distance × placement angle)",
        "out_name": "lidar_polar_checkerboard_setup.png",
    },
    "cardboard_L": {
        "scenes": ["L_carboard_box", "L_cardboard_box_30"],
        "target_label": "L box",
        "title": "LiDAR setup — L cardboard box",
        "out_name": "lidar_polar_L_box_setup.png",
    },
    "cardboard_M": {
        "scenes": ["M_cardboard_box", "M_cardboardbox_30"],
        "target_label": "M box",
        "title": "LiDAR setup — M cardboard box",
        "out_name": "lidar_polar_M_box_setup.png",
    },
    "cardboard_S": {
        "scenes": ["S_cardboard_box", "S_cardboard_box_30"],
        "target_label": "S box",
        "title": "LiDAR setup — S cardboard box",
        "out_name": "lidar_polar_S_box_setup.png",
    },
    "cardboard_sml": {
        "combined": True,
        "title": "LiDAR setup — cardboard S / M / L (edge width)",
        "out_name": "lidar_polar_SML_box_setup.png",
    },
}


def parse_args():
    p = argparse.ArgumentParser(description="Polar LiDAR GT setup grid")
    p.add_argument("--data-root", type=Path, default=_REPO_ROOT / "data")
    p.add_argument("--out-dir", type=Path, default=RUNS_ROOT / "_combined")
    p.add_argument(
        "--preset",
        choices=["checkerboard", "cardboard", "cardboard_sml", "all"],
        default="checkerboard",
    )
    p.add_argument("--size", choices=["L", "M", "S"], default=None, help="With --preset cardboard")
    p.add_argument("--scenes", nargs="*", default=None)
    p.add_argument("--distance-tol-cm", type=float, default=5.0)
    p.add_argument(
        "--angular-tol-deg",
        type=float,
        default=20.0,
        help="Bearing gate half-width (degrees) around placement or auto bearing",
    )
    p.add_argument(
        "--bearing-mode",
        choices=["setup", "auto", "placement", "distance_only"],
        default="setup",
        help="setup=placement gate then auto fallback (default); placement|auto|distance_only",
    )
    p.add_argument(
        "--coplanar-tol-cm",
        type=float,
        default=2.0,
        help="Keep cluster returns within this depth of cluster median (cm)",
    )
    p.add_argument(
        "--distances-cm",
        nargs="*",
        type=float,
        default=None,
        help="Column distances (default 10..100 checkerboard)",
    )
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    jobs: list[dict] = []

    if args.scenes:
        jobs.append(
            {
                "scenes": args.scenes,
                "target_label": "Target",
                "title": "LiDAR setup grid",
                "out_name": "lidar_polar_custom_setup.png",
            }
        )
    elif args.preset == "checkerboard":
        jobs.append(PRESETS["checkerboard"])
    elif args.preset == "cardboard":
        size = args.size or "L"
        jobs.append(PRESETS[f"cardboard_{size}"])
    elif args.preset == "cardboard_sml":
        jobs.append(PRESETS["cardboard_sml"])
    elif args.preset == "all":
        jobs = [
            PRESETS["checkerboard"],
            PRESETS["cardboard_L"],
            PRESETS["cardboard_M"],
            PRESETS["cardboard_S"],
        ]

    distances_default = list(DEFAULT_DISTANCE_COLUMNS_CM)
    distances_cardboard = list(CARDBOARD_DISTANCE_COLUMNS_CM)

    for job in jobs:
        out_png = out_dir / job["out_name"]
        if job.get("combined"):
            meta = render_cardboard_sml_combined_grid(
                args.data_root,
                out_png,
                title=job["title"],
                distance_columns_cm=args.distances_cm or distances_cardboard,
                distance_tol_cm=args.distance_tol_cm,
                angular_tol_deg=args.angular_tol_deg,
                bearing_mode=args.bearing_mode,
                coplanar_tol_m=args.coplanar_tol_cm / 100.0,
            )
            meta_out = {k: v for k, v in meta.items() if k != "cells"}
            if "cells_serial" in meta:
                meta_out["cells"] = meta["cells_serial"]
        else:
            dists = args.distances_cm or (
                distances_cardboard if job["out_name"].startswith("lidar_polar_") and "box" in job["out_name"]
                else distances_default
            )
            meta = render_setup_grid(
                args.data_root,
                job["scenes"],
                out_png,
                distance_columns_cm=dists,
                target_label=job["target_label"],
                title=job["title"],
                distance_tol_cm=args.distance_tol_cm,
                angular_tol_deg=args.angular_tol_deg,
                bearing_mode=args.bearing_mode,
                coplanar_tol_m=args.coplanar_tol_cm / 100.0,
            )
            cells_ser = {
                f"angle_{k[0]}_dist_{k[1]}": (None if v is None else f"{v['scene']}/pair_{v['pair_id']}")
                for k, v in meta.get("cells", {}).items()
            }
            meta_out = {k: v for k, v in meta.items() if k != "cells"}
            meta_out["cells"] = cells_ser
        json_path = out_png.with_suffix(".json")
        json_path.write_text(json.dumps(meta_out, indent=2, default=str) + "\n")
        print(f"Wrote {out_png}")
        print(f"Wrote {json_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
