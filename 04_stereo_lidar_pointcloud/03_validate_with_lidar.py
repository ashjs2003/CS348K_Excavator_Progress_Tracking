"""
Validate depth / point cloud against 2D LiDAR (ray depth + optional NN cloud).

Prefer running the full scorecard:
    python 06_evaluate_run.py --run latest

This script evaluates one method (same metrics as 06, subset):
    python 03_validate_with_lidar.py --run latest
    python 03_validate_with_lidar.py --run latest --stereo-suffix _dav2 --metrics-suffix _dav2
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths, write_run_info
from evaluation.depth_maps import METHODS, load_metric_depth, load_or_compute_stereo_geometry
from evaluation.lidar_ray_metrics import (
    compute_lidar_ray_metrics,
    compute_nn_cloud_metrics,
    lidar_points_in_rectified_frame,
    write_ray_per_point_csv,
)
from calib_utils import project_rectified_points
from pointcloud_utils import write_ply

# Map CLI suffix to method key
SUFFIX_TO_METHOD = {v["suffix"]: k for k, v in METHODS.items()}


def parse_args():
    parser = argparse.ArgumentParser()
    add_run_cli_arguments(parser)
    parser.add_argument(
        "--stereo-suffix",
        default="",
        help="PLY suffix, e.g. _foundation → stereo_pointcloud_downsampled_foundation.ply",
    )
    parser.add_argument(
        "--metrics-suffix",
        default="",
        help="JSON suffix, e.g. _foundation → lidar_ray_depth_metrics_foundation.json",
    )
    parser.add_argument("--ray-inlier-tau", type=float, default=0.05)
    parser.add_argument("--free-space-tau", type=float, default=0.03)
    return parser.parse_args()


def main():
    args = parse_args()
    if handle_list_runs(args):
        return
    paths = resolve_run_paths(args.run)
    validation_dir = paths.validation
    validation_dir.mkdir(parents=True, exist_ok=True)

    suffix = args.metrics_suffix if args.metrics_suffix is not None else args.stereo_suffix
    method = SUFFIX_TO_METHOD.get(suffix, "opencv")

    if paths.run_dir:
        print(f"Run: {paths.run_dir.name}  method: {method}")
    print(f"Writing validation outputs to {validation_dir}")

    geometry = load_or_compute_stereo_geometry(paths.stereo, paths.rgb1_image, paths.rgb2_image)
    points_rect, lidar_path = lidar_points_in_rectified_frame(
        paths.lidar_csv, geometry["image_size"], geometry
    )
    print(f"Loaded LiDAR-to-RGB1 extrinsics: {lidar_path}")

    depth_m = load_metric_depth(paths.stereo, method, geometry)
    if depth_m is None:
        raise RuntimeError(
            f"No depth map for method {method!r}. Run the matching 02_make_* script first."
        )

    write_ply(validation_dir / "lidar_points_in_rgb1_frame.ply", points_rect,
              np.tile(np.array([[1.0, 0.05, 0.05]]), (len(points_rect), 1)))

    ray = compute_lidar_ray_metrics(
        depth_m,
        points_rect,
        geometry["P1"],
        ray_inlier_tau=args.ray_inlier_tau,
        free_space_tau=args.free_space_tau,
    )

    uv, z_lidar, proj_valid = project_rectified_points(geometry["P1"], points_rect)
    z_est = np.full(len(points_rect), np.nan)
    h, w = depth_m.shape
    for i in np.flatnonzero(proj_valid):
        col, row = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if 0 <= col < w and 0 <= row < h:
            z = depth_m[row, col]
            if np.isfinite(z) and z > 0:
                z_est[i] = z
    write_ray_per_point_csv(
        validation_dir / f"lidar_ray_per_point{suffix}.csv",
        points_rect, uv, z_lidar, z_est, proj_valid, np.isfinite(z_est),
    )

    ray_path = validation_dir / f"lidar_ray_depth_metrics{suffix}.json"
    ray_path.write_text(json.dumps(ray, indent=2) + "\n")

    stereo_cloud = paths.stereo / f"stereo_pointcloud_downsampled{args.stereo_suffix}.ply"
    nn = compute_nn_cloud_metrics(points_rect, stereo_cloud)
    if nn is not None:
        nn["valid_lidar_points"] = int(len(points_rect))
        legacy_path = validation_dir / f"lidar_stereo_error_metrics{suffix}.json"
        legacy_path.write_text(json.dumps(nn, indent=2) + "\n")
        if paths.run_dir:
            key = "lidar_validation" if not suffix else f"lidar_validation{suffix}"
            write_run_info(paths.run_dir, **{key: nn})

    print(f"Saved {ray_path}")
    if nn is not None:
        print(f"Saved {legacy_path} (NN cloud)")
    print(json.dumps(ray, indent=2))


if __name__ == "__main__":
    main()
