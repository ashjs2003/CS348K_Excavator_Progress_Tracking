"""
Validate the stereo point cloud against the 2D LiDAR scan.

Run:
    python 03_validate_with_lidar.py
    python 03_validate_with_lidar.py --run 20260521_143022_carpet
"""

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

from calib_utils import (
    load_camera_calibration,
    load_lidar_to_rgb1,
    lidar_polar_to_xyz,
    read_lidar_csv,
    stereo_rectify_R1_rgb1,
    transform_points,
)
from pointcloud_utils import nearest_neighbor_distances, read_ply, write_ply

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths, write_run_info


def parse_args():
    parser = argparse.ArgumentParser()
    add_run_cli_arguments(parser)
    parser.add_argument(
        "--stereo-suffix",
        default="",
        help="Filename suffix for stereo PLY, e.g. _foundation → stereo_pointcloud_downsampled_foundation.ply",
    )
    parser.add_argument(
        "--metrics-suffix",
        default="",
        help="Suffix for metrics JSON, e.g. _foundation → lidar_stereo_error_metrics_foundation.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    if handle_list_runs(args):
        return
    paths = resolve_run_paths(args.run)
    lidar_csv = paths.lidar_csv
    stereo_cloud = paths.stereo / f"stereo_pointcloud_downsampled{args.stereo_suffix}.ply"
    validation_dir = paths.validation
    validation_dir.mkdir(parents=True, exist_ok=True)
    lidar_cloud_out = validation_dir / "lidar_points_in_rgb1_frame.ply"
    metrics_out = validation_dir / f"lidar_stereo_error_metrics{args.metrics_suffix}.json"
    per_point_out = validation_dir / f"lidar_stereo_error_per_point{args.metrics_suffix}.csv"

    if paths.run_dir:
        print(f"Run: {paths.run_dir.name}")
    print(f"Writing validation outputs to {validation_dir}")

    lidar_path, R_lidar_to_rgb1, t_lidar_to_rgb1 = load_lidar_to_rgb1()
    print(f"Loaded LiDAR-to-RGB1 extrinsics: {lidar_path}")

    rgb1_calib = load_camera_calibration("camera_calibration_rgb1.npz")
    rgb2_calib = load_camera_calibration("camera_calibration_rgb2.npz", "camera_calibration_rgb2_approx.npz")
    image1 = cv2.imread(str(paths.rgb1_image))
    if image1 is None:
        raise RuntimeError(f"Could not load {paths.rgb1_image}")
    image_size = (image1.shape[1], image1.shape[0])
    R1 = stereo_rectify_R1_rgb1(rgb1_calib, rgb2_calib, image_size)

    lidar_scan = read_lidar_csv(lidar_csv)
    points_lidar = lidar_polar_to_xyz(lidar_scan)
    points_rgb1 = transform_points(points_lidar, R_lidar_to_rgb1, t_lidar_to_rgb1)
    # Stereo cloud from step 02 is in rectified RGB1 frame; rotate LiDAR to match.
    points_rgb1_rect = transform_points(points_rgb1, R1, np.zeros(3))
    colors = np.tile(np.array([[1.0, 0.05, 0.05]]), (len(points_rgb1_rect), 1))
    write_ply(lidar_cloud_out, points_rgb1_rect, colors)

    stereo_points, _ = read_ply(stereo_cloud)
    if len(stereo_points) == 0:
        raise RuntimeError(
            f"No points in {stereo_cloud}. Re-run 02_make_stereo_pointcloud.py and check "
            f"{paths.stereo / 'disparity_preview.png'} and {paths.stereo / 'rectification_check.png'}."
        )

    distances, nearest_indices = nearest_neighbor_distances(points_rgb1_rect, stereo_points)
    rows = []
    for index, (point, nearest_index, error) in enumerate(zip(points_rgb1_rect, nearest_indices, distances)):
        nearest = stereo_points[nearest_index]
        rows.append([
            index,
            point[0],
            point[1],
            point[2],
            nearest[0],
            nearest[1],
            nearest[2],
            error,
        ])

    errors = np.asarray(distances, dtype=float)
    if len(errors) == 0:
        raise RuntimeError("No LiDAR points could be compared to the stereo cloud.")

    metrics = {
        "valid_lidar_points": int(len(points_rgb1_rect)),
        "stereo_point_count": int(len(stereo_points)),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "90th_percentile": float(np.percentile(errors, 90)),
        "max_error": float(np.max(errors)),
    }

    with open(metrics_out, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(per_point_out, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "lidar_rgb1_x_m",
            "lidar_rgb1_y_m",
            "lidar_rgb1_z_m",
            "nearest_stereo_x_m",
            "nearest_stereo_y_m",
            "nearest_stereo_z_m",
            "nearest_error_m",
        ])
        writer.writerows(rows)

    if paths.run_dir:
        info_key = "lidar_validation" if not args.metrics_suffix else f"lidar_validation{args.metrics_suffix}"
        write_run_info(paths.run_dir, **{info_key: metrics})

    print(f"Saved {lidar_cloud_out}")
    print(f"Saved {metrics_out}")
    print(f"Saved {per_point_out}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
