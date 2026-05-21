"""
Validate the stereo point cloud against the 2D LiDAR scan.

Run:
    python 03_validate_with_lidar.py
"""

from pathlib import Path
import csv
import json

import numpy as np

from calib_utils import load_lidar_to_rgb1, read_lidar_csv, lidar_polar_to_xyz, transform_points
from pointcloud_utils import nearest_neighbor_distances, read_ply, write_ply


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = REPO_ROOT / "capture"
OUT_DIR = REPO_ROOT / "outputs"
LIDAR_CSV = CAPTURE_DIR / "lidar_scan.csv"
STEREO_CLOUD = OUT_DIR / "stereo_pointcloud_downsampled.ply"
LIDAR_CLOUD_OUT = OUT_DIR / "lidar_points_in_rgb1_frame.ply"
METRICS_OUT = OUT_DIR / "lidar_stereo_error_metrics.json"
PER_POINT_OUT = OUT_DIR / "lidar_stereo_error_per_point.csv"


def save_lidar_cloud(points_rgb1):
    colors = np.tile(np.array([[1.0, 0.05, 0.05]]), (len(points_rgb1), 1))
    write_ply(LIDAR_CLOUD_OUT, points_rgb1, colors)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    lidar_path, R_lidar_to_rgb1, t_lidar_to_rgb1 = load_lidar_to_rgb1()
    print(f"Loaded LiDAR-to-RGB1 extrinsics: {lidar_path}")

    lidar_scan = read_lidar_csv(LIDAR_CSV)
    points_lidar = lidar_polar_to_xyz(lidar_scan)
    points_rgb1 = transform_points(points_lidar, R_lidar_to_rgb1, t_lidar_to_rgb1)
    save_lidar_cloud(points_rgb1)

    stereo_points, _ = read_ply(STEREO_CLOUD)
    if len(stereo_points) == 0:
        raise RuntimeError(
            f"No points in {STEREO_CLOUD}. Re-run 02_make_stereo_pointcloud.py and check "
            "outputs/disparity_preview.png plus outputs/rectification_check.png."
        )

    # Use scipy.spatial.cKDTree when installed, with a local brute-force fallback.
    distances, nearest_indices = nearest_neighbor_distances(points_rgb1, stereo_points)
    rows = []
    for index, (point, nearest_index, error) in enumerate(zip(points_rgb1, nearest_indices, distances)):
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
        "valid_lidar_points": int(len(points_rgb1)),
        "stereo_point_count": int(len(stereo_points)),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "90th_percentile": float(np.percentile(errors, 90)),
        "max_error": float(np.max(errors)),
    }

    with open(METRICS_OUT, "w") as f:
        json.dump(metrics, f, indent=2)

    with open(PER_POINT_OUT, "w", newline="") as f:
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

    print(f"Saved {LIDAR_CLOUD_OUT}")
    print(f"Saved {METRICS_OUT}")
    print(f"Saved {PER_POINT_OUT}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
