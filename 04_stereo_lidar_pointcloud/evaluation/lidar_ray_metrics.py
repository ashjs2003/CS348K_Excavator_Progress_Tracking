"""LiDAR ray-depth error and free-space violation metrics (rectified left frame)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from calib_utils import (
    load_camera_calibration,
    load_lidar_to_rgb1,
    lidar_polar_to_xyz,
    project_rectified_points,
    read_lidar_csv,
    stereo_rectify_R1_rgb1,
    transform_points,
)
from pointcloud_utils import nearest_neighbor_distances, read_ply


def _sample_depth_at_pixels(depth_m: np.ndarray, uv: np.ndarray, valid_mask: np.ndarray):
    """Bilinear-ish nearest-pixel sample; returns z_est and sampled_valid."""
    h, w = depth_m.shape
    n = len(uv)
    z_est = np.full(n, np.nan, dtype=np.float64)
    sampled_valid = np.zeros(n, dtype=bool)

    for i in np.flatnonzero(valid_mask):
        u, v = uv[i]
        col = int(round(u))
        row = int(round(v))
        if col < 0 or col >= w or row < 0 or row >= h:
            continue
        z = depth_m[row, col]
        if np.isfinite(z) and z > 0:
            z_est[i] = float(z)
            sampled_valid[i] = True
    return z_est, sampled_valid


def lidar_points_in_rectified_frame(lidar_csv: Path, image_size, geometry: dict):
    """LiDAR scan as Nx3 points in rectified RGB1 coordinates."""
    lidar_path, R_lidar_to_rgb1, t_lidar_to_rgb1 = load_lidar_to_rgb1()
    rgb1_calib = load_camera_calibration("camera_calibration_rgb1.npz")
    rgb2_calib = load_camera_calibration("camera_calibration_rgb2.npz", "camera_calibration_rgb2_approx.npz")
    R1 = geometry["R1"] if "R1" in geometry else stereo_rectify_R1_rgb1(rgb1_calib, rgb2_calib, image_size)

    lidar_scan = read_lidar_csv(lidar_csv)
    points_lidar = lidar_polar_to_xyz(lidar_scan)
    points_rgb1 = transform_points(points_lidar, R_lidar_to_rgb1, t_lidar_to_rgb1)
    points_rect = transform_points(points_rgb1, R1, np.zeros(3))
    return points_rect, lidar_path


def compute_lidar_ray_metrics(
    depth_m: np.ndarray,
    points_rect: np.ndarray,
    P1: np.ndarray,
    ray_inlier_tau: float = 0.05,
    free_space_tau: float = 0.03,
) -> dict:
    """
    Compare metric depth map to LiDAR hits in the rectified left frame.

    Free-space violation: Z_est < Z_lidar - free_space_tau (depth too close vs LiDAR).
    """
    uv, z_lidar, proj_valid = project_rectified_points(P1, points_rect)
    z_est, sampled_valid = _sample_depth_at_pixels(depth_m, uv, proj_valid)
    compare = proj_valid & sampled_valid
    n_compare = int(np.count_nonzero(compare))

    if n_compare == 0:
        return {
            "valid_lidar_points": int(len(points_rect)),
            "projected_in_image": int(np.count_nonzero(proj_valid)),
            "associated_pixels": 0,
            "association_rate": 0.0,
            "ray_median_error_m": None,
            "ray_mean_error_m": None,
            "ray_rmse_m": None,
            "ray_p90_m": None,
            "ray_max_error_m": None,
            "inlier_ratio": 0.0,
            "ray_inlier_tau_m": float(ray_inlier_tau),
            "free_space_violation_pct": 0.0,
            "median_free_space_violation_m": None,
            "free_space_tau_m": float(free_space_tau),
            "error_vs_range": {"bin_edges_m": [], "bins": [], "n_points": 0},
        }

    errors = np.abs(z_est[compare] - z_lidar[compare])
    inliers = errors < ray_inlier_tau
    violations = (z_est[compare] < z_lidar[compare] - free_space_tau)
    viol_vals = z_lidar[compare][violations] - z_est[compare][violations]

    from evaluation.error_vs_range import compute_error_vs_range

    error_vs_range = compute_error_vs_range(z_lidar, z_est, compare)

    return {
        "valid_lidar_points": int(len(points_rect)),
        "projected_in_image": int(np.count_nonzero(proj_valid)),
        "associated_pixels": n_compare,
        "association_rate": float(n_compare / max(1, np.count_nonzero(proj_valid))),
        "ray_median_error_m": float(np.median(errors)),
        "ray_mean_error_m": float(np.mean(errors)),
        "ray_rmse_m": float(np.sqrt(np.mean(errors**2))),
        "ray_p90_m": float(np.percentile(errors, 90)),
        "ray_max_error_m": float(np.max(errors)),
        "inlier_ratio": float(np.count_nonzero(inliers) / n_compare),
        "ray_inlier_tau_m": float(ray_inlier_tau),
        "free_space_violation_pct": float(100.0 * np.count_nonzero(violations) / n_compare),
        "median_free_space_violation_m": float(np.median(viol_vals)) if len(viol_vals) else None,
        "free_space_tau_m": float(free_space_tau),
        "error_vs_range": error_vs_range,
    }


def compute_nn_cloud_metrics(points_rect: np.ndarray, stereo_cloud_path: Path) -> dict | None:
    """Legacy nearest-neighbor cloud comparison (optional)."""
    if not stereo_cloud_path.is_file():
        return None
    stereo_points, _ = read_ply(stereo_cloud_path)
    if len(stereo_points) == 0:
        return None
    distances, _ = nearest_neighbor_distances(points_rect, stereo_points)
    errors = np.asarray(distances, dtype=float)
    return {
        "stereo_point_count": int(len(stereo_points)),
        "mean_error": float(np.mean(errors)),
        "median_error": float(np.median(errors)),
        "rmse": float(np.sqrt(np.mean(errors**2))),
        "90th_percentile": float(np.percentile(errors, 90)),
        "max_error": float(np.max(errors)),
    }


def write_ray_per_point_csv(path: Path, points_rect, uv, z_lidar, z_est, proj_valid, sampled_valid):
    path = Path(path)
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "index",
            "lidar_x_m",
            "lidar_y_m",
            "lidar_z_m",
            "pixel_u",
            "pixel_v",
            "z_lidar_m",
            "z_est_m",
            "ray_error_m",
            "free_space_violation",
        ])
        for i in range(len(points_rect)):
            err = ""
            fs = ""
            if proj_valid[i] and sampled_valid[i]:
                err = abs(z_est[i] - z_lidar[i])
                fs = int(z_est[i] < z_lidar[i])
            writer.writerow([
                i,
                points_rect[i, 0],
                points_rect[i, 1],
                points_rect[i, 2],
                uv[i, 0] if proj_valid[i] else "",
                uv[i, 1] if proj_valid[i] else "",
                z_lidar[i],
                z_est[i] if sampled_valid[i] else "",
                err,
                fs,
            ])
