"""
LiDAR width via calibration-style segment selection.

Same idea as 03_lidar_camera_calibration/01_calibrate_lidar_to_rgb1_fisheye.py:
  - keep returns near ruler distance (distance window)
  - split into angular clusters along the scan
  - pick the cluster whose chord length best matches expected face width
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.box_volume_heuristic import (
    BOX_SPECS,
    angular_diff_deg,
    box_size_class,
    load_lidar_scan,
    placement_angle_deg,
)

# Checkerboard segment used in LiDAR–camera calibration (250 mm width axis).
CHECKERBOARD_SEGMENT_WIDTH_M = 0.25
CHECKERBOARD_SEGMENT_HEIGHT_M = 0.175


def lidar_xy_from_polar(angle_deg: np.ndarray, dist_m: np.ndarray) -> np.ndarray:
    a = np.deg2rad(angle_deg.astype(float))
    x = dist_m * np.cos(a)
    y = dist_m * np.sin(a)
    return np.column_stack([x, y])


def largest_angle_cluster_mask(
    points_xy: np.ndarray,
    ranges: np.ndarray,
    *,
    window_center_m: float,
    distance_window_m: float,
    expected_length_m: float,
    length_tolerance_m: float,
) -> np.ndarray | None:
    """
    Boolean mask over scan indices for the best-matching angular cluster.

    Ported from lidar_camera_calibration largest_angle_cluster.
    """
    n = len(ranges)
    if n < 3:
        return None
    mask = np.abs(ranges - window_center_m) <= distance_window_m
    if np.count_nonzero(mask) < 4:
        nearest = np.argsort(np.abs(ranges - window_center_m))[: max(4, min(20, n))]
        mask = np.zeros(n, dtype=bool)
        mask[nearest] = True

    global_idx = np.where(mask)[0]
    candidates = points_xy[mask]
    if len(candidates) <= 2:
        out = np.zeros(n, dtype=bool)
        out[global_idx] = True
        return out

    angles = np.unwrap(np.arctan2(candidates[:, 1], candidates[:, 0]))
    order = np.argsort(angles)
    global_idx = global_idx[order]
    candidates = candidates[order]
    angles = angles[order]
    splits = np.where(np.diff(angles) > np.deg2rad(4.0))[0] + 1
    cluster_slices = np.split(np.arange(len(candidates)), splits)

    scored: list[tuple[float, int, np.ndarray]] = []
    for sl in cluster_slices:
        if len(sl) < 3:
            continue
        cluster = candidates[sl]
        length = float(np.linalg.norm(cluster[-1] - cluster[0]))
        length_error = abs(length - expected_length_m)
        gidx = global_idx[sl]
        if length_error <= length_tolerance_m:
            scored.append((length_error, -len(sl), gidx))

    if not scored:
        return None
    chosen = sorted(scored, key=lambda item: (item[0], item[1]))[0][2]
    out = np.zeros(n, dtype=bool)
    out[chosen] = True
    return out


def estimate_bearing_center_deg(
    angle_deg: np.ndarray,
    dist_m: np.ndarray,
    ruler_m: float,
    *,
    distance_window_m: float,
    placement_angle_deg: float | None = None,
    angular_prefilter_deg: float = 45.0,
) -> float:
    """Median bearing of returns near ruler distance (fallback: placement angle)."""
    dist_near = (
        np.isfinite(angle_deg) & np.isfinite(dist_m) & (dist_m > 0) & (np.abs(dist_m - ruler_m) <= distance_window_m)
    )
    if int(np.count_nonzero(dist_near)) >= 4:
        return float(np.median(angle_deg[dist_near]))
    if placement_angle_deg is not None:
        return float(placement_angle_deg)
    return 0.0


def width_from_lidar_segment(
    lidar_csv: Path,
    ruler_m: float,
    expected_width_m: float,
    *,
    distance_window_m: float = 0.06,
    length_tolerance_m: float | None = None,
    bearing_center_deg: float | None = None,
    angular_prefilter_deg: float = 45.0,
    min_points: int = 3,
) -> dict | None:
    """
    Chord length (m) of the calibration-style cluster at ruler distance.
    """
    if ruler_m is None or ruler_m <= 0 or expected_width_m <= 0:
        return None
    angle_all, dist_all = load_lidar_scan(lidar_csv)
    if len(dist_all) < min_points:
        return None

    tol = length_tolerance_m
    if tol is None:
        tol = max(0.04, 0.35 * expected_width_m)

    bearing = bearing_center_deg
    if bearing is None:
        bearing = estimate_bearing_center_deg(
            angle_all, dist_all, ruler_m, distance_window_m=distance_window_m
        )
    in_range = (
        np.isfinite(angle_all)
        & np.isfinite(dist_all)
        & (dist_all > 0)
        & (np.abs(dist_all - ruler_m) <= distance_window_m)
        & (np.abs(angular_diff_deg(angle_all, bearing)) <= angular_prefilter_deg)
    )
    if int(np.count_nonzero(in_range)) < min_points:
        return None

    idx = np.where(in_range)[0]
    angle_deg = angle_all[in_range]
    dist_m = dist_all[in_range]
    points_xy = lidar_xy_from_polar(angle_deg, dist_m)
    sel = largest_angle_cluster_mask(
        points_xy,
        dist_m,
        window_center_m=float(ruler_m),
        distance_window_m=distance_window_m,
        expected_length_m=float(expected_width_m),
        length_tolerance_m=float(tol),
    )
    if sel is None or int(np.count_nonzero(sel)) < min_points:
        return None

    cluster_xy = points_xy[sel]
    width_m = float(np.linalg.norm(cluster_xy[-1] - cluster_xy[0]))
    ang = angle_deg[sel]
    full_mask = np.zeros(len(angle_all), dtype=bool)
    full_mask[idx[sel]] = True
    return {
        "width_m": width_m,
        "width_cm": width_m * 100.0,
        "expected_width_m": float(expected_width_m),
        "expected_width_cm": float(expected_width_m) * 100.0,
        "length_tolerance_m": float(tol),
        "distance_window_m": float(distance_window_m),
        "lidar_n": int(np.count_nonzero(sel)),
        "lidar_range_m": float(np.median(dist_m[sel])),
        "bearing_center_deg": float(bearing),
        "cluster_mask": full_mask,
        "width_source": "calibrated_segment_chord",
    }


def expected_box_face_width_m(scene: str) -> float | None:
    size = box_size_class(scene)
    if size is None:
        return None
    view_deg = placement_angle_deg(scene)
    w_nominal_m = BOX_SPECS[size]["width_nominal_cm"] / 100.0
    return float(w_nominal_m * np.cos(np.deg2rad(view_deg)))


def width_from_lidar_box_segment(
    lidar_csv: Path,
    ruler_m: float,
    scene: str,
    **kwargs,
) -> dict | None:
    expected = expected_box_face_width_m(scene)
    if expected is None:
        return None
    place = placement_angle_deg(scene)
    angle_deg, dist_m = load_lidar_scan(lidar_csv)
    dist_win = float(kwargs.get("distance_window_m", 0.06))
    bearing = estimate_bearing_center_deg(
        angle_deg,
        dist_m,
        ruler_m,
        distance_window_m=dist_win,
        placement_angle_deg=place,
    )
    out = width_from_lidar_segment(
        lidar_csv,
        ruler_m,
        expected,
        bearing_center_deg=bearing,
        **kwargs,
    )
    if out is not None:
        out["scene"] = scene
        out["size_class"] = box_size_class(scene)
        out["placement_angle_deg"] = place
    return out
