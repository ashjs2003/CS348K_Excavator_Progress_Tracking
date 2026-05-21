"""Small shared helpers for the stereo RGB + 2D LiDAR point-cloud workflow."""

from pathlib import Path

import cv2
import numpy as np


# Scripts are intended to be run from this folder. These helpers also look one
# level up and in ../config so the workflow stays easy to use on Windows.
BASE_DIR = Path(__file__).resolve().parent
SEARCH_DIRS = [BASE_DIR, BASE_DIR.parent, BASE_DIR.parent / "config", BASE_DIR.parent / "stereo_calibration"]


def find_file(*names):
    """Return the first existing path matching any supplied filename."""
    for name in names:
        candidate = Path(name)
        if candidate.exists():
            return candidate
        for directory in SEARCH_DIRS:
            path = directory / name
            if path.exists():
                return path
    raise FileNotFoundError(f"Could not find any of: {names}")


def load_camera_calibration(*names):
    """Load OpenCV camera intrinsics from an npz file."""
    path = find_file(*names)
    data = np.load(path)
    return {
        "path": path,
        "K": data["camera_matrix"].astype(np.float64),
        "dist": data["dist_coeffs"].astype(np.float64),
        "image_size": tuple(data["image_size"].astype(int)),
    }


def load_lidar_to_rgb1():
    """Load LiDAR-to-RGB1 transform, accepting a few common key names."""
    path = find_file("lidar_to_rgb1_extrinsics.npz", "lidar_to_camera_extrinsics.npz")
    data = np.load(path)

    if "R_lidar_to_rgb1" in data:
        R = data["R_lidar_to_rgb1"]
    elif "R_lidar_to_camera" in data:
        R = data["R_lidar_to_camera"]
    elif "R" in data:
        R = data["R"]
    else:
        raise KeyError(f"{path} must contain R_lidar_to_rgb1, R_lidar_to_camera, or R")

    if "t_lidar_to_rgb1" in data:
        t = data["t_lidar_to_rgb1"]
    elif "t_lidar_to_camera" in data:
        t = data["t_lidar_to_camera"]
    elif "translation_meters" in data:
        t = data["translation_meters"]
    elif "t" in data:
        t = data["t"]
    else:
        raise KeyError(f"{path} must contain t_lidar_to_rgb1, t_lidar_to_camera, translation_meters, or t")

    return path, R.astype(np.float64), t.reshape(3).astype(np.float64)


def load_stereo_rgb1_to_rgb2():
    """Load RGB1-to-RGB2 stereo extrinsics."""
    path = find_file("stereo_rgb1_rgb2_extrinsics.npz")
    data = np.load(path)
    return (
        path,
        data["R_rgb1_to_rgb2"].astype(np.float64),
        data["t_rgb1_to_rgb2"].reshape(3).astype(np.float64),
    )


def open_camera(index, image_size):
    """Open a Windows webcam and request the calibrated resolution."""
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(image_size[0]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(image_size[1]))
    return cap


def read_lidar_csv(path):
    """Read angle/distance/quality LiDAR CSV, tolerating a few header variants."""
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data])
    names = set(data.dtype.names or [])

    angle_key = next((key for key in ("angle_degrees", "angle_deg", "angle") if key in names), None)
    if angle_key is None:
        raise KeyError(f"{path} needs angle_degrees, angle_deg, or angle")

    if "distance_meters" in names:
        distances = data["distance_meters"].astype(float)
    elif "distance_m" in names:
        distances = data["distance_m"].astype(float)
    elif "distance_mm" in names:
        distances = data["distance_mm"].astype(float) / 1000.0
    elif "distance" in names:
        raw = data["distance"].astype(float)
        distances = raw / 1000.0 if np.nanmedian(raw) > 20.0 else raw
    else:
        raise KeyError(f"{path} needs distance_meters, distance_m, distance_mm, or distance")

    if "quality" in names:
        quality = data["quality"].astype(float)
    else:
        quality = np.ones_like(distances)

    return np.column_stack([data[angle_key].astype(float), distances, quality])


def lidar_polar_to_xyz(lidar_scan):
    """Convert angle_degrees, distance_meters rows into LiDAR-frame xyz points."""
    valid = np.isfinite(lidar_scan[:, 0]) & np.isfinite(lidar_scan[:, 1]) & (lidar_scan[:, 1] > 0)
    lidar_scan = lidar_scan[valid]
    theta = np.deg2rad(lidar_scan[:, 0])
    radius = lidar_scan[:, 1]
    return np.column_stack([radius * np.cos(theta), radius * np.sin(theta), np.zeros_like(radius)])


def transform_points(points, R, t):
    """Apply p_out = R p_in + t to an Nx3 point array."""
    if len(points) == 0:
        return points.copy()
    return (R @ points.T).T + t.reshape(1, 3)


def project_camera_points(points_camera, K, dist, image_shape):
    """Project camera-frame 3D points to image pixels."""
    if len(points_camera) == 0:
        return np.empty((0, 2)), np.empty(0)

    valid_z = points_camera[:, 2] > 0.05
    points = points_camera[valid_z]
    if len(points) == 0:
        return np.empty((0, 2)), np.empty(0)

    uv, _ = cv2.projectPoints(points.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K, dist)
    uv = uv.reshape(-1, 2)
    h, w = image_shape[:2]
    in_image = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[in_image], points[in_image, 2]
