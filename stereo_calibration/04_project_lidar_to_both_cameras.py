"""
Project one LiDAR scan onto synchronized RGB1 and RGB2 images.

RGB2 currently uses copied RGB1 intrinsics for this quick prototype. That is
convenient, but approximate: same-model cameras can still have different
intrinsics and distortion. Calibrate RGB2 separately later for better accuracy.

Run examples:
    python 04_project_lidar_to_both_cameras.py
    python 04_project_lidar_to_both_cameras.py --pair-id 000 --lidar lidar_scan.npy
    python 04_project_lidar_to_both_cameras.py --rgb1 stereo_pairs/rgb1_000.png --rgb2 stereo_pairs/rgb2_000.png --lidar lidar_scan.csv
"""

from pathlib import Path
import argparse

import cv2
import numpy as np


RGB1_CALIBRATION_FILE = Path("../config/camera_calibration_rgb1.npz")
RGB2_CALIBRATION_FILE = Path("../config/camera_calibration_rgb2_approx.npz")
LIDAR_TO_RGB1_FILE = Path("../config/lidar_to_camera_extrinsics.npz")
STEREO_EXTRINSICS_FILE = Path("stereo_rgb1_rgb2_extrinsics.npz")
STEREO_PAIR_DIR = Path("stereo_pairs")
DEFAULT_LIDAR_SCAN = Path("lidar_scan.npy")
OUT_IMAGE = Path("lidar_projected_to_both_cameras.png")
WINDOW = "LiDAR projected to RGB1 and RGB2"
DOT_SIZE = 3


def load_camera_calibration(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"]


def load_lidar_to_rgb1(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    data = np.load(path)

    if "R_lidar_to_rgb1" in data:
        R = data["R_lidar_to_rgb1"]
    elif "R_lidar_to_camera" in data:
        R = data["R_lidar_to_camera"]
    elif "R" in data:
        R = data["R"]
    else:
        raise KeyError(f"{path} must contain R_lidar_to_rgb1 or R")

    if "t_lidar_to_rgb1" in data:
        t = data["t_lidar_to_rgb1"]
    elif "t_lidar_to_camera" in data:
        t = data["t_lidar_to_camera"]
    elif "t" in data:
        t = data["t"]
    elif "translation_meters" in data:
        t = data["translation_meters"]
    else:
        raise KeyError(f"{path} must contain t_lidar_to_rgb1, t, or translation_meters")

    return R.astype(float), t.reshape(3).astype(float)


def load_stereo_extrinsics(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    data = np.load(path)
    return data["R_rgb1_to_rgb2"].astype(float), data["t_rgb1_to_rgb2"].reshape(3).astype(float)


def distance_to_meters(distances, already_meters=False):
    distances = np.asarray(distances, dtype=float)
    if already_meters:
        return distances
    finite = distances[np.isfinite(distances) & (distances > 0)]
    if len(finite) and np.nanmedian(finite) > 20.0:
        return distances / 1000.0
    return distances


def load_lidar_scan(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing LiDAR scan: {path}")

    if path.suffix.lower() == ".npy":
        data = np.load(path)
        if data.ndim != 2 or data.shape[1] < 2:
            raise ValueError(f"{path} must be an Nx2 or Nx3 array: angle_deg, distance, optional quality")
        angles = data[:, 0]
        distances_m = distance_to_meters(data[:, 1])
    else:
        data = np.genfromtxt(path, delimiter=",", names=True)
        if data.ndim == 0:
            data = np.array([data])

        names = set(data.dtype.names or [])
        angle_key = next((key for key in ["angle_degrees", "angle_deg", "angle"] if key in names), None)
        if angle_key is None:
            raise KeyError(f"{path} must contain angle_degrees, angle_deg, or angle")

        if "distance_meters" in names:
            distance_key = "distance_meters"
            already_meters = True
        elif "distance_m" in names:
            distance_key = "distance_m"
            already_meters = True
        elif "distance_mm" in names:
            distance_key = "distance_mm"
            already_meters = False
        elif "distance" in names:
            distance_key = "distance"
            already_meters = False
        else:
            raise KeyError(f"{path} must contain a distance column")

        angles = data[angle_key]
        distances_m = distance_to_meters(data[distance_key], already_meters=already_meters)

    valid = np.isfinite(angles) & np.isfinite(distances_m) & (distances_m > 0)
    angles = angles[valid]
    distances_m = distances_m[valid]
    theta = np.deg2rad(angles)
    return np.column_stack([
        distances_m * np.cos(theta),
        distances_m * np.sin(theta),
        np.zeros_like(distances_m),
    ])


def default_pair_paths(pair_id):
    if pair_id is None:
        rgb1_paths = sorted(STEREO_PAIR_DIR.glob("rgb1_*.png"))
        if not rgb1_paths:
            raise FileNotFoundError("No stereo pairs found in stereo_pairs/")
        pair_id = rgb1_paths[0].stem.split("_")[1]

    return (
        STEREO_PAIR_DIR / f"rgb1_{pair_id}.png",
        STEREO_PAIR_DIR / f"rgb2_{pair_id}.png",
        pair_id,
    )


def project_camera_points(points_camera, K, dist, image_shape):
    valid_z = points_camera[:, 2] > 0.05
    valid_points = points_camera[valid_z]
    if len(valid_points) == 0:
        return np.empty((0, 2)), np.empty(0)

    uv, _ = cv2.projectPoints(
        valid_points.reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        K,
        dist,
    )
    uv = uv.reshape(-1, 2)
    h, w = image_shape[:2]
    in_image = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[in_image], valid_points[in_image, 2]


def draw_overlay(image, points_camera, K, dist, label):
    vis = image.copy()
    uv, depth = project_camera_points(points_camera, K, dist, image.shape)

    if len(depth):
        max_depth = max(0.1, float(np.percentile(depth, 95)))
        colors = cv2.applyColorMap(
            np.clip(depth * 255 / max_depth, 0, 255).astype(np.uint8).reshape(-1, 1),
            cv2.COLORMAP_TURBO,
        )
        for point, color in zip(uv, colors[:, 0, :]):
            cv2.circle(vis, tuple(np.round(point).astype(int)), DOT_SIZE, color.tolist(), -1)

    text = f"{label}: projected {len(uv)} LiDAR points"
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def side_by_side(left, right):
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / right.shape[0]
        right = cv2.resize(right, (int(right.shape[1] * scale), left.shape[0]))
    return np.hstack([left, right])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-id", default=None, help="Stereo pair id, for example 000")
    parser.add_argument("--rgb1", type=Path, default=None, help="Path to synchronized RGB1 image")
    parser.add_argument("--rgb2", type=Path, default=None, help="Path to synchronized RGB2 image")
    parser.add_argument("--lidar", type=Path, default=DEFAULT_LIDAR_SCAN, help="Path to LiDAR scan .npy or .csv")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.rgb1 is None or args.rgb2 is None:
        rgb1_path, rgb2_path, pair_id = default_pair_paths(args.pair_id)
    else:
        rgb1_path = args.rgb1
        rgb2_path = args.rgb2
        pair_id = args.pair_id or "custom"

    K1, dist1 = load_camera_calibration(RGB1_CALIBRATION_FILE)
    K2, dist2 = load_camera_calibration(RGB2_CALIBRATION_FILE)
    R_lidar_to_rgb1, t_lidar_to_rgb1 = load_lidar_to_rgb1(LIDAR_TO_RGB1_FILE)
    R_rgb1_to_rgb2, t_rgb1_to_rgb2 = load_stereo_extrinsics(STEREO_EXTRINSICS_FILE)
    points_lidar = load_lidar_scan(args.lidar)

    image1 = cv2.imread(str(rgb1_path))
    image2 = cv2.imread(str(rgb2_path))
    if image1 is None:
        raise RuntimeError(f"Could not load RGB1 image: {rgb1_path}")
    if image2 is None:
        raise RuntimeError(f"Could not load RGB2 image: {rgb2_path}")

    points_rgb1 = (R_lidar_to_rgb1 @ points_lidar.T).T + t_lidar_to_rgb1
    points_rgb2 = (R_rgb1_to_rgb2 @ points_rgb1.T).T + t_rgb1_to_rgb2

    overlay1 = draw_overlay(image1, points_rgb1, K1, dist1, f"RGB1 pair {pair_id}")
    overlay2 = draw_overlay(image2, points_rgb2, K2, dist2, f"RGB2 pair {pair_id}")
    combined = side_by_side(overlay1, overlay2)

    cv2.imwrite(str(OUT_IMAGE), combined)
    print(f"Loaded LiDAR scan: {args.lidar} ({len(points_lidar)} points)")
    print(f"Loaded RGB1 image: {rgb1_path}")
    print(f"Loaded RGB2 image: {rgb2_path}")
    print(f"Saved overlay: {OUT_IMAGE.resolve()}")
    print("Press any key in the overlay window to close.")

    cv2.imshow(WINDOW, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
