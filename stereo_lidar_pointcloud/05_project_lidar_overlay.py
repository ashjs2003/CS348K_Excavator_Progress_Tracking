"""
Project LiDAR points onto RGB1 and RGB2 capture images.

Run:
    python 05_project_lidar_overlay.py
"""

from pathlib import Path

import cv2
import numpy as np

from calib_utils import (
    load_camera_calibration,
    load_lidar_to_rgb1,
    load_stereo_rgb1_to_rgb2,
    read_lidar_csv,
    lidar_polar_to_xyz,
    transform_points,
    project_camera_points,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = REPO_ROOT / "capture"
OUT_DIR = REPO_ROOT / "outputs"
RGB1_IMAGE = CAPTURE_DIR / "rgb1.png"
RGB2_IMAGE = CAPTURE_DIR / "rgb2.png"
LIDAR_CSV = CAPTURE_DIR / "lidar_scan.csv"
RGB1_OUT = OUT_DIR / "lidar_overlay_rgb1.png"
RGB2_OUT = OUT_DIR / "lidar_overlay_rgb2.png"
WINDOW = "LiDAR overlays on RGB1 and RGB2"


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
            cv2.circle(vis, tuple(np.round(point).astype(int)), 4, color.tolist(), -1)

    text = f"{label}: projected {len(uv)} LiDAR points"
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def side_by_side(left, right):
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / right.shape[0]
        right = cv2.resize(right, (int(right.shape[1] * scale), left.shape[0]))
    return np.hstack([left, right])


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rgb1_calib = load_camera_calibration("camera_calibration_rgb1.npz")
    rgb2_calib = load_camera_calibration("camera_calibration_rgb2.npz", "camera_calibration_rgb2_approx.npz")
    _, R_lidar_to_rgb1, t_lidar_to_rgb1 = load_lidar_to_rgb1()
    _, R_rgb1_to_rgb2, t_rgb1_to_rgb2 = load_stereo_rgb1_to_rgb2()

    image1 = cv2.imread(str(RGB1_IMAGE))
    image2 = cv2.imread(str(RGB2_IMAGE))
    if image1 is None:
        raise RuntimeError(f"Could not load {RGB1_IMAGE}")
    if image2 is None:
        raise RuntimeError(f"Could not load {RGB2_IMAGE}")

    lidar_scan = read_lidar_csv(LIDAR_CSV)
    points_lidar = lidar_polar_to_xyz(lidar_scan)
    points_rgb1 = transform_points(points_lidar, R_lidar_to_rgb1, t_lidar_to_rgb1)
    points_rgb2 = transform_points(points_rgb1, R_rgb1_to_rgb2, t_rgb1_to_rgb2)

    overlay1 = draw_overlay(image1, points_rgb1, rgb1_calib["K"], rgb1_calib["dist"], "RGB1")
    overlay2 = draw_overlay(image2, points_rgb2, rgb2_calib["K"], rgb2_calib["dist"], "RGB2")

    cv2.imwrite(str(RGB1_OUT), overlay1)
    cv2.imwrite(str(RGB2_OUT), overlay2)
    print(f"Saved {RGB1_OUT}")
    print(f"Saved {RGB2_OUT}")

    combined = side_by_side(overlay1, overlay2)
    cv2.imshow(WINDOW, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
