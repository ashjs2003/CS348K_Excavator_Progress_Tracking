"""
Project LiDAR points onto RGB1 and RGB2 capture images.

Run:
    python 05_project_lidar_overlay.py
    python 05_project_lidar_overlay.py --run latest
"""

import argparse
import sys
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

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths

WINDOW = "LiDAR overlays on RGB1 and RGB2"


def parse_args():
    parser = argparse.ArgumentParser()
    add_run_cli_arguments(parser)
    return parser.parse_args()


def draw_overlay(image, points_camera, K, dist, label):
    vis = image.copy()
    pixels, _ = project_camera_points(points_camera, K, dist, image.shape)
    for u, v in pixels:
        cv2.circle(vis, (int(u), int(v)), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)
    cv2.putText(vis, label, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    return vis


def side_by_side(left, right):
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / right.shape[0]
        right = cv2.resize(right, (int(right.shape[1] * scale), left.shape[0]))
    return np.hstack([left, right])


def main():
    args = parse_args()
    if handle_list_runs(args):
        return
    paths = resolve_run_paths(args.run)
    paths.overlays.mkdir(parents=True, exist_ok=True)
    rgb1_out = paths.overlays / "lidar_overlay_rgb1.png"
    rgb2_out = paths.overlays / "lidar_overlay_rgb2.png"

    if paths.run_dir:
        print(f"Run: {paths.run_dir.name}")
    print(f"Writing overlays to {paths.overlays}")

    rgb1_calib = load_camera_calibration("camera_calibration_rgb1.npz")
    rgb2_calib = load_camera_calibration("camera_calibration_rgb2.npz", "camera_calibration_rgb2_approx.npz")
    _, R_lidar_to_rgb1, t_lidar_to_rgb1 = load_lidar_to_rgb1()
    _, R_rgb1_to_rgb2, t_rgb1_to_rgb2 = load_stereo_rgb1_to_rgb2()

    image1 = cv2.imread(str(paths.rgb1_image))
    image2 = cv2.imread(str(paths.rgb2_image))
    if image1 is None:
        raise RuntimeError(f"Could not load {paths.rgb1_image}")
    if image2 is None:
        raise RuntimeError(f"Could not load {paths.rgb2_image}")

    lidar_scan = read_lidar_csv(paths.lidar_csv)
    points_lidar = lidar_polar_to_xyz(lidar_scan)
    points_rgb1 = transform_points(points_lidar, R_lidar_to_rgb1, t_lidar_to_rgb1)
    points_rgb2 = transform_points(points_rgb1, R_rgb1_to_rgb2, t_rgb1_to_rgb2)

    overlay1 = draw_overlay(image1, points_rgb1, rgb1_calib["K"], rgb1_calib["dist"], "RGB1")
    overlay2 = draw_overlay(image2, points_rgb2, rgb2_calib["K"], rgb2_calib["dist"], "RGB2")

    cv2.imwrite(str(rgb1_out), overlay1)
    cv2.imwrite(str(rgb2_out), overlay2)
    print(f"Saved {rgb1_out}")
    print(f"Saved {rgb2_out}")

    combined = side_by_side(overlay1, overlay2)
    cv2.imshow(WINDOW, combined)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
