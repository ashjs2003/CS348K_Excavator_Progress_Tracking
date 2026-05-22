"""
Live LiDAR overlay on RGB1 and RGB2.

RGB2 currently uses copied RGB1 intrinsics for this quick prototype. That is
convenient, but approximate: same-model cameras can still have different
intrinsics and distortion. Calibrate RGB2 separately later for better accuracy.

Run:
    python 05_live_lidar_to_both_cameras.py

Controls:
    u - toggle raw/undistorted RGB
    r - reload LiDAR/RGB1 and RGB1/RGB2 extrinsics
    s - save current side-by-side overlay
    q/Esc - quit
"""

from pathlib import Path
import time

import cv2
import numpy as np
from rplidar import RPLidar, RPLidarException

from hardware_settings import (
    BAUDRATE,
    LIDAR_PORT,
    LIDAR_TO_RGB1_FILE,
    RGB1_CALIBRATION_FILE,
    RGB1_CAMERA_INDEX,
    RGB2_CALIBRATION_FILE,
    RGB2_CAMERA_INDEX,
    STEREO_EXTRINSICS_FILE,
)

OUT_IMAGE = Path("live_lidar_to_both_cameras.png")

WINDOW = "Live LiDAR Overlay: RGB1 + RGB2"
MIN_DISTANCE_M = 0.02
MAX_DISTANCE_M = 3.0
DOT_SIZE = 3


def load_camera_calibration(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    data = np.load(path)
    return data["camera_matrix"], data["dist_coeffs"], tuple(data["image_size"].astype(int))


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


def load_extrinsics():
    R_lidar_to_rgb1, t_lidar_to_rgb1 = load_lidar_to_rgb1(LIDAR_TO_RGB1_FILE)
    R_rgb1_to_rgb2, t_rgb1_to_rgb2 = load_stereo_extrinsics(STEREO_EXTRINSICS_FILE)
    print(f"Loaded LiDAR-to-RGB1 extrinsics: {LIDAR_TO_RGB1_FILE}")
    print(f"Loaded RGB1-to-RGB2 extrinsics: {STEREO_EXTRINSICS_FILE}")
    return R_lidar_to_rgb1, t_lidar_to_rgb1, R_rgb1_to_rgb2, t_rgb1_to_rgb2


def open_camera(index, image_size):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(image_size[0]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(image_size[1]))
    return cap


def clean_lidar_startup(lidar):
    clean_input = getattr(lidar, "clean_input", None)
    if clean_input is None:
        clean_input = getattr(lidar, "clear_input", None)

    actions = [lidar.stop, lidar.stop_motor]
    if clean_input is not None:
        actions.append(clean_input)

    for action in actions:
        try:
            action()
            time.sleep(0.3)
        except Exception:
            pass
    lidar.start_motor()
    time.sleep(2.0)


def scan_to_lidar_xyz(scan):
    rows = []
    for quality, angle_deg, distance_mm in scan:
        distance_m = distance_mm / 1000.0
        if quality > 0 and MIN_DISTANCE_M <= distance_m <= MAX_DISTANCE_M:
            theta = np.deg2rad(angle_deg)
            rows.append([
                distance_m * np.cos(theta),
                distance_m * np.sin(theta),
                0.0,
            ])

    if not rows:
        return np.empty((0, 3), dtype=float)
    return np.array(rows, dtype=float)


def project_camera_points(points_camera, K, dist, image_shape):
    if len(points_camera) == 0:
        return np.empty((0, 2)), np.empty(0)

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

    text = f"{label}: projected={len(uv)}"
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def side_by_side(left, right):
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / right.shape[0]
        right = cv2.resize(right, (int(right.shape[1] * scale), left.shape[0]))
    return np.hstack([left, right])


def main():
    K1, dist1, image_size1 = load_camera_calibration(RGB1_CALIBRATION_FILE)
    K2, dist2, image_size2 = load_camera_calibration(RGB2_CALIBRATION_FILE)
    R_lidar_to_rgb1, t_lidar_to_rgb1, R_rgb1_to_rgb2, t_rgb1_to_rgb2 = load_extrinsics()

    cap1 = open_camera(RGB1_CAMERA_INDEX, image_size1)
    cap2 = open_camera(RGB2_CAMERA_INDEX, image_size2)
    lidar = RPLidar(LIDAR_PORT, baudrate=BAUDRATE, timeout=3)

    actual1 = (int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    actual2 = (int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    print(f"RGB1 actual resolution: {actual1[0]}x{actual1[1]}")
    print(f"RGB2 actual resolution: {actual2[0]}x{actual2[1]}")

    use_undistorted = True
    last_combined = None

    try:
        clean_lidar_startup(lidar)
        print("Live LiDAR-to-both-cameras overlay started.")
        print("Press u raw/undistorted, r reload calibration, s save, q quit.")

        should_quit = False
        while not should_quit:
            try:
                for scan in lidar.iter_scans(max_buf_meas=5000):
                    ret1, frame1 = cap1.read()
                    ret2, frame2 = cap2.read()
                    if not ret1 or not ret2:
                        print("Warning: failed to read one or both RGB frames.")
                        continue

                    image1 = cv2.undistort(frame1, K1, dist1, None, K1) if use_undistorted else frame1
                    image2 = cv2.undistort(frame2, K2, dist2, None, K2) if use_undistorted else frame2

                    points_lidar = scan_to_lidar_xyz(scan)
                    points_rgb1 = (R_lidar_to_rgb1 @ points_lidar.T).T + t_lidar_to_rgb1
                    points_rgb2 = (R_rgb1_to_rgb2 @ points_rgb1.T).T + t_rgb1_to_rgb2

                    mode = "undistorted" if use_undistorted else "raw"
                    overlay1 = draw_overlay(image1, points_rgb1, K1, np.zeros_like(dist1) if use_undistorted else dist1, f"RGB1 {mode}")
                    overlay2 = draw_overlay(image2, points_rgb2, K2, np.zeros_like(dist2) if use_undistorted else dist2, f"RGB2 {mode}")
                    last_combined = side_by_side(overlay1, overlay2)

                    help_text = "u toggle | r reload | s save | q quit"
                    cv2.putText(last_combined, help_text, (20, last_combined.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(last_combined, help_text, (20, last_combined.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
                    cv2.imshow(WINDOW, last_combined)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        should_quit = True
                        break
                    if key == ord("u"):
                        use_undistorted = not use_undistorted
                    if key == ord("r"):
                        R_lidar_to_rgb1, t_lidar_to_rgb1, R_rgb1_to_rgb2, t_rgb1_to_rgb2 = load_extrinsics()
                    if key == ord("s") and last_combined is not None:
                        cv2.imwrite(str(OUT_IMAGE), last_combined)
                        print(f"Saved {OUT_IMAGE.resolve()}")
            except RPLidarException as exc:
                print(f"LiDAR stream error: {exc}")
                print("Resetting LiDAR stream...")
                clean_lidar_startup(lidar)

    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()
        try:
            lidar.stop()
            lidar.stop_motor()
        except Exception:
            pass
        lidar.disconnect()


if __name__ == "__main__":
    main()
