"""
Live RGB + LiDAR overlay using the calibrated LiDAR-to-camera extrinsics.

Run:
    python 07_live_lidar_rgb_overlay.py

Controls:
    u - toggle raw/undistorted RGB
    r - reload saved extrinsics
    s - save current overlay image
    q/Esc - quit
"""

from pathlib import Path
import sys
import time

import cv2
import numpy as np
from rplidar import RPLidar, RPLidarException

from calibration_settings import MAX_CAPTURE_DISTANCE_M, MIN_CAPTURE_DISTANCE_M

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.project_config import calibration_file, camera_index, lidar_baudrate, lidar_port

CAMERA_INDEX = camera_index("L")
BAUDRATE = lidar_baudrate()
LIDAR_PORT = lidar_port()
RGB_CALIBRATION_FILE = calibration_file("left_intrinsics")
OPT_EXTRINSICS_FILE = calibration_file("lidar_to_rgb1_extrinsics")
MANUAL_EXTRINSICS_FILE = calibration_file("lidar_to_rgb1_manual_extrinsics")
OUT_IMAGE = Path("live_lidar_rgb_overlay.png")
WINDOW = "Live RGB + LiDAR Overlay"
MIN_DISTANCE_M = MIN_CAPTURE_DISTANCE_M
MAX_DISTANCE_M = MAX_CAPTURE_DISTANCE_M
DOT_SIZE = 3


def open_camera(image_size):
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open RGB camera index {CAMERA_INDEX}.")

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


def load_extrinsics():
    path = OPT_EXTRINSICS_FILE if OPT_EXTRINSICS_FILE.exists() else MANUAL_EXTRINSICS_FILE
    if not path.exists():
        raise FileNotFoundError(
            "No LiDAR-camera extrinsics found. Run 05_optimize_lidar_to_camera_extrinsics.py "
            "or 04_manual_lidar_camera_overlay.py first."
        )
    data = np.load(path)
    print(f"Loaded extrinsics: {path}")
    return data["R"], data["t"].reshape(3), path


def scan_to_xyz(scan):
    points = []
    for quality, angle_deg, distance_mm in scan:
        distance_m = distance_mm / 1000.0
        if quality > 0 and MIN_DISTANCE_M <= distance_m <= MAX_DISTANCE_M:
            theta = np.deg2rad(angle_deg)
            points.append([
                distance_m * np.cos(theta),
                distance_m * np.sin(theta),
                0.0,
                quality,
            ])

    if not points:
        return np.empty((0, 4), dtype=float)
    return np.array(points, dtype=float)


def project_lidar(points_lidar, R, t, K, image_shape):
    if len(points_lidar) == 0:
        return np.empty((0, 2)), np.empty(0)

    xyz_lidar = points_lidar[:, :3]
    points_camera = (R @ xyz_lidar.T).T + t
    valid_z = points_camera[:, 2] > 0.05
    points_camera = points_camera[valid_z]
    if len(points_camera) == 0:
        return np.empty((0, 2)), np.empty(0)

    uv = (K @ points_camera.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    h, w = image_shape[:2]
    in_image = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[in_image], points_camera[in_image, 2]


def draw_overlay(image, points_lidar, R, t, K, use_undistorted, extrinsics_path):
    vis = image.copy()
    uv, depth = project_lidar(points_lidar, R, t, K, image.shape)

    if len(depth):
        colors = cv2.applyColorMap(
            np.clip(depth * 255 / max(MAX_DISTANCE_M, 0.1), 0, 255).astype(np.uint8).reshape(-1, 1),
            cv2.COLORMAP_TURBO,
        )
        for point, color in zip(uv, colors[:, 0, :]):
            cv2.circle(vis, tuple(np.round(point).astype(int)), DOT_SIZE, color.tolist(), -1)

    mode = "undistorted" if use_undistorted else "raw"
    text = (
        f"{mode} | projected={len(uv)} | range={MIN_DISTANCE_M:.2f}-{MAX_DISTANCE_M:.2f}m | "
        f"{extrinsics_path.name} | u toggle r reload s save q quit"
    )
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def main():
    calib = np.load(RGB_CALIBRATION_FILE)
    K = calib["camera_matrix"]
    dist = calib["dist_coeffs"]
    image_size = tuple(calib["image_size"].astype(int))

    R, t, extrinsics_path = load_extrinsics()
    cap = open_camera(image_size)
    lidar = RPLidar(LIDAR_PORT, baudrate=BAUDRATE, timeout=3)

    use_undistorted = True
    last_overlay = None

    try:
        clean_lidar_startup(lidar)
        print("Live RGB + LiDAR overlay started.")
        print("Press u to toggle raw/undistorted, r to reload calibration, s to save, q to quit.")

        should_quit = False
        while not should_quit:
            try:
                for scan in lidar.iter_scans(max_buf_meas=5000):
                    ret, frame = cap.read()
                    if not ret:
                        print("Warning: failed to read RGB frame.")
                        continue

                    image = cv2.undistort(frame, K, dist, None, K) if use_undistorted else frame
                    points_lidar = scan_to_xyz(scan)
                    last_overlay = draw_overlay(image, points_lidar, R, t, K, use_undistorted, extrinsics_path)
                    cv2.imshow(WINDOW, last_overlay)

                    key = cv2.waitKey(1) & 0xFF
                    if key in (ord("q"), 27):
                        should_quit = True
                        break
                    if key == ord("u"):
                        use_undistorted = not use_undistorted
                    if key == ord("r"):
                        R, t, extrinsics_path = load_extrinsics()
                    if key == ord("s") and last_overlay is not None:
                        cv2.imwrite(str(OUT_IMAGE), last_overlay)
                        print(f"Saved {OUT_IMAGE.resolve()}")
            except RPLidarException as exc:
                print(f"LiDAR stream error: {exc}")
                print("Resetting LiDAR stream...")
                clean_lidar_startup(lidar)

    finally:
        cap.release()
        cv2.destroyAllWindows()
        try:
            lidar.stop()
            lidar.stop_motor()
        except Exception:
            pass
        lidar.disconnect()


if __name__ == "__main__":
    main()
