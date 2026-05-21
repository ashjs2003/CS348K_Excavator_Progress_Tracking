"""
Step 6: validate LiDAR projection on an RGB image.

Run:
    python 06_validate_lidar_projection.py

Controls:
    n/right - next pair
    b/left - previous pair
    u - toggle raw/undistorted image
    s - save overlay visualization
    q/Esc - quit
"""

from pathlib import Path
import sys

import cv2
import numpy as np


RGB_CALIBRATION_FILE = Path("../rgb_calibration/camera_calibration_normal.npz")
PAIR_DIR = Path("pairs")
OPT_EXTRINSICS_FILE = Path("lidar_to_camera_extrinsics_optimized.npz")
MANUAL_EXTRINSICS_FILE = Path("lidar_to_camera_extrinsics_manual.npz")


def load_lidar_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data])
    return np.column_stack([data["angle_degrees"], data["distance_meters"], data["quality"]])


def lidar_to_xyz(lidar):
    theta = np.deg2rad(lidar[:, 0])
    return np.column_stack([
        lidar[:, 1] * np.cos(theta),
        lidar[:, 1] * np.sin(theta),
        np.zeros(len(lidar)),
    ])


def load_extrinsics():
    path = OPT_EXTRINSICS_FILE if OPT_EXTRINSICS_FILE.exists() else MANUAL_EXTRINSICS_FILE
    if not path.exists():
        raise FileNotFoundError("No LiDAR-camera extrinsics found.")
    data = np.load(path)
    print(f"Loaded extrinsics: {path}")
    return data["R"], data["t"].reshape(3)


def project(points_lidar, R, t, K, image_shape):
    points_camera = (R @ points_lidar.T).T + t
    valid_z = points_camera[:, 2] > 0.05
    points_camera = points_camera[valid_z]
    if len(points_camera) == 0:
        return np.empty((0, 2)), np.empty(0)

    uv = (K @ points_camera.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    h, w = image_shape[:2]
    valid_img = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[valid_img], points_camera[valid_img, 2]


def draw_overlay(image, points_lidar, R, t, K, label):
    vis = image.copy()
    uv, depth = project(points_lidar, R, t, K, image.shape)
    if len(depth):
        colors = cv2.applyColorMap(
            np.clip(depth * 80, 0, 255).astype(np.uint8).reshape(-1, 1),
            cv2.COLORMAP_TURBO,
        )
        for point, color in zip(uv, colors[:, 0, :]):
            cv2.circle(vis, tuple(np.round(point).astype(int)), 3, color.tolist(), -1)

    cv2.putText(vis, f"{label} | n next | b prev | u toggle | s save | q quit", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, f"{label} | n next | b prev | u toggle | s save | q quit", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 1, cv2.LINE_AA)
    return vis


def load_pair(pair_id):
    image_path = PAIR_DIR / f"pair_{pair_id}_image.png"
    lidar_path = PAIR_DIR / f"pair_{pair_id}_lidar.csv"

    raw = cv2.imread(str(image_path))
    if raw is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    lidar = load_lidar_csv(lidar_path)
    return image_path, raw, lidar_to_xyz(lidar)


def main():
    pair_id = sys.argv[1] if len(sys.argv) > 1 else None
    image_paths = sorted(PAIR_DIR.glob("pair_*_image.png"))
    if not image_paths:
        print("Error: no pairs found.")
        return
    pair_ids = [path.stem.split("_")[1] for path in image_paths]
    pair_index = pair_ids.index(pair_id) if pair_id in pair_ids else 0

    calib = np.load(RGB_CALIBRATION_FILE)
    K = calib["camera_matrix"]
    dist = calib["dist_coeffs"]
    R, t = load_extrinsics()

    use_undistorted = True
    last_overlay = None
    current_pair_id = None
    raw = None
    undistorted = None
    points_lidar = None

    while True:
        pair_id = pair_ids[pair_index]
        if pair_id != current_pair_id:
            image_path, raw, points_lidar = load_pair(pair_id)
            undistorted = cv2.undistort(raw, K, dist, None, K)
            current_pair_id = pair_id
            print(f"Showing pair {pair_id} ({pair_index + 1}/{len(pair_ids)}): {image_path}")

        image = undistorted if use_undistorted else raw
        label = f"pair {pair_id} ({pair_index + 1}/{len(pair_ids)}) | {'undistorted' if use_undistorted else 'raw'}"
        last_overlay = draw_overlay(image, points_lidar, R, t, K, label)
        cv2.imshow("Validate LiDAR Projection", last_overlay)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), 83):
            pair_index = (pair_index + 1) % len(pair_ids)
        if key in (ord("b"), 81):
            pair_index = (pair_index - 1) % len(pair_ids)
        if key == ord("u"):
            use_undistorted = not use_undistorted
        if key == ord("s"):
            out_image = Path(f"lidar_projection_validation_pair_{pair_id}.png")
            cv2.imwrite(str(out_image), last_overlay)
            print(f"Saved {out_image.resolve()}")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
