"""
Step 4: manually tune LiDAR-to-camera extrinsics with OpenCV trackbars.

Run:
    python 04_manual_lidar_camera_overlay.py

Controls:
    n/right - next pair
    b/left - previous pair
    s - save current transform
    q/Esc - quit
"""

from pathlib import Path
import sys

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.project_config import calibration_file

RGB_CALIBRATION_FILE = calibration_file("left_intrinsics")
PAIR_DIR = Path("pairs")
OUT_FILE = calibration_file("lidar_to_rgb1_manual_extrinsics")
OPT_EXTRINSICS_FILE = calibration_file("lidar_to_rgb1_extrinsics")
WINDOW = "Manual LiDAR Camera Overlay"

def load_lidar_csv(path):
    data = np.genfromtxt(path, delimiter=",", names=True)
    if data.ndim == 0:
        data = np.array([data])
    return np.column_stack([data["angle_degrees"], data["distance_meters"], data["quality"]])


def lidar_to_xyz(lidar):
    theta = np.deg2rad(lidar[:, 0])
    x = lidar[:, 1] * np.cos(theta)
    y = lidar[:, 1] * np.sin(theta)
    z = np.zeros_like(x)
    return np.column_stack([x, y, z])


def euler_to_R(roll_deg, pitch_deg, yaw_deg):
    r, p, y = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
    Rx = np.array([[1, 0, 0], [0, np.cos(r), -np.sin(r)], [0, np.sin(r), np.cos(r)]])
    Ry = np.array([[np.cos(p), 0, np.sin(p)], [0, 1, 0], [-np.sin(p), 0, np.cos(p)]])
    Rz = np.array([[np.cos(y), -np.sin(y), 0], [np.sin(y), np.cos(y), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


def R_to_euler(R):
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    if abs(np.cos(pitch)) > 1e-6:
        roll = np.arctan2(R[2, 1], R[2, 2])
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = 0.0
        yaw = np.arctan2(-R[0, 1], R[1, 1])
    return np.rad2deg([roll, pitch, yaw])


def clamp(value, low, high):
    return max(low, min(high, int(round(value))))


def load_initial_params():
    params = {
        "tx": 0.0,
        "ty": 0.0,
        "tz": 0.5,
        "roll": 0.0,
        "pitch": 0.0,
        "yaw": 0.0,
        "min_dist": 0.0,
        "max_dist": 4.0,
        "dot_size": 3,
    }

    path = OPT_EXTRINSICS_FILE if OPT_EXTRINSICS_FILE.exists() else OUT_FILE
    if not path.exists():
        print("No saved extrinsics found; starting from default sliders.")
        return params

    data = np.load(path)
    params["tx"], params["ty"], params["tz"] = data["t"].reshape(3)
    if "roll_pitch_yaw_degrees" in data:
        params["roll"], params["pitch"], params["yaw"] = data["roll_pitch_yaw_degrees"]
    else:
        params["roll"], params["pitch"], params["yaw"] = R_to_euler(data["R"])
    print(f"Loaded initial transform: {path}")
    return params


def create_trackbars(initial_params):
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, 1280, 720)
    cv2.createTrackbar("tx mm", WINDOW, clamp(initial_params["tx"] * 1000 + 1000, 0, 2000), 2000, lambda x: None)
    cv2.createTrackbar("ty mm", WINDOW, clamp(initial_params["ty"] * 1000 + 1000, 0, 2000), 2000, lambda x: None)
    cv2.createTrackbar("tz mm", WINDOW, clamp(initial_params["tz"] * 1000 + 1000, 0, 3000), 3000, lambda x: None)
    cv2.createTrackbar("roll deg", WINDOW, clamp(initial_params["roll"] + 180, 0, 360), 360, lambda x: None)
    cv2.createTrackbar("pitch deg", WINDOW, clamp(initial_params["pitch"] + 180, 0, 360), 360, lambda x: None)
    cv2.createTrackbar("yaw deg", WINDOW, clamp(initial_params["yaw"] + 180, 0, 360), 360, lambda x: None)
    cv2.createTrackbar("min dist cm", WINDOW, clamp(initial_params["min_dist"] * 100, 0, 500), 500, lambda x: None)
    cv2.createTrackbar("max dist cm", WINDOW, clamp(initial_params["max_dist"] * 100, 0, 1000), 1000, lambda x: None)
    cv2.createTrackbar("dot size", WINDOW, clamp(initial_params["dot_size"], 1, 12), 12, lambda x: None)


def read_params():
    min_dist = cv2.getTrackbarPos("min dist cm", WINDOW) / 100.0
    max_dist = cv2.getTrackbarPos("max dist cm", WINDOW) / 100.0
    if max_dist <= min_dist:
        max_dist = min_dist + 0.05
    return {
        "tx": (cv2.getTrackbarPos("tx mm", WINDOW) - 1000) / 1000.0,
        "ty": (cv2.getTrackbarPos("ty mm", WINDOW) - 1000) / 1000.0,
        "tz": (cv2.getTrackbarPos("tz mm", WINDOW) - 1000) / 1000.0,
        "roll": cv2.getTrackbarPos("roll deg", WINDOW) - 180,
        "pitch": cv2.getTrackbarPos("pitch deg", WINDOW) - 180,
        "yaw": cv2.getTrackbarPos("yaw deg", WINDOW) - 180,
        "min_dist": min_dist,
        "max_dist": max_dist,
        "dot_size": max(1, cv2.getTrackbarPos("dot size", WINDOW)),
    }


def project_lidar(points_lidar, params, camera_matrix, image_shape):
    distances = np.linalg.norm(points_lidar[:, :2], axis=1)
    mask = (distances >= params["min_dist"]) & (distances <= params["max_dist"])
    R = euler_to_R(params["roll"], params["pitch"], params["yaw"])
    t = np.array([params["tx"], params["ty"], params["tz"]])
    points_camera = (R @ points_lidar[mask].T).T + t
    valid_z = points_camera[:, 2] > 0.05
    points_camera = points_camera[valid_z]

    if len(points_camera) == 0:
        return np.empty((0, 2)), np.empty(0), R, t

    uv = (camera_matrix @ points_camera.T).T
    uv = uv[:, :2] / uv[:, 2:3]
    h, w = image_shape[:2]
    in_image = (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    return uv[in_image], points_camera[in_image, 2], R, t


def draw_overlay(image, lidar_points, params, camera_matrix, pair_label):
    vis = image.copy()
    uv, depth, R, t = project_lidar(lidar_points, params, camera_matrix, image.shape)

    if len(uv) > 0:
        colors = cv2.applyColorMap(
            np.clip(depth * 80, 0, 255).astype(np.uint8).reshape(-1, 1),
            cv2.COLORMAP_TURBO,
        )
        for point, color in zip(uv, colors[:, 0, :]):
            cv2.circle(vis, tuple(np.round(point).astype(int)), params["dot_size"], color.tolist(), -1)
        point_status = f"projected points={len(uv)}"
    else:
        point_status = "projected points=0; adjust tz/rotations or distance range"

    text = (
        f"{pair_label} | n next b prev | "
        f"t=({params['tx']:.3f},{params['ty']:.3f},{params['tz']:.3f}) m "
        f"rpy=({params['roll']},{params['pitch']},{params['yaw']}) deg | {point_status} | s save q quit"
    )
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 1, cv2.LINE_AA)
    return vis, R, t


def load_pair(pair_id, K, dist):
    image_path = PAIR_DIR / f"pair_{pair_id}_image.png"
    lidar_path = PAIR_DIR / f"pair_{pair_id}_lidar.csv"

    image = cv2.imread(str(image_path))
    if image is None:
        raise RuntimeError(f"Could not load image: {image_path}")

    lidar = load_lidar_csv(lidar_path)
    lidar_points = lidar_to_xyz(lidar)
    undistorted = cv2.undistort(image, K, dist, None, K)
    return image_path, undistorted, lidar_points


def main():
    pair_id = sys.argv[1] if len(sys.argv) > 1 else None
    image_paths = sorted(PAIR_DIR.glob("pair_*_image.png"))
    if not image_paths:
        print("Error: no pairs found. Run 01_capture_rgb_lidar_pairs.py first.")
        return
    pair_ids = [path.stem.split("_")[1] for path in image_paths]
    pair_index = pair_ids.index(pair_id) if pair_id in pair_ids else 0

    calib = np.load(RGB_CALIBRATION_FILE)
    K = calib["camera_matrix"]
    dist = calib["dist_coeffs"]
    create_trackbars(load_initial_params())

    last_R = np.eye(3)
    last_t = np.zeros(3)
    last_params = None
    current_pair_id = None
    undistorted = None
    lidar_points = None
    last_overlay = None

    while True:
        pair_id = pair_ids[pair_index]
        if pair_id != current_pair_id:
            image_path, undistorted, lidar_points = load_pair(pair_id, K, dist)
            current_pair_id = pair_id
            print(f"Showing pair {pair_id} ({pair_index + 1}/{len(pair_ids)}): {image_path}")

        params = read_params()
        pair_label = f"pair {pair_id} ({pair_index + 1}/{len(pair_ids)})"
        overlay, last_R, last_t = draw_overlay(undistorted, lidar_points, params, K, pair_label)
        last_params = params
        last_overlay = overlay
        cv2.imshow(WINDOW, overlay)

        key = cv2.waitKey(30) & 0xFF
        if key in (ord("q"), 27):
            break
        if key in (ord("n"), 83):
            pair_index = (pair_index + 1) % len(pair_ids)
        if key in (ord("b"), 81):
            pair_index = (pair_index - 1) % len(pair_ids)
        if key == ord("s"):
            save_kwargs = {
                "R": last_R,
                "t": last_t,
                "roll_pitch_yaw_degrees": np.array([last_params["roll"], last_params["pitch"], last_params["yaw"]]),
                "translation_meters": last_t,
                "source_pair_id": np.array(pair_id),
            }
            OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            OPT_EXTRINSICS_FILE.parent.mkdir(parents=True, exist_ok=True)
            np.savez(OUT_FILE, **save_kwargs)
            np.savez(OPT_EXTRINSICS_FILE, **save_kwargs)
            cv2.imwrite("manual_overlay_preview.png", last_overlay)
            print(f"Saved {OUT_FILE.resolve()}")
            print(f"Updated {OPT_EXTRINSICS_FILE.resolve()} for validation.")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
