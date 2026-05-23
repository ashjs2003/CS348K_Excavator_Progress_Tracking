"""
Step 5: optimize LiDAR-to-camera extrinsics using plane constraints.

Run:
    python 05_optimize_lidar_to_camera_extrinsics.py
"""

from pathlib import Path
import sys

import cv2
import numpy as np
from scipy.optimize import least_squares

from calibration_settings import (
    CHECKERBOARD_INNER_CORNERS,
    EXPECTED_SEGMENT_AXIS,
    EXPECTED_SEGMENT_LENGTH_M,
    SQUARE_SIZE_M,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.project_config import calibration_file

POSE_FILE = Path("checkerboard_poses/checkerboard_poses.npz")
SELECTED_DIR = Path("selected_lidar_points")
MANUAL_EXTRINSICS_FILE = calibration_file("lidar_to_rgb1_manual_extrinsics")
OUT_FILE = calibration_file("lidar_to_rgb1_extrinsics")
EDGE_RESIDUAL_WEIGHT = 5.0
BOARD_X_MIN_M = -SQUARE_SIZE_M
BOARD_X_MAX_M = CHECKERBOARD_INNER_CORNERS[0] * SQUARE_SIZE_M
BOARD_Y_MIN_M = -SQUARE_SIZE_M
BOARD_Y_MAX_M = CHECKERBOARD_INNER_CORNERS[1] * SQUARE_SIZE_M

def rodrigues_to_R(rvec):
    R, _ = cv2.Rodrigues(np.asarray(rvec, dtype=float).reshape(3, 1))
    return R


def R_to_rodrigues(R):
    rvec, _ = cv2.Rodrigues(R)
    return rvec.reshape(3)


def load_initial_guess():
    if MANUAL_EXTRINSICS_FILE.exists():
        data = np.load(MANUAL_EXTRINSICS_FILE)
        return np.r_[R_to_rodrigues(data["R"]), data["t"].reshape(3)]
    print("Warning: no manual extrinsics found. Starting optimization from identity.")
    return np.zeros(6)


def endpoint_edge_residuals(endpoints_lidar, R_lidar_to_camera, t_lidar_to_camera, rvec_board, tvec_board):
    """Constrain transformed LiDAR segment endpoints to opposite board edges."""
    R_board_to_camera, _ = cv2.Rodrigues(np.asarray(rvec_board).reshape(3, 1))
    t_board_to_camera = np.asarray(tvec_board).reshape(3)

    endpoints_camera = (R_lidar_to_camera @ endpoints_lidar.T).T + t_lidar_to_camera
    endpoints_board = (R_board_to_camera.T @ (endpoints_camera - t_board_to_camera).T).T
    p0 = endpoints_board[0]
    p1 = endpoints_board[1]

    if EXPECTED_SEGMENT_AXIS == "width":
        # The segment crosses the full board width, so endpoints lie on x-min/x-max.
        assign_a = np.array([p0[0] - BOARD_X_MIN_M, p1[0] - BOARD_X_MAX_M])
        assign_b = np.array([p0[0] - BOARD_X_MAX_M, p1[0] - BOARD_X_MIN_M])
        edge_residual = assign_a if np.linalg.norm(assign_a) <= np.linalg.norm(assign_b) else assign_b
        same_cross_axis = np.array([p0[1] - p1[1]])
    else:
        # The segment crosses the full board height, so endpoints lie on y-min/y-max.
        assign_a = np.array([p0[1] - BOARD_Y_MIN_M, p1[1] - BOARD_Y_MAX_M])
        assign_b = np.array([p0[1] - BOARD_Y_MAX_M, p1[1] - BOARD_Y_MIN_M])
        edge_residual = assign_a if np.linalg.norm(assign_a) <= np.linalg.norm(assign_b) else assign_b
        same_cross_axis = np.array([p0[0] - p1[0]])

    # z=0 means the endpoints are on the checkerboard plane. The selected
    # points also get a plane residual, but endpoint z keeps edge constraints sane.
    plane_residual = np.array([p0[2], p1[2]])
    return np.concatenate([edge_residual, same_cross_axis, plane_residual])


def residuals(x, observations):
    R = rodrigues_to_R(x[:3])
    t = x[3:6]
    all_residuals = []

    for points_lidar, endpoints_lidar, normal, plane_point, rvec_board, tvec_board in observations:
        points_camera = (R @ points_lidar.T).T + t
        signed_distances = (points_camera - plane_point.reshape(1, 3)) @ normal.reshape(3)
        all_residuals.append(signed_distances)

        edge_res = endpoint_edge_residuals(endpoints_lidar, R, t, rvec_board, tvec_board)
        edge_res *= EDGE_RESIDUAL_WEIGHT
        all_residuals.append(edge_res)

    return np.concatenate(all_residuals)


def main():
    if not POSE_FILE.exists():
        print("Error: run 02_detect_checkerboard_pose.py first.")
        return

    poses = np.load(POSE_FILE, allow_pickle=True)
    pose_by_id = {
        str(pair_id): (normal, point, rvec, tvec)
        for pair_id, normal, point, rvec, tvec in zip(
            poses["pair_ids"],
            poses["plane_normals_camera"],
            poses["plane_points_camera"],
            poses["rvecs"],
            poses["tvecs"],
        )
    }

    observations = []
    used_pair_ids = []
    for path in sorted(SELECTED_DIR.glob("pair_*_selected_points_lidar.npy")):
        pair_id = path.stem.split("_")[1]
        if pair_id not in pose_by_id:
            print(f"Skipping pair {pair_id}: no RGB plane pose")
            continue
        points_lidar = np.load(path)
        if len(points_lidar) < 2:
            print(f"Skipping pair {pair_id}: too few selected LiDAR points")
            continue
        segment_path = SELECTED_DIR / f"pair_{pair_id}_segment_lidar.npz"
        if not segment_path.exists():
            print(f"Skipping pair {pair_id}: missing fitted segment endpoints")
            continue
        segment = np.load(segment_path)
        endpoints_lidar = segment["endpoints_lidar"]
        length_error = float(segment["length_error_m"])
        print(
            f"Pair {pair_id}: selected segment length={float(segment['estimated_length_m']):.3f}m, "
            f"target={EXPECTED_SEGMENT_LENGTH_M:.3f}m, error={length_error:.3f}m"
        )
        normal, plane_point, rvec_board, tvec_board = pose_by_id[pair_id]
        observations.append((points_lidar, endpoints_lidar, normal, plane_point, rvec_board, tvec_board))
        used_pair_ids.append(pair_id)

    if len(observations) < 2:
        print("Error: need selected LiDAR board points for at least two pairs.")
        return

    x0 = load_initial_guess()
    result = least_squares(residuals, x0, args=(observations,), loss="soft_l1", f_scale=0.02)
    R = rodrigues_to_R(result.x[:3])
    t = result.x[3:6]
    residual_m = residuals(result.x, observations)

    print("Optimization complete.")
    print(f"Expected segment axis: {EXPECTED_SEGMENT_AXIS}")
    print(f"Expected segment length: {EXPECTED_SEGMENT_LENGTH_M:.3f} m")
    print(f"Used pairs: {used_pair_ids}")
    print(f"Mean absolute plane error: {np.mean(np.abs(residual_m)):.4f} m")
    print(f"RMS plane error: {np.sqrt(np.mean(residual_m ** 2)):.4f} m")
    print("R:")
    print(R)
    print("t meters:")
    print(t)

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        OUT_FILE,
        R=R,
        t=t,
        rvec=result.x[:3],
        translation_meters=t,
        used_pair_ids=np.array(used_pair_ids),
        mean_abs_plane_error_m=np.array(np.mean(np.abs(residual_m))),
        rms_plane_error_m=np.array(np.sqrt(np.mean(residual_m ** 2))),
    )
    print(f"Saved {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
