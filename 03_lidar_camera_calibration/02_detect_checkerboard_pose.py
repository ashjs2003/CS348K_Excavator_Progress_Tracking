"""
Step 2: estimate checkerboard pose in each captured RGB image.

Run:
    python 02_detect_checkerboard_pose.py
"""

from pathlib import Path
import sys

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.project_config import calibration_file

CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.025  # meters
RGB_CALIBRATION_FILE = calibration_file("left_intrinsics")
PAIR_DIR = Path("pairs")
POSE_DIR = Path("checkerboard_poses")
OUT_FILE = POSE_DIR / "checkerboard_poses.npz"

def make_object_points():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


def find_corners(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, flags)
        if found:
            return True, corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    if not found:
        return False, None

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def pose_to_plane(rvec, tvec):
    """Return checkerboard plane normal and one plane point in camera coordinates."""
    R, _ = cv2.Rodrigues(rvec)
    normal_camera = R @ np.array([[0.0], [0.0], [1.0]])
    normal_camera = normal_camera.reshape(3)
    normal_camera /= np.linalg.norm(normal_camera)
    point_camera = tvec.reshape(3)
    return normal_camera, point_camera


def main():
    POSE_DIR.mkdir(exist_ok=True)

    if not RGB_CALIBRATION_FILE.exists():
        print(f"Error: missing {RGB_CALIBRATION_FILE}")
        return

    calib = np.load(RGB_CALIBRATION_FILE)
    camera_matrix = calib["camera_matrix"]
    dist_coeffs = calib["dist_coeffs"]
    objp = make_object_points()

    pair_ids = []
    rvecs = []
    tvecs = []
    normals = []
    plane_points = []

    for image_path in sorted(PAIR_DIR.glob("pair_*_image.png")):
        pair_id = image_path.stem.split("_")[1]
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"Skipping unreadable image: {image_path}")
            continue

        found, corners = find_corners(image)
        if not found:
            print(f"Skipped pair {pair_id}: checkerboard not found")
            continue

        ok, rvec, tvec = cv2.solvePnP(
            objp,
            corners,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            print(f"Skipped pair {pair_id}: solvePnP failed")
            continue

        normal, plane_point = pose_to_plane(rvec, tvec)
        pair_ids.append(pair_id)
        rvecs.append(rvec.reshape(3))
        tvecs.append(tvec.reshape(3))
        normals.append(normal)
        plane_points.append(plane_point)

        np.savez(
            POSE_DIR / f"pair_{pair_id}_pose.npz",
            rvec=rvec,
            tvec=tvec,
            plane_normal_camera=normal,
            plane_point_camera=plane_point,
        )

        vis = image.copy()
        cv2.drawChessboardCorners(vis, CHECKERBOARD, corners, found)
        cv2.imwrite(str(POSE_DIR / f"pair_{pair_id}_corners.png"), vis)
        print(f"Pose saved for pair {pair_id}")

    if not pair_ids:
        print("Error: no checkerboard poses were detected.")
        return

    np.savez(
        OUT_FILE,
        pair_ids=np.array(pair_ids),
        rvecs=np.array(rvecs),
        tvecs=np.array(tvecs),
        plane_normals_camera=np.array(normals),
        plane_points_camera=np.array(plane_points),
    )
    print(f"Saved {OUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
