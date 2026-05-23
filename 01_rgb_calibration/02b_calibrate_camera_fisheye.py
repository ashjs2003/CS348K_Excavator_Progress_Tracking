"""
Step 4: run OpenCV fisheye calibration for wide-angle cameras.

Run:
    python 02b_calibrate_camera_fisheye.py
    python 02b_calibrate_camera_fisheye.py --camera L
    python 02b_calibrate_camera_fisheye.py --camera R
"""

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from calib_targets import prompt_camera, resolve_camera


CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.025  # meters


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate fisheye model for the L or R RGB camera.")
    parser.add_argument(
        "--camera",
        choices=["L", "R", "l", "r", "left", "right", "rgb1", "rgb2"],
        default=None,
        help="Which camera to calibrate: L or R. If omitted, you will be prompted.",
    )
    return parser.parse_args()


def make_object_points():
    objp = np.zeros((1, CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float64)
    grid = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp[0, :, :2] = grid * SQUARE_SIZE
    return objp


def find_corners(image):
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_variants = (
        gray_original,
        cv2.equalizeHist(gray_original),
        cv2.GaussianBlur(cv2.equalizeHist(gray_original), (3, 3), 0),
    )

    # Match the capture script: try the newer, more robust SB detector first.
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        for gray in gray_variants:
            found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, sb_flags)
            if found:
                return True, corners.reshape(1, -1, 2).astype(np.float64)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for gray in gray_variants:
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners.reshape(1, -1, 2).astype(np.float64)

    return False, None


def main():
    args = parse_args()
    camera_name = args.camera if args.camera is not None else prompt_camera()
    target = resolve_camera(camera_name)
    image_dir = target["image_dir"]
    output_file = target["fisheye_npz"]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        print(f"Error: no images found in {image_dir.resolve()}")
        return

    obj_template = make_object_points()
    objpoints = []
    imgpoints = []
    image_size = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue

        image_size = (image.shape[1], image.shape[0])
        found, corners = find_corners(image)
        if found:
            objpoints.append(obj_template.copy())
            imgpoints.append(corners)
            print(f"Detected checkerboard: {path}")
        else:
            print(f"Skipped, checkerboard not found: {path}")

    if len(objpoints) < 10:
        print(f"Error: only {len(objpoints)} usable images. Fisheye calibration works best with 20-30.")
        return

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    flags = (
        cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
        + cv2.fisheye.CALIB_CHECK_COND
        + cv2.fisheye.CALIB_FIX_SKEW
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    try:
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            image_size,
            K,
            D,
            rvecs,
            tvecs,
            flags,
            criteria,
        )
    except cv2.error as exc:
        print("Fisheye calibration failed.")
        print("Try more varied images, especially tilted views near image edges.")
        print(exc)
        return

    print(f"\nFisheye calibration complete for {target['label']}.")
    print(f"RMS error: {rms}")
    print("Fisheye K matrix:")
    print(K)
    print("Fisheye D coefficients:")
    print(D)

    np.savez(
        output_file,
        K=K,
        D=D,
        camera_matrix=K,
        dist_coeffs=D,
        image_size=np.array(image_size),
        checkerboard_size=np.array(CHECKERBOARD),
        square_size=np.array(SQUARE_SIZE),
        rms_error=np.array(rms),
    )
    print(f"Saved {output_file.resolve()}")


if __name__ == "__main__":
    main()
