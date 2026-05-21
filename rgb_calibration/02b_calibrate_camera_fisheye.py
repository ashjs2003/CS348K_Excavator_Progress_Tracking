"""
Step 4: run OpenCV fisheye calibration for wide-angle cameras.

Run:
    python 02b_calibrate_camera_fisheye.py
"""

from pathlib import Path

import cv2
import numpy as np


CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.025  # meters
IMAGE_DIR = Path("calibration_images")
OUTPUT_FILE = Path("camera_calibration_fisheye.npz")


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
    image_paths = sorted(IMAGE_DIR.glob("*.png")) + sorted(IMAGE_DIR.glob("*.jpg"))
    if not image_paths:
        print(f"Error: no images found in {IMAGE_DIR.resolve()}")
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

    print("\nFisheye calibration complete.")
    print(f"RMS error: {rms}")
    print("Fisheye K matrix:")
    print(K)
    print("Fisheye D coefficients:")
    print(D)

    np.savez(
        OUTPUT_FILE,
        K=K,
        D=D,
        camera_matrix=K,
        dist_coeffs=D,
        image_size=np.array(image_size),
        checkerboard_size=np.array(CHECKERBOARD),
        square_size=np.array(SQUARE_SIZE),
        rms_error=np.array(rms),
    )
    print(f"Saved {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
