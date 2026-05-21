"""
Step 2: run normal OpenCV pinhole camera calibration.

Run:
    python 02_calibrate_camera_normal.py
"""

from pathlib import Path

import cv2
import numpy as np


CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.025  # meters
IMAGE_DIR = Path("calibration_images")
OUTPUT_FILE = Path("camera_calibration_normal.npz")


def make_object_points():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
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
                return True, corners.astype(np.float32), gray

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for gray in gray_variants:
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        if found:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners, gray

    return False, None, gray_original


def mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0.0
    total_points = 0

    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected, cv2.NORM_L2)
        total_error += error * error
        total_points += len(objp)

    return float(np.sqrt(total_error / total_points))


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
        found, corners, _ = find_corners(image)
        if found:
            objpoints.append(obj_template.copy())
            imgpoints.append(corners)
            print(f"Detected checkerboard: {path}")
        else:
            print(f"Skipped, checkerboard not found: {path}")

    if len(objpoints) < 5:
        print(f"Error: only {len(objpoints)} usable images. Capture at least 5; 20-30 is better.")
        return

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    mean_error = mean_reprojection_error(
        objpoints,
        imgpoints,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )

    print("\nNormal calibration complete.")
    print(f"RMS reprojection error: {rms}")
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(dist_coeffs)
    print(f"Mean reprojection error: {mean_error}")

    np.savez(
        OUTPUT_FILE,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        checkerboard_size=np.array(CHECKERBOARD),
        square_size=np.array(SQUARE_SIZE),
        rms_error=np.array(rms),
        mean_reprojection_error=np.array(mean_error),
    )
    print(f"Saved {OUTPUT_FILE.resolve()}")


if __name__ == "__main__":
    main()
