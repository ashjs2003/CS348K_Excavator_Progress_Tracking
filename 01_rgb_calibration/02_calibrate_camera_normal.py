"""
Step 2: run normal OpenCV pinhole camera calibration.

Run:
    python 02_calibrate_camera_normal.py
    python 02_calibrate_camera_normal.py --camera L
    python 02_calibrate_camera_normal.py --camera R
"""

import argparse
import shutil
from pathlib import Path

import cv2
import numpy as np

from calib_targets import prompt_camera, resolve_camera

CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.025  # meters


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate the L or R RGB camera from checkerboard images.")
    parser.add_argument(
        "--camera",
        choices=["L", "R", "l", "r", "left", "right", "rgb1", "rgb2"],
        default=None,
        help="Which camera to calibrate: L or R. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--copy-to-config",
        action="store_true",
        help="Also copy the result into config/ for stereo and LiDAR scripts",
    )
    return parser.parse_args()


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
    args = parse_args()
    camera_name = args.camera if args.camera is not None else prompt_camera()
    target = resolve_camera(camera_name)
    image_dir = target["image_dir"]
    output_file = target["local_npz"]
    output_file.parent.mkdir(parents=True, exist_ok=True)

    image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    if not image_paths:
        print(f"Error: no images found in {image_dir.resolve()}")
        print(f"Run: python 01_capture_checkerboard_images.py --camera {target['label']}")
        return

    print(f"Calibrating {target['label']} from {image_dir.resolve()}")

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
        output_file,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        checkerboard_size=np.array(CHECKERBOARD),
        square_size=np.array(SQUARE_SIZE),
        rms_error=np.array(rms),
        mean_reprojection_error=np.array(mean_error),
    )
    print(f"Saved {output_file.resolve()}")

    if args.copy_to_config:
        target["config_npz"].parent.mkdir(parents=True, exist_ok=True)
        if output_file.resolve() != target["config_npz"].resolve():
            shutil.copy2(output_file, target["config_npz"])
            print(f"Copied to {target['config_npz'].resolve()}")
        else:
            print(f"Calibration is already saved at {target['config_npz'].resolve()}")
    else:
        print(f"For the full pipeline, copy to: {target['config_npz'].resolve()}")
        print(f"  cp {output_file} {target['config_npz']}")


if __name__ == "__main__":
    main()
