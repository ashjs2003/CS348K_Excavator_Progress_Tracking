"""
Estimate RGB1-to-RGB2 extrinsics from stereo checkerboard pairs.

This prototype fixes both cameras' intrinsics. RGB2's intrinsics are copied
from RGB1 for now, but RGB2 should ideally be calibrated separately later:
same-model cameras can still have different focal lengths, principal points,
and lens distortion.

Run:
    python 03_stereo_calibrate_rgb1_rgb2_fixed_intrinsics.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

from hardware_settings import (
    RGB1_CALIBRATION_FILE,
    RGB2_CALIBRATION_FILE,
    STEREO_EXTRINSICS_FILE,
    STEREO_PAIRS_DIR,
)

PAIR_DIR = STEREO_PAIRS_DIR
OUT_FILE = STEREO_EXTRINSICS_FILE


def load_calibration(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    data = np.load(path)
    return {
        "camera_matrix": data["camera_matrix"].astype(np.float64),
        "dist_coeffs": data["dist_coeffs"].astype(np.float64),
        "image_size": tuple(data["image_size"].astype(int)),
        "checkerboard_size": tuple(data["checkerboard_size"].astype(int)),
        "square_size": float(np.asarray(data["square_size"]).reshape(())),
    }


def make_object_points(checkerboard, square_size):
    cols, rows = checkerboard
    objp = np.zeros((rows * cols, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= square_size
    return objp


def detect_refined_corners(image, checkerboard):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, checkerboard, flags)

    # findChessboardCornersSB can segfault on macOS; classic detector only there.
    if (
        not found
        and sys.platform != "darwin"
        and hasattr(cv2, "findChessboardCornersSB")
    ):
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, checkerboard, sb_flags)

    if not found:
        return False, None

    corners = corners.astype(np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return True, corners


def normalize_imgpoints(corners):
    """OpenCV stereoCalibrate expects contiguous float32 (N, 1, 2) arrays."""
    points = np.asarray(corners, dtype=np.float32)
    if points.ndim == 2:
        points = points.reshape(-1, 1, 2)
    return np.ascontiguousarray(points)


def stereo_pair_paths():
    rgb1_paths = sorted(PAIR_DIR.glob("rgb1_*.png"))
    pairs = []
    for rgb1_path in rgb1_paths:
        pair_id = rgb1_path.stem.split("_")[1]
        rgb2_path = PAIR_DIR / f"rgb2_{pair_id}.png"
        if rgb2_path.exists():
            pairs.append((pair_id, rgb1_path, rgb2_path))
        else:
            print(f"Skipping pair {pair_id}: missing {rgb2_path}")
    return pairs


def main():
    rgb1_calib = load_calibration(RGB1_CALIBRATION_FILE)
    rgb2_calib = load_calibration(RGB2_CALIBRATION_FILE)

    checkerboard = rgb1_calib["checkerboard_size"]
    square_size = rgb1_calib["square_size"]
    if checkerboard != rgb2_calib["checkerboard_size"]:
        raise ValueError("RGB1 and RGB2 checkerboard sizes do not match.")
    if rgb1_calib["image_size"] != rgb2_calib["image_size"]:
        raise ValueError("RGB1 and RGB2 calibration image sizes do not match.")

    objp = make_object_points(checkerboard, square_size)
    objpoints = []
    imgpoints1 = []
    imgpoints2 = []
    used_pair_ids = []

    for pair_id, rgb1_path, rgb2_path in stereo_pair_paths():
        image1 = cv2.imread(str(rgb1_path))
        image2 = cv2.imread(str(rgb2_path))
        if image1 is None or image2 is None:
            print(f"Skipping pair {pair_id}: failed to load image.")
            continue

        found1, corners1 = detect_refined_corners(image1, checkerboard)
        found2, corners2 = detect_refined_corners(image2, checkerboard)
        if not (found1 and found2):
            print(f"Skipping pair {pair_id}: checkerboard not found in both images.")
            continue

        objpoints.append(objp.copy())
        imgpoints1.append(normalize_imgpoints(corners1))
        imgpoints2.append(normalize_imgpoints(corners2))
        used_pair_ids.append(pair_id)
        print(f"Using pair {pair_id}")

    if len(objpoints) < 3:
        print("Error: need at least 3 usable stereo pairs. More is better.")
        return

    K1 = rgb1_calib["camera_matrix"].astype(np.float64).copy()
    d1 = rgb1_calib["dist_coeffs"].astype(np.float64).copy()
    K2 = rgb2_calib["camera_matrix"].astype(np.float64).copy()
    d2 = rgb2_calib["dist_coeffs"].astype(np.float64).copy()
    image_size = tuple(int(v) for v in rgb1_calib["image_size"])

    print(f"Running stereoCalibrate on {len(objpoints)} pairs...")
    rms, K1, d1, K2, d2, R, t, E, F = cv2.stereoCalibrate(
        objpoints,
        imgpoints1,
        imgpoints2,
        K1,
        d1,
        K2,
        d2,
        image_size,
        flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6),
    )

    t = t.reshape(3)
    baseline_m = float(np.linalg.norm(t))

    np.savez(
        OUT_FILE,
        R_rgb1_to_rgb2=R,
        t_rgb1_to_rgb2=t,
        essential_matrix=E,
        fundamental_matrix=F,
        stereo_rms_error=np.array(rms),
        baseline_meters=np.array(baseline_m),
        used_pair_ids=np.array(used_pair_ids),
    )

    print()
    print("Stereo calibration complete.")
    print(f"Used pairs: {used_pair_ids}")
    print(f"Stereo RMS error: {rms:.4f}")
    print(f"Baseline distance: {baseline_m:.4f} m")
    print("R_rgb1_to_rgb2:")
    print(R)
    print("t_rgb1_to_rgb2 meters:")
    print(t)
    print(f"Saved {OUT_FILE.resolve()}")
    print()
    print("Reminder: RGB2 intrinsics are approximate. Calibrate RGB2 separately later for better accuracy.")


if __name__ == "__main__":
    main()
