"""Calibrate L-to-R fisheye stereo extrinsics with fixed intrinsics.

This is the main stereo calibration script for the current workflow. It reads
paired checkerboard captures from 00_data_capture/int_ext_calib_rgb/L and R,
keeps only pairs where all internal checkerboard corners are detected in both
images, and runs cv2.fisheye.stereoCalibrate with fixed L/R intrinsics.

Run from this folder:
    python 02b_stereo_calibrate_fisheye_fixed_intrinsics.py
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.project_config import calibration_file


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate fisheye stereo extrinsics with fixed intrinsics.")
    parser.add_argument("--capture-dir", type=Path, default=REPO_ROOT / "00_data_capture" / "int_ext_calib_rgb")
    parser.add_argument("--left-dir", type=Path, default=None)
    parser.add_argument("--right-dir", type=Path, default=None)
    parser.add_argument("--left-calibration", type=Path, default=calibration_file("left_fisheye_intrinsics_no_outliers"))
    parser.add_argument("--right-calibration", type=Path, default=calibration_file("right_fisheye_intrinsics_no_outliers"))
    parser.add_argument("--output-file", type=Path, default=calibration_file("stereo_rgb1_rgb2_extrinsics"))
    parser.add_argument("--pair-ids", nargs="+", default=None, help="Optional pair ids to use, for example: 013 018 019.")
    return parser.parse_args()


def load_fisheye_intrinsics(path):
    if not path.exists():
        raise FileNotFoundError(f"Missing calibration file: {path}")
    data = np.load(path)
    return {
        "camera_matrix": data["camera_matrix"].astype(np.float64),
        "dist_coeffs": data["dist_coeffs"].reshape(-1, 1).astype(np.float64),
        "image_size": tuple(int(v) for v in data["image_size"].astype(int)),
        "checkerboard_size": tuple(int(v) for v in data["checkerboard_size"].astype(int)),
        "square_size": float(np.asarray(data["square_size"]).reshape(())),
    }


def make_object_points(checkerboard, square_size):
    cols, rows = checkerboard
    objp = np.zeros((1, rows * cols, 3), np.float64)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_size
    return objp


def detect_refined_corners(image, checkerboard):
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants = (
        gray_original,
        cv2.equalizeHist(gray_original),
        cv2.GaussianBlur(cv2.equalizeHist(gray_original), (3, 3), 0),
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        for gray in variants:
            found, corners = cv2.findChessboardCornersSB(gray, checkerboard, sb_flags)
            if found:
                corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria)
                return True, corners.reshape(1, -1, 2).astype(np.float64)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for gray in variants:
        found, corners = cv2.findChessboardCorners(gray, checkerboard, flags)
        if found:
            corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria)
            return True, corners.reshape(1, -1, 2).astype(np.float64)

    return False, None


def image_map(folder, prefixes):
    mapping = {}
    for prefix in prefixes:
        for path in sorted(folder.glob(f"{prefix}_*.png")):
            parts = path.stem.split("_", 1)
            if len(parts) == 2:
                mapping[parts[1]] = path
    return mapping


def pair_paths(left_dir, right_dir, requested_ids=None):
    left_images = image_map(left_dir, ("calib", "rgb1"))
    right_images = image_map(right_dir, ("calib", "rgb2"))
    ids = sorted(set(left_images).intersection(right_images))
    if requested_ids is not None:
        requested = {pair_id.zfill(3) for pair_id in requested_ids}
        ids = [pair_id for pair_id in ids if pair_id in requested]
    return [(pair_id, left_images[pair_id], right_images[pair_id]) for pair_id in ids]


def skew(vector):
    x, y, z = vector.reshape(3)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def fundamental_from_extrinsics(K1, K2, R, t):
    return np.linalg.inv(K2).T @ skew(t) @ R @ np.linalg.inv(K1)


def main():
    args = parse_args()
    left_dir = args.left_dir if args.left_dir is not None else args.capture_dir / "L"
    right_dir = args.right_dir if args.right_dir is not None else args.capture_dir / "R"

    left_calib = load_fisheye_intrinsics(args.left_calibration)
    right_calib = load_fisheye_intrinsics(args.right_calibration)
    if left_calib["checkerboard_size"] != right_calib["checkerboard_size"]:
        raise ValueError("L/R checkerboard sizes do not match.")
    if left_calib["image_size"] != right_calib["image_size"]:
        raise ValueError("L/R image sizes do not match.")

    checkerboard = left_calib["checkerboard_size"]
    image_size = left_calib["image_size"]
    obj_template = make_object_points(checkerboard, left_calib["square_size"])

    objpoints = []
    imgpoints_left = []
    imgpoints_right = []
    used_pair_ids = []
    skipped = []

    pairs = pair_paths(left_dir, right_dir, args.pair_ids)
    if not pairs:
        raise RuntimeError(f"No matching L/R pairs found in {left_dir} and {right_dir}")

    for pair_id, left_path, right_path in pairs:
        left = cv2.imread(str(left_path))
        right = cv2.imread(str(right_path))
        if left is None or right is None:
            skipped.append((pair_id, "unreadable image"))
            continue

        found_left, corners_left = detect_refined_corners(left, checkerboard)
        found_right, corners_right = detect_refined_corners(right, checkerboard)
        if not (found_left and found_right):
            skipped.append((pair_id, f"checkerboard missing: L={found_left}, R={found_right}"))
            continue

        objpoints.append(obj_template.copy())
        imgpoints_left.append(corners_left)
        imgpoints_right.append(corners_right)
        used_pair_ids.append(pair_id)
        print(f"Using clean full-corner pair {pair_id}")

    if len(objpoints) < 3:
        raise RuntimeError(f"Need at least 3 clean stereo pairs; found {len(objpoints)}")

    K1 = left_calib["camera_matrix"].copy()
    D1 = left_calib["dist_coeffs"].copy()
    K2 = right_calib["camera_matrix"].copy()
    D2 = right_calib["dist_coeffs"].copy()
    R = np.eye(3, dtype=np.float64)
    T = np.zeros((3, 1), dtype=np.float64)
    flags = cv2.fisheye.CALIB_FIX_INTRINSIC + cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7)

    print(f"Running cv2.fisheye.stereoCalibrate on {len(objpoints)} clean pairs...")
    rms, K1, D1, K2, D2, R, T = cv2.fisheye.stereoCalibrate(
        objpoints,
        imgpoints_left,
        imgpoints_right,
        K1,
        D1,
        K2,
        D2,
        image_size,
        R,
        T,
        flags,
        criteria,
    )

    baseline_m = float(np.linalg.norm(T))
    E = skew(T) @ R
    F = fundamental_from_extrinsics(K1, K2, R, T)

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_file,
        R_rgb1_to_rgb2=R,
        t_rgb1_to_rgb2=T.reshape(3),
        essential_matrix=E,
        fundamental_matrix=F,
        stereo_rms_error=np.array(rms),
        baseline_meters=np.array(baseline_m),
        used_pair_ids=np.array(used_pair_ids),
        skipped_pair_ids=np.array([pair_id for pair_id, _ in skipped]),
        model=np.array("fisheye"),
        left_calibration=np.array(str(args.left_calibration)),
        right_calibration=np.array(str(args.right_calibration)),
        left_image_dir=np.array(str(left_dir)),
        right_image_dir=np.array(str(right_dir)),
    )

    print()
    print("Fisheye stereo calibration complete.")
    print(f"Used clean full-corner pairs: {len(used_pair_ids)}/{len(pairs)}")
    print(f"Stereo RMS: {rms:.4f} px")
    print(f"Baseline: {baseline_m:.4f} m")
    print(f"Saved {args.output_file.resolve()}")
    if skipped:
        print("Skipped pairs:")
        for pair_id, reason in skipped:
            print(f"- {pair_id}: {reason}")


if __name__ == "__main__":
    main()
