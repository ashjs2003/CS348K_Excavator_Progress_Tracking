"""
Step 2d: run OpenCV fisheye calibration after removing outlier images.

Run:
    python 02d_calibrate_camera_fisheye_without_outliers.py

By default this reads outputs/calibration_L_outlier_filter_report.csv or
outputs/calibration_R_outlier_filter_report.csv from the normal outlier-filtering
pass and uses only rows marked kept=True.
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from calib_targets import prompt_camera, resolve_camera


CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.025  # meters


def parse_args():
    parser = argparse.ArgumentParser(description="Calibrate fisheye model without outlier images.")
    parser.add_argument(
        "--camera",
        choices=["L", "R", "l", "r", "left", "right", "rgb1", "rgb2"],
        default=None,
        help="Which camera to calibrate: L or R. If omitted, you will be prompted.",
    )
    parser.add_argument("--image-dir", type=Path, default=None)
    parser.add_argument("--outlier-report", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    return parser.parse_args()


def load_kept_names(report_path):
    """Read image names where the normal outlier pass marked kept=True."""
    if not report_path.exists():
        raise FileNotFoundError(f"Missing {report_path}. Run 02c first.")

    kept = []
    with report_path.open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row.get("kept", "").strip().lower() == "true":
                kept.append(row["image"])
    return kept


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

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        for gray in gray_variants:
            found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, sb_flags)
            if found:
                return True, corners.reshape(1, -1, 2).astype(np.float64)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
    for gray in gray_variants:
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners.reshape(1, -1, 2).astype(np.float64)

    return False, None


def main():
    args = parse_args()
    camera_name = args.camera if args.camera is not None else prompt_camera()
    target = resolve_camera(camera_name)
    image_dir = args.image_dir if args.image_dir is not None else target["image_dir"]
    outlier_report = args.outlier_report
    if outlier_report is None:
        outlier_report = Path("outputs") / f"calibration_{target['label']}_outlier_filter_report.csv"
    output_file = args.output_file if args.output_file is not None else target["fisheye_outlier_npz"]

    kept_names = load_kept_names(outlier_report)
    if len(kept_names) < 10:
        print(f"Error: only {len(kept_names)} kept images. Fisheye calibration needs more variety.")
        return

    obj_template = make_object_points()
    objpoints = []
    imgpoints = []
    used_names = []
    image_size = None

    for name in kept_names:
        path = image_dir / name
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue

        image_size = (image.shape[1], image.shape[0])
        found, corners = find_corners(image)
        if not found:
            print(f"Skipped, checkerboard not found: {path}")
            continue

        objpoints.append(obj_template.copy())
        imgpoints.append(corners)
        used_names.append(name)
        print(f"Using {path}")

    if len(objpoints) < 10:
        print(f"Error: only {len(objpoints)} usable kept images.")
        return

    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
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
        print("Fisheye calibration without outliers failed.")
        print(exc)
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
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
        used_image_names=np.array(used_names),
        source_outlier_report=np.array(str(outlier_report)),
    )

    print(f"\nFisheye calibration without outliers complete for {target['label']}.")
    print(f"Used images: {len(used_names)}")
    print(f"RMS error: {rms}")
    print("Fisheye K matrix:")
    print(K)
    print("Fisheye D coefficients:")
    print(D)
    print(f"Saved {output_file.resolve()}")


if __name__ == "__main__":
    main()
