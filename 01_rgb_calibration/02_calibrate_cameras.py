"""
Run all RGB calibration variants for the L and R checkerboard datasets.

By default this script runs both cameras and saves four calibrations for each:

1. Normal pinhole calibration
2. Fisheye calibration
3. Normal pinhole calibration after reprojection-error outlier filtering
4. Fisheye calibration using the same kept images from the normal outlier pass

Run:
    python 02_calibrate_cameras.py
    python 02_calibrate_cameras.py --camera L
    python 02_calibrate_cameras.py --camera R --max-error 1.5
"""

import argparse
import csv
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[0]
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from calib_targets import resolve_camera


CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.025  # meters
DEFAULT_MAX_ERROR_PX = 1.5
MIN_NORMAL_IMAGES = 5
MIN_FILTERED_IMAGES = 15
MIN_FISHEYE_IMAGES = 10
IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")


def parse_args():
    parser = argparse.ArgumentParser(description="Run all RGB calibration variants for L and R datasets.")
    parser.add_argument(
        "--camera",
        choices=["L", "R", "l", "r", "left", "right", "rgb1", "rgb2"],
        action="append",
        default=None,
        help="Camera to calibrate. Omit to run both L and R. May be passed more than once.",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=DEFAULT_MAX_ERROR_PX,
        help="Drop images whose initial normal per-image reprojection RMSE is above this many pixels.",
    )
    parser.add_argument(
        "--summary-csv",
        type=Path,
        default=_SCRIPT_DIR / "outputs" / "calibration_results.csv",
        help="Where to save the per-camera, per-variant calibration summary CSV.",
    )
    return parser.parse_args()


def make_normal_object_points():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


def make_fisheye_object_points():
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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        for gray in gray_variants:
            found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, sb_flags)
            if found:
                corners = corners.astype(np.float32)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                return True, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for gray in gray_variants:
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners

    return False, None


def list_image_paths(image_dir):
    paths = []
    for extension in IMAGE_EXTENSIONS:
        paths.extend(sorted(image_dir.glob(extension)))
    return paths


def load_detected_points(image_dir):
    normal_template = make_normal_object_points()
    fisheye_template = make_fisheye_object_points()
    image_paths = list_image_paths(image_dir)
    records = []
    image_size = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"Skipping unreadable image: {path}")
            continue

        image_size = (image.shape[1], image.shape[0])
        found, corners = find_corners(image)
        if not found:
            print(f"Skipped, checkerboard not found: {path}")
            continue

        records.append(
            {
                "path": path,
                "normal_objpoints": normal_template.copy(),
                "normal_imgpoints": corners.astype(np.float32),
                "fisheye_objpoints": fisheye_template.copy(),
                "fisheye_imgpoints": corners.reshape(1, -1, 2).astype(np.float64),
            }
        )
        print(f"Detected checkerboard: {path}")

    return records, image_size, len(image_paths)


def normal_points(records):
    return (
        [record["normal_objpoints"] for record in records],
        [record["normal_imgpoints"] for record in records],
    )


def fisheye_points(records):
    return (
        [record["fisheye_objpoints"] for record in records],
        [record["fisheye_imgpoints"] for record in records],
    )


def normal_mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0.0
    total_points = 0
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected, cv2.NORM_L2)
        total_error += error * error
        total_points += len(objp)
    return float(np.sqrt(total_error / total_points))


def normal_per_image_errors(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    errors = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        point_errors = np.linalg.norm(imgp.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        errors.append(float(np.sqrt(np.mean(point_errors * point_errors))))
    return errors


def fisheye_mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    total_error = 0.0
    total_points = 0
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.fisheye.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        point_errors = np.linalg.norm(imgp.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        total_error += float(np.sum(point_errors * point_errors))
        total_points += point_errors.size
    return float(np.sqrt(total_error / total_points))


def calibrate_normal(records, image_size):
    objpoints, imgpoints = normal_points(records)
    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    mean_error = normal_mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)
    per_image_errors = normal_per_image_errors(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs)
    return {
        "rms": float(rms),
        "camera_matrix": camera_matrix,
        "dist_coeffs": dist_coeffs,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "mean_error": mean_error,
        "per_image_errors": per_image_errors,
    }


def calibrate_fisheye(records, image_size):
    objpoints, imgpoints = fisheye_points(records)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for _ in objpoints]
    flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND + cv2.fisheye.CALIB_FIX_SKEW
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
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
    mean_error = fisheye_mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, D)
    return {
        "rms": float(rms),
        "camera_matrix": K,
        "dist_coeffs": D,
        "rvecs": rvecs,
        "tvecs": tvecs,
        "mean_error": mean_error,
    }


def save_normal(path, result, image_size, image_names, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "camera_matrix": result["camera_matrix"],
        "dist_coeffs": result["dist_coeffs"],
        "image_size": np.array(image_size),
        "checkerboard_size": np.array(CHECKERBOARD),
        "square_size": np.array(SQUARE_SIZE),
        "rms_error": np.array(result["rms"]),
        "mean_reprojection_error": np.array(result["mean_error"]),
        "used_image_names": np.array(image_names),
    }
    if extra:
        payload.update(extra)
    np.savez(path, **payload)


def save_fisheye(path, result, image_size, image_names, extra=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "K": result["camera_matrix"],
        "D": result["dist_coeffs"],
        "camera_matrix": result["camera_matrix"],
        "dist_coeffs": result["dist_coeffs"],
        "image_size": np.array(image_size),
        "checkerboard_size": np.array(CHECKERBOARD),
        "square_size": np.array(SQUARE_SIZE),
        "rms_error": np.array(result["rms"]),
        "mean_reprojection_error": np.array(result["mean_error"]),
        "used_image_names": np.array(image_names),
    }
    if extra:
        payload.update(extra)
    np.savez(path, **payload)


def save_outlier_report(path, records, initial_errors, kept_mask):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "initial_reprojection_error_px", "kept"])
        writer.writeheader()
        for record, error, kept in zip(records, initial_errors, kept_mask):
            writer.writerow(
                {
                    "image": record["path"].name,
                    "initial_reprojection_error_px": error,
                    "kept": kept,
                }
            )


def success_row(camera, variation, model, output_file, detected_count, used_count, result, extra=None):
    row = {
        "camera": camera,
        "variation": variation,
        "model": model,
        "status": "ok",
        "output_file": str(output_file),
        "detected_images": detected_count,
        "used_images": used_count,
        "removed_images": detected_count - used_count,
        "rms_error": result["rms"],
        "mean_reprojection_error": result["mean_error"],
        "initial_rms_error": "",
        "initial_mean_reprojection_error": "",
        "max_outlier_error_px": "",
        "error": "",
    }
    if extra:
        row.update(extra)
    return row


def error_row(camera, variation, model, output_file, detected_count, message):
    return {
        "camera": camera,
        "variation": variation,
        "model": model,
        "status": "error",
        "output_file": str(output_file) if output_file else "",
        "detected_images": detected_count,
        "used_images": "",
        "removed_images": "",
        "rms_error": "",
        "mean_reprojection_error": "",
        "initial_rms_error": "",
        "initial_mean_reprojection_error": "",
        "max_outlier_error_px": "",
        "error": message,
    }


def run_camera(camera_name, max_error):
    target = resolve_camera(camera_name)
    label = target["label"]
    print(f"\n=== Calibrating camera {label} ===")
    print(f"Dataset: {target['image_dir'].resolve()}")

    records, image_size, total_images = load_detected_points(target["image_dir"])
    detected_count = len(records)
    rows = []

    if total_images == 0:
        message = f"No images found in {target['image_dir'].resolve()}"
        print(f"Error: {message}")
        for variation, model, key in (
            ("normal", "pinhole", "normal_npz"),
            ("fisheye", "fisheye", "fisheye_npz"),
            ("normal_no_outliers", "pinhole", "outlier_npz"),
            ("fisheye_no_outliers", "fisheye", "fisheye_outlier_npz"),
        ):
            rows.append(error_row(label, variation, model, target[key], 0, message))
        return rows

    if detected_count < MIN_NORMAL_IMAGES:
        message = f"Only {detected_count} usable images; need at least {MIN_NORMAL_IMAGES}."
        print(f"Error: {message}")
        for variation, model, key in (
            ("normal", "pinhole", "normal_npz"),
            ("normal_no_outliers", "pinhole", "outlier_npz"),
        ):
            rows.append(error_row(label, variation, model, target[key], detected_count, message))
    else:
        normal = calibrate_normal(records, image_size)
        image_names = [record["path"].name for record in records]
        save_normal(target["normal_npz"], normal, image_size, image_names)
        print(f"Saved normal calibration: {target['normal_npz'].resolve()}")
        rows.append(success_row(label, "normal", "pinhole", target["normal_npz"], detected_count, detected_count, normal))

        kept_mask = [error <= max_error for error in normal["per_image_errors"]]
        kept_records = [record for record, kept in zip(records, kept_mask) if kept]
        removed_names = [record["path"].name for record, kept in zip(records, kept_mask) if not kept]
        kept_names = [record["path"].name for record in kept_records]
        report_file = _SCRIPT_DIR / "outputs" / label / "outlier_filter_report.csv"
        legacy_report_file = _SCRIPT_DIR / "outputs" / f"calibration_{label}_outlier_filter_report.csv"
        save_outlier_report(report_file, records, normal["per_image_errors"], kept_mask)
        save_outlier_report(legacy_report_file, records, normal["per_image_errors"], kept_mask)
        print(f"Saved outlier report: {report_file.resolve()}")

        if len(kept_records) < MIN_FILTERED_IMAGES:
            message = (
                f"Outlier filtering would leave {len(kept_records)} images; "
                f"need at least {MIN_FILTERED_IMAGES}."
            )
            print(f"Error: {message}")
            rows.append(error_row(label, "normal_no_outliers", "pinhole", target["outlier_npz"], detected_count, message))
        else:
            normal_filtered = calibrate_normal(kept_records, image_size)
            save_normal(
                target["outlier_npz"],
                normal_filtered,
                image_size,
                kept_names,
                {
                    "max_outlier_error_px": np.array(max_error),
                    "initial_rms_error": np.array(normal["rms"]),
                    "initial_mean_reprojection_error": np.array(normal["mean_error"]),
                    "initial_per_image_reprojection_errors": np.array(normal["per_image_errors"]),
                    "final_per_image_reprojection_errors": np.array(normal_filtered["per_image_errors"]),
                    "kept_image_names": np.array(kept_names),
                    "removed_image_names": np.array(removed_names),
                },
            )
            print(f"Saved normal no-outliers calibration: {target['outlier_npz'].resolve()}")
            rows.append(
                success_row(
                    label,
                    "normal_no_outliers",
                    "pinhole",
                    target["outlier_npz"],
                    detected_count,
                    len(kept_records),
                    normal_filtered,
                    {
                        "initial_rms_error": normal["rms"],
                        "initial_mean_reprojection_error": normal["mean_error"],
                        "max_outlier_error_px": max_error,
                    },
                )
            )

    if detected_count < MIN_FISHEYE_IMAGES:
        message = f"Only {detected_count} usable images; fisheye needs at least {MIN_FISHEYE_IMAGES}."
        print(f"Error: {message}")
        rows.append(error_row(label, "fisheye", "fisheye", target["fisheye_npz"], detected_count, message))
    else:
        try:
            fisheye = calibrate_fisheye(records, image_size)
            image_names = [record["path"].name for record in records]
            save_fisheye(target["fisheye_npz"], fisheye, image_size, image_names)
            print(f"Saved fisheye calibration: {target['fisheye_npz'].resolve()}")
            rows.append(success_row(label, "fisheye", "fisheye", target["fisheye_npz"], detected_count, detected_count, fisheye))
        except cv2.error as exc:
            message = str(exc).replace("\n", " ")
            print(f"Fisheye calibration failed: {message}")
            rows.append(error_row(label, "fisheye", "fisheye", target["fisheye_npz"], detected_count, message))

    kept_records = None
    kept_names = None
    if detected_count >= MIN_NORMAL_IMAGES:
        kept_mask = [error <= max_error for error in normal["per_image_errors"]]
        kept_records = [record for record, kept in zip(records, kept_mask) if kept]
        kept_names = [record["path"].name for record in kept_records]

    if not kept_records or len(kept_records) < MIN_FISHEYE_IMAGES:
        kept_count = 0 if kept_records is None else len(kept_records)
        message = f"Only {kept_count} kept images; fisheye without outliers needs at least {MIN_FISHEYE_IMAGES}."
        print(f"Error: {message}")
        rows.append(error_row(label, "fisheye_no_outliers", "fisheye", target["fisheye_outlier_npz"], detected_count, message))
    else:
        try:
            fisheye_filtered = calibrate_fisheye(kept_records, image_size)
            save_fisheye(
                target["fisheye_outlier_npz"],
                fisheye_filtered,
                image_size,
                kept_names,
                {
                    "max_outlier_error_px": np.array(max_error),
                    "source_outlier_report": np.array(str(_SCRIPT_DIR / "outputs" / label / "outlier_filter_report.csv")),
                },
            )
            print(f"Saved fisheye no-outliers calibration: {target['fisheye_outlier_npz'].resolve()}")
            rows.append(
                success_row(
                    label,
                    "fisheye_no_outliers",
                    "fisheye",
                    target["fisheye_outlier_npz"],
                    detected_count,
                    len(kept_records),
                    fisheye_filtered,
                    {"max_outlier_error_px": max_error},
                )
            )
        except cv2.error as exc:
            message = str(exc).replace("\n", " ")
            print(f"Fisheye without outliers failed: {message}")
            rows.append(error_row(label, "fisheye_no_outliers", "fisheye", target["fisheye_outlier_npz"], detected_count, message))

    return rows


def write_summary_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "camera",
        "variation",
        "model",
        "status",
        "output_file",
        "detected_images",
        "used_images",
        "removed_images",
        "rms_error",
        "mean_reprojection_error",
        "initial_rms_error",
        "initial_mean_reprojection_error",
        "max_outlier_error_px",
        "error",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    args = parse_args()
    cameras = args.camera if args.camera else ["L", "R"]
    rows = []
    for camera in cameras:
        rows.extend(run_camera(camera, args.max_error))

    write_summary_csv(args.summary_csv, rows)
    print(f"\nSaved calibration summary: {args.summary_csv.resolve()}")

    ok_count = sum(1 for row in rows if row["status"] == "ok")
    print(f"Finished {ok_count}/{len(rows)} calibration variants successfully.")


if __name__ == "__main__":
    main()
