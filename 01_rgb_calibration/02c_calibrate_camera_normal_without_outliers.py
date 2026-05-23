"""
Step 2c: run normal OpenCV calibration after removing reprojection outliers.

Run:
    python 02c_calibrate_camera_normal_without_outliers.py
    python 02c_calibrate_camera_normal_without_outliers.py --camera L --max-error 1.5

The script does two calibration passes:
1. Calibrate with every image where the checkerboard is detected.
2. Compute each image's reprojection RMSE and remove images above --max-error.
3. Recalibrate using only the kept images.

The cleaned result is saved separately by default so the original calibration is
not overwritten by accident.
"""

import argparse
import csv
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from calib_targets import prompt_camera, resolve_camera
except ModuleNotFoundError:
    # Some copies of this project use numbered folders such as
    # 03_stereo_calibration/ instead of stereo_calibration/. In that case the
    # older calib_targets.py import can fail, so keep this script runnable with
    # the same default paths.
    def resolve_camera(name):
        aliases = {"l": "rgb1", "left": "rgb1", "rgb1": "rgb1", "r": "rgb2", "right": "rgb2", "rgb2": "rgb2"}
        repo_root = Path(__file__).resolve().parents[1]
        config_dir = repo_root / "01_rgb_calibration" / "config"
        targets = {
            "rgb1": {
                "label": "L",
                "image_dir": Path("calibration_images"),
                "outlier_npz": config_dir / "camera_calibration_L_normal_no_outliers.npz",
            },
            "rgb2": {
                "label": "R",
                "image_dir": Path("calibration_images_R"),
                "outlier_npz": config_dir / "camera_calibration_R_normal_no_outliers.npz",
            },
        }
        return targets[aliases[name.lower()]]

    def prompt_camera():
        return input("Which RGB camera is this for? [L/R]: ").strip()


CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.025  # meters
DEFAULT_MAX_ERROR_PX = 1.5
MIN_IMAGES_AFTER_FILTER = 15


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calibrate RGB camera after removing high-reprojection-error images."
    )
    parser.add_argument(
        "--camera",
        choices=["L", "R", "l", "r", "left", "right", "rgb1", "rgb2"],
        default=None,
        help="Which camera to calibrate: L or R. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--max-error",
        type=float,
        default=DEFAULT_MAX_ERROR_PX,
        help="Drop images whose initial per-image reprojection RMSE is above this many pixels.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=None,
        help="Where to save the cleaned calibration .npz.",
    )
    parser.add_argument(
        "--copy-to-config",
        action="store_true",
        help="Also copy the cleaned result into config/ for stereo and LiDAR scripts.",
    )
    return parser.parse_args()


def make_object_points():
    """Return checkerboard corner locations in real-world board coordinates."""
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


def find_corners(image):
    """Detect and refine checkerboard corners, matching the normal calibration script."""
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_variants = (
        gray_original,
        cv2.equalizeHist(gray_original),
        cv2.GaussianBlur(cv2.equalizeHist(gray_original), (3, 3), 0),
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # Try OpenCV's newer detector first. It tends to be more reliable on photos.
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


def load_detected_points(image_dir):
    """Load images and keep only the ones with a detected checkerboard."""
    image_paths = sorted(image_dir.glob("*.png")) + sorted(image_dir.glob("*.jpg"))
    image_paths += sorted(image_dir.glob("*.jpeg"))
    image_paths += sorted(image_dir.glob("*.bmp"))

    obj_template = make_object_points()
    objpoints = []
    imgpoints = []
    kept_paths = []
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
            kept_paths.append(path)
            print(f"Detected checkerboard: {path}")
        else:
            print(f"Skipped, checkerboard not found: {path}")

    return kept_paths, objpoints, imgpoints, image_size


def per_image_reprojection_errors(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    """Return one reprojection RMSE value per calibration image."""
    errors = []
    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        point_errors = np.linalg.norm(imgp.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        errors.append(float(np.sqrt(np.mean(point_errors * point_errors))))
    return errors


def mean_reprojection_error(objpoints, imgpoints, rvecs, tvecs, camera_matrix, dist_coeffs):
    """Compute global RMSE across all detected checkerboard corners."""
    total_error = 0.0
    total_points = 0

    for objp, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
        error = cv2.norm(imgp, projected, cv2.NORM_L2)
        total_error += error * error
        total_points += len(objp)

    return float(np.sqrt(total_error / total_points))


def calibrate(objpoints, imgpoints, image_size):
    """Run OpenCV pinhole calibration and return RMS, intrinsics, poses, and global RMSE."""
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
    return rms, camera_matrix, dist_coeffs, rvecs, tvecs, mean_error


def save_outlier_report(path, image_paths, initial_errors, kept_mask):
    """Write a small CSV so it is clear which images were removed."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "initial_reprojection_error_px", "kept"])
        writer.writeheader()
        for image_path, error, kept in zip(image_paths, initial_errors, kept_mask):
            writer.writerow(
                {
                    "image": image_path.name,
                    "initial_reprojection_error_px": error,
                    "kept": kept,
                }
            )


def main():
    args = parse_args()
    camera_name = args.camera if args.camera is not None else prompt_camera()
    target = resolve_camera(camera_name)
    image_dir = target["image_dir"]
    final_config_file = target["outlier_npz"]
    output_file = args.output_file
    if output_file is None:
        output_file = final_config_file
    output_file.parent.mkdir(parents=True, exist_ok=True)

    image_paths, objpoints, imgpoints, image_size = load_detected_points(image_dir)
    if not image_paths:
        print(f"Error: no usable checkerboard images found in {image_dir.resolve()}")
        return

    if len(objpoints) < MIN_IMAGES_AFTER_FILTER:
        print(
            f"Error: only {len(objpoints)} usable images before filtering. "
            f"Capture at least {MIN_IMAGES_AFTER_FILTER}; 20-30 is better."
        )
        return

    print(f"\nInitial calibration with {len(objpoints)} detected images...")
    initial = calibrate(objpoints, imgpoints, image_size)
    initial_rms, initial_camera_matrix, initial_dist_coeffs, initial_rvecs, initial_tvecs, initial_mean = initial
    initial_errors = per_image_reprojection_errors(
        objpoints,
        imgpoints,
        initial_rvecs,
        initial_tvecs,
        initial_camera_matrix,
        initial_dist_coeffs,
    )

    kept_mask = [error <= args.max_error for error in initial_errors]
    kept_count = sum(kept_mask)
    removed_count = len(kept_mask) - kept_count

    print(f"Initial RMS reprojection error: {initial_rms:.6f}")
    print(f"Initial mean reprojection error: {initial_mean:.6f}")
    print(f"Outlier cutoff: {args.max_error:.3f} px")
    print(f"Keeping {kept_count} images; removing {removed_count} images.")

    for image_path, error, kept in zip(image_paths, initial_errors, kept_mask):
        status = "keep" if kept else "drop"
        print(f"{status:>4}  {error:.3f} px  {image_path.name}")

    if kept_count < MIN_IMAGES_AFTER_FILTER:
        print(
            f"\nError: filtering would leave only {kept_count} images. "
            f"Raise --max-error or capture more images."
        )
        return

    filtered_objpoints = [objp for objp, kept in zip(objpoints, kept_mask) if kept]
    filtered_imgpoints = [imgp for imgp, kept in zip(imgpoints, kept_mask) if kept]
    kept_names = [path.name for path, kept in zip(image_paths, kept_mask) if kept]
    removed_names = [path.name for path, kept in zip(image_paths, kept_mask) if not kept]

    print(f"\nFinal calibration with {kept_count} kept images...")
    final = calibrate(filtered_objpoints, filtered_imgpoints, image_size)
    final_rms, camera_matrix, dist_coeffs, rvecs, tvecs, final_mean = final
    final_errors = per_image_reprojection_errors(
        filtered_objpoints,
        filtered_imgpoints,
        rvecs,
        tvecs,
        camera_matrix,
        dist_coeffs,
    )

    report_file = Path("outputs") / f"calibration_{target['label']}_outlier_filter_report.csv"
    save_outlier_report(report_file, image_paths, initial_errors, kept_mask)

    np.savez(
        output_file,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        image_size=np.array(image_size),
        checkerboard_size=np.array(CHECKERBOARD),
        square_size=np.array(SQUARE_SIZE),
        rms_error=np.array(final_rms),
        mean_reprojection_error=np.array(final_mean),
        max_outlier_error_px=np.array(args.max_error),
        initial_rms_error=np.array(initial_rms),
        initial_mean_reprojection_error=np.array(initial_mean),
        initial_per_image_reprojection_errors=np.array(initial_errors),
        final_per_image_reprojection_errors=np.array(final_errors),
        kept_image_names=np.array(kept_names),
        removed_image_names=np.array(removed_names),
    )

    print("\nOutlier-filtered normal calibration complete.")
    print(f"Initial RMS reprojection error: {initial_rms:.6f}")
    print(f"Final RMS reprojection error: {final_rms:.6f}")
    print(f"Initial mean reprojection error: {initial_mean:.6f}")
    print(f"Final mean reprojection error: {final_mean:.6f}")
    print(f"Removed images: {removed_names if removed_names else 'none'}")
    print("Camera matrix:")
    print(camera_matrix)
    print("Distortion coefficients:")
    print(dist_coeffs)
    print(f"Saved {output_file.resolve()}")
    print(f"Saved outlier report {report_file.resolve()}")

    if args.copy_to_config:
        final_config_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.resolve() != final_config_file.resolve():
            shutil.copy2(output_file, final_config_file)
            print(f"Copied cleaned calibration to {final_config_file.resolve()}")
        else:
            print(f"Cleaned calibration is already at {final_config_file.resolve()}")
    else:
        print(f"Final path from config.yaml: {final_config_file.resolve()}")


if __name__ == "__main__":
    main()
