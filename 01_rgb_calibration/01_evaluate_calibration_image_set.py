"""
Step 0: evaluate checkerboard calibration image quality before calibration.

Run from this folder:
    python 00_evaluate_calibration_image_set.py
    python 00_evaluate_calibration_image_set.py --camera L
    python 00_evaluate_calibration_image_set.py --camera R

This script checks whether the L/R calibration images are numerous,
sharp, well distributed across the frame, and diverse in size/pose.
If the configured L/R calibration exists, it also reports solvePnP pose and
per-image reprojection errors using the saved camera intrinsics.
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np


matplotlib.use("Agg")  # Save plots without opening GUI windows.
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from calib_targets import prompt_camera, resolve_camera

CHECKERBOARD = (9, 6)  # OpenCV inner corners: columns, rows.
SQUARE_SIZE = 0.025  # meters.
GRID_SIZE = 5  # 5x5 coverage grid over the image.

MIN_VALID_IMAGES = 15
MIN_GRID_COVERAGE_PERCENT = 50.0
MIN_AREA_RATIO_RANGE = 0.08
MIN_POSE_RANGE_DEGREES = 20.0
BLURRY_LAPLACIAN_THRESHOLD = 100.0
HIGH_REPROJECTION_ERROR_PX = 1.0



def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OpenCV checkerboard calibration images."
    )
    parser.add_argument(
        "--camera",
        choices=["L", "R", "l", "r", "left", "right", "rgb1", "rgb2"],
        default=None,
        help="Which camera image set to evaluate: L or R. If omitted, you will be prompted.",
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=None,
        help="Override the folder containing calibration images.",
    )
    parser.add_argument(
        "--calibration-file",
        type=Path,
        default=None,
        help="Override the optional .npz with camera_matrix and dist_coeffs.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(_REPO_ROOT / "01_rgb_calibration"/ "outputs"),
        help="Folder where reports and plots are written.",
    )
    return parser.parse_args()


def default_calibration_file(target):
    """Prefer the cleaned calibration if it exists, otherwise use the baseline normal calibration."""
    return target["outlier_npz"] if target["outlier_npz"].exists() else target["normal_npz"]


def make_object_points():
    """Return the checkerboard corner coordinates in board-local meters."""
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


def load_image_paths(image_dir):
    """Collect common image file types without failing if one type is absent."""
    extensions = ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff")
    paths = []
    for extension in extensions:
        paths.extend(sorted(image_dir.glob(extension)))
    return sorted(set(paths))


def detect_checkerboard(image):
    """Detect and refine checkerboard corners in a few grayscale variants."""
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_variants = (
        gray_original,
        cv2.equalizeHist(gray_original),
        cv2.GaussianBlur(cv2.equalizeHist(gray_original), (3, 3), 0),
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # The SB detector is usually more robust on real calibration photos.
    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        for gray in gray_variants:
            found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, sb_flags)
            if found:
                corners = corners.astype(np.float32)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                return True, corners, gray_original

    # Fall back to the classic detector for OpenCV builds/images where SB misses.
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for gray in gray_variants:
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners, gray_original

    return False, None, gray_original


def corner_bbox(corners):
    """Return min/max pixel bounds and the normalized bbox area."""
    pts = corners.reshape(-1, 2)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    width_height = max_xy - min_xy
    return min_xy, max_xy, float(width_height[0] * width_height[1])


def grid_cells_for_corners(corners, image_width, image_height):
    """Return the set of 5x5 image-grid cells touched by detected corners."""
    cells = set()
    for x, y in corners.reshape(-1, 2):
        col = min(GRID_SIZE - 1, max(0, int(x / image_width * GRID_SIZE)))
        row = min(GRID_SIZE - 1, max(0, int(y / image_height * GRID_SIZE)))
        cells.add((row, col))
    return cells


def rotation_vector_to_euler_degrees(rvec):
    """
    Convert an OpenCV rotation vector to approximate pitch/yaw/roll degrees.

    The values are most useful as diversity indicators, not as a rigid
    robotics convention. They are derived from the board-to-camera rotation.
    """
    rotation_matrix, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = math.atan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = math.atan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
    else:
        roll = math.atan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        pitch = math.atan2(-rotation_matrix[2, 0], sy)
        yaw = 0.0

    return {
        "pitch_deg": float(math.degrees(pitch)),
        "yaw_deg": float(math.degrees(yaw)),
        "roll_deg": float(math.degrees(roll)),
    }


def reprojection_rmse(objp, corners, rvec, tvec, camera_matrix, dist_coeffs):
    """Compute per-corner reprojection RMSE in pixels for one image."""
    projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    errors = np.linalg.norm(corners.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(errors * errors)))


def safe_summary(values):
    """Return common stats for a list, or None values when empty."""
    if not values:
        return {
            "min": None,
            "max": None,
            "mean": None,
            "std": None,
            "median": None,
            "rmse": None,
            "p90": None,
        }

    arr = np.array(values, dtype=float)
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "p90": float(np.percentile(arr, 90)),
    }


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(path, rows):
    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def draw_text(image, text, origin, scale=0.55, color=(255, 255, 255), thickness=1):
    """Draw readable text with a dark outline."""
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def save_heatmap(path, heatmap_counts):
    """Save a matplotlib heatmap of 5x5 image-grid coverage."""
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    image = ax.imshow(heatmap_counts, cmap="viridis", interpolation="nearest")

    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            count = int(heatmap_counts[row, col])
            ax.text(col, row, str(count), ha="center", va="center", color="white", fontsize=11)

    ax.set_title("Checkerboard corner coverage")
    ax.set_xlabel("Image grid column")
    ax.set_ylabel("Image grid row")
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    fig.colorbar(image, ax=ax, label="Detected corners in cell")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_line_plot(path, values, title, y_label):
    """Save a matplotlib line plot."""
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax.set_title(title)
    ax.set_xlabel("Valid image index")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)

    if values:
        arr = np.array(values, dtype=float)
        ax.plot(range(len(arr)), arr, marker="o", linewidth=1.8)
        ax.axhline(HIGH_REPROJECTION_ERROR_PX, color="tab:red", linestyle="--", linewidth=1.2)
        ax.text(
            0.01,
            0.98,
            f"mean={np.mean(arr):.3f}, median={np.median(arr):.3f}, max={np.max(arr):.3f}",
            transform=ax.transAxes,
            ha="left",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "0.8", "alpha": 0.9},
        )
    else:
        ax.text(0.5, 0.5, "No values available", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_histogram(path, values, title, x_label):
    """Save a matplotlib histogram."""
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Image count")
    ax.grid(True, axis="y", alpha=0.3)

    if values:
        arr = np.array(values, dtype=float)
        bins = min(12, max(4, int(math.sqrt(len(arr)))))
        ax.hist(arr, bins=bins, color="tab:green", edgecolor="black", alpha=0.8)
        ax.axvline(np.mean(arr), color="tab:blue", linestyle="--", linewidth=1.3, label="mean")
        ax.axvline(np.median(arr), color="tab:orange", linestyle=":", linewidth=1.8, label="median")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No values available", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_contact_sheet(path, valid_items):
    """Draw detected corners and labels on a tiled overview image."""
    if not valid_items:
        return

    thumb_w, thumb_h = 320, 210
    label_h = 42
    cols = 4
    rows = int(math.ceil(len(valid_items) / cols))
    sheet = np.full((rows * (thumb_h + label_h), cols * thumb_w, 3), 245, dtype=np.uint8)

    for index, item in enumerate(valid_items):
        image = item["image"].copy()
        corners = item["corners"]
        cv2.drawChessboardCorners(image, CHECKERBOARD, corners, True)

        scale = min(thumb_w / image.shape[1], thumb_h / image.shape[0])
        resized_w = int(image.shape[1] * scale)
        resized_h = int(image.shape[0] * scale)
        thumb = cv2.resize(image, (resized_w, resized_h), interpolation=cv2.INTER_AREA)

        row = index // cols
        col = index % cols
        x0 = col * thumb_w
        y0 = row * (thumb_h + label_h)
        x_pad = (thumb_w - resized_w) // 2
        y_pad = (thumb_h - resized_h) // 2
        sheet[y0 + y_pad : y0 + y_pad + resized_h, x0 + x_pad : x0 + x_pad + resized_w] = thumb

        reproj = item.get("reprojection_error_px")
        reproj_text = "reproj n/a" if reproj is None else f"reproj {reproj:.3f}px"
        label_1 = item["path"].name[:36]
        label_2 = f"area {item['area_ratio']:.3f}, {reproj_text}"
        draw_text(sheet, label_1, (x0 + 8, y0 + thumb_h + 17), scale=0.45, color=(20, 20, 20))
        draw_text(sheet, label_2, (x0 + 8, y0 + thumb_h + 36), scale=0.45, color=(20, 20, 20))

    cv2.imwrite(str(path), sheet)


def load_calibration(calibration_file):
    """Load intrinsics if the optional normal calibration file exists."""
    if not calibration_file.exists():
        return None, None

    data = np.load(calibration_file)
    if "camera_matrix" not in data or "dist_coeffs" not in data:
        print(f"Warning: {calibration_file} is missing camera_matrix or dist_coeffs.")
        return None, None

    return data["camera_matrix"], data["dist_coeffs"]


def format_range(stats):
    if stats["min"] is None:
        return "n/a"
    return f"{stats['min']:.3f} to {stats['max']:.3f}"


def main():
    args = parse_args()
    camera_name = args.camera if args.camera is not None else prompt_camera()
    target = resolve_camera(camera_name)
    image_dir = args.image_dir if args.image_dir is not None else target["image_dir"]
    calibration_path = (
        args.calibration_file if args.calibration_file is not None else default_calibration_file(target)
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = load_image_paths(image_dir)
    if not image_paths:
        print(f"Error: no images found in {image_dir.resolve()}")
        print(f"Run: python 01_capture_checkerboard_images.py --camera {target['label']}")
        return

    camera_matrix, dist_coeffs = load_calibration(calibration_path)
    have_calibration = camera_matrix is not None and dist_coeffs is not None
    if have_calibration:
        print(f"Loaded optional calibration: {calibration_path.resolve()}")
    else:
        print(f"No usable configured {target['label']} calibration found; skipping pose/reprojection metrics.")
    print(f"Evaluating {target['label']} images from: {image_dir.resolve()}")

    objp = make_object_points()
    valid_items = []
    csv_rows = []
    heatmap_counts = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    invalid_images = []
    image_size = None

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            invalid_images.append(path.name)
            continue

        image_height, image_width = image.shape[:2]
        image_size = (image_width, image_height)
        found, corners, gray = detect_checkerboard(image)
        sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())

        if not found:
            invalid_images.append(path.name)
            csv_rows.append(
                {
                    "image": path.name,
                    "valid": False,
                    "area_ratio": "",
                    "center_x_norm": "",
                    "center_y_norm": "",
                    "grid_cells": "",
                    "sharpness_laplacian_var": sharpness,
                    "board_distance_m": "",
                    "pitch_deg": "",
                    "yaw_deg": "",
                    "roll_deg": "",
                    "reprojection_error_px": "",
                }
            )
            continue

        min_xy, max_xy, bbox_area = corner_bbox(corners)
        area_ratio = bbox_area / float(image_width * image_height)
        center = corners.reshape(-1, 2).mean(axis=0)
        cells = grid_cells_for_corners(corners, image_width, image_height)
        for row, col in cells:
            heatmap_counts[row, col] += 1

        item = {
            "path": path,
            "image": image,
            "corners": corners,
            "area_ratio": area_ratio,
            "center_x_norm": float(center[0] / image_width),
            "center_y_norm": float(center[1] / image_height),
            "grid_cells": sorted([f"{row},{col}" for row, col in cells]),
            "sharpness_laplacian_var": sharpness,
            "bbox_min_x": float(min_xy[0]),
            "bbox_min_y": float(min_xy[1]),
            "bbox_max_x": float(max_xy[0]),
            "bbox_max_y": float(max_xy[1]),
            "board_distance_m": None,
            "pitch_deg": None,
            "yaw_deg": None,
            "roll_deg": None,
            "reprojection_error_px": None,
        }

        if have_calibration:
            ok, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
            if ok:
                item["board_distance_m"] = float(np.linalg.norm(tvec))
                item.update(rotation_vector_to_euler_degrees(rvec))
                item["reprojection_error_px"] = reprojection_rmse(
                    objp, corners, rvec, tvec, camera_matrix, dist_coeffs
                )

        valid_items.append(item)
        csv_rows.append(
            {
                "image": path.name,
                "valid": True,
                "area_ratio": area_ratio,
                "center_x_norm": item["center_x_norm"],
                "center_y_norm": item["center_y_norm"],
                "grid_cells": ";".join(item["grid_cells"]),
                "sharpness_laplacian_var": sharpness,
                "board_distance_m": "" if item["board_distance_m"] is None else item["board_distance_m"],
                "pitch_deg": "" if item["pitch_deg"] is None else item["pitch_deg"],
                "yaw_deg": "" if item["yaw_deg"] is None else item["yaw_deg"],
                "roll_deg": "" if item["roll_deg"] is None else item["roll_deg"],
                "reprojection_error_px": ""
                if item["reprojection_error_px"] is None
                else item["reprojection_error_px"],
            }
        )

    total_images = len(image_paths)
    valid_count = len(valid_items)
    invalid_count = total_images - valid_count
    covered_cells = int(np.count_nonzero(heatmap_counts))
    grid_coverage_percent = covered_cells / float(GRID_SIZE * GRID_SIZE) * 100.0

    area_ratios = [item["area_ratio"] for item in valid_items]
    sharpness_values = [item["sharpness_laplacian_var"] for item in valid_items]
    distances = [item["board_distance_m"] for item in valid_items if item["board_distance_m"] is not None]
    pitch_values = [item["pitch_deg"] for item in valid_items if item["pitch_deg"] is not None]
    yaw_values = [item["yaw_deg"] for item in valid_items if item["yaw_deg"] is not None]
    roll_values = [item["roll_deg"] for item in valid_items if item["roll_deg"] is not None]
    reprojection_errors = [
        item["reprojection_error_px"]
        for item in valid_items
        if item["reprojection_error_px"] is not None
    ]

    area_stats = safe_summary(area_ratios)
    sharpness_stats = safe_summary(sharpness_values)
    distance_stats = safe_summary(distances)
    pitch_stats = safe_summary(pitch_values)
    yaw_stats = safe_summary(yaw_values)
    roll_stats = safe_summary(roll_values)
    reprojection_stats = safe_summary(reprojection_errors)

    weak_points = []
    if valid_count < MIN_VALID_IMAGES:
        weak_points.append(f"Too few valid images: {valid_count} valid, target at least {MIN_VALID_IMAGES}.")
    if grid_coverage_percent < MIN_GRID_COVERAGE_PERCENT:
        weak_points.append(
            f"Grid coverage below {MIN_GRID_COVERAGE_PERCENT:.0f}%: {grid_coverage_percent:.1f}%."
        )
    if area_stats["min"] is not None and (area_stats["max"] - area_stats["min"]) < MIN_AREA_RATIO_RANGE:
        weak_points.append(
            f"Board area range is narrow: {area_stats['min']:.3f} to {area_stats['max']:.3f}."
        )
    if pitch_stats["min"] is not None:
        pitch_range = pitch_stats["max"] - pitch_stats["min"]
        yaw_range = yaw_stats["max"] - yaw_stats["min"]
        roll_range = roll_stats["max"] - roll_stats["min"]
        if max(pitch_range, yaw_range, roll_range) < MIN_POSE_RANGE_DEGREES:
            weak_points.append("Low pose diversity: pitch/yaw/roll ranges are all small.")
    if reprojection_stats["p90"] is not None and reprojection_stats["p90"] > HIGH_REPROJECTION_ERROR_PX:
        weak_points.append(
            f"High reprojection outliers: 90th percentile is {reprojection_stats['p90']:.3f} px."
        )
    blurry_count = sum(value < BLURRY_LAPLACIAN_THRESHOLD for value in sharpness_values)
    if blurry_count:
        weak_points.append(
            f"Potentially blurry images: {blurry_count} valid images below Laplacian variance "
            f"{BLURRY_LAPLACIAN_THRESHOLD:.0f}."
        )

    metrics = {
        "settings": {
            "camera": target["label"],
            "image_dir": str(image_dir),
            "checkerboard_inner_corners": CHECKERBOARD,
            "square_size_m": SQUARE_SIZE,
            "grid_size": [GRID_SIZE, GRID_SIZE],
            "calibration_file": str(calibration_path),
            "used_calibration_file": have_calibration,
            "image_size": image_size,
        },
        "summary": {
            "total_images": total_images,
            "valid_detections": valid_count,
            "invalid_images": invalid_count,
            "grid_coverage_percent": grid_coverage_percent,
            "covered_grid_cells": covered_cells,
            "board_area_ratio": area_stats,
            "board_distance_m": distance_stats,
            "pitch_deg": pitch_stats,
            "yaw_deg": yaw_stats,
            "roll_deg": roll_stats,
            "reprojection_error_px": reprojection_stats,
            "sharpness_laplacian_var": sharpness_stats,
            "weak_points": weak_points,
        },
        "invalid_image_names": invalid_images,
    }

    save_json(args.output_dir / "calibration_dataset_metrics.json", metrics)
    save_csv(args.output_dir / "calibration_per_image_metrics.csv", csv_rows)
    save_heatmap(args.output_dir / "calibration_coverage_heatmap.png", heatmap_counts)
    save_line_plot(
        args.output_dir / "reprojection_error_plot.png",
        reprojection_errors,
        "Per-image reprojection error",
        "pixels",
    )
    save_histogram(
        args.output_dir / "board_area_distribution.png",
        area_ratios,
        "Checkerboard area ratio distribution",
        "area ratio",
    )
    save_contact_sheet(args.output_dir / "calibration_contact_sheet.png", valid_items)

    print("\nCalibration dataset quality summary")
    print("-----------------------------------")
    print(f"Camera: {target['label']}")
    print(f"Images: {total_images}")
    print(f"Valid detections: {valid_count}")
    print(f"Invalid detections: {invalid_count}")
    print(f"Grid coverage: {grid_coverage_percent:.1f}% ({covered_cells}/{GRID_SIZE * GRID_SIZE} cells)")
    print(
        "Board area ratio min/max/mean/std: "
        f"{area_stats['min']:.3f} / {area_stats['max']:.3f} / "
        f"{area_stats['mean']:.3f} / {area_stats['std']:.3f}"
        if area_stats["min"] is not None
        else "Board area ratio min/max/mean/std: n/a"
    )
    print(f"Board distance min/max: {format_range(distance_stats)} m")
    print(f"Pitch range: {format_range(pitch_stats)} deg")
    print(f"Yaw range: {format_range(yaw_stats)} deg")
    print(f"Roll range: {format_range(roll_stats)} deg")
    if reprojection_stats["mean"] is None:
        print("Reprojection error mean/median/RMSE/90th/max: n/a")
    else:
        print(
            "Reprojection error mean/median/RMSE/90th/max: "
            f"{reprojection_stats['mean']:.3f} / {reprojection_stats['median']:.3f} / "
            f"{reprojection_stats['rmse']:.3f} / {reprojection_stats['p90']:.3f} / "
            f"{reprojection_stats['max']:.3f} px"
        )
    print(
        f"Sharpness min/median: {sharpness_stats['min']:.1f} / {sharpness_stats['median']:.1f}"
        if sharpness_stats["min"] is not None
        else "Sharpness min/median: n/a"
    )

    print("\nWeak points")
    print("-----------")
    if weak_points:
        for weak_point in weak_points:
            print(f"- {weak_point}")
    else:
        print("- No obvious weak points found by the simple thresholds.")

    print("\nSaved outputs")
    print("-------------")
    print(args.output_dir / "calibration_dataset_metrics.json")
    print(args.output_dir / "calibration_per_image_metrics.csv")
    print(args.output_dir / "calibration_coverage_heatmap.png")
    print(args.output_dir / "reprojection_error_plot.png")
    print(args.output_dir / "board_area_distribution.png")
    print(args.output_dir / "calibration_contact_sheet.png")


if __name__ == "__main__":
    main()
