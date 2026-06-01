"""
Step 0: evaluate stereo checkerboard image-pair quality before stereo calibration.

Run from this folder:
    python 00_evaluate_stereo_image_set.py

This is a pre-flight check for paired L/R checkerboard captures. By default it
reads ../00_data_capture/int_ext_calib_rgb/L and R, and it also supports the
older flat stereo_pairs/rgb1_*.png and rgb2_*.png layout. It
measures checkerboard detection success, coverage, sharpness, size/pose
diversity, per-camera solvePnP reprojection errors, and fixed-intrinsics stereo
calibration RMS when configured intrinsics are available.
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

matplotlib.use("Agg")
import matplotlib.pyplot as plt

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from utils.project_config import calibration_file

SCRIPT_DIR = Path(__file__).resolve().parent

CHECKERBOARD = (9, 6)
SQUARE_SIZE = 0.025  # meters
GRID_SIZE = 5

MIN_VALID_PAIRS = 10
MIN_GRID_COVERAGE_PERCENT = 50.0
MIN_AREA_RATIO_RANGE = 0.06
MIN_POSE_RANGE_DEGREES = 15.0
BLURRY_LAPLACIAN_THRESHOLD = 100.0
HIGH_REPROJECTION_ERROR_PX = 1.0
HIGH_STEREO_RMS_PX = 1.5



def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate stereo checkerboard image-pair quality.")
    parser.add_argument("--pair-dir", type=Path, default=_REPO_ROOT / "00_data_capture" / "int_ext_calib_rgb")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--rgb1-calibration", type=Path, default=calibration_file("left_intrinsics"))
    parser.add_argument("--rgb2-calibration", type=Path, default=calibration_file("right_intrinsics"))
    return parser.parse_args()


def make_object_points():
    objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : CHECKERBOARD[0], 0 : CHECKERBOARD[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


def stereo_pair_paths(pair_dir):
    left_dir = pair_dir / "L"
    right_dir = pair_dir / "R"
    if left_dir.exists() and right_dir.exists():
        left_images = {}
        right_images = {}
        for prefix in ("calib", "rgb1"):
            for path in sorted(left_dir.glob(f"{prefix}_*.png")):
                left_images[path.stem.split("_", 1)[1]] = path
        for prefix in ("calib", "rgb2"):
            for path in sorted(right_dir.glob(f"{prefix}_*.png")):
                right_images[path.stem.split("_", 1)[1]] = path

        pair_ids = sorted(set(left_images).union(right_images))
        return [
            (
                pair_id,
                left_images.get(pair_id),
                right_images.get(pair_id),
            )
            for pair_id in pair_ids
        ]

    pairs = []
    for rgb1_path in sorted(pair_dir.glob("rgb1_*.png")):
        pair_id = rgb1_path.stem.split("_")[1]
        rgb2_path = pair_dir / f"rgb2_{pair_id}.png"
        pairs.append((pair_id, rgb1_path, rgb2_path if rgb2_path.exists() else None))
    return pairs


def detect_checkerboard(image):
    gray_original = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_variants = (
        gray_original,
        cv2.equalizeHist(gray_original),
        cv2.GaussianBlur(cv2.equalizeHist(gray_original), (3, 3), 0),
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        for gray in gray_variants:
            found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, sb_flags)
            if found:
                corners = corners.astype(np.float32)
                corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
                return True, corners, gray_original

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for gray in gray_variants:
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners, gray_original

    return False, None, gray_original


def load_calibration(path):
    if not path.exists():
        print(f"Warning: missing calibration file {path}")
        return None
    data = np.load(path)
    required = {"camera_matrix", "dist_coeffs", "image_size"}
    missing = required.difference(data.files)
    if missing:
        print(f"Warning: {path} is missing keys {sorted(missing)}")
        return None
    return {
        "path": str(path),
        "camera_matrix": data["camera_matrix"].astype(np.float64),
        "dist_coeffs": data["dist_coeffs"].reshape(-1, 1).astype(np.float64),
        "image_size": tuple(data["image_size"].astype(int)),
        "model": "fisheye" if "fisheye" in path.name.lower() else "pinhole",
    }


def bbox_area_ratio(corners, image_shape):
    pts = corners.reshape(-1, 2)
    min_xy = pts.min(axis=0)
    max_xy = pts.max(axis=0)
    area = float(np.prod(max_xy - min_xy))
    image_area = float(image_shape[0] * image_shape[1])
    center = pts.mean(axis=0)
    return {
        "bbox_min_x": float(min_xy[0]),
        "bbox_min_y": float(min_xy[1]),
        "bbox_max_x": float(max_xy[0]),
        "bbox_max_y": float(max_xy[1]),
        "area_ratio": area / image_area,
        "center_x_norm": float(center[0] / image_shape[1]),
        "center_y_norm": float(center[1] / image_shape[0]),
    }


def grid_cells(corners, image_shape):
    height, width = image_shape[:2]
    cells = set()
    for x, y in corners.reshape(-1, 2):
        col = min(GRID_SIZE - 1, max(0, int(x / width * GRID_SIZE)))
        row = min(GRID_SIZE - 1, max(0, int(y / height * GRID_SIZE)))
        cells.add((row, col))
    return cells


def rotation_vector_to_euler_degrees(rvec):
    R, _ = cv2.Rodrigues(rvec)
    sy = math.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = math.atan2(R[2, 1], R[2, 2])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = math.atan2(R[1, 0], R[0, 0])
    else:
        roll = math.atan2(-R[1, 2], R[1, 1])
        pitch = math.atan2(-R[2, 0], sy)
        yaw = 0.0
    return {
        "pitch_deg": float(math.degrees(pitch)),
        "yaw_deg": float(math.degrees(yaw)),
        "roll_deg": float(math.degrees(roll)),
    }


def reprojection_rmse(objp, corners, rvec, tvec, camera_matrix, dist_coeffs):
    projected, _ = cv2.projectPoints(objp, rvec, tvec, camera_matrix, dist_coeffs)
    errors = np.linalg.norm(corners.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(errors * errors)))


def fisheye_reprojection_rmse(objp, corners, rvec, tvec, camera_matrix, dist_coeffs):
    projected, _ = cv2.fisheye.projectPoints(
        objp.reshape(1, -1, 3).astype(np.float64),
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs.reshape(-1, 1),
    )
    errors = np.linalg.norm(corners.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
    return float(np.sqrt(np.mean(errors * errors)))


def safe_summary(values):
    if not values:
        return {"min": None, "max": None, "mean": None, "std": None, "median": None, "rmse": None, "p90": None}
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


def range_text(stats, suffix=""):
    if stats["min"] is None:
        return "n/a"
    return f"{stats['min']:.3f} to {stats['max']:.3f}{suffix}"


def run_stereo_calibration(valid_items, rgb1_calib, rgb2_calib, objp):
    if not valid_items or rgb1_calib is None or rgb2_calib is None:
        return None

    image_size = rgb1_calib["image_size"]
    model = "fisheye" if rgb1_calib.get("model") == "fisheye" or rgb2_calib.get("model") == "fisheye" else "pinhole"
    K1 = rgb1_calib["camera_matrix"].copy()
    d1 = rgb1_calib["dist_coeffs"].copy()
    K2 = rgb2_calib["camera_matrix"].copy()
    d2 = rgb2_calib["dist_coeffs"].copy()

    if model == "fisheye":
        objpoints = [objp.reshape(1, -1, 3).astype(np.float64).copy() for _ in valid_items]
        imgpoints1 = [np.ascontiguousarray(item["rgb1_corners"].reshape(1, -1, 2).astype(np.float64)) for item in valid_items]
        imgpoints2 = [np.ascontiguousarray(item["rgb2_corners"].reshape(1, -1, 2).astype(np.float64)) for item in valid_items]
        R = np.eye(3, dtype=np.float64)
        t = np.zeros((3, 1), dtype=np.float64)
        flags = cv2.fisheye.CALIB_FIX_INTRINSIC + cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv2.fisheye.CALIB_CHECK_COND
        rms, _, _, _, _, R, t = cv2.fisheye.stereoCalibrate(
            objpoints,
            imgpoints1,
            imgpoints2,
            K1,
            d1,
            K2,
            d2,
            image_size,
            R,
            t,
            flags,
            criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 1e-7),
        )
        E = None
        F = None
    else:
        objpoints = [objp.copy() for _ in valid_items]
        imgpoints1 = [np.ascontiguousarray(item["rgb1_corners"].reshape(-1, 1, 2).astype(np.float32)) for item in valid_items]
        imgpoints2 = [np.ascontiguousarray(item["rgb2_corners"].reshape(-1, 1, 2).astype(np.float32)) for item in valid_items]
        rms, _, _, _, _, R, t, E, F = cv2.stereoCalibrate(
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

    return {
        "model": model,
        "stereo_rms_error_px": float(rms),
        "baseline_m": float(np.linalg.norm(t)),
        "R_rgb1_to_rgb2": R.tolist(),
        "t_rgb1_to_rgb2": t.reshape(3).tolist(),
        "essential_matrix": None if E is None else E.tolist(),
        "fundamental_matrix": None if F is None else F.tolist(),
    }


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def draw_text(image, text, origin, scale=0.5, color=(255, 255, 255), thickness=1):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def save_heatmap(path, counts, title):
    fig, ax = plt.subplots(figsize=(6, 5), dpi=150)
    image = ax.imshow(counts, cmap="viridis", interpolation="nearest")
    for row in range(GRID_SIZE):
        for col in range(GRID_SIZE):
            ax.text(col, row, str(int(counts[row, col])), ha="center", va="center", color="white")
    ax.set_title(title)
    ax.set_xlabel("Image grid column")
    ax.set_ylabel("Image grid row")
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    fig.colorbar(image, ax=ax, label="Detected corners in cell")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_line_plot(path, values_by_label, title, y_label, threshold=None):
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax.set_title(title)
    ax.set_xlabel("Valid stereo pair index")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.3)
    any_values = False
    for label, values in values_by_label.items():
        if not values:
            continue
        any_values = True
        ax.plot(range(len(values)), values, marker="o", linewidth=1.6, label=label)
    if threshold is not None:
        ax.axhline(threshold, color="tab:red", linestyle="--", linewidth=1.2, label=f"{threshold:g} px")
    if any_values:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No values available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_histogram(path, values_by_label, title, x_label):
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    ax.set_title(title)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Pair count")
    ax.grid(True, axis="y", alpha=0.3)
    any_values = False
    for label, values in values_by_label.items():
        if not values:
            continue
        any_values = True
        bins = min(12, max(4, int(math.sqrt(len(values)))))
        ax.hist(values, bins=bins, alpha=0.55, edgecolor="black", label=label)
    if any_values:
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No values available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_contact_sheet(path, valid_items):
    if not valid_items:
        return
    thumb_w, thumb_h = 260, 150
    label_h = 48
    cols = 2
    rows = len(valid_items)
    sheet = np.full((rows * (thumb_h + label_h), cols * thumb_w, 3), 245, dtype=np.uint8)

    for row, item in enumerate(valid_items):
        for col, camera in enumerate(("rgb1", "rgb2")):
            image = item[f"{camera}_image"].copy()
            corners = item[f"{camera}_corners"]
            cv2.drawChessboardCorners(image, CHECKERBOARD, corners, True)
            scale = min(thumb_w / image.shape[1], thumb_h / image.shape[0])
            resized = cv2.resize(image, (int(image.shape[1] * scale), int(image.shape[0] * scale)), interpolation=cv2.INTER_AREA)
            x0 = col * thumb_w
            y0 = row * (thumb_h + label_h)
            x_pad = (thumb_w - resized.shape[1]) // 2
            y_pad = (thumb_h - resized.shape[0]) // 2
            sheet[y0 + y_pad : y0 + y_pad + resized.shape[0], x0 + x_pad : x0 + x_pad + resized.shape[1]] = resized
            area = item[f"{camera}_area_ratio"]
            reproj = item.get(f"{camera}_reprojection_error_px")
            reproj_text = "n/a" if reproj is None else f"{reproj:.3f}px"
            draw_text(sheet, f"{camera.upper()} pair {item['pair_id']}", (x0 + 8, y0 + thumb_h + 18), color=(20, 20, 20))
            draw_text(sheet, f"area {area:.3f}, reproj {reproj_text}", (x0 + 8, y0 + thumb_h + 38), color=(20, 20, 20))

    cv2.imwrite(str(path), sheet)


def camera_metrics(prefix, image, corners, gray, objp, calib):
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    bbox = bbox_area_ratio(corners, image.shape)
    metrics = {
        f"{prefix}_area_ratio": bbox["area_ratio"],
        f"{prefix}_center_x_norm": bbox["center_x_norm"],
        f"{prefix}_center_y_norm": bbox["center_y_norm"],
        f"{prefix}_sharpness_laplacian_var": sharpness,
        f"{prefix}_board_distance_m": None,
        f"{prefix}_pitch_deg": None,
        f"{prefix}_yaw_deg": None,
        f"{prefix}_roll_deg": None,
        f"{prefix}_reprojection_error_px": None,
        f"{prefix}_grid_cells": sorted([f"{row},{col}" for row, col in grid_cells(corners, image.shape)]),
    }
    for key, value in bbox.items():
        metrics[f"{prefix}_{key}"] = value

    if calib is not None:
        if calib.get("model") == "fisheye":
            pose_corners = cv2.fisheye.undistortPoints(
                corners.reshape(-1, 1, 2).astype(np.float64),
                calib["camera_matrix"],
                calib["dist_coeffs"].reshape(-1, 1),
                P=calib["camera_matrix"],
            ).astype(np.float32)
            ok, rvec, tvec = cv2.solvePnP(objp, pose_corners, calib["camera_matrix"], None)
        else:
            ok, rvec, tvec = cv2.solvePnP(objp, corners, calib["camera_matrix"], calib["dist_coeffs"])
        if ok:
            metrics[f"{prefix}_board_distance_m"] = float(np.linalg.norm(tvec))
            for key, value in rotation_vector_to_euler_degrees(rvec).items():
                metrics[f"{prefix}_{key}"] = value
            if calib.get("model") == "fisheye":
                metrics[f"{prefix}_reprojection_error_px"] = fisheye_reprojection_rmse(
                    objp, corners, rvec, tvec, calib["camera_matrix"], calib["dist_coeffs"]
                )
            else:
                metrics[f"{prefix}_reprojection_error_px"] = reprojection_rmse(
                    objp, corners, rvec, tvec, calib["camera_matrix"], calib["dist_coeffs"]
                )
    return metrics


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pairs = stereo_pair_paths(args.pair_dir)
    if not pairs:
        print(f"Error: no stereo image pairs found in {args.pair_dir.resolve()}")
        return

    rgb1_calib = load_calibration(args.rgb1_calibration)
    rgb2_calib = load_calibration(args.rgb2_calibration)
    objp = make_object_points()
    valid_items = []
    csv_rows = []
    invalid_pairs = []
    heatmap1 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)
    heatmap2 = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int32)

    for pair_id, rgb1_path, rgb2_path in pairs:
        row = {
            "pair_id": pair_id,
            "rgb1_image": "" if rgb1_path is None else rgb1_path.name,
            "rgb2_image": "" if rgb2_path is None else rgb2_path.name,
            "valid_pair": False,
        }
        if rgb1_path is None:
            invalid_pairs.append({"pair_id": pair_id, "reason": "missing rgb1 image"})
            csv_rows.append(row)
            continue
        if rgb2_path is None:
            invalid_pairs.append({"pair_id": pair_id, "reason": "missing rgb2 image"})
            csv_rows.append(row)
            continue

        image1 = cv2.imread(str(rgb1_path))
        image2 = cv2.imread(str(rgb2_path))
        if image1 is None or image2 is None:
            invalid_pairs.append({"pair_id": pair_id, "reason": "unreadable image"})
            csv_rows.append(row)
            continue

        found1, corners1, gray1 = detect_checkerboard(image1)
        found2, corners2, gray2 = detect_checkerboard(image2)
        if not (found1 and found2):
            reason = f"checkerboard missing: rgb1={found1}, rgb2={found2}"
            invalid_pairs.append({"pair_id": pair_id, "reason": reason})
            csv_rows.append(row)
            continue

        metrics = {"pair_id": pair_id}
        metrics.update(camera_metrics("rgb1", image1, corners1, gray1, objp, rgb1_calib))
        metrics.update(camera_metrics("rgb2", image2, corners2, gray2, objp, rgb2_calib))

        for r, c in grid_cells(corners1, image1.shape):
            heatmap1[r, c] += 1
        for r, c in grid_cells(corners2, image2.shape):
            heatmap2[r, c] += 1

        item = {
            **metrics,
            "rgb1_image": image1,
            "rgb2_image": image2,
            "rgb1_corners": corners1,
            "rgb2_corners": corners2,
        }
        valid_items.append(item)

        row.update({"valid_pair": True})
        for key, value in metrics.items():
            row[key] = ";".join(value) if isinstance(value, list) else ("" if value is None else value)
        csv_rows.append(row)

    stereo_result = run_stereo_calibration(valid_items, rgb1_calib, rgb2_calib, objp)
    valid_count = len(valid_items)
    total_pairs = len(pairs)
    grid1_coverage = float(np.count_nonzero(heatmap1) / (GRID_SIZE * GRID_SIZE) * 100.0)
    grid2_coverage = float(np.count_nonzero(heatmap2) / (GRID_SIZE * GRID_SIZE) * 100.0)

    def vals(name):
        return [item[name] for item in valid_items if item.get(name) is not None]

    summaries = {
        "rgb1_area_ratio": safe_summary(vals("rgb1_area_ratio")),
        "rgb2_area_ratio": safe_summary(vals("rgb2_area_ratio")),
        "rgb1_distance_m": safe_summary(vals("rgb1_board_distance_m")),
        "rgb2_distance_m": safe_summary(vals("rgb2_board_distance_m")),
        "rgb1_reprojection_error_px": safe_summary(vals("rgb1_reprojection_error_px")),
        "rgb2_reprojection_error_px": safe_summary(vals("rgb2_reprojection_error_px")),
        "rgb1_sharpness": safe_summary(vals("rgb1_sharpness_laplacian_var")),
        "rgb2_sharpness": safe_summary(vals("rgb2_sharpness_laplacian_var")),
        "rgb1_pitch_deg": safe_summary(vals("rgb1_pitch_deg")),
        "rgb2_pitch_deg": safe_summary(vals("rgb2_pitch_deg")),
        "rgb1_yaw_deg": safe_summary(vals("rgb1_yaw_deg")),
        "rgb2_yaw_deg": safe_summary(vals("rgb2_yaw_deg")),
        "rgb1_roll_deg": safe_summary(vals("rgb1_roll_deg")),
        "rgb2_roll_deg": safe_summary(vals("rgb2_roll_deg")),
    }

    weak_points = []
    if valid_count < MIN_VALID_PAIRS:
        weak_points.append(f"Too few valid stereo pairs: {valid_count}, target at least {MIN_VALID_PAIRS}.")
    if min(grid1_coverage, grid2_coverage) < MIN_GRID_COVERAGE_PERCENT:
        weak_points.append(f"Low grid coverage: RGB1 {grid1_coverage:.1f}%, RGB2 {grid2_coverage:.1f}%.")
    for cam in ("rgb1", "rgb2"):
        area = summaries[f"{cam}_area_ratio"]
        if area["min"] is not None and area["max"] - area["min"] < MIN_AREA_RATIO_RANGE:
            weak_points.append(f"{cam.upper()} board area range is narrow: {area['min']:.3f} to {area['max']:.3f}.")
        pitch = summaries[f"{cam}_pitch_deg"]
        yaw = summaries[f"{cam}_yaw_deg"]
        roll = summaries[f"{cam}_roll_deg"]
        if pitch["min"] is not None and max(pitch["max"] - pitch["min"], yaw["max"] - yaw["min"], roll["max"] - roll["min"]) < MIN_POSE_RANGE_DEGREES:
            weak_points.append(f"{cam.upper()} pose diversity is low.")
        reproj = summaries[f"{cam}_reprojection_error_px"]
        if reproj["p90"] is not None and reproj["p90"] > HIGH_REPROJECTION_ERROR_PX:
            weak_points.append(f"{cam.upper()} reprojection 90th percentile is high: {reproj['p90']:.3f} px.")
        blurry = sum(v < BLURRY_LAPLACIAN_THRESHOLD for v in vals(f"{cam}_sharpness_laplacian_var"))
        if blurry:
            weak_points.append(f"{cam.upper()} has {blurry} potentially blurry valid images.")
    if stereo_result and stereo_result["stereo_rms_error_px"] > HIGH_STEREO_RMS_PX:
        weak_points.append(f"Stereo fixed-intrinsics RMS is high: {stereo_result['stereo_rms_error_px']:.3f} px.")

    metrics = {
        "settings": {
            "pair_dir": str(args.pair_dir),
            "checkerboard_inner_corners": CHECKERBOARD,
            "square_size_m": SQUARE_SIZE,
            "grid_size": [GRID_SIZE, GRID_SIZE],
            "rgb1_calibration": None if rgb1_calib is None else rgb1_calib["path"],
            "rgb2_calibration": None if rgb2_calib is None else rgb2_calib["path"],
        },
        "summary": {
            "total_pairs": total_pairs,
            "valid_pairs": valid_count,
            "invalid_pairs": total_pairs - valid_count,
            "rgb1_grid_coverage_percent": grid1_coverage,
            "rgb2_grid_coverage_percent": grid2_coverage,
            "stereo": stereo_result,
            "stats": summaries,
            "weak_points": weak_points,
        },
        "invalid_pair_details": invalid_pairs,
    }

    save_json(args.output_dir / "stereo_dataset_metrics.json", metrics)
    save_csv(args.output_dir / "stereo_per_pair_metrics.csv", csv_rows)
    save_heatmap(args.output_dir / "stereo_rgb1_coverage_heatmap.png", heatmap1, "RGB1 checkerboard coverage")
    save_heatmap(args.output_dir / "stereo_rgb2_coverage_heatmap.png", heatmap2, "RGB2 checkerboard coverage")
    save_line_plot(
        args.output_dir / "stereo_reprojection_error_plot.png",
        {
            "RGB1": vals("rgb1_reprojection_error_px"),
            "RGB2": vals("rgb2_reprojection_error_px"),
        },
        "Per-pair solvePnP reprojection error",
        "pixels",
        threshold=HIGH_REPROJECTION_ERROR_PX,
    )
    save_histogram(
        args.output_dir / "stereo_board_area_distribution.png",
        {"RGB1": vals("rgb1_area_ratio"), "RGB2": vals("rgb2_area_ratio")},
        "Stereo checkerboard area ratio distribution",
        "area ratio",
    )
    save_contact_sheet(args.output_dir / "stereo_contact_sheet.png", valid_items)

    print("\nStereo dataset quality summary")
    print("------------------------------")
    print(f"Pairs: {total_pairs}")
    print(f"Valid stereo detections: {valid_count}")
    print(f"Invalid pairs: {total_pairs - valid_count}")
    print(f"RGB1 grid coverage: {grid1_coverage:.1f}%")
    print(f"RGB2 grid coverage: {grid2_coverage:.1f}%")
    print(f"RGB1 area ratio range: {range_text(summaries['rgb1_area_ratio'])}")
    print(f"RGB2 area ratio range: {range_text(summaries['rgb2_area_ratio'])}")
    print(f"RGB1 distance range: {range_text(summaries['rgb1_distance_m'], ' m')}")
    print(f"RGB2 distance range: {range_text(summaries['rgb2_distance_m'], ' m')}")
    print(f"RGB1 reprojection mean/median/p90/max: {summaries['rgb1_reprojection_error_px']['mean']:.3f} / {summaries['rgb1_reprojection_error_px']['median']:.3f} / {summaries['rgb1_reprojection_error_px']['p90']:.3f} / {summaries['rgb1_reprojection_error_px']['max']:.3f} px")
    print(f"RGB2 reprojection mean/median/p90/max: {summaries['rgb2_reprojection_error_px']['mean']:.3f} / {summaries['rgb2_reprojection_error_px']['median']:.3f} / {summaries['rgb2_reprojection_error_px']['p90']:.3f} / {summaries['rgb2_reprojection_error_px']['max']:.3f} px")
    if stereo_result:
        print(f"Stereo fixed-intrinsics RMS: {stereo_result['stereo_rms_error_px']:.3f} px")
        print(f"Estimated baseline: {stereo_result['baseline_m']:.3f} m")
    print("\nWeak points")
    print("-----------")
    if weak_points:
        for weak_point in weak_points:
            print(f"- {weak_point}")
    else:
        print("- No obvious weak points found by the simple thresholds.")
    print("\nSaved outputs")
    print("-------------")
    for name in (
        "stereo_dataset_metrics.json",
        "stereo_per_pair_metrics.csv",
        "stereo_rgb1_coverage_heatmap.png",
        "stereo_rgb2_coverage_heatmap.png",
        "stereo_reprojection_error_plot.png",
        "stereo_board_area_distribution.png",
        "stereo_contact_sheet.png",
    ):
        print(args.output_dir / name)


if __name__ == "__main__":
    main()
