"""Evaluate saved stereo calibration quality.

Run from this folder:
    python 04_evaluate_stereo_calibration.py

Outputs:
    outputs/stereo_calibration_eval.json
    outputs/stereo_calibration_eval_per_pair.csv
    outputs/stereo_calibration_epipolar_error_plot.png
    outputs/stereo_calibration_rectification_vertical_error_plot.png
    outputs/stereo_calibration_error_histograms.png
    outputs/stereo_rectified_alignment_example.png
    README.md
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


cv2.ocl.setUseOpenCL(False)
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.project_config import calibration_file


CHECKERBOARD = (9, 6)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate stereo calibration geometry.")
    parser.add_argument("--pair-dir", type=Path, default=SCRIPT_DIR / "stereo_pairs")
    parser.add_argument("--output-dir", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--rgb1-calibration", type=Path, default=calibration_file("left_intrinsics"))
    parser.add_argument("--rgb2-calibration", type=Path, default=calibration_file("right_intrinsics"))
    parser.add_argument("--stereo-calibration", type=Path, default=calibration_file("stereo_rgb1_rgb2_extrinsics"))
    parser.add_argument("--readme", type=Path, default=SCRIPT_DIR / "README.md")
    return parser.parse_args()


def load_intrinsics(path):
    data = np.load(path)
    return {
        "camera_matrix": data["camera_matrix"].astype(np.float64),
        "dist_coeffs": data["dist_coeffs"].astype(np.float64),
        "image_size": tuple(data["image_size"].astype(int)),
    }


def load_stereo(path):
    data = np.load(path)
    return {
        "R": data["R_rgb1_to_rgb2"].astype(np.float64),
        "t": data["t_rgb1_to_rgb2"].reshape(3, 1).astype(np.float64),
        "F": data["fundamental_matrix"].astype(np.float64),
        "stereo_rms_error_px": float(np.asarray(data["stereo_rms_error"]).reshape(())),
        "baseline_m": float(np.asarray(data["baseline_meters"]).reshape(())),
        "used_pair_ids": [str(x) for x in data["used_pair_ids"]] if "used_pair_ids" in data.files else None,
    }


def stereo_pair_paths(pair_dir, allowed_ids=None):
    allowed = None if allowed_ids is None else set(allowed_ids)
    pairs = []
    for rgb1_path in sorted(pair_dir.glob("rgb1_*.png")):
        pair_id = rgb1_path.stem.split("_")[1]
        if allowed is not None and pair_id not in allowed:
            continue
        rgb2_path = pair_dir / f"rgb2_{pair_id}.png"
        if rgb2_path.exists():
            pairs.append((pair_id, rgb1_path, rgb2_path))
    return pairs


def detect_checkerboard(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants = (
        gray,
        cv2.equalizeHist(gray),
        cv2.GaussianBlur(cv2.equalizeHist(gray), (3, 3), 0),
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    if hasattr(cv2, "findChessboardCornersSB"):
        sb_flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        for candidate in variants:
            found, corners = cv2.findChessboardCornersSB(candidate, CHECKERBOARD, sb_flags)
            if found:
                corners = cv2.cornerSubPix(candidate, corners.astype(np.float32), (11, 11), (-1, -1), criteria)
                return True, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for candidate in variants:
        found, corners = cv2.findChessboardCorners(candidate, CHECKERBOARD, flags)
        if found:
            corners = cv2.cornerSubPix(candidate, corners, (11, 11), (-1, -1), criteria)
            return True, corners
    return False, None


def point_line_distances(points, lines):
    pts = points.reshape(-1, 2)
    numerator = np.abs(lines[:, 0] * pts[:, 0] + lines[:, 1] * pts[:, 1] + lines[:, 2])
    denominator = np.sqrt(lines[:, 0] ** 2 + lines[:, 1] ** 2)
    return numerator / denominator


def epipolar_errors_before_rectification(corners1, corners2, fundamental_matrix):
    pts1 = corners1.reshape(-1, 2)
    pts2 = corners2.reshape(-1, 2)
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, fundamental_matrix).reshape(-1, 3)
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fundamental_matrix).reshape(-1, 3)
    d2 = point_line_distances(pts2, lines2)
    d1 = point_line_distances(pts1, lines1)
    return 0.5 * (d1 + d2)


def rectified_vertical_errors(corners1, corners2, rectification, calib1, calib2):
    r1, r2, p1, p2 = rectification
    rect1 = cv2.undistortPoints(corners1, calib1["camera_matrix"], calib1["dist_coeffs"], R=r1, P=p1)
    rect2 = cv2.undistortPoints(corners2, calib2["camera_matrix"], calib2["dist_coeffs"], R=r2, P=p2)
    y1 = rect1.reshape(-1, 2)[:, 1]
    y2 = rect2.reshape(-1, 2)[:, 1]
    return np.abs(y1 - y2)


def summarize(values):
    if len(values) == 0:
        return {"mean": None, "median": None, "rmse": None, "p90": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "rmse": float(np.sqrt(np.mean(arr * arr))),
        "p90": float(np.percentile(arr, 90)),
        "max": float(np.max(arr)),
    }


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def save_json(path, data):
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def save_metric_plot(path, rows, metric_prefix, title):
    fig, ax = plt.subplots(figsize=(10, 4.8), dpi=150)
    ax.set_title(title)
    ax.set_xlabel("Stereo pair id")
    ax.set_ylabel("pixels")
    ax.grid(True, alpha=0.3)

    if rows:
        labels = [row["pair_id"] for row in rows]
        x = np.arange(len(labels))
        for suffix, label in (("mean_px", "mean"), ("p90_px", "p90"), ("max_px", "max")):
            key = f"{metric_prefix}_{suffix}"
            ax.plot(x, [float(row[key]) for row in rows], marker="o", linewidth=1.5, label=label)
        step = max(1, len(labels) // 12)
        ax.set_xticks(x[::step])
        ax.set_xticklabels(labels[::step], rotation=45, ha="right")
        ax.legend()
    else:
        ax.text(0.5, 0.5, "No valid stereo pairs", ha="center", va="center", transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_error_histograms(path, epipolar_values, vertical_values):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), dpi=150)
    plots = (
        (axes[0], epipolar_values, "Epipolar error before rectification", "tab:blue"),
        (axes[1], vertical_values, "Rectification vertical error", "tab:green"),
    )
    for ax, values, title, color in plots:
        ax.set_title(title)
        ax.set_xlabel("pixels")
        ax.set_ylabel("corner count")
        ax.grid(True, axis="y", alpha=0.3)
        if values:
            arr = np.asarray(values, dtype=float)
            bins = min(24, max(6, int(np.sqrt(len(arr)))))
            ax.hist(arr, bins=bins, color=color, edgecolor="black", alpha=0.75)
            ax.axvline(np.mean(arr), color="black", linestyle="--", linewidth=1.2, label=f"mean {np.mean(arr):.2f}")
            ax.axvline(np.percentile(arr, 90), color="tab:red", linestyle=":", linewidth=1.6, label=f"p90 {np.percentile(arr, 90):.2f}")
            ax.legend()
        else:
            ax.text(0.5, 0.5, "No values", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def draw_label(image, text, origin=(16, 32)):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)


def draw_rectified_corners(image, corners, x_offset=0, color=(0, 255, 255)):
    for x, y in corners.reshape(-1, 2):
        cv2.circle(image, (int(round(x + x_offset)), int(round(y))), 4, color, -1, cv2.LINE_AA)


def save_rectified_alignment_example(path, example, rectification, calib1, calib2, image_size):
    if example is None:
        return False

    r1, r2, p1, p2 = rectification
    map1x, map1y = cv2.initUndistortRectifyMap(
        calib1["camera_matrix"], calib1["dist_coeffs"], r1, p1, image_size, cv2.CV_16SC2
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        calib2["camera_matrix"], calib2["dist_coeffs"], r2, p2, image_size, cv2.CV_16SC2
    )
    rect1 = cv2.remap(example["image1"], map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(example["image2"], map2x, map2y, cv2.INTER_LINEAR)

    rect_corners1 = cv2.undistortPoints(
        example["corners1"], calib1["camera_matrix"], calib1["dist_coeffs"], R=r1, P=p1
    )
    rect_corners2 = cv2.undistortPoints(
        example["corners2"], calib2["camera_matrix"], calib2["dist_coeffs"], R=r2, P=p2
    )

    combined = np.hstack([rect1, rect2])
    height, width = combined.shape[:2]
    single_width = rect1.shape[1]
    for y in range(40, height, 60):
        cv2.line(combined, (0, y), (width - 1, y), (0, 255, 0), 1, cv2.LINE_AA)
    cv2.line(combined, (single_width, 0), (single_width, height - 1), (255, 255, 255), 2, cv2.LINE_AA)
    draw_rectified_corners(combined, rect_corners1, color=(0, 255, 255))
    draw_rectified_corners(combined, rect_corners2, x_offset=single_width, color=(0, 128, 255))
    draw_label(combined, f"Rectified RGB1 pair {example['pair_id']}", (16, 32))
    draw_label(combined, "Rectified RGB2", (single_width + 16, 32))
    draw_label(
        combined,
        f"vertical mean={example['vertical_mean_px']:.2f}px, max={example['vertical_max_px']:.2f}px",
        (16, height - 18),
    )

    if not cv2.imwrite(str(path), combined):
        raise RuntimeError(f"Could not write {path}")
    return True


def fmt_px(value):
    return "n/a" if value is None else f"{value:.3f} px"


def write_readme(path, summary, settings):
    text = f"""# Stereo Calibration

## Summary

| Metric | Value |
| --- | ---: |
| Stereo RMS calibration error | {fmt_px(summary["stereo_rms_error_px"])} |
| Rectification vertical error, mean / p90 / max | {fmt_px(summary["rectification_vertical_error_px"]["mean"])} / {fmt_px(summary["rectification_vertical_error_px"]["p90"])} / {fmt_px(summary["rectification_vertical_error_px"]["max"])} |
| Epipolar error before rectification, mean / p90 / max | {fmt_px(summary["epipolar_error_before_rectification_px"]["mean"])} / {fmt_px(summary["epipolar_error_before_rectification_px"]["p90"])} / {fmt_px(summary["epipolar_error_before_rectification_px"]["max"])} |
| Baseline | {summary["baseline_m"]:.4f} m |
| Evaluated pairs | {summary["evaluated_pairs"]} |

Lower is better for all pixel-error metrics. The rectification vertical error is
the remaining y-mismatch between corresponding checkerboard corners after
`cv2.stereoRectify`; good rectification should make this close to zero.

## Commands

Run from this folder:

```powershell
python 01_evaluate_stereo_image_set.py
python 02_stereo_calibrate_rgb1_rgb2_fixed_intrinsics.py
python 03_save_stereo_calibration_pair_image.py
python 04_evaluate_stereo_calibration.py
```

## Outputs

- `outputs/stereo_calibration_pair_example.png`
- `outputs/stereo_calibration_eval.json`
- `outputs/stereo_calibration_eval_per_pair.csv`
- `outputs/stereo_calibration_epipolar_error_plot.png`
- `outputs/stereo_calibration_rectification_vertical_error_plot.png`
- `outputs/stereo_calibration_error_histograms.png`
- `outputs/stereo_rectified_alignment_example.png`
- `config/stereo_rgb1_rgb2_extrinsics.npz`

## Inputs

- Stereo pairs: `{settings["pair_dir"]}`
- RGB1 intrinsics: `{settings["rgb1_calibration"]}`
- RGB2 intrinsics: `{settings["rgb2_calibration"]}`
- Stereo extrinsics: `{settings["stereo_calibration"]}`
"""
    path.write_text(text, encoding="utf-8")


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    calib1 = load_intrinsics(args.rgb1_calibration)
    calib2 = load_intrinsics(args.rgb2_calibration)
    stereo = load_stereo(args.stereo_calibration)
    if calib1["image_size"] != calib2["image_size"]:
        raise ValueError("RGB1 and RGB2 calibration image sizes do not match.")

    image_size = tuple(int(v) for v in calib1["image_size"])
    r1, r2, p1, p2, _, _, _ = cv2.stereoRectify(
        calib1["camera_matrix"],
        calib1["dist_coeffs"],
        calib2["camera_matrix"],
        calib2["dist_coeffs"],
        image_size,
        stereo["R"],
        stereo["t"],
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    rectification = (r1, r2, p1, p2)

    rows = []
    all_epipolar = []
    all_vertical = []
    skipped = []
    alignment_example = None

    for pair_id, rgb1_path, rgb2_path in stereo_pair_paths(args.pair_dir, stereo["used_pair_ids"]):
        image1 = cv2.imread(str(rgb1_path))
        image2 = cv2.imread(str(rgb2_path))
        if image1 is None or image2 is None:
            skipped.append({"pair_id": pair_id, "reason": "unreadable image"})
            continue

        found1, corners1 = detect_checkerboard(image1)
        found2, corners2 = detect_checkerboard(image2)
        if not (found1 and found2):
            skipped.append({"pair_id": pair_id, "reason": f"checkerboard missing: rgb1={found1}, rgb2={found2}"})
            continue

        epipolar = epipolar_errors_before_rectification(corners1, corners2, stereo["F"])
        vertical = rectified_vertical_errors(corners1, corners2, rectification, calib1, calib2)
        all_epipolar.extend(epipolar.tolist())
        all_vertical.extend(vertical.tolist())
        epi_stats = summarize(epipolar)
        vert_stats = summarize(vertical)
        if alignment_example is None:
            alignment_example = {
                "pair_id": pair_id,
                "image1": image1,
                "image2": image2,
                "corners1": corners1,
                "corners2": corners2,
                "vertical_mean_px": vert_stats["mean"],
                "vertical_max_px": vert_stats["max"],
            }
        rows.append(
            {
                "pair_id": pair_id,
                "epipolar_error_before_rectification_mean_px": epi_stats["mean"],
                "epipolar_error_before_rectification_p90_px": epi_stats["p90"],
                "epipolar_error_before_rectification_max_px": epi_stats["max"],
                "rectification_vertical_error_mean_px": vert_stats["mean"],
                "rectification_vertical_error_p90_px": vert_stats["p90"],
                "rectification_vertical_error_max_px": vert_stats["max"],
            }
        )

    summary = {
        "stereo_rms_error_px": stereo["stereo_rms_error_px"],
        "baseline_m": stereo["baseline_m"],
        "evaluated_pairs": len(rows),
        "skipped_pairs": len(skipped),
        "epipolar_error_before_rectification_px": summarize(all_epipolar),
        "rectification_vertical_error_px": summarize(all_vertical),
    }
    settings = {
        "pair_dir": str(args.pair_dir),
        "rgb1_calibration": str(args.rgb1_calibration),
        "rgb2_calibration": str(args.rgb2_calibration),
        "stereo_calibration": str(args.stereo_calibration),
    }
    result = {"settings": settings, "summary": summary, "skipped_pairs": skipped}

    save_json(args.output_dir / "stereo_calibration_eval.json", result)
    save_csv(args.output_dir / "stereo_calibration_eval_per_pair.csv", rows)
    save_metric_plot(
        args.output_dir / "stereo_calibration_epipolar_error_plot.png",
        rows,
        "epipolar_error_before_rectification",
        "Epipolar Error Before Rectification",
    )
    save_metric_plot(
        args.output_dir / "stereo_calibration_rectification_vertical_error_plot.png",
        rows,
        "rectification_vertical_error",
        "Rectification Vertical Error",
    )
    save_error_histograms(
        args.output_dir / "stereo_calibration_error_histograms.png",
        all_epipolar,
        all_vertical,
    )
    wrote_alignment = save_rectified_alignment_example(
        args.output_dir / "stereo_rectified_alignment_example.png",
        alignment_example,
        rectification,
        calib1,
        calib2,
        image_size,
    )
    write_readme(args.readme, summary, settings)

    print("\nStereo calibration evaluation")
    print("-----------------------------")
    print(f"Stereo RMS calibration error: {summary['stereo_rms_error_px']:.3f} px")
    print(
        "Rectification vertical error mean/p90/max: "
        f"{fmt_px(summary['rectification_vertical_error_px']['mean'])} / "
        f"{fmt_px(summary['rectification_vertical_error_px']['p90'])} / "
        f"{fmt_px(summary['rectification_vertical_error_px']['max'])}"
    )
    print(
        "Epipolar error before rectification mean/p90/max: "
        f"{fmt_px(summary['epipolar_error_before_rectification_px']['mean'])} / "
        f"{fmt_px(summary['epipolar_error_before_rectification_px']['p90'])} / "
        f"{fmt_px(summary['epipolar_error_before_rectification_px']['max'])}"
    )
    print(f"Evaluated pairs: {summary['evaluated_pairs']}")
    print(f"Saved {args.output_dir / 'stereo_calibration_eval.json'}")
    print(f"Saved {args.output_dir / 'stereo_calibration_eval_per_pair.csv'}")
    print(f"Saved {args.output_dir / 'stereo_calibration_epipolar_error_plot.png'}")
    print(f"Saved {args.output_dir / 'stereo_calibration_rectification_vertical_error_plot.png'}")
    print(f"Saved {args.output_dir / 'stereo_calibration_error_histograms.png'}")
    if wrote_alignment:
        print(f"Saved {args.output_dir / 'stereo_rectified_alignment_example.png'}")
    print(f"Updated {args.readme}")


if __name__ == "__main__":
    main()
