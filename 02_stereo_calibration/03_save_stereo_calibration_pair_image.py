"""Save an annotated stereo calibration pair preview.

Run from this folder:
    python 03_save_stereo_calibration_pair_image.py
    python 03_save_stereo_calibration_pair_image.py --pair-id 005
"""

import argparse
import os
from pathlib import Path

os.environ["OPENCV_OPENCL_RUNTIME"] = "disabled"

import cv2
import numpy as np


cv2.ocl.setUseOpenCL(False)
SCRIPT_DIR = Path(__file__).resolve().parent
CHECKERBOARD = (9, 6)


def parse_args():
    parser = argparse.ArgumentParser(description="Save an annotated stereo calibration pair image.")
    parser.add_argument("--pair-dir", type=Path, default=SCRIPT_DIR / "stereo_pairs")
    parser.add_argument("--pair-id", default=None, help="Stereo image id, for example 005.")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "outputs" / "stereo_calibration_pair_example.png")
    return parser.parse_args()


def pair_ids(pair_dir):
    ids = []
    for rgb1_path in sorted(pair_dir.glob("rgb1_*.png")):
        pair_id = rgb1_path.stem.split("_")[1]
        if (pair_dir / f"rgb2_{pair_id}.png").exists():
            ids.append(pair_id)
    return ids


def detect_corners(image):
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


def draw_label(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 36), (0, 0, 0), -1)
    cv2.putText(image, text, (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2)


def annotated_image(path, label):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read {path}")
    found, corners = detect_corners(image)
    if found:
        cv2.drawChessboardCorners(image, CHECKERBOARD, corners, True)
    draw_label(image, f"{label}: {path.name} checkerboard={'yes' if found else 'no'}")
    return image


def main():
    args = parse_args()
    ids = pair_ids(args.pair_dir)
    if not ids:
        raise RuntimeError(f"No matching rgb1_*.png/rgb2_*.png pairs found in {args.pair_dir}")

    pair_id = args.pair_id if args.pair_id is not None else ids[0]
    if pair_id not in ids:
        raise ValueError(f"Pair {pair_id} not found. Available ids include: {', '.join(ids[:10])}")

    rgb1 = annotated_image(args.pair_dir / f"rgb1_{pair_id}.png", "RGB1")
    rgb2 = annotated_image(args.pair_dir / f"rgb2_{pair_id}.png", "RGB2")
    if rgb1.shape[:2] != rgb2.shape[:2]:
        rgb2 = cv2.resize(rgb2, (rgb1.shape[1], rgb1.shape[0]), interpolation=cv2.INTER_AREA)

    combined = np.hstack([rgb1, rgb2])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), combined):
        raise RuntimeError(f"Could not write {args.output}")
    print(f"Saved {args.output.resolve()}")


if __name__ == "__main__":
    main()
