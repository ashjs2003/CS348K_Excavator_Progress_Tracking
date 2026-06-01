"""Save left/right stereo pairs with numbered checkerboard corners.

Run from this folder:
    python 03b_save_numbered_stereo_corners.py
    python 03b_save_numbered_stereo_corners.py --pair-id 005

If --pair-id is omitted, every matching pair in stereo_pairs/ is processed.
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
    parser = argparse.ArgumentParser(description="Save stereo pairs with numbered checkerboard corners.")
    parser.add_argument("--pair-dir", type=Path, default=SCRIPT_DIR / "stereo_pairs")
    parser.add_argument("--pair-id", default=None, help="Optional single stereo image id, for example 005.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for numbered pair images. Defaults to --pair-dir.",
    )
    return parser.parse_args()


def stereo_pair_ids(pair_dir):
    ids = []
    for rgb1_path in sorted(left_image_dir(pair_dir).glob("*.png")):
        pair_id = rgb1_path.stem.split("_")[1]
        if right_image_path(pair_dir, pair_id).exists():
            ids.append(pair_id)
    return ids


def left_image_dir(pair_dir):
    left_dir = pair_dir / "L"
    return left_dir if left_dir.exists() else pair_dir


def right_image_dir(pair_dir):
    right_dir = pair_dir / "R"
    return right_dir if right_dir.exists() else pair_dir


def left_image_path(pair_dir, pair_id):
    image_dir = left_image_dir(pair_dir)
    for prefix in ("rgb1", "calib"):
        path = image_dir / f"{prefix}_{pair_id}.png"
        if path.exists():
            return path
    return image_dir / f"rgb1_{pair_id}.png"


def right_image_path(pair_dir, pair_id):
    image_dir = right_image_dir(pair_dir)
    for prefix in ("rgb2", "calib"):
        path = image_dir / f"{prefix}_{pair_id}.png"
        if path.exists():
            return path
    return image_dir / f"rgb2_{pair_id}.png"


def detect_corners(image):
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
                corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria)
                return True, corners

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    for gray in gray_variants:
        found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
        if found:
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            return True, corners

    return False, None


def draw_text(image, text, origin, scale, color, thickness=1):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def draw_header(image, text):
    cv2.rectangle(image, (0, 0), (image.shape[1], 42), (0, 0, 0), -1)
    draw_text(image, text, (12, 29), 0.78, (255, 255, 255), 2)


def draw_numbered_corners(image, corners):
    points = corners.reshape(-1, 2)
    cv2.drawChessboardCorners(image, CHECKERBOARD, corners, True)

    font_scale = max(0.38, min(image.shape[0], image.shape[1]) / 1700.0)
    radius = max(3, int(min(image.shape[:2]) / 180))
    offset = max(8, int(radius * 2.2))

    for idx, (x_float, y_float) in enumerate(points):
        x = int(round(x_float))
        y = int(round(y_float))
        cv2.circle(image, (x, y), radius, (0, 255, 255), -1, cv2.LINE_AA)
        cv2.circle(image, (x, y), radius + 1, (0, 0, 0), 1, cv2.LINE_AA)

        label = str(idx)
        text_x = min(max(x + offset, 2), image.shape[1] - 38)
        text_y = min(max(y - offset, 52), image.shape[0] - 6)
        draw_text(image, label, (text_x, text_y), font_scale, (0, 255, 255), 1)


def annotated_image(path, label):
    image = cv2.imread(str(path))
    if image is None:
        raise FileNotFoundError(f"Could not read {path}")

    found, corners = detect_corners(image)
    if found:
        draw_numbered_corners(image, corners)
    draw_header(image, f"{label}: {path.name}  checkerboard={'yes' if found else 'no'}")
    return image, found


def choose_pair_ids(pair_dir, requested_pair_id):
    ids = stereo_pair_ids(pair_dir)
    if not ids:
        raise RuntimeError(f"No matching rgb1_*.png/rgb2_*.png pairs found in {pair_dir}")

    if requested_pair_id is not None:
        if requested_pair_id not in ids:
            raise ValueError(f"Pair {requested_pair_id} not found. Available ids include: {', '.join(ids[:10])}")
        return [requested_pair_id]

    return ids


def save_numbered_pair(pair_dir, output_dir, pair_id):
    left, found_left = annotated_image(left_image_path(pair_dir, pair_id), "Left / RGB1")
    right, found_right = annotated_image(right_image_path(pair_dir, pair_id), "Right / RGB2")

    if left.shape[:2] != right.shape[:2]:
        right = cv2.resize(right, (left.shape[1], left.shape[0]), interpolation=cv2.INTER_AREA)

    separator = np.full((left.shape[0], 8, 3), 30, dtype=np.uint8)
    combined = np.hstack([left, separator, right])

    output_path = output_dir / f"numbered_stereo_corners_{pair_id}.png"
    if not cv2.imwrite(str(output_path), combined):
        raise RuntimeError(f"Could not write {output_path}")

    return {
        "pair_id": pair_id,
        "found_left": found_left,
        "found_right": found_right,
        "output_path": output_path,
    }


def main():
    args = parse_args()
    output_dir = args.output_dir if args.output_dir is not None else args.pair_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    pair_ids = choose_pair_ids(args.pair_dir, args.pair_id)
    saved = []
    failed = []

    for pair_id in pair_ids:
        try:
            result = save_numbered_pair(args.pair_dir, output_dir, pair_id)
            saved.append(result)
            print(
                f"Saved pair {pair_id}: left={result['found_left']} "
                f"right={result['found_right']} -> {result['output_path']}"
            )
        except Exception as exc:
            failed.append((pair_id, exc))
            print(f"Failed pair {pair_id}: {exc}")

    print(f"\nSaved {len(saved)}/{len(pair_ids)} numbered stereo pair images.")
    if failed:
        print("Failed pairs:")
        for pair_id, exc in failed:
            print(f"- {pair_id}: {exc}")


if __name__ == "__main__":
    main()
