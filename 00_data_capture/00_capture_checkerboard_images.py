"""
Capture paired L/R checkerboard images for RGB intrinsic and stereo extrinsic calibration.

Run:
    python 00_capture_checkerboard_images.py
    python 00_capture_checkerboard_images.py --output-dir int_ext_calib_rgb

Controls:
    s - save the current L/R pair if the checkerboard is detected in both images
    q - quit
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parents[0]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.project_config import camera_index, frame_size, open_configured_camera


CHECKERBOARD = (9, 6)
PREVIEW_SCALE = 0.5


def parse_args():
    parser = argparse.ArgumentParser(description="Capture paired L/R checkerboard images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=_SCRIPT_DIR / "int_ext_calib_rgb",
        help="Dataset directory where L/calib_###.png and R/calib_###.png images are saved.",
    )
    return parser.parse_args()


def next_pair_index(output_dir):
    """Continue numbering after existing L/R calib_###.png pair files."""
    indices = []
    for side in ("L", "R"):
        for path in (output_dir / side).glob("calib_*.png"):
            try:
                indices.append(int(path.stem.split("_")[1]))
            except (IndexError, ValueError):
                pass
    return max(indices, default=-1) + 1


def detect_checkerboard(frame):
    """Detect checkerboard corners for live preview and save gating."""
    gray_original = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_variants = (
        gray_original,
        cv2.equalizeHist(gray_original),
        cv2.GaussianBlur(cv2.equalizeHist(gray_original), (3, 3), 0),
    )
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)

    if sys.platform != "darwin" and hasattr(cv2, "findChessboardCornersSB"):
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
            corners = cv2.cornerSubPix(gray, corners.astype(np.float32), (11, 11), (-1, -1), criteria)
            return True, corners

    return False, None


def draw_text(image, text, origin, color, scale=0.65, thickness=2):
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def annotate_preview(frame, label, found, corners):
    preview = frame.copy()
    if found:
        cv2.drawChessboardCorners(preview, CHECKERBOARD, corners, found)
    color = (0, 255, 0) if found else (0, 0, 255)
    status = "FOUND" if found else "not found"
    draw_text(preview, f"{label} checkerboard: {status}", (20, 35), color)
    return preview


def make_combined_preview(left_preview, right_preview, pair_index, output_dir):
    if left_preview.shape[:2] != right_preview.shape[:2]:
        right_preview = cv2.resize(
            right_preview,
            (left_preview.shape[1], left_preview.shape[0]),
            interpolation=cv2.INTER_AREA,
        )

    separator = np.full((left_preview.shape[0], 8, 3), 30, dtype=np.uint8)
    combined = np.hstack([left_preview, separator, right_preview])
    draw_text(
        combined,
        f"next L/R calib_{pair_index:03d}.png | s save pair | q quit | {output_dir}",
        (20, combined.shape[0] - 24),
        (255, 255, 255),
        scale=0.58,
        thickness=1,
    )
    if PREVIEW_SCALE != 1.0:
        combined = cv2.resize(
            combined,
            None,
            fx=PREVIEW_SCALE,
            fy=PREVIEW_SCALE,
            interpolation=cv2.INTER_AREA,
        )
    return combined


def save_pair(output_dir, pair_index, left_frame, right_frame):
    left_dir = output_dir / "L"
    right_dir = output_dir / "R"
    left_dir.mkdir(parents=True, exist_ok=True)
    right_dir.mkdir(parents=True, exist_ok=True)

    left_path = left_dir / f"calib_{pair_index:03d}.png"
    right_path = right_dir / f"calib_{pair_index:03d}.png"
    left_ok = cv2.imwrite(str(left_path), left_frame)
    right_ok = cv2.imwrite(str(right_path), right_frame)
    if not (left_ok and right_ok):
        raise RuntimeError(f"Failed to save pair {pair_index:03d} to {output_dir}")
    return left_path, right_path


def main():
    args = parse_args()
    output_dir = args.output_dir
    if not output_dir.is_absolute():
        output_dir = _SCRIPT_DIR / output_dir
    (output_dir / "L").mkdir(parents=True, exist_ok=True)
    (output_dir / "R").mkdir(parents=True, exist_ok=True)

    image_size = frame_size()
    left_index = camera_index("L")
    right_index = camera_index("R")
    pair_index = next_pair_index(output_dir)

    left_cap = None
    right_cap = None
    try:
        left_cap = open_configured_camera(left_index, image_size)
        right_cap = open_configured_camera(right_index, image_size)
    except RuntimeError as exc:
        if left_cap is not None:
            left_cap.release()
        if right_cap is not None:
            right_cap.release()
        print(f"Error: {exc}")
        return

    left_actual = (int(left_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(left_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    right_actual = (int(right_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(right_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))

    print(f"Capturing paired checkerboards for RGB intrinsic/extrinsic calibration")
    print(f"Left / RGB1 camera index: {left_index}, resolution: {left_actual[0]}x{left_actual[1]}")
    print(f"Right / RGB2 camera index: {right_index}, resolution: {right_actual[0]}x{right_actual[1]}")
    print(f"Saving pairs to: {output_dir.resolve()}")
    print("Move the checkerboard around both frames. Press 's' to save only when both are detected; 'q' to quit.")

    window = "Capture L/R Checkerboard Pair"

    try:
        while True:
            left_ok, left_frame = left_cap.read()
            right_ok, right_frame = right_cap.read()
            if not left_ok or not right_ok:
                print(f"Warning: failed to read frame. left={left_ok}, right={right_ok}")
                continue

            left_found, left_corners = detect_checkerboard(left_frame)
            right_found, right_corners = detect_checkerboard(right_frame)

            left_preview = annotate_preview(left_frame, "Left / RGB1", left_found, left_corners)
            right_preview = annotate_preview(right_frame, "Right / RGB2", right_found, right_corners)
            preview = make_combined_preview(left_preview, right_preview, pair_index, output_dir)
            cv2.imshow(window, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("s"):
                if not (left_found and right_found):
                    print(f"Not saved: checkerboard detection left={left_found}, right={right_found}.")
                    continue

                try:
                    left_path, right_path = save_pair(output_dir, pair_index, left_frame, right_frame)
                except RuntimeError as exc:
                    print(f"Error: {exc}")
                    continue

                print(f"Saved pair {pair_index:03d}: {left_path.name}, {right_path.name}")
                pair_index += 1

    finally:
        left_cap.release()
        right_cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
