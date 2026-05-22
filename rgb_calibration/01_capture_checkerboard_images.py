"""
Step 1: capture checkerboard images for RGB camera calibration.

Run:
    python 01_capture_checkerboard_images.py --camera rgb1
    python 01_capture_checkerboard_images.py --camera rgb2

Controls:
    s - save the current frame if the checkerboard is detected
    q - quit
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from calib_targets import FRAME_SIZE, resolve_camera

_STEREO_CALIB_DIR = Path(__file__).resolve().parents[1] / "stereo_calibration"
if str(_STEREO_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_STEREO_CALIB_DIR))
from hardware_settings import open_camera

CHECKERBOARD = (9, 6)  # Inner corners for a 10 x 7 square checkerboard.


def parse_args():
    parser = argparse.ArgumentParser(description="Capture checkerboard images for RGB1 or RGB2.")
    parser.add_argument(
        "--camera",
        choices=["rgb1", "rgb2"],
        default="rgb1",
        help="Which camera to calibrate (default: rgb1)",
    )
    return parser.parse_args()


def next_image_index(output_dir):
    """Continue numbering after any existing calib_*.png files."""
    existing = sorted(output_dir.glob("calib_*.png"))
    if not existing:
        return 0

    indices = []
    for path in existing:
        try:
            indices.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(indices, default=-1) + 1


def detect_checkerboard(frame):
    """Detect corners for live preview using robust OpenCV fallbacks."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # findChessboardCornersSB can segfault on some macOS + OpenCV builds; use classic first.
    if sys.platform != "darwin" and hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, CHECKERBOARD, flags)
        if found:
            return True, corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, CHECKERBOARD, flags)
    if not found:
        equalized = cv2.equalizeHist(gray)
        found, corners = cv2.findChessboardCorners(equalized, CHECKERBOARD, flags)

    return found, corners


def main():
    args = parse_args()
    target = resolve_camera(args.camera)
    output_dir = target["image_dir"]
    camera_index = target["camera_index"]

    output_dir.mkdir(exist_ok=True)
    image_index = next_image_index(output_dir)

    try:
        cap = open_camera(camera_index, FRAME_SIZE)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Calibrating {target['label']} (camera index {camera_index})")
    print(f"Saving images to {output_dir.resolve()}")
    print(f"Camera resolution: {actual_width}x{actual_height}")
    print("Move the checkerboard around the frame. Press 's' to save, 'q' to quit.")

    last_found = False
    last_corners = None
    window = f"Capture Checkerboard — {target['label']}"

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read frame.")
                continue

            found, corners = detect_checkerboard(frame)
            last_found = found
            last_corners = corners

            preview = frame.copy()
            if found:
                cv2.drawChessboardCorners(preview, CHECKERBOARD, corners, found)

            status = "FOUND" if found else "not found"
            cv2.putText(
                preview,
                f"{target['label']} | checkerboard: {status} | next calib_{image_index:03d}.png | s save q quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (0, 255, 0) if found else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(window, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("s"):
                if not last_found or last_corners is None:
                    print("Not saved: checkerboard was not detected in this frame.")
                    continue

                output_path = output_dir / f"calib_{image_index:03d}.png"
                if cv2.imwrite(str(output_path), frame):
                    print(f"Saved {output_path}")
                    image_index += 1
                else:
                    print(f"Error: failed to write {output_path}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
