"""
Step 1: capture checkerboard images for RGB camera calibration.

Run:
    python 01_capture_checkerboard_images.py

Controls:
    s - save the current frame if the checkerboard is detected
    q - quit
"""

from pathlib import Path

import cv2
import numpy as np


CHECKERBOARD = (9, 6)  # Inner corners for a 10 x 7 square checkerboard.
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
OUTPUT_DIR = Path("calibration_images")


def open_camera():
    """Open webcam and request 1280x720. The camera may choose a nearby mode."""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


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

    if hasattr(cv2, "findChessboardCornersSB"):
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
    OUTPUT_DIR.mkdir(exist_ok=True)
    image_index = next_image_index(OUTPUT_DIR)

    try:
        cap = open_camera()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"Camera resolution: {actual_width}x{actual_height}")
    print("Move the checkerboard around the frame. Press 's' to save, 'q' to quit.")

    last_found = False
    last_corners = None

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
                f"Checkerboard: {status} | saved next: calib_{image_index:03d}.png | s: save q: quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 255, 0) if found else (0, 0, 255),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Capture Checkerboard Images", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            if key == ord("s"):
                if not last_found or last_corners is None:
                    print("Not saved: checkerboard was not detected in this frame.")
                    continue

                output_path = OUTPUT_DIR / f"calib_{image_index:03d}.png"
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
