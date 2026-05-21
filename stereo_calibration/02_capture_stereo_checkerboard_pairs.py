"""
Capture stereo checkerboard image pairs from RGB1 and RGB2.

Run:
    python 02_capture_stereo_checkerboard_pairs.py

Controls:
    s - save pair only when checkerboard is detected in both cameras
    q - quit
"""

from pathlib import Path

import cv2
import numpy as np


RGB1_CALIBRATION_FILE = Path("../config/camera_calibration_rgb1.npz")
OUT_DIR = Path("stereo_pairs")
RGB1_CAMERA_INDEX = 0
RGB2_CAMERA_INDEX = 2


def load_rgb1_settings():
    data = np.load(RGB1_CALIBRATION_FILE)
    image_size = tuple(data["image_size"].astype(int))
    checkerboard = tuple(data["checkerboard_size"].astype(int))
    return image_size, checkerboard


def open_camera(index, image_size):
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(image_size[0]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(image_size[1]))
    return cap


def next_pair_index():
    existing = sorted(OUT_DIR.glob("rgb1_*.png"))
    indices = []
    for path in existing:
        try:
            indices.append(int(path.stem.split("_")[1]))
        except (IndexError, ValueError):
            pass
    return max(indices, default=-1) + 1


def detect_checkerboard(frame, checkerboard):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    if hasattr(cv2, "findChessboardCornersSB"):
        flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
        found, corners = cv2.findChessboardCornersSB(gray, checkerboard, flags)
        if found:
            return True, corners.astype(np.float32)

    flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    found, corners = cv2.findChessboardCorners(gray, checkerboard, flags)
    if found:
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
    return found, corners


def draw_status(frame, camera_name, found, pair_index):
    vis = frame.copy()
    status = "FOUND" if found else "not found"
    color = (0, 255, 0) if found else (0, 0, 255)
    text = f"{camera_name}: checkerboard {status} | next {pair_index:03d}"
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(vis, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)
    return vis


def side_by_side(left, right):
    if left.shape[0] != right.shape[0]:
        scale = left.shape[0] / right.shape[0]
        right = cv2.resize(right, (int(right.shape[1] * scale), left.shape[0]))
    return np.hstack([left, right])


def main():
    OUT_DIR.mkdir(exist_ok=True)
    image_size, checkerboard = load_rgb1_settings()

    cap1 = open_camera(RGB1_CAMERA_INDEX, image_size)
    cap2 = open_camera(RGB2_CAMERA_INDEX, image_size)
    pair_index = next_pair_index()

    actual1 = (
        int(cap1.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap1.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    actual2 = (
        int(cap2.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap2.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )

    print(f"Requested resolution: {image_size[0]}x{image_size[1]}")
    print(f"RGB1 actual resolution: {actual1[0]}x{actual1[1]}")
    print(f"RGB2 actual resolution: {actual2[0]}x{actual2[1]}")
    print("Press s to save when both checkerboards are detected, q to quit.")

    try:
        while True:
            ret1, frame1 = cap1.read()
            ret2, frame2 = cap2.read()
            if not ret1 or not ret2:
                print("Warning: failed to read one or both camera frames.")
                continue

            found1, corners1 = detect_checkerboard(frame1, checkerboard)
            found2, corners2 = detect_checkerboard(frame2, checkerboard)

            vis1 = draw_status(frame1, "RGB1", found1, pair_index)
            vis2 = draw_status(frame2, "RGB2", found2, pair_index)
            if found1:
                cv2.drawChessboardCorners(vis1, checkerboard, corners1, found1)
            if found2:
                cv2.drawChessboardCorners(vis2, checkerboard, corners2, found2)

            preview = side_by_side(vis1, vis2)
            cv2.putText(preview, "s save pair | q quit", (20, preview.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(preview, "s save pair | q quit", (20, preview.shape[0] - 24), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.imshow("Stereo Checkerboard Capture", preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("s"):
                if not (found1 and found2):
                    print("Not saved: checkerboard must be detected in both cameras.")
                    continue

                path1 = OUT_DIR / f"rgb1_{pair_index:03d}.png"
                path2 = OUT_DIR / f"rgb2_{pair_index:03d}.png"
                cv2.imwrite(str(path1), frame1)
                cv2.imwrite(str(path2), frame2)
                print(f"Saved stereo pair {pair_index:03d}: {path1}, {path2}")
                pair_index += 1
    finally:
        cap1.release()
        cap2.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
