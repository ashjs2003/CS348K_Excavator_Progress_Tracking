"""
Load camera_calibration.npz and show an undistorted live webcam feed.

Run after calibration:
    python undistort_live_rgb.py

Controls:
    q - quit
"""

from pathlib import Path

import cv2
import numpy as np


CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CALIBRATION_FILE = Path("camera_calibration.npz")


def load_calibration(path):
    """Load camera matrix and distortion coefficients from an npz file."""
    if not path.exists():
        raise FileNotFoundError(
            f"{path} was not found. Run calibrate_rgb_camera.py first and press 'c' "
            "after collecting checkerboard detections."
        )

    data = np.load(path)
    required_keys = {"camera_matrix", "dist_coeffs", "image_size"}
    missing_keys = required_keys.difference(data.files)
    if missing_keys:
        raise KeyError(f"Calibration file is missing keys: {sorted(missing_keys)}")

    return data["camera_matrix"], data["dist_coeffs"], tuple(data["image_size"].astype(int))


def open_camera():
    """Open the webcam and request the same resolution used for calibration."""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def main():
    try:
        camera_matrix, dist_coeffs, calibration_size = load_calibration(CALIBRATION_FILE)
        cap = open_camera()
    except (FileNotFoundError, KeyError, RuntimeError) as exc:
        print(f"Error: {exc}")
        return

    print("Undistorted live preview started. Press 'q' to quit.")

    map1 = None
    map2 = None
    current_size = None

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read frame from webcam.")
                continue

            frame_size = (frame.shape[1], frame.shape[0])
            if frame_size != current_size:
                current_size = frame_size
                new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
                    camera_matrix,
                    dist_coeffs,
                    frame_size,
                    alpha=1,
                    newImgSize=frame_size,
                )
                map1, map2 = cv2.initUndistortRectifyMap(
                    camera_matrix,
                    dist_coeffs,
                    None,
                    new_camera_matrix,
                    frame_size,
                    cv2.CV_16SC2,
                )

                if frame_size != calibration_size:
                    print(
                        "Warning: live frame size "
                        f"{frame_size} differs from calibration size {calibration_size}."
                    )

            undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            cv2.imshow("Original RGB Feed", frame)
            cv2.imshow("Undistorted RGB Feed", undistorted)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
