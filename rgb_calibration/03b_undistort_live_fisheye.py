"""
Step 5: view OpenCV fisheye undistortion live.

Run:
    python 03b_undistort_live_fisheye.py

Controls:
    [ - decrease balance, more crop
    ] - increase balance, more field of view
    0 - balance = 0.0
    1 - balance = 1.0
    + - enlarge display preview
    - - shrink display preview
    q - quit
"""

from pathlib import Path

import cv2
import numpy as np


CALIBRATION_FILE = Path("camera_calibration_fisheye.npz")
CAMERA_INDEX = 0
DISPLAY_MAX_WIDTH = 1200
DISPLAY_MAX_HEIGHT = 900
DISPLAY_SCALE_STEP = 0.1


def open_camera(image_size):
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {CAMERA_INDEX}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(image_size[0]))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(image_size[1]))
    return cap


def make_maps(K, D, frame_size, balance):
    new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(
        K,
        D,
        frame_size,
        np.eye(3),
        balance=balance,
        new_size=frame_size,
    )
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K,
        D,
        np.eye(3),
        new_K,
        frame_size,
        cv2.CV_16SC2,
    )
    return map1, map2


def label_image(image, text):
    labeled = image.copy()
    cv2.rectangle(labeled, (0, 0), (labeled.shape[1], 45), (0, 0, 0), -1)
    cv2.putText(
        labeled,
        text,
        (20, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return labeled


def fit_to_screen(image, user_scale):
    """Scale only the displayed preview so the OpenCV window fits on screen."""
    h, w = image.shape[:2]
    auto_scale = min(DISPLAY_MAX_WIDTH / w, DISPLAY_MAX_HEIGHT / h, 1.0)
    scale = max(0.1, min(1.0, auto_scale * user_scale))
    if scale >= 0.999:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main():
    if not CALIBRATION_FILE.exists():
        print(f"Error: {CALIBRATION_FILE} not found. Run 02b_calibrate_camera_fisheye.py first.")
        return

    data = np.load(CALIBRATION_FILE)
    K = data["K"] if "K" in data.files else data["camera_matrix"]
    D = data["D"] if "D" in data.files else data["dist_coeffs"]
    image_size = tuple(data["image_size"].astype(int))

    try:
        cap = open_camera(image_size)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    actual_size = (
        int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
    )
    print(f"Camera resolution: {actual_size[0]}x{actual_size[1]}")
    print("Press '[' or ']' to change balance, '0'/'1' for endpoints, '+'/'-' to resize display, 'q' to quit.")

    balance = 0.0
    user_scale = 1.0
    map1, map2 = make_maps(K, D, actual_size, balance)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read frame.")
                continue

            frame_size = (frame.shape[1], frame.shape[0])
            if frame_size != actual_size:
                actual_size = frame_size
                map1, map2 = make_maps(K, D, actual_size, balance)

            undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            raw_labeled = label_image(frame, "Raw camera feed")
            undistorted_labeled = label_image(
                undistorted,
                f"Fisheye undistorted feed | balance={balance:.1f} | display scale={user_scale:.1f}",
            )
            combined = np.vstack((raw_labeled, undistorted_labeled))
            cv2.imshow("Fisheye Undistortion", fit_to_screen(combined, user_scale))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("["):
                balance = max(0.0, round(balance - 0.1, 1))
                map1, map2 = make_maps(K, D, actual_size, balance)
                print(f"balance = {balance:.1f}")
            if key == ord("]"):
                balance = min(1.0, round(balance + 0.1, 1))
                map1, map2 = make_maps(K, D, actual_size, balance)
                print(f"balance = {balance:.1f}")
            if key in (ord("0"), ord("1")):
                balance = float(chr(key))
                map1, map2 = make_maps(K, D, actual_size, balance)
                print(f"balance = {balance:.1f}")
            if key in (ord("+"), ord("=")):
                user_scale = min(2.0, round(user_scale + DISPLAY_SCALE_STEP, 1))
                print(f"display scale = {user_scale:.1f}")
            if key in (ord("-"), ord("_")):
                user_scale = max(0.2, round(user_scale - DISPLAY_SCALE_STEP, 1))
                print(f"display scale = {user_scale:.1f}")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
