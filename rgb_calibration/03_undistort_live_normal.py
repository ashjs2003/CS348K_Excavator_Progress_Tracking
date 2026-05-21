"""
Step 3: view normal pinhole undistortion live.

Run:
    python 03_undistort_live_normal.py

Controls:
    0 - alpha = 0, more crop
    1 - alpha = 1, more field of view
    + - enlarge display preview
    - - shrink display preview
    q - quit
"""

from pathlib import Path

import cv2
import numpy as np


CALIBRATION_FILE = Path("camera_calibration_normal.npz")
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


def make_maps(camera_matrix, dist_coeffs, frame_size, alpha):
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        frame_size,
        alpha,
        frame_size,
    )
    map1, map2 = cv2.initUndistortRectifyMap(
        camera_matrix,
        dist_coeffs,
        None,
        new_camera_matrix,
        frame_size,
        cv2.CV_16SC2,
    )
    return map1, map2, roi


def crop_to_roi(image, roi):
    x, y, w, h = roi
    if w <= 0 or h <= 0:
        return image
    return image[y : y + h, x : x + w]


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
    """Scale only the display image so the OpenCV window fits on screen."""
    h, w = image.shape[:2]
    auto_scale = min(DISPLAY_MAX_WIDTH / w, DISPLAY_MAX_HEIGHT / h, 1.0)
    scale = max(0.1, min(1.0, auto_scale * user_scale))
    if scale >= 0.999:
        return image
    return cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def main():
    if not CALIBRATION_FILE.exists():
        print(f"Error: {CALIBRATION_FILE} not found. Run 02_calibrate_camera_normal.py first.")
        return

    data = np.load(CALIBRATION_FILE)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
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
    print("Press '0' for alpha=0, '1' for alpha=1, '+'/'-' to resize display, 'q' to quit.")

    alpha = 0
    user_scale = 1.0
    map1, map2, roi = make_maps(camera_matrix, dist_coeffs, actual_size, alpha)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read frame.")
                continue

            frame_size = (frame.shape[1], frame.shape[0])
            if frame_size != actual_size:
                actual_size = frame_size
                map1, map2, roi = make_maps(camera_matrix, dist_coeffs, actual_size, alpha)

            undistorted = cv2.remap(frame, map1, map2, interpolation=cv2.INTER_LINEAR)
            if alpha == 0:
                undistorted_display = crop_to_roi(undistorted, roi)
                undistorted_display = cv2.resize(undistorted_display, frame_size)
            else:
                undistorted_display = undistorted

            raw_labeled = label_image(frame, "Raw camera feed")
            undistorted_labeled = label_image(
                undistorted_display,
                f"Undistorted feed | alpha={alpha} | display scale={user_scale:.1f}",
            )
            combined = np.vstack((raw_labeled, undistorted_labeled))
            cv2.imshow("Normal Undistortion", fit_to_screen(combined, user_scale))

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key in (ord("0"), ord("1")):
                alpha = int(chr(key))
                map1, map2, roi = make_maps(camera_matrix, dist_coeffs, actual_size, alpha)
                print(f"alpha = {alpha}")
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
