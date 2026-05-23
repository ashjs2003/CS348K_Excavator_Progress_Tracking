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

import argparse
from pathlib import Path
import sys

import cv2
import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from calib_targets import prompt_camera, resolve_camera

DISPLAY_MAX_WIDTH = 1400
DISPLAY_MAX_HEIGHT = 900
DISPLAY_SCALE_STEP = 0.1


def parse_args():
    parser = argparse.ArgumentParser(description="View normal undistortion for the L or R RGB camera.")
    parser.add_argument(
        "--camera",
        choices=["L", "R", "l", "r", "left", "right", "rgb1", "rgb2"],
        default=None,
        help="Which camera to view: L or R. If omitted, you will be prompted.",
    )
    return parser.parse_args()


def open_camera(camera_index, image_size):
    cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(camera_index)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}.")

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
    args = parse_args()
    camera_name = args.camera if args.camera is not None else prompt_camera()
    target = resolve_camera(camera_name)
    calibration_file = target["outlier_npz"] if target["outlier_npz"].exists() else target["normal_npz"]

    if not calibration_file.exists():
        print(f"Error: {calibration_file} not found. Run 02_calibrate_camera_normal.py first.")
        return

    data = np.load(calibration_file)
    camera_matrix = data["camera_matrix"]
    dist_coeffs = data["dist_coeffs"]
    image_size = tuple(data["image_size"].astype(int))

    try:
        cap = open_camera(target["camera_index"], image_size)
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    print(f"Viewing {target['label']} with calibration: {calibration_file.resolve()}")
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
