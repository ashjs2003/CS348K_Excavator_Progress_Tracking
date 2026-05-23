"""
Calibrate an RGB webcam with a printed checkerboard.

Run on Windows:
    python calibrate_rgb_camera.py

Controls:
    s - detect and save the current checkerboard view
    c - calibrate using all saved detections
    q - quit
"""

from pathlib import Path

import cv2
import numpy as np


CHECKERBOARD = (9, 6)  # OpenCV inner corners: 10x7 squares -> 9x6 corners
CHECKERBOARD_CANDIDATES = (CHECKERBOARD, (CHECKERBOARD[1], CHECKERBOARD[0]))
SQUARE_SIZE = 0.025  # meters
CAMERA_INDEX = 0
FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
CALIBRATION_IMAGE_DIR = Path("calibration_images")
OUTPUT_FILE = Path("camera_calibration.npz")


def make_object_points(pattern_size=CHECKERBOARD):
    """Create the real-world checkerboard corner coordinates."""
    objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
    objp[:, :2] = np.mgrid[0 : pattern_size[0], 0 : pattern_size[1]].T.reshape(-1, 2)
    objp *= SQUARE_SIZE
    return objp


def checkerboard_gray_images(frame):
    """Return grayscale variants that make checkerboard detection less brittle."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    blurred = cv2.GaussianBlur(equalized, (3, 3), 0)
    return (gray, equalized, blurred)


def detect_checkerboard(frame, criteria):
    """Detect checkerboard corners with robust fallbacks."""
    classic_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
    sb_flags = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY

    for pattern_size in CHECKERBOARD_CANDIDATES:
        for gray in checkerboard_gray_images(frame):
            found, corners = cv2.findChessboardCornersSB(gray, pattern_size, sb_flags)
            if found:
                return True, pattern_size, gray, corners.astype(np.float32)

            found, corners = cv2.findChessboardCorners(gray, pattern_size, classic_flags)
            if found:
                refined_corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    (11, 11),
                    (-1, -1),
                    criteria,
                )
                return True, pattern_size, gray, refined_corners

    return False, None, None, None


def open_camera():
    """Open the webcam and request a fixed 1280x720 capture size."""
    cap = cv2.VideoCapture(CAMERA_INDEX, cv2.CAP_DSHOW)
    if not cap.isOpened():
        cap = cv2.VideoCapture(CAMERA_INDEX)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open webcam at index {CAMERA_INDEX}.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    return cap


def calibrate_camera(objpoints, imgpoints, image_size):
    """Run OpenCV camera calibration and return the results."""
    if len(objpoints) < 5:
        raise RuntimeError(
            f"Only {len(objpoints)} valid images captured. Capture at least 5, "
            "and 10-20 from varied angles is better."
        )

    rms, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
        objpoints,
        imgpoints,
        image_size,
        None,
        None,
    )
    return rms, camera_matrix, dist_coeffs, rvecs, tvecs


def main():
    CALIBRATION_IMAGE_DIR.mkdir(exist_ok=True)

    objpoints = []
    imgpoints = []
    saved_count = 0
    image_size = None

    try:
        cap = open_camera()
    except RuntimeError as exc:
        print(f"Error: {exc}")
        return

    print("Live preview started.")
    print("Press 's' to save a checkerboard detection, 'c' to calibrate, or 'q' to quit.")

    criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
        30,
        0.001,
    )

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Warning: failed to read frame from webcam.")
                continue

            image_size = (frame.shape[1], frame.shape[0])
            preview = frame.copy()
            cv2.putText(
                preview,
                f"Saved detections: {saved_count} | s: save  c: calibrate  q: quit",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("RGB Camera Calibration", preview)

            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break

            if key == ord("s"):
                found, pattern_size, gray, refined_corners = detect_checkerboard(
                    frame,
                    criteria,
                )

                if not found:
                    print(
                        "Checkerboard not found. Tips: keep all inner corners visible, "
                        "use even lighting, avoid glare, and leave a white border around "
                        "the printed board."
                    )
                    continue

                objpoints.append(make_object_points(pattern_size))
                imgpoints.append(refined_corners)
                saved_count += 1

                image_path = CALIBRATION_IMAGE_DIR / f"calibration_{saved_count:03d}.png"
                cv2.imwrite(str(image_path), frame)

                drawn = frame.copy()
                cv2.drawChessboardCorners(drawn, pattern_size, refined_corners, found)
                cv2.imshow("Detected Checkerboard", drawn)
                print(
                    f"Saved detection {saved_count}: {image_path} "
                    f"(detected pattern {pattern_size})"
                )

            if key == ord("c"):
                if image_size is None:
                    print("No camera frame available yet.")
                    continue

                try:
                    rms, camera_matrix, dist_coeffs, _, _ = calibrate_camera(
                        objpoints,
                        imgpoints,
                        image_size,
                    )
                except RuntimeError as exc:
                    print(f"Error: {exc}")
                    continue

                print("\nCalibration complete.")
                print(f"RMS reprojection error: {rms}")
                print("Camera matrix:")
                print(camera_matrix)
                print("Distortion coefficients:")
                print(dist_coeffs)

                np.savez(
                    OUTPUT_FILE,
                    camera_matrix=camera_matrix,
                    dist_coeffs=dist_coeffs,
                    image_size=np.array(image_size),
                    checkerboard_size=np.array(CHECKERBOARD),
                    square_size=np.array(SQUARE_SIZE),
                )
                print(f"Saved calibration results to {OUTPUT_FILE.resolve()}\n")

    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
