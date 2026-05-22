"""
Shared hardware and path settings for stereo_calibration scripts.

Edit camera indices, LiDAR port, and baud rate here once; import from this
module in stereo_calibration, lidar_camera_calibration, and stereo_lidar_pointcloud scripts.

Run scripts from stereo_calibration/:
    python 02_capture_stereo_checkerboard_pairs.py
"""

from pathlib import Path
import sys

STEREO_CALIB_DIR = Path(__file__).resolve().parent
REPO_ROOT = STEREO_CALIB_DIR.parent
CONFIG_DIR = REPO_ROOT / "config"
STEREO_PAIRS_DIR = STEREO_CALIB_DIR / "stereo_pairs"

# OpenCV camera device indices
RGB1_CAMERA_INDEX = 0
RGB2_CAMERA_INDEX = 1

# RPLidar C1 (Slamtec default baud: 460800)
BAUDRATE = 460800
if sys.platform == "darwin":
    LIDAR_PORT = "/dev/cu.usbserial-1140"
else:
    LIDAR_PORT = "COM5"

# Calibration and data paths (absolute — safe regardless of cwd)
RGB1_CALIBRATION_FILE = CONFIG_DIR / "camera_calibration_rgb1.npz"
RGB2_CALIBRATION_FILE = CONFIG_DIR / "camera_calibration_rgb2_approx.npz"
RGB2_APPROX_CALIBRATION_FILE = RGB2_CALIBRATION_FILE
LIDAR_TO_RGB1_FILE = CONFIG_DIR / "lidar_to_camera_extrinsics.npz"
STEREO_EXTRINSICS_FILE = STEREO_CALIB_DIR / "stereo_rgb1_rgb2_extrinsics.npz"


def open_camera(index, image_size=None):
    """Open a webcam with a platform-appropriate OpenCV backend."""
    import cv2

    if sys.platform == "darwin":
        cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    elif sys.platform == "win32":
        cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
    else:
        cap = cv2.VideoCapture(index)

    if not cap.isOpened():
        cap = cv2.VideoCapture(index)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {index}.")

    if image_size is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(image_size[0]))
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(image_size[1]))
    return cap
