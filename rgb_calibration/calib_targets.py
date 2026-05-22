"""RGB1 / RGB2 checkerboard calibration paths and camera indices."""

import sys
from pathlib import Path

_STEREO_CALIB_DIR = Path(__file__).resolve().parents[1] / "stereo_calibration"
if str(_STEREO_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_STEREO_CALIB_DIR))

from hardware_settings import CONFIG_DIR, RGB1_CAMERA_INDEX, RGB2_CAMERA_INDEX

FRAME_SIZE = (1280, 720)

TARGETS = {
    "rgb1": {
        "label": "RGB1",
        "camera_index": RGB1_CAMERA_INDEX,
        "image_dir": Path("calibration_images"),
        "local_npz": Path("camera_calibration_rgb1.npz"),
        "config_npz": CONFIG_DIR / "camera_calibration_rgb1.npz",
    },
    "rgb2": {
        "label": "RGB2",
        "camera_index": RGB2_CAMERA_INDEX,
        "image_dir": Path("calibration_images_rgb2"),
        "local_npz": Path("camera_calibration_rgb2.npz"),
        "config_npz": CONFIG_DIR / "camera_calibration_rgb2.npz",
    },
}


def resolve_camera(name):
    key = name.lower()
    if key not in TARGETS:
        raise ValueError(f"Unknown camera {name!r}. Use rgb1 or rgb2.")
    return TARGETS[key]
