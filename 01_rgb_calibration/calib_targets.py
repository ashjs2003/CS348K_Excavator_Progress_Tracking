"""Left / right RGB checkerboard calibration paths and camera indices."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STEREO_CALIB_DIR = _REPO_ROOT / "02_stereo_calibration"
if str(_STEREO_CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(_STEREO_CALIB_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from utils.project_config import camera_index, frame_size, workflow_config_dir

FRAME_SIZE = frame_size()
CONFIG_DIR = workflow_config_dir("rgb_calibration")

TARGETS = {
    "l": {
        "label": "L",
        "camera_index": camera_index("L"),
        "image_dir": _REPO_ROOT / "00_data_capture" / "int_ext_calib_rgb" / "L",
        "normal_npz": CONFIG_DIR / "camera_calibration_L_normal.npz",
        "outlier_npz": CONFIG_DIR / "camera_calibration_L_normal_no_outliers.npz",
        "fisheye_npz": CONFIG_DIR / "camera_calibration_L_fisheye.npz",
        "fisheye_outlier_npz": CONFIG_DIR / "camera_calibration_L_fisheye_no_outliers.npz",
    },
    "r": {
        "label": "R",
        "camera_index": camera_index("R"),
        "image_dir": _REPO_ROOT / "00_data_capture" / "int_ext_calib_rgb" / "R",
        "normal_npz": CONFIG_DIR / "camera_calibration_R_normal.npz",
        "outlier_npz": CONFIG_DIR / "camera_calibration_R_normal_no_outliers.npz",
        "fisheye_npz": CONFIG_DIR / "camera_calibration_R_fisheye.npz",
        "fisheye_outlier_npz": CONFIG_DIR / "camera_calibration_R_fisheye_no_outliers.npz",
    },
}

ALIASES = {
    "left": "l",
    "l": "l",
    "rgb1": "l",
    "right": "r",
    "r": "r",
    "rgb2": "r",
}



def prompt_camera():
    """Ask which physical RGB camera should be used."""
    while True:
        choice = input("Which RGB camera is this for? [L/R]: ").strip().lower()
        if choice in ALIASES:
            return ALIASES[choice]
        print("Please enter L or R.")


def resolve_camera(name):
    key = ALIASES.get(name.lower())
    if key not in TARGETS:
        raise ValueError(f"Unknown camera {name!r}. Use L or R.")
    target = TARGETS[key].copy()
    target["key"] = key
    target["local_npz"] = target["normal_npz"]
    target["config_npz"] = target["normal_npz"]
    return target
