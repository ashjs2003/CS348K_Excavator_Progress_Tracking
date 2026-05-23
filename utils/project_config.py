"""Small helpers for reading the repo-level config.yaml."""

from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_FILE = REPO_ROOT / "config.yaml"


def load_config():
    """Parse the simple two-level config.yaml used by this project."""
    data = {}
    section = None
    for raw_line in CONFIG_FILE.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].rstrip()
        if not line:
            continue
        if not line.startswith(" "):
            section = line.rstrip(":")
            data[section] = {}
            continue
        if section and ":" in line:
            key, value = line.strip().split(":", 1)
            data[section][key.strip()] = value.strip().strip("'\"")
    return data


def config_path(section, key):
    """Return an absolute path for a path-valued config entry."""
    return REPO_ROOT / load_config()[section][key]


def calibration_file(key):
    """Return an absolute path from config.yaml's calibration_files section."""
    return config_path("calibration_files", key)


def workflow_config_dir(key):
    """Return an absolute path from config.yaml's workflow_config_dirs section."""
    return config_path("workflow_config_dirs", key)


def hardware_value(key, default=None, cast=str):
    """Return a scalar value from config.yaml's hardware section."""
    value = load_config().get("hardware", {}).get(key)
    if value is None:
        return default
    if cast is bool:
        return value.lower() in {"1", "true", "yes", "on"}
    return cast(value)


def camera_index(side):
    """Return configured camera index for L/R."""
    if str(side).lower() in {"l", "left", "rgb1"}:
        return hardware_value("left_camera_index", 0, int)
    if str(side).lower() in {"r", "right", "rgb2"}:
        return hardware_value("right_camera_index", 1, int)
    raise ValueError(f"Unknown camera side {side!r}. Use L or R.")


def frame_size():
    """Return configured camera frame size as (width, height)."""
    return (
        hardware_value("frame_width", 1280, int),
        hardware_value("frame_height", 720, int),
    )


def lidar_baudrate():
    """Return configured LiDAR baudrate."""
    return hardware_value("lidar_baudrate", 460800, int)


def lidar_port():
    """Return platform-appropriate configured LiDAR serial port."""
    if sys.platform == "darwin":
        return hardware_value("lidar_port_macos", "/dev/cu.usbserial-1140", str)
    return hardware_value("lidar_port_windows", "COM5", str)


def open_configured_camera(index, image_size=None):
    """Open a camera index with a platform-appropriate OpenCV backend."""
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
