"""Load rectified stereo geometry and per-method metric depth maps (HxW, meters)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from calib_utils import load_camera_calibration, load_stereo_rgb1_to_rgb2
from dav2_scale import depth_map_from_disparity
from stereo_shared import rectify_stereo_pair

STEREO_GEOMETRY_FILE = "stereo_geometry.npz"

METHODS = {
    "opencv": {
        "suffix": "",
        "disparity_name": "disparity.npy",
        "depth_name": "depth_metric_opencv.npy",
    },
    "dav2": {
        "suffix": "_dav2",
        "disparity_name": None,
        "depth_name": "depth_metric_dav2.npy",
    },
    "foundation": {
        "suffix": "_foundation",
        "disparity_name": "disparity_foundation.npy",
        "depth_name": None,
    },
}


def save_stereo_geometry(out_dir: Path, R1, P1, Q, baseline_m: float, image_size) -> Path:
    """Persist rectification outputs used by evaluation (written by 02_make_stereo_pointcloud)."""
    out_dir = Path(out_dir)
    path = out_dir / STEREO_GEOMETRY_FILE
    np.savez(
        path,
        R1=R1.astype(np.float64),
        P1=P1.astype(np.float64),
        Q=Q.astype(np.float64),
        baseline_m=np.array(float(baseline_m)),
        image_size=np.array(image_size, dtype=np.int32),
    )
    return path


def load_stereo_geometry(stereo_dir: Path) -> dict:
    path = Path(stereo_dir) / STEREO_GEOMETRY_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Re-run: python 02_make_stereo_pointcloud.py --run <id>"
        )
    data = np.load(path)
    size = tuple(int(x) for x in data["image_size"])
    return {
        "path": path,
        "R1": data["R1"].astype(np.float64),
        "P1": data["P1"].astype(np.float64),
        "Q": data["Q"].astype(np.float64),
        "baseline_m": float(np.asarray(data["baseline_m"]).reshape(())),
        "image_size": size,
    }


def load_or_compute_stereo_geometry(stereo_dir: Path, rgb1_path: Path, rgb2_path: Path) -> dict:
    """Use saved geometry or recompute from capture (for older runs)."""
    geo_path = Path(stereo_dir) / STEREO_GEOMETRY_FILE
    if geo_path.is_file():
        return load_stereo_geometry(stereo_dir)

    rgb1_calib = load_camera_calibration("camera_calibration_rgb1.npz")
    rgb2_calib = load_camera_calibration("camera_calibration_rgb2.npz", "camera_calibration_rgb2_approx.npz")
    _, R, t = load_stereo_rgb1_to_rgb2()
    image1 = cv2.imread(str(rgb1_path))
    image2 = cv2.imread(str(rgb2_path))
    if image1 is None or image2 is None:
        raise RuntimeError("Could not load capture images for stereo geometry.")
    stereo = rectify_stereo_pair(image1, image2, rgb1_calib, rgb2_calib, R, t)
    save_stereo_geometry(
        stereo_dir,
        stereo["R1"],
        stereo["P1"],
        stereo["Q"],
        stereo["baseline_m"],
        stereo["image_size"],
    )
    return load_stereo_geometry(stereo_dir)


def depth_from_disparity_file(disparity_path: Path, Q: np.ndarray) -> np.ndarray:
    disp = np.load(disparity_path).astype(np.float32)
    return depth_map_from_disparity(disp, Q)


def load_metric_depth(stereo_dir: Path, method: str, geometry: dict) -> np.ndarray | None:
    """
    Return metric Z (m) per pixel in rectified left frame, or None if files missing.
    """
    if method not in METHODS:
        raise ValueError(f"Unknown method {method!r}; expected one of {list(METHODS)}")

    stereo_dir = Path(stereo_dir)
    meta = METHODS[method]
    depth_path = stereo_dir / meta["depth_name"] if meta["depth_name"] else None
    if depth_path is not None and depth_path.is_file():
        depth = np.load(depth_path).astype(np.float64)
        return depth

    disp_name = meta["disparity_name"]
    if disp_name is None:
        return None
    disp_path = stereo_dir / disp_name
    if not disp_path.is_file():
        return None
    return depth_from_disparity_file(disp_path, geometry["Q"])


def discover_methods(stereo_dir: Path) -> list[str]:
    """Methods that have enough files to evaluate."""
    stereo_dir = Path(stereo_dir)
    available = []
    for name, meta in METHODS.items():
        if meta["depth_name"] and (stereo_dir / meta["depth_name"]).is_file():
            available.append(name)
            continue
        if meta["disparity_name"] and (stereo_dir / meta["disparity_name"]).is_file():
            available.append(name)
    return available
