"""Left-right photometric consistency on rectified stereo (OpenCV / Foundation disparity)."""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def photometric_reprojection_score(
    rect1_bgr: np.ndarray,
    rect2_bgr: np.ndarray,
    disparity: np.ndarray,
    max_samples: int = 80_000,
) -> dict:
    """
    Warp rectified right view to left using disparity; report mean absolute gray error.

    Assumes disparity[u,v] = x_left - x_right (OpenCV stereo convention on rectified pair).
    """
    gray1 = cv2.cvtColor(rect1_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray2 = cv2.cvtColor(rect2_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    h, w = gray1.shape
    valid = disparity > 0
    indices = np.flatnonzero(valid)
    if len(indices) == 0:
        return {
            "mean_photometric_error": None,
            "valid_pixel_pct": 0.0,
            "sample_count": 0,
        }
    if len(indices) > max_samples:
        indices = np.random.choice(indices, max_samples, replace=False)

    errors = []
    for flat in indices:
        v, u = divmod(int(flat), w)
        d = float(disparity[v, u])
        u_r = u - d
        if u_r < 0 or u_r >= w - 1:
            continue
        # Bilinear sample on right image
        val = cv2.remap(
            gray2,
            np.array([[[u_r, float(v)]]], dtype=np.float32),
            None,
            cv2.INTER_LINEAR,
        )[0, 0]
        if np.isfinite(val):
            errors.append(abs(float(gray1[v, u]) - float(val)))

    if not errors:
        return {
            "mean_photometric_error": None,
            "valid_pixel_pct": 100.0 * float(np.count_nonzero(valid)) / valid.size,
            "sample_count": 0,
        }

    err = np.asarray(errors, dtype=np.float64)
    return {
        "mean_photometric_error": float(np.mean(err)),
        "median_photometric_error": float(np.median(err)),
        "valid_pixel_pct": 100.0 * float(np.count_nonzero(valid)) / valid.size,
        "sample_count": int(len(err)),
    }


def run_photometric_for_method(stereo_dir: Path, method_key: str, *, geometry: dict | None = None) -> dict | None:
    """method_key: 'opencv' or 'foundation'."""
    from depth_layout import resolve_path

    stereo_dir = Path(stereo_dir)
    rect1_path = resolve_path(stereo_dir, None, "rgb1_rectified.png")
    rect2_path = resolve_path(stereo_dir, None, "rgb2_rectified.png")
    if method_key not in ("opencv", "foundation"):
        return None
    if rect1_path is None or rect2_path is None:
        return None

    rect1 = cv2.imread(str(rect1_path))
    rect2 = cv2.imread(str(rect2_path))
    if rect1 is None or rect2 is None:
        return None

    disp_path = resolve_path(stereo_dir, method_key, "disparity.npy")
    if disp_path is not None:
        disparity = np.load(disp_path).astype(np.float32)
    elif method_key == "foundation" and geometry is not None:
        from evaluation.foundation_vis_disparity import load_foundation_disparity

        disparity, _meta = load_foundation_disparity(stereo_dir, geometry)
        if disparity is None:
            return None
    else:
        return None

    return photometric_reprojection_score(rect1, rect2, disparity)
