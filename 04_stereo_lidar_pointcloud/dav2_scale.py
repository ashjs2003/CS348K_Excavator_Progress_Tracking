"""Scale Depth Anything V2 relative depth to metric meters using a reference depth map."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


def depth_map_from_disparity(disparity: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Metric Z (m) per pixel from OpenCV disparity and Q (invalid = nan)."""
    points_3d = cv2.reprojectImageTo3D(disparity.astype(np.float32), Q)
    depth = points_3d[:, :, 2].astype(np.float64)
    depth[~np.isfinite(depth)] = np.nan
    depth[disparity <= 0] = np.nan
    if np.nanmedian(depth) < 0:
        depth = -depth
    return depth


def depth_matches_reference(
    depth_m: np.ndarray,
    reference_m: float,
    tolerance_m: float,
) -> np.ndarray:
    """Pixels with finite positive depth within tolerance of a reference distance (m)."""
    valid = np.isfinite(depth_m) & (depth_m > 0)
    return valid & (np.abs(depth_m - reference_m) <= tolerance_m)


def opencv_gt_anchor_mask(
    opencv_depth_m: np.ndarray,
    target_gt_m: float,
    wall_gt_m: float,
    tolerance_m: float,
) -> np.ndarray:
    """Union of OpenCV pixels matching ruler target (pair txt) or back-wall GT."""
    return depth_matches_reference(opencv_depth_m, target_gt_m, tolerance_m) | depth_matches_reference(
        opencv_depth_m, wall_gt_m, tolerance_m
    )


def _linear_fit(reference_m: np.ndarray, relative: np.ndarray):
    rel = relative.astype(np.float64)
    ref = reference_m.astype(np.float64)
    A = np.column_stack([rel, np.ones(len(rel))])
    scale, shift = np.linalg.lstsq(A, ref, rcond=None)[0]
    return float(scale), float(shift)


def _pick_relative_variant(da: np.ndarray, reference_m: np.ndarray, valid: np.ndarray):
    ref = reference_m[valid]
    if len(ref) < 50:
        raise RuntimeError("Too few valid reference depth pixels to scale Depth Anything V2.")

    variants = {
        "direct": da,
        "inverse": 1.0 / np.clip(da, 1e-6, None),
        "flipped": float(np.nanmax(da)) - da,
    }
    best_name = "direct"
    best_score = -np.inf
    best_scale, best_shift = 1.0, 0.0
    for name, rel_map in variants.items():
        rel = rel_map[valid]
        if np.std(rel) < 1e-9:
            continue
        score = abs(float(np.corrcoef(rel, ref)[0, 1]))
        scale, shift = _linear_fit(ref, rel)
        if score > best_score:
            best_score = score
            best_name = name
            best_scale, best_shift = scale, shift
    return best_name, best_scale, best_shift, variants[best_name]


def metric_depth_from_relative(
    da_relative: np.ndarray,
    reference_depth_m: np.ndarray,
    max_samples: int = 200_000,
    *,
    fit_mask: np.ndarray | None = None,
    min_fit_pixels: int = 50,
) -> tuple[np.ndarray, dict]:
    """
    Fit metric depth = scale * relative + shift using pixels where reference depth is valid.

    If fit_mask is set, only those pixels are used for the linear fit (e.g. OpenCV pixels
    that match ruler target / wall GT). The full image is still scaled with the fitted
    scale and shift.
    """
    valid = np.isfinite(reference_depth_m) & (reference_depth_m > 0.05)
    if fit_mask is not None:
        if fit_mask.shape != reference_depth_m.shape:
            raise ValueError("fit_mask shape must match reference_depth_m")
        valid = valid & fit_mask
    if not np.any(valid):
        raise RuntimeError("Reference depth map has no valid pixels for scaling.")

    n_valid = int(np.count_nonzero(valid))
    if n_valid < min_fit_pixels:
        raise RuntimeError(
            f"Too few pixels for DA-V2 scaling ({n_valid} < {min_fit_pixels}). "
            "Try --scale-modes opencv or check stereo / pair_*.txt GT for opencv-gt."
        )

    indices = np.flatnonzero(valid)
    if len(indices) > max_samples:
        indices = np.random.choice(indices, max_samples, replace=False)

    mask = np.zeros_like(valid, dtype=bool)
    mask.ravel()[indices] = True

    variant, scale, shift, rel_used = _pick_relative_variant(da_relative, reference_depth_m, mask)
    metric = (scale * rel_used + shift).astype(np.float32)
    metric[metric < 0] = np.nan
    info = {
        "variant": variant,
        "scale": scale,
        "shift_m": shift,
        "reference_valid_pixels": int(np.count_nonzero(np.isfinite(reference_depth_m) & (reference_depth_m > 0.05))),
        "fit_pixels": int(np.count_nonzero(mask)),
        "fit_correlation": float(np.corrcoef(rel_used[mask], reference_depth_m[mask])[0, 1]),
    }
    if fit_mask is not None:
        info["fit_mask_pixels"] = n_valid
    return metric, info


def pointcloud_from_depth_map(
    depth_m: np.ndarray,
    K: np.ndarray,
    image_bgr: np.ndarray,
    depth_min_m: float,
    depth_max_m: float,
):
    """Back-project metric depth in the rectified RGB1 frame."""
    h, w = depth_m.shape
    u_grid, v_grid = np.meshgrid(np.arange(w, dtype=np.float64), np.arange(h, dtype=np.float64))
    valid = (
        np.isfinite(depth_m)
        & (depth_m >= depth_min_m)
        & (depth_m <= depth_max_m)
    )
    z = depth_m[valid]
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    x = (u_grid[valid] - cx) * z / fx
    y = (v_grid[valid] - cy) * z / fy
    points = np.column_stack([x, y, z])
    colors = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)[valid].astype(np.float64) / 255.0
    return points, colors


def save_depth_preview(path, depth_m: np.ndarray):
    valid = np.isfinite(depth_m) & (depth_m > 0)
    preview = np.zeros(depth_m.shape, dtype=np.uint8)
    if np.any(valid):
        values = depth_m[valid]
        lo, hi = np.percentile(values, [5, 95])
        if hi <= lo:
            lo, hi = float(np.min(values)), float(np.max(values))
        scaled = np.clip((values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        preview[valid] = (scaled * 255.0).astype(np.uint8)
    cv2.imwrite(str(path), cv2.applyColorMap(preview, cv2.COLORMAP_TURBO))


def write_scale_info(path, info: dict) -> None:
    Path(path).write_text(json.dumps(info, indent=2) + "\n")
