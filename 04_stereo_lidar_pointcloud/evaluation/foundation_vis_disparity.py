"""
FoundationStereo stores results as vis.png: rectified RGB (left) + turbo disparity (right).

When disparity.npy is missing, decode the colormap panel and scale to pixel disparity
using OpenCV stereo as reference (same geometry / rectified frame).
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from depth_layout import method_dir, resolve_path
from dav2_scale import depth_map_from_disparity, metric_depth_from_relative

# Matches stereo_shared.save_disparity_preview (project OpenCV previews).
COLORMAP = cv2.COLORMAP_TURBO
BLACK_THRESH = 12


def _turbo_palette_bgr() -> np.ndarray:
    lut = np.arange(256, dtype=np.uint8).reshape(256, 1)
    return cv2.applyColorMap(lut, COLORMAP).reshape(256, 3)


def split_vis_composite(vis_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (left_rgb, right_disparity_color) from a side-by-side vis.png."""
    h, w = vis_bgr.shape[:2]
    mid = w // 2
    return vis_bgr[:, :mid].copy(), vis_bgr[:, mid:].copy()


def decode_colormap_to_index(colormap_bgr: np.ndarray, palette_bgr: np.ndarray | None = None) -> np.ndarray:
    """
    Nearest-neighbor decode of a turbo colormap image to uint8 indices 0..255.

    Invalid / black pixels are set to 0.
    """
    palette = palette_bgr if palette_bgr is not None else _turbo_palette_bgr()
    h, w = colormap_bgr.shape[:2]
    flat = colormap_bgr.reshape(-1, 3).astype(np.float32)
    gray = cv2.cvtColor(colormap_bgr, cv2.COLOR_BGR2GRAY)
    valid = (gray > BLACK_THRESH).ravel()

    idx = np.zeros(flat.shape[0], dtype=np.uint8)
    if np.any(valid):
        pts = flat[valid]
        # (N, 1, 3) - (1, 256, 3) -> (N, 256)
        diff = pts[:, None, :] - palette[None, :, :]
        dist2 = np.sum(diff * diff, axis=2)
        idx[valid] = np.argmin(dist2, axis=1).astype(np.uint8)

    return idx.reshape(h, w)


def disparity_index_from_vis(
    vis_path: Path,
    target_size: tuple[int, int],
) -> tuple[np.ndarray, dict]:
    """
    Extract a disparity index map (0 = invalid) from vis.png.

    target_size: (width, height) of rectified rgb1 / stereo geometry.
    """
    vis_path = Path(vis_path)
    vis = cv2.imread(str(vis_path), cv2.IMREAD_COLOR)
    if vis is None:
        raise FileNotFoundError(f"Could not read {vis_path}")

    _left, disp_color = split_vis_composite(vis)
    tw, th = target_size
    if disp_color.shape[1] != tw or disp_color.shape[0] != th:
        disp_color = cv2.resize(disp_color, (tw, th), interpolation=cv2.INTER_AREA)

    index = decode_colormap_to_index(disp_color)
    valid = index > 0
    meta = {
        "vis_path": str(vis_path),
        "vis_shape": list(vis.shape),
        "colormap": "COLORMAP_TURBO",
        "valid_pixel_pct": 100.0 * float(np.count_nonzero(valid)) / index.size,
        "index_range": [int(index[valid].min()), int(index[valid].max())] if np.any(valid) else None,
    }
    return index, meta


def _index_variants(index: np.ndarray) -> dict[str, np.ndarray]:
    inv = np.zeros_like(index, dtype=np.uint8)
    valid = index > 0
    inv[valid] = (255 - index[valid]).astype(np.uint8)
    return {"direct": index, "inverted": inv}


def disparity_from_vis_png(
    vis_path: Path,
    target_size: tuple[int, int],
    Q: np.ndarray,
    opencv_depth_m: np.ndarray,
    *,
    baseline_m: float,
    cache_path: Path | None = None,
) -> tuple[np.ndarray, dict]:
    """
    Build float32 pixel disparity (0 = invalid) from vis.png.

    Colormap indices are only ordinal; scale to metric depth using OpenCV depth
    (same approach as DA-V2), then convert back to disparity for the rest of the pipeline.
    """
    index, meta = disparity_index_from_vis(vis_path, target_size)
    if opencv_depth_m.shape != index.shape:
        raise ValueError("OpenCV depth shape must match rectified image size.")

    best_name = "direct"
    best_score = -np.inf
    best_depth = None
    best_stats: dict = {}
    for name, idx_map in _index_variants(index).items():
        proxy = idx_map.astype(np.float32)
        proxy[proxy <= 0] = np.nan
        depth_rel = depth_map_from_disparity(proxy, Q)
        try:
            depth_m, stats = metric_depth_from_relative(depth_rel, opencv_depth_m)
        except RuntimeError:
            continue
        valid = np.isfinite(depth_rel) & (depth_rel > 0) & np.isfinite(opencv_depth_m) & (opencv_depth_m > 0)
        if np.count_nonzero(valid) < 50:
            continue
        corr = abs(float(np.corrcoef(depth_rel[valid].ravel(), opencv_depth_m[valid].ravel())[0, 1]))
        if corr > best_score:
            best_score = corr
            best_name = name
            best_depth = depth_m
            best_stats = stats

    if best_depth is None:
        raise RuntimeError("Could not scale Foundation vis.png to OpenCV depth (too few overlap pixels).")

    fx = float(Q[2, 3])
    if fx <= 0 or baseline_m <= 0:
        raise RuntimeError("Invalid stereo geometry for disparity recovery.")

    disp = np.zeros(index.shape, dtype=np.float32)
    valid_z = np.isfinite(best_depth) & (best_depth > 0)
    disp[valid_z] = (fx * baseline_m / best_depth[valid_z]).astype(np.float32)

    meta = {
        **meta,
        **best_stats,
        "index_variant": best_name,
        "opencv_depth_correlation": best_score,
        "disparity_source": "vis.png_colormap_scaled_to_opencv_depth",
    }

    if cache_path is not None:
        cache_path = Path(cache_path)
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(cache_path, disp)
        meta["cached_disparity"] = str(cache_path)

    return disp, meta


def load_foundation_disparity(stereo_dir: Path, geometry: dict) -> tuple[np.ndarray | None, dict]:
    """
    Load foundation disparity.npy, or derive from vis.png + OpenCV scale fit.
    """
    stereo_dir = Path(stereo_dir)
    w, h = geometry["image_size"]
    target = (w, h)

    disp_path = resolve_path(stereo_dir, "foundation", "disparity.npy")
    if disp_path is not None:
        disp = np.load(disp_path).astype(np.float32)
        if disp.shape != (h, w):
            disp = cv2.resize(disp, target, interpolation=cv2.INTER_LINEAR)
        return disp, {"disparity_source": str(disp_path)}

    vis_path = resolve_path(stereo_dir, "foundation", "vis.png")
    if vis_path is None:
        return None, {"reason": "no foundation/vis.png or disparity.npy"}

    cache_path = method_dir(stereo_dir, "foundation") / "work" / "disparity_from_vis.npy"
    if cache_path.is_file():
        disp = np.load(cache_path).astype(np.float32)
        if disp.shape == (h, w):
            return disp, {
                "disparity_source": str(cache_path),
                "from_vis": True,
                "vis_path": str(vis_path),
            }

    opencv_depth_path = resolve_path(stereo_dir, "opencv", "depth_metric.npy")
    if opencv_depth_path is None:
        opencv_disp_path = resolve_path(stereo_dir, "opencv", "disparity.npy")
        if opencv_disp_path is None:
            return None, {"reason": "vis.png present but OpenCV depth/disparity required for scale"}
        opencv_depth_m = depth_map_from_disparity(np.load(opencv_disp_path), geometry["Q"])
    else:
        opencv_depth_m = np.load(opencv_depth_path).astype(np.float64)

    disp, meta = disparity_from_vis_png(
        vis_path,
        target,
        geometry["Q"],
        opencv_depth_m,
        baseline_m=float(geometry["baseline_m"]),
        cache_path=cache_path,
    )
    meta["from_vis"] = True
    meta["vis_path"] = str(vis_path)
    return disp, meta


def foundation_metric_depth_from_vis(stereo_dir: Path, geometry: dict) -> tuple[np.ndarray | None, dict]:
    disp_path = resolve_path(Path(stereo_dir), "foundation", "disparity.npy")
    if disp_path is not None:
        depth = depth_map_from_disparity(np.load(disp_path), geometry["Q"])
        return depth, {"disparity_source": str(disp_path)}

    disp, meta = load_foundation_disparity(stereo_dir, geometry)
    if disp is None:
        return None, meta
    depth = depth_map_from_disparity(disp, geometry["Q"])
    meta["depth_source"] = meta.get("disparity_source", "foundation")
    return depth, meta
