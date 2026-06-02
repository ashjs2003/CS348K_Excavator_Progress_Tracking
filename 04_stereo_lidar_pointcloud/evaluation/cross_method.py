"""Cross-method depth/disparity agreement (no ground truth)."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from depth_layout import resolve_path
from evaluation.depth_maps import METHODS, load_metric_depth


def _overlap_stats(a: np.ndarray, b: np.ndarray, eps: float = 0.02) -> dict:
    valid = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    n = int(np.count_nonzero(valid))
    if n == 0:
        return {"overlap_pixels": 0, "median_abs_diff_m": None, "mean_abs_diff_m": None, "correlation": None}
    diff = np.abs(a[valid] - b[valid])
    corr = float(np.corrcoef(a[valid], b[valid])[0, 1]) if n > 10 else None
    return {
        "overlap_pixels": n,
        "median_abs_diff_m": float(np.median(diff)),
        "mean_abs_diff_m": float(np.mean(diff)),
        "fraction_within_eps": float(np.count_nonzero(diff < eps) / n),
        "eps_m": eps,
        "correlation": corr,
    }


def compute_cross_method_metrics(stereo_dir, geometry: dict, eps: float = 0.02) -> dict:
    stereo_dir = str(stereo_dir)
    depths = {}
    for name in METHODS:
        z = load_metric_depth(stereo_dir, name, geometry)
        if z is not None:
            depths[name] = z

    out = {"methods_present": list(depths.keys()), "pairwise_depth": {}, "consensus": {}}
    names = list(depths.keys())
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            key = f"{a}_vs_{b}"
            out["pairwise_depth"][key] = _overlap_stats(depths[a], depths[b], eps=eps)

    if len(names) >= 2:
        stack = np.stack([depths[n] for n in names], axis=0)
        valid_count = np.sum(np.isfinite(stack) & (stack > 0), axis=0)
        all_valid = valid_count == len(names)
        if np.any(all_valid):
            std_map = np.nanstd(stack, axis=0)
            std_vals = std_map[all_valid]
            out["consensus"] = {
                "pixels_all_methods": int(np.count_nonzero(all_valid)),
                "median_std_m": float(np.median(std_vals)),
                "mean_std_m": float(np.mean(std_vals)),
                "p90_std_m": float(np.percentile(std_vals, 90)),
            }
        else:
            multi = valid_count >= 2
            if np.any(multi):
                std_map = np.nanstd(stack, axis=0)
                std_vals = std_map[multi]
                out["consensus"] = {
                    "pixels_at_least_two_methods": int(np.count_nonzero(multi)),
                    "median_std_m": float(np.median(std_vals)),
                    "mean_std_m": float(np.mean(std_vals)),
                    "p90_std_m": float(np.percentile(std_vals, 90)),
                }

    disp_pairs = {}
    sd = Path(stereo_dir)
    if resolve_path(sd, "opencv", "disparity.npy") and resolve_path(sd, "foundation", "disparity.npy"):
        d1 = load_metric_depth(stereo_dir, "opencv", geometry)
        d3 = load_metric_depth(stereo_dir, "foundation", geometry)
        if d1 is not None and d3 is not None:
            disp_pairs["opencv_vs_foundation_depth"] = _overlap_stats(d1, d3, eps=eps)
    out["pairwise_depth"].update(disp_pairs)
    return out
