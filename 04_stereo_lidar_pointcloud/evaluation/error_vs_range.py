"""Binned LiDAR ray depth error vs range (Z_lidar in rectified RGB1 frame)."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from evaluation.depth_maps import METHODS


def ray_pairs_from_arrays(
    z_lidar: np.ndarray,
    z_est: np.ndarray,
    compare: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (range_m, abs_error_m) for compared LiDAR hits."""
    if not np.any(compare):
        return np.array([]), np.array([])
    z = z_lidar[compare].astype(np.float64)
    e = np.abs(z_est[compare] - z_lidar[compare]).astype(np.float64)
    ok = np.isfinite(z) & np.isfinite(e) & (z > 0)
    return z[ok], e[ok]


def shared_bin_edges(
    range_arrays: list[np.ndarray],
    n_bins: int = 8,
    min_points: int = 3,
) -> np.ndarray | None:
    """Edges in meters spanning pooled LiDAR ranges; None if too few points."""
    pooled = np.concatenate([r for r in range_arrays if len(r)]) if range_arrays else np.array([])
    if len(pooled) < min_points:
        return None
    lo = float(np.percentile(pooled, 2))
    hi = float(np.percentile(pooled, 98))
    if hi - lo < 0.02:
        mid = 0.5 * (lo + hi)
        lo = max(0.05, mid - 0.15)
        hi = mid + 0.15
    return np.linspace(lo, hi, n_bins + 1)


def binned_error_vs_range(
    range_m: np.ndarray,
    error_m: np.ndarray,
    bin_edges_m: np.ndarray,
) -> dict:
    """Per-bin stats; last bin includes right edge."""
    bins_out = []
    edges = np.asarray(bin_edges_m, dtype=np.float64)
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if i < len(edges) - 2:
            mask = (range_m >= lo) & (range_m < hi)
        else:
            mask = (range_m >= lo) & (range_m <= hi)
        n = int(np.count_nonzero(mask))
        center = 0.5 * (lo + hi)
        if n == 0:
            bins_out.append({
                "range_min_m": lo,
                "range_max_m": hi,
                "range_center_m": center,
                "count": 0,
                "median_error_m": None,
                "mean_error_m": None,
                "rmse_m": None,
                "p90_error_m": None,
            })
            continue
        err = error_m[mask]
        bins_out.append({
            "range_min_m": lo,
            "range_max_m": hi,
            "range_center_m": center,
            "count": n,
            "median_error_m": float(np.median(err)),
            "mean_error_m": float(np.mean(err)),
            "rmse_m": float(np.sqrt(np.mean(err**2))),
            "p90_error_m": float(np.percentile(err, 90)),
        })
    return {
        "bin_edges_m": [float(x) for x in edges],
        "bins": bins_out,
        "n_points": int(len(range_m)),
    }


def compute_error_vs_range(
    z_lidar: np.ndarray,
    z_est: np.ndarray,
    compare: np.ndarray,
    bin_edges_m: np.ndarray | None = None,
    n_bins: int = 8,
) -> dict:
    range_m, error_m = ray_pairs_from_arrays(z_lidar, z_est, compare)
    if len(range_m) == 0:
        return {"bin_edges_m": [], "bins": [], "n_points": 0}
    if bin_edges_m is None:
        bin_edges_m = shared_bin_edges([range_m], n_bins=n_bins)
        if bin_edges_m is None:
            bin_edges_m = np.linspace(float(np.min(range_m)), float(np.max(range_m)), n_bins + 1)
    return binned_error_vs_range(range_m, error_m, bin_edges_m)


def load_ray_pairs_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load (z_lidar_m, ray_error_m) for rows with a valid comparison."""
    path = Path(path)
    if not path.is_file():
        return np.array([]), np.array([])
    ranges, errors = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            z_s, e_s = row.get("z_lidar_m", ""), row.get("ray_error_m", "")
            if z_s == "" or e_s == "":
                continue
            try:
                z, e = float(z_s), float(e_s)
                if np.isfinite(z) and np.isfinite(e) and z > 0:
                    ranges.append(z)
                    errors.append(e)
            except ValueError:
                pass
    return np.asarray(ranges, dtype=float), np.asarray(errors, dtype=float)


def method_ray_csv_suffix(method_id: str) -> str:
    return METHODS.get(method_id, {}).get("suffix", f"_{method_id}")
