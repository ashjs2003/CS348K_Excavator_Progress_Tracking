"""ROI polygon mask, wall (farthest) vs ruler GT, per-pixel depth comparison."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from evaluation.depth_maps import METHODS, discover_methods, load_metric_depth, load_or_compute_stereo_geometry
from evaluation.gt_depth_overlay import (
    DEFAULT_GT_TOLERANCE_M,
    WALL_GT_CM,
    load_gt_reference_for_run,
    parse_pair_distance_txt,
)

ROI_SUFFIX = "_roi.json"
DEFAULT_WALL_PERCENTILE = 90.0  # top 10% depth in ROI = farthest (wall)
# No 100 cm wall GT on excavator captures (ROI vs .txt only).
EXCAVATOR_NO_WALL_SCENES = frozenset({"excavator_M", "excavator_S"})

METHOD_CHART_LABELS = {
    "opencv": "OpenCV",
    "dav2": "DA-V2",
    "dav2_gt": "DA-V2 GT",
    "foundation": "Foundation",
}

METHOD_ORDER = ("opencv", "dav2", "dav2_gt", "foundation")


def scene_uses_wall_gt(scene: str) -> bool:
    return scene not in EXCAVATOR_NO_WALL_SCENES


def pair_id_from_stem(stem: str) -> str:
    if stem.startswith("pair_"):
        return stem.replace("pair_", "", 1)
    return stem


def roi_json_path(data_scene_dir: Path, pair_id: str) -> Path:
    return Path(data_scene_dir) / f"pair_{pair_id}{ROI_SUFFIX}"


def discover_data_pairs(data_scene_dir: Path) -> list[str]:
    ids = []
    for p in sorted(data_scene_dir.glob("pair_*_rgb_L.png")):
        ids.append(pair_id_from_stem(p.stem.replace("_rgb_L", "")))
    return ids


def annotation_image_paths(
    repo_root: Path,
    scene: str,
    pair_id: str,
) -> tuple[Path | None, Path | None]:
    """Prefer rectified RGB from run; fallback to raw left capture in data/."""
    repo_root = Path(repo_root)
    rect = repo_root / "outputs" / "runs" / scene / f"pair_{pair_id}" / "depth" / "shared" / "rgb1_rectified.png"
    raw = repo_root / "data" / scene / f"pair_{pair_id}_rgb_L.png"
    rect_p = rect if rect.is_file() else None
    raw_p = raw if raw.is_file() else None
    return rect_p, raw_p


def save_roi_polygon(
    path: Path,
    *,
    scene: str,
    pair_id: str,
    image_path: Path,
    polygon_xy: list[list[int]],
    wall_percentile: float = DEFAULT_WALL_PERCENTILE,
) -> None:
    path = Path(path)
    payload = {
        "scene": scene,
        "pair_id": pair_id,
        "image_path": str(image_path),
        "polygon_xy": polygon_xy,
        "wall_depth_percentile": float(wall_percentile),
        "notes": "Polygon in image pixel coords. Wall = farthest depth band inside ROI (top percentile).",
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def load_roi_polygon(path: Path) -> dict:
    path = Path(path)
    data = json.loads(path.read_text())
    pts = np.asarray(data["polygon_xy"], dtype=np.int32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
        raise ValueError(f"Invalid polygon in {path}")
    return data


def polygon_mask(shape_hw: tuple[int, int], polygon_xy: np.ndarray) -> np.ndarray:
    h, w = shape_hw
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(mask, [polygon_xy.reshape(-1, 1, 2)], 1)
    return mask.astype(bool)


def classify_wall_ruler_in_roi(
    depth_m: np.ndarray,
    roi_mask: np.ndarray,
    *,
    target_gt_m: float | None,
    wall_gt_m: float,
    wall_percentile: float = DEFAULT_WALL_PERCENTILE,
    use_wall_gt: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Inside ROI: farthest-depth pixels -> wall (compare to wall_gt_m).
    Remaining valid pixels -> ruler (compare to target_gt_m from .txt).
    If use_wall_gt is False, all valid ROI pixels are ruler-only (no wall band).
    """
    valid = roi_mask & np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return np.zeros_like(roi_mask), np.zeros_like(roi_mask)

    if not use_wall_gt:
        ruler = valid if target_gt_m is not None else np.zeros_like(roi_mask)
        return np.zeros_like(roi_mask), ruler

    z = depth_m[valid]
    thresh = float(np.percentile(z, wall_percentile))
    wall = valid & (depth_m >= thresh)
    if target_gt_m is None:
        ruler = np.zeros_like(roi_mask)
    else:
        ruler = valid & ~wall
    return wall, ruler


def per_point_errors(
    depth_m: np.ndarray,
    wall_mask: np.ndarray,
    ruler_mask: np.ndarray,
    *,
    target_gt_m: float | None,
    wall_gt_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (gt_m, gt_label 0=wall 1=ruler, error_m) only at compared pixels."""
    compare = wall_mask | ruler_mask
    n = int(np.count_nonzero(compare))
    if n == 0:
        return np.array([]), np.array([]), np.array([])

    gt = np.full(depth_m.shape, np.nan, dtype=np.float64)
    label = np.full(depth_m.shape, -1, dtype=np.int8)
    gt[wall_mask] = wall_gt_m
    label[wall_mask] = 0
    if target_gt_m is not None:
        gt[ruler_mask] = target_gt_m
        label[ruler_mask] = 1

    err = np.abs(depth_m - gt)
    return gt[compare], label[compare], err[compare]


def summarize_errors(error_m: np.ndarray, tolerance_m: float) -> dict:
    if len(error_m) == 0:
        return {
            "n_pixels": 0,
            "median_error_cm": None,
            "mean_error_cm": None,
            "rmse_cm": None,
            "frac_within_tolerance_pct": None,
        }
    return {
        "n_pixels": int(len(error_m)),
        "median_error_cm": float(np.median(error_m) * 100.0),
        "mean_error_cm": float(np.mean(error_m) * 100.0),
        "rmse_cm": float(np.sqrt(np.mean(error_m**2)) * 100.0),
        "frac_within_tolerance_pct": float(100.0 * np.count_nonzero(error_m <= tolerance_m) / len(error_m)),
    }


def summarize_depth_in_mask(depth_m: np.ndarray, mask: np.ndarray) -> dict:
    z = depth_m[mask & np.isfinite(depth_m) & (depth_m > 0)]
    if len(z) == 0:
        return {"median_depth_cm": None, "mean_depth_cm": None}
    return {
        "median_depth_cm": float(np.median(z) * 100.0),
        "mean_depth_cm": float(np.mean(z) * 100.0),
    }


def discover_all_roi_jobs(data_root: Path, runs_root: Path) -> list[tuple[str, str, Path, Path]]:
    """(scene, pair_id, roi_path, run_dir) for each pair_*_roi.json with a matching run."""
    data_root = Path(data_root)
    runs_root = Path(runs_root)
    jobs: list[tuple[str, str, Path, Path]] = []
    for roi_path in sorted(data_root.rglob(f"pair_*{ROI_SUFFIX}")):
        scene = roi_path.parent.name
        pair_id = roi_path.stem.replace("pair_", "").replace("_roi", "")
        run_dir = runs_root / scene / f"pair_{pair_id}"
        if run_dir.is_dir() and (run_dir / "capture" / "rgb1.png").is_file():
            jobs.append((scene, pair_id, roi_path, run_dir))
    return jobs


def error_map_cm(
    depth_m: np.ndarray,
    wall_mask: np.ndarray,
    ruler_mask: np.ndarray,
    *,
    target_gt_m: float | None,
    wall_gt_m: float,
) -> np.ndarray:
    """Per-pixel |Z_est - GT| in cm; nan outside compared pixels."""
    h, w = depth_m.shape
    err_cm = np.full((h, w), np.nan, dtype=np.float64)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if np.any(wall_mask & valid):
        err_cm[wall_mask & valid] = np.abs(depth_m[wall_mask & valid] - wall_gt_m) * 100.0
    if target_gt_m is not None and np.any(ruler_mask & valid):
        err_cm[ruler_mask & valid] = np.abs(depth_m[ruler_mask & valid] - target_gt_m) * 100.0
    return err_cm


def evaluate_roi_for_run(
    run_dir: Path,
    roi_path: Path,
    repo_root: Path,
    *,
    tolerance_m: float = DEFAULT_GT_TOLERANCE_M,
    use_wall_gt: bool | None = None,
) -> dict:
    run_dir = Path(run_dir)
    repo_root = Path(repo_root)
    roi = load_roi_polygon(roi_path)
    depth_dir = run_dir / "depth"
    validation_dir = run_dir / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    gt_ref = load_gt_reference_for_run(run_dir, repo_root)
    scene_name = roi.get("scene") or run_dir.parent.name
    if use_wall_gt is None:
        use_wall_gt = scene_uses_wall_gt(scene_name)
    target_gt_m = gt_ref["target_gt_m"]
    wall_gt_m = gt_ref["wall_gt_m"]
    wall_pct = float(roi.get("wall_depth_percentile", DEFAULT_WALL_PERCENTILE))

    capture_rgb1 = run_dir / "capture" / "rgb1.png"
    capture_rgb2 = run_dir / "capture" / "rgb2.png"
    if not capture_rgb1.is_file():
        raise FileNotFoundError(f"Missing {capture_rgb1}")
    geometry = load_or_compute_stereo_geometry(depth_dir, capture_rgb1, capture_rgb2)
    h, w = geometry["image_size"][1], geometry["image_size"][0]

    poly = np.asarray(roi["polygon_xy"], dtype=np.int32)
    if poly[:, 0].max() >= w or poly[:, 1].max() >= h:
        raise ValueError(
            f"ROI polygon coords exceed rectified size {w}x{h}. "
            "Re-annotate on depth/shared/rgb1_rectified.png for this pair."
        )
    roi_mask = polygon_mask((h, w), poly)

    methods = discover_methods(depth_dir)
    if not methods:
        raise RuntimeError(f"No depth methods in {depth_dir}")

    ref_depth = None
    for name in methods:
        z = load_metric_depth(depth_dir, name, geometry)
        if z is not None:
            ref_depth = z
            break
    if ref_depth is None:
        raise RuntimeError("No loadable depth map")

    wall_mask, ruler_mask = classify_wall_ruler_in_roi(
        ref_depth,
        roi_mask,
        target_gt_m=target_gt_m,
        wall_gt_m=wall_gt_m,
        wall_percentile=wall_pct,
        use_wall_gt=use_wall_gt,
    )

    result = {
        "run": run_dir.name,
        "scene": scene_name,
        "pair_id": roi.get("pair_id"),
        "roi_path": str(roi_path),
        "use_wall_gt": use_wall_gt,
        "target_gt_cm": gt_ref.get("target_gt_cm"),
        "wall_gt_cm": gt_ref["wall_gt_cm"] if use_wall_gt else None,
        "tolerance_cm": tolerance_m * 100.0,
        "wall_percentile": wall_pct,
        "roi_pixels": int(np.count_nonzero(roi_mask)),
        "wall_pixels": int(np.count_nonzero(wall_mask)),
        "ruler_pixels": int(np.count_nonzero(ruler_mask)),
        "methods": {},
    }

    for method in methods:
        depth_m = load_metric_depth(depth_dir, method, geometry)
        if depth_m is None:
            continue
        gt_c, lab_c, err_c = per_point_errors(
            depth_m, wall_mask, ruler_mask, target_gt_m=target_gt_m, wall_gt_m=wall_gt_m
        )
        wall_err = err_c[lab_c == 0] if len(err_c) else np.array([])
        ruler_err = err_c[lab_c == 1] if len(err_c) else np.array([])

        wall_stats = summarize_errors(wall_err, tolerance_m)
        ruler_stats = summarize_errors(ruler_err, tolerance_m)
        all_stats = summarize_errors(err_c, tolerance_m)
        wall_stats.update(summarize_depth_in_mask(depth_m, wall_mask))
        ruler_stats.update(summarize_depth_in_mask(depth_m, ruler_mask))
        compare_mask = wall_mask | ruler_mask
        all_stats.update(summarize_depth_in_mask(depth_m, compare_mask))

        result["methods"][method] = {
            "wall": wall_stats,
            "ruler": ruler_stats,
            "all_roi_compared": all_stats,
        }

    return result


def write_per_point_csv(
    path: Path,
    depth_m: np.ndarray,
    wall_mask: np.ndarray,
    ruler_mask: np.ndarray,
    *,
    target_gt_m: float | None,
    wall_gt_m: float,
    method: str,
) -> None:
    import csv

    path = Path(path)
    rows = []
    h, w = depth_m.shape
    for v in range(h):
        for u in range(w):
            if wall_mask[v, u]:
                gt_m, role = wall_gt_m, "wall"
            elif ruler_mask[v, u]:
                if target_gt_m is None:
                    continue
                gt_m, role = target_gt_m, "ruler"
            else:
                continue
            z = float(depth_m[v, u])
            rows.append(
                {
                    "u": u,
                    "v": v,
                    "method": method,
                    "role": role,
                    "z_est_m": z,
                    "gt_m": gt_m,
                    "gt_cm": gt_m * 100.0,
                    "error_cm": abs(z - gt_m) * 100.0,
                }
            )
    with open(path, "w", newline="") as f:
        if not rows:
            f.write("u,v,method,role,z_est_m,gt_m,gt_cm,error_cm\n")
            return
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def render_roi_overlay(
    rgb_bgr: np.ndarray,
    polygon_xy: np.ndarray,
    wall_mask: np.ndarray,
    ruler_mask: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    vis = rgb_bgr.copy()
    cv2.polylines(vis, [polygon_xy.reshape(-1, 1, 2)], True, (255, 255, 0), 2)
    vis[ruler_mask] = (vis[ruler_mask] * 0.45 + np.array([0, 200, 0]) * 0.55).astype(np.uint8)
    vis[wall_mask] = (vis[wall_mask] * 0.45 + np.array([255, 80, 80]) * 0.55).astype(np.uint8)
    cv2.putText(vis, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(vis, "green=ruler GT  red=wall GT", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), vis)


def run_roi_gt_evaluation(
    run_dir: Path,
    roi_path: Path,
    repo_root: Path,
    validation_dir: Path | None = None,
    *,
    tolerance_m: float = DEFAULT_GT_TOLERANCE_M,
    heatmap_vmax_cm: float = 30.0,
) -> dict | None:
    """
    Full ROI GT pipeline: JSON summary, overlay, CSVs, numeric grid, heatmaps.
    Returns summary dict or None on failure.
    """
    from depth_layout import resolve_path
    from evaluation.roi_gt_grid import (
        render_heatmap_grid_combined,
        render_heatmap_grid_split_roles,
        render_numeric_grid,
    )

    run_dir = Path(run_dir)
    validation_dir = Path(validation_dir or run_dir / "validation")
    validation_dir.mkdir(parents=True, exist_ok=True)

    try:
        summary = evaluate_roi_for_run(
            run_dir, roi_path, repo_root, tolerance_m=tolerance_m
        )
    except Exception as exc:
        return {"error": str(exc)}

    (validation_dir / "roi_gt_depth_compare.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )

    roi = load_roi_polygon(roi_path)
    depth_dir = run_dir / "depth"
    capture_rgb1 = run_dir / "capture" / "rgb1.png"
    capture_rgb2 = run_dir / "capture" / "rgb2.png"
    geometry = load_or_compute_stereo_geometry(depth_dir, capture_rgb1, capture_rgb2)
    h, w = geometry["image_size"][1], geometry["image_size"][0]
    poly = np.asarray(roi["polygon_xy"], dtype=np.int32)
    roi_mask = polygon_mask((h, w), poly)

    gt_ref = load_gt_reference_for_run(run_dir, repo_root)
    use_wall = summary.get("use_wall_gt", True)
    target_gt_m = gt_ref["target_gt_m"]
    wall_gt_m = gt_ref["wall_gt_m"]

    ref = load_metric_depth(depth_dir, discover_methods(depth_dir)[0], geometry)
    wall_mask, ruler_mask = classify_wall_ruler_in_roi(
        ref,
        roi_mask,
        target_gt_m=target_gt_m,
        wall_gt_m=wall_gt_m,
        wall_percentile=float(roi.get("wall_depth_percentile", DEFAULT_WALL_PERCENTILE)),
        use_wall_gt=use_wall,
    )

    rect_path = resolve_path(depth_dir, None, "rgb1_rectified.png")
    rgb = cv2.imread(str(rect_path or capture_rgb1))
    if rgb is not None:
        render_roi_overlay(
            rgb,
            poly,
            wall_mask,
            ruler_mask,
            validation_dir / "roi_gt_overlay.png",
            f"{summary['scene']} pair_{summary['pair_id']}",
        )

    error_maps: dict[str, np.ndarray] = {}
    for method in summary.get("methods", {}):
        depth_m = load_metric_depth(depth_dir, method, geometry)
        if depth_m is None:
            continue
        error_maps[method] = error_map_cm(
            depth_m, wall_mask, ruler_mask, target_gt_m=target_gt_m, wall_gt_m=wall_gt_m
        )
        write_per_point_csv(
            validation_dir / f"roi_gt_per_point_{method}.csv",
            depth_m,
            wall_mask,
            ruler_mask,
            target_gt_m=target_gt_m,
            wall_gt_m=wall_gt_m,
            method=method,
        )

    render_numeric_grid(summary, validation_dir / "roi_gt_eval_grid.png")
    if rgb is not None and error_maps:
        render_heatmap_grid_combined(
            rgb, poly, error_maps, summary,
            validation_dir / "roi_gt_eval_heatmap.png",
            vmax_cm=heatmap_vmax_cm,
        )
        render_heatmap_grid_split_roles(
            rgb, poly, error_maps, wall_mask, ruler_mask, summary,
            validation_dir / "roi_gt_eval_heatmap_roles.png",
            vmax_cm=heatmap_vmax_cm,
        )

    return summary
