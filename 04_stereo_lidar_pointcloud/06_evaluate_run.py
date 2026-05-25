"""
Structured evaluation for one capture run: LiDAR ray + free-space, photometric, cross-method, consensus.

Run from this folder after depth pipelines:
    python 06_evaluate_run.py --run latest
    python 06_evaluate_run.py --run 20260521_235229_carpet
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths, write_run_info
from evaluation.consensus_map import save_consensus_depth_std_png
from evaluation.cross_method import compute_cross_method_metrics
from evaluation.depth_maps import METHODS, discover_methods, load_metric_depth, load_or_compute_stereo_geometry
from evaluation.lidar_ray_metrics import (
    compute_lidar_ray_metrics,
    compute_nn_cloud_metrics,
    lidar_points_in_rectified_frame,
    write_ray_per_point_csv,
)
from evaluation.photometric import run_photometric_for_method
from calib_utils import project_rectified_points
from pointcloud_utils import write_ply


def parse_args():
    parser = argparse.ArgumentParser(description="Structured evaluation for one run")
    add_run_cli_arguments(parser)
    parser.add_argument("--ray-inlier-tau", type=float, default=0.05, help="LiDAR ray inlier threshold (m)")
    parser.add_argument("--free-space-tau", type=float, default=0.03, help="Z_est < Z_lidar - tau counts as violation")
    parser.add_argument("--cross-eps", type=float, default=0.02, help="Depth agreement epsilon for cross-method stats")
    parser.add_argument("--skip-photometric", action="store_true")
    parser.add_argument("--skip-consensus", action="store_true")
    parser.add_argument("--skip-nn", action="store_true", help="Skip legacy nearest-neighbor cloud metrics")
    return parser.parse_args()


def depth_coverage_pct(depth_m: np.ndarray) -> float:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    return 100.0 * float(np.count_nonzero(valid)) / depth_m.size


def evaluate_method(
    method: str,
    depth_m: np.ndarray,
    geometry: dict,
    points_rect: np.ndarray,
    paths,
    args,
) -> dict:
    suffix = METHODS[method]["suffix"]
    P1 = geometry["P1"]

    ray = compute_lidar_ray_metrics(
        depth_m,
        points_rect,
        P1,
        ray_inlier_tau=args.ray_inlier_tau,
        free_space_tau=args.free_space_tau,
    )
    ray["method"] = method
    ray["depth_coverage_pct"] = depth_coverage_pct(depth_m)

    uv, z_lidar, proj_valid = project_rectified_points(P1, points_rect)
    z_est = np.full(len(points_rect), np.nan)
    h, w = depth_m.shape
    for i in np.flatnonzero(proj_valid):
        col, row = int(round(uv[i, 0])), int(round(uv[i, 1]))
        if 0 <= col < w and 0 <= row < h:
            z = depth_m[row, col]
            if np.isfinite(z) and z > 0:
                z_est[i] = z
    sampled_valid = np.isfinite(z_est)
    ray_csv = paths.validation / f"lidar_ray_per_point{suffix}.csv"
    write_ray_per_point_csv(ray_csv, points_rect, uv, z_lidar, z_est, proj_valid, sampled_valid)

    ray_path = paths.validation / f"lidar_ray_depth_metrics{suffix}.json"
    ray_path.write_text(json.dumps(ray, indent=2) + "\n")

    if not args.skip_nn:
        cloud_path = paths.stereo / f"stereo_pointcloud_downsampled{suffix}.ply"
        nn = compute_nn_cloud_metrics(points_rect, cloud_path)
        if nn is not None:
            nn["valid_lidar_points"] = int(len(points_rect))
            legacy_path = paths.validation / f"lidar_stereo_error_metrics{suffix}.json"
            legacy_path.write_text(json.dumps(nn, indent=2) + "\n")
            ray["nn_cloud"] = nn

    return ray


def main():
    args = parse_args()
    if handle_list_runs(args):
        return

    paths = resolve_run_paths(args.run)
    if paths.run_dir:
        print(f"Run: {paths.run_dir.name}")
    paths.validation.mkdir(parents=True, exist_ok=True)

    geometry = load_or_compute_stereo_geometry(paths.stereo, paths.rgb1_image, paths.rgb2_image)
    image_size = geometry["image_size"]
    points_rect, lidar_path = lidar_points_in_rectified_frame(paths.lidar_csv, image_size, geometry)
    print(f"LiDAR extrinsics: {lidar_path}")
    print(f"LiDAR points in rectified frame: {len(points_rect)}")

    colors = np.tile(np.array([[1.0, 0.05, 0.05]]), (len(points_rect), 1))
    write_ply(paths.validation / "lidar_points_in_rgb1_frame.ply", points_rect, colors)

    methods = discover_methods(paths.stereo)
    if not methods:
        raise RuntimeError(
            f"No depth products in {paths.stereo}. Run 02_make_stereo_pointcloud.py (and optional DA-V2 / Foundation) first."
        )
    print(f"Methods found: {', '.join(methods)}")

    summary = {"run": paths.run_dir.name if paths.run_dir else "legacy", "methods": {}, "photometric": {}}
    for method in methods:
        depth_m = load_metric_depth(paths.stereo, method, geometry)
        if depth_m is None:
            continue
        print(f"\n=== {method} ===")
        row = evaluate_method(method, depth_m, geometry, points_rect, paths, args)
        summary["methods"][method] = row
        med = row.get("ray_median_error_m")
        fs = row.get("free_space_violation_pct")
        print(f"  coverage: {row['depth_coverage_pct']:.1f}%")
        if med is not None:
            print(f"  ray median: {med:.4f} m  inlier: {row['inlier_ratio']:.2%}  free-space viol: {fs:.1f}%")

    if not args.skip_photometric:
        photometric = {}
        for key in ("opencv", "foundation"):
            result = run_photometric_for_method(paths.stereo, key)
            if result is not None:
                photometric[key] = result
                print(f"\nPhotometric {key}: mean error = {result.get('mean_photometric_error')}")
        if photometric:
            phot_path = paths.validation / "photometric_reprojection.json"
            phot_path.write_text(json.dumps(photometric, indent=2) + "\n")
            summary["photometric"] = photometric

    if len(methods) >= 2:
        cross = compute_cross_method_metrics(paths.stereo, geometry, eps=args.cross_eps)
        cross_path = paths.validation / "cross_method_metrics.json"
        cross_path.write_text(json.dumps(cross, indent=2) + "\n")
        summary["cross_method"] = cross
        print(f"\nCross-method consensus median std: {cross.get('consensus', {}).get('median_std_m')}")

    if not args.skip_consensus and len(methods) >= 2:
        consensus_info = save_consensus_depth_std_png(
            paths.stereo,
            geometry,
            paths.validation / "consensus_depth_std.png",
            scene_label=paths.run_dir.name if paths.run_dir else None,
        )
        summary["consensus_map"] = consensus_info
        if consensus_info.get("saved"):
            print(f"Saved {consensus_info['path']}")
            if consensus_info.get("overlay_saved"):
                print(f"Saved {consensus_info['overlay_path']}")
            elif consensus_info.get("overlay_reason"):
                print(f"  (no RGB overlay: {consensus_info['overlay_reason']})")

    summary_path = paths.validation / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if paths.run_dir:
        write_run_info(paths.run_dir, evaluation_summary=summary)

    print(f"\nSaved {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
