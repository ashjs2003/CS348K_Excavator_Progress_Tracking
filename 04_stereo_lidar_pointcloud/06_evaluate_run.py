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
from evaluation.gt_depth_overlay import save_gt_depth_reference_overlays
from evaluation.cross_method import compute_cross_method_metrics
from evaluation.depth_maps import METHODS, discover_methods, load_metric_depth, load_or_compute_stereo_geometry
from evaluation.foundation_vis_disparity import foundation_metric_depth_from_vis
from evaluation.error_vs_range import (
    binned_error_vs_range,
    ray_pairs_from_arrays,
    shared_bin_edges,
)
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
    parser.add_argument(
        "--skip-gt-overlay",
        action="store_true",
        help="Skip manual GT overlays (target from data/pair_*.txt, wall 100 cm, ±5 cm)",
    )
    parser.add_argument("--skip-nn", action="store_true", help="Skip legacy nearest-neighbor cloud metrics")
    parser.add_argument(
        "--range-bins",
        type=int,
        default=8,
        help="Bins for error-vs-range curves (shared edges across methods)",
    )
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

    compare = proj_valid & sampled_valid
    range_m, error_m = ray_pairs_from_arrays(z_lidar, z_est, compare)
    ray["_ray_pairs"] = {"range_m": range_m, "error_m": error_m}

    if not args.skip_nn:
        cloud_path = paths.depth / f"stereo_pointcloud_downsampled{suffix}.ply"
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

    geometry = load_or_compute_stereo_geometry(paths.depth, paths.rgb1_image, paths.rgb2_image)
    image_size = geometry["image_size"]
    points_rect, lidar_path = lidar_points_in_rectified_frame(paths.lidar_csv, image_size, geometry)
    print(f"LiDAR extrinsics: {lidar_path}")
    print(f"LiDAR points in rectified frame: {len(points_rect)}")

    colors = np.tile(np.array([[1.0, 0.05, 0.05]]), (len(points_rect), 1))
    write_ply(paths.validation / "lidar_points_in_rgb1_frame.ply", points_rect, colors)

    methods = discover_methods(paths.depth)
    if not methods:
        raise RuntimeError(
            f"No depth products in {paths.depth}. Run 02_make_stereo_pointcloud.py (and optional DA-V2 / Foundation) first."
        )
    print(f"Methods found: {', '.join(methods)}")

    summary = {
        "run": paths.run_dir.name if paths.run_dir else "legacy",
        "methods": {},
        "photometric": {},
    }
    for method in methods:
        fs_meta = None
        if method == "foundation":
            depth_m, fs_meta = foundation_metric_depth_from_vis(paths.depth, geometry)
        else:
            depth_m = load_metric_depth(paths.depth, method, geometry)
        if depth_m is None:
            continue
        print(f"\n=== {method} ===")
        if fs_meta and (
            fs_meta.get("from_vis")
            or "vis.png" in str(fs_meta.get("disparity_source", ""))
            or fs_meta.get("disparity_source", "").endswith("disparity_from_vis.npy")
        ):
            src = fs_meta.get("disparity_source", "vis.png")
            scale = fs_meta.get("scale")
            variant = fs_meta.get("index_variant", "")
            print(
                f"  disparity from vis.png (right panel, {variant})"
                + (f", depth scale={scale:.4f}" if scale is not None else "")
            )
        row = evaluate_method(method, depth_m, geometry, points_rect, paths, args)
        if fs_meta:
            row["foundation_disparity"] = fs_meta
        summary["methods"][method] = row
        med = row.get("ray_median_error_m")
        fs = row.get("free_space_violation_pct")
        print(f"  coverage: {row['depth_coverage_pct']:.1f}%")
        if med is not None:
            print(f"  ray median: {med:.4f} m  inlier: {row['inlier_ratio']:.2%}  free-space viol: {fs:.1f}%")

    pairs_by_method = {}
    range_arrays = []
    for method in summary["methods"]:
        row = summary["methods"][method]
        pairs = row.pop("_ray_pairs", {"range_m": np.array([]), "error_m": np.array([])})
        pairs_by_method[method] = pairs
        range_arrays.append(pairs["range_m"])

    shared_edges = shared_bin_edges(range_arrays, n_bins=args.range_bins)
    if shared_edges is not None:
        summary["error_vs_range_bin_edges_m"] = [float(x) for x in shared_edges]
        for method in summary["methods"]:
            if method not in pairs_by_method:
                continue
            r_m = pairs_by_method[method]["range_m"]
            e_m = pairs_by_method[method]["error_m"]
            if len(r_m):
                evr = binned_error_vs_range(r_m, e_m, shared_edges)
            else:
                evr = {"bin_edges_m": summary["error_vs_range_bin_edges_m"], "bins": [], "n_points": 0}
            summary["methods"][method]["error_vs_range"] = evr
            suffix = METHODS[method]["suffix"]
            ray_json = paths.validation / f"lidar_ray_depth_metrics{suffix}.json"
            if ray_json.is_file():
                payload = json.loads(ray_json.read_text())
                payload["error_vs_range"] = evr
                ray_json.write_text(json.dumps(payload, indent=2) + "\n")
        print(f"\nError vs range: shared {len(shared_edges) - 1} bins "
              f"({summary['error_vs_range_bin_edges_m'][0]:.2f}–"
              f"{summary['error_vs_range_bin_edges_m'][-1]:.2f} m)")
    if not args.skip_photometric:
        photometric = {}
        for key in ("opencv", "foundation"):
            result = run_photometric_for_method(paths.depth, key, geometry=geometry)
            if result is not None:
                photometric[key] = result
                print(f"\nPhotometric {key}: mean error = {result.get('mean_photometric_error')}")
        if photometric:
            phot_path = paths.validation / "photometric_reprojection.json"
            phot_path.write_text(json.dumps(photometric, indent=2) + "\n")
            summary["photometric"] = photometric

    depth_methods = list(summary["methods"].keys())
    if len(depth_methods) >= 2:
        cross = compute_cross_method_metrics(paths.depth, geometry, eps=args.cross_eps)
        cross_path = paths.validation / "cross_method_metrics.json"
        cross_path.write_text(json.dumps(cross, indent=2) + "\n")
        summary["cross_method"] = cross
        print(f"\nCross-method consensus median std: {cross.get('consensus', {}).get('median_std_m')}")

    if not args.skip_consensus and len(depth_methods) >= 2:
        consensus_info = save_consensus_depth_std_png(
            paths.depth,
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

    if not args.skip_gt_overlay:
        gt_info = save_gt_depth_reference_overlays(
            paths.depth,
            geometry,
            paths.validation,
            run_dir=paths.run_dir,
            repo_root=_REPO_ROOT,
            scene_label=paths.run_dir.name if paths.run_dir else None,
        )
        summary["gt_depth_reference"] = gt_info
        if gt_info.get("saved"):
            tol_cm = gt_info.get("tolerance_cm", 5.0)
            print(f"\nGT depth overlays (±{tol_cm:.0f} cm, from project defaults):")
            print(f"  target = {gt_info['target_gt_cm']:.1f} cm (pair_*.txt)  wall = {gt_info['wall_gt_cm']:.0f} cm (fixed)")
            if gt_info.get("txt_path"):
                print(f"  txt: {gt_info['txt_path']}")
            print(f"  Saved {gt_info['combined_on_rgb']}")
            print(f"  Saved {gt_info['all_methods_on_rgb']}")
            print(f"  Saved {gt_info['labeled_figure']}")
            for name, p in gt_info.get("per_method_on_rgb", {}).items():
                row = gt_info["per_method"].get(name, {})
                print(
                    f"  {name}: target {row.get('frac_target_pct', 0):.2f}%  "
                    f"wall {row.get('frac_wall_pct', 0):.2f}%  -> {p}"
                )
        else:
            print(f"\nGT depth overlays skipped: {gt_info.get('reason')}")

    summary_path = paths.validation / "evaluation_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    if paths.run_dir:
        write_run_info(paths.run_dir, evaluation_summary=summary)

    print(f"\nSaved {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()
