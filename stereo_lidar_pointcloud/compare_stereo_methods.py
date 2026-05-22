"""
Print side-by-side stats for OpenCV, FoundationStereo, and Depth Anything V2.

Run:
    python compare_stereo_methods.py
    python compare_stereo_methods.py --run 20260521_222300_legacy_import
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths
from stereo_shared import disparity_coverage


def load_metrics(path: Path):
    if not path.is_file():
        return None
    return json.loads(path.read_text())


def disparity_stats(stereo_dir: Path, prefix: str):
    disp_path = stereo_dir / f"disparity{prefix}.npy"
    ply_path = stereo_dir / f"stereo_pointcloud_downsampled{prefix}.ply"
    if not disp_path.is_file():
        return None
    disp = np.load(disp_path)
    cov, dmin, dmax = disparity_coverage(disp)
    return {
        "coverage_pct": cov,
        "range": f"{dmin:.1f} .. {dmax:.1f} px",
        "ply": ply_path.name if ply_path.is_file() else "(missing)",
    }


def dav2_stats(stereo_dir: Path):
    depth_path = stereo_dir / "depth_metric_dav2.npy"
    ply_path = stereo_dir / "stereo_pointcloud_downsampled_dav2.ply"
    scale_path = stereo_dir / "depth_scaling_dav2.json"
    if not depth_path.is_file():
        return None
    depth = np.load(depth_path)
    valid = np.isfinite(depth) & (depth > 0)
    cov = 100.0 * float(np.count_nonzero(valid)) / depth.size
    dmin = float(np.min(depth[valid])) if np.any(valid) else 0.0
    dmax = float(np.max(depth[valid])) if np.any(valid) else 0.0
    scale_note = ""
    if scale_path.is_file():
        info = json.loads(scale_path.read_text())
        scale_note = f"scale={info.get('scale', '?'):.4f} variant={info.get('variant', '?')}"
    return {
        "coverage_pct": cov,
        "range": f"{dmin:.2f} .. {dmax:.2f} m",
        "ply": ply_path.name if ply_path.is_file() else "(missing)",
        "scale_note": scale_note,
    }


def print_validation(paths: Path):
    suffixes = [
        ("OpenCV", ""),
        ("FoundationStereo", "_foundation"),
        ("Depth Anything V2", "_dav2"),
    ]
    any_metrics = False
    for label, suffix in suffixes:
        metrics_path = paths.validation / f"lidar_stereo_error_metrics{suffix}.json"
        metrics = load_metrics(metrics_path)
        if metrics:
            any_metrics = True
            print(
                f"  {label:20s} median={metrics.get('median_error', 0):.4f} m  "
                f"rmse={metrics.get('rmse', 0):.4f} m"
            )
    if not any_metrics:
        print("  (run 03_validate_with_lidar.py per method — see DEPTH_ANYTHING_V2.md)")


def main():
    parser = argparse.ArgumentParser(description="Compare depth / stereo methods for one run")
    add_run_cli_arguments(parser)
    args = parser.parse_args()
    if handle_list_runs(args):
        return

    paths = resolve_run_paths(args.run)
    stereo_dir = paths.stereo
    print(f"Run: {paths.run_dir.name if paths.run_dir else 'legacy'}")
    print(f"Stereo dir: {stereo_dir}\n")

    rows = [
        ("OpenCV", disparity_stats(stereo_dir, ""), "python 02_make_stereo_pointcloud.py"),
        ("FoundationStereo", disparity_stats(stereo_dir, "_foundation"), "python 02_make_stereo_pointcloud_foundation.py"),
        ("Depth Anything V2", dav2_stats(stereo_dir), "python 02_make_depth_anything_pointcloud.py"),
    ]
    for label, stats, hint in rows:
        print(f"=== {label} ===")
        if stats is None:
            print(f"  (not found — {hint})\n")
            continue
        print(f"  coverage: {stats['coverage_pct']:.1f}%")
        print(f"  range: {stats['range']}")
        print(f"  PLY: {stats['ply']}")
        if stats.get("scale_note"):
            print(f"  {stats['scale_note']}")
        print()

    print("=== LiDAR validation (median error) ===")
    print_validation(paths)


if __name__ == "__main__":
    main()
