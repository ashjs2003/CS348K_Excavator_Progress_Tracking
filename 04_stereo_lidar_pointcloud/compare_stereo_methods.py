"""
Print scorecard for OpenCV, DA-V2, and FoundationStereo.

Run after:
    python 06_evaluate_run.py --run latest
"""

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

from depth_layout import resolve_path
from evaluation.depth_maps import METHODS, discover_methods
from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths
from stereo_shared import disparity_coverage


def load_json(path: Path):
    return json.loads(path.read_text()) if path.is_file() else None


def main():
    parser = argparse.ArgumentParser(description="Compare depth methods for one run")
    add_run_cli_arguments(parser)
    args = parser.parse_args()
    if handle_list_runs(args):
        return

    paths = resolve_run_paths(args.run)
    stereo_dir = paths.depth
    val = paths.validation
    print(f"Run: {paths.run_dir.name if paths.run_dir else 'legacy'}\n")

    summary = load_json(val / "evaluation_summary.json")
    photometric = load_json(val / "photometric_reprojection.json") or {}
    if summary:
        print("=== Scorecard (evaluation_summary.json) ===")
        for method, row in summary.get("methods", {}).items():
            med = row.get("ray_median_error_m")
            fs = row.get("free_space_violation_pct")
            cov = row.get("depth_coverage_pct")
            cov_s = f"{cov:5.1f}%" if cov is not None else "  n/a"
            med_s = f"{med:.4f} m" if med is not None else "n/a"
            fs_s = f"{fs:.1f}%" if fs is not None else "n/a"
            print(
                f"  {method:12s}  cov={cov_s}  ray_med={med_s}  "
                f"inlier={row.get('inlier_ratio', 0):.2%}  free_space={fs_s}"
            )
        print()
    else:
        print("Run: python 06_evaluate_run.py --run latest\n")

    if photometric:
        print("=== Photometric (lower is better) ===")
        for key, row in photometric.items():
            print(f"  {key:12s}  mean={row.get('mean_photometric_error')}  samples={row.get('sample_count')}")
        print()

    gt = summary.get("gt_depth_reference") if summary else None
    if gt and gt.get("saved"):
        print(
            f"=== Manual GT depth (±{gt.get('tolerance_m', 0.05) * 100:.0f} cm) ===\n"
            f"  target {gt.get('target_gt_cm')} cm from txt  |  wall {gt.get('wall_gt_cm')} cm\n"
            f"  any method: target {gt.get('frac_target_any_method_pct', 0):.2f}%  "
            f"wall {gt.get('frac_wall_any_method_pct', 0):.2f}% of image\n"
            f"  overlay: {gt.get('combined_on_rgb')}\n"
        )
        for name, row in gt.get("per_method", {}).items():
            print(
                f"  {name:12s}  target {row.get('frac_target_pct', 0):5.2f}%  "
                f"wall {row.get('frac_wall_pct', 0):5.2f}%"
            )
        print()

    cross = load_json(val / "cross_method_metrics.json")
    if cross and cross.get("consensus"):
        c = cross["consensus"]
        print(f"=== Cross-method depth std (median {c.get('median_std_m')} m) ===\n")

    for method in discover_methods(stereo_dir):
        suffix = METHODS[method]["suffix"]
        ray = load_json(val / f"lidar_ray_depth_metrics{suffix}.json")
        nn = load_json(val / f"lidar_stereo_error_metrics{suffix}.json")
        meta = METHODS[method]
        depth_path = resolve_path(stereo_dir, method, "depth_metric.npy")
        disp_path = resolve_path(stereo_dir, method, "disparity.npy")
        if depth_path is not None:
            d = np.load(depth_path)
            v = np.isfinite(d) & (d > 0)
            extra = f"depth cov={100 * np.count_nonzero(v) / d.size:.1f}%"
        elif disp_path is not None:
            cov, _, _ = disparity_coverage(np.load(disp_path))
            extra = f"disp cov={cov:.1f}%"
        elif method == "foundation":
            from evaluation.depth_maps import foundation_preview_available, method_has_metric_depth

            if method_has_metric_depth(stereo_dir, "foundation"):
                fs = (summary or {}).get("methods", {}).get("foundation", {}).get("foundation_disparity", {})
                if disp_path is not None:
                    cov, _, _ = disparity_coverage(np.load(disp_path))
                    extra = f"disp cov={cov:.1f}%"
                elif fs:
                    extra = "depth from vis.png (scaled via OpenCV)"
                else:
                    extra = "vis.png → metric depth"
            elif foundation_preview_available(stereo_dir):
                extra = "vis.png (need OpenCV disparity for scale)"
            else:
                extra = "(missing)"
        else:
            extra = "(missing)"

        print(f"=== {method} ===  {extra}")
        if ray:
            print(
                f"  ray median={ray.get('ray_median_error_m')} m  "
                f"free-space viol={ray.get('free_space_violation_pct')}%  "
                f"assoc={ray.get('association_rate', 0):.2%}"
            )
        elif nn:
            print(f"  NN median (legacy only)={nn.get('median_error')} m")
        else:
            print("  (no metrics — run 06_evaluate_run.py)")
        print()


if __name__ == "__main__":
    main()
