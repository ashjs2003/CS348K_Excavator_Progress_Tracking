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

from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths
from stereo_shared import disparity_coverage


METHOD_ROWS = [
    ("opencv", ""),
    ("dav2", "_dav2"),
    ("foundation", "_foundation"),
]


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
            med_s = f"{med:.4f} m" if med is not None else "n/a"
            print(
                f"  {method:12s}  cov={cov:5.1f}%  ray_med={med_s}  "
                f"inlier={row.get('inlier_ratio', 0):.2%}  free_space={fs:.1f}%"
            )
        print()
    else:
        print("Run: python 06_evaluate_run.py --run latest\n")

    if photometric:
        print("=== Photometric (lower is better) ===")
        for key, row in photometric.items():
            print(f"  {key:12s}  mean={row.get('mean_photometric_error')}  samples={row.get('sample_count')}")
        print()

    cross = load_json(val / "cross_method_metrics.json")
    if cross and cross.get("consensus"):
        c = cross["consensus"]
        print(f"=== Cross-method depth std (median {c.get('median_std_m')} m) ===\n")

    for label, suffix in METHOD_ROWS:
        ray = load_json(val / f"lidar_ray_depth_metrics{suffix}.json")
        nn = load_json(val / f"lidar_stereo_error_metrics{suffix}.json")
        disp_path = stereo_dir / (f"disparity{suffix}.npy" if suffix != "_dav2" else "disparity.npy")
        if suffix == "_dav2":
            depth_path = stereo_dir / "depth_metric_dav2.npy"
            extra = ""
            if depth_path.is_file():
                d = np.load(depth_path)
                v = np.isfinite(d) & (d > 0)
                extra = f"depth cov={100*np.count_nonzero(v)/d.size:.1f}%"
        elif disp_path.is_file():
            cov, _, _ = disparity_coverage(np.load(disp_path))
            extra = f"disp cov={cov:.1f}%"
        else:
            extra = "(missing)"

        print(f"=== {label} ===  {extra}")
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
