"""
Build a stereo point cloud using FoundationStereo (GPU) on the same rectified pair as OpenCV.

Requires setup: see FOUNDATIONSTEREO.md and scripts/setup_foundationstereo.sh

Run (from stereo_lidar_pointcloud/, on a CUDA machine):
    python 02_make_stereo_pointcloud_foundation.py
    python 02_make_stereo_pointcloud_foundation.py --run 20260521_222300_legacy_import

Outputs sit beside OpenCV files with a _foundation suffix:
    disparity_foundation.npy
    disparity_preview_foundation.png
    stereo_pointcloud_foundation.ply
    stereo_pointcloud_downsampled_foundation.ply
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from calib_utils import load_camera_calibration, load_stereo_rgb1_to_rgb2
from foundation_stereo_platform import require_foundation_stereo_or_exit, setup_command
from stereo_shared import (
    draw_rectification_check,
    rectify_stereo_pair,
    save_stereo_pointclouds,
    write_foundation_intrinsic_file,
)

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths, write_run_info

DEFAULT_FS_ROOT = _REPO_ROOT / "third_party" / "FoundationStereo"
DEFAULT_CKPT = DEFAULT_FS_ROOT / "pretrained_models" / "23-51-11" / "model_best_bp2.pth"
INFER_SCRIPT = Path(__file__).resolve().parent / "tools" / "fs_infer_disparity.py"
SCENE_DEPTH_MIN_M = 0.45
SCENE_DEPTH_MAX_M = 2.0


def parse_args():
    parser = argparse.ArgumentParser(description="FoundationStereo disparity + point cloud")
    add_run_cli_arguments(parser)
    parser.add_argument(
        "--fs-root",
        type=Path,
        default=DEFAULT_FS_ROOT,
        help="Cloned FoundationStereo repository",
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        default=DEFAULT_CKPT,
        help="Path to model_best_bp2.pth",
    )
    parser.add_argument(
        "--conda-env",
        default="foundation_stereo",
        help="Conda env name with FoundationStereo dependencies + CUDA",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=1.0,
        help="Inference scale (<=1). 0.5 is faster; 1.0 matches OpenCV resolution.",
    )
    parser.add_argument("--valid-iters", type=int, default=16, help="FoundationStereo refinement iterations")
    parser.add_argument("--hiera", type=int, default=0, help="Hierarchical inference for >1K images")
    parser.add_argument("--depth-min", type=float, default=SCENE_DEPTH_MIN_M)
    parser.add_argument("--depth-max", type=float, default=SCENE_DEPTH_MAX_M)
    parser.add_argument(
        "--reuse-rectified",
        action="store_true",
        help="Use existing rgb1_rectified.png / rgb2_rectified.png if present",
    )
    return parser.parse_args()


def ensure_setup(fs_root: Path, ckpt: Path, conda_env: str) -> None:
    if not fs_root.is_dir():
        raise FileNotFoundError(
            f"FoundationStereo not found at {fs_root}\n"
            f"Setup:\n  {setup_command()}"
        )
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Download weights per FOUNDATIONSTEREO.md (Google Drive link in NVlabs README)."
        )
    if shutil.which("conda") is None:
        raise RuntimeError("conda not found on PATH; install Miniconda/Anaconda for FoundationStereo.")
    probe = subprocess.run(
        ["conda", "run", "-n", conda_env, "python", "-c", "import torch; print(torch.cuda.is_available())"],
        capture_output=True,
        text=True,
    )
    if probe.returncode != 0:
        raise RuntimeError(
            f"Conda env '{conda_env}' is missing or broken.\n"
            f"Create it with: bash scripts/setup_foundationstereo.sh\n{probe.stderr}"
        )
    if probe.stdout.strip() != "True":
        raise RuntimeError(
            f"Conda env '{conda_env}' has no CUDA GPU available.\n"
            "FoundationStereo must run on an NVIDIA machine (not Apple Silicon CPU-only)."
        )


def run_foundation_disparity(
    rect1_path: Path,
    rect2_path: Path,
    out_disp: Path,
    fs_root: Path,
    ckpt: Path,
    conda_env: str,
    scale: float,
    valid_iters: int,
    hiera: int,
) -> None:
    env = os.environ.copy()
    env["FOUNDATION_STEREO_ROOT"] = str(fs_root.resolve())
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "--no-capture-output",
        "python",
        str(INFER_SCRIPT),
        "--left",
        str(rect1_path),
        "--right",
        str(rect2_path),
        "--ckpt",
        str(ckpt),
        "--out_disp",
        str(out_disp),
        "--scale",
        str(scale),
        "--valid_iters",
        str(valid_iters),
        "--hiera",
        str(hiera),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def main():
    args = parse_args()
    if handle_list_runs(args):
        return

    require_foundation_stereo_or_exit()
    # Clone + weights check (Windows/Linux); skipped on macOS via require_* above.
    ensure_setup(args.fs_root, args.ckpt, args.conda_env)
    paths = resolve_run_paths(args.run)
    out_dir = paths.depth
    out_dir.mkdir(parents=True, exist_ok=True)

    if paths.run_dir:
        print(f"Run: {paths.run_dir.name}")
    print(f"FoundationStereo root: {args.fs_root}")
    print(f"Checkpoint: {args.ckpt}")
    print(f"Writing FoundationStereo outputs to {out_dir}")

    rgb1_calib = load_camera_calibration("camera_calibration_rgb1.npz")
    rgb2_calib = load_camera_calibration("camera_calibration_rgb2.npz", "camera_calibration_rgb2_approx.npz")
    stereo_path, R, t = load_stereo_rgb1_to_rgb2()
    print(f"Loaded stereo extrinsics: {stereo_path}")

    rect1_path = out_dir / "rgb1_rectified.png"
    rect2_path = out_dir / "rgb2_rectified.png"
    image1 = cv2.imread(str(paths.rgb1_image))
    image2 = cv2.imread(str(paths.rgb2_image))
    if image1 is None or image2 is None:
        raise RuntimeError("Could not load capture images.")
    stereo = rectify_stereo_pair(image1, image2, rgb1_calib, rgb2_calib, R, t)
    Q, P1 = stereo["Q"], stereo["P1"]

    if args.reuse_rectified and rect1_path.is_file() and rect2_path.is_file():
        rect1 = cv2.imread(str(rect1_path))
        rect2 = cv2.imread(str(rect2_path))
        if rect1 is None or rect2 is None:
            raise RuntimeError("Could not load existing rectified images.")
    else:
        rect1, rect2 = stereo["rect1"], stereo["rect2"]
        cv2.imwrite(str(rect1_path), rect1)
        cv2.imwrite(str(rect2_path), rect2)
        cv2.imwrite(str(out_dir / "rectification_check.png"), draw_rectification_check(rect1, rect2))

    k_txt = out_dir / "K_rectified_for_foundationstereo.txt"
    write_foundation_intrinsic_file(k_txt, P1, stereo["baseline_m"])

    work_dir = out_dir / "foundation_stereo_work"
    work_dir.mkdir(parents=True, exist_ok=True)
    disp_path = work_dir / "disparity_raw.npy"
    run_foundation_disparity(
        rect1_path,
        rect2_path,
        disp_path,
        args.fs_root,
        args.ckpt,
        args.conda_env,
        args.scale,
        args.valid_iters,
        args.hiera,
    )
    disparity = np.load(disp_path)

    stats = save_stereo_pointclouds(
        out_dir,
        disparity,
        Q,
        rect1,
        args.depth_min,
        args.depth_max,
        prefix="_foundation",
    )
    print(
        f"FoundationStereo coverage: {stats['coverage_pct']:.1f}% "
        f"(disp {stats['disp_min']:.1f}..{stats['disp_max']:.1f} px)"
    )
    print(f"FoundationStereo points: {stats['point_count']} "
          f"(downsampled {stats['point_count_downsampled']})")

    if paths.run_dir:
        write_run_info(
            paths.run_dir,
            foundation_disparity_coverage_pct=stats["coverage_pct"],
            foundation_stereo_point_count=stats["point_count"],
            foundation_stereo_point_count_downsampled=stats["point_count_downsampled"],
            foundation_stereo_ckpt=str(args.ckpt),
            foundation_stereo_scale=float(args.scale),
        )

    print("Saved:")
    for name in (
        "disparity_foundation.npy",
        "disparity_preview_foundation.png",
        "stereo_pointcloud_foundation.ply",
        "stereo_pointcloud_downsampled_foundation.ply",
    ):
        print(f"  {out_dir / name}")


if __name__ == "__main__":
    main()
