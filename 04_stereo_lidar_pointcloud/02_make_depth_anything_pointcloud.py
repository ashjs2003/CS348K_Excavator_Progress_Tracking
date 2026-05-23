"""
Monocular depth from Depth Anything V2 (RGB1), scaled to metric meters, then point cloud.

Works on Mac (MPS) and Windows (CUDA) via conda env depth_anything_v2.
See DEPTH_ANYTHING_V2.md and scripts/setup_depth_anything_v2.*

Run:
    python 02_make_depth_anything_pointcloud.py --run latest
    python 02_make_depth_anything_pointcloud.py --run latest --reuse-rectified
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np

from calib_utils import load_camera_calibration, load_stereo_rgb1_to_rgb2
from dav2_scale import (
    depth_map_from_disparity,
    metric_depth_from_relative,
    pointcloud_from_depth_map,
    save_depth_preview,
    write_scale_info,
)
from ml_inference_platform import dav2_setup_command, require_dav2_or_exit
from pointcloud_utils import voxel_downsample, write_ply
from stereo_shared import draw_rectification_check, rectify_stereo_pair

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths, write_run_info

DEFAULT_DA_ROOT = _REPO_ROOT / "third_party" / "Depth-Anything-V2"
DEFAULT_CKPT = DEFAULT_DA_ROOT / "checkpoints" / "depth_anything_v2_vits.pth"
INFER_SCRIPT = Path(__file__).resolve().parent / "tools" / "dav2_infer_depth.py"
SCENE_DEPTH_MIN_M = 0.45
SCENE_DEPTH_MAX_M = 2.0
CONDA_ENV = "depth_anything_v2"


def parse_args():
    parser = argparse.ArgumentParser(description="Depth Anything V2 point cloud (RGB1)")
    add_run_cli_arguments(parser)
    parser.add_argument("--da-root", type=Path, default=DEFAULT_DA_ROOT)
    parser.add_argument("--ckpt", type=Path, default=DEFAULT_CKPT)
    parser.add_argument("--conda-env", default=CONDA_ENV)
    parser.add_argument("--encoder", default="vits", choices=["vits", "vitb", "vitl"])
    parser.add_argument("--input-size", type=int, default=518)
    parser.add_argument("--depth-min", type=float, default=SCENE_DEPTH_MIN_M)
    parser.add_argument("--depth-max", type=float, default=SCENE_DEPTH_MAX_M)
    parser.add_argument("--reuse-rectified", action="store_true")
    parser.add_argument(
        "--scale-from",
        choices=["opencv", "auto"],
        default="auto",
        help="Metric scale: opencv=use disparity.npy; auto=opencv if present else error",
    )
    return parser.parse_args()


def ensure_files(da_root: Path, ckpt: Path) -> None:
    if not da_root.is_dir():
        raise FileNotFoundError(
            f"Depth-Anything-V2 not found at {da_root}\nSetup: {dav2_setup_command()}"
        )
    if not ckpt.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt}\n"
            "Download vits weights per DEPTH_ANYTHING_V2.md"
        )


def run_dav2_inference(image_path, out_relative, da_root, ckpt, conda_env, encoder, input_size):
    env = os.environ.copy()
    env["DEPTH_ANYTHING_V2_ROOT"] = str(da_root.resolve())
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "--no-capture-output",
        "python",
        str(INFER_SCRIPT),
        "--image",
        str(image_path),
        "--ckpt",
        str(ckpt),
        "--out",
        str(out_relative),
        "--encoder",
        encoder,
        "--input-size",
        str(input_size),
    ]
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


def load_reference_depth(stereo_dir: Path, Q: np.ndarray, scale_from: str) -> np.ndarray:
    disp_path = stereo_dir / "disparity.npy"
    if scale_from == "auto" and not disp_path.is_file():
        raise FileNotFoundError(
            f"Need {disp_path} to scale DA-V2 to meters.\n"
            "Run first: python 02_make_stereo_pointcloud.py --run <same_run>"
        )
    if not disp_path.is_file():
        raise FileNotFoundError(f"Missing {disp_path}")
    disparity = np.load(disp_path)
    return depth_map_from_disparity(disparity, Q)


def save_dav2_outputs(out_dir, depth_m, rect1, depth_min, depth_max, scale_info):
    tag = "_dav2"
    np.save(out_dir / f"depth_metric{tag}.npy", depth_m)
    save_depth_preview(out_dir / f"depth_preview{tag}.png", depth_m)
    write_scale_info(out_dir / f"depth_scaling{tag}.json", scale_info)

    K = np.asarray(scale_info["K"], dtype=np.float64)
    points, colors = pointcloud_from_depth_map(depth_m, K, rect1, depth_min, depth_max)
    if len(points) == 0:
        raise RuntimeError("No points after depth filtering; widen --depth-min/--depth-max")
    write_ply(out_dir / f"stereo_pointcloud{tag}.ply", points, colors)
    down_pts, down_cols = voxel_downsample(points, colors, voxel_size=0.02)
    write_ply(out_dir / f"stereo_pointcloud_downsampled{tag}.ply", down_pts, down_cols)

    valid = np.isfinite(depth_m) & (depth_m > 0)
    coverage = 100.0 * float(np.count_nonzero(valid)) / depth_m.size
    return {
        "coverage_pct": coverage,
        "point_count": int(len(points)),
        "point_count_downsampled": int(len(down_pts)),
    }


def main():
    args = parse_args()
    if handle_list_runs(args):
        return

    require_dav2_or_exit(args.conda_env)
    ensure_files(args.da_root, args.ckpt)

    paths = resolve_run_paths(args.run)
    out_dir = paths.stereo
    out_dir.mkdir(parents=True, exist_ok=True)
    if paths.run_dir:
        print(f"Run: {paths.run_dir.name}")

    rgb1_calib = load_camera_calibration("camera_calibration_rgb1.npz")
    rgb2_calib = load_camera_calibration("camera_calibration_rgb2.npz", "camera_calibration_rgb2_approx.npz")
    _, R, t = load_stereo_rgb1_to_rgb2()

    image1 = cv2.imread(str(paths.rgb1_image))
    image2 = cv2.imread(str(paths.rgb2_image))
    if image1 is None or image2 is None:
        raise RuntimeError("Could not load capture images.")

    stereo = rectify_stereo_pair(image1, image2, rgb1_calib, rgb2_calib, R, t)
    Q, P1, rect1 = stereo["Q"], stereo["P1"], stereo["rect1"]
    K = P1[:3, :3]

    rect1_path = out_dir / "rgb1_rectified.png"
    if args.reuse_rectified and rect1_path.is_file():
        rect1 = cv2.imread(str(rect1_path))
        if rect1 is None:
            raise RuntimeError(f"Could not read {rect1_path}")
    else:
        cv2.imwrite(str(rect1_path), rect1)
        cv2.imwrite(str(out_dir / "rgb2_rectified.png"), stereo["rect2"])
        cv2.imwrite(str(out_dir / "rectification_check.png"), draw_rectification_check(rect1, stereo["rect2"]))

    work = out_dir / "depth_anything_v2_work"
    work.mkdir(parents=True, exist_ok=True)
    rel_path = work / "depth_relative.npy"
    run_dav2_inference(
        rect1_path, rel_path, args.da_root, args.ckpt, args.conda_env, args.encoder, args.input_size
    )
    da_relative = np.load(rel_path)

    reference_depth = load_reference_depth(out_dir, Q, args.scale_from)
    depth_m, scale_info = metric_depth_from_relative(da_relative, reference_depth)
    scale_info["K"] = K.tolist()
    scale_info["scale_reference"] = args.scale_from
    scale_info["encoder"] = args.encoder

    stats = save_dav2_outputs(out_dir, depth_m, rect1, args.depth_min, args.depth_max, scale_info)
    print(f"Scaling: {json.dumps({k: scale_info[k] for k in scale_info if k != 'K'}, indent=2)}")
    print(f"DA-V2 coverage: {stats['coverage_pct']:.1f}%  points: {stats['point_count']}")

    if paths.run_dir:
        write_run_info(
            paths.run_dir,
            dav2_coverage_pct=stats["coverage_pct"],
            dav2_point_count=stats["point_count"],
            dav2_point_count_downsampled=stats["point_count_downsampled"],
            dav2_scale_info=scale_info,
        )

    print("Saved:")
    for name in (
        "depth_relative.npy (work/)",
        "depth_metric_dav2.npy",
        "depth_preview_dav2.png",
        "depth_scaling_dav2.json",
        "stereo_pointcloud_dav2.ply",
        "stereo_pointcloud_downsampled_dav2.ply",
    ):
        print(f"  {out_dir} — {name}")


if __name__ == "__main__":
    main()
