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
    opencv_gt_anchor_mask,
    pointcloud_from_depth_map,
    save_depth_preview,
    write_scale_info,
)
from depth_layout import ensure_method_tree, method_dir, resolve_path, shared_dir
from evaluation.gt_depth_overlay import load_gt_reference_for_run
from evaluation.depth_maps import load_or_compute_stereo_geometry
from ml_inference_platform import dav2_python_command, dav2_setup_command, require_dav2_or_exit
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
        "--scale-modes",
        choices=["both", "opencv", "opencv-gt"],
        default="both",
        help=(
            "Metric scaling variants to save: both=dav2 + dav2_gt (default); "
            "opencv=all valid OpenCV Z; opencv-gt=GT-anchor pixels only (needs pair_*.txt)"
        ),
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
        *dav2_python_command(conda_env),
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
    disp_path = resolve_path(stereo_dir, "opencv", "disparity.npy")
    if disp_path is None:
        raise FileNotFoundError(
            "Need OpenCV disparity to scale DA-V2 to meters.\n"
            "Run first: python 02_make_stereo_pointcloud.py --run <same_run>"
        )
    disparity = np.load(disp_path)
    return depth_map_from_disparity(disparity, Q)


def save_dav2_outputs(
    out_dir: Path,
    depth_m: np.ndarray,
    rect1: np.ndarray,
    depth_min: float,
    depth_max: float,
    scale_info: dict,
    *,
    method_id: str,
) -> dict:
    mdir = method_dir(out_dir, method_id)
    mdir.mkdir(parents=True, exist_ok=True)
    np.save(mdir / "depth_metric.npy", depth_m)
    save_depth_preview(mdir / "depth_preview.png", depth_m, vmin_m=depth_min, vmax_m=depth_max)
    write_scale_info(mdir / "scaling.json", scale_info)

    K = np.asarray(scale_info["K"], dtype=np.float64)
    points, colors = pointcloud_from_depth_map(depth_m, K, rect1, depth_min, depth_max)
    if len(points) == 0:
        raise RuntimeError(
            f"No points after depth filtering for {method_id}; widen --depth-min/--depth-max"
        )
    write_ply(mdir / "pointcloud.ply", points, colors)
    down_pts, down_cols = voxel_downsample(points, colors, voxel_size=0.02)
    write_ply(mdir / "pointcloud_downsampled.ply", down_pts, down_cols)

    valid = np.isfinite(depth_m) & (depth_m > 0)
    coverage = 100.0 * float(np.count_nonzero(valid)) / depth_m.size
    return {
        "method_id": method_id,
        "coverage_pct": coverage,
        "point_count": int(len(points)),
        "point_count_downsampled": int(len(down_pts)),
        "scale_info": scale_info,
    }


def _scale_modes_list(scale_modes: str) -> list[str]:
    if scale_modes == "both":
        return ["opencv", "opencv-gt"]
    return [scale_modes]


def _fit_dav2_variant(
    da_relative: np.ndarray,
    reference_depth: np.ndarray,
    mode: str,
    *,
    run_dir: Path | None,
    K: np.ndarray,
    encoder: str,
) -> tuple[np.ndarray, dict]:
    fit_mask = None
    extra: dict = {"scale_reference": mode, "method_id": "dav2" if mode == "opencv" else "dav2_gt"}

    if mode == "opencv-gt":
        gt_ref = load_gt_reference_for_run(run_dir, _REPO_ROOT)
        target_gt_m = gt_ref["target_gt_m"]
        if target_gt_m is None:
            raise RuntimeError(
                "opencv-gt scaling needs data/<scene>/pair_<id>.txt with a numeric target distance."
            )
        fit_mask = opencv_gt_anchor_mask(
            reference_depth,
            target_gt_m,
            gt_ref["wall_gt_m"],
            gt_ref["tolerance_m"],
        )
        n_anchor = int(np.count_nonzero(fit_mask))
        print(
            f"GT anchor fit: target={target_gt_m * 100:.1f} cm, wall={gt_ref['wall_gt_cm']:.0f} cm, "
            f"±{gt_ref['tolerance_cm']:.0f} cm — {n_anchor} OpenCV pixels "
            f"({100.0 * n_anchor / fit_mask.size:.2f}%)"
        )
        extra.update(
            {
                "gt_txt": str(gt_ref["txt_path"]) if gt_ref["txt_path"] else None,
                "target_gt_cm": gt_ref["target_gt_cm"],
                "wall_gt_cm": gt_ref["wall_gt_cm"],
                "tolerance_cm": gt_ref["tolerance_cm"],
                "anchor_pixels": n_anchor,
                "anchor_pixels_pct": 100.0 * n_anchor / fit_mask.size,
            }
        )
    elif mode != "opencv":
        raise ValueError(f"Unknown scale mode {mode!r}")

    depth_m, scale_info = metric_depth_from_relative(
        da_relative, reference_depth, fit_mask=fit_mask
    )
    scale_info["K"] = K.tolist()
    scale_info["encoder"] = encoder
    scale_info.update(extra)
    return depth_m, scale_info


def main():
    args = parse_args()
    if handle_list_runs(args):
        return

    require_dav2_or_exit(args.conda_env)
    ensure_files(args.da_root, args.ckpt)

    paths = resolve_run_paths(args.run)
    out_dir = paths.depth
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

    geometry = load_or_compute_stereo_geometry(out_dir, paths.rgb1_image, paths.rgb2_image)
    Q = geometry["Q"]
    K = geometry["P1"][:3, :3]

    ensure_method_tree(out_dir)
    sdir = shared_dir(out_dir)
    rect1_path = sdir / "rgb1_rectified.png"
    if args.reuse_rectified and rect1_path.is_file():
        rect1 = cv2.imread(str(rect1_path))
        if rect1 is None:
            raise RuntimeError(f"Could not read {rect1_path}")
    else:
        stereo = rectify_stereo_pair(image1, image2, rgb1_calib, rgb2_calib, R, t)
        rect1 = stereo["rect1"]
        cv2.imwrite(str(rect1_path), rect1)
        cv2.imwrite(str(sdir / "rgb2_rectified.png"), stereo["rect2"])
        cv2.imwrite(str(sdir / "rectification_check.png"), draw_rectification_check(rect1, stereo["rect2"]))

    work = method_dir(out_dir, "dav2") / "work"
    work.mkdir(parents=True, exist_ok=True)
    rel_path = work / "depth_relative.npy"
    run_dav2_inference(
        rect1_path, rel_path, args.da_root, args.ckpt, args.conda_env, args.encoder, args.input_size
    )
    da_relative = np.load(rel_path)

    reference_depth = load_reference_depth(out_dir, Q, "opencv")
    mode_to_method = {"opencv": "dav2", "opencv-gt": "dav2_gt"}
    saved_variants: list[dict] = []

    for mode in _scale_modes_list(args.scale_modes):
        method_id = mode_to_method[mode]
        print(f"\n--- DA-V2 scale mode: {mode} → {method_id}/ ---")
        try:
            depth_m, scale_info = _fit_dav2_variant(
                da_relative,
                reference_depth,
                mode,
                run_dir=paths.run_dir,
                K=K,
                encoder=args.encoder,
            )
        except RuntimeError as exc:
            if mode == "opencv-gt" and args.scale_modes == "both":
                print(f"Skipping GT-anchor variant: {exc}")
                continue
            raise

        stats = save_dav2_outputs(
            out_dir, depth_m, rect1, args.depth_min, args.depth_max, scale_info, method_id=method_id
        )
        saved_variants.append(stats)
        print(f"Scaling: {json.dumps({k: scale_info[k] for k in scale_info if k != 'K'}, indent=2)}")
        print(f"Coverage: {stats['coverage_pct']:.1f}%  points: {stats['point_count']}")

    if not saved_variants:
        raise RuntimeError("No DA-V2 depth variants were saved.")

    if paths.run_dir:
        info_kwargs: dict = {}
        for stats in saved_variants:
            mid = stats["method_id"]
            si = stats["scale_info"]
            if mid == "dav2":
                info_kwargs["dav2_coverage_pct"] = stats["coverage_pct"]
                info_kwargs["dav2_point_count"] = stats["point_count"]
                info_kwargs["dav2_point_count_downsampled"] = stats["point_count_downsampled"]
                info_kwargs["dav2_scale_info"] = si
            elif mid == "dav2_gt":
                info_kwargs["dav2_gt_coverage_pct"] = stats["coverage_pct"]
                info_kwargs["dav2_gt_point_count"] = stats["point_count"]
                info_kwargs["dav2_gt_point_count_downsampled"] = stats["point_count_downsampled"]
                info_kwargs["dav2_gt_scale_info"] = si
        write_run_info(paths.run_dir, **info_kwargs)

    print("\nSaved:")
    print(f"  {method_dir(out_dir, 'dav2') / 'work'} — depth_relative.npy")
    for stats in saved_variants:
        mdir = method_dir(out_dir, stats["method_id"])
        for name in (
            "depth_metric.npy",
            "depth_preview.png",
            "scaling.json",
            "pointcloud.ply",
            "pointcloud_downsampled.ply",
        ):
            print(f"  {mdir} — {name}")


if __name__ == "__main__":
    main()
