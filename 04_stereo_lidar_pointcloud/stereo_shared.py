"""Shared rectification and disparity-to-point-cloud helpers for OpenCV and FoundationStereo."""

from pathlib import Path

import cv2
import numpy as np

from calib_utils import load_camera_calibration, load_stereo_rgb1_to_rgb2
from pointcloud_utils import voxel_downsample, write_ply

MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 5.0
FISHEYE_RECTIFY_BALANCE = 1.0


def is_fisheye_calibration(calib) -> bool:
    return np.asarray(calib["dist"]).size == 4


def stereo_rectify_maps(rgb1_calib, rgb2_calib, image_size, R, t):
    """Rectification maps and Q/P1 — matches 02_make_stereo_pointcloud (fisheye when needed)."""
    if is_fisheye_calibration(rgb1_calib) or is_fisheye_calibration(rgb2_calib):
        print(f"Using fisheye stereo rectification (balance={FISHEYE_RECTIFY_BALANCE:.2f})")
        R1, R2, P1, P2, Q = cv2.fisheye.stereoRectify(
            rgb1_calib["K"],
            rgb1_calib["dist"].reshape(-1, 1),
            rgb2_calib["K"],
            rgb2_calib["dist"].reshape(-1, 1),
            image_size,
            R,
            t.reshape(3, 1),
            flags=cv2.fisheye.CALIB_ZERO_DISPARITY,
            newImageSize=image_size,
            balance=FISHEYE_RECTIFY_BALANCE,
            fov_scale=1.0,
        )
        map1x, map1y = cv2.fisheye.initUndistortRectifyMap(
            rgb1_calib["K"], rgb1_calib["dist"].reshape(-1, 1), R1, P1, image_size, cv2.CV_32FC1
        )
        map2x, map2y = cv2.fisheye.initUndistortRectifyMap(
            rgb2_calib["K"], rgb2_calib["dist"].reshape(-1, 1), R2, P2, image_size, cv2.CV_32FC1
        )
        return R1, R2, P1, P2, Q, map1x, map1y, map2x, map2y

    R1, R2, P1, P2, Q, _, _ = cv2.stereoRectify(
        rgb1_calib["K"],
        rgb1_calib["dist"],
        rgb2_calib["K"],
        rgb2_calib["dist"],
        image_size,
        R,
        t,
        flags=cv2.CALIB_ZERO_DISPARITY,
        alpha=0,
    )
    map1x, map1y = cv2.initUndistortRectifyMap(
        rgb1_calib["K"], rgb1_calib["dist"], R1, P1, image_size, cv2.CV_32FC1
    )
    map2x, map2y = cv2.initUndistortRectifyMap(
        rgb2_calib["K"], rgb2_calib["dist"], R2, P2, image_size, cv2.CV_32FC1
    )
    return R1, R2, P1, P2, Q, map1x, map1y, map2x, map2y


def draw_rectification_check(rect1, rect2, line_step=40):
    combined = np.hstack([rect1, rect2])
    for y in range(0, combined.shape[0], line_step):
        cv2.line(combined, (0, y), (combined.shape[1] - 1, y), (0, 255, 255), 1)
    return combined


def save_disparity_preview(path, disparity):
    valid_disp = disparity > 0
    preview = np.zeros_like(disparity, dtype=np.uint8)
    if np.any(valid_disp):
        valid_values = disparity[valid_disp]
        lo, hi = np.percentile(valid_values, [5, 95])
        if hi <= lo:
            lo, hi = float(np.min(valid_values)), float(np.max(valid_values))
        scaled = np.clip((valid_values - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
        preview[valid_disp] = (scaled * 255.0).astype(np.uint8)
    preview_color = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    preview_color[~valid_disp] = (0, 0, 0)
    cv2.imwrite(str(path), preview_color)


def disparity_coverage(disparity):
    valid = disparity > 0
    if not np.any(valid):
        return 0.0, 0.0, 0.0
    values = disparity[valid]
    return (
        100.0 * float(np.count_nonzero(valid)) / disparity.size,
        float(np.min(values)),
        float(np.max(values)),
    )


def rectify_stereo_pair(image1, image2, rgb1_calib, rgb2_calib, R, t):
    """Rectify a capture pair; returns rectified images and stereo geometry."""
    image_size = (image1.shape[1], image1.shape[0])
    R1, R2, P1, P2, Q, map1x, map1y, map2x, map2y = stereo_rectify_maps(
        rgb1_calib, rgb2_calib, image_size, R, t
    )
    rect1 = cv2.remap(image1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(image2, map2x, map2y, cv2.INTER_LINEAR)
    baseline_m = float(np.linalg.norm(t))
    return {
        "rect1": rect1,
        "rect2": rect2,
        "Q": Q,
        "P1": P1,
        "R1": R1,
        "baseline_m": baseline_m,
        "image_size": image_size,
    }


def write_foundation_intrinsic_file(path, P1, baseline_m):
    """K.txt format expected by FoundationStereo demo (rectified left camera)."""
    K = P1[:3, :3].astype(np.float64)
    flat = " ".join(f"{v:.12g}" for v in K.reshape(-1))
    Path(path).write_text(f"{flat}\n{baseline_m:.12g}\n")


def pointcloud_from_disparity(
    disparity,
    Q,
    rect1_bgr,
    depth_min_m,
    depth_max_m,
    min_depth_m=MIN_DEPTH_M,
    max_depth_m=MAX_DEPTH_M,
):
    """Build colored points in rectified RGB1 frame from a disparity map."""
    valid_disp = disparity > 0
    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    raw_depth = points_3d[:, :, 2]
    raw_valid_depth = raw_depth[valid_disp & np.isfinite(raw_depth)]
    if len(raw_valid_depth) and np.median(raw_valid_depth) < 0:
        points_3d *= -1.0

    finite = np.isfinite(points_3d).all(axis=2)
    depth = points_3d[:, :, 2]
    mask = (
        valid_disp
        & finite
        & (depth >= max(min_depth_m, depth_min_m))
        & (depth <= min(max_depth_m, depth_max_m))
    )
    points = points_3d[mask]
    colors = cv2.cvtColor(rect1_bgr, cv2.COLOR_BGR2RGB)[mask].astype(np.float64) / 255.0
    return points, colors


def save_stereo_pointclouds(
    out_dir,
    disparity,
    Q,
    rect1,
    depth_min_m,
    depth_max_m,
    prefix="",
):
    """Write disparity, preview, full PLY, and downsampled PLY with optional filename prefix."""
    out_dir = Path(out_dir)
    tag = prefix or ""
    np.save(out_dir / f"disparity{tag}.npy", disparity)
    save_disparity_preview(out_dir / f"disparity_preview{tag}.png", disparity)
    points, colors = pointcloud_from_disparity(
        disparity, Q, rect1, depth_min_m, depth_max_m
    )
    if len(points) == 0:
        raise RuntimeError(
            f"No stereo points for disparity{tag}; check disparity_preview{tag}.png"
        )
    write_ply(out_dir / f"stereo_pointcloud{tag}.ply", points, colors)
    down_points, down_colors = voxel_downsample(points, colors, voxel_size=0.02)
    write_ply(out_dir / f"stereo_pointcloud_downsampled{tag}.ply", down_points, down_colors)
    coverage, disp_min, disp_max = disparity_coverage(disparity)
    return {
        "coverage_pct": float(coverage),
        "disp_min": disp_min,
        "disp_max": disp_max,
        "point_count": int(len(points)),
        "point_count_downsampled": int(len(down_points)),
    }
