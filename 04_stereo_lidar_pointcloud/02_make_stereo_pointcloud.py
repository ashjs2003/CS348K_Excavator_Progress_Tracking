"""
Build a colored stereo point cloud from the latest capture run.

Run:
    python 02_make_stereo_pointcloud.py
    python 02_make_stereo_pointcloud.py --method stereobm
    python 02_make_stereo_pointcloud.py --run 20260521_143022_carpet --list-runs
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from calib_utils import load_camera_calibration, load_stereo_rgb1_to_rgb2
from dav2_scale import depth_map_from_disparity
from evaluation.depth_maps import save_stereo_geometry
from pointcloud_utils import voxel_downsample, write_ply
from stereo_methods import DEFAULT_STEREO_METHOD, normalize_stereo_method
from stereo_shared import stereo_rectify_maps

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths, write_run_info
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 5.0
# Scene depth band for carpet-on-floor captures (filters ambiguous full-range matches).
SCENE_DEPTH_MIN_M = 0.45
SCENE_DEPTH_MAX_M = 2.0

# SGBM tuning: repetitive carpet/cloth needs looser uniqueness + larger blocks.
SGBM_BLOCK_SIZE = 7
SGBM_UNIQUENESS_RATIO = 8
SGBM_SPECKLE_WINDOW = 150
SGBM_SPECKLE_RANGE = 32
USE_CLAHE = True
MEDIAN_FILTER_K = 5
LR_MAX_DIFF_PX = 2.5
# Rectified flow still has vertical component when RGB2 intrinsics are approximate.
FLOW_EPIPOLAR_MAX_PX = 8.0
def draw_rectification_check(rect1, rect2, line_step=40):
    """Stack rectified images side-by-side and draw horizontal guide lines."""
    combined = np.hstack([rect1, rect2])
    for y in range(0, combined.shape[0], line_step):
        cv2.line(combined, (0, y), (combined.shape[1] - 1, y), (0, 255, 255), 1)
    return combined


def make_sgbm(num_disparities):
    """Create SGBM matcher. numDisparities must be divisible by 16."""
    block_size = SGBM_BLOCK_SIZE
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=2,
        uniquenessRatio=SGBM_UNIQUENESS_RATIO,
        speckleWindowSize=SGBM_SPECKLE_WINDOW,
        speckleRange=SGBM_SPECKLE_RANGE,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM,
    )


def prepare_gray_for_matching(gray):
    """Boost local contrast — helps carpets with lighting gradients."""
    if not USE_CLAHE:
        return gray
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    return clahe.apply(gray)


def refine_disparity(disparity):
    """Light denoise on valid disparity pixels."""
    if MEDIAN_FILTER_K < 3 or not np.any(disparity > 0):
        return disparity
    refined = disparity.copy()
    filtered = cv2.medianBlur(refined, MEDIAN_FILTER_K)
    valid = disparity > 0
    refined[valid] = filtered[valid]
    return refined


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


def focal_length_px(rgb1_calib) -> float:
    K = rgb1_calib["K"]
    return float((K[0, 0] + K[1, 1]) / 2.0)


def disparity_search_range(
    rgb1_calib,
    baseline_m,
    depth_min_m=SCENE_DEPTH_MIN_M,
    depth_max_m=SCENE_DEPTH_MAX_M,
):
    """
    Search only disparities consistent with scene depth (d = f * B / Z).

    A 160 px search on repetitive carpet invites wrong matches at unrelated depths.
    """
    focal = focal_length_px(rgb1_calib)
    d_near = focal * baseline_m / max(depth_min_m, 0.2)
    d_far = focal * baseline_m / max(depth_max_m, 0.25)
    if d_near < d_far:
        d_near, d_far = d_far, d_near
    # Search from zero; cap width so carpet cannot lock onto unrelated far matches.
    min_disparity = 0
    span = int(np.ceil(d_near - d_far) + 20)
    num_disparities = int(np.clip(((max(48, span) + 15) // 16) * 16, 64, 128))
    return min_disparity, num_disparities


def reproject_depth_z(disparity, Q) -> np.ndarray:
    points = cv2.reprojectImageTo3D(disparity, Q)
    z = points[:, :, 2].astype(np.float32)
    return np.where(np.isfinite(z), z, np.nan)


def score_disparity_depth(disparity, Q, depth_min_m, depth_max_m) -> float:
    valid = disparity > 0
    n_valid = int(np.count_nonzero(valid))
    if n_valid < 500:
        return -1.0
    z = reproject_depth_z(disparity, Q)[valid]
    z = np.abs(z) if np.nanmedian(z) < 0 else z
    in_band = (z >= depth_min_m) & (z <= depth_max_m)
    depth_frac = float(np.count_nonzero(in_band)) / len(z)
    coverage_frac = n_valid / disparity.size
    return depth_frac * coverage_frac


def left_right_consistency(disp_left, disp_right, max_diff=LR_MAX_DIFF_PX):
    """Keep left-view disparities that agree with a right-to-left match."""
    out = np.zeros_like(disp_left)
    h, w = disp_left.shape
    for y in range(h):
        row_l = disp_left[y]
        row_r = disp_right[y]
        valid = row_l > 0
        if not np.any(valid):
            continue
        xs = np.flatnonzero(valid)
        d = row_l[xs]
        xr = np.round(xs - d).astype(np.int32)
        in_bounds = (xr >= 0) & (xr < w)
        xs, d, xr = xs[in_bounds], d[in_bounds], xr[in_bounds]
        if len(xs) == 0:
            continue
        agree = np.abs(row_r[xr] - d) <= max_diff
        keep = xs[agree]
        out[y, keep] = row_l[keep]
    return out


def filter_disparity_depth_band(disparity, Q, depth_min_m, depth_max_m):
    z = reproject_depth_z(disparity, Q)
    valid = disparity > 0
    use_z = np.abs(z) if np.nanmedian(z[valid]) < 0 else z
    mask = valid & np.isfinite(use_z) & (use_z >= depth_min_m) & (use_z <= depth_max_m)
    filtered = disparity.copy()
    filtered[~mask] = 0.0
    return filtered, bool(np.nanmedian(z[valid]) < 0 if np.any(valid) else False)


def make_stereo_bm(num_disparities, min_disparity=0, *, block_size=21, uniqueness=5):
    matcher = cv2.StereoBM.create(numDisparities=num_disparities, blockSize=block_size)
    matcher.setPreFilterCap(31)
    matcher.setBlockSize(block_size)
    matcher.setMinDisparity(int(min_disparity))
    matcher.setNumDisparities(int(num_disparities))
    matcher.setTextureThreshold(0)
    matcher.setUniquenessRatio(int(uniqueness))
    matcher.setSpeckleWindowSize(100)
    matcher.setSpeckleRange(16)
    matcher.setDisp12MaxDiff(2)
    return matcher


def run_stereo_bm(gray_left, gray_right, min_disparity, num_disparities, **bm_kwargs):
    matcher = make_stereo_bm(num_disparities, min_disparity, **bm_kwargs)
    raw = matcher.compute(gray_left, gray_right).astype(np.float32) / 16.0
    raw[raw < float(min_disparity)] = 0.0
    return raw


def compute_disparity_sgbm(gray1, gray2, min_disparity, num_disparities):
    matcher = make_sgbm(num_disparities)
    matcher.setMinDisparity(int(min_disparity))
    print(
        f"SGBM minDisparity={matcher.getMinDisparity()}, "
        f"numDisparities={matcher.getNumDisparities()}, blockSize={matcher.getBlockSize()}"
    )
    raw = matcher.compute(gray1, gray2).astype(np.float32) / 16.0
    raw[raw < float(min_disparity)] = 0.0
    return refine_disparity(raw)


def compute_disparity_bm(gray1, gray2, min_disparity, num_disparities, Q=None, depth_min_m=SCENE_DEPTH_MIN_M, depth_max_m=SCENE_DEPTH_MAX_M):
    """
    OpenCV StereoBM (stereobm): narrow search band, stricter uniqueness, LR check.
    """
    block_size = 21
    forward = run_stereo_bm(gray1, gray2, min_disparity, num_disparities, block_size=block_size, uniqueness=5)
    reverse = run_stereo_bm(gray2, gray1, min_disparity, num_disparities, block_size=block_size, uniqueness=5)

    forward_lr = refine_disparity(left_right_consistency(forward, reverse))
    reverse_lr = refine_disparity(left_right_consistency(reverse, forward))

    variants = [("rgb1->rgb2 raw", forward), ("rgb1->rgb2 LR", forward_lr)]
    if Q is not None:
        rev_score = score_disparity_depth(reverse_lr, Q, depth_min_m, depth_max_m)
        if rev_score > 0:
            variants.append(("rgb2->rgb1 LR", reverse_lr))

    candidates = []
    for label, disp in variants:
        score = score_disparity_depth(disp, Q, depth_min_m, depth_max_m) if Q is not None else 0.0
        cov, dmin, dmax = disparity_coverage(disp)
        candidates.append((score, cov, label, disp))
        print(
            f"StereoBM {label}: numDisp={num_disparities}, block={block_size}, "
            f"coverage={cov:.1f}% (range {dmin:.1f}..{dmax:.1f}), score={score:.4f}"
        )

    best = max(candidates, key=lambda item: (item[0], item[1]))
    print(f"StereoBM selected: {best[2]} (score={best[0]:.4f}, coverage={best[1]:.1f}%)")
    return best[3]


def compute_disparity_flow(gray1, gray2, min_disparity, num_disparities):
    """Dense optical flow on rectified views (use --method stereobm if this is sparse)."""
    max_disp = float(num_disparities)
    flow = cv2.calcOpticalFlowFarneback(
        gray1,
        gray2,
        None,
        pyr_scale=0.5,
        levels=5,
        winsize=31,
        iterations=5,
        poly_n=7,
        poly_sigma=1.5,
        flags=0,
    )
    flow_x = flow[:, :, 0]
    flow_y = flow[:, :, 1]
    epi_mask = np.abs(flow_y) < FLOW_EPIPOLAR_MAX_PX

    best = None
    for sign in (-1.0, 1.0):
        disp = np.maximum(sign * flow_x, 0.0)
        valid = epi_mask & (disp >= float(min_disparity)) & (disp < max_disp)
        count = int(np.count_nonzero(valid))
        if best is None or count > best[0]:
            best = (count, sign, disp, valid)

    count, sign, disparity, valid = best
    print(
        f"Optical-flow disparity: sign={sign:+.0f}, coverage={100.0 * count / disparity.size:.1f}%, "
        f"epipolar_max_y={FLOW_EPIPOLAR_MAX_PX:.1f}px"
    )
    disparity[~valid] = 0.0
    return refine_disparity(disparity)


def compute_disparity_blend(gray1, gray2, min_disparity, num_disparities, Q=None):
    sgbm = compute_disparity_sgbm(gray1, gray2, min_disparity, num_disparities)
    stereobm_disp = compute_disparity_bm(gray1, gray2, min_disparity, num_disparities, Q=Q)
    valid_sgbm = sgbm > 0
    out = stereobm_disp.copy()
    out[valid_sgbm] = sgbm[valid_sgbm]
    print(
        f"Blend: kept SGBM on {100.0 * np.count_nonzero(valid_sgbm) / sgbm.size:.1f}% pixels, "
        f"StereoBM fill elsewhere"
    )
    return refine_disparity(out)


def compute_disparity(gray1, gray2, min_disparity, num_disparities, method, Q=None, depth_min_m=SCENE_DEPTH_MIN_M, depth_max_m=SCENE_DEPTH_MAX_M):
    if method == "sgbm":
        disp = compute_disparity_sgbm(gray1, gray2, min_disparity, num_disparities)
    elif method == "stereobm":
        disp = compute_disparity_bm(gray1, gray2, min_disparity, num_disparities, Q=Q, depth_min_m=depth_min_m, depth_max_m=depth_max_m)
    elif method == "flow":
        disp = compute_disparity_flow(gray1, gray2, min_disparity, num_disparities)
    elif method == "blend":
        disp = compute_disparity_blend(gray1, gray2, min_disparity, num_disparities, Q=Q)
    else:
        raise ValueError(f"Unknown method: {method}")

    if Q is not None:
        before_cov, _, _ = disparity_coverage(disp)
        disp, _ = filter_disparity_depth_band(disp, Q, depth_min_m, depth_max_m)
        after_cov, dmin, dmax = disparity_coverage(disp)
        print(
            f"Depth band filter ({depth_min_m:.2f}..{depth_max_m:.2f} m): "
            f"coverage {before_cov:.1f}% -> {after_cov:.1f}% (disp {dmin:.1f}..{dmax:.1f} px)"
        )
    return disp


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


def save_point_cloud(path, points, colors):
    write_ply(path, points, colors)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Stereo point cloud from capture/rgb1.png + rgb2.png"
    )
    parser.add_argument(
        "--method",
        type=normalize_stereo_method,
        default=DEFAULT_STEREO_METHOD,
        help="stereobm (OpenCV StereoBM, default), sgbm, flow, blend; aliases: carpet, bm",
    )
    parser.add_argument("--depth-min", type=float, default=SCENE_DEPTH_MIN_M, help="Min plausible depth (m) for disparity filter")
    parser.add_argument("--depth-max", type=float, default=SCENE_DEPTH_MAX_M, help="Max plausible depth (m) for disparity filter")
    add_run_cli_arguments(parser)
    return parser.parse_args()


def main():
    args = parse_args()
    if handle_list_runs(args):
        return
    paths = resolve_run_paths(args.run)
    out_dir = paths.depth
    rgb1_image = paths.rgb1_image
    rgb2_image = paths.rgb2_image
    if paths.run_dir:
        print(f"Run: {paths.run_dir.name}")
    print(f"Disparity method: {args.method}")
    print(f"Reading capture from {paths.capture}")
    print(f"Writing stereo outputs to {out_dir}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rgb1_calib = load_camera_calibration("camera_calibration_rgb1.npz")
    rgb2_calib = load_camera_calibration("camera_calibration_rgb2.npz", "camera_calibration_rgb2_approx.npz")
    stereo_path, R, t = load_stereo_rgb1_to_rgb2()

    image1 = cv2.imread(str(rgb1_image))
    image2 = cv2.imread(str(rgb2_image))
    if image1 is None:
        raise RuntimeError(f"Could not load {rgb1_image}")
    if image2 is None:
        raise RuntimeError(f"Could not load {rgb2_image}")

    image_size = (image1.shape[1], image1.shape[0])
    baseline_m = float(np.linalg.norm(t))
    min_disparity, num_disparities = disparity_search_range(
        rgb1_calib, baseline_m, args.depth_min, args.depth_max
    )
    print(f"Loaded stereo extrinsics: {stereo_path}")
    print(f"Image size: {image_size[0]}x{image_size[1]}")
    print(f"Stereo baseline: {baseline_m * 100:.2f} cm")
    print(
        f"Disparity search: min={min_disparity} px, num={num_disparities} px "
        f"(depth band {args.depth_min:.2f}..{args.depth_max:.2f} m)"
    )

    # Rectify both cameras into a common stereo geometry. Q converts disparity
    # back into 3D points in the rectified RGB1 camera frame.
    R1, R2, P1, P2, Q, map1x, map1y, map2x, map2y = stereo_rectify_maps(
        rgb1_calib, rgb2_calib, image_size, R, t
    )
    rect1 = cv2.remap(image1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(image2, map2x, map2y, cv2.INTER_LINEAR)

    cv2.imwrite(str(out_dir / "rgb1_rectified.png"), rect1)
    cv2.imwrite(str(out_dir / "rgb2_rectified.png"), rect2)
    cv2.imwrite(str(out_dir / "rectification_check.png"), draw_rectification_check(rect1, rect2))
    save_stereo_geometry(out_dir, R1, P1, Q, baseline_m, image_size)

    gray1 = prepare_gray_for_matching(cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY))
    gray2 = prepare_gray_for_matching(cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY))
    disparity = compute_disparity(
        gray1,
        gray2,
        min_disparity,
        num_disparities,
        args.method,
        Q=Q,
        depth_min_m=args.depth_min,
        depth_max_m=args.depth_max,
    )
    coverage, disp_min, disp_max = disparity_coverage(disparity)
    print(f"Disparity coverage: {coverage:.1f}% (range {disp_min:.1f}..{disp_max:.1f} px)")

    np.save(out_dir / "disparity.npy", disparity)
    save_disparity_preview(out_dir / "disparity_preview.png", disparity)
    depth_metric = depth_map_from_disparity(disparity, Q)
    np.save(out_dir / "depth_metric_opencv.npy", depth_metric.astype(np.float32))

    if coverage < 5.0:
        print(
            "Warning: very low disparity coverage (StereoBM). "
            "Calibrate RGB2 (--camera rgb2), add light/texture, or use LiDAR for the floor plane."
        )

    valid_disp = disparity > 0

    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    raw_depth = points_3d[:, :, 2]
    raw_valid_depth = raw_depth[valid_disp & np.isfinite(raw_depth)]
    flip_sign = False
    if len(raw_valid_depth):
        print(
            "Raw reprojected depth before filtering: "
            f"median={np.median(raw_valid_depth):.3f}m, "
            f"min={np.min(raw_valid_depth):.3f}m, max={np.max(raw_valid_depth):.3f}m"
        )
        flip_sign = np.median(raw_valid_depth) < 0
        if flip_sign:
            print("Depth median is negative; flipping reprojected XYZ sign.")
            points_3d *= -1.0

    finite = np.isfinite(points_3d).all(axis=2)
    depth = points_3d[:, :, 2]
    mask = valid_disp & finite & (depth >= max(MIN_DEPTH_M, args.depth_min)) & (depth <= min(MAX_DEPTH_M, args.depth_max))

    points = points_3d[mask]
    colors = cv2.cvtColor(rect1, cv2.COLOR_BGR2RGB)[mask].astype(np.float64) / 255.0
    print(f"Valid disparity pixels: {int(np.count_nonzero(valid_disp))}")
    print(f"Stereo points after filtering: {len(points)}")
    if len(points) == 0:
        valid_depth = depth[valid_disp & finite]
        if len(valid_depth):
            print(
                "No points survived depth filtering. "
                f"Depth range among valid disparities was {np.min(valid_depth):.3f}..{np.max(valid_depth):.3f} m. "
                f"Current filter is {MIN_DEPTH_M:.1f}..{MAX_DEPTH_M:.1f} m."
            )
        raise RuntimeError("Stereo point cloud is empty; check disparity_preview.png and rectification_check.png.")

    save_point_cloud(out_dir / "stereo_pointcloud.ply", points, colors)
    down_points, down_colors = voxel_downsample(points, colors, voxel_size=0.02)
    save_point_cloud(out_dir / "stereo_pointcloud_downsampled.ply", down_points, down_colors)

    if paths.run_dir:
        write_run_info(
            paths.run_dir,
            disparity_method=args.method,
            disparity_coverage_pct=float(coverage),
            stereo_point_count=int(len(points)),
            stereo_point_count_downsampled=int(len(down_points)),
        )

    print(f"Saved {out_dir / 'rgb1_rectified.png'}")
    print(f"Saved {out_dir / 'rgb2_rectified.png'}")
    print(f"Saved {out_dir / 'rectification_check.png'}")
    print(f"Saved {out_dir / 'stereo_geometry.npz'}")
    print(f"Saved {out_dir / 'disparity.npy'}")
    print(f"Saved {out_dir / 'depth_metric_opencv.npy'}")
    print(f"Saved {out_dir / 'disparity_preview.png'}")
    print(f"Saved {out_dir / 'stereo_pointcloud.ply'}")
    print(f"Saved {out_dir / 'stereo_pointcloud_downsampled.ply'}")


if __name__ == "__main__":
    main()
