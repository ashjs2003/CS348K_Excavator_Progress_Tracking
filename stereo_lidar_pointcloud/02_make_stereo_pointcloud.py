"""
Build a colored stereo point cloud from capture/rgb1.png and capture/rgb2.png.

Run:
    python 02_make_stereo_pointcloud.py
"""

from pathlib import Path

import cv2
import numpy as np

from calib_utils import load_camera_calibration, load_stereo_rgb1_to_rgb2
from pointcloud_utils import voxel_downsample, write_ply


REPO_ROOT = Path(__file__).resolve().parents[1]
CAPTURE_DIR = REPO_ROOT / "capture"
OUT_DIR = REPO_ROOT / "outputs"
RGB1_IMAGE = CAPTURE_DIR / "rgb1.png"
RGB2_IMAGE = CAPTURE_DIR / "rgb2.png"
MIN_DEPTH_M = 0.1
MAX_DEPTH_M = 5.0


def draw_rectification_check(rect1, rect2, line_step=40):
    """Stack rectified images side-by-side and draw horizontal guide lines."""
    combined = np.hstack([rect1, rect2])
    for y in range(0, combined.shape[0], line_step):
        cv2.line(combined, (0, y), (combined.shape[1] - 1, y), (0, 255, 255), 1)
    return combined


def make_sgbm(width):
    """Create a simple SGBM matcher. numDisparities must be divisible by 16."""
    num_disparities = max(16 * 6, ((width // 8) // 16) * 16)
    block_size = 5
    return cv2.StereoSGBM_create(
        minDisparity=0,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * block_size**2,
        P2=32 * 3 * block_size**2,
        disp12MaxDiff=1,
        uniquenessRatio=8,
        speckleWindowSize=80,
        speckleRange=2,
        preFilterCap=63,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
    )


def save_point_cloud(path, points, colors):
    write_ply(path, points, colors)


def main():
    OUT_DIR.mkdir(exist_ok=True)
    rgb1_calib = load_camera_calibration("camera_calibration_rgb1.npz")
    rgb2_calib = load_camera_calibration("camera_calibration_rgb2.npz", "camera_calibration_rgb2_approx.npz")
    stereo_path, R, t = load_stereo_rgb1_to_rgb2()

    image1 = cv2.imread(str(RGB1_IMAGE))
    image2 = cv2.imread(str(RGB2_IMAGE))
    if image1 is None:
        raise RuntimeError(f"Could not load {RGB1_IMAGE}")
    if image2 is None:
        raise RuntimeError(f"Could not load {RGB2_IMAGE}")

    image_size = (image1.shape[1], image1.shape[0])
    print(f"Loaded stereo extrinsics: {stereo_path}")
    print(f"Image size: {image_size[0]}x{image_size[1]}")

    # Rectify both cameras into a common stereo geometry. Q converts disparity
    # back into 3D points in the rectified RGB1 camera frame.
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
    rect1 = cv2.remap(image1, map1x, map1y, cv2.INTER_LINEAR)
    rect2 = cv2.remap(image2, map2x, map2y, cv2.INTER_LINEAR)

    cv2.imwrite(str(OUT_DIR / "rgb1_rectified.png"), rect1)
    cv2.imwrite(str(OUT_DIR / "rgb2_rectified.png"), rect2)
    cv2.imwrite(str(OUT_DIR / "rectification_check.png"), draw_rectification_check(rect1, rect2))

    gray1 = cv2.cvtColor(rect1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(rect2, cv2.COLOR_BGR2GRAY)
    matcher = make_sgbm(image_size[0])
    disparity = matcher.compute(gray1, gray2).astype(np.float32) / 16.0
    np.save(OUT_DIR / "disparity.npy", disparity)

    valid_disp = disparity > 0
    preview = np.zeros_like(disparity, dtype=np.uint8)
    if np.any(valid_disp):
        valid_values = disparity[valid_disp]
        normalized = cv2.normalize(valid_values, None, 0, 255, cv2.NORM_MINMAX)
        preview[valid_disp] = normalized.reshape(-1).astype(np.uint8)
    preview_color = cv2.applyColorMap(preview, cv2.COLORMAP_TURBO)
    preview_color[~valid_disp] = (0, 0, 0)
    cv2.imwrite(str(OUT_DIR / "disparity_preview.png"), preview_color)

    points_3d = cv2.reprojectImageTo3D(disparity, Q)
    raw_depth = points_3d[:, :, 2]
    raw_valid_depth = raw_depth[valid_disp & np.isfinite(raw_depth)]
    if len(raw_valid_depth):
        print(
            "Raw reprojected depth before filtering: "
            f"median={np.median(raw_valid_depth):.3f}m, "
            f"min={np.min(raw_valid_depth):.3f}m, max={np.max(raw_valid_depth):.3f}m"
        )
        # Depending on stereo baseline sign, OpenCV's Q can produce the same
        # geometry with negative Z. Flip the cloud so depth is positive.
        if np.median(raw_valid_depth) < 0:
            print("Depth median is negative; flipping reprojected XYZ sign.")
            points_3d *= -1.0

    finite = np.isfinite(points_3d).all(axis=2)
    depth = points_3d[:, :, 2]
    mask = valid_disp & finite & (depth >= MIN_DEPTH_M) & (depth <= MAX_DEPTH_M)

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

    save_point_cloud(OUT_DIR / "stereo_pointcloud.ply", points, colors)
    down_points, down_colors = voxel_downsample(points, colors, voxel_size=0.02)
    save_point_cloud(OUT_DIR / "stereo_pointcloud_downsampled.ply", down_points, down_colors)

    print(f"Saved {OUT_DIR / 'rgb1_rectified.png'}")
    print(f"Saved {OUT_DIR / 'rgb2_rectified.png'}")
    print(f"Saved {OUT_DIR / 'rectification_check.png'}")
    print(f"Saved {OUT_DIR / 'disparity.npy'}")
    print(f"Saved {OUT_DIR / 'disparity_preview.png'}")
    print(f"Saved {OUT_DIR / 'stereo_pointcloud.ply'}")
    print(f"Saved {OUT_DIR / 'stereo_pointcloud_downsampled.ply'}")


if __name__ == "__main__":
    main()
