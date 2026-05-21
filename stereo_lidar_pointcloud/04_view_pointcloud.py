"""
Open a lightweight Matplotlib viewer for the stereo point cloud and/or LiDAR points.

Run:
    python 04_view_pointcloud.py
    python 04_view_pointcloud.py --mode stereo
    python 04_view_pointcloud.py --mode lidar
    python 04_view_pointcloud.py --mode both
"""

from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np

from pointcloud_utils import read_ply


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "outputs"
STEREO_CLOUD = OUT_DIR / "stereo_pointcloud_downsampled.ply"
LIDAR_CLOUD = OUT_DIR / "lidar_points_in_rgb1_frame.ply"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stereo", "lidar", "both"], default="both")
    parser.add_argument("--max-points", type=int, default=80000, help="Maximum stereo points to draw")
    parser.add_argument("--point-size", type=float, default=2.0, help="Stereo point size")
    parser.add_argument("--brightness", type=float, default=2.0, help="Brightness multiplier for stereo RGB")
    parser.add_argument("--gamma", type=float, default=0.6, help="Gamma correction for stereo RGB")
    parser.add_argument("--depth-shade", action="store_true", help="Let Matplotlib darken points by depth")
    return parser.parse_args()


def sample_points(points, colors, max_points):
    if len(points) <= max_points:
        return points, colors
    indices = np.linspace(0, len(points) - 1, max_points).astype(int)
    return points[indices], colors[indices]


def set_equal_axes(ax, points):
    if len(points) == 0:
        return
    center = np.mean(points, axis=0)
    radius = max(0.1, float(np.max(np.ptp(points, axis=0))) / 2.0)
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)


def enhance_colors(colors, brightness, gamma):
    colors = np.asarray(colors, dtype=float)
    if len(colors) == 0:
        return colors
    colors = np.clip(colors * brightness, 0.0, 1.0)
    if gamma > 0:
        colors = np.power(colors, gamma)
    return np.clip(colors, 0.0, 1.0)


def print_color_stats(label, colors):
    if len(colors) == 0:
        print(f"{label} color stats: no colors")
        return
    print(
        f"{label} color stats: "
        f"min={np.min(colors, axis=0)}, "
        f"median={np.median(colors, axis=0)}, "
        f"max={np.max(colors, axis=0)}"
    )


def main():
    args = parse_args()
    all_points = []
    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    if args.mode in ("stereo", "both"):
        stereo_points, stereo_colors = read_ply(STEREO_CLOUD)
        if len(stereo_points) == 0:
            raise RuntimeError(f"No points loaded from {STEREO_CLOUD}")
        stereo_points, stereo_colors = sample_points(stereo_points, stereo_colors, args.max_points)
        print_color_stats("Stereo raw", stereo_colors)
        stereo_colors = enhance_colors(stereo_colors, args.brightness, args.gamma)
        print_color_stats("Stereo displayed", stereo_colors)
        ax.scatter(
            stereo_points[:, 0],
            stereo_points[:, 1],
            stereo_points[:, 2],
            c=stereo_colors,
            s=args.point_size,
            depthshade=args.depth_shade,
            label="stereo",
        )
        all_points.append(stereo_points)

    if args.mode in ("lidar", "both"):
        lidar_points, _ = read_ply(LIDAR_CLOUD)
        if len(lidar_points) == 0:
            raise RuntimeError(f"No points loaded from {LIDAR_CLOUD}")
        ax.scatter(
            lidar_points[:, 0],
            lidar_points[:, 1],
            lidar_points[:, 2],
            c="red",
            s=12,
            label="lidar",
        )
        all_points.append(lidar_points)

    # Coordinate frame is in the rectified/RGB1 camera coordinate system.
    axis_len = 0.25
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="r", linewidth=2)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="g", linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="b", linewidth=2)
    ax.text(axis_len, 0, 0, "X", color="r")
    ax.text(0, axis_len, 0, "Y", color="g")
    ax.text(0, 0, axis_len, "Z", color="b")

    if all_points:
        set_equal_axes(ax, np.vstack(all_points))
    ax.set_xlabel("X m")
    ax.set_ylabel("Y m")
    ax.set_zlabel("Z m")
    ax.legend()
    ax.set_title(f"Stereo/LiDAR point cloud ({args.mode})")
    plt.show()


if __name__ == "__main__":
    main()
