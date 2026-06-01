"""
Open a lightweight Matplotlib viewer for the stereo point cloud and/or LiDAR points.

Run:
    python 04_view_pointcloud.py
    python 04_view_pointcloud.py --mode stereo
    python 04_view_pointcloud.py --mode lidar
    python 04_view_pointcloud.py --mode both
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from pointcloud_utils import read_ply

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from output_runs import add_run_cli_arguments, handle_list_runs, resolve_run_paths


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["stereo", "lidar", "both"], default="stereo")
    parser.add_argument(
        "--stereo-backend",
        choices=["opencv", "foundation", "dav2", "compare", "compare-all"],
        default="opencv",
        help="opencv | foundation | dav2 | compare (opencv+FS) | compare-all (all three)",
    )
    add_run_cli_arguments(parser)
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


def plot_stereo_on_ax(ax, stereo_cloud, args, label):
    stereo_points, stereo_colors = read_ply(stereo_cloud)
    if len(stereo_points) == 0:
        raise RuntimeError(f"No points loaded from {stereo_cloud}")
    stereo_points, stereo_colors = sample_points(stereo_points, stereo_colors, args.max_points)
    stereo_colors = enhance_colors(stereo_colors, args.brightness, args.gamma)
    ax.scatter(
        stereo_points[:, 0],
        stereo_points[:, 1],
        stereo_points[:, 2],
        c=stereo_colors,
        s=args.point_size,
        depthshade=args.depth_shade,
        label=label,
    )
    return stereo_points


def add_axes_gizmo(ax, axis_len=0.25):
    ax.quiver(0, 0, 0, axis_len, 0, 0, color="r", linewidth=2)
    ax.quiver(0, 0, 0, 0, axis_len, 0, color="g", linewidth=2)
    ax.quiver(0, 0, 0, 0, 0, axis_len, color="b", linewidth=2)
    ax.text(axis_len, 0, 0, "X", color="r")
    ax.text(0, axis_len, 0, "Y", color="g")
    ax.text(0, 0, axis_len, "Z", color="b")


def main():
    args = parse_args()
    if handle_list_runs(args):
        return
    paths = resolve_run_paths(args.run)
    lidar_cloud = paths.validation / "lidar_points_in_rgb1_frame.ply"
    if paths.run_dir:
        print(f"Run: {paths.run_dir.name}")

    if args.stereo_backend == "compare-all":
        fig = plt.figure(figsize=(20, 6))
        axes = [fig.add_subplot(131, projection="3d"), fig.add_subplot(132, projection="3d"), fig.add_subplot(133, projection="3d")]
        backends = [("", "OpenCV"), ("_foundation", "FoundationStereo"), ("_dav2", "Depth Anything V2")]
    elif args.stereo_backend == "compare":
        fig = plt.figure(figsize=(16, 7))
        axes = [fig.add_subplot(121, projection="3d"), fig.add_subplot(122, projection="3d")]
        backends = [("", "OpenCV"), ("_foundation", "FoundationStereo")]
    else:
        suffix = {"opencv": "", "foundation": "_foundation", "dav2": "_dav2"}[args.stereo_backend]
        fig = plt.figure(figsize=(11, 8))
        axes = [fig.add_subplot(111, projection="3d")]
        backends = [(suffix, args.stereo_backend)]

    for ax, (suffix, name) in zip(axes, backends):
        panel_points = []
        if args.mode in ("stereo", "both"):
            stereo_cloud = paths.stereo / f"stereo_pointcloud_downsampled{suffix}.ply"
            pts = plot_stereo_on_ax(ax, stereo_cloud, args, label=f"stereo ({name})")
            panel_points.append(pts)

        if args.mode in ("lidar", "both") and len(axes) == 1:
            if not lidar_cloud.is_file():
                if args.mode == "lidar":
                    raise FileNotFoundError(
                        f"{lidar_cloud} missing. Run: python 03_validate_with_lidar.py --run {args.run}"
                    )
                print(f"Skipping LiDAR overlay because {lidar_cloud} is missing.")
            else:
                lidar_points, _ = read_ply(lidar_cloud)
                ax.scatter(
                    lidar_points[:, 0],
                    lidar_points[:, 1],
                    lidar_points[:, 2],
                    c="red",
                    s=12,
                    label="lidar",
                )
                panel_points.append(lidar_points)

        add_axes_gizmo(ax)
        if panel_points:
            set_equal_axes(ax, np.vstack(panel_points))
        ax.set_xlabel("X m")
        ax.set_ylabel("Y m")
        ax.set_zlabel("Z m")
        ax.legend()
        ax.set_title(f"{name} — {args.mode}")

    if args.mode in ("lidar", "both") and len(axes) > 1:
        print("Note: compare mode shows stereo only; run --stereo-backend opencv --mode both for LiDAR overlay.")

    fig.suptitle(f"Run {paths.run_dir.name if paths.run_dir else 'legacy'}")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
