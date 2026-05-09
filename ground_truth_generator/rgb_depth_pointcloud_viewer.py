import argparse
from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path):
    image = Image.open(path).convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def load_depth(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        depth = np.load(path)
    else:
        image = Image.open(path)
        depth = np.asarray(image, dtype=np.float32)
        if depth.dtype != np.float32:
            depth = depth.astype(np.float32)
        if depth.max() > 1000:
            depth /= 1000.0
    return np.squeeze(depth).astype(np.float32)


def make_point_cloud(rgb, depth, fx, fy, cx=None, cy=None, stride=1, max_depth=None):
    height, width = depth.shape
    if rgb.shape[:2] != depth.shape:
        rgb_image = Image.fromarray((rgb * 255).astype(np.uint8))
        rgb_image = rgb_image.resize((width, height), Image.Resampling.BILINEAR)
        rgb = np.asarray(rgb_image, dtype=np.float32) / 255.0

    if cx is None:
        cx = (width - 1) / 2.0
    if cy is None:
        cy = (height - 1) / 2.0

    v, u = np.mgrid[0:height:stride, 0:width:stride]
    z = depth[0:height:stride, 0:width:stride]
    colors = rgb[0:height:stride, 0:width:stride]

    valid = np.isfinite(z) & (z > 0)
    if max_depth is not None:
        valid &= z <= max_depth

    u = u[valid].astype(np.float32)
    v = v[valid].astype(np.float32)
    z = z[valid].astype(np.float32)
    colors = colors[valid].reshape(-1, 3)

    x = (u - cx) * z / fx
    y = -(v - cy) * z / fy
    points = np.column_stack((x, y, z))
    return points, colors


def save_ply(path, points, colors):
    colors_u8 = np.clip(colors * 255, 0, 255).astype(np.uint8)
    with open(path, "w", encoding="ascii") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {len(points)}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")
        for point, color in zip(points, colors_u8):
            file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{color[0]} {color[1]} {color[2]}\n"
            )


def view_with_open3d(points, colors):
    import open3d as o3d

    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)

    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.5)
    o3d.visualization.draw_geometries(
        [cloud, frame],
        window_name="RGB-D Point Cloud Viewer",
        width=1280,
        height=800,
    )


def view_with_matplotlib(points, colors, sample_count=120_000):
    import matplotlib.pyplot as plt

    if len(points) > sample_count:
        rng = np.random.default_rng(0)
        indices = rng.choice(len(points), size=sample_count, replace=False)
        points = points[indices]
        colors = colors[indices]

    figure = plt.figure("RGB-D Point Cloud Viewer")
    axis = figure.add_subplot(111, projection="3d")
    axis.scatter(points[:, 0], points[:, 2], points[:, 1], c=colors, s=0.2)
    axis.set_xlabel("X")
    axis.set_ylabel("Z")
    axis.set_zlabel("Y")
    axis.set_box_aspect((np.ptp(points[:, 0]), np.ptp(points[:, 2]), np.ptp(points[:, 1])))
    plt.show()


def main():
    parser = argparse.ArgumentParser(
        description="Create and view a colored point cloud from an RGB image and depth map."
    )
    parser.add_argument("--rgb", default="terrain_rgb.png", help="Path to the RGB image.")
    parser.add_argument(
        "--depth",
        default="terrain_depth_meters.npy",
        help="Path to a depth map in meters as .npy, or a 16-bit depth PNG in millimeters.",
    )
    parser.add_argument("--out", default="terrain_pointcloud.ply", help="Output PLY path.")
    parser.add_argument("--fx", type=float, default=None, help="Camera focal length in pixels.")
    parser.add_argument("--fy", type=float, default=None, help="Camera focal length in pixels.")
    parser.add_argument("--cx", type=float, default=None, help="Camera principal point x.")
    parser.add_argument("--cy", type=float, default=None, help="Camera principal point y.")
    parser.add_argument("--stride", type=int, default=2, help="Use every Nth pixel.")
    parser.add_argument("--max-depth", type=float, default=None, help="Drop points beyond this depth.")
    parser.add_argument("--no-view", action="store_true", help="Only save the PLY file.")
    args = parser.parse_args()

    rgb = load_rgb(args.rgb)
    depth = load_depth(args.depth)

    height, width = depth.shape
    fx = args.fx if args.fx is not None else width
    fy = args.fy if args.fy is not None else width

    points, colors = make_point_cloud(
        rgb=rgb,
        depth=depth,
        fx=fx,
        fy=fy,
        cx=args.cx,
        cy=args.cy,
        stride=max(1, args.stride),
        max_depth=args.max_depth,
    )

    save_ply(args.out, points, colors)
    print(f"Saved {len(points):,} points to {args.out}")

    if args.no_view:
        return

    try:
        view_with_open3d(points, colors)
    except ImportError:
        print("Open3D is not installed, falling back to matplotlib viewer.")
        view_with_matplotlib(points, colors)


if __name__ == "__main__":
    main()
