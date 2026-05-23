"""Tiny point-cloud utilities so the workflow does not depend on Open3D."""

from pathlib import Path

import numpy as np


def write_ply(path, points, colors=None):
    """Write an ASCII PLY file with optional RGB colors in 0..1 or 0..255."""
    path = Path(path)
    points = np.asarray(points, dtype=float)

    if colors is None:
        colors_u8 = np.full((len(points), 3), 255, dtype=np.uint8)
    else:
        colors = np.asarray(colors)
        if colors.size == 0:
            colors_u8 = np.full((len(points), 3), 255, dtype=np.uint8)
        elif np.nanmax(colors) <= 1.0:
            colors_u8 = np.clip(colors * 255, 0, 255).astype(np.uint8)
        else:
            colors_u8 = np.clip(colors, 0, 255).astype(np.uint8)

    with open(path, "w", newline="\n") as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        for point, color in zip(points, colors_u8):
            f.write(
                f"{point[0]:.8f} {point[1]:.8f} {point[2]:.8f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def read_ply(path):
    """Read ASCII PLY files written by write_ply."""
    path = Path(path)
    with open(path, "r") as f:
        vertex_count = None
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Invalid PLY header: {path}")
            line = line.strip()
            if line.startswith("element vertex"):
                vertex_count = int(line.split()[-1])
            if line == "end_header":
                break

        if vertex_count is None:
            raise ValueError(f"PLY missing vertex count: {path}")
        if vertex_count == 0:
            return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=float)

        data = np.loadtxt(f, max_rows=vertex_count)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    points = data[:, :3].astype(float)
    if data.shape[1] >= 6:
        colors = data[:, 3:6].astype(float) / 255.0
    else:
        colors = np.ones((len(points), 3), dtype=float)
    return points, colors


def voxel_downsample(points, colors, voxel_size):
    """Simple voxel-grid downsample by averaging points/colors per voxel."""
    points = np.asarray(points, dtype=float)
    colors = np.asarray(colors, dtype=float)
    if len(points) == 0:
        return points, colors

    voxel_indices = np.floor(points / voxel_size).astype(np.int64)
    _, inverse = np.unique(voxel_indices, axis=0, return_inverse=True)
    counts = np.bincount(inverse)

    down_points = np.column_stack([
        np.bincount(inverse, weights=points[:, axis]) / counts for axis in range(3)
    ])
    down_colors = np.column_stack([
        np.bincount(inverse, weights=colors[:, axis]) / counts for axis in range(3)
    ])
    return down_points, down_colors


def nearest_neighbor_distances(query_points, reference_points, chunk_size=5000):
    """Nearest-reference distance for every query point.

    Uses scipy.spatial.cKDTree when available. Falls back to chunked brute force.
    """
    query_points = np.asarray(query_points, dtype=float)
    reference_points = np.asarray(reference_points, dtype=float)
    if len(query_points) == 0:
        return np.empty(0, dtype=float), np.empty(0, dtype=int)
    if len(reference_points) == 0:
        raise ValueError("Cannot compute nearest neighbors: reference point cloud is empty.")

    try:
        from scipy.spatial import cKDTree

        distances, indices = cKDTree(reference_points).query(query_points, k=1)
        return distances.astype(float), indices.astype(int)
    except Exception:
        distances = np.empty(len(query_points), dtype=float)
        indices = np.empty(len(query_points), dtype=int)
        for start in range(0, len(query_points), chunk_size):
            stop = min(start + chunk_size, len(query_points))
            delta = query_points[start:stop, None, :] - reference_points[None, :, :]
            dist_sq = np.sum(delta * delta, axis=2)
            nearest = np.argmin(dist_sq, axis=1)
            indices[start:stop] = nearest
            distances[start:stop] = np.sqrt(dist_sq[np.arange(stop - start), nearest])
        return distances, indices
