"""Per-method folders under run depth/ (shared rectification + opencv | dav2 | dav2_gt | foundation)."""

from __future__ import annotations

import shutil
from pathlib import Path

SHARED_DIR = "shared"
METHOD_IDS = ("opencv", "dav2", "dav2_gt", "foundation")

# Legacy flat filenames (pre method-subfolder layout)
LEGACY_SHARED = (
    "rgb1_rectified.png",
    "rgb2_rectified.png",
    "rectification_check.png",
    "stereo_geometry.npz",
)
LEGACY_BY_METHOD: dict[str, dict[str, str]] = {
    "opencv": {
        "disparity.npy": "disparity.npy",
        "disparity_preview.png": "disparity_preview.png",
        "depth_metric.npy": "depth_metric_opencv.npy",
        "pointcloud.ply": "stereo_pointcloud.ply",
        "pointcloud_downsampled.ply": "stereo_pointcloud_downsampled.ply",
    },
    "dav2": {
        "depth_metric.npy": "depth_metric_dav2.npy",
        "depth_preview.png": "depth_preview_dav2.png",
        "scaling.json": "depth_scaling_dav2.json",
        "pointcloud.ply": "stereo_pointcloud_dav2.ply",
        "pointcloud_downsampled.ply": "stereo_pointcloud_downsampled_dav2.ply",
    },
    "dav2_gt": {
        "depth_metric.npy": "depth_metric_dav2_gt.npy",
        "depth_preview.png": "depth_preview_dav2_gt.png",
        "scaling.json": "depth_scaling_dav2_gt.json",
        "pointcloud.ply": "stereo_pointcloud_dav2_gt.ply",
        "pointcloud_downsampled.ply": "stereo_pointcloud_downsampled_dav2_gt.ply",
    },
    "foundation": {
        "vis.png": "vis.png",
        "disparity.npy": "disparity_foundation.npy",
        "disparity_preview.png": "disparity_preview_foundation.png",
        "pointcloud.ply": "stereo_pointcloud_foundation.ply",
        "pointcloud_downsampled.ply": "stereo_pointcloud_downsampled_foundation.ply",
    },
}


def shared_dir(depth_root: Path) -> Path:
    return Path(depth_root) / SHARED_DIR


def method_dir(depth_root: Path, method: str) -> Path:
    if method not in METHOD_IDS:
        raise ValueError(f"Unknown method {method!r}")
    return Path(depth_root) / method


def resolve_path(depth_root: Path, method: str | None, canonical_name: str) -> Path | None:
    """
    Resolve a product path (new layout first, then legacy flat names).

    method=None uses shared/ (or depth root for legacy).
    """
    depth_root = Path(depth_root)
    if method is None:
        p = shared_dir(depth_root) / canonical_name
        if p.is_file():
            return p
        leg = depth_root / canonical_name
        return leg if leg.is_file() else None

    mdir = method_dir(depth_root, method)
    p = mdir / canonical_name
    if p.is_file():
        return p
    legacy_name = LEGACY_BY_METHOD.get(method, {}).get(canonical_name)
    if legacy_name:
        leg = depth_root / legacy_name
        if leg.is_file():
            return leg
    if method == "foundation" and canonical_name == "vis.png":
        for alt in (shared_dir(depth_root) / "vis.png", depth_root / "vis.png"):
            if alt.is_file():
                return alt
    return None


def rgb1_rectified_path(depth_root: Path) -> Path:
    p = resolve_path(depth_root, None, "rgb1_rectified.png")
    if p is None:
        raise FileNotFoundError(f"Missing rgb1_rectified.png under {depth_root}")
    return p


def stereo_geometry_path(depth_root: Path) -> Path:
    p = resolve_path(depth_root, None, "stereo_geometry.npz")
    if p is None:
        return shared_dir(depth_root) / "stereo_geometry.npz"
    return p


def ensure_method_tree(depth_root: Path) -> None:
    shared_dir(depth_root).mkdir(parents=True, exist_ok=True)
    for mid in METHOD_IDS:
        method_dir(depth_root, mid).mkdir(parents=True, exist_ok=True)


def migrate_depth_folder(depth_root: Path, dry_run: bool = False) -> list[str]:
    """Move flat depth/ files into shared/ and method subfolders. Returns log lines."""
    depth_root = Path(depth_root)
    if not depth_root.is_dir():
        return []
    logs: list[str] = []
    ensure_method_tree(depth_root)

    for name in LEGACY_SHARED:
        src = depth_root / name
        if src.is_file():
            dest = shared_dir(depth_root) / name
            logs.append(_move_line(src, dest, dry_run))

    fdir = method_dir(depth_root, "foundation")
    for src in (depth_root / "vis.png", shared_dir(depth_root) / "vis.png"):
        if src.is_file():
            logs.append(_move_line(src, fdir / "vis.png", dry_run))

    if (depth_root / "depth_anything_v2_work").is_dir():
        dest_work = method_dir(depth_root, "dav2") / "work"
        logs.append(f"{'would move' if dry_run else 'moved'} work/ -> dav2/work/")
        if not dry_run:
            if dest_work.exists():
                shutil.rmtree(dest_work)
            shutil.move(str(depth_root / "depth_anything_v2_work"), str(dest_work))

    for method, mapping in LEGACY_BY_METHOD.items():
        mdir = method_dir(depth_root, method)
        for canonical, legacy in mapping.items():
            src = depth_root / legacy
            if src.is_file():
                dest = mdir / canonical
                logs.append(_move_line(src, dest, dry_run))

    return logs


def _move_line(src: Path, dest: Path, dry_run: bool) -> str:
    rel = dest.relative_to(src.parent.parent) if dest.parent.parent in src.parents else dest
    msg = f"{'would move' if dry_run else 'moved'} {src.name} -> {rel}"
    if not dry_run:
        dest.parent.mkdir(parents=True, exist_ok=True)
        if dest.is_file():
            src.unlink()
        else:
            shutil.move(str(src), str(dest))
    return msg
