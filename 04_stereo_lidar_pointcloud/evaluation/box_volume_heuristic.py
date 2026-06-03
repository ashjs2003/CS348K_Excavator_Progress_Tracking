"""
Cardboard box volume heuristic (S / M / L).

- Depth (into box): median Z inside ROI − median Z on consistent outside flap ring.
- Height: fixed manual size (S=7, M=19, L=24 cm).
- Width: catalog face width × cos(placement angle); optional LiDAR width via
  range-profile edge detection at ruler distance + placement bearing (no catalog width).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from evaluation.gt_depth_overlay import parse_pair_distance_txt

# Nominal face width (cm) along the horizontal edge visible at 0°; height is user-measured vertical edge.
BOX_SPECS: dict[str, dict[str, float]] = {
    "S": {"height_cm": 7.0, "width_nominal_cm": 7.0, "gt_volume_cm3": 294.0},
    "M": {"height_cm": 19.0, "width_nominal_cm": 26.0, "gt_volume_cm3": 4940.0},
    "L": {"height_cm": 24.0, "width_nominal_cm": 40.0, "gt_volume_cm3": 15360.0},
}

# ROI−flap depth: flaps must be clearly in front of ROI; tight band on flap Z (tuned for S/M/L cardboard).
DEFAULT_OUTSIDE_CLOSER_MARGIN_M = 0.01
DEFAULT_OUTSIDE_CONSISTENCY_M = 0.005


def gt_nominal_depth_cm(size: str) -> float:
    """Catalog depth edge (cm) from nominal volume / (width × height)."""
    spec = BOX_SPECS[size]
    denom = spec["width_nominal_cm"] * spec["height_cm"]
    return float(spec["gt_volume_cm3"] / denom)


def box_size_class(scene: str) -> str | None:
    if scene.startswith("S_"):
        return "S"
    if scene.startswith("M_"):
        return "M"
    if scene.startswith("L_"):
        return "L"
    return None


def placement_angle_deg(scene: str) -> float:
    """Cardboard/checkerboard *_30 scenes → −30° in LiDAR bearing; else 0°."""
    name = scene.lower()
    if name.endswith("_30") or "box_30" in name or "box30" in name:
        return -30.0
    return 0.0


def angular_diff_deg(angle_deg: np.ndarray, center_deg: float) -> np.ndarray:
    """Smallest signed difference to center_deg, in [-180, 180]."""
    return (angle_deg - float(center_deg) + 180.0) % 360.0 - 180.0


def mask_lidar_at_setup(
    angle_deg: np.ndarray,
    dist_m: np.ndarray,
    *,
    ruler_m: float,
    placement_angle_deg: float,
    distance_tol_m: float,
    angular_tol_deg: float,
    quality: np.ndarray | None = None,
    min_quality: float = 0.0,
) -> np.ndarray:
    """True where return is at ruler range AND near nominal placement bearing."""
    valid = np.isfinite(angle_deg) & np.isfinite(dist_m) & (dist_m > 0)
    if quality is not None:
        valid &= np.isfinite(quality) & (quality >= min_quality)
    near_r = np.abs(dist_m - ruler_m) <= distance_tol_m
    near_a = np.abs(angular_diff_deg(angle_deg, placement_angle_deg)) <= float(angular_tol_deg)
    return valid & near_r & near_a


def circular_span_deg(angle_deg: np.ndarray) -> float:
    """Angular extent handling 0°/360° wrap (relative to median bearing)."""
    if len(angle_deg) < 2:
        return 0.0
    center = np.deg2rad(float(np.median(angle_deg)))
    rel = np.arctan2(
        np.sin(np.deg2rad(angle_deg) - center),
        np.cos(np.deg2rad(angle_deg) - center),
    )
    return float(np.rad2deg(np.max(rel) - np.min(rel)))


def robust_circular_span_deg(
    angle_deg: np.ndarray,
    *,
    q_low: float = 10.0,
    q_high: float = 90.0,
) -> float:
    """Percentile angular span (robust to outliers)."""
    if len(angle_deg) < 2:
        return 0.0
    center = np.deg2rad(float(np.median(angle_deg)))
    rel = np.rad2deg(
        np.arctan2(
            np.sin(np.deg2rad(angle_deg) - center),
            np.cos(np.deg2rad(angle_deg) - center),
        )
    )
    lo, hi = np.percentile(rel, [q_low, q_high])
    return float(hi - lo)


def _largest_cluster_local_indices(
    points_xy: np.ndarray,
    *,
    gap_deg: float = 4.0,
    min_pts: int = 3,
) -> np.ndarray | None:
    """Indices into ``points_xy`` for the largest contiguous angular cluster."""
    n = len(points_xy)
    if n < 2:
        return np.arange(n) if n else None
    if n < min_pts:
        return np.arange(n)

    angles = np.unwrap(np.arctan2(points_xy[:, 1], points_xy[:, 0]))
    order = np.argsort(angles)
    angles = angles[order]
    splits = np.where(np.diff(angles) > np.deg2rad(gap_deg))[0] + 1
    slices = np.split(order, splits)

    best: np.ndarray | None = None
    best_len = 0
    for sl in slices:
        if len(sl) < min_pts:
            continue
        if len(sl) > best_len:
            best_len = len(sl)
            best = sl
    if best is None:
        return order
    return best


def select_lidar_face_cluster_mask(
    angle_deg: np.ndarray,
    dist_m: np.ndarray,
    *,
    ruler_m: float,
    placement_angle_deg: float,
    distance_tol_m: float = 0.03,
    angular_tol_deg: float = 15.0,
    bearing_mode: str = "setup",
    coplanar_tol_m: float = 0.02,
    quality: np.ndarray | None = None,
    min_quality: float = 0.0,
    min_points: int = 3,
) -> tuple[np.ndarray, np.ndarray, str] | None:
    """
    Setup gate (ruler + placement angle) then largest angular cluster + coplanar filter.

    Returns (cluster_mask, gate_mask, gate_mode_used) over the full scan, or None.
    """
    gate_modes = ["placement", "auto"] if (bearing_mode or "setup").lower() == "setup" else [bearing_mode]
    gate = None
    gate_mode_used = bearing_mode
    for gm in gate_modes:
        g = mask_lidar_span_at_setup(
            angle_deg,
            dist_m,
            ruler_m=ruler_m,
            placement_angle_deg=placement_angle_deg,
            distance_tol_m=distance_tol_m,
            angular_tol_deg=angular_tol_deg,
            bearing_mode=gm,
            quality=quality,
            min_quality=min_quality,
        )
        if int(np.count_nonzero(g)) >= min_points:
            gate = g
            gate_mode_used = gm
            break
    if gate is None or int(np.count_nonzero(gate)) < min_points:
        return None

    idx = np.where(gate)[0]
    ang_g = angle_deg[idx]
    d_g = dist_m[idx]
    from evaluation.lidar_segment_width import lidar_xy_from_polar

    xy = lidar_xy_from_polar(ang_g, d_g)
    local = _largest_cluster_local_indices(xy, min_pts=3)
    if local is None or len(local) < 2:
        return None

    cluster = np.zeros(len(angle_deg), dtype=bool)
    cluster[idx[local]] = True

    d_c = dist_m[cluster]
    r_med = float(np.median(d_c))
    if coplanar_tol_m > 0:
        coplanar = cluster & (np.abs(dist_m - r_med) <= coplanar_tol_m)
        if int(np.count_nonzero(coplanar)) >= min_points:
            cluster = coplanar
        elif int(np.count_nonzero(coplanar)) >= 2:
            cluster = coplanar

    if int(np.count_nonzero(cluster)) < min_points:
        return None
    return cluster, gate, gate_mode_used


def load_lidar_scan(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (angle_deg, distance_m) arrays from pair_*_lidar.csv."""
    path = Path(path)
    angles, dists = [], []
    with open(path, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            try:
                angles.append(float(row["angle_degrees"]))
                dists.append(float(row["distance_meters"]))
            except (KeyError, ValueError):
                continue
    if not angles:
        return np.array([]), np.array([])
    return np.asarray(angles, dtype=float), np.asarray(dists, dtype=float)


def mask_lidar_span_at_setup(
    angle_deg: np.ndarray,
    dist_m: np.ndarray,
    *,
    ruler_m: float,
    placement_angle_deg: float | None = None,
    distance_tol_m: float = 0.03,
    angular_tol_deg: float = 15.0,
    bearing_mode: str = "auto",
    quality: np.ndarray | None = None,
    min_quality: float = 0.0,
) -> np.ndarray:
    """
    Select LiDAR returns for span-based width (method 2).

    bearing_mode:
      - ``setup`` (default): placement ± tol, else auto median bearing at ruler range
      - ``placement``: scene placement angle ± angular_tol only
      - ``auto``: median bearing at ruler distance ± angular_tol
      - ``distance_only``: ruler distance only (no bearing gate)
    """
    valid = np.isfinite(angle_deg) & np.isfinite(dist_m) & (dist_m > 0)
    if quality is not None:
        valid &= np.isfinite(quality) & (quality >= min_quality)
    near_r = np.abs(dist_m - ruler_m) <= distance_tol_m
    mode = (bearing_mode or "setup").lower()

    def _auto_gate() -> np.ndarray:
        from evaluation.lidar_segment_width import estimate_bearing_center_deg

        bearing = estimate_bearing_center_deg(
            angle_deg,
            dist_m,
            ruler_m,
            distance_window_m=max(distance_tol_m, 0.06),
            placement_angle_deg=placement_angle_deg,
        )
        near_a = np.abs(angular_diff_deg(angle_deg, bearing)) <= float(angular_tol_deg)
        return valid & near_r & near_a

    if mode == "distance_only":
        return valid & near_r
    if mode == "placement" and placement_angle_deg is not None:
        near_a = np.abs(angular_diff_deg(angle_deg, placement_angle_deg)) <= float(angular_tol_deg)
        return valid & near_r & near_a
    if mode == "auto":
        return _auto_gate()
    if mode == "setup" and placement_angle_deg is not None:
        near_a = np.abs(angular_diff_deg(angle_deg, placement_angle_deg)) <= float(angular_tol_deg)
        placement_gate = valid & near_r & near_a
        if int(np.count_nonzero(placement_gate)) >= 3:
            return placement_gate
        return _auto_gate()
    return _auto_gate()


def width_from_lidar_span(
    lidar_csv: Path,
    ruler_m: float,
    *,
    placement_angle_deg: float | None = None,
    distance_tol_m: float = 0.05,
    angular_tol_deg: float = 20.0,
    bearing_mode: str = "setup",
    coplanar_tol_m: float = 0.02,
    min_points: int = 3,
) -> dict | None:
    """
    LiDAR width via range-profile edge detection (see ``lidar_edge_profile_width``).

    No catalog box width. Uses ``pair_*_lidar.csv`` only.
    """
    if placement_angle_deg is None or ruler_m is None or ruler_m <= 0:
        return None
    from evaluation.lidar_edge_profile_width import width_from_lidar_edge_profile

    out = width_from_lidar_edge_profile(
        lidar_csv,
        ruler_m,
        placement_angle_deg,
        scene=scene,
        distance_tol_m=distance_tol_m,
        angular_tol_deg=angular_tol_deg,
        bearing_mode=bearing_mode,
        plateau_tol_m=max(coplanar_tol_m, 0.02),
        min_gate_points=max(min_points, 5),
        min_face_points=max(min_points, 4),
    )
    if out is None:
        return None
    # Back-compat keys for callers
    out["cluster_mask"] = out.get("face_mask")
    return out


def width_from_setup(
    scene: str,
    *,
    ruler_m: float | None = None,
    z_inside_m: float | None = None,
    use_lidar_span: bool = False,
    lidar_csv: Path | None = None,
    distance_tol_m: float = 0.03,
    angular_tol_deg: float = 15.0,
    ruler_scale_clamp: tuple[float, float] = (0.6, 1.6),
) -> dict:
    """
    Width (m) for volume: nominal face width × cos(placement angle).

    Optional: LiDAR angular-span width at ruler distance (--use-lidar-width).
    Optional: scale width by ruler_m / z_inside (clamped) when stereo back depth differs.
    """
    size = box_size_class(scene)
    if size is None:
        raise ValueError(f"Unknown box size for scene {scene!r}")
    spec = BOX_SPECS[size]
    view_deg = placement_angle_deg(scene)
    w_nominal_m = (spec["width_nominal_cm"] / 100.0) * float(np.cos(np.deg2rad(view_deg)))

    out: dict = {
        "size_class": size,
        "placement_angle_deg": view_deg,
        "width_nominal_cm": spec["width_nominal_cm"],
        "width_cos_cm": w_nominal_m * 100.0,
        "width_m": w_nominal_m,
        "width_source": "nominal_cos_placement",
    }

    lidar_w = None
    if use_lidar_span and lidar_csv is not None and ruler_m is not None:
        lidar_w = width_from_lidar_span(
            lidar_csv,
            ruler_m,
            placement_angle_deg=view_deg,
            distance_tol_m=distance_tol_m,
            angular_tol_deg=angular_tol_deg,
            bearing_mode="setup",
        )
    if lidar_w is not None:
        out["width_m"] = float(lidar_w["width_m"])
        out["width_source"] = lidar_w.get("width_source", "lidar_placement_face_heuristic")
        out["lidar_width"] = lidar_w
    elif ruler_m is not None and z_inside_m is not None and z_inside_m > 0:
        lo, hi = ruler_scale_clamp
        scale = float(np.clip(ruler_m / z_inside_m, lo, hi))
        out["width_m"] = w_nominal_m * scale
        out["width_source"] = "nominal_cos_x_ruler_scale_clamped"
        out["ruler_scale"] = scale

    out["width_cm"] = out["width_m"] * 100.0
    out["height_cm"] = spec["height_cm"]
    out["gt_volume_cm3"] = spec["gt_volume_cm3"]
    return out


def outside_flap_depth(
    depth_m: np.ndarray,
    roi_mask: np.ndarray,
    ring_mask: np.ndarray,
    *,
    outside_closer_margin_m: float = DEFAULT_OUTSIDE_CLOSER_MARGIN_M,
    consistency_tol_m: float = DEFAULT_OUTSIDE_CONSISTENCY_M,
    min_ring_px: int = 8,
    return_masks: bool = False,
) -> dict | None:
    """
    Z_inside = median depth in ROI (back marker).
    Z_outside = median on outside ring, preferring closer-than-ROI flap pixels
    with consistent depth (|z − z_flap| ≤ consistency_tol).
    depth_m = z_inside − z_outside (positive when inside is farther).
    """
    valid = np.isfinite(depth_m) & (depth_m > 0)
    inside = depth_m[roi_mask & valid]
    outside = depth_m[ring_mask & valid]
    if len(inside) < min_ring_px or len(outside) < min_ring_px:
        return None

    z_inside = float(np.median(inside))
    closer = outside[outside <= (z_inside - outside_closer_margin_m)]
    if len(closer) >= min_ring_px:
        z_seed = float(np.median(closer))
        outside_mode = "closer_flap_seed"
    else:
        z_seed = float(np.median(outside))
        outside_mode = "all_outside_seed"

    consistent = ring_mask & valid & (np.abs(depth_m - z_seed) <= consistency_tol_m)
    n_consistent = int(np.count_nonzero(consistent))
    if n_consistent >= min_ring_px:
        z_outside = float(np.median(depth_m[consistent]))
        outside_mode += "+consistent_band"
    else:
        z_outside = z_seed
        outside_mode += "+seed_only"

    box_depth = z_inside - z_outside
    if box_depth <= 0:
        box_depth = abs(box_depth)
        outside_mode += "|abs|"

    out = {
        "z_inside_m": z_inside,
        "z_outside_m": z_outside,
        "depth_m": float(box_depth),
        "outside_mode": outside_mode,
        "outside_consistent_n": n_consistent,
        "outside_total_n": int(len(outside)),
        "outside_closer_n": int(len(closer)),
    }
    if return_masks:
        out["consistent_ring_mask"] = consistent
        out["ring_mask"] = ring_mask.copy()
        out["roi_mask"] = roi_mask.copy()
    return out


def estimate_box_volume(
    depth_m: np.ndarray,
    roi_mask: np.ndarray,
    ring_mask: np.ndarray,
    scene: str,
    *,
    ruler_m: float | None,
    lidar_csv: Path | None = None,
    use_lidar_span: bool = False,
    outside_closer_margin_m: float = DEFAULT_OUTSIDE_CLOSER_MARGIN_M,
    consistency_tol_m: float = DEFAULT_OUTSIDE_CONSISTENCY_M,
) -> dict | None:
    depth_part = outside_flap_depth(
        depth_m,
        roi_mask,
        ring_mask,
        outside_closer_margin_m=outside_closer_margin_m,
        consistency_tol_m=consistency_tol_m,
    )
    if depth_part is None:
        return None

    try:
        geom = width_from_setup(
            scene,
            ruler_m=ruler_m,
            z_inside_m=depth_part["z_inside_m"],
            lidar_csv=lidar_csv,
            use_lidar_span=use_lidar_span,
        )
    except ValueError:
        return None

    height_m = geom["height_cm"] / 100.0
    width_m = geom["width_m"]
    depth_box = depth_part["depth_m"]
    volume_m3 = width_m * height_m * depth_box

    return {
        **depth_part,
        **{k: v for k, v in geom.items() if k not in depth_part},
        "width_m": width_m,
        "height_m": height_m,
        "volume_m3": float(volume_m3),
        "volume_cm3": float(volume_m3 * 1_000_000.0),
    }


def resolve_ruler_m(data_root: Path, scene: str, pair_id: str) -> float | None:
    txt = Path(data_root) / scene / f"pair_{pair_id}.txt"
    return parse_pair_distance_txt(txt)


def resolve_lidar_csv(data_root: Path, scene: str, pair_id: str) -> Path | None:
    p = Path(data_root) / scene / f"pair_{pair_id}_lidar.csv"
    return p if p.is_file() else None
