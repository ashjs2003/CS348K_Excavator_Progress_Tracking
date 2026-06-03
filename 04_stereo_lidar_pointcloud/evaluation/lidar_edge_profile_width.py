"""
LiDAR width from 1D range-profile edge detection (no catalog face width).

Pipeline: ruler + placement/setup bearing gate → sort by angle → smooth range →
find range jumps (face edges) → width ≈ r_face × Δθ (and XY chord between edges).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evaluation.box_volume_heuristic import (
    angular_diff_deg,
    load_lidar_scan,
    mask_lidar_span_at_setup,
    robust_circular_span_deg,
)
from evaluation.lidar_segment_width import lidar_xy_from_polar

# LiDAR edge tuning per box size (width_hint_m is for span search only, not measured width).
SIZE_EDGE_TUNING: dict[str, dict] = {
    "S": {
        "angular_tol_deg": 12.0,
        "angular_tol_widen_deg": 18.0,
        "bearing_mode": "placement",
        "max_span_deg": 22.0,
        "max_center_offset_deg": 12.0,
        "max_edge_offset_deg": 14.0,
        "center_offset_penalty": 18.0,
        "edge_offset_penalty": 6.0,
        "min_gate_points": 3,
        "min_face_points": 3,
        "min_points_between": 3,
    },
    "M": {
        "angular_tol_deg": 28.0,
        "angular_tol_widen_deg": 36.0,
        "bearing_mode": "setup",
        "placement_median_max_offset_deg": 18.0,
        "max_span_deg": 32.0,
        "max_span_cap_deg": 48.0,
        "max_span_margin_deg": 10.0,
        "width_hint_m": 0.26,
        "min_jump_m": 0.008,
        "min_points_between": 4,
        "min_gate_points": 4,
        "min_face_points": 4,
        "max_center_offset_deg": 20.0,
        "center_offset_penalty": 6.0,
        "edge_offset_penalty": 4.0,
        "wide_span_deg": 22.0,
        "max_width_scale": 1.35,
        "plateau_tol_m": 0.03,
    },
    "L": {
        "angular_tol_deg": 32.0,
        "angular_tol_widen_deg": 42.0,
        "bearing_mode": "setup",
        "placement_median_max_offset_deg": 20.0,
        "max_span_deg": 38.0,
        "max_span_cap_deg": 65.0,
        "max_span_margin_deg": 12.0,
        "width_hint_m": 0.40,
        "min_jump_m": 0.006,
        "min_points_between": 3,
        "min_gate_points": 4,
        "min_face_points": 4,
        "max_center_offset_deg": 22.0,
        "center_offset_penalty": 5.0,
        "edge_offset_penalty": 3.0,
        "wide_span_deg": 24.0,
        "max_width_scale": 1.35,
        "plateau_tol_m": 0.035,
    },
}


def _tune_for_size(size_class: str | None) -> dict:
    if size_class and size_class in SIZE_EDGE_TUNING:
        return dict(SIZE_EDGE_TUNING[size_class])
    return {}


def _effective_max_span_deg(tune: dict, ruler_m: float) -> float:
    """Allow wider angular segments when the box is close (geometry prior for search)."""
    base = float(tune.get("max_span_deg", 28.0))
    hint = tune.get("width_hint_m")
    if hint is None:
        return base
    cap = float(tune.get("max_span_cap_deg", 90.0))
    margin = float(tune.get("max_span_margin_deg", 8.0))
    geom = float(np.rad2deg(2.0 * np.arctan(0.55 * float(hint) / max(float(ruler_m), 0.12))))
    return float(np.clip(max(base, geom + margin), base, cap))


def _distance_tol_m(tune: dict, ruler_m: float, default: float) -> float:
    if ruler_m <= 0.35:
        return float(tune.get("distance_tol_close_m", max(default, 0.07)))
    return default


def _expected_span_deg(hint_m: float, ruler_m: float) -> float:
    return float(np.rad2deg(2.0 * np.arctan(0.5 * hint_m / max(ruler_m, 0.12))))


def _width_from_gate_extent(
    ang_sorted: np.ndarray,
    r_sorted: np.ndarray,
    *,
    ruler_m: float,
    distance_tol_m: float,
    q_low: float = 8.0,
    q_high: float = 92.0,
) -> tuple[float, int, int] | None:
    """Fallback width from angular extent of gated returns near face range."""
    if len(ang_sorted) < 4:
        return None
    r_med = float(np.median(r_sorted))
    on_r = np.abs(r_sorted - r_med) <= max(distance_tol_m, 0.04)
    if int(np.count_nonzero(on_r)) < 4:
        on_r = np.ones(len(r_sorted), dtype=bool)
    ang = ang_sorted[on_r]
    r = r_sorted[on_r]
    center = np.deg2rad(float(np.median(ang)))
    rel = np.rad2deg(
        np.arctan2(
            np.sin(np.deg2rad(ang) - center),
            np.cos(np.deg2rad(ang) - center),
        )
    )
    lo, hi = np.percentile(rel, [q_low, q_high])
    i0 = int(np.argmin(np.abs(rel - lo)))
    i1 = int(np.argmin(np.abs(rel - hi)))
    ang0, ang1 = float(ang[i0]), float(ang[i1])
    r_face = float(np.median(r))
    delta = abs(np.deg2rad(ang1) - np.deg2rad(ang0))
    if delta > np.pi:
        delta = 2.0 * np.pi - delta
    width_arc = r_face * delta
    xy = lidar_xy_from_polar(np.array([ang0, ang1]), np.array([r_face, r_face]))
    width_chord = float(np.linalg.norm(xy[1] - xy[0]))
    return float(max(width_arc, width_chord)), i0, i1


def _bearing_offset_deg(angle_deg: np.ndarray, center_deg: float) -> np.ndarray:
    return np.abs(angular_diff_deg(angle_deg, center_deg))


def _setup_gate_with_fallback(
    angle_deg: np.ndarray,
    dist_m: np.ndarray,
    *,
    ruler_m: float,
    placement_angle_deg: float,
    distance_tol_m: float,
    angular_tol_deg: float,
    bearing_mode: str,
    min_gate_points: int = 3,
    size_class: str | None = None,
) -> tuple[np.ndarray, str] | None:
    tune = _tune_for_size(size_class)
    if tune.get("bearing_mode") == "placement":
        tol_steps = [
            float(tune.get("angular_tol_deg", angular_tol_deg)),
            float(tune.get("angular_tol_widen_deg", angular_tol_deg)),
        ]
        min_gate = int(tune.get("min_gate_points", min_gate_points))
        for tol in tol_steps:
            gate = mask_lidar_span_at_setup(
                angle_deg,
                dist_m,
                ruler_m=ruler_m,
                placement_angle_deg=placement_angle_deg,
                distance_tol_m=distance_tol_m,
                angular_tol_deg=tol,
                bearing_mode="placement",
            )
            if int(np.count_nonzero(gate)) >= min_gate:
                label = "placement" if tol == tol_steps[0] else "placement_wide"
                return gate, label
        return None

    mode = (bearing_mode or "setup").lower()
    modes = ["placement", "auto"] if mode == "setup" else [mode]

    def _gate_for(gm: str, tol: float) -> np.ndarray:
        return mask_lidar_span_at_setup(
            angle_deg,
            dist_m,
            ruler_m=ruler_m,
            placement_angle_deg=placement_angle_deg,
            distance_tol_m=distance_tol_m,
            angular_tol_deg=tol,
            bearing_mode=gm,
        )

    placement_gate = None
    for gm in modes:
        gate = _gate_for(gm, angular_tol_deg)
        if int(np.count_nonzero(gate)) < min_gate_points:
            continue
        if gm == "placement":
            med = float(np.median(angle_deg[gate]))
            off = float(np.abs(angular_diff_deg(np.array([med]), placement_angle_deg))[0])
            max_off = float(tune.get("placement_median_max_offset_deg", 8.0))
            if off > max_off and mode == "setup":
                placement_gate = gate
                continue
            return gate, gm
        return gate, gm

    if placement_gate is not None:
        auto_gate = _gate_for("auto", angular_tol_deg)
        if int(np.count_nonzero(auto_gate)) >= min_gate_points:
            return auto_gate, "auto"
        return placement_gate, "placement"
    return None


def _longest_plateau_segment(
    ang_sorted: np.ndarray,
    r_sorted: np.ndarray,
    *,
    plateau_tol_m: float,
    placement_angle_deg: float | None = None,
    max_center_offset_deg: float | None = None,
    span_weight: float = 0.0,
) -> tuple[int, int] | None:
    """Inclusive index range [i0, i1] of longest contiguous near-median range run."""
    if len(r_sorted) < 2:
        return None
    r_med = float(np.median(r_sorted))
    on = np.abs(r_sorted - r_med) <= plateau_tol_m
    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(ang_sorted)))
    gaps = np.where(np.diff(unwrapped) > 4.0)[0] + 1
    slices = np.split(np.arange(len(on)), gaps)

    best: tuple[int, int] | None = None
    best_score = -np.inf
    for sl in slices:
        seg_on = on[sl]
        if not np.any(seg_on):
            continue
        idx = np.where(seg_on)[0]
        if len(idx) == 0:
            continue
        splits = np.where(np.diff(idx) > 1)[0] + 1
        runs = np.split(idx, splits)
        for run in runs:
            if len(run) < 2:
                continue
            i0 = int(sl[run[0]])
            i1 = int(sl[run[-1]])
            seg_ang = ang_sorted[i0 : i1 + 1]
            center = float(np.median(seg_ang))
            off = 0.0
            if placement_angle_deg is not None:
                off = float(np.abs(angular_diff_deg(np.array([center]), placement_angle_deg))[0])
                if max_center_offset_deg is not None and off > max_center_offset_deg:
                    continue
            span_run = float(unwrapped[i1] - unwrapped[i0])
            score = float(len(run)) + span_weight * span_run - 3.0 * off
            if score > best_score:
                best_score = score
                best = (i0, i1)
    return best


def _find_edge_pair(
    ang_sorted: np.ndarray,
    r_sorted: np.ndarray,
    *,
    ruler_m: float,
    distance_tol_m: float,
    placement_angle_deg: float | None = None,
    min_jump_m: float = 0.012,
    max_span_deg: float = 28.0,
    min_span_deg: float = 3.0,
    min_points_between: int = 6,
    max_center_offset_deg: float | None = None,
    max_edge_offset_deg: float | None = None,
    center_offset_penalty: float = 0.0,
    edge_offset_penalty: float = 0.0,
    width_hint_m: float | None = None,
) -> tuple[int, int] | None:
    """
    Return sorted-array indices (i_left, i_right) of range-profile edges.
    """
    n = len(r_sorted)
    if n < 4:
        return None

    if n >= 5:
        k = 5
        r_smooth = np.convolve(r_sorted, np.ones(k) / k, mode="same")
    else:
        r_smooth = r_sorted.copy()

    dr = np.abs(np.diff(r_smooth))
    thresh = max(min_jump_m, float(np.percentile(dr, 80)))
    peak_idx = [int(i) for i in np.where(dr >= thresh)[0]]

    if len(peak_idx) < 2:
        return None

    unwrapped = np.rad2deg(np.unwrap(np.deg2rad(ang_sorted)))
    best: tuple[int, int] | None = None
    best_score = -np.inf

    for a in range(len(peak_idx)):
        for b in range(a + 1, len(peak_idx)):
            i = peak_idx[a]
            j = peak_idx[b]
            i0, i1 = i, j + 1
            if i1 >= n:
                continue
            if i1 - i0 + 1 < min_points_between:
                continue
            span = float(unwrapped[i1] - unwrapped[i0])
            if span < min_span_deg or span > max_span_deg:
                continue
            seg_r = r_sorted[i0 : i1 + 1]
            r_face = float(np.median(seg_r))
            if abs(r_face - ruler_m) > max(distance_tol_m * 2.0, 0.08):
                continue
            on_face_frac = float(np.mean(np.abs(seg_r - r_face) <= 0.03))
            if on_face_frac < 0.45:
                continue
            seg_ang = ang_sorted[i0 : i1 + 1]
            center = float(np.median(seg_ang))
            center_off = 0.0
            max_edge_off = 0.0
            if placement_angle_deg is not None:
                center_off = float(
                    np.abs(angular_diff_deg(np.array([center]), placement_angle_deg))[0]
                )
                max_edge_off = float(
                    np.max(_bearing_offset_deg(seg_ang[[0, -1]], placement_angle_deg))
                )
                if max_center_offset_deg is not None and center_off > max_center_offset_deg:
                    continue
                if max_edge_offset_deg is not None and max_edge_off > max_edge_offset_deg:
                    continue
            score = (
                span
                + 20.0 * on_face_frac
                - 40.0 * abs(r_face - ruler_m)
                - center_offset_penalty * center_off
                - edge_offset_penalty * max_edge_off
            )
            if width_hint_m is not None:
                exp = _expected_span_deg(width_hint_m, ruler_m)
                if span < 0.55 * exp:
                    score -= 12.0 * (1.0 - span / max(exp, 1.0))
                elif span > 1.15 * exp:
                    score -= 6.0 * (span / max(exp, 1.0) - 1.15)
            if score > best_score:
                best_score = score
                best = (i0, i1)

    return best


def width_from_lidar_edge_profile_arrays(
    angle_deg: np.ndarray,
    dist_m: np.ndarray,
    ruler_m: float,
    placement_angle_deg: float,
    *,
    distance_tol_m: float = 0.05,
    angular_tol_deg: float = 20.0,
    bearing_mode: str = "setup",
    plateau_tol_m: float = 0.025,
    min_gate_points: int = 5,
    min_face_points: int = 4,
    size_class: str | None = None,
) -> dict | None:
    tune = _tune_for_size(size_class)
    angular_tol_deg = float(tune.get("angular_tol_deg", angular_tol_deg))
    bearing_mode = str(tune.get("bearing_mode", bearing_mode))
    min_gate_points = int(tune.get("min_gate_points", min_gate_points))
    if size_class in ("M", "L") and ruler_m <= 0.25:
        min_gate_points = min(min_gate_points, 3)
    min_face_points = int(tune.get("min_face_points", min_face_points))
    distance_tol_m = _distance_tol_m(tune, ruler_m, distance_tol_m)
    plateau_tol_m = float(tune.get("plateau_tol_m", plateau_tol_m))
    max_span_deg = _effective_max_span_deg(tune, ruler_m)
    if size_class == "S" and len(angle_deg) > 0:
        min_face_points = min(min_face_points, 2)

    gated = _setup_gate_with_fallback(
        angle_deg,
        dist_m,
        ruler_m=ruler_m,
        placement_angle_deg=placement_angle_deg,
        distance_tol_m=distance_tol_m,
        angular_tol_deg=angular_tol_deg,
        bearing_mode=bearing_mode,
        min_gate_points=min_gate_points,
        size_class=size_class,
    )
    if gated is None:
        return None
    gate, gate_mode_used = gated

    ang = angle_deg[gate]
    r = dist_m[gate]
    order = np.argsort(np.unwrap(np.deg2rad(ang)))
    ang_s = ang[order]
    r_s = r[order]

    min_pts_between = int(tune.get("min_points_between", 6))
    if size_class == "S":
        min_pts_between = min(min_pts_between, max(2, len(ang_s) - 1))

    edge_pair = None
    if len(ang_s) >= 5:
        edge_pair = _find_edge_pair(
            ang_s,
            r_s,
            ruler_m=ruler_m,
            distance_tol_m=distance_tol_m,
            placement_angle_deg=placement_angle_deg,
            min_jump_m=float(tune.get("min_jump_m", 0.012)),
            max_span_deg=max_span_deg,
            min_points_between=min_pts_between,
            max_center_offset_deg=tune.get("max_center_offset_deg"),
            max_edge_offset_deg=tune.get("max_edge_offset_deg"),
            center_offset_penalty=float(tune.get("center_offset_penalty", 0.0)),
            edge_offset_penalty=float(tune.get("edge_offset_penalty", 0.0)),
            width_hint_m=tune.get("width_hint_m"),
        )
    method_detail = "range_jump_edges"

    if edge_pair is None:
        plateau = _longest_plateau_segment(
            ang_s,
            r_s,
            plateau_tol_m=plateau_tol_m,
            placement_angle_deg=placement_angle_deg,
            max_center_offset_deg=tune.get("max_center_offset_deg"),
            span_weight=0.85 if size_class in ("M", "L") else 0.0,
        )
        if plateau is None:
            return None
        i0, i1 = plateau
        method_detail = "longest_plateau_fallback"
    else:
        i0, i1 = edge_pair

    if i1 - i0 + 1 < min_face_points:
        return None

    ang_face = ang_s[i0 : i1 + 1]
    r_face_arr = r_s[i0 : i1 + 1]
    r_med = float(np.median(r_face_arr))
    span_deg = robust_circular_span_deg(ang_face)
    max_span = max_span_deg
    min_span = 1.5 if size_class == "S" else 2.0
    if span_deg > max(max_span, 60.0) or span_deg < min_span:
        return None
    if placement_angle_deg is not None and tune.get("max_center_offset_deg") is not None:
        center_off = float(
            np.abs(angular_diff_deg(np.array([np.median(ang_face)]), placement_angle_deg))[0]
        )
        if center_off > float(tune["max_center_offset_deg"]):
            return None

    unwrapped = np.unwrap(np.deg2rad(ang_s))
    delta_rad = abs(unwrapped[i1] - unwrapped[i0])
    width_arc_m = r_med * delta_rad

    xy = lidar_xy_from_polar(ang_face, r_face_arr)
    u = np.unwrap(np.arctan2(xy[:, 1], xy[:, 0]))
    o = np.argsort(u)
    width_chord_m = float(np.linalg.norm(xy[o][-1] - xy[o][0]))

    wide_span = float(tune.get("wide_span_deg", 25.0))
    if span_deg >= wide_span and ruler_m >= 0.65:
        width_m = float(min(width_arc_m, width_chord_m))
    else:
        width_m = float(max(width_arc_m, width_chord_m))

    hint = tune.get("width_hint_m")
    if hint is not None:
        cap = float(tune.get("max_width_scale", 1.35)) * float(hint)
        if width_m > cap:
            return None
        if width_m < 0.72 * float(hint):
            ang_g = angle_deg[gate]
            r_g = dist_m[gate]
            ord_g = np.argsort(np.unwrap(np.deg2rad(ang_g)))
            fb = _width_from_gate_extent(
                ang_g[ord_g],
                r_g[ord_g],
                ruler_m=ruler_m,
                distance_tol_m=distance_tol_m,
            )
            if fb is not None:
                w_fb, _, _ = fb
                if w_fb > width_m:
                    width_m = float(min(w_fb, cap))
                    method_detail = "gate_extent_fallback"

    if width_m < 0.02 or width_m > 1.2:
        return None

    face_mask = np.zeros(len(angle_deg), dtype=bool)
    gate_idx = np.where(gate)[0]
    face_local = np.zeros(len(ang_s), dtype=bool)
    face_local[i0 : i1 + 1] = True
    face_mask[gate_idx[face_local]] = True

    return {
        "width_m": width_m,
        "width_cm": width_m * 100.0,
        "width_arc_cm": width_arc_m * 100.0,
        "width_chord_cm": width_chord_m * 100.0,
        "lidar_span_deg": span_deg,
        "edge_angle_min_deg": float(ang_s[i0]),
        "edge_angle_max_deg": float(ang_s[i1]),
        "lidar_n": int(np.count_nonzero(face_mask)),
        "lidar_n_gate": int(np.count_nonzero(gate)),
        "lidar_range_m": r_med,
        "placement_angle_deg": placement_angle_deg,
        "bearing_mode": bearing_mode,
        "gate_bearing_mode": gate_mode_used,
        "edge_method_detail": method_detail,
        "width_source": "lidar_range_profile_edges",
        "size_class": size_class,
        "face_mask": face_mask,
        "gate_mask": gate,
    }


def width_from_lidar_edge_profile(
    lidar_csv: Path,
    ruler_m: float,
    placement_angle_deg: float,
    *,
    scene: str | None = None,
    size_class: str | None = None,
    **kwargs,
) -> dict | None:
    from evaluation.box_volume_heuristic import box_size_class

    if size_class is None and scene is not None:
        size_class = box_size_class(scene)
    angle_deg, dist_m = load_lidar_scan(lidar_csv)
    if len(dist_m) == 0:
        return None
    return width_from_lidar_edge_profile_arrays(
        angle_deg,
        dist_m,
        ruler_m,
        placement_angle_deg,
        size_class=size_class,
        **kwargs,
    )
