"""Excavator volume: LiDAR edge width × catalog height × stereo ROI depth."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from evaluation.box_volume_heuristic import (
    BOX_SPECS,
    angular_diff_deg,
    circular_span_deg,
    load_lidar_scan,
    mask_lidar_span_at_setup,
    resolve_lidar_csv,
    robust_circular_span_deg,
)
from evaluation.lidar_edge_profile_width import width_from_lidar_edge_profile
from evaluation.roi_gt_compare import METHOD_ORDER

EXCAVATOR_SCENES = ("excavator_M", "excavator_S")
EXCAVATOR_SIZE_CLASS = {"excavator_M": "M", "excavator_S": "S"}
EXCAVATOR_GT_VOLUME_CM3 = {"excavator_M": 4940.0, "excavator_S": 294.0}
METHOD_VOLUME_COLS = [(m, f"{m}_volume_cm3") for m in METHOD_ORDER]

_REF_METHODS = ("dav2", "foundation", "opencv", "dav2_gt")


def _norm_pair_id(pair_id: str) -> str:
    p = str(pair_id).strip().removeprefix("pair_")
    return p.zfill(3) if p.isdigit() else p


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if x != x:  # NaN
        return None
    return x


def _ruler_m_from_json_row(row: dict) -> float | None:
    """Use stereo back depth (z_inside) as LiDAR range gate when pair_*.txt is absent."""
    for method in _REF_METHODS:
        block = row.get(method)
        if isinstance(block, dict):
            z = block.get("z_inside_m")
            if z is not None and float(z) > 0.05:
                return float(z)
    return None


def _load_json_rows(runs_root: Path, scene: str) -> dict[str, dict]:
    path = runs_root / scene / "roi_bbox_volume_estimates.json"
    if not path.is_file():
        return {}
    data = json.loads(path.read_text())
    out: dict[str, dict] = {}
    for row in data.get("rows", []):
        pid = _norm_pair_id(row.get("pair_id", ""))
        out[pid] = row
    return out


def _lidar_width_gate_span_cm(
    angle_deg: np.ndarray,
    dist_m: np.ndarray,
    ruler_m: float,
    *,
    size_class: str,
) -> float | None:
    """Fallback opening span when edge-profile width fails (close-range excavator hole)."""
    dtol_steps = (0.20, 0.32, 0.45) if size_class == "M" else (0.14, 0.22, 0.34)
    atol_steps = (30.0, 50.0, 75.0)
    cap_m = 0.90 if size_class == "M" else 0.35
    for dtol in dtol_steps:
        for atol in atol_steps:
            for bearing_mode in ("auto", "placement"):
                gate = mask_lidar_span_at_setup(
                    angle_deg,
                    dist_m,
                    ruler_m=ruler_m,
                    placement_angle_deg=0.0,
                    distance_tol_m=dtol,
                    angular_tol_deg=atol,
                    bearing_mode=bearing_mode,
                )
                n = int(np.count_nonzero(gate))
                if n < 4:
                    continue
                ang = angle_deg[gate]
                r = dist_m[gate]
                r_med = float(np.median(r))
                span_deg = robust_circular_span_deg(ang)
                if span_deg < 1.5 or span_deg > 130.0:
                    span_deg = circular_span_deg(ang)
                if span_deg < 1.5:
                    continue
                width_m = r_med * float(np.deg2rad(span_deg))
                if width_m < 0.02 or width_m > cap_m:
                    continue
                return width_m * 100.0
    return None


def lidar_width_cm_for_pair(
    data_root: Path,
    scene: str,
    pair_id: str,
    ruler_m: float,
) -> float | None:
    size = EXCAVATOR_SIZE_CLASS.get(scene)
    if size is None:
        return None
    lidar_csv = resolve_lidar_csv(data_root, scene, pair_id)
    if lidar_csv is None:
        return None
    lw = width_from_lidar_edge_profile(
        lidar_csv,
        ruler_m,
        0.0,
        scene=scene,
        size_class=size,
        distance_tol_m=0.15,
        angular_tol_deg=30.0,
        bearing_mode="auto",
    )
    if lw is not None:
        return float(lw["width_cm"])

    angle_deg, dist_m = load_lidar_scan(lidar_csv)
    if len(dist_m) == 0:
        return None
    return _lidar_width_gate_span_cm(angle_deg, dist_m, ruler_m, size_class=size)


def lidar_depth_cm_for_pair(
    data_root: Path,
    scene: str,
    pair_id: str,
    ruler_m: float,
) -> float | None:
    """Edge→back depth: range span on placement bearing at stereo z_inside gate."""
    size = EXCAVATOR_SIZE_CLASS.get(scene)
    if size is None:
        return None
    lidar_csv = resolve_lidar_csv(data_root, scene, pair_id)
    if lidar_csv is None:
        return None
    angle_deg, dist_m = load_lidar_scan(lidar_csv)
    if len(dist_m) < 4:
        return None

    gt_depth_m = BOX_SPECS[size]["gt_volume_cm3"] / (
        BOX_SPECS[size]["width_nominal_cm"] * BOX_SPECS[size]["height_cm"]
    ) / 100.0

    for dist_tol in (0.15, 0.25, 0.38):
        for atol in (8.0, 14.0, 22.0):
            off = np.abs(angular_diff_deg(angle_deg, 0.0))
            gate = (
                np.isfinite(angle_deg)
                & np.isfinite(dist_m)
                & (dist_m > 0)
                & (np.abs(dist_m - ruler_m) <= dist_tol)
                & (off <= atol)
            )
            r = dist_m[gate]
            if len(r) < 3:
                continue
            r_hi = ruler_m + max(gt_depth_m * 3.0, 0.08) + 0.05
            r_lo = max(0.05, ruler_m - 0.05)
            in_box = r[(r >= r_lo) & (r <= r_hi)]
            if len(in_box) < 3:
                in_box = r
            depth_m = float(np.percentile(in_box, 90) - np.percentile(in_box, 10))
            if depth_m < 0.003:
                depth_m = float(np.max(in_box) - np.min(in_box))
            if 0.003 <= depth_m <= 0.40:
                return depth_m * 100.0
    return None


def build_excavator_lidar_stereo_volume_rows(
    runs_root: Path,
    data_root: Path,
    *,
    pair_filter: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """
    Same pairs as excavator ROI runs; volume = LiDAR width × catalog height × stereo depth.
    """
    runs_root = Path(runs_root)
    data_root = Path(data_root)
    rows_out: list[dict] = []

    for scene in EXCAVATOR_SCENES:
        size = EXCAVATOR_SIZE_CLASS[scene]
        height_cm = float(BOX_SPECS[size]["height_cm"])
        gt_vol = EXCAVATOR_GT_VOLUME_CM3[scene]
        csv_path = runs_root / scene / "roi_bbox_volume_estimates.csv"
        json_by_pair = _load_json_rows(runs_root, scene)
        if not csv_path.is_file():
            continue

        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                pair_id = _norm_pair_id(r.get("pair_id", ""))
                if pair_filter is not None and (scene, pair_id) not in pair_filter:
                    continue

                jrow = json_by_pair.get(pair_id, {})
                ruler_m = _ruler_m_from_json_row(jrow)
                if ruler_m is None:
                    continue

                width_cm = lidar_width_cm_for_pair(data_root, scene, pair_id, ruler_m)
                if width_cm is None or width_cm <= 0:
                    continue

                row = {
                    "scene": scene,
                    "pair_id": pair_id,
                    "gt_volume_cm3": gt_vol,
                    "width_lidar_cm": width_cm,
                    "height_cm": height_cm,
                    "ruler_proxy_cm": ruler_m * 100.0,
                }
                for method, vol_col in METHOD_VOLUME_COLS:
                    depth_cm = _to_float(r.get(f"{method}_depth_cm"))
                    row[f"{method}_depth_cm"] = depth_cm
                    if depth_cm is None or depth_cm <= 0:
                        row[vol_col] = ""
                        continue
                    vol = width_cm * height_cm * depth_cm
                    row[vol_col] = f"{vol:.1f}"

                rows_out.append(row)

    rows_out.sort(key=lambda x: (x["scene"], x["pair_id"]))
    return rows_out


def build_excavator_lidar_depth_volume_rows(
    runs_root: Path,
    data_root: Path,
    *,
    pair_filter: list[tuple[str, str]] | None = None,
) -> list[dict]:
    """V = LiDAR width × catalog height × LiDAR edge→back depth."""
    runs_root = Path(runs_root)
    data_root = Path(data_root)
    rows_out: list[dict] = []

    for scene in EXCAVATOR_SCENES:
        size = EXCAVATOR_SIZE_CLASS[scene]
        height_cm = float(BOX_SPECS[size]["height_cm"])
        gt_vol = EXCAVATOR_GT_VOLUME_CM3[scene]
        json_by_pair = _load_json_rows(runs_root, scene)

        for pair_id, jrow in sorted(json_by_pair.items()):
            if pair_filter is not None and (scene, pair_id) not in pair_filter:
                continue
            ruler_m = _ruler_m_from_json_row(jrow)
            if ruler_m is None:
                continue
            width_cm = lidar_width_cm_for_pair(data_root, scene, pair_id, ruler_m)
            depth_cm = lidar_depth_cm_for_pair(data_root, scene, pair_id, ruler_m)
            if width_cm is None or depth_cm is None or depth_cm <= 0:
                continue
            vol = width_cm * height_cm * depth_cm
            rows_out.append(
                {
                    "scene": scene,
                    "pair_id": pair_id,
                    "gt_volume_cm3": gt_vol,
                    "width_lidar_cm": width_cm,
                    "height_cm": height_cm,
                    "depth_lidar_cm": depth_cm,
                    "lidar_volume_cm3": f"{vol:.1f}",
                }
            )

    return rows_out


def pair_list_from_combined_json(path: Path) -> list[tuple[str, str]]:
    """Preserve scene/pair set from an existing combined export."""
    if not path.is_file():
        return []
    data = json.loads(path.read_text())
    out: list[tuple[str, str]] = []
    for row in data.get("rows", []):
        scene = row.get("scene", "")
        pair_id = _norm_pair_id(row.get("pair_id", ""))
        if scene and pair_id:
            out.append((scene, pair_id))
    return out
