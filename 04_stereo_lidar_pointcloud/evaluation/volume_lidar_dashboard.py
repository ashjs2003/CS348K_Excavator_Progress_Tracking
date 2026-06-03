"""Recompute cardboard box volume for dashboard: LiDAR width × height × ROI depth."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from evaluation.box_volume_heuristic import BOX_SPECS, box_size_class
from evaluation.lidar_width_vs_gt import collect_lidar_width_points
from evaluation.roi_gt_compare import METHOD_ORDER

CARDBOARD_SCENES = [
    "L_carboard_box",
    "L_cardboard_box_30",
    "M_cardboard_box",
    "M_cardboardbox_30",
    "S_cardboard_box",
    "S_cardboard_box_30",
]

METHOD_VOLUME_COLS = [(m, f"{m}_volume_cm3") for m in METHOD_ORDER]
METHOD_DEPTH_COLS = [(m, f"{m}_depth_cm") for m in METHOD_ORDER]


def _norm_pair_id(pair_id: str) -> str:
    p = str(pair_id).strip().removeprefix("pair_")
    return p.zfill(3) if p.isdigit() else p


def _row_key_from_scene(scene: str) -> str | None:
    mapping = {
        "L_carboard_box": "L-0deg",
        "L_cardboard_box_30": "L-30deg",
        "M_cardboard_box": "M-0deg",
        "M_cardboardbox_30": "M-30deg",
        "S_cardboard_box": "S-0deg",
        "S_cardboard_box_30": "S-30deg",
    }
    return mapping.get(scene)


def build_lidar_width_lookup(
    scene_points: dict[str, list[dict]] | None = None,
    *,
    data_root: Path | None = None,
) -> dict[tuple[str, str], float]:
    if scene_points is None:
        if data_root is None:
            raise ValueError("Provide scene_points or data_root")
        scene_points = collect_lidar_width_points(data_root)
    out: dict[tuple[str, str], float] = {}
    for scene, pts in scene_points.items():
        for p in pts:
            w = p.get("width_lidar_cm")
            if w is None:
                continue
            pid = _norm_pair_id(p["pair_id"])
            out[(scene, pid)] = float(w)
    return out


def _to_float(v) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(x):
        return None
    return x


def load_lidar_volume_rows(
    runs_root: Path,
    *,
    data_root: Path,
    width_lookup: dict[tuple[str, str], float] | None = None,
) -> list[dict]:
    """
    One dict per capture with recomputed ``{method}_volume_cm3`` and depth/width fields.
    """
    if width_lookup is None:
        width_lookup = build_lidar_width_lookup(data_root=data_root)

    rows: list[dict] = []
    for scene in CARDBOARD_SCENES:
        csv_path = runs_root / scene / "roi_bbox_volume_estimates.csv"
        if not csv_path.is_file():
            continue
        row_key = _row_key_from_scene(scene)
        if row_key is None:
            continue
        size = box_size_class(scene)
        default_h = BOX_SPECS[size]["height_cm"] if size else None

        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                pair_id = _norm_pair_id(r.get("pair_id", ""))
                dist = _to_float(r.get("ruler_distance_cm"))
                gt = _to_float(r.get("gt_volume_cm3"))
                if dist is None or gt is None or gt <= 0:
                    continue

                width_cm = width_lookup.get((scene, pair_id))
                height_cm = _to_float(r.get("height_cm")) or default_h
                if width_cm is None or height_cm is None or height_cm <= 0:
                    continue

                row = {
                    "scene": scene,
                    "pair_id": pair_id,
                    "row_key": row_key,
                    "distance_cm": dist,
                    "gt_volume_cm3": gt,
                    "width_lidar_cm": width_cm,
                    "height_cm": height_cm,
                }
                for method, vol_col in METHOD_VOLUME_COLS:
                    depth_cm = _to_float(r.get(f"{method}_depth_cm"))
                    row[f"{method}_depth_cm"] = depth_cm
                    if depth_cm is None or depth_cm <= 0:
                        row[vol_col] = None
                        continue
                    row[vol_col] = float(width_cm * height_cm * depth_cm)

                rows.append(row)
    return rows
