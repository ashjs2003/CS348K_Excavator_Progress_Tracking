"""
Polar LiDAR grid: GT distance (columns) × placement angle (rows).

Shows LiDAR returns for span-based width (method 2):
  - ruler distance from pair_*.txt (±distance_tol)
  - bearing gate: auto median at that range, or scene placement (±angular_tol)

Width = range-profile edge detection (placement + ruler gate). No catalog width.

Black ray = scene placement angle; orange = min/max bearing of filtered returns.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from calib_utils import read_lidar_csv
from evaluation.box_volume_heuristic import box_size_class, circular_span_deg, width_from_lidar_span
from evaluation.lidar_edge_profile_width import width_from_lidar_edge_profile_arrays
from evaluation.gt_depth_overlay import parse_pair_distance_txt

SCENE_PLACEMENT_ANGLE_DEG: dict[str, float] = {
    "checkerboard_data": 0.0,
    "checkerboard_data_30": -30.0,
    "checkerboard_data_60": 60.0,
    "L_carboard_box": 0.0,
    "L_cardboard_box_30": -30.0,
    "M_cardboard_box": 0.0,
    "M_cardboardbox_30": -30.0,
    "S_cardboard_box": 0.0,
    "S_cardboard_box_30": -30.0,
}

DEFAULT_DISTANCE_COLUMNS_CM = [10, 15, 20, 25, 50, 75, 100]
CARDBOARD_DISTANCE_COLUMNS_CM = [20, 25, 50, 75, 100]

CARDBOARD_SML_SCENES = [
    "S_cardboard_box",
    "S_cardboard_box_30",
    "M_cardboard_box",
    "M_cardboardbox_30",
    "L_carboard_box",
    "L_cardboard_box_30",
]

# Rows for combined S/M/L figure: (size label, placement angle deg)
def _set_column_header_labels(
    fig: plt.Figure,
    axes: np.ndarray,
    distance_columns_cm: list[float],
    *,
    header_row: int = 0,
    y_pad: float = 0.014,
) -> None:
    for ci, d_cm in enumerate(distance_columns_cm):
        ax = axes[header_row, ci]
        if not ax.get_visible():
            continue
        pos = ax.get_position()
        fig.text(
            (pos.x0 + pos.x1) / 2,
            pos.y1 + y_pad,
            f"{int(d_cm)} cm",
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="medium",
        )


def _add_polar_grid_legend(fig: plt.Figure, axes: np.ndarray) -> None:
    ax_ref = None
    for ax in axes.flat:
        if ax.get_visible():
            ax_ref = ax
            break
    if ax_ref is None:
        return
    handles, labels = ax_ref.get_legend_handles_labels()
    if not handles:
        return
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=min(4, len(handles)),
        fontsize=8,
        frameon=True,
        handlelength=1.4,
        columnspacing=1.4,
    )


CARDBOARD_SML_ROWS: list[tuple[str, float]] = [
    ("S", 0.0),
    ("S", -30.0),
    ("M", 0.0),
    ("M", -30.0),
    ("L", 0.0),
    ("L", -30.0),
]


def placement_angle_for_scene(scene: str) -> float:
    if scene in SCENE_PLACEMENT_ANGLE_DEG:
        return SCENE_PLACEMENT_ANGLE_DEG[scene]
    if scene.endswith("_30") or "box_30" in scene or "box30" in scene:
        return 30.0
    return 0.0


def discover_scene_captures(data_root: Path, scene: str) -> list[dict]:
    scene_dir = Path(data_root) / scene
    if not scene_dir.is_dir():
        return []
    out = []
    for lidar_path in sorted(scene_dir.glob("pair_*_lidar.csv")):
        pair_id = lidar_path.stem.replace("pair_", "").replace("_lidar", "")
        txt_path = scene_dir / f"pair_{pair_id}.txt"
        gt_m = parse_pair_distance_txt(txt_path)
        if gt_m is None:
            continue
        out.append(
            {
                "scene": scene,
                "pair_id": pair_id,
                "lidar_path": lidar_path,
                "txt_path": txt_path,
                "gt_cm": float(gt_m * 100.0),
                "gt_m": float(gt_m),
                "placement_angle_deg": placement_angle_for_scene(scene),
            }
        )
    return out


def index_by_distance_and_angle(
    data_root: Path,
    scenes: list[str],
    distance_columns_cm: list[float],
) -> dict[tuple[float, float], dict | None]:
    cells: dict[tuple[float, float], dict | None] = {
        (ang, float(d)): None
        for ang in {placement_angle_for_scene(s) for s in scenes}
        for d in distance_columns_cm
    }
    for scene in scenes:
        angle = placement_angle_for_scene(scene)
        for cap in discover_scene_captures(data_root, scene):
            key = (angle, cap["gt_cm"])
            if key in cells and cells[key] is None:
                cells[key] = cap
    return cells


def index_cardboard_sml_cells(
    data_root: Path,
    scenes: list[str],
    distance_columns_cm: list[float],
) -> dict[tuple[str, float, float], dict | None]:
    """Map (size S/M/L, placement angle, ruler cm) → one capture."""
    dist_set = {float(d) for d in distance_columns_cm}
    cells: dict[tuple[str, float, float], dict | None] = {
        (size, ang, float(d)): None for size, ang in CARDBOARD_SML_ROWS for d in distance_columns_cm
    }
    for scene in scenes:
        size = box_size_class(scene)
        if size is None:
            continue
        angle = placement_angle_for_scene(scene)
        for cap in discover_scene_captures(data_root, scene):
            d_cm = float(cap["gt_cm"])
            if d_cm not in dist_set:
                continue
            key = (size, angle, d_cm)
            if key in cells and cells[key] is None:
                cells[key] = cap
    return cells


def plot_polar_cell(
    ax,
    cap: dict,
    *,
    target_label: str = "Target",
    distance_tol_m: float = 0.05,
    angular_tol_deg: float = 15.0,
    bearing_mode: str = "setup",
    coplanar_tol_m: float = 0.02,
    min_quality: float = 0.0,
    r_max_m: float = 1.15,
) -> dict:
    scan = read_lidar_csv(cap["lidar_path"])
    ang_deg = scan[:, 0]
    dist_m = scan[:, 1]
    qual = scan[:, 2]

    gt_m = cap["gt_m"]
    place_deg = cap["placement_angle_deg"]
    size_class = box_size_class(cap.get("scene", ""))
    lw = width_from_lidar_edge_profile_arrays(
        ang_deg,
        dist_m,
        gt_m,
        place_deg,
        distance_tol_m=distance_tol_m,
        angular_tol_deg=angular_tol_deg,
        bearing_mode=bearing_mode,
        plateau_tol_m=max(coplanar_tol_m, 0.02),
        size_class=size_class,
    )
    gate_mode_used = lw.get("gate_bearing_mode") if lw else None
    if lw is not None:
        target = lw["face_mask"].copy()
        gate = lw["gate_mask"].copy()
    else:
        target = np.zeros(len(ang_deg), dtype=bool)
        gate = target.copy()

    ax.set_theta_zero_location("E")
    ax.set_theta_direction(1)
    ax.set_ylim(0, r_max_m)
    ax.set_yticklabels([])
    ax.set_xticklabels([])
    ax.grid(True, color="#cccccc", linewidth=0.4, alpha=0.6)

    valid = np.isfinite(ang_deg) & np.isfinite(dist_m) & (dist_m > 0) & (dist_m <= r_max_m)
    if min_quality > 0:
        valid &= np.isfinite(qual) & (qual >= min_quality)
    gate_only = valid & gate & ~target
    noise = valid & ~gate

    if np.any(gate_only):
        ax.scatter(
            np.deg2rad(ang_deg[gate_only]),
            dist_m[gate_only],
            s=10,
            c="#f4a582",
            alpha=0.75,
            linewidths=0,
            label="Setup gate (not in cluster)",
            zorder=2,
        )
    if np.any(noise):
        ax.scatter(
            np.deg2rad(ang_deg[noise]),
            dist_m[noise],
            s=8,
            c="#c8c8c8",
            alpha=0.55,
            linewidths=0,
            label="Scan (outside gate)",
            zorder=2,
        )
    if np.any(target):
        ax.scatter(
            np.deg2rad(ang_deg[target]),
            dist_m[target],
            s=16,
            c="#d62728",
            alpha=0.9,
            linewidths=0,
            label="Face (between edges)",
            zorder=3,
        )

    place_rad = np.deg2rad(place_deg)
    ax.plot([place_rad, place_rad], [0, r_max_m], color="black", linewidth=2.2, zorder=5)

    stats: dict = {
        "gt_cm": cap["gt_cm"],
        "placement_angle_deg": place_deg,
        "angular_tol_deg": angular_tol_deg,
        "bearing_mode": bearing_mode,
        "distance_tol_cm": distance_tol_m * 100.0,
        "n_face_cluster": int(np.count_nonzero(target)),
        "n_gate": int(np.count_nonzero(gate)),
        "n_target": int(np.count_nonzero(target)),
        "n_scan": int(np.count_nonzero(valid)),
        "n_noise": int(np.count_nonzero(noise)),
    }
    if lw is not None:
        r_med = float(lw["lidar_range_m"])
        stats["span_deg"] = lw.get("lidar_span_deg")
        stats["angle_min_deg"] = lw.get("edge_angle_min_deg")
        stats["angle_max_deg"] = lw.get("edge_angle_max_deg")
        stats["edge_method_detail"] = lw.get("edge_method_detail")
        stats["bearing_median_offset_deg"] = float(
            np.median((ang_deg[target] - place_deg + 180.0) % 360.0 - 180.0)
        ) if np.any(target) else None
        t_min = np.deg2rad(float(lw["edge_angle_min_deg"]))
        t_max = np.deg2rad(float(lw["edge_angle_max_deg"]))
        ax.plot([t_min, t_min], [0, r_med], color="#c45c00", linewidth=1.8, zorder=4)
        ax.plot([t_max, t_max], [0, r_med], color="#c45c00", linewidth=1.8, zorder=4)
        stats["width_lidar_cm"] = lw["width_cm"]
        stats["width_lidar_m"] = lw["width_m"]
        stats["width_arc_cm"] = lw.get("width_arc_cm")
        stats["width_chord_cm"] = lw.get("width_chord_cm")
        stats["width_method"] = lw.get("width_source", "lidar_range_profile_edges")
        stats["gate_bearing_mode"] = lw.get("gate_bearing_mode")
    elif np.count_nonzero(target) >= 2:
        ang_t = ang_deg[target]
        stats["span_deg"] = circular_span_deg(ang_t)
    return stats


def render_cardboard_sml_combined_grid(
    data_root: Path,
    out_path: Path,
    *,
    scenes: list[str] | None = None,
    distance_columns_cm: list[float] | None = None,
    title: str = "LiDAR setup — cardboard S / M / L (edge width)",
    distance_tol_cm: float = 5.0,
    angular_tol_deg: float = 20.0,
    bearing_mode: str = "setup",
    coplanar_tol_m: float = 0.02,
) -> dict:
    """One polar grid: rows = S/M/L × (0°, −30°), columns = ruler distances (no 10/15 cm)."""
    scenes = scenes or CARDBOARD_SML_SCENES
    distance_columns_cm = distance_columns_cm or list(CARDBOARD_DISTANCE_COLUMNS_CM)
    cells = index_cardboard_sml_cells(data_root, scenes, distance_columns_cm)

    n_rows = len(CARDBOARD_SML_ROWS)
    n_cols = len(distance_columns_cm)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.1 * n_cols, 1.85 * n_rows),
        subplot_kw={"projection": "polar"},
    )
    if n_rows == 1:
        axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    all_stats: dict[str, dict] = {}
    for ri, (size, ang) in enumerate(CARDBOARD_SML_ROWS):
        for ci, d_cm in enumerate(distance_columns_cm):
            ax = axes[ri, ci]
            cap = cells.get((size, ang, float(d_cm)))
            if cap is None:
                ax.axis("off")
                continue
            key = f"{cap['scene']}/pair_{cap['pair_id']}"
            all_stats[key] = plot_polar_cell(
                ax,
                cap,
                target_label=f"{size} box",
                distance_tol_m=distance_tol_cm / 100.0,
                angular_tol_deg=angular_tol_deg,
                bearing_mode=bearing_mode,
                coplanar_tol_m=coplanar_tol_m,
            )
    for ri, (size, ang) in enumerate(CARDBOARD_SML_ROWS):
        ang_lbl = f"{int(ang)}°" if ang == int(ang) else f"{ang:.0f}°"
        axes[ri, 0].text(
            -0.42,
            0.5,
            f"{size} · {ang_lbl}",
            transform=axes[ri, 0].transAxes,
            fontsize=10,
            fontweight="bold",
            va="center",
            ha="right",
        )

    fig.subplots_adjust(left=0.10, right=0.96, top=0.90, bottom=0.10, wspace=0.28, hspace=0.38)
    _set_column_header_labels(fig, axes, distance_columns_cm)
    _add_polar_grid_legend(fig, axes)

    fig.suptitle(title, fontsize=14, fontweight="bold", y=0.98)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)

    cells_ser = {
        f"{size}_angle_{ang}_dist_{d}": (
            None if v is None else f"{v['scene']}/pair_{v['pair_id']}"
        )
        for (size, ang, d), v in cells.items()
    }
    return {
        "cells": cells,
        "cells_serial": cells_ser,
        "panel_stats": all_stats,
        "distance_columns_cm": distance_columns_cm,
        "rows": [{"size": s, "angle_deg": a} for s, a in CARDBOARD_SML_ROWS],
        "angular_tol_deg": angular_tol_deg,
        "distance_tol_cm": distance_tol_cm,
        "bearing_mode": bearing_mode,
        "width_method": "lidar_range_profile_edges",
        "coplanar_tol_cm": coplanar_tol_m * 100.0,
    }


def render_setup_grid(
    data_root: Path,
    scenes: list[str],
    out_path: Path,
    *,
    distance_columns_cm: list[float] | None = None,
    target_label: str = "Target",
    title: str = "LiDAR setup grid",
    distance_tol_cm: float = 5.0,
    angular_tol_deg: float = 15.0,
    bearing_mode: str = "setup",
    coplanar_tol_m: float = 0.02,
) -> dict:
    distance_columns_cm = distance_columns_cm or list(DEFAULT_DISTANCE_COLUMNS_CM)
    angles = sorted({placement_angle_for_scene(s) for s in scenes}, key=lambda a: (-a, a))
    cells = index_by_distance_and_angle(data_root, scenes, distance_columns_cm)

    n_rows = len(angles)
    n_cols = len(distance_columns_cm)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.0 * n_cols, 2.0 * n_rows),
        subplot_kw={"projection": "polar"},
    )
    if n_rows == 1:
        axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    all_stats: dict[str, dict] = {}
    for ri, ang in enumerate(angles):
        for ci, d_cm in enumerate(distance_columns_cm):
            ax = axes[ri, ci]
            cap = cells.get((ang, float(d_cm)))
            if cap is None:
                ax.axis("off")
                continue
            key = f"{cap['scene']}/pair_{cap['pair_id']}"
            all_stats[key] = plot_polar_cell(
                ax,
                cap,
                target_label=target_label,
                distance_tol_m=distance_tol_cm / 100.0,
                angular_tol_deg=angular_tol_deg,
                bearing_mode=bearing_mode,
                coplanar_tol_m=coplanar_tol_m,
            )

    for ri, ang in enumerate(angles):
        label = f"{int(ang)}°" if ang == int(ang) else f"{ang:.0f}°"
        axes[ri, 0].text(
            -0.35,
            0.5,
            label,
            transform=axes[ri, 0].transAxes,
            fontsize=11,
            fontweight="bold",
            va="center",
            ha="right",
        )

    fig.subplots_adjust(left=0.08, right=0.96, top=0.88, bottom=0.10, wspace=0.25, hspace=0.35)
    _set_column_header_labels(fig, axes, distance_columns_cm)
    _add_polar_grid_legend(fig, axes)

    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.15, facecolor="white")
    plt.close(fig)
    return {
        "cells": cells,
        "panel_stats": all_stats,
        "distance_columns_cm": distance_columns_cm,
        "angles_deg": angles,
        "angular_tol_deg": angular_tol_deg,
        "distance_tol_cm": distance_tol_cm,
        "bearing_mode": bearing_mode,
        "width_method": "lidar_range_profile_edges",
        "coplanar_tol_cm": coplanar_tol_m * 100.0,
    }
