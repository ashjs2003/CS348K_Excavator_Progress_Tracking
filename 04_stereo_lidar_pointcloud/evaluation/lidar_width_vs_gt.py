"""LiDAR edge-profile width vs ruler GT distance (cardboard S / M / L)."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.box_volume_heuristic import BOX_SPECS, box_size_class, placement_angle_deg
from evaluation.lidar_edge_profile_width import width_from_lidar_edge_profile
from evaluation.lidar_polar_setup_grid import CARDBOARD_SML_SCENES, discover_scene_captures

SIZE_ORDER = ["S", "M", "L"]

LIDAR_MEASURED_STYLE = {
    "color": "#1f77b4",
    "linestyle": "-",
    "marker": "o",
    "linewidth": 1.5,
    "markersize": 6,
}
GT_WIDTH_STYLE = {"color": "#333333", "linestyle": ":", "linewidth": 1.4}
ANGLE_ORDER = [0.0, -30.0]

SCENE_BY_SIZE_ANGLE: dict[tuple[str, float], str] = {
    ("S", 0.0): "S_cardboard_box",
    ("S", -30.0): "S_cardboard_box_30",
    ("M", 0.0): "M_cardboard_box",
    ("M", -30.0): "M_cardboardbox_30",
    ("L", 0.0): "L_carboard_box",
    ("L", -30.0): "L_cardboard_box_30",
}


def gt_nominal_width_cm(size: str, angle_deg: float | None = None) -> float:
    """Physical catalog face width (cm). Same at 0° and −30°; angle is not used."""
    del angle_deg  # placement changes pose, not box width
    return float(BOX_SPECS[size]["width_nominal_cm"])


def collect_lidar_width_points(data_root: Path) -> dict[str, list[dict]]:
    """Per scene: sorted points with ruler_cm, width_lidar_cm, gt_width_cm."""
    out: dict[str, list[dict]] = {}
    for scene in CARDBOARD_SML_SCENES:
        pts: list[dict] = []
        size = box_size_class(scene)
        if size is None:
            continue
        ang = placement_angle_deg(scene)
        gt_w = gt_nominal_width_cm(size, ang)
        for cap in discover_scene_captures(data_root, scene):
            lw = width_from_lidar_edge_profile(
                cap["lidar_path"],
                cap["gt_m"],
                cap["placement_angle_deg"],
                scene=scene,
                size_class=size,
            )
            width_cm = float(lw["width_cm"]) if lw is not None else None
            err_cm = (width_cm - gt_w) if width_cm is not None else None
            pts.append(
                {
                    "pair_id": cap["pair_id"],
                    "ruler_cm": float(cap["gt_cm"]),
                    "width_lidar_cm": width_cm,
                    "gt_width_cm": gt_w,
                    "width_error_cm": err_cm,
                    "placement_angle_deg": ang,
                    "size_class": size,
                    "gate_bearing_mode": lw.get("gate_bearing_mode") if lw else None,
                    "edge_method_detail": lw.get("edge_method_detail") if lw else None,
                }
            )
        pts.sort(key=lambda p: p["ruler_cm"])
        out[scene] = pts
    return out


def _row_ylim_cm(size: str, scene_points: dict[str, list[dict]]) -> tuple[float, float]:
    """Shared y limits for both placement columns of one box size."""
    vals: list[float] = []
    for ang in ANGLE_ORDER:
        scene = SCENE_BY_SIZE_ANGLE.get((size, ang))
        if scene is None:
            continue
        vals.append(gt_nominal_width_cm(size, ang))
        for p in scene_points.get(scene, []):
            if p["width_lidar_cm"] is not None:
                vals.append(float(p["width_lidar_cm"]))
    if not vals:
        return 0.0, 10.0
    ymax = max(vals)
    pad = max(1.5, 0.12 * ymax)
    return 0.0, ymax + pad


def render_width_vs_gt_chart(
    scene_points: dict[str, list[dict]],
    out_path: Path,
    *,
    title: str = "LiDAR width vs ruler distance (cardboard S / M / L)",
) -> None:
    n_rows = len(SIZE_ORDER)
    n_cols = len(ANGLE_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(9.5, 8.5), sharex=True, sharey="row")
    if n_rows == 1:
        axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for ri, size in enumerate(SIZE_ORDER):
        ylo, yhi = _row_ylim_cm(size, scene_points)
        for ci, ang in enumerate(ANGLE_ORDER):
            ax = axes[ri, ci]
            scene = SCENE_BY_SIZE_ANGLE.get((size, ang))
            if scene is None:
                ax.axis("off")
                continue
            pts = scene_points.get(scene, [])
            gt_w = gt_nominal_width_cm(size, ang)
            ax.axhline(gt_w, zorder=1, **GT_WIDTH_STYLE)

            measured = [(p["ruler_cm"], p["width_lidar_cm"]) for p in pts if p["width_lidar_cm"] is not None]
            if measured:
                x, y = zip(*measured)
                ax.plot(x, y, zorder=3, **LIDAR_MEASURED_STYLE)

            ax.set_ylim(ylo, yhi)
            ax.grid(True, alpha=0.35)
            if ri == 0:
                ang_lbl = f"{int(ang)}°" if ang == int(ang) else f"{ang:.0f}°"
                ax.set_title(f"Placement {ang_lbl}", fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{size} box\nWidth (cm)", fontsize=10)
            if ri == n_rows - 1:
                ax.set_xlabel("Ruler distance (cm)", fontsize=10)

    handles = [
        plt.Line2D([0], [0], label="GT width (catalog face)", **GT_WIDTH_STYLE),
        plt.Line2D([0], [0], label="LiDAR measured", **LIDAR_MEASURED_STYLE),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=2,
        fontsize=9,
        frameon=True,
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10, hspace=0.32, wspace=0.22)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


def write_summary_json(scene_points: dict[str, list[dict]], out_path: Path) -> None:
    payload = {
        "scenes": CARDBOARD_SML_SCENES,
        "scene_by_size_angle": {f"{s}_{a}": sc for (s, a), sc in SCENE_BY_SIZE_ANGLE.items()},
        "gt_width_formula": "BOX_SPECS[width_nominal_cm] (physical face; same at all placement angles)",
        "points_by_scene": scene_points,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
