"""Stereo ROI−flap depth vs catalog GT depth (cardboard S / M / L)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from evaluation.box_volume_heuristic import box_size_class, gt_nominal_depth_cm, placement_angle_deg
from evaluation.lidar_width_vs_gt import ANGLE_ORDER, SCENE_BY_SIZE_ANGLE, SIZE_ORDER
from evaluation.roi_error_vs_distance import METHOD_COLORS, METHOD_LABELS, METHOD_ORDER
from evaluation.volume_lidar_dashboard import CARDBOARD_SCENES, _norm_pair_id, _to_float

GT_DEPTH_STYLE = {"color": "#333333", "linestyle": ":", "linewidth": 1.4}

METHOD_LINE_STYLE = {
    m: {
        "color": METHOD_COLORS[m],
        "linestyle": "-",
        "marker": "o",
        "linewidth": 1.4,
        "markersize": 5,
    }
    for m in METHOD_ORDER
}


def collect_depth_points(runs_root: Path) -> dict[str, list[dict]]:
    """Per scene: ruler distance and per-method depth_cm from volume estimates CSV."""
    out: dict[str, list[dict]] = {}
    for scene in CARDBOARD_SCENES:
        csv_path = runs_root / scene / "roi_bbox_volume_estimates.csv"
        if not csv_path.is_file():
            continue
        size = box_size_class(scene)
        if size is None:
            continue
        gt_d = gt_nominal_depth_cm(size)
        pts: list[dict] = []
        with open(csv_path, newline="") as f:
            for r in csv.DictReader(f):
                dist = _to_float(r.get("ruler_distance_cm"))
                if dist is None:
                    continue
                row = {
                    "pair_id": _norm_pair_id(r.get("pair_id", "")),
                    "ruler_cm": dist,
                    "gt_depth_cm": gt_d,
                    "size_class": size,
                    "placement_angle_deg": placement_angle_deg(scene),
                }
                for method in METHOD_ORDER:
                    d = _to_float(r.get(f"{method}_depth_cm"))
                    row[f"{method}_depth_cm"] = d
                    row[f"{method}_depth_error_cm"] = (d - gt_d) if d is not None else None
                pts.append(row)
        pts.sort(key=lambda p: p["ruler_cm"])
        out[scene] = pts
    return out


def _row_ylim_cm(size: str, scene_points: dict[str, list[dict]]) -> tuple[float, float]:
    vals: list[float] = [gt_nominal_depth_cm(size)]
    for ang in ANGLE_ORDER:
        scene = SCENE_BY_SIZE_ANGLE.get((size, ang))
        if scene is None:
            continue
        for p in scene_points.get(scene, []):
            for method in METHOD_ORDER:
                d = p.get(f"{method}_depth_cm")
                if d is not None and d > 0:
                    vals.append(float(d))
    if not vals:
        return 0.0, 20.0
    ymax = max(vals)
    pad = max(1.0, 0.15 * ymax)
    return 0.0, ymax + pad


def render_depth_vs_gt_chart(
    scene_points: dict[str, list[dict]],
    out_path: Path,
    *,
    title: str = "Box depth vs ruler distance",
) -> None:
    n_rows = len(SIZE_ORDER)
    n_cols = len(ANGLE_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10.5, 8.5), sharex=True, sharey="row")
    if n_rows == 1:
        axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    for ri, size in enumerate(SIZE_ORDER):
        ylo, yhi = _row_ylim_cm(size, scene_points)
        gt_d = gt_nominal_depth_cm(size)
        for ci, ang in enumerate(ANGLE_ORDER):
            ax = axes[ri, ci]
            scene = SCENE_BY_SIZE_ANGLE.get((size, ang))
            if scene is None:
                ax.axis("off")
                continue
            pts = scene_points.get(scene, [])
            ax.axhline(gt_d, zorder=1, label="GT depth" if ri == 0 and ci == 0 else None, **GT_DEPTH_STYLE)

            for method in METHOD_ORDER:
                series = [
                    (p["ruler_cm"], p[f"{method}_depth_cm"])
                    for p in pts
                    if p.get(f"{method}_depth_cm") is not None and p[f"{method}_depth_cm"] > 0
                ]
                if not series:
                    continue
                x, y = zip(*series)
                ax.plot(
                    x,
                    y,
                    zorder=3,
                    label=METHOD_LABELS[method] if ri == 0 and ci == 0 else None,
                    **METHOD_LINE_STYLE[method],
                )

            ax.set_ylim(ylo, yhi)
            ax.grid(True, alpha=0.35)
            if ri == 0:
                ang_lbl = f"{int(ang)}°" if ang == int(ang) else f"{ang:.0f}°"
                ax.set_title(f"Placement {ang_lbl}", fontsize=11, fontweight="bold")
            if ci == 0:
                ax.set_ylabel(f"{size} box\nDepth (cm)", fontsize=10)
            if ri == n_rows - 1:
                ax.set_xlabel("Ruler distance (cm)", fontsize=10)

    handles = [
        plt.Line2D([0], [0], label="GT depth", **GT_DEPTH_STYLE),
    ]
    for method in METHOD_ORDER:
        handles.append(
            plt.Line2D([0], [0], label=METHOD_LABELS[method], **METHOD_LINE_STYLE[method])
        )
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=3,
        fontsize=8,
        frameon=True,
    )
    fig.suptitle(title, fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.14, hspace=0.32, wspace=0.22)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", pad_inches=0.12, facecolor="white")
    plt.close(fig)


def write_summary_json(scene_points: dict[str, list[dict]], out_path: Path) -> None:
    payload = {
        "scenes": CARDBOARD_SCENES,
        "gt_depth_cm": {s: gt_nominal_depth_cm(s) for s in SIZE_ORDER},
        "gt_depth_formula": "gt_volume_cm3 / (width_nominal_cm × height_cm)",
        "depth_heuristic": "median(ROI depth) − median(consistent outside flap ring)",
        "methods": list(METHOD_ORDER),
        "points_by_scene": scene_points,
    }
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n")
