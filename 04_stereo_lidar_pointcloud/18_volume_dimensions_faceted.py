"""
Faceted chart: volume-heuristic dimensions vs ruler distance (cardboard S/M/L).

Layout (like roi_error_by_method_faceted):
  - Columns: OpenCV, DA-V2, DA-V2 GT, Foundation
  - Rows: width (setup), depth (ROI−flap), height (fixed)
  - Lines: one per cardboard scene (0° and 30°)
  - X-axis: ruler distance from pair_*.txt (cm)

Also writes optional LiDAR angular-span width row when pair_*_lidar.csv exists.

Output:
  outputs/runs/_combined/volume_dimensions_by_method.png
  outputs/runs/_combined/volume_dimensions_by_method.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.box_volume_heuristic import (
    BOX_SPECS,
    placement_angle_deg,
    resolve_lidar_csv,
    resolve_ruler_m,
    width_from_lidar_span,
)
from evaluation.roi_error_vs_distance import METHOD_LABELS, METHOD_ORDER, scene_plot_style
from output_runs import RUNS_ROOT

CARD_BOX_SCENES = [
    "L_carboard_box",
    "L_cardboard_box_30",
    "M_cardboard_box",
    "M_cardboardbox_30",
    "S_cardboard_box",
    "S_cardboard_box_30",
]

DIM_ROWS = [
    ("width_setup_cm", "Width — setup (nominal × cos θ, ruler scale)", "width_cm"),
    ("depth_cm", "Depth — median(ROI) − flap ring", None),
    ("height_cm", "Height — fixed manual (S/M/L)", "height_cm"),
]

LIDAR_ROW = ("width_lidar_cm", "Width — LiDAR span @ ruler distance", None)


def parse_args():
    p = argparse.ArgumentParser(description="Faceted volume dimension chart (cardboard boxes)")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--data-root", type=Path, default=_REPO_ROOT / "data")
    p.add_argument("--include-lidar-width-row", action="store_true", default=True)
    p.add_argument("--no-lidar-width-row", action="store_false", dest="include_lidar_width_row")
    return p.parse_args()


def gt_nominal_depth_cm(scene: str) -> float | None:
    """Catalog depth edge (cm) for dashed reference — not used in heuristic depth."""
    size = scene[0] if scene else None
    if size == "L":
        return 16.0
    if size == "M":
        return 10.0
    if size == "S":
        return 6.0
    return None


def gt_nominal_width_cm(scene: str) -> float:
    size = scene[0]
    spec = BOX_SPECS.get(size, {})
    w = spec.get("width_nominal_cm", 0.0)
    return w * float(np.cos(np.deg2rad(placement_angle_deg(scene))))


def load_scene_volume_rows(runs_root: Path, scene: str) -> list[dict]:
    path = runs_root / scene / "roi_bbox_volume_estimates.json"
    if not path.is_file():
        return []
    payload = json.loads(path.read_text())
    return payload.get("rows", [])


def collect_points(
    runs_root: Path,
    data_root: Path,
    *,
    include_lidar_row: bool,
) -> dict:
    """
    Return nested structure:
      methods[method][scene] -> list of {ruler_cm, width_cm, depth_cm, height_cm, width_lidar_cm?}
    """
    out: dict[str, dict[str, list[dict]]] = {m: {} for m in METHOD_ORDER}

    for scene in CARD_BOX_SCENES:
        rows = load_scene_volume_rows(runs_root, scene)
        if not rows:
            continue
        for row in rows:
            pair_id = row.get("pair_id", "")
            ruler_cm = row.get("ruler_distance_cm")
            if ruler_cm is None:
                continue
            ruler_m = resolve_ruler_m(data_root, scene, pair_id)
            lidar_w = None
            if include_lidar_row and ruler_m is not None:
                lc = resolve_lidar_csv(data_root, scene, pair_id)
                if lc is not None:
                    lw = width_from_lidar_span(
                        lc,
                        ruler_m,
                        placement_angle_deg=placement_angle_deg(scene),
                        bearing_mode="setup",
                    )
                    if lw is not None:
                        lidar_w = lw["width_cm"]

            for method in METHOD_ORDER:
                est = row.get(method)
                if not est:
                    continue
                pt = {
                    "pair_id": pair_id,
                    "ruler_cm": float(ruler_cm),
                    "width_cm": float(est.get("width_cm", np.nan)),
                    "depth_cm": float(est.get("depth_m", np.nan) * 100.0),
                    "height_cm": float(est.get("height_cm", est.get("height_m", np.nan) * 100.0)),
                    "width_source": est.get("width_source", ""),
                    "width_lidar_cm": lidar_w,
                }
                out[method].setdefault(scene, []).append(pt)

    for method in out:
        for scene in out[method]:
            out[method][scene].sort(key=lambda p: p["ruler_cm"])
    return out


def _plot_scene_curves(ax, scene: str, points: list[dict], y_key: str) -> None:
    if not points:
        return
    x = [p["ruler_cm"] for p in points]
    y = [p.get(y_key) for p in points]
    if any(v is None or (isinstance(v, float) and not np.isfinite(v)) for v in y):
        y = [np.nan if v is None else v for v in y]
    st = scene_plot_style(scene)
    ax.plot(
        x,
        y,
        color=st["color"],
        linestyle=st["linestyle"],
        marker=st["marker"],
        linewidth=1.2,
        markersize=4.5,
    )


def render_faceted_chart(
    data: dict,
    out_path: Path,
    *,
    include_lidar_row: bool,
) -> None:
    row_specs = list(DIM_ROWS)
    if include_lidar_row:
        row_specs.append(LIDAR_ROW)

    n_rows = len(row_specs)
    n_cols = len(METHOD_ORDER)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(14.5, 3.2 * n_rows), sharex="col")
    if n_rows == 1:
        axes = np.atleast_2d(axes)
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    scenes_present = [s for s in CARD_BOX_SCENES if any(s in data.get(m, {}) for m in METHOD_ORDER)]

    for col, method in enumerate(METHOD_ORDER):
        scene_dict = data.get(method, {})
        for row_idx, (y_key, y_label, est_key) in enumerate(row_specs):
            ax = axes[row_idx, col]
            key = est_key or y_key
            for scene in scenes_present:
                pts = scene_dict.get(scene, [])
                if not pts:
                    continue
                _plot_scene_curves(ax, scene, pts, key)

            # GT reference on width / depth rows
            if y_key == "width_setup_cm" and scenes_present:
                for scene in scenes_present:
                    gt_w = gt_nominal_width_cm(scene)
                    ax.axhline(gt_w, color=scene_plot_style(scene)["color"], alpha=0.2, linewidth=0.8, linestyle=":")
            if y_key == "depth_cm" and scenes_present:
                for scene in scenes_present:
                    gt_d = gt_nominal_depth_cm(scene)
                    if gt_d is not None:
                        ax.axhline(gt_d, color=scene_plot_style(scene)["color"], alpha=0.2, linewidth=0.8, linestyle=":")
            if y_key == "height_cm" and scenes_present:
                for scene in scenes_present:
                    size = scene[0]
                    h = BOX_SPECS.get(size, {}).get("height_cm")
                    if h:
                        ax.axhline(h, color=scene_plot_style(scene)["color"], alpha=0.2, linewidth=0.8, linestyle=":")

            if row_idx == 0:
                ax.set_title(METHOD_LABELS.get(method, method), fontsize=11, fontweight="bold")
            if col == 0:
                ax.set_ylabel(y_label, fontsize=8)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(bottom=0)

    for col in range(n_cols):
        axes[-1, col].set_xlabel("Ruler distance (pair_*.txt, cm)", fontsize=10)

    handles = []
    for s in scenes_present:
        st = scene_plot_style(s)
        handles.append(
            plt.Line2D(
                [0],
                [0],
                color=st["color"],
                marker=st["marker"],
                linestyle=st["linestyle"],
                linewidth=1.2,
                markersize=4.5,
                label=s,
            )
        )
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=min(len(handles), 6),
        fontsize=7.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.suptitle(
        "Volume heuristic dimensions vs ruler distance (cardboard S / M / L)",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )
    fig.text(
        0.5,
        0.01,
        "Dotted horizontal: catalog GT edge (width×cos θ, depth, height). LiDAR row = arc span at ruler range.",
        ha="center",
        fontsize=8,
        color="#555",
    )
    fig.subplots_adjust(left=0.08, right=0.98, top=0.92, bottom=0.14, hspace=0.35, wspace=0.12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    data_root = Path(args.data_root)
    data = collect_points(
        runs_root,
        data_root,
        include_lidar_row=args.include_lidar_width_row,
    )

    if not any(data[m] for m in METHOD_ORDER):
        print("No roi_bbox_volume_estimates.json found for cardboard scenes. Run 12_estimate_box_volume_from_roi.py first.")
        return 1

    out_dir = runs_root / "_combined"
    png = out_dir / "volume_dimensions_by_method.png"
    render_faceted_chart(data, png, include_lidar_row=args.include_lidar_width_row)

    summary = {
        "scenes": CARD_BOX_SCENES,
        "methods": METHOD_ORDER,
        "dimension_rows": [r[0] for r in DIM_ROWS]
        + ([LIDAR_ROW[0]] if args.include_lidar_width_row else []),
        "data": data,
    }
    json_path = out_dir / "volume_dimensions_by_method.json"
    json_path.write_text(json.dumps(summary, indent=2, default=str) + "\n")
    print(f"Wrote {png}")
    print(f"Wrote {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
