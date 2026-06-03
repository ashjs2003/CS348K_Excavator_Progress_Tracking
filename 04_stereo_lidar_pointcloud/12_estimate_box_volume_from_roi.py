"""
Estimate cardboard box volume (S / M / L) from ROI depth + setup priors.

Heuristic:
- Depth: median Z inside ROI − Z on consistent outside flap ring (small depth band).
- Height: fixed per size class (S=7, M=19, L=24 cm).
- Width: nominal face width × cos(placement angle); optional LiDAR span at ruler distance.
- Ruler distance: data/<scene>/pair_*.txt (cm). Placement angle: 0° or 30° from scene name.
- LiDAR: data/<scene>/pair_*_lidar.csv (angle_degrees, distance_meters).

Outputs per scene:
- outputs/runs/<scene>/roi_bbox_volume_estimates.csv
- outputs/runs/<scene>/roi_bbox_volume_table.png
- outputs/runs/<scene>/roi_bbox_volume_estimates.json
"""

from __future__ import annotations

import argparse
import csv
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
    DEFAULT_OUTSIDE_CLOSER_MARGIN_M,
    DEFAULT_OUTSIDE_CONSISTENCY_M,
    box_size_class,
    estimate_box_volume,
    placement_angle_deg,
    resolve_lidar_csv,
    resolve_ruler_m,
)
from evaluation.depth_maps import discover_methods, load_metric_depth, load_or_compute_stereo_geometry
from evaluation.roi_gt_compare import METHOD_CHART_LABELS, METHOD_ORDER, load_roi_polygon, polygon_mask
from output_runs import RUNS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description="Estimate S/M/L box volume (ROI depth + setup width/height)")
    p.add_argument(
        "--scenes",
        nargs="+",
        default=[
            "L_carboard_box",
            "L_cardboard_box_30",
            "M_cardboard_box",
            "M_cardboardbox_30",
            "S_cardboard_box",
            "S_cardboard_box_30",
        ],
        help="Scene folders under outputs/runs/",
    )
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--data-root", type=Path, default=_REPO_ROOT / "data")
    p.add_argument("--ring-pad-px", type=int, default=24)
    p.add_argument("--ring-inner-gap-px", type=int, default=4)
    p.add_argument(
        "--outside-closer-margin-cm",
        type=float,
        default=DEFAULT_OUTSIDE_CLOSER_MARGIN_M * 100.0,
        help="Outside ring must be at least this much closer than ROI median (flaps)",
    )
    p.add_argument(
        "--outside-consistency-cm",
        type=float,
        default=DEFAULT_OUTSIDE_CONSISTENCY_M * 100.0,
        help="Keep outside pixels within this depth of flap seed median",
    )
    p.add_argument(
        "--use-lidar-width",
        action="store_true",
        help="LiDAR width: range-profile edges at ruler+placement gate (no catalog width)",
    )
    return p.parse_args()


def bbox_from_polygon(poly_xy: np.ndarray) -> tuple[int, int, int, int]:
    x0 = int(np.min(poly_xy[:, 0]))
    x1 = int(np.max(poly_xy[:, 0]))
    y0 = int(np.min(poly_xy[:, 1]))
    y1 = int(np.max(poly_xy[:, 1]))
    return x0, y0, x1, y1


def rectangle_mask(h: int, w: int, x0: int, y0: int, x1: int, y1: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    if x1 >= x0 and y1 >= y0:
        m[y0 : y1 + 1, x0 : x1 + 1] = True
    return m


def outside_ring_mask(
    h: int,
    w: int,
    x0: int,
    y0: int,
    x1: int,
    y1: int,
    *,
    pad_px: int,
    inner_gap_px: int,
) -> np.ndarray:
    ox0 = max(0, x0 - pad_px)
    oy0 = max(0, y0 - pad_px)
    ox1 = min(w - 1, x1 + pad_px)
    oy1 = min(h - 1, y1 + pad_px)
    outer = rectangle_mask(h, w, ox0, oy0, ox1, oy1)

    ix0 = max(0, x0 - inner_gap_px)
    iy0 = max(0, y0 - inner_gap_px)
    ix1 = min(w - 1, x1 + inner_gap_px)
    iy1 = min(h - 1, y1 + inner_gap_px)
    inner = rectangle_mask(h, w, ix0, iy0, ix1, iy1)
    return outer & ~inner


def collect_scene_rows(
    scene: str,
    runs_root: Path,
    data_root: Path,
    *,
    ring_pad_px: int,
    ring_inner_gap_px: int,
    outside_closer_margin_m: float,
    consistency_tol_m: float,
    use_lidar_span: bool,
) -> list[dict]:
    scene_dir = Path(runs_root) / scene
    if not scene_dir.is_dir():
        return []

    size = box_size_class(scene)
    view_deg = placement_angle_deg(scene)
    gt_vol = BOX_SPECS[size]["gt_volume_cm3"] if size else None

    rows: list[dict] = []
    for run_dir in sorted(scene_dir.glob("pair_*")):
        pair_id = run_dir.name.replace("pair_", "")
        roi_path = Path(data_root) / scene / f"pair_{pair_id}_roi.json"
        if not roi_path.is_file():
            continue
        depth_dir = run_dir / "depth"
        if not depth_dir.is_dir():
            continue

        roi = load_roi_polygon(roi_path)
        poly = np.asarray(roi["polygon_xy"], dtype=np.int32)
        geometry = load_or_compute_stereo_geometry(
            depth_dir, run_dir / "capture" / "rgb1.png", run_dir / "capture" / "rgb2.png"
        )
        w_img, h_img = geometry["image_size"]
        roi_mask = polygon_mask((h_img, w_img), poly)
        x0, y0, x1, y1 = bbox_from_polygon(poly)

        ruler_m = resolve_ruler_m(data_root, scene, pair_id)
        lidar_csv = resolve_lidar_csv(data_root, scene, pair_id)
        methods = discover_methods(depth_dir)
        if not methods:
            continue

        row = {
            "scene": scene,
            "pair_id": pair_id,
            "size_class": size,
            "placement_angle_deg": view_deg,
            "gt_volume_cm3": gt_vol,
            "ruler_distance_cm": None if ruler_m is None else ruler_m * 100.0,
            "height_cm": BOX_SPECS[size]["height_cm"] if size else None,
        }
        for method in methods:
            try:
                depth = load_metric_depth(depth_dir, method, geometry)
            except Exception:
                depth = None
            if depth is None:
                continue
            dh, dw = depth.shape[:2]
            if (dh, dw) != (h_img, w_img):
                scale_x = dw / float(w_img)
                scale_y = dh / float(h_img)
                poly_s = poly.astype(np.float64).copy()
                poly_s[:, 0] *= scale_x
                poly_s[:, 1] *= scale_y
                roi_m = polygon_mask((dh, dw), poly_s.astype(np.int32))
                x0s = int(x0 * scale_x)
                x1s = int(x1 * scale_x)
                y0s = int(y0 * scale_y)
                y1s = int(y1 * scale_y)
                ring_m = outside_ring_mask(
                    dh, dw, x0s, y0s, x1s, y1s, pad_px=ring_pad_px, inner_gap_px=ring_inner_gap_px
                )
            else:
                roi_m = roi_mask
                ring_m = outside_ring_mask(
                    h_img, w_img, x0, y0, x1, y1, pad_px=ring_pad_px, inner_gap_px=ring_inner_gap_px
                )
            est = estimate_box_volume(
                depth,
                roi_m,
                ring_m,
                scene,
                ruler_m=ruler_m,
                lidar_csv=lidar_csv,
                use_lidar_span=use_lidar_span,
                outside_closer_margin_m=outside_closer_margin_m,
                consistency_tol_m=consistency_tol_m,
            )
            if est is not None:
                row[method] = est
        if any(m in row for m in methods):
            rows.append(row)

    rows.sort(
        key=lambda r: (
            float("inf") if r["ruler_distance_cm"] is None else r["ruler_distance_cm"],
            r["pair_id"],
        )
    )
    return rows


def write_scene_csv(scene_dir: Path, rows: list[dict]) -> Path:
    out_csv = scene_dir / "roi_bbox_volume_estimates.csv"
    methods = list(METHOD_ORDER)
    fields = [
        "pair_id",
        "ruler_distance_cm",
        "placement_angle_deg",
        "height_cm",
        "gt_volume_cm3",
    ]
    for m in methods:
        fields += [
            f"{m}_volume_cm3",
            f"{m}_depth_cm",
            f"{m}_width_cm",
            f"{m}_width_source",
            f"{m}_outside_mode",
        ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = {
                "pair_id": r["pair_id"],
                "ruler_distance_cm": "" if r["ruler_distance_cm"] is None else f"{r['ruler_distance_cm']:.1f}",
                "placement_angle_deg": f"{r.get('placement_angle_deg', '')}",
                "height_cm": "" if r.get("height_cm") is None else f"{r['height_cm']:.1f}",
                "gt_volume_cm3": "" if r.get("gt_volume_cm3") is None else f"{r['gt_volume_cm3']:.0f}",
            }
            for m in methods:
                est = r.get(m)
                if est:
                    row[f"{m}_volume_cm3"] = f"{est['volume_cm3']:.1f}"
                    row[f"{m}_depth_cm"] = f"{est['depth_m'] * 100.0:.2f}"
                    row[f"{m}_width_cm"] = f"{est['width_cm']:.2f}"
                    row[f"{m}_width_source"] = est.get("width_source", "")
                    row[f"{m}_outside_mode"] = est.get("outside_mode", "")
                else:
                    for k in fields:
                        if k.startswith(f"{m}_"):
                            row[k] = ""
            writer.writerow(row)
    return out_csv


def render_scene_table_png(scene_dir: Path, scene: str, rows: list[dict]) -> Path | None:
    methods = [m for m in METHOD_ORDER if any(m in r for r in rows)]
    if not methods or not rows:
        return None

    gt = rows[0].get("gt_volume_cm3")
    col_labels = ["Pair", "Ruler cm"] + [f"{METHOD_CHART_LABELS.get(m, m)} vol" for m in methods]
    cell_text = []
    for r in rows:
        one = [r["pair_id"], "n/a" if r["ruler_distance_cm"] is None else f"{r['ruler_distance_cm']:.0f}"]
        for m in methods:
            est = r.get(m)
            one.append("—" if not est else f"{est['volume_cm3']:,.0f}")
        cell_text.append(one)

    fig_h = 1.4 + 0.35 * len(cell_text)
    fig_w = max(8.0, 2.2 + 1.8 * len(col_labels))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    gt_s = f" · GT {gt:,.0f} cm³" if gt else ""
    fig.suptitle(f"{scene}: estimated volume{gt_s}", fontsize=12, fontweight="bold", y=0.98)
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
    h_cm = rows[0].get("height_cm")
    ang = rows[0].get("placement_angle_deg")
    ax.text(
        0.5,
        0.02,
        f"V = width(setup) × height({h_cm:.0f} cm) × depth(ROI−flap); placement {ang:.0f}°",
        ha="center",
        transform=ax.transAxes,
        fontsize=8,
        color="#666",
    )
    out_png = scene_dir / "roi_bbox_volume_table.png"
    fig.tight_layout(rect=[0, 0.05, 1, 0.95])
    fig.savefig(out_png, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_png


def write_scene_json(scene_dir: Path, scene: str, rows: list[dict], args) -> Path:
    out_json = scene_dir / "roi_bbox_volume_estimates.json"
    size = box_size_class(scene)
    payload = {
        "scene": scene,
        "size_class": size,
        "box_specs": BOX_SPECS.get(size) if size else None,
        "placement_angle_deg": placement_angle_deg(scene),
        "assumptions": {
            "depth_m": "median(ROI) − median(consistent outside flap ring)",
            "height_m": "fixed manual height per S/M/L",
            "width_m": "nominal_face_width × cos(placement°); optional LiDAR span at ruler distance",
            "ruler_m": "data/<scene>/pair_*.txt",
            "lidar": "data/<scene>/pair_*_lidar.csv",
        },
        "params": {
            "ring_pad_px": args.ring_pad_px,
            "ring_inner_gap_px": args.ring_inner_gap_px,
            "outside_closer_margin_cm": args.outside_closer_margin_cm,
            "outside_consistency_cm": args.outside_consistency_cm,
        },
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2, default=_json_default) + "\n")
    return out_json


def _json_default(obj):
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    raise TypeError(type(obj))


def main():
    args = parse_args()
    runs_root = Path(args.runs_root)
    data_root = Path(args.data_root)
    ok = 0
    for scene in args.scenes:
        scene_dir = runs_root / scene
        rows = collect_scene_rows(
            scene,
            runs_root,
            data_root,
            ring_pad_px=args.ring_pad_px,
            ring_inner_gap_px=args.ring_inner_gap_px,
            outside_closer_margin_m=args.outside_closer_margin_cm / 100.0,
            consistency_tol_m=args.outside_consistency_cm / 100.0,
            use_lidar_span=args.use_lidar_width,
        )
        if not rows:
            print(f"SKIP {scene}: no ROI runs found")
            continue
        csv_path = write_scene_csv(scene_dir, rows)
        png_path = render_scene_table_png(scene_dir, scene, rows)
        json_path = write_scene_json(scene_dir, scene, rows, args)
        ok += 1
        print(f"OK {scene}: {len(rows)} pairs -> {csv_path.name}, {json_path.name}, {png_path.name if png_path else '(no png)'}")

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
