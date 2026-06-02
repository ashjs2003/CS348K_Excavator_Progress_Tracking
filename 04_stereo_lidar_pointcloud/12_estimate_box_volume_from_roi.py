"""
Estimate cardboard box volume from existing ROI and nearby outside depth.

Heuristic:
- Z_inside: median depth inside ROI polygon (back-of-box marker area)
- Z_outside: median depth in a ring just outside ROI bbox
- box_depth_m = |Z_outside - Z_inside|
- box_width_m, box_height_m from ROI bbox pixels and intrinsics at Z_inside
- volume_m3 = box_width_m * box_height_m * box_depth_m

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

from evaluation.depth_maps import discover_methods, load_metric_depth, load_or_compute_stereo_geometry
from evaluation.gt_depth_overlay import load_gt_reference_for_run
from evaluation.roi_gt_compare import METHOD_CHART_LABELS, METHOD_ORDER, load_roi_polygon, polygon_mask
from output_runs import RUNS_ROOT


def parse_args():
    p = argparse.ArgumentParser(description="Estimate box volume from ROI + outside depth ring")
    p.add_argument(
        "--scenes",
        nargs="+",
        default=["L_carboard_box", "M_cardboard_box", "S_cardboard_box"],
        help="Scene folders under outputs/runs/",
    )
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--data-root", type=Path, default=_REPO_ROOT / "data")
    p.add_argument("--ring-pad-px", type=int, default=24, help="Expand ROI bbox by this many pixels")
    p.add_argument(
        "--ring-inner-gap-px",
        type=int,
        default=4,
        help="Leave this many pixels gap around ROI bbox before outside ring",
    )
    p.add_argument(
        "--outside-closer-margin-cm",
        type=float,
        default=0.5,
        help="Use outside-ring depths <= Z_inside - margin as front-face candidates",
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


def estimate_for_method(
    depth_m: np.ndarray,
    roi_mask: np.ndarray,
    ring_mask: np.ndarray,
    *,
    bbox_w_px: int,
    bbox_h_px: int,
    fx: float,
    fy: float,
    outside_closer_margin_m: float,
) -> dict | None:
    valid = np.isfinite(depth_m) & (depth_m > 0)
    inside = depth_m[roi_mask & valid]
    outside = depth_m[ring_mask & valid]
    if len(inside) < 8 or len(outside) < 8:
        return None

    z_inside = float(np.median(inside))
    # Prefer outside pixels closer than ROI depth (front face candidate).
    outside_closer = outside[outside <= (z_inside - outside_closer_margin_m)]
    if len(outside_closer) >= 8:
        z_outside = float(np.median(outside_closer))
        outside_mode = "closer_only"
    else:
        z_outside = float(np.median(outside))
        outside_mode = "fallback_all_outside"
    box_depth = abs(z_outside - z_inside)
    width_m = float((bbox_w_px / fx) * z_inside)
    height_m = float((bbox_h_px / fy) * z_inside)
    volume = float(width_m * height_m * box_depth)
    return {
        "z_inside_m": z_inside,
        "z_outside_m": z_outside,
        "depth_m": box_depth,
        "width_m": width_m,
        "height_m": height_m,
        "volume_m3": volume,
        "volume_cm3": volume * 1_000_000.0,
        "outside_mode": outside_mode,
        "outside_candidates_n": int(len(outside_closer)),
        "outside_total_n": int(len(outside)),
    }


def collect_scene_rows(
    scene: str,
    runs_root: Path,
    data_root: Path,
    *,
    ring_pad_px: int,
    ring_inner_gap_px: int,
    outside_closer_margin_m: float,
) -> list[dict]:
    scene_dir = Path(runs_root) / scene
    if not scene_dir.is_dir():
        return []

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
        geometry = load_or_compute_stereo_geometry(depth_dir, run_dir / "capture" / "rgb1.png", run_dir / "capture" / "rgb2.png")
        w_img, h_img = geometry["image_size"]
        roi_mask = polygon_mask((h_img, w_img), poly)
        x0, y0, x1, y1 = bbox_from_polygon(poly)
        bbox_w_px = max(1, x1 - x0 + 1)
        bbox_h_px = max(1, y1 - y0 + 1)
        ring_mask = outside_ring_mask(
            h_img,
            w_img,
            x0,
            y0,
            x1,
            y1,
            pad_px=ring_pad_px,
            inner_gap_px=ring_inner_gap_px,
        )

        fx = float(geometry["P1"][0, 0])
        fy = float(geometry["P1"][1, 1])
        gt = load_gt_reference_for_run(run_dir, _REPO_ROOT)
        methods = discover_methods(depth_dir)
        if not methods:
            continue

        row = {
            "scene": scene,
            "pair_id": pair_id,
            "ruler_distance_cm": gt.get("target_gt_cm"),
            "bbox_w_px": bbox_w_px,
            "bbox_h_px": bbox_h_px,
        }
        for method in methods:
            try:
                depth = load_metric_depth(depth_dir, method, geometry)
            except Exception:
                depth = None
            if depth is None:
                continue
            est = estimate_for_method(
                depth,
                roi_mask,
                ring_mask,
                bbox_w_px=bbox_w_px,
                bbox_h_px=bbox_h_px,
                fx=fx,
                fy=fy,
                outside_closer_margin_m=outside_closer_margin_m,
            )
            if est is None:
                continue
            row[method] = est
        if any(m in row for m in methods):
            rows.append(row)
    rows.sort(key=lambda r: (float("inf") if r["ruler_distance_cm"] is None else r["ruler_distance_cm"], r["pair_id"]))
    return rows


def write_scene_csv(scene_dir: Path, rows: list[dict]) -> Path:
    out_csv = scene_dir / "roi_bbox_volume_estimates.csv"
    methods = list(METHOD_ORDER)
    fields = ["pair_id", "ruler_distance_cm"]
    for m in methods:
        fields += [
            f"{m}_volume_cm3",
            f"{m}_depth_cm",
            f"{m}_w_cm",
            f"{m}_h_cm",
            f"{m}_outside_mode",
            f"{m}_outside_candidates_n",
            f"{m}_outside_total_n",
        ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = {
                "pair_id": r["pair_id"],
                "ruler_distance_cm": "" if r["ruler_distance_cm"] is None else f"{r['ruler_distance_cm']:.1f}",
            }
            for m in methods:
                est = r.get(m)
                if est:
                    row[f"{m}_volume_cm3"] = f"{est['volume_cm3']:.1f}"
                    row[f"{m}_depth_cm"] = f"{est['depth_m'] * 100.0:.2f}"
                    row[f"{m}_w_cm"] = f"{est['width_m'] * 100.0:.2f}"
                    row[f"{m}_h_cm"] = f"{est['height_m'] * 100.0:.2f}"
                    row[f"{m}_outside_mode"] = est.get("outside_mode", "")
                    row[f"{m}_outside_candidates_n"] = str(est.get("outside_candidates_n", ""))
                    row[f"{m}_outside_total_n"] = str(est.get("outside_total_n", ""))
                else:
                    row[f"{m}_volume_cm3"] = ""
                    row[f"{m}_depth_cm"] = ""
                    row[f"{m}_w_cm"] = ""
                    row[f"{m}_h_cm"] = ""
                    row[f"{m}_outside_mode"] = ""
                    row[f"{m}_outside_candidates_n"] = ""
                    row[f"{m}_outside_total_n"] = ""
            writer.writerow(row)
    return out_csv


def render_scene_table_png(scene_dir: Path, scene: str, rows: list[dict]) -> Path | None:
    methods = [m for m in METHOD_ORDER if any(m in r for r in rows)]
    if not methods or not rows:
        return None

    col_labels = ["Pair", "Ruler cm"] + [f"{METHOD_CHART_LABELS.get(m,m)} vol (cm³)" for m in methods]
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
    fig.suptitle(f"{scene}: estimated box volume vs ruler distance", fontsize=12, fontweight="bold", y=0.98)
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, cellLoc="center", loc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.4)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor("#e8e8e8")
            cell.set_text_props(fontweight="bold")
    ax.text(
        0.5,
        0.02,
        "Volume in cm³. Heuristic from ROI bbox + outside depth ring.",
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
    payload = {
        "scene": scene,
        "assumptions": {
            "z_inside": "median depth inside ROI polygon",
            "z_outside": "median outside-ring depth; prefer values closer than z_inside by margin",
            "depth_m": "abs(z_outside - z_inside)",
            "width_height": "ROI bbox pixels projected with fx/fy at z_inside",
        },
        "params": {
            "ring_pad_px": args.ring_pad_px,
            "ring_inner_gap_px": args.ring_inner_gap_px,
            "outside_closer_margin_cm": args.outside_closer_margin_cm,
        },
        "rows": rows,
    }
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    return out_json


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
