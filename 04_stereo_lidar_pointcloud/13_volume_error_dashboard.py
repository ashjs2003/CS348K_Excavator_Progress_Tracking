"""
Create one consolidated volume-error dashboard across box sizes and views.

Outputs:
  outputs/runs/_combined/volume_error_dashboard.png
  outputs/runs/_combined/volume_error_dashboard.csv
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.volume_lidar_dashboard import load_lidar_volume_rows
from output_runs import RUNS_ROOT

METHOD_COLUMNS = [
    ("opencv_volume_cm3", "OpenCV"),
    ("dav2_volume_cm3", "DA-V2"),
    ("dav2_gt_volume_cm3", "DA-V2 GT"),
    ("foundation_volume_cm3", "Foundation"),
]

ROW_ORDER = [
    "L-0deg",
    "L-30deg",
    "M-0deg",
    "M-30deg",
    "S-0deg",
    "S-30deg",
]

ROW_LABELS = {
    "L-0deg": "L 0°",
    "L-30deg": "L 30°",
    "M-0deg": "M 0°",
    "M-30deg": "M 30°",
    "S-0deg": "S 0°",
    "S-30deg": "S 30°",
}


def parse_args():
    p = argparse.ArgumentParser(description="Consolidated volume percent-error dashboard")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--data-root", type=Path, default=_REPO_ROOT / "data")
    p.add_argument("--combined-dir", type=Path, default=None)
    p.add_argument(
        "--error-cap-pct",
        type=float,
        default=100.0,
        help="Symmetric color scale limit (%%); cells beyond show as saturated",
    )
    return p.parse_args()


def _signed_color_limits(grids: dict[str, np.ndarray], cap_pct: float) -> tuple[float, float]:
    """Symmetric limits for diverging map (signed %% error)."""
    vals = np.concatenate([g[np.isfinite(g)].ravel() for g in grids.values() if np.any(np.isfinite(g))])
    if len(vals) == 0:
        return -cap_pct, cap_pct
    p95 = float(np.percentile(np.abs(vals), 95))
    lim = min(float(cap_pct), max(40.0, p95))
    return -lim, lim


def _cell_label(err_pct: float, lim: float) -> str:
    if not np.isfinite(err_pct):
        return ""
    if abs(err_pct) >= lim - 0.5:
        sign = "+" if err_pct > 0 else "−"
        return f"{sign}{lim:.0f}%+"
    sign = "+" if err_pct > 0 else "−"
    return f"{sign}{abs(err_pct):.0f}%"


def _text_color_for_value(val: float, lim: float) -> str:
    """Readable annotation on diverging background."""
    if not np.isfinite(val):
        return "#333"
    t = min(abs(val) / max(lim, 1.0), 1.0)
    if t < 0.35:
        return "#111"
    return "white"


def compute_error_grid(rows: list[dict]):
    distances = sorted({float(r["distance_cm"]) for r in rows})
    if not distances:
        return None, None, None
    row_to_i = {k: i for i, k in enumerate(ROW_ORDER)}
    d_to_j = {d: j for j, d in enumerate(distances)}

    grids: dict[str, np.ndarray] = {}
    for col, _ in METHOD_COLUMNS:
        arr = np.full((len(ROW_ORDER), len(distances)), np.nan, dtype=float)
        for r in rows:
            i = row_to_i[r["row_key"]]
            j = d_to_j[float(r["distance_cm"])]
            est = r.get(col)
            gt = r["gt_volume_cm3"]
            if est is None or gt <= 0:
                continue
            arr[i, j] = 100.0 * (est - gt) / gt
        grids[col] = arr
    return distances, grids, row_to_i


def export_long_csv(rows: list[dict], out_csv: Path) -> None:
    fields = [
        "row_key",
        "scene",
        "pair_id",
        "distance_cm",
        "method",
        "width_lidar_cm",
        "height_cm",
        "depth_cm",
        "gt_volume_cm3",
        "est_volume_cm3",
        "error_pct",
    ]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            gt = r["gt_volume_cm3"]
            for col, label in METHOD_COLUMNS:
                est = r.get(col)
                if est is None:
                    continue
                method_key = col.replace("_volume_cm3", "")
                depth_cm = r.get(f"{method_key}_depth_cm")
                w.writerow(
                    {
                        "row_key": r["row_key"],
                        "scene": r.get("scene", ""),
                        "pair_id": r.get("pair_id", ""),
                        "distance_cm": f"{r['distance_cm']:.1f}",
                        "method": label,
                        "width_lidar_cm": f"{r['width_lidar_cm']:.2f}",
                        "height_cm": f"{r['height_cm']:.1f}",
                        "depth_cm": "" if depth_cm is None else f"{depth_cm:.2f}",
                        "gt_volume_cm3": f"{gt:.1f}",
                        "est_volume_cm3": f"{est:.1f}",
                        "error_pct": f"{(100.0 * (est - gt) / gt):.2f}",
                    }
                )


def render_dashboard(distances, grids, out_png: Path, error_cap_pct: float) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.2, 8.8), squeeze=False)
    axes_flat = axes.ravel()

    vmin, vmax = _signed_color_limits(grids, error_cap_pct)
    norm = TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#e8e8e8")

    im = None
    for ax, (col, label) in zip(axes_flat, METHOD_COLUMNS):
        data = grids[col]
        shown = np.ma.masked_invalid(np.clip(data, vmin, vmax))
        im = ax.imshow(shown, aspect="auto", cmap=cmap, norm=norm)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=10)
        ax.set_xticks(range(len(distances)))
        ax.set_xticklabels([f"{int(d)}" for d in distances], fontsize=10)
        if ax in (axes_flat[0], axes_flat[2]):
            ax.set_yticks(range(len(ROW_ORDER)))
            ax.set_yticklabels([ROW_LABELS[k] for k in ROW_ORDER], fontsize=10)
        else:
            ax.set_yticks(range(len(ROW_ORDER)))
            ax.set_yticklabels([])
        if ax in (axes_flat[2], axes_flat[3]):
            ax.set_xlabel("Ruler distance (cm)", fontsize=10)
        else:
            ax.set_xlabel("")

        ax.set_xticks(np.arange(-0.5, len(distances), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ROW_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=1.0)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isfinite(val):
                    ax.text(
                        j,
                        i,
                        "—",
                        ha="center",
                        va="center",
                        fontsize=9,
                        color="#888",
                    )
                    continue
                ax.text(
                    j,
                    i,
                    _cell_label(val, vmax),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=_text_color_for_value(val, vmax),
                    fontweight="bold",
                )

    fig.suptitle("Volume % error vs ground truth", fontsize=14, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.86, top=0.92, bottom=0.08, hspace=0.28, wspace=0.12)
    cax = fig.add_axes([0.88, 0.10, 0.022, 0.78])
    cbar = fig.colorbar(im, cax=cax)
    ticks = [vmin, vmin / 2, 0, vmax / 2, vmax]
    cbar.set_ticks([t for t in ticks if vmin <= t <= vmax])
    cbar.ax.set_yticklabels([f"{t:+.0f}%" for t in cbar.get_ticks()])
    cbar.set_label("% error", fontsize=9)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    combined_dir = args.combined_dir or (Path(args.runs_root) / "_combined")
    rows = load_lidar_volume_rows(
        Path(args.runs_root),
        data_root=Path(args.data_root),
    )
    if not rows:
        print(
            "No rows found. Need roi_bbox_volume_estimates.csv per scene "
            "(run 12_estimate_box_volume_from_roi.py) and LiDAR captures under data/."
        )
        return 1

    distances, grids, _ = compute_error_grid(rows)
    if distances is None:
        print("No valid distances found.")
        return 1

    out_png = combined_dir / "volume_error_dashboard.png"
    out_csv = combined_dir / "volume_error_dashboard.csv"
    export_long_csv(rows, out_csv)
    render_dashboard(distances, grids, out_png, args.error_cap_pct)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

