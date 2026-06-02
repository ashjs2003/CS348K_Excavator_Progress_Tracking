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
import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

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
    p.add_argument("--combined-dir", type=Path, default=None)
    p.add_argument("--vmax-pct", type=float, default=300.0, help="Color scale clamp for abs(%% error)")
    return p.parse_args()


def _row_key_from_scene(scene: str) -> str | None:
    if scene == "L_carboard_box":
        return "L-0deg"
    if scene == "L_cardboard_box_30":
        return "L-30deg"
    if scene == "M_cardboard_box":
        return "M-0deg"
    if scene == "M_cardboardbox_30":
        return "M-30deg"
    if scene == "S_cardboard_box":
        return "S-0deg"
    if scene == "S_cardboard_box_30":
        return "S-30deg"
    return None


def _to_float(v: str) -> float | None:
    if v is None or v == "":
        return None
    try:
        x = float(v)
    except ValueError:
        return None
    if not np.isfinite(x):
        return None
    return x


def load_rows(combined_dir: Path) -> list[dict]:
    rows = []
    for name in ("L_box_all_views.csv", "M_box_all_views.csv", "S_box_all_views.csv"):
        path = combined_dir / name
        if not path.is_file():
            continue
        with open(path, newline="") as f:
            for r in csv.DictReader(f):
                key = _row_key_from_scene(r.get("scene", ""))
                if key is None:
                    continue
                gt = _to_float(r.get("gt_volume_cm3", ""))
                dist = _to_float(r.get("ruler_distance_cm", ""))
                if gt is None or gt <= 0 or dist is None:
                    continue
                row = {
                    "row_key": key,
                    "distance_cm": dist,
                    "gt_volume_cm3": gt,
                }
                for col, _ in METHOD_COLUMNS:
                    row[col] = _to_float(r.get(col, ""))
                rows.append(row)
    return rows


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
    fields = ["row_key", "distance_cm", "method", "gt_volume_cm3", "est_volume_cm3", "error_pct"]
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            gt = r["gt_volume_cm3"]
            for col, label in METHOD_COLUMNS:
                est = r.get(col)
                if est is None:
                    continue
                w.writerow(
                    {
                        "row_key": r["row_key"],
                        "distance_cm": f"{r['distance_cm']:.1f}",
                        "method": label,
                        "gt_volume_cm3": f"{gt:.1f}",
                        "est_volume_cm3": f"{est:.1f}",
                        "error_pct": f"{(100.0 * (est - gt) / gt):.2f}",
                    }
                )


def render_dashboard(distances, grids, out_png: Path, vmax_pct: float) -> None:
    n_methods = len(METHOD_COLUMNS)
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), squeeze=False, constrained_layout=True)
    axes_flat = axes.ravel()

    # Use robust limits for readability, then cap by --vmax-pct.
    all_vals = []
    for col, _ in METHOD_COLUMNS:
        arr = grids[col]
        vals = np.abs(arr[np.isfinite(arr)])
        if len(vals):
            all_vals.append(vals)
    if all_vals:
        pooled = np.concatenate(all_vals)
        robust = float(np.percentile(pooled, 90))
        vlim = min(float(max(20.0, robust)), float(max(20.0, vmax_pct)))
    else:
        vlim = float(max(20.0, vmax_pct))
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#eeeeee")

    im = None
    for ax, (col, label) in zip(axes_flat, METHOD_COLUMNS):
        data = grids[col]
        shown = np.ma.masked_invalid(np.clip(np.abs(data), 0.0, vlim))
        im = ax.imshow(shown, aspect="auto", cmap=cmap, vmin=0.0, vmax=vlim)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(range(len(distances)))
        ax.set_xticklabels([f"{int(d)}" for d in distances], fontsize=9)
        if ax in (axes_flat[0], axes_flat[2]):
            ax.set_yticks(range(len(ROW_ORDER)))
            ax.set_yticklabels([ROW_LABELS[k] for k in ROW_ORDER], fontsize=9)
        else:
            ax.set_yticks(range(len(ROW_ORDER)))
            ax.set_yticklabels([])
        ax.set_xlabel("Ruler distance (cm)", fontsize=9)

        # Cell borders for easier scan.
        ax.set_xticks(np.arange(-0.5, len(distances), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(ROW_ORDER), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.8)
        ax.tick_params(which="minor", bottom=False, left=False)

        # Annotate values (lightly) for readability.
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                if not np.isfinite(val):
                    continue
                txt = f"{val:+.0f}%"
                if abs(val) >= vlim:
                    txt = f"{'+' if val > 0 else '-'}>{vlim:.0f}%"
                frac = min(max(abs(float(val)) / vlim, 0.0), 1.0)
                tcolor = "white" if frac >= 0.55 else "#111"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color=tcolor, fontweight="bold")

    fig.suptitle("Volume Error Dashboard (vs GT)", fontsize=14, fontweight="bold")
    cbar = fig.colorbar(im, ax=axes_flat, fraction=0.03, pad=0.02)
    cbar.set_label("Absolute % error vs GT volume")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    combined_dir = args.combined_dir or (Path(args.runs_root) / "_combined")
    rows = load_rows(combined_dir)
    if not rows:
        print("No combined GT rows found. Make sure L/M/S *_all_views.csv exist with gt_volume_cm3.")
        return 1

    distances, grids, _ = compute_error_grid(rows)
    if distances is None:
        print("No valid distances found.")
        return 1

    out_png = combined_dir / "volume_error_dashboard.png"
    out_csv = combined_dir / "volume_error_dashboard.csv"
    export_long_csv(rows, out_csv)
    render_dashboard(distances, grids, out_png, args.vmax_pct)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

