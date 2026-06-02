"""
One-figure dashboard: ROI error vs GT distance across all scenes and methods.

Reads:
  outputs/runs/*/roi_error_vs_gt_distance.json

Writes:
  outputs/runs/_combined/roi_error_all_scenes_dashboard.png
  outputs/runs/_combined/roi_error_all_scenes_dashboard.csv
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

from output_runs import RUNS_ROOT

METHODS = [
    ("opencv", "OpenCV"),
    ("dav2", "DA-V2"),
    ("dav2_gt", "DA-V2 GT"),
    ("foundation", "Foundation"),
]


def parse_args():
    p = argparse.ArgumentParser(description="All-scenes ROI error dashboard")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--vmax-cm", type=float, default=40.0, help="Color clamp for median error (cm)")
    return p.parse_args()


def scene_order_key(scene: str) -> tuple[int, str]:
    order = [
        "L_carboard_box",
        "L_cardboard_box_30",
        "M_cardboard_box",
        "M_cardboardbox_30",
        "S_cardboard_box",
        "S_cardboard_box_30",
        "checkerboard_data",
        "checkerboard_data_30",
        "checkerboard_data_60",
    ]
    try:
        return (order.index(scene), scene)
    except ValueError:
        return (999, scene)


def load_scene_payloads(runs_root: Path) -> list[tuple[str, dict]]:
    out = []
    for path in sorted(Path(runs_root).glob("*/roi_error_vs_gt_distance.json")):
        try:
            payload = json.loads(path.read_text())
        except Exception:
            continue
        scene = payload.get("scene") or path.parent.name
        out.append((scene, payload))
    out.sort(key=lambda t: scene_order_key(t[0]))
    return out


def collect_distance_bins(scene_payloads: list[tuple[str, dict]]) -> list[int]:
    # Use rounded GT-bin centers in cm so all scenes can be shown together.
    vals: set[int] = set()
    for _scene, payload in scene_payloads:
        methods = payload.get("methods", {})
        for method_key, _ in METHODS:
            bins = methods.get(method_key, {}).get("error_vs_gt", {}).get("bins", [])
            for b in bins:
                c = b.get("range_center_m")
                if c is None:
                    continue
                vals.add(int(round(float(c) * 100.0)))
    return sorted(vals)


def build_grids(scene_payloads: list[tuple[str, dict]], distance_cm_bins: list[int]):
    scenes = [s for s, _ in scene_payloads]
    d_to_j = {d: j for j, d in enumerate(distance_cm_bins)}
    grids = {m: np.full((len(scenes), len(distance_cm_bins)), np.nan, dtype=float) for m, _ in METHODS}
    long_rows = []

    for i, (scene, payload) in enumerate(scene_payloads):
        methods = payload.get("methods", {})
        for method_key, method_label in METHODS:
            bins = methods.get(method_key, {}).get("error_vs_gt", {}).get("bins", [])
            for b in bins:
                center_m = b.get("range_center_m")
                med_m = b.get("median_error_m")
                count = b.get("count", 0)
                if center_m is None:
                    continue
                d_cm = int(round(float(center_m) * 100.0))
                if d_cm not in d_to_j:
                    continue
                if med_m is None or count == 0:
                    continue
                err_cm = float(med_m) * 100.0
                j = d_to_j[d_cm]
                grids[method_key][i, j] = err_cm
                long_rows.append(
                    {
                        "scene": scene,
                        "method": method_label,
                        "distance_cm_bin": d_cm,
                        "median_error_cm": f"{err_cm:.3f}",
                        "count": int(count),
                    }
                )
    return scenes, grids, long_rows


def write_long_csv(path: Path, rows: list[dict]) -> None:
    fields = ["scene", "method", "distance_cm_bin", "median_error_cm", "count"]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def render_dashboard(
    scenes: list[str],
    distance_cm_bins: list[int],
    grids: dict[str, np.ndarray],
    out_png: Path,
    vmax_cm: float,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 8.6), constrained_layout=True)
    axes_flat = axes.ravel()
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#efefef")
    vlim = max(10.0, float(vmax_cm))

    im = None
    for ax, (method_key, label) in zip(axes_flat, METHODS):
        data = grids[method_key]
        shown = np.ma.masked_invalid(np.clip(data, 0.0, vlim))
        im = ax.imshow(shown, aspect="auto", cmap=cmap, vmin=0.0, vmax=vlim)
        ax.set_title(label, fontsize=12, fontweight="bold", pad=8)
        ax.set_xticks(range(len(distance_cm_bins)))
        ax.set_xticklabels([str(d) for d in distance_cm_bins], fontsize=8)
        if ax in (axes_flat[0], axes_flat[2]):
            ax.set_yticks(range(len(scenes)))
            ax.set_yticklabels(scenes, fontsize=8)
        else:
            ax.set_yticks(range(len(scenes)))
            ax.set_yticklabels([])
        ax.set_xlabel("GT distance bin (cm)", fontsize=9)

        ax.set_xticks(np.arange(-0.5, len(distance_cm_bins), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(scenes), 1), minor=True)
        ax.grid(which="minor", color="white", linestyle="-", linewidth=0.8, alpha=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)

        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                v = data[i, j]
                if not np.isfinite(v):
                    continue
                text = f"{v:.1f}"
                if v >= vlim:
                    text = f">{vlim:.0f}"
                frac = min(max(float(v) / vlim, 0.0), 1.0)
                tcolor = "white" if frac >= 0.55 else "#111"
                ax.text(j, i, text, ha="center", va="center", fontsize=6.7, color=tcolor, fontweight="bold")

    cbar = fig.colorbar(im, ax=axes_flat, fraction=0.03, pad=0.02)
    cbar.set_label("Median ROI error vs GT (cm)")
    fig.suptitle("All scenes: ROI median error vs GT distance", fontsize=14, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    scene_payloads = load_scene_payloads(args.runs_root)
    if not scene_payloads:
        print("No roi_error_vs_gt_distance.json files found.")
        return 1
    distance_bins = collect_distance_bins(scene_payloads)
    if not distance_bins:
        print("No distance bins found.")
        return 1

    scenes, grids, long_rows = build_grids(scene_payloads, distance_bins)
    out_dir = Path(args.runs_root) / "_combined"
    out_png = out_dir / "roi_error_all_scenes_dashboard.png"
    out_csv = out_dir / "roi_error_all_scenes_dashboard.csv"
    write_long_csv(out_csv, long_rows)
    render_dashboard(scenes, distance_bins, grids, out_png, args.vmax_cm)
    print(f"Wrote {out_png}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

