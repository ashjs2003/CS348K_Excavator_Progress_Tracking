"""
Checkerboard depth comparison with a single metric Z colormap (turbo) for all methods.

Uses depth_metric.npy or disparity.npy (+ Q) — not legacy per-image disparity previews.

Outputs (under outputs/runs/_combined/):
  checkerboard_depth_uniform_grid.png
  checkerboard_depth_uniform_scale.json
  checkerboard_depth_uniform_panels/<pair>_<method>.png
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.colors import Normalize
from matplotlib.ticker import FixedLocator, FuncFormatter

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dav2_scale import save_depth_preview
from depth_layout import rgb1_rectified_path, resolve_path
from evaluation.depth_maps import load_metric_depth, load_stereo_geometry
from output_runs import RUNS_ROOT

SCENE = "checkerboard_data"
PAIRS = [
    ("pair_001", 10),
    ("pair_005", 50),
    ("pair_006", 75),
]
DEPTH_ROWS = [
    ("dav2_gt", "Depth Anything with GT"),
    ("dav2", "Depth Anything"),
    ("foundation", "Foundation Stereo"),
    ("opencv", "OpenCV"),
]
# Checkerboard captures are sub-meter; 0–100 cm spreads method differences vs 0–200 cm.
DEFAULT_VMIN_M = 0.0
DEFAULT_VMAX_M = 1.0


def parse_args():
    p = argparse.ArgumentParser(description="Uniform metric depth grid for checkerboard pairs")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--scene", default=SCENE)
    p.add_argument("--vmin-m", type=float, default=DEFAULT_VMIN_M, help="Near clip (m)")
    p.add_argument("--vmax-m", type=float, default=DEFAULT_VMAX_M, help="Far clip (m)")
    p.add_argument("--out-name", default="checkerboard_depth_uniform_grid")
    return p.parse_args()


def load_gt_cm(scene_name: str, pair_id: str) -> int | None:
    txt = _REPO_ROOT / "data" / scene_name / f"{pair_id}.txt"
    if not txt.is_file():
        return None
    raw = txt.read_text().strip().splitlines()[0].strip()
    if not raw or raw.upper() == "NA":
        return None
    try:
        return int(float(raw))
    except ValueError:
        return None


def depth_to_rgb_float(depth_m: np.ndarray, vmin_m: float, vmax_m: float) -> np.ndarray:
    """RGB float [0,1] for matplotlib; invalid NaN."""
    out = np.full((*depth_m.shape, 3), np.nan, dtype=np.float32)
    valid = np.isfinite(depth_m) & (depth_m > 0)
    if not np.any(valid):
        return out
    norm = Normalize(vmin=vmin_m, vmax=vmax_m, clip=True)
    rgba = cm.turbo(norm(depth_m[valid]))
    out[valid] = rgba[:, :3]
    return out


def panel_stats(depth_m: np.ndarray) -> dict:
    valid = depth_m[np.isfinite(depth_m) & (depth_m > 0)]
    if valid.size == 0:
        return {"valid_px": 0}
    return {
        "valid_px": int(valid.size),
        "min_m": float(np.min(valid)),
        "p50_m": float(np.median(valid)),
        "max_m": float(np.max(valid)),
    }


def main():
    args = parse_args()
    scene_dir = Path(args.runs_root) / args.scene
    out_dir = Path(args.runs_root) / "_combined"
    panel_dir = out_dir / "checkerboard_depth_uniform_panels"
    panel_dir.mkdir(parents=True, exist_ok=True)

    vmin_m, vmax_m = float(args.vmin_m), float(args.vmax_m)
    if vmax_m <= vmin_m:
        raise SystemExit("--vmax-m must be greater than --vmin-m")

    meta: dict = {
        "scene": args.scene,
        "vmin_m": vmin_m,
        "vmax_m": vmax_m,
        "colormap": "turbo",
        "unit": "m",
        "colorbar_label": "metric depth",
        "colorbar_ticks_cm": f"{vmin_m * 100:.0f}–{vmax_m * 100:.0f}",
        "pairs": [],
    }

    n_depth_rows = len(DEPTH_ROWS)
    n_cols = len(PAIRS)
    fig, axes = plt.subplots(
        n_depth_rows + 1,
        n_cols,
        figsize=(4.2 * n_cols, 2.6 * (n_depth_rows + 1)),
        gridspec_kw={"wspace": 0.05, "hspace": 0.12},
    )
    if n_cols == 1:
        axes = np.atleast_2d(axes)

    for col, (pair_id, gt_cm_fallback) in enumerate(PAIRS):
        run_dir = scene_dir / pair_id
        depth_dir = run_dir / "depth"
        if not depth_dir.is_dir():
            raise FileNotFoundError(f"Missing {depth_dir}")

        gt_cm = load_gt_cm(args.scene, pair_id) or gt_cm_fallback
        geometry = load_stereo_geometry(depth_dir)
        pair_meta = {"pair": pair_id, "gt_cm": gt_cm, "methods": {}}
        meta["pairs"].append(pair_meta)

        col_title = f"{gt_cm} cm"
        axes[0, col].set_title(col_title, fontsize=12, fontweight="bold")

        for row, (method_key, row_label) in enumerate(DEPTH_ROWS):
            ax = axes[row, col]
            depth = load_metric_depth(depth_dir, method_key, geometry)
            if depth is None:
                ax.text(0.5, 0.5, "missing", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                pair_meta["methods"][method_key] = {"status": "missing"}
                continue

            src = resolve_path(depth_dir, method_key, "depth_metric.npy")
            if src is None:
                src = resolve_path(depth_dir, method_key, "disparity.npy")
            pair_meta["methods"][method_key] = {
                "source": str(src.relative_to(run_dir)) if src else None,
                **panel_stats(depth),
            }

            panel_path = panel_dir / f"{pair_id}_{method_key}.png"
            save_depth_preview(panel_path, depth.astype(np.float32), vmin_m=vmin_m, vmax_m=vmax_m)

            rgb = depth_to_rgb_float(depth, vmin_m, vmax_m)
            ax.imshow(rgb)
            ax.set_axis_off()
            if col == 0:
                ax.set_ylabel(row_label, fontsize=9, rotation=90, labelpad=36, va="center")

        # RGB row
        ax_rgb = axes[n_depth_rows, col]
        rgb_path = rgb1_rectified_path(depth_dir)
        bgr = cv2.imread(str(rgb_path))
        if bgr is None:
            ax_rgb.text(0.5, 0.5, "no RGB", ha="center", va="center", transform=ax_rgb.transAxes)
        else:
            ax_rgb.imshow(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        ax_rgb.set_axis_off()
        if col == 0:
            ax_rgb.set_ylabel("RGB", fontsize=9, rotation=90, labelpad=36, va="center")

    # Shared colorbar for depth rows
    sm = cm.ScalarMappable(norm=Normalize(vmin=vmin_m, vmax=vmax_m), cmap="turbo")
    sm.set_array([])
    cbar = fig.colorbar(
        sm,
        ax=axes[:n_depth_rows, :].ravel().tolist(),
        fraction=0.02,
        pad=0.02,
        aspect=30,
    )
    tick_m = np.linspace(vmin_m, vmax_m, 6)
    cbar.ax.yaxis.set_major_locator(FixedLocator(tick_m))
    cbar.ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _pos: f"{x * 100:.0f}"))
    cbar.set_label("metric depth", fontsize=10)

    fig.suptitle(
        f"{args.scene}: uniform depth scale across methods (pairs 001 / 005 / 006)",
        fontsize=13,
        y=0.98,
    )

    grid_path = out_dir / f"{args.out_name}.png"
    fig.savefig(grid_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    scale_path = out_dir / f"{args.out_name}_scale.json"
    scale_path.write_text(json.dumps(meta, indent=2) + "\n")

    print(f"Wrote {grid_path}")
    print(f"Wrote {scale_path}")
    print(f"Panels under {panel_dir}")


if __name__ == "__main__":
    main()
