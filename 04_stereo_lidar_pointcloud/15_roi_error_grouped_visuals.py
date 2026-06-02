"""
Create cleaner ROI error visuals across all scenes/methods.

Outputs:
  outputs/runs/_combined/roi_error_all_scenes_grouped_dashboard.png
  outputs/runs/_combined/roi_error_method_scene_summary.png
  outputs/runs/_combined/roi_error_grouped_long.csv
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

METHODS = [("opencv", "OpenCV"), ("dav2", "DA-V2"), ("dav2_gt", "DA-V2 GT"), ("foundation", "Foundation")]
SCENE_ORDER = [
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

GROUPS = [
    ("Near", 15.0, 30.0),
    ("Mid", 30.0, 60.0),
    ("Far", 60.0, 85.0),
    ("Very Far", 85.0, 100.0),
]


def parse_args():
    p = argparse.ArgumentParser(description="Grouped ROI error visuals")
    p.add_argument("--runs-root", type=Path, default=RUNS_ROOT)
    p.add_argument("--vmax-cm", type=float, default=35.0)
    return p.parse_args()


def scene_sort_key(scene: str) -> tuple[int, str]:
    try:
        return (SCENE_ORDER.index(scene), scene)
    except ValueError:
        return (999, scene)


def load_payloads(runs_root: Path) -> list[tuple[str, dict]]:
    out = []
    for p in sorted(Path(runs_root).glob("*/roi_error_vs_gt_distance.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        scene = d.get("scene") or p.parent.name
        out.append((scene, d))
    out.sort(key=lambda t: scene_sort_key(t[0]))
    return out


def grouped_stats_for_bins(bins: list[dict], lo_cm: float, hi_cm: float) -> tuple[float | None, int]:
    vals = []
    counts = []
    for b in bins:
        c = b.get("range_center_m")
        med = b.get("median_error_m")
        n = int(b.get("count", 0))
        if c is None or med is None or n <= 0:
            continue
        c_cm = float(c) * 100.0
        if lo_cm <= c_cm < hi_cm:
            vals.append(float(med) * 100.0)
            counts.append(n)
    if not vals:
        return None, 0
    # weighted average of per-bin medians by bin counts (pragmatic summary)
    w = np.asarray(counts, dtype=float)
    x = np.asarray(vals, dtype=float)
    return float(np.sum(w * x) / np.sum(w)), int(np.sum(counts))


def build_grouped_tables(payloads: list[tuple[str, dict]]):
    scenes = [s for s, _ in payloads]
    val = {m: np.full((len(scenes), len(GROUPS)), np.nan, dtype=float) for m, _ in METHODS}
    cnt = {m: np.zeros((len(scenes), len(GROUPS)), dtype=int) for m, _ in METHODS}
    long_rows = []
    for i, (scene, payload) in enumerate(payloads):
        md = payload.get("methods", {})
        for m_key, m_label in METHODS:
            bins = md.get(m_key, {}).get("error_vs_gt", {}).get("bins", [])
            for j, (g_name, lo, hi) in enumerate(GROUPS):
                v, n = grouped_stats_for_bins(bins, lo, hi)
                if v is None:
                    continue
                val[m_key][i, j] = v
                cnt[m_key][i, j] = n
                long_rows.append(
                    {
                        "scene": scene,
                        "method": m_label,
                        "group": g_name,
                        "median_error_cm": f"{v:.3f}",
                        "count": n,
                    }
                )
    return scenes, val, cnt, long_rows


def write_long_csv(path: Path, rows: list[dict]) -> None:
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["scene", "method", "group", "median_error_cm", "count"])
        w.writeheader()
        w.writerows(rows)


def render_grouped_dashboard(
    scenes: list[str],
    val: dict[str, np.ndarray],
    cnt: dict[str, np.ndarray],
    out_png: Path,
    vmax_cm: float,
):
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.6), constrained_layout=True)
    axes = axes.ravel()
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#efefef")
    vlim = max(10.0, float(vmax_cm))

    im = None
    for ax, (m_key, m_label) in zip(axes, METHODS):
        data = val[m_key]
        counts = cnt[m_key]
        # Add an "All" numeric column: weighted mean over Near/Mid/Far/Very Far.
        data_all = np.full((data.shape[0], 1), np.nan, dtype=float)
        count_all = np.zeros((counts.shape[0], 1), dtype=int)
        for i in range(data.shape[0]):
            ok = np.isfinite(data[i, :]) & (counts[i, :] > 0)
            if np.any(ok):
                w = counts[i, ok].astype(float)
                x = data[i, ok].astype(float)
                data_all[i, 0] = float(np.sum(w * x) / np.sum(w))
                count_all[i, 0] = int(np.sum(counts[i, ok]))
        data_plot = np.concatenate([data, data_all], axis=1)
        counts_plot = np.concatenate([counts, count_all], axis=1)
        shown = np.ma.masked_invalid(np.clip(data_plot, 0.0, vlim))
        im = ax.imshow(shown, aspect="auto", cmap=cmap, vmin=0.0, vmax=vlim)
        ax.set_title(m_label, fontsize=12, fontweight="bold")
        x_labels = [f"{name}\n[{int(lo)}-{int(hi)} cm)" for name, lo, hi in GROUPS] + ["All\n[15-100 cm)"]
        ax.set_xticks(range(len(x_labels)))
        ax.set_xticklabels(x_labels, fontsize=9)
        if ax in (axes[0], axes[2]):
            ax.set_yticks(range(len(scenes)))
            ax.set_yticklabels(scenes, fontsize=8)
        else:
            ax.set_yticks(range(len(scenes)))
            ax.set_yticklabels([])

        # annotate with median + tiny n
        for i in range(data_plot.shape[0]):
            for j in range(data_plot.shape[1]):
                v = data_plot[i, j]
                n = counts_plot[i, j]
                if not np.isfinite(v):
                    continue
                txt = f"{v:.1f}\n(n={n})"
                if v >= vlim:
                    txt = f">{vlim:.0f}\n(n={n})"
                frac = min(max(float(v) / vlim, 0.0), 1.0)
                tcolor = "white" if frac >= 0.55 else "#111"
                ax.text(
                    j,
                    i,
                    txt,
                    ha="center",
                    va="center",
                    fontsize=6.3,
                    color=tcolor,
                    fontweight="bold",
                )

        ax.set_xticks(np.arange(-0.5, len(x_labels), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(scenes), 1), minor=True)
        ax.grid(which="minor", color="white", linewidth=0.8, alpha=0.9)
        ax.tick_params(which="minor", bottom=False, left=False)

    cbar = fig.colorbar(im, ax=axes, fraction=0.03, pad=0.02)
    cbar.set_label("Grouped median ROI error vs GT (cm)")
    fig.suptitle("All scenes ROI error (grouped by GT distance)", fontsize=14, fontweight="bold")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def render_method_scene_summary(scenes: list[str], val: dict[str, np.ndarray], out_png: Path):
    # aggregate each scene/method across groups (simple mean over available group medians)
    arr = np.full((len(scenes), len(METHODS)), np.nan, dtype=float)
    for j, (m_key, _label) in enumerate(METHODS):
        data = val[m_key]
        for i in range(len(scenes)):
            row = data[i, :]
            ok = np.isfinite(row)
            if np.any(ok):
                arr[i, j] = float(np.mean(row[ok]))

    finite = arr[np.isfinite(arr)]
    vmax = float(np.percentile(finite, 90)) if len(finite) else 20.0
    vmax = max(8.0, vmax)
    cmap = plt.get_cmap("YlOrRd").copy()
    cmap.set_bad("#efefef")

    fig, ax = plt.subplots(figsize=(7.6, 6.0), constrained_layout=True)
    shown = np.ma.masked_invalid(np.clip(arr, 0.0, vmax))
    im = ax.imshow(shown, aspect="auto", cmap=cmap, vmin=0.0, vmax=vmax)
    ax.set_title("Scene × Method summary (mean grouped error cm)", fontsize=12, fontweight="bold")
    ax.set_xticks(range(len(METHODS)))
    ax.set_xticklabels([m[1] for m in METHODS], fontsize=9)
    ax.set_yticks(range(len(scenes)))
    ax.set_yticklabels(scenes, fontsize=8)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if np.isfinite(v):
                frac = min(max(float(v) / vmax, 0.0), 1.0)
                tcolor = "white" if frac >= 0.55 else "#111"
                ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8, color=tcolor, fontweight="bold")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Error (cm)")
    fig.savefig(out_png, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = parse_args()
    payloads = load_payloads(args.runs_root)
    if not payloads:
        print("No roi_error_vs_gt_distance.json files found.")
        return 1

    scenes, val, cnt, rows = build_grouped_tables(payloads)
    out_dir = Path(args.runs_root) / "_combined"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_dash = out_dir / "roi_error_all_scenes_grouped_dashboard.png"
    out_sum = out_dir / "roi_error_method_scene_summary.png"
    out_csv = out_dir / "roi_error_grouped_long.csv"

    write_long_csv(out_csv, rows)
    render_grouped_dashboard(scenes, val, cnt, out_dash, args.vmax_cm)
    render_method_scene_summary(scenes, val, out_sum)
    print(f"Wrote {out_dash}")
    print(f"Wrote {out_sum}")
    print(f"Wrote {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

