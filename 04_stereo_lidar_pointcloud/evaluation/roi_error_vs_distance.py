"""Pool ROI per-point GT comparisons across captures; bin error vs GT depth."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from evaluation.error_vs_range import binned_error_vs_range, shared_bin_edges
from evaluation.roi_gt_compare import METHOD_ORDER

METHOD_LABELS = {
    "opencv": "OpenCV",
    "dav2": "DA-V2",
    "dav2_gt": "DA-V2 GT",
    "foundation": "Foundation",
}
METHOD_COLORS = {
    "opencv": "#4C78A8",
    "dav2": "#F58518",
    "dav2_gt": "#E45756",
    "foundation": "#54A24B",
}

ROI_PER_POINT_PREFIX = "roi_gt_per_point_"


def load_roi_gt_pairs_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return (gt_m, error_m) for compared ROI pixels."""
    path = Path(path)
    if not path.is_file():
        return np.array([]), np.array([])
    gt_vals, err_vals = [], []
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gt_s = row.get("gt_m", "")
            err_s = row.get("error_cm", "")
            if not gt_s or not err_s:
                continue
            try:
                gt_m = float(gt_s)
                err_cm = float(err_s)
            except ValueError:
                continue
            if not (np.isfinite(gt_m) and np.isfinite(err_cm) and gt_m > 0):
                continue
            gt_vals.append(gt_m)
            err_vals.append(err_cm / 100.0)
    return np.asarray(gt_vals, dtype=float), np.asarray(err_vals, dtype=float)


def discover_roi_methods_in_run(validation_dir: Path) -> list[str]:
    methods = []
    for p in sorted(validation_dir.glob(f"{ROI_PER_POINT_PREFIX}*.csv")):
        key = p.stem.replace(ROI_PER_POINT_PREFIX, "")
        if key:
            methods.append(key)
    return [m for m in METHOD_ORDER if m in methods] + [
        m for m in methods if m not in METHOD_ORDER
    ]


def collect_scene_roi_pairs(scene_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Pool (gt_m, error_m) per method across all pair_* with ROI CSVs."""
    scene_dir = Path(scene_dir)
    pooled: dict[str, list[np.ndarray]] = {}
    pooled_err: dict[str, list[np.ndarray]] = {}

    for pair_dir in sorted(scene_dir.glob("pair_*")):
        val = pair_dir / "validation"
        if not val.is_dir():
            continue
        for method in discover_roi_methods_in_run(val):
            gt_m, err_m = load_roi_gt_pairs_csv(val / f"{ROI_PER_POINT_PREFIX}{method}.csv")
            if len(gt_m) == 0:
                continue
            pooled.setdefault(method, []).append(gt_m)
            pooled_err.setdefault(method, []).append(err_m)

    out: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for method in pooled:
        out[method] = (
            np.concatenate(pooled[method]),
            np.concatenate(pooled_err[method]),
        )
    return out


def aggregate_scene_roi_error_vs_gt(
    scene_dir: Path,
    *,
    n_bins: int = 8,
) -> dict | None:
    """
    Binned median |Z_est - GT| vs GT depth (m) per method, shared bin edges.
  """
    series = collect_scene_roi_pairs(scene_dir)
    if not series:
        return None

    scene_name = Path(scene_dir).name
    all_gt = [gt for gt, _ in series.values()]
    bin_edges = shared_bin_edges(all_gt, n_bins=n_bins)
    if bin_edges is None:
        pooled_gt = np.concatenate(all_gt)
        if len(pooled_gt) < 2:
            return None
        bin_edges = np.linspace(float(np.min(pooled_gt)), float(np.max(pooled_gt)), n_bins + 1)

    methods_out = {}
    for method, (gt_m, err_m) in series.items():
        order_key = METHOD_ORDER.index(method) if method in METHOD_ORDER else 99
        methods_out[method] = {
            "order": order_key,
            "n_points": int(len(gt_m)),
            "error_vs_gt": binned_error_vs_range(gt_m, err_m, bin_edges),
        }

    pair_ids = []
    for pair_dir in sorted(Path(scene_dir).glob("pair_*")):
        val = pair_dir / "validation"
        if val.is_dir() and any(val.glob(f"{ROI_PER_POINT_PREFIX}*.csv")):
            pair_ids.append(pair_dir.name.replace("pair_", ""))

    return {
        "scene": scene_name,
        "n_pairs": len(pair_ids),
        "pair_ids": pair_ids,
        "bin_edges_m": [float(x) for x in bin_edges],
        "methods": methods_out,
    }


def render_scene_roi_error_vs_gt_chart(
    summary: dict,
    out_path: Path,
    *,
    show_p90: bool = True,
) -> bool:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    methods_data = summary.get("methods", {})
    if not methods_data:
        return False

    bin_edges = np.asarray(summary["bin_edges_m"], dtype=float)
    scene = summary.get("scene", "")

    methods = sorted(
        methods_data.keys(),
        key=lambda m: methods_data[m].get("order", 99),
    )

    fig, ax = plt.subplots(figsize=(9, 5))
    fig.suptitle(
        f"ROI error vs ground-truth distance — {scene}",
        fontsize=13,
        fontweight="bold",
    )

    for method in methods:
        evr = methods_data[method]["error_vs_gt"]
        color = METHOD_COLORS.get(method, "#333")
        label = METHOD_LABELS.get(method, method)
        centers, med_cm, p90_cm, counts = [], [], [], []
        for b in evr["bins"]:
            if b["count"] == 0 or b["median_error_m"] is None:
                continue
            centers.append(100.0 * b["range_center_m"])
            med_cm.append(100.0 * b["median_error_m"])
            p90_cm.append(100.0 * b["p90_error_m"])
            counts.append(b["count"])
        if not centers:
            continue
        centers = np.asarray(centers)
        med_cm = np.asarray(med_cm)
        p90_cm = np.asarray(p90_cm)
        counts = np.asarray(counts)
        ax.plot(centers, med_cm, "o-", color=color, linewidth=2, markersize=7, label=f"{label} · median")
        if show_p90:
            ax.plot(
                centers,
                p90_cm,
                "--",
                color=color,
                linewidth=1.2,
                alpha=0.75,
                label=f"{label} · p90",
            )
        for x, y, n in zip(centers, med_cm, counts):
            ax.annotate(
                str(n),
                (x, y),
                textcoords="offset points",
                xytext=(0, 6),
                ha="center",
                fontsize=7,
                color=color,
            )

    ax.set_xlabel("Ground truth depth in ROI (cm)")
    ax.set_ylabel("|Z_est − GT| (cm)")
    ax.set_title(
        f"{summary.get('n_pairs', 0)} annotated captures · n = ROI pixels per bin"
    )
    ax.axhline(5, color="#888", linestyle=":", linewidth=0.9, alpha=0.7)
    ax.axhline(15, color="#888", linestyle="--", linewidth=0.9, alpha=0.7)
    ax.grid(True, alpha=0.3)

    style_handles = [
        Line2D([0], [0], color="#555", linestyle="-", marker="o", markersize=6, label="Solid = median"),
        Line2D(
            [0],
            [0],
            color="#555",
            linestyle="--",
            linewidth=1.5,
            label="Dashed = p90",
        ),
    ]
    method_handles, method_labels = ax.get_legend_handles_labels()
    ax.legend(
        style_handles + method_handles,
        [h.get_label() for h in style_handles] + list(method_labels),
        loc="best",
        fontsize=8,
    )
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def process_scene(scene_dir: Path, *, n_bins: int = 8) -> dict | None:
    summary = aggregate_scene_roi_error_vs_gt(scene_dir, n_bins=n_bins)
    if summary is None:
        return None
    scene_dir = Path(scene_dir)
    json_path = scene_dir / "roi_error_vs_gt_distance.json"
    png_path = scene_dir / "roi_error_vs_gt_distance.png"
    json_path.write_text(json.dumps(summary, indent=2) + "\n")
    render_scene_roi_error_vs_gt_chart(summary, png_path)
    return summary
