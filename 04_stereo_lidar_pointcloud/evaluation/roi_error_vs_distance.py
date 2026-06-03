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

# Shared GT-depth bins (m) for cross-scene pooled / grouped summaries.
POOLED_BIN_EDGES_M = np.array([0.15, 0.30, 0.60, 0.85, 1.00], dtype=float)

# Per-scene style: hue family + linestyle/marker so 0° vs 30° stay distinguishable.
# 0° = solid; 30°/60° = dashed or dash-dot.
SCENE_PLOT_STYLE: dict[str, dict[str, str]] = {
    # L — blue / cyan
    "L_carboard_box": {"color": "#0047ff", "linestyle": "-", "marker": "o"},
    "L_cardboard_box_30": {"color": "#00d4ff", "linestyle": "--", "marker": "s"},
    # M — red / orange
    "M_cardboard_box": {"color": "#ff0000", "linestyle": "-", "marker": "o"},
    "M_cardboardbox_30": {"color": "#ff8c00", "linestyle": "--", "marker": "^"},
    # S — green / lime
    "S_cardboard_box": {"color": "#008000", "linestyle": "-", "marker": "o"},
    "S_cardboard_box_30": {"color": "#9acd32", "linestyle": "--", "marker": "D"},
    # Checkerboard — violet / magenta / coral
    "checkerboard_data": {"color": "#7b00ff", "linestyle": "-", "marker": "o"},
    "checkerboard_data_30": {"color": "#ff00aa", "linestyle": "--", "marker": "s"},
    "checkerboard_data_60": {"color": "#ff4466", "linestyle": "-.", "marker": "v"},
}


def scene_plot_style(scene: str) -> dict[str, str]:
    return SCENE_PLOT_STYLE.get(
        scene,
        {"color": "#333333", "linestyle": "-", "marker": "o"},
    )


def _median_error_curve_cm(error_vs_gt: dict) -> tuple[list[float], list[float]]:
    """X = bin midpoint (cm), Y = median |Z−GT| (cm)."""
    centers, med_cm = [], []
    for b in error_vs_gt.get("bins", []):
        if b["count"] == 0 or b["median_error_m"] is None:
            continue
        lo_cm = 100.0 * b["range_min_m"]
        hi_cm = 100.0 * b["range_max_m"]
        centers.append(0.5 * (lo_cm + hi_cm))
        med_cm.append(100.0 * b["median_error_m"])
    return centers, med_cm


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


def aggregate_pooled_roi_error_vs_gt(
    runs_root: Path,
    *,
    scenes: list[str] | None = None,
) -> dict | None:
    """Pool ROI pixels across scenes; bin with POOLED_BIN_EDGES_M."""
    runs_root = Path(runs_root)
    pooled_gt: dict[str, list[np.ndarray]] = {}
    pooled_err: dict[str, list[np.ndarray]] = {}
    scenes_used: list[str] = []

    candidates = [runs_root / s for s in scenes] if scenes else sorted(runs_root.iterdir())
    for scene_dir in candidates:
        scene_dir = Path(scene_dir)
        if not scene_dir.is_dir():
            continue
        series = collect_scene_roi_pairs(scene_dir)
        if not series:
            continue
        scenes_used.append(scene_dir.name)
        for method, (gt_m, err_m) in series.items():
            pooled_gt.setdefault(method, []).append(gt_m)
            pooled_err.setdefault(method, []).append(err_m)

    if not pooled_gt:
        return None

    methods_out: dict[str, dict] = {}
    for method in sorted(pooled_gt.keys(), key=lambda m: METHOD_ORDER.index(m) if m in METHOD_ORDER else 99):
        gt_m = np.concatenate(pooled_gt[method])
        err_m = np.concatenate(pooled_err[method])
        order_key = METHOD_ORDER.index(method) if method in METHOD_ORDER else 99
        methods_out[method] = {
            "order": order_key,
            "n_points": int(len(gt_m)),
            "error_vs_gt": binned_error_vs_range(gt_m, err_m, POOLED_BIN_EDGES_M),
        }

    return {
        "kind": "pooled",
        "scenes": scenes_used,
        "n_scenes": len(scenes_used),
        "bin_edges_m": [float(x) for x in POOLED_BIN_EDGES_M],
        "methods": methods_out,
    }


def aggregate_per_scene_roi_error_by_method(
    runs_root: Path,
    *,
    scenes: list[str] | None = None,
) -> dict | None:
    """Per scene × method: median error in POOLED_BIN_EDGES_M (not merged across scenes)."""
    runs_root = Path(runs_root)
    scenes_used: list[str] = []
    methods_out: dict[str, dict[str, dict]] = {}

    candidates = [runs_root / s for s in scenes] if scenes else sorted(runs_root.iterdir())
    for scene_dir in candidates:
        scene_dir = Path(scene_dir)
        if not scene_dir.is_dir():
            continue
        series = collect_scene_roi_pairs(scene_dir)
        if not series:
            continue
        scene_name = scene_dir.name
        scenes_used.append(scene_name)
        for method, (gt_m, err_m) in series.items():
            methods_out.setdefault(method, {})[scene_name] = {
                "n_points": int(len(gt_m)),
                "error_vs_gt": binned_error_vs_range(gt_m, err_m, POOLED_BIN_EDGES_M),
            }

    if not scenes_used:
        return None

    return {
        "kind": "per_scene_by_method",
        "scenes": scenes_used,
        "n_scenes": len(scenes_used),
        "bin_edges_m": [float(x) for x in POOLED_BIN_EDGES_M],
        "methods": methods_out,
    }


def render_scene_roi_error_vs_gt_chart(
    summary: dict,
    out_path: Path,
    *,
    show_p90: bool = True,
) -> bool:
    import matplotlib.pyplot as plt

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
        centers, med_cm, p90_cm = [], [], []
        for b in evr["bins"]:
            if b["count"] == 0 or b["median_error_m"] is None:
                continue
            centers.append(100.0 * b["range_center_m"])
            med_cm.append(100.0 * b["median_error_m"])
            p90 = b.get("p90_error_m")
            p90_cm.append(100.0 * p90 if p90 is not None else np.nan)
        if not centers:
            continue
        centers = np.asarray(centers)
        med_cm = np.asarray(med_cm)
        p90_cm = np.asarray(p90_cm)
        ax.plot(centers, med_cm, "o-", color=color, linewidth=2, markersize=7, label=label)
        if show_p90 and np.any(np.isfinite(p90_cm)):
            ax.plot(
                centers,
                p90_cm,
                "--",
                color=color,
                linewidth=1.2,
                alpha=0.75,
                label=f"{label} (p90)",
            )

    ax.set_xlabel("Ground truth depth in ROI (cm)")
    ax.set_ylabel("|Z_est − GT| (cm)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def render_pooled_roi_error_chart(summary: dict, out_path: Path) -> bool:
    """One chart: median ROI error vs GT distance, all scenes pooled per method."""
    import matplotlib.pyplot as plt

    methods_data = summary.get("methods", {})
    if not methods_data:
        return False

    methods = sorted(methods_data.keys(), key=lambda m: methods_data[m].get("order", 99))
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    n_scenes = summary.get("n_scenes", len(summary.get("scenes", [])))

    for method in methods:
        evr = methods_data[method]["error_vs_gt"]
        color = METHOD_COLORS.get(method, "#333")
        label = METHOD_LABELS.get(method, method)
        centers, med_cm = _median_error_curve_cm(evr)
        if not centers:
            continue
        ax.plot(centers, med_cm, "o-", color=color, linewidth=2.2, markersize=8, label=label)

    ax.set_xlabel("Ground-truth depth in ROI (cm)")
    ax.set_ylabel("Median |Z_est − GT| (cm)")
    ax.set_title(f"Pooled across {n_scenes} scenes · fixed bins [15–30, 30–60, 60–85, 85–100] cm")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10)
    fig.suptitle("ROI depth error vs GT distance (all scenes)", fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return True


def render_method_faceted_roi_error_chart(summary: dict, out_path: Path) -> bool:
    """2×2 panels: one stereo method each; one line per scene (same GT bins)."""
    import matplotlib.pyplot as plt

    methods_data = summary.get("methods", {})
    scenes = summary.get("scenes", [])
    if not methods_data or not scenes:
        return False

    fig, axes = plt.subplots(2, 2, figsize=(14.5, 9.0), sharex=True, sharey=True, constrained_layout=False)

    for ax, method in zip(axes.ravel(), METHOD_ORDER):
        scene_dict = methods_data.get(method, {})
        if not scene_dict:
            ax.axis("off")
            continue
        for scene in scenes:
            entry = scene_dict.get(scene)
            if not entry:
                continue
            centers, med_cm = _median_error_curve_cm(entry["error_vs_gt"])
            if not centers:
                continue
            st = scene_plot_style(scene)
            ax.plot(
                centers,
                med_cm,
                color=st["color"],
                linestyle=st["linestyle"],
                marker=st["marker"],
                linewidth=1.2,
                markersize=4.5,
                alpha=1.0,
            )
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=12, fontweight="bold")
        ax.grid(True, alpha=0.3)

    axes[1, 0].set_xlabel("Ground-truth depth in ROI (cm)", fontsize=10)
    axes[1, 1].set_xlabel("Ground-truth depth in ROI (cm)", fontsize=10)
    axes[0, 0].set_ylabel("Median |Z_est − GT| (cm)", fontsize=10)
    axes[1, 0].set_ylabel("Median |Z_est − GT| (cm)", fontsize=10)

    handles = []
    for s in scenes:
        if not any(s in methods_data.get(m, {}) for m in METHOD_ORDER):
            continue
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
    n_legend = len(handles)
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=n_legend,
        fontsize=7.5,
        frameon=True,
        bbox_to_anchor=(0.5, 0.0),
        columnspacing=1.0,
        handletextpad=0.35,
    )
    fig.suptitle("ROI depth error vs GT distance", fontsize=13, fontweight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.98, top=0.90, bottom=0.20, hspace=0.22, wspace=0.12)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
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
